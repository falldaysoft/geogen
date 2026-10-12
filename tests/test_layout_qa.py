"""Layout QA: overlaps, walls, door swings, windows, doors and reachability."""

import numpy as np
import pytest

from geogen.core.transform import Transform
from geogen.layout import LayoutLoader, SceneComposer
from geogen.layout.qa import check_layout
from geogen.player import PlayerSpec


@pytest.mark.parametrize("scene", ["hotel_room", "hotel_room_auto"])
def test_hotel_rooms_pass_qa(scene):
    assert [str(i) for i in check_layout(SceneComposer().compose(f"assets/scenes/{scene}.yaml"))] == []


def _suite_with(*items):
    """Hotel suite plan with items (asset, room-local x, z, yaw degrees) in the bedroom."""
    root = LayoutLoader().load("assets/hotel_suite.yaml")
    bedroom = root.find("bedroom")
    for i, (asset, x, z, yaw) in enumerate(items):
        node = LayoutLoader().load(f"assets/{asset}")
        node.name = f"{asset.removesuffix('.yaml')}_{i}"
        node.transform = Transform(translation=np.array([x, 0.0, z]), rotation=np.array([0, np.radians(yaw), 0]))
        bedroom.add_child(node)
    return root


def _kinds(root, player=None):
    return sorted({i.kind for i in check_layout(root, player)})


def test_detects_overlap_wall_and_swing():
    root = _suite_with(("wardrobe.yaml", 0.0, 0.0, 0), ("nightstand.yaml", 0.2, 0.1, 0))
    assert "overlap" in _kinds(root)
    root = _suite_with(("wardrobe.yaml", 1.95, 0.0, -90))   # half inside the east wall
    assert "outside_room" in _kinds(root)
    # The corridor door swings into the bedroom near its east wall (hinge at z ~ 1.3).
    swing = root.find("bedroom").meta["door_swings"][0]
    hx, _, hz = swing["hinge"]
    root = _suite_with(("armchair.yaml", hx - 0.5, hz - 0.3, 0))
    assert "door_swing" in _kinds(root)


def test_detects_tall_item_in_front_of_window():
    # Bedroom window is on the south wall, sill 0.7 m.
    root = _suite_with(("wardrobe.yaml", 0.0, -1.85, 0))
    assert "window_blocked" in _kinds(root)
    root = _suite_with(("desk.yaml", 0.0, -1.85, 0))   # lower than sill + 0.4: fine
    assert "window_blocked" not in _kinds(root)


def test_detects_unreachable_items():
    # A wardrobe boxed into the south-west corner by two more wardrobes.
    root = _suite_with(("nightstand.yaml", -1.85, -1.9, 0),
                       ("wardrobe.yaml", -1.0, -1.85, 90), ("wardrobe.yaml", -1.75, -1.25, 0))
    # Reach is a straight-line distance, so use a short reach for the check.
    issues = check_layout(root, PlayerSpec(reach=0.6))
    assert any(i.kind == "unreachable" and "nightstand_0" in i.items for i in issues)
    open_room = _suite_with(("nightstand.yaml", -1.85, -1.9, 0))
    assert "unreachable" not in _kinds(open_room, PlayerSpec(reach=0.6))


def test_narrow_doors_use_player_spec():
    root = LayoutLoader().load("assets/hotel_suite.yaml")
    wide_player = PlayerSpec(radius=0.5, door_min_width=1.2, corridor_min_width=1.2)
    assert "narrow_door" in _kinds(root, wide_player)
    assert "narrow_door" not in _kinds(root)


def test_coplanar_overlaps_finds_z_fighting_floors():
    import numpy as np

    from geogen.core.node import SceneNode
    from geogen.core.transform import Transform
    from geogen.generators.primitives import CubeGenerator
    from geogen.layout.qa import coplanar_overlaps

    def slab(name, y0, y1, x=0.0):
        mesh = CubeGenerator(size_x=2, size_y=y1 - y0, size_z=2, bevel=0).generate()
        return SceneNode(name, mesh=mesh, transform=Transform(translation=np.array([x, (y0 + y1) / 2, 0.0])))

    root = SceneNode("root")
    root.add_child(slab("lot", 0.0, 0.15))
    root.add_child(slab("floor", 0.10, 0.15, x=0.5))       # top level with the lot's: z-fights
    issues = coplanar_overlaps(root)
    assert [i.items for i in issues] == [("lot", "floor")] and "y=0.150" in issues[0].message
    root.children[1].transform.translation[1] += 0.003     # 3 mm proud: fine
    assert coplanar_overlaps(root) == []


# --- ways in: dead-end rooms aren't entrances (geogen.23) ----------------------------------------

STREET_DOOR = {"room": "shop", "side": "north", "width": 1.2}


def _shop(doors, rooms=None, counter=True, items=()):
    """A shop (4 x 5 m, north) with a stock room behind it, joined by one door.

    The counter spans the shop, customers' side to the street, so its staff
    side can only be reached through the stock room. ``items`` are
    nightstands, one in each room named.
    """
    from geogen.generators.floorplan import FloorPlan

    spec = {"rooms": {"shop": {"rect": [0, 3, 4, 5]}, "stock": {"rect": [0, 0, 4, 3]}, **(rooms or {})},
            "doors": [{"between": ["shop", "stock"], "at": 0.8, "width": 0.9}, *doors]}
    root = FloorPlan.from_spec(spec).build()
    if counter:
        till = LayoutLoader().load("assets/till_counter.yaml", params={"width": 3.7})
        till.name = "till"
        root.find("shop").add_child(till)
    for room in items:
        node = LayoutLoader().load("assets/nightstand.yaml")
        node.name = f"{room}_item"
        root.find(room).add_child(node)
    return root


def _unreachable(root):
    return sorted({i.items[0] for i in check_layout(root) if i.kind == "unreachable"})


def test_a_dead_end_back_room_is_not_a_way_in():
    from geogen.layout.qa import room_reachability

    root = _shop([STREET_DOOR])
    assert {o["to"] for o in root.find("shop").meta["openings"]} == {"stock", None}
    # The staff side of the counter is cut off from the street: only the stock room's door reaches it.
    issues = [i for i in check_layout(root) if i.kind == "unreachable"]
    assert [i.items for i in issues] == [("till",)] and "where you stand" in issues[0].message
    assert room_reachability(root.find("shop")) == ["till"]          # what the furnishing solver asks


def test_a_chain_of_dead_ends_is_not_a_way_in():
    from geogen.layout.qa import room_reachability

    toilet = {"toilet": {"rect": [4, 0, 2, 3]}}
    doors = [STREET_DOOR, {"between": ["stock", "toilet"], "width": 0.9}]
    root = _shop(doors, toilet)
    assert room_reachability(root.find("shop")) == ["till"]
    assert "till" in _unreachable(root)
    # Without the counter the dead ends are reached through the shop: their one door counts.
    root = _shop(doors, toilet, counter=False, items=("shop", "stock", "toilet"))
    assert _unreachable(root) == []
    for room in ("shop", "stock", "toilet"):
        assert room_reachability(root.find(room)) == []


def test_a_back_room_with_another_way_in_is_an_entrance():
    from geogen.layout.qa import room_reachability

    variants = {
        "a back door": ([STREET_DOOR, {"room": "stock", "side": "south", "width": 0.9}], None),
        "a back door two rooms on": ([STREET_DOOR, {"between": ["stock", "yard"], "width": 0.9},
                                      {"room": "yard", "side": "south", "width": 0.9}],
                                     {"yard": {"rect": [4, 0, 2, 3]}}),
        "a stair room": ([STREET_DOOR], {"stock": {"rect": [0, 0, 4, 3], "type": "stair"}}),
        "a lift": ([STREET_DOOR, {"between": ["stock", "shaft"], "width": 0.9}],
                   {"shaft": {"rect": [4, 0, 2, 3], "type": "lift_shaft"}}),
    }
    for name, (doors, rooms) in variants.items():
        root = _shop(doors, rooms)
        assert _unreachable(root) == [], name
        assert room_reachability(root.find("shop")) == [], name


def test_a_plan_with_no_way_in_keeps_every_door():
    from geogen.layout.qa import room_reachability

    root = _shop([], items=("stock",))       # no street door: nothing says which side is the dead end
    assert _unreachable(root) == []
    assert room_reachability(root.find("shop")) == [] and room_reachability(root.find("stock")) == []


def test_rooms_behind_a_blocked_room_are_unreachable_too():
    # QA floods the storey from its entrances, so what's only reached past the counter is cut off as well.
    root = _shop([STREET_DOOR], items=("stock",))
    assert _unreachable(root) == ["stock_item", "till"]
