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
