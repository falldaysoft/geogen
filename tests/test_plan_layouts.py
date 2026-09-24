"""Procedural floor-plan layouts."""

import pytest

from geogen.core import meshops
from geogen.generators.floorplan import FloorPlan
from geogen.generators.plan_layouts import expand, hotel_corridor
from geogen.layout.qa import check_layout


def _rooms_of(root, room_type):
    return [n for n in root.iter_nodes()
            if isinstance(n.meta.get("room"), dict) and n.meta.get("type") != "room_volume"
            and n.meta["room"]["type"] == room_type]


@pytest.mark.parametrize("spec", [
    {"generate": "hotel_corridor", "length": 30, "depth": 15, "module": 3.8},
    {"generate": "hotel_corridor", "length": 40, "depth": 16, "module": 3.6, "ensuite": False},
    {"generate": "hotel_lobby", "length": 30, "depth": 15},
])
def test_layouts_build_clean_plans(spec):
    root = FloorPlan.from_spec(spec).build("plan")
    assert meshops.validate(root.find("walls").mesh).watertight
    assert [str(i) for i in check_layout(root)] == []


def test_corridor_room_count_and_modules():
    spec = hotel_corridor(length=40, depth=16, module=3.6, stair_width=3.0, core_width=4.8)
    bedrooms = [n for n, r in spec["rooms"].items() if r["type"] == "hotel_bedroom"]
    # (40 - 6 - 4.8) / 2 // 3.6 = 4 modules per half, two halves, two sides.
    assert len(bedrooms) == 16
    assert all(spec["rooms"][n]["rect"][2] == pytest.approx(3.6) for n in bedrooms)
    root = FloorPlan.from_spec({"generate": "hotel_corridor", "length": 40, "depth": 16}).build("plan")
    assert len(_rooms_of(root, "hotel_bathroom")) == 16
    # Every guest room has a window on the façade and a door to the corridor.
    for room in _rooms_of(root, "hotel_bedroom"):
        kinds = {o["kind"] for o in room.meta["openings"]}
        assert kinds == {"door", "window"}


def test_lobby_has_street_entrance():
    root = FloorPlan.from_spec({"generate": "hotel_lobby"}).build("lobby")
    lobby = _rooms_of(root, "lobby")[0]
    assert any(o["kind"] == "door" and o["side"] == "south" and o["hi"] - o["lo"] == pytest.approx(2.0)
               for o in lobby.meta["openings"])
    assert root.find("entrance").tags == ["door", "door.exterior"]


def test_plan_keys_pass_through_and_unknown_keys_raise():
    spec = expand({"generate": "hotel_lobby", "wall_height": 3.6, "finishes": False})
    assert spec["wall_height"] == 3.6 and spec["finishes"] is False and "rooms" in spec
    with pytest.raises(ValueError, match="Unknown keys"):
        expand({"generate": "hotel_lobby", "nope": 1})
    with pytest.raises(ValueError, match="Unknown floor plan layout"):
        expand({"generate": "castle"})


def test_too_short_for_a_module():
    with pytest.raises(ValueError, match="too short"):
        hotel_corridor(length=12)


def test_generated_floor_furnishes_cleanly():
    from geogen.layout import SceneComposer
    scene = SceneComposer().compose("assets/scenes/hotel_floor_furnished.yaml")
    assert scene.find("floor").meta["furnish_report"] == []
    assert [str(i) for i in check_layout(scene)] == []
