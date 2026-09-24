"""Interior finishes on floor-plan rooms."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.generators.floorplan import FloorPlan
from geogen.layout import LayoutLoader

SPEC = {
    "rooms": {
        "bedroom": {"rect": [0, 0, 4.4, 4.6], "type": "hotel_bedroom"},
        "bathroom": {"rect": [4.4, 0, 2.4, 2.6], "type": "hotel_bathroom"},
        "corridor": {"rect": [4.4, 2.6, 2.4, 2.0], "type": "corridor"},
    },
    "doors": [{"between": ["corridor", "bedroom"], "width": 0.9}],
    "windows": [{"room": "bedroom", "side": "south", "width": 2.0, "height": 1.5, "sill": 0.7}],
}


def _world(node):
    v = node.mesh.vertices
    return (node.world_transform() @ np.c_[v, np.ones(len(v))].T).T[:, :3]


def test_rooms_get_finishes_and_type_materials():
    root = FloorPlan.from_spec(SPEC).build("suite")
    bedroom, bathroom = root.find("bedroom"), root.find("bathroom")
    for room in (bedroom, bathroom):
        names = {c.name for c in room.children}
        assert {"lining", "skirting", "cornice", f"{room.name}_light"} <= names
        for part in ("lining", "skirting", "cornice"):
            assert meshops.validate(room.find(part).mesh).watertight, (room.name, part)
    # Materials come from the room types' finishes.
    assert bedroom.find("lining").mesh.material.name == "paint_greige"
    assert bedroom.find("floor").mesh.material.name == "carpet"
    assert bathroom.find("lining").mesh.material.name == "tile_white"
    assert bathroom.find("floor").mesh.material.name == "tile_floor_grey"


def test_room_keys_override_type_finishes():
    spec = dict(SPEC, rooms=dict(SPEC["rooms"], bedroom={**SPEC["rooms"]["bedroom"], "walls": "paint_sage"}))
    root = FloorPlan.from_spec(spec).build("suite")
    assert root.find("bedroom").find("lining").mesh.material.name == "paint_sage"


def test_skirting_breaks_at_doorways_and_lining_is_cut():
    root = FloorPlan.from_spec(SPEC).build("suite")
    bedroom = root.find("bedroom")
    door = root.find("door_corridor_bedroom")
    door_z = door.world_transform()[2, 3]
    skirting = _world(bedroom.find("skirting"))
    east = skirting[skirting[:, 0] > skirting[:, 0].max() - 0.02]
    # No skirting across the door (0.9 m wide) or its architraves.
    assert not np.any(np.abs(east[:, 2] - door_z) < 0.45 + 0.05)
    lining = _world(bedroom.find("lining"))
    east_lining = lining[lining[:, 0] > lining[:, 0].max() - 0.02]
    below_head = east_lining[east_lining[:, 1] < 2.1 - 1e-6]
    # Below the door head, no lining vertex lies inside the 0.9 m opening.
    assert np.all(np.abs(below_head[:, 2] - door_z) >= 0.45 - 1e-6)


def test_light_and_switches():
    root = FloorPlan.from_spec(SPEC).build("suite")
    bedroom = root.find("bedroom")
    light = bedroom.find("bedroom_light")
    assert light.meta["light"]["type"] == "omni"
    switch = bedroom.find("bedroom_switch_1")
    assert switch.meta["switch"] == {"light": "bedroom_light"}
    assert switch.world_transform()[1, 3] == pytest.approx(1.1)
    door_z = root.find("door_corridor_bedroom").world_transform()[2, 3]
    assert abs(switch.world_transform()[2, 3] - door_z) > 0.45  # beside, not over, the door


def test_finishes_can_be_turned_off():
    bedroom = FloorPlan.from_spec(dict(SPEC, finishes=False)).build("suite").find("bedroom")
    assert bedroom.find("skirting") is None and bedroom.find("lining") is None
    bedroom = FloorPlan.from_spec(dict(SPEC, finishes={"cornice": False})).build("suite").find("bedroom")
    assert bedroom.find("cornice") is None and bedroom.find("skirting") is not None


def test_furnished_items_stand_clear_of_skirting():
    from geogen.layout import SceneComposer
    scene = SceneComposer().compose("assets/scenes/hotel_room_auto.yaml")
    bedroom = scene.find("suite").find("bedroom")
    half = bedroom.size[[0, 2]] / 2
    for item in ("bed", "wardrobe", "tv_console"):
        node = bedroom.find(item)
        pts = np.concatenate([_world(n) for n in node.iter_nodes() if n.mesh is not None])
        local = pts - bedroom.world_transform()[:3, 3]
        assert np.all(np.abs(local[:, [0, 2]]) <= half - 0.024 + 1e-6), item


def test_cutaway_removes_ceilings_and_fixtures():
    from geogen.render import cutaway
    root = FloorPlan.from_spec(SPEC).build("suite")
    cut = cutaway(root)
    names = {n.name for n in cut.iter_nodes()}
    assert "ceiling" not in names and "bedroom_light" not in names and "cornice" not in names
    assert "skirting" in names and root.find("ceiling") is not None  # original untouched
