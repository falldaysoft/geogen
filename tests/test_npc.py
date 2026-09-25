"""NPC data: definitions, actions, affordance/portal resolution and scene placement."""

import numpy as np
import pytest

from geogen.layout import LayoutLoader, SceneComposer
from geogen.npc import ASSETS_DIR, approach_point, load_actions, load_definition, parse_actions, parse_poses

COTTAGE_NPC = """
name: t
place:
  house: { asset: house_peaked.yaml }
  door: { asset: door.yaml, on: house.front_wall, at: { u: 0.5, v: { abs: 0 } } }
  resident:
    npc: npcs/resident.yaml
    on: house.floor
    at: { u: 0.6, v: 0.6 }
    home: house.floor
    seed: 7
"""


def _npc(root):
    return next(n for n in root.iter_nodes() if n.meta.get("type") == "npc")


def test_actions_file_parses():
    actions = load_actions()
    assert {"sit", "lie", "look", "stand", "use", "pass", "wander", "idle"} <= set(actions)
    assert actions["pass"]["steps"][0] == {"go_to": "near"}
    assert actions["sit"]["duration"] == [15.0, 40.0]


@pytest.mark.parametrize("step, message", [
    ({"go_to": "moon"}, "go_to must be"),
    ({"wait": 2, "face": "anchor"}, "exactly one"),
    ({"dance": 1}, "exactly one"),
    ({"pose": "sit", "at": "ceiling"}, "at must be"),
    ({"wait": [5, 1]}, "lo, hi"),
])
def test_bad_steps_are_rejected(step, message):
    with pytest.raises(ValueError, match=message):
        parse_actions({"kind": "npc_actions", "version": 1, "actions": {"x": {"steps": [step]}}})


def test_definition_resolves():
    d = load_definition(ASSETS_DIR / "npcs" / "resident.yaml")
    assert d["definition"] == "resident"
    assert set(d["needs"]) == {"rest", "curiosity", "fresh_air"}
    assert d["activities"]["wander"]["action"] == "wander"
    assert d["scoring"]["retry"] == 30.0
    assert d["flags"]["closes_doors"] is True


def test_definition_rejects_unknown_need(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("kind: npc\nversion: 1\nneeds: {rest: {}}\n"
                    "activities: {idle: {action: idle, advertises: {hunger: 1}}}\n")
    with pytest.raises(ValueError, match="unknown needs"):
        load_definition(path)


def test_poses_need_stand():
    with pytest.raises(ValueError, match="stand"):
        parse_poses({"sit": {"scale": [1, 0.5, 1]}})
    assert parse_poses({"stand": {}})["stand"]["scale"] == [1.0, 1.0, 1.0]


def test_approach_points_follow_the_anchor_heading():
    # sit: 0.5 m in front of the seat; a seat facing +X puts it at +X.
    assert approach_point("sit", np.array([1.0, 0.45, 0.0]), 90.0) == pytest.approx([1.5, 0.0, 0.0])
    # An explicit [x, y, z] is in the anchor frame (x right of the facing).
    assert approach_point("sit", np.zeros(3), 0.0, [0.6, 0, 0.15]) == pytest.approx([0.6, 0.0, 0.15])
    assert approach_point("look", np.zeros(3), 180.0, 0.4) == pytest.approx([0.0, 0.0, -0.4])


def test_affordances_carry_npc_fields():
    chair = LayoutLoader().load(ASSETS_DIR / "chair.yaml")
    (a,) = chair.meta["affordances"]
    assert a["action"] == "sit" and a["tags"] == ["seat", "table"] and a["slots"] == 1
    assert a["approach"] == pytest.approx([0.6, 0.0, 0.18])
    window = LayoutLoader().load(ASSETS_DIR / "window.yaml")
    assert window.meta["affordances"][0]["action"] == "look"


def test_door_portal_references_its_interaction():
    door = LayoutLoader().load(ASSETS_DIR / "door.yaml")
    portal = door.meta["portal"]
    assert portal["interaction"] == "swing" and portal["open"] == "open"
    assert portal["normal"] == [0.0, 0.0, 1.0]
    assert portal["width"] == pytest.approx(0.95 - 2 * 0.07)


def test_npc_placement_builds_body_and_home():
    root = SceneComposer().compose_string(COTTAGE_NPC)
    npc = _npc(root)
    data = npc.meta["npc"]
    assert npc.name == "resident" and npc.tags == ["npc"]
    assert [c.name for c in npc.children] == ["body"]
    assert data["seed"] == 7
    assert data["height"] == pytest.approx(1.75) and data["radius"] == pytest.approx(0.24)
    assert set(data["body"]["poses"]) == {"stand", "sit", "lie"}
    assert "pass" in data["actions"]
    # The home polygon (house floor, 7.4 x 5.4 m) is in the NPC's frame: it contains the NPC.
    poly = np.array(data["home"]["polygon"])
    assert poly.shape == (4, 2)
    assert poly[:, 0].min() < 0 < poly[:, 0].max() and poly[:, 1].min() < 0 < poly[:, 1].max()
    assert np.ptp(poly[:, 0]) * np.ptp(poly[:, 1]) == pytest.approx(7.4 * 5.4, rel=0.01)
    # Body parts don't collide (the runtime gives the NPC its own capsule).
    assert all(n.meta.get("collider") == "none" for n in npc.iter_nodes() if n.mesh is not None)


def test_home_must_be_a_horizontal_surface():
    with pytest.raises(ValueError, match="horizontal"):
        SceneComposer().compose_string(COTTAGE_NPC.replace("home: house.floor", "home: house.front_wall"))


def test_npc_exports_extras(tmp_path):
    import json
    import struct

    from geogen.export import export_scene

    path = tmp_path / "t.glb"
    export_scene(SceneComposer().compose_string(COTTAGE_NPC), path)
    data = path.read_bytes()
    gltf = json.loads(data[20:20 + struct.unpack("<I", data[12:16])[0]])
    npc = next(n for n in gltf["nodes"] if n.get("extras", {}).get("geogen", {}).get("type") == "npc")
    data = npc["extras"]["geogen"]["npc"]
    assert data["definition"] == "resident"
    names = {gltf["nodes"][i]["name"] for i in npc["children"]}
    assert "body" in names
