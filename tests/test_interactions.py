"""Declarative interactions: parsing, posing, and export into extras.geogen."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from geogen.export import export_scene
from geogen.layout import LayoutLoader, SceneComposer
from geogen.layout.interactions import apply_state

SCHEMA = Path(__file__).parent.parent / "docs/schema/geogen-extras.v1.schema.json"


def _gltf_json(path):
    data = Path(path).read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


def _world_points(node):
    v = node.mesh.vertices
    return (node.world_transform() @ np.c_[v, np.ones(len(v))].T).T[:, :3]


def test_door_swing_parses_and_opens_inward_about_hinge():
    door = LayoutLoader().load("assets/door.yaml")
    (swing,) = door.interactions
    assert swing.name == "swing" and swing.initial == "closed"
    assert [n.name for n in swing.targets] == ["leaf", "knob", "knob_inside"]
    motion = swing.motions[0]
    assert motion.kind == "rotate" and motion.values == {"open": 100.0}
    hinge = motion.pivot.copy()

    leaf = door.find("leaf")
    closed = _world_points(leaf)
    apply_state(door, swing, "open")
    opened = _world_points(leaf)
    # Swings in (-Z, into the house) and stays on the hinge side.
    assert opened[:, 2].min() < closed[:, 2].min() - 0.7
    assert opened[:, 0].max() < hinge[0] + 0.1
    # Points on the hinge axis don't move.
    near_hinge = np.linalg.norm(closed[:, [0, 2]] - hinge[[0, 2]], axis=1) < 0.03
    assert near_hinge.any()
    assert np.allclose(opened[near_hinge][:, [0, 2]], closed[near_hinge][:, [0, 2]], atol=0.03)
    apply_state(door, swing, "closed")
    assert np.allclose(_world_points(leaf), closed, atol=1e-9)


def test_initial_state_poses_parts():
    yaml = Path("assets/door.yaml").read_text().replace("initial: closed", "initial: open")
    open_door = LayoutLoader().load_string(yaml)
    closed_door = LayoutLoader().load("assets/door.yaml")
    assert _world_points(open_door.find("leaf"))[:, 2].min() < _world_points(closed_door.find("leaf"))[:, 2].min() - 0.7


@pytest.mark.parametrize("block, match", [
    ({"s": {"motions": [{"parts": ["a"], "rotate": "y"}]}}, "states"),
    ({"s": {"states": {"a": {"next": "zz"}}, "motions": [{"parts": ["a"], "rotate": "y"}]}}, "unknown state"),
    ({"s": {"states": {"a": {}}, "motions": [{"parts": ["nope"], "rotate": "y"}]}}, "unknown part"),
    ({"s": {"states": {"a": {}}, "motions": [{"parts": ["a"], "rotate": "q"}]}}, "axis"),
    ({"s": {"states": {"a": {}}, "motions": [{"parts": ["a"]}]}}, "rotate/translate"),
])
def test_bad_interactions_raise(block, match):
    import yaml as pyyaml
    asset = {"name": "t", "size": [1, 1, 1],
             "parts": {"a": {"primitive": "cube", "size": [1, 1, 1], "anchor": "bottom_center"}},
             "interactions": block}
    with pytest.raises(ValueError, match=match):
        LayoutLoader().load_string(pyyaml.safe_dump(asset))


def test_interactions_exported_with_node_names(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    scene = SceneComposer().compose(Path("assets/scenes/cottage.yaml"))
    gltf = _gltf_json(export_scene(scene, tmp_path / "cottage.glb"))
    names = {n["name"] for n in gltf["nodes"]}
    door = next(n for n in gltf["nodes"] if n["name"] == "door")
    swing = door["extras"]["geogen"]["interactions"]["swing"]
    assert swing["initial"] == "closed"
    assert swing["states"]["closed"] == {"next": "open", "prompt": "Open"}
    assert set(swing["motions"][0]["nodes"]) <= names
    assert swing["motions"][0]["values"] == {"closed": 0.0, "open": 100.0}
    jsonschema.validate(door["extras"], json.loads(SCHEMA.read_text()))


def test_floorplan_doors_get_swing_interactions(tmp_path):
    root = LayoutLoader().load("assets/hotel_suite.yaml")
    entry = root.find("entry")
    (swing,) = entry.interactions
    assert swing.initial == "open"  # the suite opens its entry door 90 degrees
    assert swing.motions[0].parts[0].name == "leaf_pivot"
    bath = root.find("door_corridor_bathroom")
    assert bath.interactions[0].initial == "closed"
    gltf = _gltf_json(export_scene(root, tmp_path / "suite.glb"))
    exported = [n for n in gltf["nodes"] if "swing" in n.get("extras", {}).get("geogen", {}).get("interactions", {})]
    assert len(exported) == 3                                  # the three doors (switches have their own)
    # Exported node names are unique, and each interaction points at its own leaf pivot.
    pivots = [n["extras"]["geogen"]["interactions"]["swing"]["motions"][0]["nodes"][0] for n in exported]
    assert len(set(pivots)) == 3


CABINET = """
name: cabinet
size: [0.6, 0.8, 0.4]
parts:
  body: { primitive: cube, size: [1, 1, 0.95], anchor: bottom_center, offset: [0, 0, -0.025] }
  door:
    primitive: cube
    size: [0.98, 0.95, 0.05]
    anchor: bottom_center
    offset: [0, 0.02, 0.475]
    joint: { type: hinge, pivot: left, limits: [0, 110], with: [knob] }
  knob: { primitive: sphere, size: [0.06, 0.06, 0.06], anchor: bottom_center, offset: [0.4, 0.5, 0.55] }
  drawer:
    primitive: cube
    size: [0.9, 0.1, 0.9]
    anchor: bottom_center
    offset: [0, 0.9, 0]
    joint: { type: slide, range: [0, 0.3] }
"""


def test_joint_shorthand_makes_interactions():
    root = LayoutLoader().load_string(CABINET)
    by_name = {i.name: i for i in root.interactions}
    door, drawer = by_name["door"], by_name["drawer"]
    assert list(door.states) == ["closed", "open"] and door.initial == "closed"
    assert [p.name for p in door.motions[0].parts] == ["door", "knob"]
    assert door.motions[0].pivot[0] == pytest.approx(-0.294, abs=1e-3)  # the door's left face
    before = _world_points(root.find("door"))
    apply_state(root, door, "open")
    after = _world_points(root.find("door"))
    assert after[:, 2].max() > before[:, 2].max() + 0.4    # swings out toward +Z
    assert drawer.motions[0].kind == "translate" and drawer.motions[0].values == {"closed": 0.0, "open": 0.3}
    apply_state(root, drawer, "open")
    assert root.find("drawer").world_transform()[2, 3] == pytest.approx(0.3, abs=1e-6)


def test_bad_joint_type():
    with pytest.raises(ValueError, match="joint type"):
        LayoutLoader().load_string(CABINET.replace("type: slide", "type: ball"))
