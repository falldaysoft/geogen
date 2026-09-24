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
    exported = [n for n in gltf["nodes"] if "interactions" in n.get("extras", {}).get("geogen", {})]
    assert len(exported) == 3
    # Exported node names are unique, and each interaction points at its own leaf pivot.
    pivots = [n["extras"]["geogen"]["interactions"]["swing"]["motions"][0]["nodes"][0] for n in exported]
    assert len(set(pivots)) == 3
