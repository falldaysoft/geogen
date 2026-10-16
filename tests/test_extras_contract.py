"""The node extras contract (geogen.extras): the schema is the one declaration of what is
exported, and the lists that mirror it can't drift. Every registered scene is checked
against it in tests/test_asset_quality.py::test_extras_match_closed_schema."""

import re
from pathlib import Path

import pytest

from geogen import extras
from geogen.core.node import SceneNode
from geogen.export import exported_extras, node_extras

ROOT = Path(__file__).parent.parent
GEOGEN = extras.schema()["properties"]["geogen"]

# Keys a runtime reads: their sub-objects list their properties and take no others.
CLOSED = ["room", "door", "joint", "clearance", "light", "portal", "place", "physics", "travel", "container",
          "city", "block", "lot", "building", "lod", "interior", "signal", "character", "stairs",
          "water", "driving", "fleet", "train", "npc", "vehicle", "item", "flows"]


def test_schema_is_closed():
    assert GEOGEN["additionalProperties"] is False
    for key in CLOSED:
        spec = GEOGEN["properties"][key]
        assert spec["type"] == "object" and spec["properties"] and spec["additionalProperties"] is False, key
    interaction = GEOGEN["properties"]["interactions"]["additionalProperties"]
    assert interaction["additionalProperties"] is False and interaction["properties"]["motions"]["items"]["additionalProperties"] is False
    assert extras.schema()["$defs"]["pose"]["additionalProperties"] is False
    for key, spec in GEOGEN["properties"].items():     # nothing is a bare {"type": "object"}
        if spec.get("type") == "object":
            assert "properties" in spec or "additionalProperties" in spec, key
        assert spec.get("description") or key == "version", f"'{key}' needs a description"


def test_every_object_that_lists_its_properties_takes_no_others():
    """Nested too (an NPC's attention, a vehicle's sound): an object either lists its keys and is
    closed, or is a map ('additionalProperties' a schema for each value)."""
    open_ = []

    def walk(spec, path):
        if isinstance(spec, list):
            for i, item in enumerate(spec):
                walk(item, f"{path}[{i}]")
            return
        if not isinstance(spec, dict):
            return
        if "properties" in spec and spec.get("type", "object") == "object" and spec.get("additionalProperties") is not False:
            open_.append(path)
        for key, value in spec.items():
            if key not in ("if", "properties"):         # 'if' only tests a value
                walk(value, f"{path}/{key}")
        for key, value in spec.get("properties", {}).items():
            walk(value, f"{path}.{key}")

    walk(GEOGEN, "geogen")
    walk(extras.schema()["$defs"], "$defs")
    assert not open_, open_


def test_build_only_keys_are_not_declared():
    assert not set(extras.BUILD_ONLY) & extras.declared_keys()
    assert all(why for why in extras.BUILD_ONLY.values())


def test_only_declared_meta_is_exported():
    node = SceneNode("lamp", tags=["decor.lamp"], meta={
        "walkable": True, "collider": "auto",
        "source": {"asset": "lamp.yaml", "params": {}}, "placed_by": "furnish", "openings": []})
    assert node_extras(node) == {"geogen": {"version": 1, "tags": ["decor.lamp"], "walkable": True}}
    assert node_extras(node, "box")["geogen"]["collider"] == "box"
    assert node_extras(SceneNode("empty", meta={"source": {}})) == {}


def test_an_undeclared_meta_key_fails_the_export(tmp_path):
    from geogen.export import export_scene

    root = SceneNode("room")
    root.add_child(SceneNode("shelf", meta={"mystery": 3}))
    with pytest.raises(extras.ExtrasError, match=r"node 'room/shelf': meta key 'mystery'"):
        exported_extras(root.children[0])
    with pytest.raises(extras.ExtrasError, match="mystery"):
        export_scene(root, tmp_path / "room.glb")
    assert extras.scene_problems(root) and "mystery" in extras.scene_problems(root)[0]


def test_scene_problems_reports_what_the_schema_rejects():
    pytest.importorskip("jsonschema")
    root = SceneNode("yard")
    root.add_child(SceneNode("pond", meta={"water": {"foam": 0.4, "colour": [0, 0, 1]}}))
    root.add_child(SceneNode("pond2", meta={"water": {"foam": 0.6, "colour": [0, 1, 1]}}))
    problems = extras.scene_problems(root)
    assert len(problems) == 1 and "yard/pond'" in problems[0] and "geogen/water" in problems[0]
    assert all("uid" not in n.meta for n in root.iter_nodes())


def test_scene_builder_known_keys_are_the_schema_keys():
    """GeogenSceneBuilder warns about extras keys it doesn't know: its list is the schema's."""
    source = (ROOT / "runtime/godot/addons/geogen/scene_builder.gd").read_text()
    listed = re.search(r"const KNOWN_KEYS := \[(.*?)\]", source, re.S).group(1)
    assert re.findall(r'"(\w+)"', listed) == list(GEOGEN["properties"])


def test_runtime_readme_lists_every_key():
    """The 'Node extras' table in runtime/godot/README.md has a row for each key, and no others."""
    readme = (ROOT / "runtime/godot/README.md").read_text()
    section = readme.split("\n## Node extras\n", 1)[1].split("\n## ", 1)[0]
    listed = [key for row in re.findall(r"^\| (`[^|]*) \|", section, re.M) for key in re.findall(r"`(\w+)`", row)]
    assert sorted(listed) == sorted(GEOGEN["properties"])
