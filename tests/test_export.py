"""Round-trip tests for glTF/GLB and OBJ export."""

import json

import numpy as np
import pytest
import trimesh

from geogen.export import export_scene, manifest_path
from geogen.player import PlayerSpec, load_player_spec
from geogen.render import scene_bounds


def _visual(geometry: dict) -> dict:
    """Geometry minus the collider-only meshes (Godot -colonly/-convcolonly)."""
    return {k: g for k, g in geometry.items() if "-col" not in k and "-convcol" not in k}


@pytest.mark.parametrize("scene", ["chair", "dining_set", "house_peaked", "room"])
def test_glb_round_trip_preserves_geometry(built_scene, scene, tmp_path):
    root = built_scene(scene)
    path = export_scene(root, tmp_path / f"{scene}.glb")
    loaded = trimesh.load(path)
    np.testing.assert_allclose(loaded.bounds, scene_bounds(root), atol=1e-4)
    src_tris = sum(len(m.faces) for _, m in root.iter_meshes())
    # Instances share one glTF mesh, so count per node.
    nodes = [n for n in loaded.graph.nodes_geometry if "-col" not in n and "-convcol" not in n]
    assert sum(len(loaded.geometry[loaded.graph[n][1]].faces) for n in nodes) == src_tris
    # Hierarchy: every scene node with a mesh is its own glTF node.
    assert len(nodes) == sum(1 for _, m in root.iter_meshes() if len(m.faces))


def test_glb_has_textured_pbr_materials(built_scene, tmp_path):
    loaded = trimesh.load(export_scene(built_scene("table"), tmp_path / "table.glb"))
    for geom in _visual(loaded.geometry).values():
        assert isinstance(geom.visual, trimesh.visual.TextureVisuals)
        mat = geom.visual.material
        assert mat.baseColorTexture is not None
        assert mat.metallicRoughnessTexture is not None
    # Texture-space UVs: a 1.2 m tabletop with a 0.8 m wood tile spans > 1 repeat.
    top = loaded.geometry["top"]
    assert np.ptp(top.visual.uv, axis=0).max() > 1.2


def test_obj_export_writes_material_and_textures(built_scene, tmp_path):
    path = export_scene(built_scene("chair"), tmp_path / "chair.obj")
    assert path.exists()
    assert any(p.suffix == ".mtl" for p in tmp_path.iterdir())
    assert any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_unknown_format_rejected(built_scene, tmp_path):
    with pytest.raises(ValueError):
        export_scene(built_scene("chair"), tmp_path / "chair.fbx")


def test_glb_export_writes_manifest_with_player_spec(built_scene, tmp_path):
    path = export_scene(built_scene("chair"), tmp_path / "chair.glb")
    manifest = json.loads(manifest_path(path).read_text())
    assert manifest["format"] == "geogen-manifest"
    assert manifest["model"] == "chair.glb"
    assert manifest["units"] == "m"
    assert PlayerSpec.from_dict(manifest["player"]) == load_player_spec()


def test_obj_export_has_no_manifest(built_scene, tmp_path):
    path = export_scene(built_scene("chair"), tmp_path / "chair.obj")
    assert not manifest_path(path).exists()


def _gltf_json(path):
    import struct
    data = path.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


def test_node_extras_match_schema(built_scene, tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    from pathlib import Path
    schema = json.loads((Path(__file__).parent.parent / "docs/schema/geogen-extras.v1.schema.json").read_text())
    for scene in ("hotel_suite", "cottage", "dining_set"):
        gltf = _gltf_json(export_scene(built_scene(scene), tmp_path / f"{scene}.glb"))
        extras = [n["extras"] for n in gltf["nodes"] if "extras" in n]
        assert extras, scene
        for e in extras:
            jsonschema.validate(e, schema)


def test_colliders_are_godot_suffixed_children(built_scene, tmp_path):
    gltf = _gltf_json(export_scene(built_scene("hotel_suite"), tmp_path / "suite.glb"))
    nodes = gltf["nodes"]
    by_name = {n["name"]: n for n in nodes}
    walls = by_name["walls"]
    assert walls["extras"]["geogen"]["collider"] == "mesh"  # walls have openings: trimesh
    child_names = [nodes[i]["name"] for i in walls.get("children", [])]
    assert child_names == ["walls-colonly"]
    assert by_name["walls-colonly"]["extras"]["geogen"] == {"version": 1, "type": "collider", "shape": "mesh"}
    floor = by_name["floor"]["extras"]["geogen"]
    assert floor["collider"] == "box" and floor["walkable"] is True
    assert "floor-convcolonly" in by_name


def test_collider_override_and_auto_policy(built_scene):
    from geogen.core.node import SceneNode
    from geogen.export import resolve_collider
    from geogen.generators.primitives import CubeGenerator, SphereGenerator

    cube = SceneNode("c", mesh=CubeGenerator(bevel=0).generate())
    assert resolve_collider(cube) == "box"
    ball = SceneNode("s", mesh=SphereGenerator(radius=0.3).generate())
    assert resolve_collider(ball) == "hull"
    tiny = SceneNode("t", mesh=CubeGenerator(size_x=0.02, size_y=0.02, size_z=0.02).generate())
    assert resolve_collider(tiny) == "none"
    ball.meta["collider"] = "none"
    assert resolve_collider(ball) == "none"
    hollow = built_scene("house_peaked").find("walls")
    assert resolve_collider(hollow) == "mesh"


def test_manifest_lists_rooms_and_spawns(tmp_path):
    from geogen.layout import SceneComposer
    root = SceneComposer().compose_string("""
name: s
tags: [level.test]
place:
  suite: { asset: hotel_suite.yaml }
spawns:
  start: { position: [2.2, 0, 1.3], facing: west }
""")
    manifest = json.loads(manifest_path(export_scene(root, tmp_path / "s.glb")).read_text())
    assert [r["id"] for r in manifest["rooms"]] == ["bedroom", "bathroom", "corridor"]
    bedroom = manifest["rooms"][0]
    assert bedroom["type"] == "hotel_bedroom" and bedroom["size"][1] == pytest.approx(2.5)
    assert manifest["spawns"][0]["name"] == "start"
    assert manifest["spawns"][0]["position"] == pytest.approx([2.2, 0, 1.3])
    assert manifest["spawns"][0]["forward"] == pytest.approx([-1, 0, 0], abs=1e-6)
    assert manifest["extras"]["version"] == 1


def _read_accessor(glb: bytes, gltf: dict, index: int) -> np.ndarray:
    import struct

    length = struct.unpack("<I", glb[12:16])[0]
    rest = glb[20 + length:]
    data = rest[8:8 + struct.unpack("<I", rest[:4])[0]]
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return np.frombuffer(data[start:start + acc["count"] * width * 4], dtype="<f4").reshape(acc["count"], width)


def test_interaction_transitions_export_as_gltf_animations(built_scene, tmp_path):
    from geogen.layout.interactions import apply_state

    root = built_scene("cottage")
    path = export_scene(root, tmp_path / "cottage.glb")
    glb = path.read_bytes()
    gltf = _gltf_json(path)
    names = {a["name"]: a for a in gltf["animations"]}
    assert {"door/swing/closed->open", "door/swing/open->closed"} <= set(names)
    anim = names["door/swing/closed->open"]
    nodes = gltf["nodes"]
    leaf_index = next(i for i, n in enumerate(nodes) if n.get("name") == "leaf")
    assert "matrix" not in nodes[leaf_index] and "rotation" in nodes[leaf_index]
    rotation = next(c for c in anim["channels"] if c["target"] == {"node": leaf_index, "path": "rotation"})
    translation = next(c for c in anim["channels"] if c["target"] == {"node": leaf_index, "path": "translation"})
    times = _read_accessor(glb, gltf, anim["samplers"][rotation["sampler"]]["input"])
    assert times[0, 0] == 0.0 and times[-1, 0] == pytest.approx(0.9)
    assert gltf["accessors"][anim["samplers"][rotation["sampler"]]["input"]]["max"] == [pytest.approx(0.9)]
    quats = _read_accessor(glb, gltf, anim["samplers"][rotation["sampler"]]["output"])
    moves = _read_accessor(glb, gltf, anim["samplers"][translation["sampler"]]["output"])
    np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-5)
    # The first key is the exported (closed) pose; the last is the pose apply_state gives for "open".
    np.testing.assert_allclose(quats[0], nodes[leaf_index]["rotation"], atol=1e-5)
    door = next(n for n in root.iter_nodes() if any(i.name == "swing" for i in n.interactions))
    apply_state(door, door.interactions[0], "open")
    leaf = next(n for n in door.iter_nodes() if n.name == "leaf")
    from geogen.export import _quat

    expected = _quat(leaf.transform.to_matrix()[:3, :3])
    assert min(np.abs(quats[-1] - expected).max(), np.abs(quats[-1] + expected).max()) < 1e-4
    np.testing.assert_allclose(moves[-1], leaf.transform.translation, atol=1e-4)


def test_animations_can_be_disabled(built_scene, tmp_path):
    path = export_scene(built_scene("cottage"), tmp_path / "plain.glb", animations=False)
    assert "animations" not in _gltf_json(path)
