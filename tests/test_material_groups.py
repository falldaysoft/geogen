"""Per-face material groups and vertex colours through mesh ops, CSG, export and rendering."""

import json
import struct

import numpy as np
import pytest

from geogen.core import csg, meshops, uvmap
from geogen.core.mesh import Mesh
from geogen.core.node import SceneNode
from geogen.core.subdiv import subdivide
from geogen.generators.primitives import CubeGenerator
from geogen.materials.loader import MaterialLoader

LOADER = MaterialLoader()


def _cube(x=0.0, material="brick"):
    m = np.eye(4)
    m[0, 3] = x
    mesh = CubeGenerator(size_x=1, size_y=1, size_z=1, bevel=0).generate().transform(m)
    mesh.material = LOADER.load(material)
    return mesh


def _names(mesh):
    return sorted((mat.name, len(sub.faces)) for mat, sub in mesh.groups())


def test_merge_makes_groups_only_when_materials_differ():
    same = Mesh.merge([_cube(), _cube(2)])
    assert not same.multi_material and same.material.name == "brick"
    mixed = Mesh.merge([_cube(), _cube(2, "wood")])
    assert mixed.multi_material and _names(mixed) == [("brick", 12), ("wood", 12)]
    # Groups compact their vertices and keep UVs.
    for _, sub in mixed.groups():
        assert len(sub.vertices) == 24 and sub.uvs is not None


def test_ops_keep_slots_and_colours():
    mixed = Mesh.merge([_cube(), _cube(2, "wood")])
    mixed.colors = np.tile([1.0, 0.5, 0.25, 1.0], (len(mixed.vertices), 1))
    for op in (lambda m: m.transform(np.eye(4) * 2), lambda m: m.copy(), meshops.weld_vertices,
               lambda m: meshops.compute_normals(m, 30), uvmap.box_project):
        out = op(mixed)
        assert _names(out) == [("brick", 12), ("wood", 12)]
        assert out.colors is not None and len(out.colors) == len(out.vertices)
        np.testing.assert_allclose(out.colors[0], [1.0, 0.5, 0.25, 1.0])
    sub = subdivide(mixed, 2)
    assert _names(sub) == [("brick", 192), ("wood", 192)]


def test_csg_keeps_materials():
    union = csg.union(_cube(), _cube(0.7, "wood"))
    assert {mat.name for mat, _ in union.groups()} == {"brick", "wood"}
    assert meshops.validate(union).ok
    hole = CubeGenerator(size_x=0.3, size_y=0.3, size_z=3, bevel=0).generate()
    cut = csg.difference(union, hole)
    assert {mat.name for mat, _ in cut.groups()} == {"brick", "wood"}
    assert len(cut.faces) > len(union.faces) and meshops.validate(cut).ok
    reduced = meshops.decimate(meshops.compute_normals(subdivide(union, 2), 30), 0.5)
    assert {mat.name for mat, _ in reduced.groups()} == {"brick", "wood"}
    # A single-material target stays single-material.
    assert not csg.difference(_cube(), hole).multi_material


def _gltf(path):
    data = path.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


def test_export_writes_one_primitive_per_group_and_color_0(tmp_path):
    from geogen.export import export_scene

    mixed = Mesh.merge([_cube(), _cube(2, "wood")])
    mixed.colors = np.tile([1.0, 0.4, 0.4, 1.0], (len(mixed.vertices), 1))
    root = SceneNode("root")
    for i in range(2):   # two instances share the mesh
        root.add_child(SceneNode(f"box{i}", mesh=mixed))
    gltf = _gltf(export_scene(root, tmp_path / "groups.glb", manifest=False))
    nodes = {n["name"]: n for n in gltf["nodes"]}
    box = gltf["meshes"][nodes["box0"]["mesh"]]
    assert nodes["box1"]["mesh"] == nodes["box0"]["mesh"]
    assert len(box["primitives"]) == 2
    assert len({p["material"] for p in box["primitives"]}) == 2
    assert all("COLOR_0" in p["attributes"] for p in box["primitives"])
    # The helper group nodes are no longer part of the hierarchy.
    referenced = {c for n in gltf["nodes"] for c in n.get("children", [])}
    assert all(i not in referenced for i, n in enumerate(gltf["nodes"]) if "__group" in n.get("name", ""))


def test_pyrender_draws_each_group():
    from geogen.render import SceneRenderer

    mixed = Mesh.merge([_cube(), _cube(2, "wood")])
    root = SceneNode("root")
    root.add_child(SceneNode("box", mesh=mixed))
    renderer = SceneRenderer(root)
    primitives = [p for m in renderer.scene.meshes for p in m.primitives]
    assert len([p for p in primitives if p.material is not None and p.material.name in ("brick", "wood")]) == 2
