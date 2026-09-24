"""Scale features: instancing, shared glTF meshes, chunked export, shared textures, disk cache."""

import json
import struct

import numpy as np
import pytest

from geogen.chunks import export_chunks, find_units, split_interiors
from geogen.core.mesh import Mesh
from geogen.core.node import SceneNode
from geogen.core.transform import Transform
from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from geogen.layout.loader import LayoutLoader

TOWN = """
name: mini_town
city:
  seed: 1
  blocks: [2, 1]
  block_size: [30, 32]
  avenues: {{ ew: [0] }}
  lot_width: [10, 14]
  buildings:
    residential: [{{ asset: house_simple.yaml, params: {{ width: 7, depth: 7 }} }}]
    commercial: [{{ asset: shop.yaml, setback: 0 }}]
  furniture:
    lamp: {{ asset: street_lamp.yaml, spacing: 15 }}
{extra}
"""


def _gltf(path):
    data = path.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


@pytest.fixture(scope="module")
def town():
    return SceneComposer().compose_string(TOWN.format(extra=""))


def test_instance_shares_meshes_and_remaps_interactions():
    shop = LayoutLoader().load("assets/shop.yaml")
    copy = shop.instance()
    originals = {id(n.mesh) for n in shop.iter_nodes() if n.mesh is not None}
    assert {id(n.mesh) for n in copy.iter_nodes() if n.mesh is not None} == originals
    copied_nodes = {id(n) for n in copy.iter_nodes()}
    for node in copy.iter_nodes():
        for interaction in node.interactions:
            assert all(id(t) in copied_nodes for t in interaction.targets)
    copy.transform = Transform(translation=np.array([5.0, 0, 0]))
    assert shop.transform.translation[0] == 0.0


def test_repeated_loads_share_meshes():
    loader = LayoutLoader()
    a, b = loader.load("assets/bed.yaml"), loader.load("assets/bed.yaml")
    c = loader.load("assets/bed.yaml", params={"width": 1.4})
    mesh = lambda n: next(x.mesh for x in n.iter_nodes() if x.mesh is not None)  # noqa: E731
    assert a is not b and mesh(a) is mesh(b)
    assert mesh(c) is not mesh(a)


def test_export_writes_shared_meshes_once(tmp_path):
    box = Mesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]),
               faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]))
    root = SceneNode("root")
    for i in range(5):
        root.add_child(SceneNode(f"n{i}", mesh=box, transform=Transform(translation=np.array([2.0 * i, 0, 0])),
                                 meta={"collider": "hull", "index": i}))
    gltf = _gltf(export_scene(root, tmp_path / "shared.glb"))
    named = {n["name"]: n for n in gltf["nodes"]}
    assert len(gltf["meshes"]) == 2                      # the mesh and its hull collider
    assert {named[f"n{i}"]["mesh"] for i in range(5)} == {named["n0"]["mesh"]}
    assert [named[f"n{i}"]["extras"]["geogen"]["index"] for i in range(5)] == list(range(5))
    assert all(f"n{i}-convcolonly" in named for i in range(5))


def test_split_interiors_moves_rooms_out(town):
    unit = next(u for u in find_units(town) if any("building.shop" in n.tags for n in u.iter_nodes()))
    exterior = unit.instance()
    exterior.transform = Transform.from_matrix(unit.world_transform())
    world_floor = next(n for n in exterior.iter_nodes() if "floor" in n.tags).world_transform()
    interiors = split_interiors(exterior)
    assert interiors
    assert not any(isinstance(n.meta.get("room"), dict) for n in exterior.iter_nodes())
    _, interior = interiors[0]
    floor = next(n for n in interior.iter_nodes() if "floor" in n.tags)
    np.testing.assert_allclose(floor.world_transform(), world_floor, atol=1e-9)
    assert any("door.exterior" in n.tags for n in exterior.iter_nodes())


def test_chunked_export(town, tmp_path):
    index_path = export_chunks(town, tmp_path, name="mini_town")
    index = json.loads(index_path.read_text())
    assert index["format"] == "geogen-chunks" and index["version"] == 1
    names = [c["name"] for c in index["chunks"]]
    assert names[0] == "base" and {"block_0_0", "block_1_0"} <= set(names)
    for chunk in index["chunks"]:
        assert (tmp_path / chunk["file"]).exists()
        lo, hi = np.array(chunk["bounds"])
        assert (hi > lo).all()
        for interior in chunk["interiors"]:
            assert (tmp_path / interior["file"]).exists()
    blocks = [c for c in index["chunks"] if c["name"].startswith("block")]
    assert all("lod" in c for c in blocks) and any(c["interiors"] for c in blocks)
    assert index["spawns"] and "player" in index
    # Textures live once in textures/, referenced by URI; no image data inside the GLBs.
    textures = list((tmp_path / "textures").iterdir())
    assert textures
    for chunk in blocks:
        gltf = _gltf(tmp_path / chunk["file"])
        assert all("uri" in img and "bufferView" not in img for img in gltf.get("images", []))
        assert all((tmp_path / img["uri"]).exists() for img in gltf.get("images", []))


def test_externalized_glb_loads(town, tmp_path):
    import trimesh

    index = json.loads(export_chunks(town, tmp_path).read_text())
    block = next(c for c in index["chunks"] if c["name"] == "block_0_0")
    scene = trimesh.load(tmp_path / block["file"])
    assert len(scene.geometry) > 10
    lod = trimesh.load(tmp_path / block["lod"])
    full_tris = sum(len(g.faces) for g in scene.geometry.values())
    lod_tris = sum(len(g.faces) for g in lod.geometry.values())
    assert 0 < lod_tris < full_tris


def test_disk_cache_round_trip(tmp_path):
    cold = SceneComposer(cache_dir=tmp_path)
    first = cold._load_object({"asset": "shop.yaml"})
    assert list(tmp_path.glob("*.pickle"))
    warm = SceneComposer(cache_dir=tmp_path)
    second = warm._load_object({"asset": "shop.yaml"})
    names = lambda n: [x.name for x in n.iter_nodes()]  # noqa: E731
    assert names(first) == names(second)
    verts = lambda n: sum(len(x.mesh.vertices) for x in n.iter_nodes() if x.mesh is not None)  # noqa: E731
    assert verts(first) == verts(second)
    # Equal materials from the cache are one object.
    materials = {}
    for x in second.iter_nodes():
        if x.mesh is not None and x.mesh.material is not None:
            materials.setdefault(x.mesh.material.name, set()).add(id(x.mesh.material))
    assert all(len(ids) == 1 for ids in materials.values())


def test_disk_cache_key_includes_params(tmp_path):
    composer = SceneComposer(cache_dir=tmp_path)
    composer._load_object({"asset": "house_simple.yaml", "params": {"width": 7}})
    composer._load_object({"asset": "house_simple.yaml", "params": {"width": 9}})
    assert len(list(tmp_path.glob("*.pickle"))) == 2
