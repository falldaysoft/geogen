"""LODs: decimation, MSFT_lod export and parametric detail."""

import json
import struct

import pytest

from geogen.core import meshops
from geogen.export import export_scene
from geogen.layout import LayoutLoader


def _gltf(path):
    data = path.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


@pytest.mark.parametrize("ratio", [0.5, 0.25])
def test_decimate_hits_ratio_and_stays_closed(ratio):
    mesh = LayoutLoader().load("assets/table_lamp.yaml").find("base").mesh
    reduced = meshops.decimate(mesh, ratio)
    assert len(reduced.faces) == pytest.approx(len(mesh.faces) * ratio, rel=0.15)
    assert meshops.validate(reduced).watertight and reduced.uvs is not None


def test_decimate_leaves_open_or_tiny_meshes():
    from geogen.generators.primitives import PlaneGenerator
    plane = PlaneGenerator().generate()
    assert meshops.decimate(plane, 0.5) is plane


def test_msft_lod_export(tmp_path):
    gltf = _gltf(export_scene(LayoutLoader().load("assets/table_lamp.yaml"), tmp_path / "l.glb", lods=[0.5, 0.25]))
    assert "MSFT_lod" in gltf["extensionsUsed"]
    nodes = gltf["nodes"]
    base = next(n for n in nodes if n["name"] == "base")
    ids = base["extensions"]["MSFT_lod"]["ids"]
    assert [nodes[i]["name"] for i in ids] == ["base_LOD1", "base_LOD2"]
    assert all(i not in base.get("children", []) for i in ids)          # detached from the hierarchy
    assert len(base["extras"]["MSFT_screencoverage"]) == 3
    tris = [gltf["accessors"][gltf["meshes"][n["mesh"]]["primitives"][0]["indices"]]["count"] // 3
            for n in (base, nodes[ids[0]], nodes[ids[1]])]
    assert tris[0] > tris[1] > tris[2]


def test_parametric_detail():
    coarse = LayoutLoader(detail=0.5).load("assets/table_lamp.yaml").find("base").mesh
    fine = LayoutLoader().load("assets/table_lamp.yaml").find("base").mesh
    assert len(coarse.faces) < len(fine.faces) * 0.6
    assert meshops.validate(coarse).watertight
