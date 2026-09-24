"""Transparent and emissive materials through loader, exporter and renderer."""

import json
import struct

import numpy as np
import pytest

from geogen.core.node import SceneNode
from geogen.export import export_scene
from geogen.generators.primitives import CubeGenerator
from geogen.materials.loader import MaterialLoader


def _gltf(path):
    data = path.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


def test_loader_reads_opacity_and_emission():
    glass = MaterialLoader().load("glass")
    assert glass.transparent and glass.opacity == pytest.approx(0.3)
    shade = MaterialLoader().load("lamp_shade")
    assert not shade.transparent and any(shade.emissive)
    assert shade.emissive_factor == pytest.approx((0.6, 0.492, 0.33))
    assert not MaterialLoader().load("brick").transparent


def test_gltf_export_blends_glass_and_emits_light(tmp_path):
    root = SceneNode("r")
    for name in ("glass", "lamp_shade", "brick"):
        mesh = CubeGenerator(bevel=0).generate()
        mesh.material = MaterialLoader().load(name)
        root.add_child(SceneNode(name, mesh=mesh))
    materials = {m["name"]: m for m in _gltf(export_scene(root, tmp_path / "m.glb"))["materials"]}
    assert materials["glass"]["alphaMode"] == "BLEND" and materials["glass"]["doubleSided"]
    assert materials["glass"]["pbrMetallicRoughness"]["baseColorFactor"][3] == pytest.approx(0.3, abs=0.01)
    assert materials["lamp_shade"]["emissiveFactor"] == pytest.approx([0.6, 0.492, 0.33], abs=0.01)
    assert "alphaMode" not in materials["brick"] and "emissiveFactor" not in materials["brick"]


def test_render_sees_through_glass():
    from geogen.render import RenderOptions, render_scene
    loader = MaterialLoader()

    def pane(material):
        root = SceneNode("r")
        wall = CubeGenerator(size_x=2, size_y=2, size_z=0.1, bevel=0).generate()
        wall.material = loader.load("brick")
        glass = CubeGenerator(size_x=2, size_y=2, size_z=0.02, bevel=0).generate()
        glass.material = loader.load(material)
        root.add_child(SceneNode("wall", mesh=wall))
        front = SceneNode("glass", mesh=glass)
        front.transform.translation = np.array([0, 0, 0.5])
        root.add_child(front)
        img = render_scene(root, RenderOptions(width=160, height=120, camera=np.array([0, 0, 4.0]),
                                               target=np.zeros(3), ground=False, shadows=False))
        return np.asarray(img, dtype=float)[40:80, 60:100].mean(axis=(0, 1))

    through_glass, through_opaque = pane("glass"), pane("chrome")
    # Brick (red-dominant) shows through the transparent pane, not the opaque one.
    assert through_glass[0] - through_glass[2] > through_opaque[0] - through_opaque[2] + 10
