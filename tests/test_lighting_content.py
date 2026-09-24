"""Light fixtures: asset light blocks, switches, KHR_lights_punctual export, viewer fixture lights."""

import json
import struct

import numpy as np
import pytest

from geogen.export import CANDELA_PER_ENERGY, export_scene
from geogen.layout import LayoutLoader
from geogen.layout.interactions import apply_state


def _gltf(path):
    data = path.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


def test_lamp_assets_declare_lights():
    lamp = LayoutLoader().load("assets/table_lamp.yaml")
    assert lamp.meta["light"]["offset"] == pytest.approx([0, 0.43, 0])
    with pytest.raises(ValueError, match="unknown keys"):
        LayoutLoader().load_string("name: x\nsize: [1, 1, 1]\nlight: { watts: 60 }\n")


def test_switches_toggle_their_room_light():
    suite = LayoutLoader().load("assets/hotel_suite.yaml")
    switch = suite.find("bedroom_switch_1")
    (it,) = switch.interactions
    assert switch.meta["switch"] == {"light": "bedroom_light"}
    assert it.initial == "on" and it.states["on"].next == "off"
    rocker = switch.find("rocker")
    before = rocker.world_transform().copy()
    apply_state(switch, it, "off")
    assert not np.allclose(rocker.world_transform(), before)


def test_khr_lights_punctual_export(tmp_path):
    gltf = _gltf(export_scene(LayoutLoader().load("assets/hotel_suite.yaml"), tmp_path / "s.glb"))
    assert "KHR_lights_punctual" in gltf["extensionsUsed"]
    lights = gltf["extensions"]["KHR_lights_punctual"]["lights"]
    assert len(lights) == 3 and all(l["type"] == "point" for l in lights)
    assert lights[0]["intensity"] == pytest.approx(1.2 * CANDELA_PER_ENERGY)
    nodes = gltf["nodes"]
    holder = next(n for n in nodes if n["name"] == "bedroom_light_light")
    assert holder["translation"] == pytest.approx([0, -0.47, 0])
    parent = next(n for n in nodes if holder is not n and nodes.index(holder) in n.get("children", []))
    assert parent["name"] == "bedroom_light"
    import trimesh
    assert len(trimesh.load(tmp_path / "s.glb").geometry) > 0   # still a valid GLB


def test_viewer_collects_fixture_lights():
    pytest.importorskip("PyQt6")
    from geogen.layout import SceneComposer
    from geogen.viewer.gl_view import scene_fixture_lights
    scene = SceneComposer().compose("assets/scenes/hotel_room_auto.yaml")
    fixtures = scene_fixture_lights(scene)
    assert len(fixtures) == 6   # 3 pendants, 2 table lamps, 1 floor lamp
    heights = sorted(round(float(p[1]), 2) for p, *_ in fixtures)
    assert heights[-1] == pytest.approx(2.5 - 0.47, abs=0.01)
