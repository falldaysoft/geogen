"""primitive: water (generators/water.py): meta for the runtime shader, and the shore barrier."""

import json

import pytest

from geogen.core import meshops
from geogen.export import export_scene
from geogen.layout.loader import LayoutLoader

SEA = """
name: sea_test
size: [1, 1, 1]
bounds: geometry
spawns_note: none
parts:
  island:
    primitive: cylinder
    size: [20, 0.3, 20]
    anchor: bottom_center
    material: dirt
    walkable: true
  sea:
    primitive: water
    shape: { outer: { rect: [80, 80] }, holes: [{ circle: 10, segments: 32 }] }
    size: [80, 0.02, 80]
    anchor: bottom_center
    offset: [0, 0.1, 0]
    barrier: 1.5
    wave_speed: 1.2
"""


@pytest.fixture(scope="module")
def sea(tmp_path_factory):
    path = tmp_path_factory.mktemp("assets") / "sea_test.yaml"
    path.write_text(SEA.replace("spawns_note: none\n", ""))
    return LayoutLoader().load(path)


def test_water_meta_material_and_no_collider(sea):
    water = sea.find("sea")
    assert water.meta["water"] == {"wave_speed": 1.2}
    assert water.meta["collider"] == "none" and water.mesh.material.name == "water"
    assert not meshops.validate(water.mesh).issues


def test_shore_barrier_is_an_invisible_collider(sea):
    wall = sea.find("shore_barrier-colonly")
    assert wall.meta["type"] == "collider"
    assert "shore_barrier-colonly" not in [n.name for n, _ in sea.iter_meshes()]   # renderers skip it
    lo, hi = wall.mesh.vertices.min(axis=0), wall.mesh.vertices.max(axis=0)
    assert hi[1] >= 1.5 and lo[1] < 0 and hi[0] > 39                  # round the island and the sea's edge


def test_player_cant_walk_into_the_sea(run_godot, sea, tmp_path):
    export_scene(sea, tmp_path / "sea_test.glb")
    out = run_godot("--scene=sea_test", f"--generated={tmp_path}", "--spawn=0,0.3,0", "--yaw=0", "--walk=5")
    result = json.loads(next(l for l in out.splitlines() if l.startswith("walk result: ")).removeprefix("walk result: "))
    # Walking toward -Z from the middle of the island (radius 10) stops at the shore wall.
    assert -10.3 < result["z"] < -9.0, result
