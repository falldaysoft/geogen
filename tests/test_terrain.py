"""Terrain heightfields (generators/terrain.py) and scatter on terrain (scenes/terrain_test.yaml)."""

import json

import numpy as np
import pytest

from geogen.core import meshops
from geogen.export import export_scene
from geogen.generators.terrain import GRASS, ROCK, SAND, TerrainGenerator, terrain_from_meta

ISLAND = dict(size=(120, 120), height=14, sea_level=2.0, depth=1.9, seed=5, scale=38,
              falloff={"radius": 0.36, "edge": 0.14, "power": 1.4})


def test_terrain_is_a_closed_solid_with_material_groups():
    mesh = TerrainGenerator(**ISLAND).generate()
    report = meshops.validate(mesh)
    assert report.issues == [], report
    assert mesh.to_trimesh().volume > 0                       # outward winding
    assert {SAND, GRASS} <= set(np.unique(mesh.face_materials))
    steep = TerrainGenerator(**{**ISLAND, "height": 40, "scale": 14}).generate()
    assert ROCK in set(np.unique(steep.face_materials))           # steep ground turns to rock
    assert mesh.uvs is not None


def test_island_rises_from_the_sea_bed_and_stays_climbable():
    t = TerrainGenerator(**ISLAND)
    assert t.height_at(0, 0)[0] > t.sea_level + 8                       # a hill in the middle
    assert t.height_at(59, 59)[0] == pytest.approx(t.sea_level - t.depth)   # sea bed at the corners
    xs, zs = np.meshgrid(np.linspace(-40, 40, 81), np.linspace(-40, 40, 81))
    assert t.slope_at(xs.ravel(), zs.ravel()).max() < 40.0             # the player's max slope


def test_pads_level_the_ground_and_coastline_follows_the_water():
    t = TerrainGenerator(**ISLAND, pads=[{"center": [10, -15], "radius": 5, "blend": 4}])
    ring = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    h = t.height_at(10 + 4.5 * np.cos(ring), -15 + 4.5 * np.sin(ring))
    assert np.ptp(h) < 1e-6
    level = t.sea_level - 0.5
    coast = t.coastline(level)
    assert np.all(t.height_at(coast[:, 0], coast[:, 1]) <= level)       # just past the waterline
    assert np.all(t.height_at(coast[:, 0] * 0.97, coast[:, 1] * 0.97) > level - 0.3)
    assert terrain_from_meta(t.params()).height_at(3, 4)[0] == pytest.approx(t.height_at(3, 4)[0])


@pytest.fixture(scope="module")
def terrain_scene(built_scene):
    return built_scene("terrain_test")


def test_island_asset_has_terrain_meta_shore_and_materials(terrain_scene):
    ground = next(n for n in terrain_scene.iter_nodes() if "terrain" in n.meta)
    assert ground.meta.get("walkable")
    assert [m.name for m in ground.mesh.materials] == ["sand", "grass", "rock", "gravel"]
    assert any(c.name.startswith("shore_barrier") for c in ground.children)


def test_scatter_on_terrain_respects_height_and_slope(terrain_scene):
    ground = next(n for n in terrain_scene.iter_nodes() if "terrain" in n.meta)
    t = terrain_from_meta(ground.meta["terrain"])
    trees = [n for n in terrain_scene.children if n.meta.get("scatter", {}).get("group") == "trees"]
    rocks = [n for n in terrain_scene.children if n.meta.get("scatter", {}).get("group") == "rocks"]
    assert len(trees) == 40 and len(rocks) > 5
    for node, (h_lo, h_hi), (s_lo, s_hi) in [(n, (2.0, 9), (0, 18)) for n in trees] + \
            [(n, (0.2, 20), (22, 90)) for n in rocks]:
        x, y, z = node.transform.translation
        assert y == pytest.approx(t.height_at(x, z)[0], abs=1e-6)
        assert h_lo <= y - t.sea_level <= h_hi
        assert s_lo <= t.slope_at(x, z)[0] <= s_hi
    xs = np.array([n.transform.translation[0] for n in trees])
    assert xs.min() < -15 and xs.max() > 15                 # spread over the island, not one clump


@pytest.fixture(scope="module")
def terrain_dir(tmp_path_factory, terrain_scene):
    out = tmp_path_factory.mktemp("generated")
    export_scene(terrain_scene, out / "terrain_test.glb")
    return out


def test_navmesh_covers_the_island(run_godot, terrain_dir):
    t = TerrainGenerator(**ISLAND)
    for x, z in [(0, -38), (-35, 0), (30, 10)]:
        y = float(t.height_at(x, z)[0])
        out = run_godot("--scene=terrain_test", f"--generated={terrain_dir}", f"--nav={x},{y},{z}:0,17,0")
        path = json.loads(next(l for l in out.splitlines() if l.startswith("nav path: ")).removeprefix("nav path: "))
        assert path["reached"] and path["end_gap"] < 0.5, path
        assert abs(path["points"][0][1] - y) < 0.5, path   # started on the island, not the sea bed


def test_island_grounds_paths_and_jetty_gap(built_scene):
    grounds = next(n for n in built_scene("island").iter_nodes() if "terrain" in n.meta)
    paths = next(c for c in grounds.children if c.name == "paths")
    assert paths.mesh.material.name == "gravel" and meshops.validate(paths.mesh).issues == []
    t = terrain_from_meta(grounds.meta["terrain"])
    v = paths.world_mesh().vertices
    top = v[v[:, 1] > t.height_at(v[:, 0], v[:, 2]) + 0.02]
    assert len(top) > len(v) / 3                          # draped just above the ground
    wall = next(c for c in grounds.children if c.name.startswith("shore_barrier"))
    w = wall.world_mesh().vertices
    assert not np.any(np.hypot(w[:, 0], w[:, 2] + 55) < 0.9)     # the jetty crosses the shore here
    trees = [n for n in built_scene("island").children if n.meta.get("scatter", {}).get("group") in ("palms", "pines")]
    xz = np.array([n.transform.translation[[0, 2]] for n in trees])
    assert not t.on_path(xz).any()                        # scatter keeps off the paths
