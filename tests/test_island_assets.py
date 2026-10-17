"""Island assets (palm tree, jetty, rowing boat, lighthouse) and the features they brought:
part arrays, round/floor/ceil, circle paths, closed lathe profiles, spiral stair ramps and links."""

import json

import numpy as np
import pytest

from geogen.core import meshops
from geogen.export import export_scene
from geogen.generators.paths import path_points
from geogen.generators.profiles import LatheGenerator
from geogen.generators.stairs import StairsGenerator
from geogen.layout.expressions import evaluate
from geogen.layout.loader import LayoutLoader
from geogen.runtime_client import step


def _load(name: str, **params):
    return LayoutLoader().load(f"assets/{name}.yaml", params=params or None)


def test_array_parts_repeat_and_subtract_all_copies():
    jetty = _load("jetty", length=12)
    piles = sorted(n.name for n in jetty.children if n.name.startswith("pile_left_"))
    assert piles == [f"pile_left_{i}" for i in range(5)]          # round(12 / 3) + 1
    z = [next(n for n in jetty.children if n.name == p).transform.translation[2] for p in piles]
    assert np.allclose(np.diff(z), (12 - 0.4) / 4)
    # The lighthouse's tower subtracts `window`, an arrayed cutter: both slits are cut.
    tower = next(n for n in _load("lighthouse").children if n.name == "tower")
    v = tower.world_mesh().vertices
    back = v[(v[:, 2] < -1.5) & (np.abs(v[:, 0]) < 0.2)]
    assert np.sum(np.abs(back[:, 1] - 3.0) < 0.01) and np.sum(np.abs(back[:, 1] - 6.2) < 0.01)


def test_round_floor_ceil_and_circle_paths():
    assert evaluate("round(2.5)", {}) == 3 and evaluate("floor(2.7)", {}) == 2 and evaluate("ceil(2.1)", {}) == 3
    pts = path_points({"circle": 2.0, "segments": 12})
    assert len(pts) == 12 and np.allclose(np.hypot(*pts.T), 2.0)


def test_closed_lathe_profile_is_a_ring_without_caps():
    ring = LatheGenerator(profile=np.array([[1.0, 0], [1.2, 0], [1.2, 1], [1.0, 1], [1.0, 0]]), segments=24).generate()
    assert np.hypot(ring.vertices[:, 0], ring.vertices[:, 2]).min() > 0.99
    assert meshops.validate(ring).issues == []
    capped = LatheGenerator(profile=np.array([[1.0, 0], [1.2, 0], [1.2, 1], [1.0, 1]]), segments=24).generate()
    assert np.hypot(capped.vertices[:, 0], capped.vertices[:, 2]).min() < 1e-9   # open profiles still cap


def test_spiral_stairs_have_a_ramp_and_linked_ends():
    gen = StairsGenerator(style="spiral", rise=10, width=1.1, railing="none")
    node = gen.to_node("stair")
    ramp = next(c for c in node.children if c.name.endswith("_ramp-colonly"))
    assert ramp.meta["type"] == "collider" and meshops.validate(ramp.mesh).issues == []
    lo, hi = node.meta["stairs"]["ends"]
    assert hi[1] - lo[1] == pytest.approx(10.0)
    # The way round between them, for a body that walks: on the ramp, in steps a walker can't cut across.
    walk = np.array(node.meta["stairs"]["path"])
    assert np.allclose(walk[0], lo, atol=1e-3) and np.allclose(walk[-1], hi, atol=1e-3)
    assert np.linalg.norm(np.diff(walk, axis=0), axis=1).max() < 0.35 and (np.diff(walk[:, 1]) >= 0).all()
    radius = np.hypot(*(walk[:, [0, 2]] - node.mesh.vertices[:, [0, 2]].mean(axis=0)).T)
    assert radius.min() > 0.6 and radius.max() < 0.9
    # The ramp is a helix, not a sawtooth: no facet a body on the walking line touches is steeper than it can climb.
    v, f = ramp.mesh.vertices, ramp.mesh.faces
    normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    middle = v[f].mean(axis=1)
    walked = (normals[:, 1] > 0.2) & (np.hypot(middle[:, 0], middle[:, 2]) > 0.6)
    assert walked.sum() > 500 and np.degrees(np.arccos(normals[walked, 1])).max() < 38
    # The pole is on the origin in plan; the ramp ends level with the top tread.
    assert np.allclose(node.mesh.vertices[:, [0, 2]].mean(axis=0), 0, atol=0.2)
    assert ramp.mesh.vertices[:, 1].max() == pytest.approx(hi[1], abs=0.01)     # flat across the top tread


@pytest.mark.parametrize("name", ["palm_tree", "jetty", "rowing_boat", "lighthouse"])
def test_island_assets_are_clean(name):
    root = _load(name)
    for node in root.iter_nodes():
        if node.mesh is not None:
            assert meshops.validate(node.mesh).issues == [], (name, node.name)


def test_palm_has_curved_trunk_and_fronds():
    palm = _load("palm_tree", seed=4)
    tree = next(n for n in palm.iter_nodes() if n.meta.get("tree"))
    assert tree.meta["tree"]["style"] == "palm" and tree.meta["tree"]["fronds"] == 11
    foliage = next(n for n in tree.children if n.name == "foliage")
    assert [m.name for m in foliage.mesh.materials] == ["foliage_palm", "bark"]
    xz = tree.mesh.vertices[:, [0, 2]]
    assert np.ptp(xz, axis=0).max() > 0.6          # it leans


@pytest.fixture(scope="module")
def lighthouse_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("generated")
    export_scene(_load("lighthouse"), out / "lighthouse.glb")
    return out


def test_lighthouse_gallery_is_reachable_on_foot(godot_script, lighthouse_dir):
    run = godot_script([step("wait_ready", timeout=120), step("world.nav_path", **{"from": [0, 0, 4]}, to=[0, 10, -2.8])],
                       "--scene=lighthouse", f"--generated={lighthouse_dir}")
    path = run["world.nav_path"]
    assert path["reached"] and path["end_gap"] < 0.3 and path["points"][-1][1] > 9.9, path


def test_lighthouse_stair_is_climbed_by_the_player(godot_runtime, lighthouse_dir):
    # With the real body, up and back down: the navmesh path has only the stair link's two ends,
    # and the walker goes round by the stair's own walking line.
    rt = godot_runtime("lighthouse", lighthouse_dir)
    rt.call("wait_ready")
    rt.call("player.teleport", pos=[0, 0.1, 4], yaw=0)
    rt.call("time.scale", scale=4)
    for target, height in (([0, 10, -2.8], 10.0), ([0, 0.1, 4], 0.0)):
        walked = rt.send("player.walk_to", pos=target, timeout=60).get(timeout=200)["result"]
        assert walked["reached"] and walked["distance"] < 0.4, (target, walked["status"], walked["blocked_by"])
        assert walked["position"][1] == pytest.approx(height, abs=0.1)
        assert max(p[1] for p in walked["trace"]) <= 10.2                # it went round, never over
    rt.call("time.scale", scale=1)


def test_lighthouse_lamp_comes_on_at_night(godot_script, lighthouse_dir):
    for time, on in (("23:00", True), ("12:00", False)):
        run = godot_script([step("wait", frames=5), step("world.lights")], "--scene=lighthouse",
                           f"--generated={lighthouse_dir}", start={"time": time})
        lights = run["world.lights"]
        assert lights["fixtures"]["lighthouse"]["visible"] is on, (time, lights)
