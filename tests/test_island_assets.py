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


def _nav(out: str) -> dict:
    return json.loads(next(l for l in out.splitlines() if l.startswith("nav path: ")).removeprefix("nav path: "))


def test_lighthouse_gallery_is_reachable_on_foot(run_godot, lighthouse_dir):
    out = run_godot("--scene=lighthouse", f"--generated={lighthouse_dir}", "--nav=0,0,4:0,10,-2.8")
    path = _nav(out)
    assert path["reached"] and path["end_gap"] < 0.3 and path["points"][-1][1] > 9.9, path


def test_lighthouse_lamp_comes_on_at_night(run_godot, lighthouse_dir):
    for time, on in (("23:00", True), ("12:00", False)):
        out = run_godot("--scene=lighthouse", f"--generated={lighthouse_dir}", f"--time={time}", "--lights",
                        "--quit-after=5")
        lights = json.loads(next(l for l in out.splitlines() if l.startswith("lights: ")).removeprefix("lights: "))
        assert lights["fixtures"]["lighthouse"]["visible"] is on, (time, lights)
