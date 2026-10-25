"""Seeded scatter placement."""

import numpy as np
import pytest

from geogen.layout import SceneComposer
from geogen.layout.scatter import poisson_disk, resolve_random_params

BASE = """
name: s
place:
  house: { asset: house_peaked.yaml }
  trees:
    asset: pine_tree.yaml
    params: { scale: { random: [0.8, 1.4] } }
    scatter: { seed: 3, rect: [-15, -15, 15, 15], spacing: 4, count: 20, avoid: [house], margin: 1 }
  rocks:
    asset: rock_small.yaml
    scatter: { seed: 4, rect: [-15, -15, 15, 15], spacing: 2, count: 15, radius: 0.5 }
"""


def _xz(scene, group):
    return np.array([c.transform.translation[[0, 2]] for c in scene.children
                     if c.meta.get("scatter", {}).get("group") == group])


def test_poisson_disk_respects_spacing_and_bounds():
    pts = np.array(poisson_disk(np.random.default_rng(1), np.array([0.0, 0.0]), np.array([10.0, 10.0]), 1.5))
    assert len(pts) > 20
    assert pts.min() >= 0 and pts.max() <= 10
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(len(pts)) * 99
    assert d.min() >= 1.5 - 1e-9


def test_scatter_is_deterministic_and_avoids_things():
    a, b = SceneComposer().compose_string(BASE), SceneComposer().compose_string(BASE)
    trees = _xz(a, "trees")
    assert len(trees) == 20 and np.allclose(trees, _xz(b, "trees"))
    # Off the house footprint (8 x 6 m + 1 m margin) ...
    assert not np.any((np.abs(trees[:, 0]) < 5) & (np.abs(trees[:, 1]) < 4))
    # ... and rocks keep their radius away from the trees (2 m + 0.5 m).
    rocks = _xz(a, "rocks")
    assert len(rocks) > 0
    assert np.linalg.norm(rocks[:, None] - trees[None], axis=2).min() >= 2.5 - 1e-9
    # Per-copy random params: tree sizes differ.
    heights = {round(float(c.size[1]), 3) for c in a.children if c.meta.get("scatter", {}).get("group") == "trees"}
    assert len(heights) > 5


def test_different_seed_different_layout():
    other = SceneComposer().compose_string(BASE.replace("seed: 3", "seed: 30"))
    assert not np.allclose(_xz(other, "trees")[:5], _xz(SceneComposer().compose_string(BASE), "trees")[:5])


def test_path_scatter_spacing_and_offset():
    scene = SceneComposer().compose_string("""
name: s
place:
  lamps:
    asset: street_lamp.yaml
    scatter: { seed: 1, path: [[0, 0], [20, 0]], spacing: 5, offset: 2 }
""")
    lamps = _xz(scene, "lamps")
    assert lamps[:, 0] == pytest.approx([0, 5, 10, 15, 20])
    assert np.allclose(lamps[:, 1], 2.0)


FACING = """
name: s
place:
  stalls:
    asset: bench.yaml
    scatter: { seed: 7, path: [[0, -10], [0, 10]], spacing: 5, count: 4, facing: right }
  ring:
    asset: bench.yaml
    scatter: { seed: 8, rect: [-20, -20, 20, 20], spacing: 6, count: 6, facing: center }
  posts:
    asset: bench.yaml
    scatter: { seed: 9, rect: [-20, -20, 20, 20], spacing: 6, count: 3, facing: 45, yaw: [-5, 5] }
"""


def test_scatter_facing_turns_copies_toward_a_path_or_a_centre():
    """`facing` sets each copy's yaw (degrees about +Y, 0 = +Z): `right` of a path heading
    north is +X, `center` looks at the rect's middle, a number is fixed (plus `yaw` jitter)."""
    scene = SceneComposer().compose_string(FACING)
    by_group = {}
    for c in scene.children:
        group = c.meta.get("scatter", {}).get("group")
        if group:
            by_group.setdefault(group, []).append(c)
    assert all(np.degrees(c.transform.rotation[1]) == pytest.approx(90.0) for c in by_group["stalls"])
    for c in by_group["ring"]:
        x, _, z = c.transform.translation
        expected = np.degrees(np.arctan2(-x, -z))
        assert ((np.degrees(c.transform.rotation[1]) - expected + 180) % 360 - 180) == pytest.approx(0, abs=1e-6)
    assert all(40 <= np.degrees(c.transform.rotation[1]) <= 50 for c in by_group["posts"])
    with pytest.raises(ValueError, match="needs a path"):
        SceneComposer().compose_string(FACING.replace("facing: center", "facing: along"))
    with pytest.raises(ValueError, match="facing is"):
        SceneComposer().compose_string(FACING.replace("facing: center", "facing: sideways"))


def test_random_param_values():
    rng = np.random.default_rng(0)
    params = resolve_random_params({"a": {"random": [1, 2]}, "b": {"choice": ["x", "y"]}, "c": 5}, rng)
    assert 1 <= params["a"] <= 2 and params["b"] in ("x", "y") and params["c"] == 5


def test_bad_scatter_spec():
    with pytest.raises(ValueError, match="rect:, path:, on:, on_tag: or on_terrain:"):
        SceneComposer().compose_string("name: s\nplace:\n  t: { asset: rock_small.yaml, scatter: { count: 3 } }\n")
    with pytest.raises(ValueError, match="unknown keys"):
        SceneComposer().compose_string(
            "name: s\nplace:\n  t: { asset: rock_small.yaml, scatter: { rect: [0,0,1,1], denisty: 3 } }\n")
