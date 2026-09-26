"""Humanoid body asset (geogen-z2b.16.4): presets, shape params, budget, skinning integrity, export."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.export import _split_glb, export_scene
from geogen.layout import LayoutLoader

HUMANOID = "assets/characters/humanoid.yaml"


@pytest.fixture(scope="module")
def bodies():
    loader = LayoutLoader()
    return {preset: loader.load(HUMANOID, params={"preset": preset} if preset else None)
            for preset in (None, "feminine", "masculine")}


def _torso_widths(body):
    """(chest, waist, hips) widths of the rest (T-pose) mesh from horizontal sections, arms excluded."""
    import trimesh

    mesh = body.find("body").mesh
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    k = body.meta["humanoid"]["height"] / 1.7

    def widths(lo, hi):
        out = []
        for y in np.linspace(lo * k, hi * k, 7):
            pts = tm.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0]).vertices
            out.append(np.ptp(pts[np.abs(pts[:, 0]) < 0.2 * k][:, 0]))
        return np.array(out)

    return widths(1.15, 1.26).max(), widths(1.0, 1.12).min(), widths(0.8, 0.92).max()


@pytest.mark.parametrize("preset, height", [(None, 1.72), ("feminine", 1.66), ("masculine", 1.79)])
def test_presets_build_clean_bodies_within_budget(bodies, preset, height):
    body = bodies[preset]
    assert body.meta["humanoid"]["height"] == pytest.approx(height)
    assert body.meta["humanoid"]["pose"] == "stand"
    mesh = body.find("body").mesh
    assert len(mesh.faces) <= 1600
    assert meshops.validate(mesh).ok
    posed = body.find("body").world_mesh()
    assert meshops.validate(posed).watertight
    assert posed.vertices[:, 1].max() == pytest.approx(height, abs=0.01)
    assert posed.vertices[:, 1].min() == pytest.approx(0.0, abs=0.01)     # feet on the floor
    np.testing.assert_allclose(mesh.weights.sum(axis=1), 1.0)
    assert body.size[1] == pytest.approx(height, abs=0.01)


def test_coincident_vertices_move_together(bodies):
    """Union seams split vertices by UV; the copies must share weights or posing tears the skin."""
    body = bodies["feminine"].find("body")
    rest, posed = body.mesh.vertices, body.world_mesh().vertices
    _, group = np.unique(np.round(rest / 1e-6).astype(np.int64), axis=0, return_inverse=True)
    group = group.reshape(-1)
    spread = np.zeros((group.max() + 1, 3))
    lo = np.full((group.max() + 1, 3), np.inf)
    hi = np.full((group.max() + 1, 3), -np.inf)
    np.minimum.at(lo, group, posed)
    np.maximum.at(hi, group, posed)
    assert np.max(hi - lo) < 1e-9


def test_feminine_and_masculine_shapes_differ(bodies):
    fem_chest, fem_waist, fem_hips = _torso_widths(bodies["feminine"])
    masc_chest, masc_waist, masc_hips = _torso_widths(bodies["masculine"])
    assert masc_chest / masc_hips > fem_chest / fem_hips + 0.1     # broad chest vs hips
    assert fem_waist / fem_hips < masc_waist / masc_hips - 0.06    # hourglass
    assert fem_waist / fem_hips < 0.72


def test_explicit_params_override_the_preset():
    body = LayoutLoader().load(HUMANOID, params={"preset": "feminine", "height": 1.55})
    assert body.meta["humanoid"]["height"] == pytest.approx(1.55)
    assert body.meta["preset"] == "feminine"
    with pytest.raises(ValueError, match="unknown preset"):
        LayoutLoader().load(HUMANOID, params={"preset": "giant"})


def test_every_profile_bone_is_a_joint(bodies):
    body = bodies[None]
    skin = body.find("body").skin
    assert len(skin.joints) == 56
    assert body.find("Root").parent is body
    names = [j.name for j in skin.joints]
    for bone in ("Hips", "LeftHand", "RightLittleDistal", "Jaw", "LeftEye", "RightToes"):
        assert bone in names


def test_mitten_follows_the_centre_fingers(bodies):
    body = bodies[None].find("body")
    names = [j.name for j in body.skin.joints]
    used = {names[j] for j in np.unique(body.mesh.joints[body.mesh.weights > 0.05])}
    for bone in ("MiddleProximal", "RingProximal", "MiddleDistal", "RingDistal", "ThumbDistal"):
        assert f"Left{bone}" in used and f"Right{bone}" in used
    # Up to four influences everywhere, none silently dropped: every vertex's weights still sum to 1.
    np.testing.assert_allclose(body.mesh.weights.sum(axis=1), 1.0)


def test_humanoid_exports_as_a_skinned_glb(bodies, tmp_path):
    out = export_scene(bodies["feminine"], tmp_path / "woman.glb")
    gltf, _, _ = _split_glb(out.read_bytes())
    (skin,) = gltf["skins"]
    assert len(skin["joints"]) == 56
    assert gltf["nodes"][skin["skeleton"]]["name"] == "Root"
    assert not any(n["name"].endswith("colonly") for n in gltf["nodes"])
