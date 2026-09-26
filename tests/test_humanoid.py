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
    assert height <= body.size[1] < height + 0.05          # the hair adds a little


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
    skins = gltf["skins"]
    assert {gltf["nodes"][i]["name"] for i, n in enumerate(gltf["nodes"]) if "skin" in n} == {"body", "hair"}
    for skin in skins:
        assert len(skin["joints"]) == 56
        assert gltf["nodes"][skin["skeleton"]]["name"] == "Root"
    body_mesh = gltf["meshes"][next(n["mesh"] for n in gltf["nodes"] if n["name"] == "body")]
    assert len(body_mesh["primitives"]) == 2                  # skin + face decal
    assert all("COLOR_0" in p["attributes"] and "JOINTS_0" in p["attributes"] for p in body_mesh["primitives"])
    assert not any(n["name"].endswith("colonly") for n in gltf["nodes"])


def _posed(body, pose):
    from geogen.core.skin import pose_clips

    copy = body.instance()
    pose_clips(copy, f"pose_{pose}", 0.0)
    return copy


@pytest.mark.parametrize("preset", ["feminine", "masculine"])
def test_sit_puts_feet_on_the_floor_and_hips_on_the_seat(bodies, preset):
    """The sit pose's IK is solved per body: placed at a 0.45 m seat anchor with the runtime's
    root offset, both ankles sit at ankle height and the hips just above the seat."""
    body = _posed(bodies[preset], "sit")
    offset = np.array(body.meta["poses"]["sit"]["offset"])
    body.transform.translation = np.array([0.0, 0.45, 0.0]) + offset
    height = body.meta["humanoid"]["height"]
    for side in ("Left", "Right"):
        ankle = body.find(f"{side}Foot").world_transform()[:3, 3]
        assert ankle[1] == pytest.approx(0.045 * height, abs=0.01), side
        assert ankle[2] > 0.3                                   # feet out in front of the seat
    hips = body.find("Hips").world_transform()[:3, 3]
    assert 0.45 < hips[1] < 0.6
    lowest = body.find("body").world_mesh().vertices[:, 1].min()
    assert lowest > -0.02


def test_ik_poses_reach_their_targets(bodies):
    body = bodies["feminine"]
    lean = _posed(body, "lean_on_sill")
    for side in ("Left", "Right"):
        hand = lean.find(f"{side}Hand").world_transform()[:3, 3]
        assert hand[1] == pytest.approx(0.97, abs=0.01)
        assert lean.find(f"{side}Foot").world_transform()[1, 3] == pytest.approx(0.045 * 1.66, abs=0.005)
    reach = _posed(body, "reach_low")
    assert reach.find("RightHand").world_transform()[1, 3] < 0.55
    for pose in ("sit_at_table", "look", "reach_low", "lean_on_sill"):
        assert meshops.validate(_posed(body, pose).find("body").world_mesh()).watertight


STYLES = ["buzz", "crop", "bob", "long", "ponytail", "bun"]


@pytest.mark.parametrize("style", STYLES)
def test_hair_styles_build_closed_skinned_shells(style):
    body = LayoutLoader().load(HUMANOID, params={"hair_style": style, "hair_color": "auburn"})
    hair = body.find("hair")
    assert hair is not None and hair.skin is not None and hair.meta["collider"] == "none"
    assert meshops.validate(hair.mesh).watertight
    np.testing.assert_allclose(hair.mesh.weights.sum(axis=1), 1.0)
    names = [j.name for j in hair.skin.joints]
    assert "Head" in {names[j] for j in np.unique(hair.mesh.joints[hair.mesh.weights > 0.5])}
    np.testing.assert_allclose(hair.mesh.colors[0, :3], (0.55, 0.26, 0.14))
    # Hair sits on the head: above the chin, over the crown, clear of the face in front.
    top = body.meta["humanoid"]["height"]
    assert hair.world_mesh().vertices[:, 1].max() > top
    assert body.meta["humanoid"]["triangles"] <= 2500


def test_no_hair_style_means_no_hair_node():
    assert LayoutLoader().load(HUMANOID, params={"hair_style": "none"}).find("hair") is None
    with pytest.raises(Exception, match="must be one of"):
        LayoutLoader().load(HUMANOID, params={"hair_style": "mohawk"})


def test_face_decal_slot_and_skin_tone(bodies):
    mesh = bodies["feminine"].find("body").mesh
    assert mesh.multi_material and [m.name for m in mesh.materials] == ["skin", "face"]
    face = mesh.faces[mesh.face_materials == 1]
    assert 30 < len(face) < 300
    uv = mesh.uvs[np.unique(face)]
    assert uv.min() > -0.2 and uv.max() < 1.2
    # The face region is on the front of the head.
    pts = mesh.vertices[np.unique(face)]
    assert pts[:, 2].min() > 0 and pts[:, 1].min() > 1.35
    dark = LayoutLoader().load(HUMANOID, params={"preset": "feminine", "skin_tone": 0.9}).find("body").mesh
    assert dark.colors[0, 0] < mesh.colors[0, 0] - 0.3
    # Face texture: distinct per face spec, shared otherwise.
    again = LayoutLoader().load(HUMANOID, params={"preset": "feminine", "height": 1.6}).find("body").mesh
    assert again.materials[1] is mesh.materials[1]
    assert bodies["masculine"].find("body").mesh.materials[1] is not mesh.materials[1]


def test_face_texture_paints_features():
    from geogen.textures.character import FaceTextureGenerator

    img = np.asarray(FaceTextureGenerator(width=128, height=128, lashes=1.0).generate()).astype(float)
    eye_row = int(128 * (1 - 0.61))
    left_eye = img[eye_row - 3:eye_row + 3, int(128 * 0.28):int(128 * 0.32)].mean()
    cheek = img[int(128 * 0.55):int(128 * 0.6), int(128 * 0.1):int(128 * 0.15)].mean()
    assert left_eye < cheek - 40
