"""The geometric drape of loose garments (generators/drape.py) and garment stiffness (ringloft)."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.core.skin import pose_clips
from geogen.generators import drape
from geogen.generators.ringloft import _roll
from geogen.layout import LayoutLoader

HUMANOID = "assets/characters/humanoid.yaml"


@pytest.fixture(scope="module")
def skirted():
    return LayoutLoader().load(HUMANOID, params={"preset": "feminine", "outfit": "casual_f"})


def _unbaked(character, clip: str, seat: bool = False):
    """(skinned, draped) clothes vertices of the character in ``clip``, before the morphs."""
    actor = character.instance()
    assert pose_clips(actor, clip, 0.0)
    for node in actor.iter_nodes():
        node.morph_weights = {}
    draper = drape.Draper(character, character.meta["humanoid"]["shells"])
    skinned, draped, _ = draper.positions(actor, seat)
    return actor, skinned, draped


def test_a_ball_spans_a_notch_and_follows_a_wide_hollow():
    s = np.linspace(0.0, 1.0, 201)
    notch = np.where(np.abs(s - 0.5) < 0.03, 0.0, 0.1)            # 6 cm wide, 10 cm deep
    np.testing.assert_allclose(_roll(notch, s, 0.2), 0.1, atol=0.003)
    hollow = 0.1 - 0.08 * np.exp(-((s - 0.5) / 0.25) ** 2)        # half a metre wide
    rolled = _roll(hollow, s, 0.2)
    assert np.all(rolled >= hollow - 1e-12) and np.abs(rolled - hollow).max() < 0.01
    bump = 0.1 - 0.4 * (s - 0.5) ** 2                             # nothing to span on a rise
    np.testing.assert_allclose(_roll(bump, s, 0.2), bump, atol=1e-6)


def test_hull_reach_from_inside_and_outside():
    square = np.array([[-1.0, -1], [1, -1], [1, 1], [-1, 1], [0, 0]])
    rays = np.array([[1.0, 0], [0, 1], [np.sqrt(0.5), np.sqrt(0.5)]])
    np.testing.assert_allclose(drape._hull_reach(square, rays), [1, 1, np.sqrt(2)], atol=1e-9)
    away = square + [3.0, 0]                                       # the origin is outside it
    np.testing.assert_allclose(drape._hull_reach(away, np.array([[1.0, 0], [-1, 0], [0, 1]])), [4, 0, 0], atol=1e-9)
    assert drape._hull_reach(square[:2], rays) is None


def test_standing_needs_no_drape(skirted):
    _, skinned, draped = _unbaked(skirted, "pose_stand")
    np.testing.assert_allclose(draped, skinned, atol=1e-9)
    track = next(c for c in skirted.clips if c.name == "pose_stand").morphs["clothes"]
    assert max(np.abs(w).max() for w in track.weights.values()) < 1e-9


def test_a_seated_skirt_lies_on_the_lap_and_on_the_seat(skirted):
    actor, skinned, draped = _unbaked(skirted, "pose_sit", seat=True)
    from geogen.generators.humanoid import dominant_bones

    node = actor.find("body")
    body = node.world_mesh().vertices[np.isin(dominant_bones(node.mesh, [j.name for j in node.skin.joints]),
                                              ["LeftUpperLeg", "RightUpperLeg"])]
    hip = actor.find("LeftUpperLeg").world_transform()[:3, 3]
    knee = actor.find("LeftLowerLeg").world_transform()[:3, 3]
    middle = (hip[2] + knee[2]) / 2 + 0.03                         # a band across the lap, mid-thigh
    lap = lambda v: v[(np.abs(v[:, 2] - middle) < 0.05) & (np.abs(v[:, 0]) < 0.2)]
    thighs = lap(body)[:, 1].max()                                 # the thighs' tops
    assert lap(skinned)[:, 1].max() > thighs + 0.05                # skinned alone it stands off them as a box
    assert thighs < lap(draped)[:, 1].max() < thighs + 0.03        # draped it lies on them (the band spans 10 cm of a sloping thigh)
    seat = skirted.meta["humanoid"]["seats"]["sit"]
    behind = lambda v: v[(v[:, 2] < knee[2] - 0.03) & (v[:, 1] < hip[1])]
    assert behind(skinned)[:, 1].min() < seat - 0.03               # ... and hangs through the seat
    assert behind(draped)[:, 1].min() > seat - 0.003
    # The baked morphs give the same shape (a static pose has a target of its own).
    baked = skirted.instance()
    pose_clips(baked, "pose_sit", 0.0)
    np.testing.assert_allclose(baked.find("clothes").world_mesh().vertices, draped, atol=0.001)


def test_drape_is_baked_to_a_few_morph_targets(skirted):
    report = skirted.meta["humanoid"]["drape"]
    morphs = skirted.find("clothes").mesh.morphs
    assert report["morphs"] == len(morphs) <= drape.MAX_MORPHS
    assert all(name.startswith("drape_") for name in morphs)
    assert report["clips"]["pose_sit"][1] < 0.5 and report["clips"]["walk"][1] < 8.0      # mm the targets leave
    plain = LayoutLoader().load(HUMANOID, params={"preset": "feminine", "outfit": "jeans_tee"})
    assert plain.find("clothes") is None and "drape" not in plain.meta["humanoid"]


def test_exported_morph_targets_turn_the_normals(skirted, tmp_path):
    """A morph laying the skirt across the lap must turn its normals too (glTF target NORMAL), or a
    renderer lights the lap as though it still hung down."""
    import json, struct
    from geogen.export import export_scene
    data = export_scene(skirted, tmp_path / "skirted.glb").read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + length])
    targets = [t for m in gltf["meshes"] for p in m["primitives"] for t in p.get("targets", [])]
    assert targets and all({"POSITION", "NORMAL"} <= set(t) for t in targets)

    clothes = skirted.find("clothes")
    actor = skirted.instance()
    pose_clips(actor, "pose_sit", 0.0)
    worn = actor.find("clothes")
    mesh = clothes.mesh
    delta = sum(w * mesh.morphs[k] for k, w in worn.morph_weights.items())
    turned = sum(w * meshops.morph_normal_deltas(mesh, mesh.morphs[k]) for k, w in worn.morph_weights.items())
    moved = np.linalg.norm(delta, axis=1) > 0.03                  # the cloth the drape moves a lot
    assert moved.sum() > 20 and np.linalg.norm(turned[moved], axis=1).mean() > 0.1


def test_faces_the_skirt_always_hides_are_left_out(skirted):
    """A character in a skirt doesn't carry the pelvis and thighs under it: faces whose vertices
    are well inside the draped cloth in every pose and sampled frame are left out of the drawn
    body (geogen.11.5.24). The closed body stays beside it, hidden and build-only, for seat
    fitting, draping and QA, and never reaches an export."""
    from geogen.export import to_trimesh_scene
    from geogen.generators.humanoid import closed_body, dominant_bones

    body, closed = skirted.find("body"), skirted.find("body_closed")
    assert closed is not None and closed.hidden and closed.meta["build_only"]
    assert closed_body(skirted) is closed and closed.skin is not None
    assert meshops.validate(closed.mesh).watertight and not meshops.validate(body.mesh).watertight
    dropped = skirted.meta["humanoid"]["hidden_faces"]
    assert 200 < dropped == len(closed.mesh.faces) - len(body.mesh.faces)
    assert np.array_equal(body.mesh.vertices, closed.mesh.vertices)         # indices still hold
    drawn = {tuple(f) for f in body.mesh.faces.tolist()}
    gone = np.array([f for f in closed.mesh.faces.tolist() if tuple(f) not in drawn])
    names = [j.name for j in closed.skin.joints]
    assert set(dominant_bones(closed.mesh, names)[np.unique(gone)]) <= {"Hips", "Spine", "LeftUpperLeg", "RightUpperLeg"}
    # What went is inside the cloth, seated and mid-walk alike, by more than the drape's margin.
    from geogen.clipping import _meshes, _solid, distance, posed, winding

    for pose, clip, t in (("sit", None, 0.0), (None, "walk", 0.4), (None, "sit_down", 0.5)):
        now = _meshes(posed(skirted, pose, clip, t))
        pts = now["body"].vertices[np.unique(gone)]
        solid, fans = _solid(now["clothes"], caps=True)
        assert np.all(winding(pts, solid) > 0.5)
        # ... and well off the cloth itself: the outer wall (the lining is 6 mm inside it; the
        # fans QA caps the hem's opening with cut through a lap).
        assert distance(pts, solid[:len(solid) - len(fans)]).min() > 0.004
    assert "body_closed" not in to_trimesh_scene(skirted).graph.nodes
    plain = LayoutLoader().load(HUMANOID, params={"preset": "feminine", "outfit": "jeans_tee"})
    assert plain.find("body_closed") is None and "hidden_faces" not in plain.meta["humanoid"]
