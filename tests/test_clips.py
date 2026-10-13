"""Procedural humanoid clips (geogen-z2b.16.9): walk without foot sliding, seamless loops, idle."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.core.ik import aim, two_bone
from geogen.layout import LayoutLoader

HUMANOID = "assets/characters/humanoid.yaml"


@pytest.fixture(scope="module", params=["feminine", "masculine"])
def body(request):
    return LayoutLoader().load(HUMANOID, params={"preset": request.param})


def _clip(body, name):
    return next(c for c in body.clips if c.name == name)


def test_two_bone_ik_reaches_and_bends_toward_the_pole():
    root, target, pole = np.zeros(3), np.array([0.0, -0.7, 0.2]), np.array([0.0, 0.0, 1.0])
    middle, end, reached = two_bone(root, 0.4, 0.4, target, pole)
    assert reached
    np.testing.assert_allclose(end, target, atol=1e-9)
    assert np.linalg.norm(middle - root) == pytest.approx(0.4)
    assert np.linalg.norm(end - middle) == pytest.approx(0.4)
    assert middle[2] > target[2] / 2                  # knee out toward +Z
    _, end, reached = two_bone(root, 0.4, 0.4, np.array([0.0, -2.0, 0.0]), pole)
    assert not reached and np.linalg.norm(end) == pytest.approx(0.8, abs=1e-5)


def test_aim_keeps_the_hinge_axis():
    r = aim(np.eye(3), [0, -1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0])
    np.testing.assert_allclose(r @ [0, -1, 0], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(r @ [1, 0, 0], [1, 0, 0], atol=1e-12)


def test_walk_feet_stay_planted(body):
    walk = _clip(body, "walk")
    speed, period = walk.meta["speed"], walk.duration
    assert speed == pytest.approx(1.2)
    assert walk.meta["stride"] == pytest.approx(speed * period, abs=1e-4)
    assert 0.9 < walk.meta["stride"] < 1.3                # a natural stride at 1.2 m/s
    times = np.linspace(0, period, 91)[:-1]
    ankle, toes = [], []
    for t in times:
        walk.apply(body, t)
        ankle.append(body.find("LeftFoot").world_transform()[:3, 3] + [0, 0, speed * t])
        toes.append(body.find("LeftToes").world_transform()[:3, 3] + [0, 0, speed * t])
    ankle, toes, phase = np.array(ankle), np.array(toes), times / period
    # In ground coordinates the stance foot doesn't move: flat foot, then rolling onto the toes.
    assert np.ptp(ankle[phase < 0.36], axis=0).max() < 0.004
    assert np.ptp(toes[(phase > 0.36) & (phase < 0.6)], axis=0).max() < 0.004
    # Mid-swing, the foot is lifted clear of the floor.
    planted = ankle[phase < 0.36, 1].mean()
    assert ankle[(phase > 0.72) & (phase < 0.88), 1].min() > planted + 0.03


def test_walk_loops_seamlessly(body):
    walk = _clip(body, "walk")
    walk.apply(body, 0.0)
    start = {n.name: n.transform.to_matrix() for n in body.find("Root").iter_nodes()}
    walk.apply(body, walk.duration - 1e-9)
    for node in body.find("Root").iter_nodes():
        np.testing.assert_allclose(node.transform.to_matrix(), start[node.name], atol=1e-6, err_msg=node.name)


def test_walk_and_idle_poses_stay_on_the_floor_and_closed(body):
    for name, fractions in (("walk", (0.0, 0.25, 0.5, 0.8)), ("idle", (0.1, 0.4, 0.7))):
        clip = _clip(body, name)
        for f in fractions:
            clip.apply(body, clip.duration * f)
            mesh = body.find("body").world_mesh()
            assert meshops.validate(mesh).watertight
            assert mesh.vertices[:, 1].min() > -0.015, (name, f)


def test_idle_keeps_feet_planted(body):
    idle = _clip(body, "idle")
    positions = []
    for t in np.linspace(0, idle.duration, 25):
        idle.apply(body, t)
        positions.append(body.find("RightFoot").world_transform()[:3, 3])
    assert np.ptp(np.array(positions), axis=0).max() < 0.004


def test_clip_speed_is_exported(tmp_path):
    from geogen.export import _split_glb, export_scene

    body = LayoutLoader().load(HUMANOID)
    gltf, _, _ = _split_glb(export_scene(body, tmp_path / "h.glb").read_bytes())
    extras = next(n for n in gltf["nodes"] if n["name"] == "humanoid")["extras"]["geogen"]
    clips = {c["name"]: c for c in extras["clips"]}
    assert clips["walk"]["speed"] == pytest.approx(1.2) and clips["walk"]["loop"]
    assert {"idle", "pose_sit", "pose_lie", "pose_stand"} <= set(clips)
    assert {a["name"] for a in gltf["animations"]} >= {"humanoid_walk", "humanoid_idle"}


def _placed(body, clip, t):
    """The body at ``t`` of a transition clip, its root following the clip's ``meta.root``."""
    from geogen.core.transform import Transform
    from geogen.generators.clips import root_at

    actor = body.instance()
    _clip(actor, clip.name).apply(actor, t)
    actor.transform = Transform.from_matrix(root_at(clip.meta["root"], t))
    return actor


@pytest.mark.parametrize("name", ["sit_down", "stand_up"])
def test_sitting_down_keeps_the_feet_planted(body, name):
    """The transition clips move the root from the approach to the seat (or back) while leg IK
    holds both feet on the floor: the ankles stay level and only shuffle between the two poses'
    foot positions."""
    clip = _clip(body, name)
    assert (clip.meta["from"], clip.meta["to"]) == (("stand", "sit") if name == "sit_down" else ("sit", "stand"))
    assert not clip.loop
    ankles, hips = [], []
    for t in np.linspace(0.0, clip.duration, 15):
        actor = _placed(body, clip, t)
        ankles.append(actor.find("LeftFoot").world_transform()[:3, 3])
        hips.append(actor.find("Hips").world_transform()[:3, 3])
    ankles, hips = np.array(ankles), np.array(hips)
    assert np.ptp(ankles[:, 1]) < 0.005                        # on the floor throughout
    assert np.ptp(ankles[:, 2]) < 0.2 and np.ptp(ankles[:, 0]) < 0.03
    steps = np.diff(ankles[:, 2])
    assert (steps <= 1e-6).all() or (steps >= -1e-6).all()      # one way, no wobble
    assert np.ptp(hips[:, 1]) > 0.2 and np.ptp(hips[:, 2]) > 0.4   # the hips do the travelling


def test_transitions_start_and_end_in_their_poses(body):
    """A transition's first and last frames are the two poses where the runtime holds them: the
    root path ends at the pose's root offset, and the skeleton matches the pose clip."""
    from geogen.generators.clips import root_at

    sit = np.array(body.meta["poses"]["sit"]["offset"])
    down, up = _clip(body, "sit_down"), _clip(body, "stand_up")
    np.testing.assert_allclose(root_at(down.meta["root"], down.duration)[:3, 3], sit, atol=2e-4)
    np.testing.assert_allclose(root_at(up.meta["root"], up.duration), np.eye(4), atol=2e-4)
    # Sitting down starts where standing up ends, in front of the seat and on the floor below it.
    start = root_at(down.meta["root"], 0.0)[:3, 3]
    np.testing.assert_allclose(root_at(up.meta["root"], 0.0)[:3, 3], sit - start, atol=2e-4)   # approach frame
    assert start[2] == pytest.approx(0.5) and start[1] == pytest.approx(-0.45, abs=0.03)
    for clip, t, pose in ((down, down.duration, "pose_sit"), (up, 0.0, "pose_sit"), (down, 0.0, "pose_stand")):
        moved, held = body.instance(), body.instance()
        _clip(moved, clip.name).apply(moved, t)
        _clip(held, pose).apply(held, 0.0)
        for bone in ("LeftLowerLeg", "LeftFoot", "Head", "RightHand"):
            np.testing.assert_allclose(moved.find(bone).world_transform()[:3, 3],
                                       held.find(bone).world_transform()[:3, 3], atol=0.01, err_msg=f"{clip.name} {bone}")


def test_lying_down_turns_the_body_onto_its_back(body):
    clip = _clip(body, "lie_down")
    lie = body.meta["poses"]["lie"]
    start, end = _placed(body, clip, 0.0), _placed(body, clip, clip.duration)
    np.testing.assert_allclose(end.transform.translation, lie["offset"], atol=2e-4)
    head = end.find("Head").world_transform()[:3, 3] - end.find("Hips").world_transform()[:3, 3]
    assert abs(head[1]) < 0.05 and head[2] < -0.4               # flat, head toward the pillow
    assert start.find("Hips").world_transform()[0, 3] == pytest.approx(0.8)   # beside the bed
    assert _clip(body, "get_up").meta["to"] == "stand"


def test_clips_key_the_drape_correctives():
    """A skirt's pose correctives (generators/drape.py) are morph targets the clips key: none
    standing, the seated drape at the seated end of a transition, and a walk that loops."""
    body = LayoutLoader().load(HUMANOID, params={"preset": "feminine", "outfit": "skirt_sweater"})
    morphs = body.find("clothes").mesh.morphs
    assert 1 <= len(morphs) <= 8
    tracks = {name: _clip(body, name).morphs["clothes"] for name in ("pose_stand", "pose_sit", "sit_down", "stand_up", "walk")}
    for track in tracks.values():
        assert set(track.weights) == set(morphs)
        assert all(np.abs(w).max() <= 1 + 1e-9 for w in track.weights.values())
    at = lambda name, k: np.array([tracks[name].weights[m][k] for m in morphs])
    assert np.abs(at("pose_stand", 0)).max() < 1e-6 and np.abs(at("pose_sit", 0)).max() > 0.5
    np.testing.assert_allclose(at("sit_down", -1), at("pose_sit", 0), atol=1e-6)
    np.testing.assert_allclose(at("stand_up", 0), at("pose_sit", 0), atol=1e-6)
    np.testing.assert_allclose(at("sit_down", 0), 0.0, atol=0.02)      # (its first frame plants the feet)
    np.testing.assert_allclose(at("walk", 0), at("walk", -1), atol=1e-9)
