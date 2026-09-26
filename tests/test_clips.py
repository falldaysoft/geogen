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
