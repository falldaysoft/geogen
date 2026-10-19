"""Arms that keep clear of the body and its clothes (generators/arms.py, geogen.11.5.5)."""

from pathlib import Path

import numpy as np
import pytest

ASSETS = Path(__file__).parent.parent / "assets"


def _load(preset, outfit):
    from geogen.layout import LayoutLoader

    return LayoutLoader().load(ASSETS / "characters" / "humanoid.yaml", params={"preset": preset, "outfit": outfit})


def test_probe_tells_inside_from_outside_and_how_deep():
    from geogen.generators.arms import _Surface
    from geogen.generators.primitives import CubeGenerator

    cube = CubeGenerator(bevel=0).generate()
    obstacle = _Surface(cube.faces, cube.vertices).at(cube.vertices)
    points = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.7, 0.0, 0.0], [0.6, 0.6, 0.0], [0.45, 0.45, 0.45]])
    depth, normal = obstacle.probe(points)
    assert depth == pytest.approx([0.5, 0.1, -0.2, -np.hypot(0.1, 0.1), 0.05], abs=1e-9)
    assert normal[1] == pytest.approx([1.0, 0.0, 0.0])


def test_tube_probe_measures_off_its_rings():
    """A skirt as rings: a cone 0.2 m long, radius 0.1 at the top and 0.15 at the hem, hanging
    from y = 0 (the hem ring tilted 0.02 m up at +X)."""
    from geogen.generators.arms import _Tube

    angle = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    rings, x = [], []
    for k, y in enumerate(np.linspace(0.0, -0.2, 5)):
        r = 0.1 + 0.25 * -y
        ring = np.column_stack([r * np.cos(angle), np.full(24, y), r * np.sin(angle)])
        if k == 4:
            ring[:, 1] += 0.01 * (1 + np.cos(angle))
        rings.append(np.arange(24 * k, 24 * k + 24))
        x.append(ring)
    tube = _Tube(rings, np.concatenate(x))
    slant = 1 / np.hypot(1, 0.25)
    points = np.array([[0.0, -0.1, 0.0], [0.1, -0.1, 0.0], [0.15, -0.1, 0.0], [0.0, 0.05, 0.0], [0.0, -0.25, 0.0], [0.0, -0.21, 0.0]])
    depth, normal = tube.probe(points)
    assert depth[0] == pytest.approx(0.09, abs=0.001)                      # nearer the hem (at -0.19 on the axis) than the wall
    assert depth[1] == pytest.approx(0.025 * slant, abs=0.001)
    assert depth[2] == pytest.approx(-0.025 * slant, abs=0.001)
    assert normal[2] == pytest.approx([1.0, 0.0, 0.0], abs=0.01)
    assert depth[3:] == pytest.approx([-0.05, -0.06, -0.02], abs=0.001)    # off the caps
    assert normal[3] == pytest.approx([0.0, 1.0, 0.0])


def test_swing_takes_an_arm_out_of_a_block():
    """A hanging arm whose hand is in a block beside it swings outward until it is clear, and
    one that is clear already is left alone."""
    from geogen.generators.arms import CLEARANCE, _rotation, _Surface, _swing
    from geogen.generators.primitives import CubeGenerator

    cube = CubeGenerator(bevel=0).generate()                    # a unit cube about the origin: a hip
    obstacle = _Surface(cube.faces, cube.vertices).at(cube.vertices)
    pivot, along, outward = np.array([0.55, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    hand = np.array([[0.45, 0.1, 0.0], [0.47, 0.0, 0.05], [0.44, -0.1, -0.05]])
    swing = _swing(hand, pivot, along, outward, [obstacle])
    angle = np.linalg.norm(swing)
    assert 0 < np.degrees(angle) < 15 and swing[2] / angle == pytest.approx(1.0)       # about +Z: the hand goes to +X
    moved = pivot + (hand - pivot) @ _rotation(swing / angle, angle).T
    assert obstacle.probe(moved)[0].max() <= -CLEARANCE
    assert np.linalg.norm(_swing(moved, pivot, along, outward, [obstacle])) == 0.0


def test_wide_hips_move_the_hands_out_and_slim_ones_leave_them():
    from geogen.clipping import measure

    wide, slim = _load(["feminine", "hourglass"], "none"), _load(["masculine", "slim"], "none")
    assert wide.meta["humanoid"]["arms"]["pose_lie"] > 1.0
    assert "pose_lie" not in slim.meta["humanoid"].get("arms", {})
    assert "pose_stand" not in slim.meta["humanoid"].get("arms", {})
    assert measure(wide, "lie")["hands"]["count"] == 0


def test_hands_stay_out_of_the_thighs_through_sitting_down():
    """The blend from standing to sitting used to take the hands through the thighs."""
    from geogen.clipping import measure

    character = _load(["masculine"], "jeans_tee")
    report = character.meta["humanoid"]["arms"]
    assert {"sit_down", "stand_up"} <= set(report) and max(report.values()) < 30.0
    for clip in ("sit_down", "stand_up"):
        duration = next(c.duration for c in character.clips if c.name == clip)
        for t in np.linspace(0.0, duration, 9):
            assert measure(character, None, clip, float(t))["hands"]["count"] == 0, (clip, t)


def test_clearing_twice_changes_nothing():
    from geogen.generators.arms import clear_arms

    character = _load(["feminine", "curvy"], "skirt_sweater").instance()
    assert clear_arms(character) == {}


def test_a_transition_ends_as_the_pose_it_goes_to():
    """The swing a clip adds at its ends is the static poses' own, so the hand-over doesn't pop."""
    character = _load(["feminine", "hourglass"], "dress")
    clips = {c.name: c for c in character.clips}
    for clip, key, pose in (("lie_down", -1, "pose_lie"), ("get_up", 0, "pose_lie"), ("sit_down", -1, "pose_sit")):
        for side in ("Left", "Right"):
            a, b = clips[clip].tracks[f"{side}UpperArm"].rotations[key], clips[pose].tracks[f"{side}UpperArm"].rotations[0]
            assert abs(np.dot(a, b)) == pytest.approx(1.0, abs=1e-6), (clip, side)
