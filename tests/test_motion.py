"""Motions (geogen.103): body channels, the motion format, the reference rig and the review scenarios.

The scenarios run headless here (no rendering); their numbers are what the review page shows.
"""

import pytest

from geogen import roots
from geogen.core.skeleton import load_skeleton
from geogen.motion import MotionError, load_motion, validate_motion
from geogen.motion_review import load_scenario, run_scenario, scenario_names

MOTIONS = sorted(p.stem for p in roots.glob("motions/*.yaml", roots.CORE))


def test_skeleton_channels_cover_the_body_once():
    s = load_skeleton(roots.find("skeletons/humanoid.yaml", roots.CORE))
    assert {"root", "hips", "spine", "legs", "left_hand", "right_hand", "gaze"} <= set(s.channels)
    owned = [b for c in s.channels.values() for b in c["bones"]]
    assert len(owned) == len(set(owned)), "a bone is in two channels"
    assert "RightIndexDistal" in s.channels["right_hand"]["bones"]
    assert s.channels["right_hand"]["end"] == "RightHand"
    assert s.channels["gaze"]["shared"]


@pytest.mark.parametrize("name", MOTIONS)
def test_motion_files_validate(name):
    validate_motion(load_motion(name, roots.CORE))


def test_unknown_beat_key_is_an_error():
    spec = {"kind": "motion", "name": "bad", "beats": {"a": {"pose": "sit", "wobble": 1}}}
    with pytest.raises(MotionError, match="wobble"):
        validate_motion(spec)


def test_scenarios_load():
    names = scenario_names(roots.CORE)
    assert {"sit_on_chair", "sit_interrupted", "press_tv", "sit_watch_tv"} <= set(names)
    for name in names:
        load_scenario(name, roots.CORE)


@pytest.fixture(scope="module")
def runs():
    return {}


def _run(runs, name):
    if name not in runs:
        runs[name] = run_scenario(name)
    return runs[name]


def test_sit_lands_and_rises_without_sliding(runs):
    r = _run(runs, "sit_on_chair")
    m = r.metrics
    assert m["beats"] == ["sit.arrive", "sit.lower", "sit.seated", "sit.rise"]
    assert m["foot_slide_mm"] < 10
    assert m["pops"] == 0 and m["peak_joint_speed"]["deg_s"] < 400
    assert r.rig.posture == "stand"
    assert abs(r.rig.root[1, 3]) < 1e-6, "standing again on the floor"


def test_interrupted_sit_gets_up_through_its_exit(runs):
    m = _run(runs, "sit_interrupted").metrics
    assert [e["event"] for e in m["events"]] == ["start", "interrupt", "end"]
    assert m["beats"][-1] == "sit.rise"
    assert m["duration_s"] < 6, "the ten-second sit was cut short"
    assert m["pops"] == 0 and m["settle_s"] and m["settle_s"][0] < 2


def test_press_reaches_the_button_and_emits(runs):
    m = _run(runs, "press_tv").metrics
    assert m["beats"] == ["press.reach", "press.push", "press.back"]
    assert any(e.get("emit") == "pressed" for e in m["events"])
    assert m["contact_error_mm"] < 40
    assert m["pops"] == 0


def test_glance_shares_the_body_with_sitting(runs):
    r = _run(runs, "sit_watch_tv")
    during = [s for s in r.rig.samples if "glance" in s["beats"]]
    assert during, "the glance ran"
    for s in during:
        assert s["owners"]["gaze"] == "glance"
        assert s["owners"]["legs"] == "sit"
    assert all("gaze" in s["aims"] for s in during)
    assert r.metrics["foot_slide_mm"] < 10
