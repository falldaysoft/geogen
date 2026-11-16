"""Motions (geogen.103): body channels, the motion format, the reference rig and the review scenarios.

The scenarios run headless here (no rendering); their numbers are what the review page shows.
"""

from pathlib import Path

import numpy as np
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


@pytest.mark.parametrize("name", ["sit_on_chair", "sit_interrupted", "sit_watch_tv"])
def test_sitting_keeps_the_hands_out_of_the_skirt_and_rests_them_on_it(runs, name):
    # The arms follow the body's cleared transition clip and the skirt its drape correctives;
    # blended freely, the forearms went 100 mm into the skirt on the way down and up.
    r = _run(runs, name)
    m = r.metrics
    assert "cloth_mm" in m, "the scenario's person wears a draped skirt"
    assert m["cloth_mm"]["mm"] < 2, m["cloth_mm"]
    assert m["peak_joint_accel"]["deg_s2"] < 4000, m["peak_joint_accel"]
    # Seated, the palms rest on the lap (fit_contacts lays them on the skirt as posed), not over it.
    seated = [s["cloth"] for s in r.rig.samples if s["beats"].get("sit") == "seated" and "glance" not in s["beats"]]
    assert seated and max(seated) > -0.01, f"hands {-max(seated) * 1000:.0f} mm off the lap"


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
    assert m["contact_error_mm"] < 5, "IK lands on the button (inertialization is laid before the solve)"
    assert m["pops"] == 0
    # Smooth: no joint's speed swings frame to frame (a stepped spine bend, a gaze turn restarted
    # every beat and a hard neck clamp made it 5700 deg/s2).
    assert m["peak_joint_accel"]["deg_s2"] < 4000, m["peak_joint_accel"]


def test_glance_shares_the_body_with_sitting(runs):
    r = _run(runs, "sit_watch_tv")
    during = [s for s in r.rig.samples if "glance" in s["beats"]]
    assert during, "the glance ran"
    for s in during:
        assert s["owners"]["gaze"] == "glance"
        assert s["owners"]["legs"] == "sit"
    assert all("gaze" in s["aims"] for s in during)
    assert r.metrics["foot_slide_mm"] < 10


def test_hand_shapes_are_declared_for_the_left_hand_and_mirrored():
    s = load_skeleton(roots.find("skeletons/humanoid.yaml", roots.CORE))
    assert {"relaxed", "open", "flat", "press", "power", "pinch", "hook"} <= set(s.hand_shapes)
    press = s.hand_shapes["press"]
    assert press["RightMiddleProximal"] == [0.0, 0.0, -press["LeftMiddleProximal"][2]]
    assert abs(press["LeftIndexProximal"][2]) < abs(press["LeftMiddleProximal"][2]), "the index points"


def test_press_takes_the_button_grips_hand_shape_through_the_interactions_touch(runs):
    import numpy as np

    from geogen.core.transform import Transform

    r = _run(runs, "press_tv")
    assert r.rig.trace[0]["event"] == "start"
    skeleton = r.rig.skeleton
    want = skeleton.body_quat("RightMiddleProximal", Transform(rotation=np.radians(
        skeleton.hand_shapes["press"]["RightMiddleProximal"])).to_matrix()[:3, :3])
    pushing = [s for s in r.rig.samples if s["beats"].get("press") == "push"]
    assert pushing
    got = pushing[-1]["rotations"]["RightMiddleProximal"]
    assert abs(float(np.dot(got, want))) > 0.999, "the middle finger is curled as press says"
    assert r.metrics["contact_error_mm"] < 5, "tv.power reached the power button (its touch:)"


def test_interaction_touch_must_name_an_attachment():
    from geogen.layout.loader import LayoutLoader

    text = """
name: box
size: [1, 1, 1]
parts:
  body: { primitive: cube, size: [1, 1, 1] }
attachments:
  knob: { anchor: top_center }
interactions:
  toggle:
    touch: handle
    targets: [body]
    states: { a: { next: b }, b: { next: a } }
"""
    with pytest.raises(ValueError, match="touch 'handle'"):
        LayoutLoader().load_string(text)


# --- exercises and the runtime contract (geogen.103.12) ---

def test_exercises_repeat_and_stay_smooth(runs):
    squat = _run(runs, "squat")
    assert squat.metrics["beats"] == ["squat.down", "squat.up"] * 3
    assert squat.metrics["pops"] == 0 and squat.metrics["foot_slide_mm"] < 1
    assert squat.metrics["contact_error_mm"] < 5, "the hands reach where the squat puts them"
    jacks = _run(runs, "jumping_jacks")
    assert jacks.metrics["beats"] == ["jumping_jack.jump_out", "jumping_jack.land_out", "jumping_jack.jump_in",
                                      "jumping_jack.land_in"] * 4
    assert jacks.metrics["pops"] == 0
    assert jacks.rig.posture == "stand" and jacks.rig.running() == []


def test_mirrored_body_frame_targets_cross_the_body(runs):
    r = _run(runs, "squat")
    arrived = [s["aims"] for s in r.rig.samples if "LeftHand@arrived" in s["aims"]]
    assert arrived
    left, right = arrived[0]["LeftHand@arrived"], arrived[0]["RightHand@arrived"]
    inv = np.linalg.inv(r.rig.root)
    lx, rx = (inv @ np.r_[left, 1])[0], (inv @ np.r_[right, 1])[0]
    assert lx > 0.05 and abs(lx + rx) < 1e-6, "the right hand's target is the left's across the body"


def test_rig_extras_rebuild_the_skeleton(runs):
    from geogen.core.transform import matrix_from_quat
    from geogen.motion import rig_extras

    r = _run(runs, "squat")
    extras = rig_extras(r.rig.actor)
    s = r.rig.skeleton
    world = {}
    for bone in extras["bones"]:
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = matrix_from_quat(bone["rest"][3:]), bone["rest"][:3]
        world[bone["name"]] = world[bone["parent"]] @ m if bone["parent"] else m
    assert max(np.abs(world[n][:3, :3] - s[n].rotation).max() for n in s.names) < 1e-5
    assert max(np.abs(world[n][:3, 3] - s[n].head).max() for n in s.names) < 1e-5
    assert {"stand", "sit", "t_pose"} <= set(extras["poses"])
    assert set(extras["channels"]) == set(s.channels) and "flat" in extras["hand_shapes"]


def test_motions_json_holds_every_motion(tmp_path):
    import json

    from geogen.motion import motion_names, write_motions

    data = json.loads(write_motions(tmp_path).read_text())
    assert data["format"] == "geogen-motions" and data["version"] == 1
    assert set(data["motions"]) == set(motion_names()) >= {"sit", "press", "glance", "squat", "jumping_jack"}
    assert data["motions"]["squat"]["beats"]["up"]["loop"] == {"to": "down", "times": "{reps}"}, "{expr} left for the runtime"


@pytest.mark.parametrize("name", ["squat", "jumping_jacks"])
def test_motion_traces_are_what_the_reference_rig_does(name):
    """The stored conformance traces (tests/data/motion, the runtime rig replays them) still
    match the reference rig: after a change to it, run tests/data/motion/make_traces.py."""
    import json

    from geogen.motion_review import motion_trace

    stored = json.loads((Path(__file__).parent / "data" / "motion" / f"{name}.json").read_text())
    now = motion_trace(name)
    assert now["expect"]["events"] == stored["expect"]["events"]
    worst = max(abs(a - b) for f, g in zip(now["expect"]["positions"], stored["expect"]["positions"])
                for bone in f["positions"] for a, b in zip(f["positions"][bone], g["positions"][bone]))
    assert worst < 1e-4, f"the trace is stale by {worst * 1000:.2f} mm: run tests/data/motion/make_traces.py"
