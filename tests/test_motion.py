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
    assert {"sit_on_chair", "sit_interrupted", "press_tv", "sit_watch_tv", "take_carry_set_down",
            "watch_tv"} <= set(names)
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


# --- take, carry, set down; watch (geogen.103.4) ---

def _node(r, name):
    return next(n for n in r.rig.scene.iter_nodes() if n.name == name and n.parent is r.rig.scene
                or n.name == name and n.parent.name == "stage")


def test_take_carry_set_down_moves_the_plate_onto_the_place(runs):
    r = _run(runs, "take_carry_set_down")
    m = r.metrics
    assert m["beats"] == ["take.reach", "carry.held", "take.grasp", "take.lift", "set_down.over",
                          "set_down.lower", "set_down.let_go", "set_down.back"]
    events = [(e["event"], e["motion"]) for e in r.rig.trace if e["event"] in ("attach", "release", "emit", "end")]
    assert events == [("attach", "take"), ("emit", "take"), ("end", "take"), ("release", "set_down"),
                      ("end", "carry"), ("emit", "set_down"), ("end", "set_down")], \
        "the release ends the carry; nothing is left running"
    assert m["contact_error_mm"] < 5 and m["pops"] == 0
    assert m["peak_joint_accel"]["deg_s2"] < 4000, m["peak_joint_accel"]
    assert m["cloth_mm"]["mm"] < 2, m["cloth_mm"]
    plate, table = _node(r, "plate"), _node(r, "table")
    top = table.get_attachment("top").to_matrix()[:3, 3]
    world = plate.world_transform()
    assert np.linalg.norm(world[:3, 3] - top) < 0.005, "set down on the table's top (2 mm above it)"
    assert world[1, 3] > top[1], "not into the table"
    assert abs(world[1, 1] - 1.0) < 1e-6, "level, as it was carried"
    assert r.rig.attached == {}


def test_a_carried_plate_goes_with_the_hand_and_stays_level(runs):
    r = _run(runs, "take_carry_set_down")
    held = [s for s in r.rig.samples if s["beats"].get("carry") == "held" and "take" not in s["beats"]
            and "set_down" not in s["beats"]]
    assert held and all(s["owners"]["right_hand"] == "carry" for s in held), "carry has the hand between"
    # The pinch the plate's rim grip names, held through the carry.
    skeleton = r.rig.skeleton
    from geogen.core.transform import Transform

    want = skeleton.body_quat("RightIndexProximal", Transform(rotation=np.radians(
        skeleton.hand_shapes["pinch"]["RightIndexProximal"])).to_matrix()[:3, :3])
    assert abs(float(np.dot(held[-1]["rotations"]["RightIndexProximal"], want))) > 0.999
    # Lifted: the hand at waist height, well above the table.
    assert held[-1]["ends"]["RightHand"][1] > 0.85


def test_set_down_needs_something_in_hand():
    from geogen.motion import MotionError, MotionRig
    from geogen.motion_review import _actor, _scene

    spec = load_scenario("take_carry_set_down")
    scene = _scene(spec["scene"], roots.CORE)
    rig = MotionRig(_actor(spec["actor"], scene, roots.CORE), scene)
    with pytest.raises(MotionError, match="no hand holds"):
        rig.start("set_down", item="plate", place="table.top")


def test_grip_any_takes_the_first_grip_a_hand_holds():
    from geogen.layout.loader import LayoutLoader
    from geogen.motion import contact

    plate = LayoutLoader().load(roots.find("plate.yaml", roots.CORE))
    assert contact(plate, "any").name.endswith(".rim") and contact(plate, "any").grip == "pinch"
    tv = LayoutLoader().load(roots.find("tv.yaml", roots.CORE))
    assert not contact(tv, "any").name.endswith((".power_button", ".screen")), "press and look aren't held"


def test_watch_gives_the_gaze_to_a_glance_and_takes_it_back(runs):
    r = _run(runs, "watch_tv")
    owners = [s["owners"]["gaze"] for s in r.rig.samples]
    runs_of = [o for i, o in enumerate(owners) if i == 0 or o != owners[i - 1]]
    assert runs_of == [None, "watch", "glance", "watch", None]
    m = r.metrics
    assert m["pops"] == 0 and m["peak_joint_accel"]["deg_s2"] < 4000, m["peak_joint_accel"]


def test_a_look_that_ends_eases_off_instead_of_snapping(runs):
    # take's gaze on the plate ends with take; inertialized at the gaze's 0.08 s half-life the head
    # sprang back at 690 deg/s.
    r = _run(runs, "take_carry_set_down")
    end = next(e["t"] for e in r.rig.trace if e["event"] == "end" and e["motion"] == "take")
    s = r.rig.samples
    fastest = 0.0
    for a, b in zip(s, s[1:]):
        if end - 0.1 < b["t"] < end + 0.5:
            q, p = b["rotations"]["Head"], a["rotations"]["Head"]
            fastest = max(fastest, np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(p, q)))))) / (b["t"] - a["t"]))
    assert fastest < 150, fastest


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
    assert set(data["motions"]) == set(motion_names()) >= {"sit", "press", "glance", "squat", "jumping_jack",
                                                                "take", "carry", "set_down", "watch"}
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
