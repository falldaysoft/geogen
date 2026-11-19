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
            "watch_tv", "eat_glance", "eat_interrupted", "eat_wave", "squat_shoved", "bow_wave",
            "eat_watch_tv", "watch_by_turns"} <= set(names)
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
    assert m["beats"] == ["press.reach", "press.push", "press.withdraw", "press.back"]
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
    # The rim shape the plate's rim grip names, held through the carry.
    skeleton = r.rig.skeleton
    from geogen.core.transform import Transform

    want = skeleton.body_quat("RightIndexProximal", Transform(rotation=np.radians(
        skeleton.hand_shapes["rim"]["RightIndexProximal"])).to_matrix()[:3, :3])
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
    assert contact(plate, "any").name.endswith(".rim") and contact(plate, "any").grip == "rim"
    tv = LayoutLoader().load(roots.find("tv.yaml", roots.CORE))
    assert not contact(tv, "any").name.endswith((".power_button", ".screen")), "press and look aren't held"


def test_watch_gives_the_gaze_to_a_glance_and_takes_it_back(runs):
    r = _run(runs, "watch_tv")
    owners = [s["owners"]["gaze"] for s in r.rig.samples]
    runs_of = [o for i, o in enumerate(owners) if i == 0 or o != owners[i - 1]]
    assert runs_of == [None, "watch", "glance", "watch", None]
    m = r.metrics
    assert m["pops"] == 0 and m["peak_joint_accel"]["deg_s2"] < 4000, m["peak_joint_accel"]


# --- multitasking: a shared gaze taken by turns (geogen.103.6) ---

def _gaze_turns(r):
    """[(motion, t)] each time the gaze changed hands (None: nobody has it)."""
    out = []
    for e in r.rig.trace:
        if e["event"] == "owner" and e.get("channel") == "gaze":
            out.append((e["motion"], e["t"]))
    return out


def test_eating_and_watching_take_turns_at_the_gaze(runs):
    """eat (share 1) and watch (share 1.5) both share the gaze: the TV gets 3 s turns, the plate
    2 s, by turns, while the hand eats on and every bite is taken."""
    r = _run(runs, "eat_watch_tv")
    turns = _gaze_turns(r)
    who = [m for m, _ in turns]
    assert who[:4] == ["watch", "eat", "watch", "eat"], turns
    lengths = [round(b - a, 2) for (_, a), (_, b) in zip(turns, turns[1:4])]
    assert lengths == [3.0, 2.0, 3.0], lengths
    assert [e[1] for e in _events(r, "emit")] == ["eat"] * 4, "every bite taken"
    assert all(s["owners"]["right_hand"] == "eat" for s in r.rig.samples if "eat" in s["beats"])
    assert not _events(r, "pause", "resume", "interrupt")
    m = r.metrics
    assert m["pops"] == 0 and m["contact_error_mm"] < 5 and m["cloth_mm"]["mm"] < 2


def test_a_glance_takes_a_shared_gaze_outright(runs):
    """Two watches share the gaze by turns (3 s each); a glance claims it, which outranks a share,
    and gives it back to a watch when it ends."""
    r = _run(runs, "watch_by_turns")
    turns = _gaze_turns(r)
    assert [m for m, _ in turns] == ["watch", "watch", "glance", "watch", None], turns
    assert round(turns[1][1] - turns[0][1], 2) == 3.0
    assert all(s["owners"]["gaze"] == "glance" for s in r.rig.samples if "glance" in s["beats"])
    assert r.metrics["pops"] == 0


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


# --- interruption: overlays and resume, yield and finish, break (geogen.103.5) ---

def _events(r, *kinds):
    return [(e["event"], e["motion"], e.get("beat")) for e in r.rig.trace if e["event"] in kinds]


def _smooth(m, accel=4000):
    assert m["pops"] == 0, m["peak_joint_speed"]
    assert m["peak_joint_accel"]["deg_s2"] < accel, m["peak_joint_accel"]


def test_a_glance_while_eating_leaves_the_meal_alone(runs):
    r = _run(runs, "eat_glance")
    assert not _events(r, "pause", "resume", "interrupt"), "the gaze is shared: nothing waits for it"
    assert [e[1] for e in _events(r, "emit")] == ["eat"] * 3, "every bite taken"
    during = [s for s in r.rig.samples if "glance" in s["beats"]]
    assert during and all(s["owners"]["gaze"] == "glance" and s["owners"]["right_hand"] == "eat" for s in during)
    m = r.metrics
    _smooth(m)
    assert m["contact_error_mm"] < 5 and m["cloth_mm"]["mm"] < 2


def test_an_interrupted_meal_finishes_the_bite_and_puts_the_hand_down(runs):
    r = _run(runs, "eat_interrupted")
    m = r.metrics
    eat = [b for b in m["beats"] if b.startswith("eat.")]
    assert eat[-2:] == ["eat.bite", "eat.rest"], "bite says finish: it lands, then its exit"
    assert ("interrupt", "eat", "bite") in _events(r, "interrupt")
    assert ("emit", "eat", "bite") in _events(r, "emit"), "the bite in progress was taken"
    rest_end = next(e["t"] for e in r.rig.trace if e["event"] == "end" and e["motion"] == "eat")
    rise = next(e["t"] for e in r.rig.trace if e["event"] == "beat" and e["beat"] == "rise")
    assert rise > rest_end, "she gets up once the hand is down"
    assert r.rig.posture == "stand" and abs(r.rig.root[1, 3]) < 1e-6
    _smooth(m)
    assert m["cloth_mm"]["mm"] < 2


def test_a_wave_pauses_the_meal_and_it_resumes_where_it_was(runs):
    r = _run(runs, "eat_wave")
    pauses = [e for e in r.rig.trace if e["event"] in ("pause", "resume")]
    assert [(e["event"], e["motion"]) for e in pauses] == [("pause", "eat"), ("resume", "eat")]
    assert pauses[0]["channels"] == ["right_hand"] and pauses[0]["beat"] == "chew"
    assert pauses[1]["to"] == "chew", "chew's resume: the bite was eaten, chew on"
    stopped = [s for s in r.rig.samples if "wave" in s["beats"] and pauses[0]["t"] < s["t"] < pauses[1]["t"]]
    assert stopped and len({s["beats"]["eat"] for s in stopped}) == 1, "eat's clock stands still under the wave"
    assert [e[1] for e in _events(r, "emit")] == ["eat"] * 2, "both bites, none lost or repeated"
    _smooth(r.metrics)


def test_a_paused_motion_asked_to_yield_leaves_through_its_exit(runs):
    from geogen.motion import MotionRig
    from geogen.motion_review import _actor, _scene

    spec = load_scenario("bow_wave")
    scene = _scene(spec["scene"], roots.CORE)
    rig = MotionRig(_actor(spec["actor"], scene, roots.CORE), scene)
    bow = rig.start("bow")
    for _ in range(10):
        rig.step()
    wave = rig.start("wave")
    rig.step()
    assert bow.paused == "lower"
    assert rig.interrupt(bow) == "exiting" and bow.beat == "lower", "it waits for the hand"
    while wave in rig.running():
        rig.step()
    rig.step()
    assert bow.beat == "up", "back through its exit, not its resume"


def test_a_break_drops_everything_into_a_stagger(runs):
    r = _run(runs, "squat_shoved")
    events = _events(r, "break", "end", "start")
    assert events[:4] == [("start", "squat", None), ("break", None, None), ("end", "squat", "up"),
                          ("start", "stagger", None)]
    assert r.metrics["beats"][-3:] == ["stagger.jolt", "stagger.catch", "stagger.recover"]
    assert "squat.down" not in r.metrics["beats"][2:], "no exits: the squat just stops"
    assert r.metrics["pops"] == 0 and r.metrics["foot_slide_mm"] < 1
    assert r.rig.running() == []


def test_a_break_lets_go_of_what_is_carried():
    from geogen.motion import MotionRig
    from geogen.motion_review import _actor, _scene

    spec = load_scenario("take_carry_set_down")
    scene = _scene(spec["scene"], roots.CORE)
    rig = MotionRig(_actor(spec["actor"], scene, roots.CORE), scene)
    rig.start("take", item="plate")
    rig.start("carry", item="plate")
    for _ in range(75):
        rig.step()
    plate = next(n for n in scene.iter_nodes() if n.name == "plate" and n.parent is scene)
    assert rig.attached and plate.world_transform()[1, 3] > 0.8
    rig.break_("stagger")
    assert rig.attached == {} and [r.name for r in rig.running()] == ["stagger"]
    assert ("release", None) in [(e["event"], e["motion"]) for e in rig.trace]
    for _ in range(40):
        rig.step()
    assert abs(plate.world_transform()[1, 3] - rig.floor) < 1e-9, "dropped to the floor"


def test_a_free_arm_hangs_when_the_spine_bends(runs):
    r = _run(runs, "bow_wave")
    s = r.rig.skeleton
    standing = r.rig.poses["stand"]
    fk = s.fk(standing)
    hang = fk["LeftHand"][:3, 3] - fk["LeftUpperArm"][:3, 3]
    bowed = [x for x in r.rig.samples if x["beats"].get("bow") == "held" and "wave" not in x["beats"]]
    assert bowed
    pose = standing.copy()
    pose.rotations.update(bowed[-1]["rotations"])
    fk = s.fk(pose)
    arm = fk["LeftHand"][:3, 3] - fk["LeftUpperArm"][:3, 3]
    # HANG_SHARE of the way from swinging with the chest back to hanging plumb.
    from geogen.motion import HANG_SHARE

    pose.rotations["LeftUpperArm"] = standing.rotations.get("LeftUpperArm", np.array([0, 0, 0, 1.0]))
    fk = s.fk(pose)
    swung = fk["LeftHand"][:3, 3] - fk["LeftUpperArm"][:3, 3]

    def angle(a, b):
        return np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))

    assert angle(swung, hang) > 10, "the bow bends the spine"
    assert abs(angle(arm, hang) - (1 - HANG_SHARE) * angle(swung, hang)) < 2, \
        "the left arm hangs most of the way back to plumb, not swung back with the chest"
    _smooth(r.metrics, accel=5000)


def test_a_beats_spine_eases_in_under_a_reach(runs):
    # A beat that reaches and poses the spine (the bow, the squat's lean) laid the spine whole on its
    # first frame: the reach bend started from the beat's pose, not the spine as laid.
    r = _run(runs, "bow_wave")
    s = r.rig.samples
    fastest = max(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(a["rotations"]["Spine"], b["rotations"]["Spine"]))))))
                  / (b["t"] - a["t"]) for a, b in zip(s, s[1:]))
    assert fastest < 60, fastest


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
                                                                "take", "carry", "set_down", "watch",
                                                                "eat", "wave", "stagger", "bow"}
    assert data["motions"]["squat"]["beats"]["up"]["loop"] == {"to": "down", "times": "{reps}"}, "{expr} left for the runtime"


@pytest.mark.parametrize("name", ["squat", "jumping_jacks", "squat_shoved", "bow_wave", "emotes", "idles"])
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


# --- idles, emotes, dances and things on the table (geogen.103.17) -----------------------------------

def test_every_motion_is_in_the_catalogue():
    """motion_catalogue lists every motion file with a category and a one-line summary (what the
    runtime's gallery lists); the body-only ones are what the runtime rig plays today."""
    from geogen.motion import CATEGORIES, motion_catalogue, motion_names

    entries = motion_catalogue()
    assert sorted(e["name"] for e in entries) == sorted(motion_names())
    for e in entries:
        assert e["category"] in CATEGORIES and e["summary"], e["name"]
    by_name = {e["name"]: e for e in entries}
    assert by_name["clap"]["body_only"] and by_name["clap"]["category"] == "emote"
    assert not by_name["drink"]["body_only"] and by_name["drink"]["props"] == {"cup": {"asset": "mug.yaml"}}
    assert len([e for e in entries if e["category"] == "idle"]) >= 10
    assert len([e for e in entries if e["category"] in ("emote", "dance")]) >= 20
    order = [CATEGORIES.index(e["category"]) for e in entries]
    assert order == sorted(order), "listed by category"


def test_motions_json_carries_the_catalogue(tmp_path):
    import json

    from geogen.motion import motion_catalogue, write_motions

    data = json.loads(write_motions(tmp_path).read_text())
    assert data["catalogue"] == json.loads(json.dumps(motion_catalogue()))


def test_props_name_roles_and_category_is_known():
    base = {"kind": "motion", "name": "bad", "beats": {"a": {"after": 1}}}
    with pytest.raises(MotionError, match="category"):
        validate_motion({**base, "category": "party"})
    with pytest.raises(MotionError, match="props.cup"):
        validate_motion({**base, "props": {"cup": {"asset": "mug.yaml"}}})


GALLERY = [n for n in MOTIONS if load_motion(n).get("category") in ("idle", "emote", "dance", "reaction")]


@pytest.fixture(scope="module")
def demos():
    """Every idle, emote, dance and reaction on its demonstration stage (demo_scenario), on the
    young woman (a skirt: the arms' clearance is measured) and the older man (a heavier build)."""
    from geogen.motion_review import demo_scenario

    return {(name, who): run_scenario(demo_scenario(name, who))
            for name in GALLERY for who in ("young_woman", "older_man")}


@pytest.mark.parametrize("name", GALLERY)
def test_gallery_motions_play_through_cleanly(demos, name):
    """Each runs to its end and leaves the body standing, with its feet where they were and its
    hands where its beats put them; no joint snaps (`pops`) except a start's deliberate jerk."""
    for who in ("young_woman", "older_man"):
        r = demos[(name, who)]
        m = r.metrics
        assert r.rig.running() == [] and r.rig.posture == "stand", (name, who)
        assert m["foot_slide_mm"] < 2, (name, who, m["foot_slide_mm"])
        # A fast beat (a clap's quarter-second meet) stops a frame short of its goal.
        assert m["contact_error_mm"] < 40, (name, who, m["contact_error_mm"])
        assert m["pops"] <= (2 if name in ("surprised", "fist_pump", "celebrate_jump", "stagger") else 1), (name, who, m["peak_joint_speed"])


def test_hand_turn_points_the_fingers_and_faces_the_palm():
    """`fingers` / `palm` on an ik entry turn the hand: the clap's meet has the palms facing each
    other and the fingers up and ahead, mirrored on the right."""
    from geogen.motion import MotionRig
    from geogen.motion_review import _actor, _scene

    scene = _scene({"place": {}}, roots.CORE)
    actor = _actor({"archetype": "characters/archetypes/young_man.yaml", "seed": 2}, scene, roots.CORE)
    rig = MotionRig(actor, scene)
    rig.start("clap", claps=1)
    while rig.running() and rig.running()[0].beat != "part":
        rig.step()
    fk = rig.skeleton.fk(rig.out)
    for side, sign in (("Left", 1), ("Right", -1)):
        hand, middle = fk[f"{side}Hand"][:3, 3], fk[f"{side}MiddleProximal"][:3, 3]
        fingers = (middle - hand) / np.linalg.norm(middle - hand)
        want = np.array([-0.2 * sign, 0.6, 1.0])
        assert fingers @ (want / np.linalg.norm(want)) > 0.97, (side, fingers)
        # The palm faces across the body: the thumb side of the hand is up, the hand's flat across.
        assert abs(fingers[0]) < 0.3
    assert abs(fk["LeftHand"][0, 3] + fk["RightHand"][0, 3]) < 0.01, "mirrored"


@pytest.mark.parametrize("name", ["drink", "inspect"])
def test_a_thing_taken_from_the_table_goes_back_where_it_stood(runs, name):
    """drink and inspect take the mug from the table in front, use it (it tips and turns with the
    hand) and put it back where it stood (`cup@start`), upright and let go of."""
    from geogen import refs
    from geogen.motion_review import _scene

    r = _run(runs, name)
    m = r.metrics
    assert r.rig.running() == [] and not r.rig.attached
    assert m["pops"] == 0 and m["foot_slide_mm"] < 1 and m["contact_error_mm"] < 5
    assert [e["emit"] for e in r.rig.trace if e["event"] == "emit"] == ["taken", "put"]
    before = refs.resolve(_scene(r.scenario["scene"], roots.CORE), "cup").world_transform()
    after = refs.resolve(r.rig.scene, "cup").world_transform()
    assert np.linalg.norm(after[:3, 3] - before[:3, 3]) < 0.004, "back where it stood"
    assert after[:3, 1] @ [0, 1, 0] > 0.999, "upright"


def test_every_motion_has_a_stage():
    """demo_scenario stages every motion: body-only ones on a rug, ones with things at a table
    with their props on it in front of the person, a seat (`behind`) behind them (a motion with
    roles and no props can't be shown)."""
    from geogen.motion import motion_catalogue
    from geogen.motion_review import demo_scenario

    for e in motion_catalogue():
        if e["partners"]:
            # Two-person: the other person where its role places them, its props at someone's side.
            spec = demo_scenario(e["name"])
            assert set(spec["partners"]) == set(e["partners"]), e["name"]
            continue
        if e["roles"] and not e["props"]:
            with pytest.raises(MotionError, match="no props"):
                demo_scenario(e["name"])
            continue
        spec = demo_scenario(e["name"])
        place = spec["scene"]["place"]
        on_table = {r: p for r, p in e["props"].items() if "behind" not in p}
        assert ("table" in place) == bool(on_table), e["name"]
        for role, prop in e["props"].items():
            staged = place[spec["script"][0][role]]
            if "behind" in prop:
                assert spec["actor"]["at"] == spec["script"][0][role] and spec["actor"]["offset"][2] == prop["behind"]
            else:
                assert staged["on"] == "table.top"


def _bite(name):
    """The scenario up to the end of its bites, the thing still carried."""
    spec = load_scenario(name)
    spec["script"] = [{k: v for k, v in e.items() if k != "duration"} for e in spec["script"] if e["start"] != "set_down"]
    return run_scenario(spec)


@pytest.mark.parametrize("name, hand", [("bite_apple", "RightHand"), ("bite_apple_left", "LeftHand")])
def test_bite_borrows_the_hand_that_holds_the_apple(name, hand):
    rig = _bite(name).rig
    attaches = [e for e in rig.trace if e["event"] == "attach"]
    assert {e["hand"] for e in attaches} == {hand}                  # mirrored for the left hand, not re-taken
    assert list(rig.attached) == ["item"] and rig.attached["item"].end == hand   # still held, by the role that took it
    assert [r.name for r in rig.running()] == [f"carry{'_left' if hand == 'LeftHand' else ''}"]   # carry has the hand back
    assert sum(e["event"] == "emit" and e.get("emit") == "bite" for e in rig.trace) >= 1


def test_a_held_role_needs_a_hand_holding_the_thing():
    rig = _bite("bite_apple").rig
    from geogen import refs

    table = refs.resolve(rig.scene, "table")
    with pytest.raises(MotionError, match="no hand holds it"):
        rig.start("bite", item=table)


def test_while_couples_motions_to_the_walk():
    rig = _bite("bite_apple").rig
    assert rig.coupling("walk") == {"speed": 1.0, "arm_swing": 0.5}   # carry alone
    rig.start("bite", item=rig.attached["item"].node)
    assert rig.coupling("walk") == pytest.approx({"speed": 0.8, "arm_swing": 0.3})


def test_while_only_takes_shares_of_what_it_knows():
    spec = {"kind": "motion", "name": "x", "beats": {"a": {}}, "while": {"walk": {"speed": 2}}}
    with pytest.raises(MotionError):
        validate_motion(spec)
    with pytest.raises(MotionError):
        validate_motion({**spec, "while": {"swim": {"speed": 0.5}}})


# --- two-person motions (geogen.103.10) ---

def _beats(rig, motion):
    return [(e["t"], e["beat"]) for e in rig.trace if e["event"] == "beat" and e["motion"] == motion]


def test_a_plate_is_handed_over(runs):
    """give / receive: the giver holds the plate out, the taker's grasp takes it out of the giver's
    hand (no drop: the giver's carry ends, give runs on), and both go on from each shared beat in
    the same step."""
    r = _run(runs, "hand_over")
    giver, taker = r.rig, r.partners["taker"]
    assert not giver.running() and not taker.running()
    assert list(taker.attached) == ["item"] and not giver.attached and not giver.falling
    released = [e for e in giver.trace if e["event"] == "release"]
    assert released == [{"t": released[0]["t"], "event": "release", "motion": None, "role": "item", "to": "taker"}]
    ended = {e["motion"]: e["t"] for e in giver.trace if e["event"] == "end"}
    assert ended["carry"] == released[0]["t"], "its carry ended at the hand-over"
    assert [e["emit"] for e in taker.trace if e["event"] == "emit"] == ["received"]
    give, receive = dict((b, t) for t, b in _beats(giver, "give")), dict((b, t) for t, b in _beats(taker, "receive"))
    assert give["hold"] == receive["reach"], "offered: both go on together"
    assert give["let_go"] == receive["lift"], "taken: the giver lets go as the taker lifts"
    for m in (r.metrics, r.metrics["partners"]["taker"]):
        assert m["pops"] == 0 and m["contact_error_mm"] < 5 and m["foot_slide_mm"] < 1


def test_a_hug_opens_and_parts_together(runs):
    """hug on both bodies, each the other's partner: the same beats at the same times, the hands
    round the other's back (behind their chest, seen from in front) while they hold on."""
    r = _run(runs, "hug")
    a, b = r.rig, r.partners["friend"]
    assert _beats(a, "hug") == _beats(b, "hug")
    assert [x for _, x in _beats(a, "hug")] == ["open", "embrace", "part", "back"]
    held = next(t for t, x in _beats(a, "hug") if x == "embrace") + 1.0
    for me, them in ((a, b), (b, a)):
        s = next(s for s in me.samples if s["t"] >= held)
        theirs = next(x for x in them.samples if x["t"] >= held)
        ahead = theirs["root"][:3, 3] - s["root"][:3, 3]
        ahead[1] = 0
        ahead /= np.linalg.norm(ahead)
        chest = (theirs["root"] @ them.skeleton.fk(them.out)["Chest"])[:3, 3]   # (they hold still while hugging)
        for hand in ("LeftHand", "RightHand"):
            assert (s["ends"][hand] - chest) @ ahead > 0.0, (me.actor.name, hand)
    assert r.metrics["pops"] == 0 and r.metrics["partners"]["friend"]["pops"] == 0


def test_a_shared_beat_gives_up_on_a_partner_that_never_comes():
    """give with a partner who runs nothing: the offer waits SYNC_WAIT at its shared beat, then
    leaves through its exit (back to carrying the plate)."""
    from geogen.motion import SYNC_WAIT, MotionRig
    from geogen.motion_review import _actor, _scene

    scene = _scene({"place": {"plate": {"asset": "plate.yaml"}}}, roots.CORE)
    giver = MotionRig(_actor({"archetype": "characters/archetypes/young_woman.yaml", "seed": 1}, scene, roots.CORE), scene)
    other = MotionRig(_actor({"archetype": "characters/archetypes/young_man.yaml", "seed": 2, "name": "other",
                              "offset": [0, 0, 1.0], "facing": "toward"}, scene, roots.CORE), scene)
    giver.start("take", item="plate")
    giver.start("carry", item="plate")
    while giver.running("take"):
        giver.step()
        other.step()
    giver.start("give", item="plate", partner=other)
    for _ in range(int((SYNC_WAIT + 3) * 30)):
        giver.step()
        other.step()
    failed = [e for e in giver.trace if e["event"] == "sync_failed"]
    assert len(failed) == 1 and failed[0]["sync"] == "offered" and failed[0]["partners"] == ["partner"]
    assert [b for _, b in _beats(giver, "give")] == ["offer", "keep"]
    assert list(giver.attached) == ["item"] and [r.name for r in giver.running()] == ["carry"]


def test_a_body_role_binds_a_rig():
    from geogen.motion import MotionRig
    from geogen.motion_review import _actor, _scene

    scene = _scene({"place": {}}, roots.CORE)
    rig = MotionRig(_actor({"archetype": "characters/archetypes/young_woman.yaml", "seed": 1}, scene, roots.CORE), scene)
    with pytest.raises(MotionError, match="another body"):
        rig.start("hug", partner="actor")
    with pytest.raises(MotionError, match="another body"):
        rig.start("hug", partner=rig)
    with pytest.raises(MotionError, match="body role"):
        validate_motion({"kind": "motion", "name": "x", "roles": {"thing": {"place": {"offset": [0, 0, 1]}}},
                         "beats": {"a": {}}})
