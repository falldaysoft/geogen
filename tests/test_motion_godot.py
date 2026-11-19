"""Motions in the Godot runtime (geogen.103.12): runtime/godot/scripts/motion_rig.gd replays the
conformance traces the reference rig wrote (tests/data/motion, geogen.motion_review.motion_trace),
and motion_player.gd runs motion files on the bodies of an export through the control API."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from geogen.runtime_client import step

TRACES = sorted((Path(__file__).parent / "data" / "motion").glob("*.json"))


def _results(output: str) -> dict[str, list[str]]:
    found = {}
    for line in output.splitlines():
        if line.startswith("trace: "):
            result = json.loads(line.removeprefix("trace: "))
            found[result["name"]] = result["problems"]
    return found


@pytest.fixture(scope="module")
def replayed(run_godot) -> dict[str, list[str]]:
    return _results(run_godot(*(f"--trace={path}" for path in TRACES)))


@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_motion_trace_conforms_in_godot(replayed, path):
    name = json.loads(path.read_text())["name"]
    assert name in replayed, f"no result for {name}: got {sorted(replayed)}"
    assert replayed[name] == []


def test_a_wrong_motion_trace_is_reported(tmp_path):
    from conftest import GODOT_PROJECT, _godot_binary

    if not _godot_binary():
        pytest.skip("no Godot")
    trace = json.loads(next(p for p in TRACES if p.stem == "squat").read_text())
    trace["expect"]["positions"][3]["positions"]["LeftHand"][1] += 0.01
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(trace))
    result = subprocess.run([_godot_binary(), "--headless", "--path", str(GODOT_PROJECT), "--", f"--trace={path}"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 1
    assert any("10.0 mm" in problem for problem in _results(result.stdout)["squat"])


def test_motion_files_run_on_bodies_in_the_runtime(godot_script, scene_export):
    """Squats and jumping jacks started through the control API run their beats and repetitions
    on the exported bodies and end; the squat takes the hips down and back and up again."""
    run = godot_script([
        step("motion.start", node="tomas", motion="squat", params={"reps": 2}),
        step("motion.start", node="ines", motion="jumping_jack", params={"reps": 3}),
        step("wait", seconds=1.15),
        step("motion.status", node="tomas"),
        step("wait", seconds=3.6),
        step("motion.status", node="tomas"),
        step("motion.status", node="ines"),
        step("motion.start", node="tomas", motion="no_such_motion"),
    ], "--scene=motion_test", f"--generated={scene_export('motion_test')}", check=False)
    statuses = [e["result"] for e in run.steps if e["cmd"] == "motion.status"]
    bottom, tomas, ines = statuses
    assert bottom["running"] == ["squat"]
    assert bottom["bones"]["Hips"][1] < tomas["bones"]["Hips"][1] - 0.25, "the hips went down"
    assert bottom["bones"]["Hips"][2] < tomas["bones"]["Hips"][2] - 0.1, "and back"
    beats = lambda s: [(e["event"], e["beat"]) for e in s["events"] if e["event"] in ("beat", "loop", "end")]
    assert tomas["running"] == [] and beats(tomas) == [("beat", "down"), ("beat", "up"), ("loop", "up"),
                                                       ("beat", "down"), ("beat", "up"), ("end", "up")]
    assert ines["running"] == [] and [b for b in beats(ines) if b[0] == "loop"] == [("loop", "land_in")] * 2
    missing = run.steps[-1]
    assert not missing["ok"] and "no motion 'no_such_motion'" in missing["error"]


def test_a_break_through_the_control_api_drops_into_a_stagger(godot_script, scene_export):
    """motion.break (geogen.103.5) drops the running squat with no exit and starts the stagger,
    which stands the body up again and ends."""
    run = godot_script([
        step("motion.start", node="tomas", motion="squat", params={"reps": 3}),
        step("wait", seconds=0.8),
        step("motion.break", node="tomas", reaction="stagger"),
        step("wait", seconds=2.0),
        step("motion.status", node="tomas"),
        step("motion.break", node="tomas", reaction="no_such_motion"),
    ], "--scene=motion_test", f"--generated={scene_export('motion_test')}", check=False)
    broke = next(e for e in run.steps if e["cmd"] == "motion.break")
    assert broke["ok"] and broke["result"]["running"] == ["stagger"]
    status = next(e["result"] for e in run.steps if e["cmd"] == "motion.status")
    events = [(e["event"], e["motion"]) for e in status["events"] if e["event"] in ("break", "start", "end")]
    assert events == [("start", "squat"), ("break", None), ("end", "squat"), ("start", "stagger"), ("end", "stagger")]
    assert status["running"] == []
    assert not run.steps[-1]["ok"] and "no motion 'no_such_motion'" in run.steps[-1]["error"]


def test_the_motion_gallery_lists_the_catalogue_and_plays_on_everyone(godot_script, scene_export):
    """The motion_gallery scene opens the HUD's gallery (geogen.103.18) on its own: it lists every
    motion file from the catalogue, plays a body-only one on all five people or on one, and
    refuses one that needs things the stage doesn't have (press: a TV)."""
    from geogen.motion import motion_names

    run = godot_script([
        step("hud.gallery"),
        step("hud.gallery", select="clap", play=True),
        step("wait", seconds=0.5),
        step("motion.status", node="walter"),
        step("hud.gallery", select="nod", performer="ines", play=True),
        step("wait", seconds=0.2),
        step("motion.status", node="ines"),
        step("motion.status", node="priya"),
        step("hud.gallery", select="press", play=True),
    ], "--scene=motion_gallery", f"--generated={scene_export('motion_gallery')}", check=False)
    opened = run.steps[0]["result"]
    assert opened["open"] and opened["performers"] == ["ines", "tomas", "ruth", "walter", "priya"]
    assert sorted(m["name"] for m in opened["motions"]) == sorted(motion_names())
    plays = {m["name"]: m["plays"] for m in opened["motions"]}
    assert plays["clap"] and plays["sit_ground"] and plays["drink"] and plays["take"] and plays["sit"]
    assert not plays["press"]
    assert run.steps[1]["result"]["playing"] == "clap"
    walter, ines, priya = (e["result"] for e in run.steps if e["cmd"] == "motion.status")
    assert walter["running"] == ["clap"], "everyone claps"
    assert ines["running"] == ["nod"] and priya["running"] == ["clap"], "only ines nods; the others clap on"
    refused = run.steps[-1]
    assert not refused["ok"] and "needs" in refused["error"]


def test_people_sit_on_a_chair_and_get_up_in_the_gallery(godot_script, scene_export):
    """sit moves the body (geogen.103.7): in the gallery each person steps back to the chair
    behind them, sits (the body's sit_down clip, the root on its path down to the seat), stays and
    stands up again at the chair; the next motion puts them back where they stood."""
    run = godot_script([
        step("hud.gallery", open=True),
        step("motion.status", node="walter"),
        step("hud.gallery", select="sit", play=True),
        step("wait", seconds=2.4),
        step("motion.status", node="walter"),
        step("world.node", name="walter_chair"),
        step("world.node", name="walter_table"),
        step("wait", seconds=4.6),
        step("motion.status", node="walter"),
        step("hud.gallery", select="clap", play=True),
        step("wait", seconds=0.1),
        step("motion.status", node="walter"),
        step("world.node", name="walter_chair"),
    ], "--scene=motion_gallery", f"--generated={scene_export('motion_gallery')}")
    before, seated, stood, clapping = run.all("motion.status")
    chair, table, chair_after = (e["result"] for e in run.steps if e["cmd"] == "world.node")
    assert chair["visible"] and not table["visible"], "the chair comes out, the table doesn't"
    assert seated["running"] == ["sit"] and seated["posture"] == "sit"
    assert seated["root"][1] < before["root"][1] - 0.2, "the body went down onto the seat"
    assert seated["root"][2] < before["root"][2] - 0.15, "and back over it"
    beats = [(e["event"], e["beat"]) for e in stood["events"] if e["event"] in ("beat", "end")]
    assert beats == [("beat", "arrive"), ("beat", "lower"), ("beat", "seated"), ("beat", "rise"), ("end", "rise")]
    assert stood["posture"] == "stand" and abs(stood["root"][1] - before["root"][1]) < 0.01, "up on the floor again"
    assert clapping["running"] == ["clap"] and all(abs(a - b) < 1e-3 for a, b in zip(clapping["root"], before["root"]))
    assert not chair_after["visible"]


def test_people_take_things_from_the_table_use_them_and_put_them_back(godot_script, scene_export):
    """In the gallery (geogen.103.7) a motion with things plays at the table in front of each
    person: drink takes the mug (a physics prop: out of the physics while held) by its handle up
    to the face and puts it back where it stood; take's demo takes the plate, carries it and sets
    it down where it stood (take, carry, set_down)."""
    run = godot_script([
        step("hud.gallery", open=True),
        step("world.node", name="tomas_plate"),
        step("world.bodies"),
        step("hud.gallery", select="drink", play=True),
        step("wait", seconds=3.0),
        step("motion.status", node="ines"),
        step("world.bodies"),
        step("wait", seconds=7.0),
        step("motion.status", node="ines"),
        step("world.bodies"),
        step("hud.gallery", select="take", performer="tomas", play=True),
        step("wait", seconds=2.5),
        step("motion.status", node="tomas"),
        step("wait", seconds=5.0),
        step("motion.status", node="tomas"),
        step("world.node", name="tomas_plate"),
        step("hud.gallery", select="clap", play=True),
        step("world.node", name="tomas_table"),
    ], "--scene=motion_gallery", f"--generated={scene_export('motion_gallery')}")
    mug = lambda bodies: next(b for b in bodies if b["node"] == "ines_mug")
    before, sipping, after = (mug(r) for r in run.all("world.bodies"))
    drinking, drunk, taking, put = run.all("motion.status")
    assert drinking["running"] == ["drink"] and drinking["things"]["ines_mug"]["held"]
    assert sipping["holder"] != "" and sipping["position"][1] > 1.2, "up at the face"
    head = drinking["bones"]["Head"]
    assert abs(sipping["position"][0] - head[0]) < 0.25 and abs(sipping["position"][2] - head[2]) < 0.25
    assert drunk["running"] == [] and not drunk["things"]["ines_mug"]["held"]
    assert after["holder"] == "" and all(abs(a - b) < 0.01 for a, b in zip(after["position"], before["position"])), \
        "back where it stood"
    assert taking["running"] == ["carry"] and taking["things"]["tomas_plate"]["held"]
    assert put["running"] == [] and not put["things"]["tomas_plate"]["held"]
    assert [e["motion"] for e in put["events"] if e["event"] == "start"][-3:] == ["take", "carry", "set_down"]
    plate_before, plate_after, table = run.all("world.node")
    assert not plate_before["visible"] and plate_after["visible"], "the tables come out for motions with things"
    assert all(abs(a - b) < 0.005 for a, b in zip(plate_after["position"], plate_before["position"])), \
        "the plate is back where it stood"
    assert not table["visible"], "and go away for one without"


def test_two_people_hug_through_the_control_api(godot_script, scene_export):
    """A two-person motion (geogen.103.10) in the runtime: ana and ben each run `hug` with the
    other as `partner` (a role naming a body binds that body's rig), open their arms and let go
    together (shared `sync` beats), each hand round the other's back while they hold on."""
    run = godot_script([
        step("time.pause"),
        step("motion.start", node="ana", motion="hug", roles={"partner": "ben"}),
        step("motion.start", node="ben", motion="hug", roles={"partner": "ana"}),
        step("time.resume"),
        step("wait", seconds=2.0),
        step("motion.status", node="ana"),
        step("motion.status", node="ben"),
        step("wait", seconds=2.6),
        step("motion.status", node="ana"),
        step("motion.status", node="ben"),
        step("motion.start", node="ana", motion="hug", roles={"partner": "ana"}),
    ], "--scene=motion_test", f"--generated={scene_export('motion_test')}", check=False)
    ana, ben, ana_after, ben_after = (e["result"] for e in run.steps if e["cmd"] == "motion.status")
    assert ana["running"] == ["hug"] and ben["running"] == ["hug"]
    for me, them in ((ana, ben), (ben, ana)):
        # Each hand is behind the other's back: past the middle of their body, seen from mine.
        mine, chest, theirs = (np.array(p) for p in (me["root"], them["bones"]["Hips"], them["root"]))
        ahead = (theirs - mine) / np.linalg.norm(theirs - mine)
        assert (np.array(me["bones"]["RightHand"]) - chest) @ ahead > 0.03, me["node"]
        assert (np.array(me["bones"]["LeftHand"]) - chest) @ ahead > 0.0, me["node"]
    beats = lambda s: [(e["t"], e["beat"]) for e in s["events"] if e["event"] == "beat"]
    assert [b for _, b in beats(ana_after)] == ["open", "embrace", "part", "back"] == [b for _, b in beats(ben_after)]
    # They go on from each shared beat in the same step.
    assert beats(ana_after) == beats(ben_after)
    assert ana_after["running"] == [] and ben_after["running"] == []
    assert not run.steps[-1]["ok"] and "not another body" in run.steps[-1]["error"]
