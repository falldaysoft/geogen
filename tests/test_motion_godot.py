"""Motions in the Godot runtime (geogen.103.12): runtime/godot/scripts/motion_rig.gd replays the
conformance traces the reference rig wrote (tests/data/motion, geogen.motion_review.motion_trace),
and motion_player.gd runs motion files on the bodies of an export through the control API."""

import json
import subprocess
from pathlib import Path

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
