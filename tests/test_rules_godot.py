"""The Godot rules machine (runtime/godot/scripts/machine.gd) replays the conformance
traces the reference evaluator is held to (docs/rules.md, tests/test_rules.py)."""

import json
import subprocess
from pathlib import Path

import pytest

TRACES = sorted((Path(__file__).parent / "data" / "rules").glob("*.json"))


def _results(output: str) -> dict[str, list[str]]:
    """``trace: {...}`` lines of a run, as name -> problems."""
    found = {}
    for line in output.splitlines():
        if line.startswith("trace: "):
            result = json.loads(line.removeprefix("trace: "))
            found[result["name"]] = result["problems"]
    return found


@pytest.fixture(scope="module")
def replayed(run_godot) -> dict[str, list[str]]:
    """Every trace through one headless Godot, which loads no world for it."""
    return _results(run_godot(*(f"--trace={path}" for path in TRACES)))


def test_every_trace_is_replayed():
    assert {p.stem for p in TRACES} >= {"lever_opens_gate", "bindings", "actors_and_once", "watched_conditions",
                                       "time", "cascades", "quest_go_there", "quest_change_the_world",
                                       "quest_timed"}


@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_trace_conforms_in_godot(replayed, path):
    name = json.loads(path.read_text())["name"]
    assert name in replayed, f"no result for {name}: got {sorted(replayed)}"
    assert replayed[name] == []


def test_a_wrong_expectation_is_reported(run_godot, tmp_path):
    from conftest import GODOT_PROJECT, _godot_binary

    trace = json.loads(TRACES[0].read_text())
    step = next(s for s in trace["steps"] if "expect" in s and s["expect"].get("state"))
    key = next(iter(step["expect"]["state"]))
    step["expect"]["state"][key] = "nonsense"
    fired = next(s for s in trace["steps"] if "expect" in s and s["expect"].get("fired"))
    fired["expect"]["fired"] = fired["expect"]["fired"] + ["a_rule_that_never_ran"]
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(trace))
    # run_godot insists on exit code 0, and this run must fail
    result = subprocess.run([_godot_binary(), "--headless", "--path", str(GODOT_PROJECT), "--",
                             f"--trace={wrong}", f"--trace={TRACES[0]}"],
                            capture_output=True, text=True, timeout=240)
    assert "SCRIPT ERROR" not in result.stdout + result.stderr, result.stdout + result.stderr
    assert result.returncode == 1, result.stdout + result.stderr
    lines = [json.loads(line.removeprefix("trace: ")) for line in result.stdout.splitlines()
             if line.startswith("trace: ")]
    assert len(lines) == 2                              # one line per file, the sound one too
    assert any(key in problem and "nonsense" in problem for problem in lines[0]["problems"])
    assert any("a_rule_that_never_ran" in problem for problem in lines[0]["problems"])
    assert lines[1]["problems"] == []
