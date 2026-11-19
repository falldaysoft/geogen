"""The Godot rules machine (runtime/godot/scripts/machine.gd) replays the conformance
traces the reference evaluator is held to (docs/rules.md, tests/test_rules.py)."""

import json
from pathlib import Path

import pytest
from helpers import assert_trace_conforms, replay_traces, replayed_fixture

TRACES = sorted((Path(__file__).parent / "data" / "rules").glob("*.json"))


replayed = replayed_fixture(TRACES)


def test_every_trace_is_replayed():
    assert {p.stem for p in TRACES} >= {"lever_opens_gate", "bindings", "actors_and_once", "watched_conditions",
                                       "time", "cascades", "quest_go_there", "quest_change_the_world",
                                       "quest_timed"}


@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_trace_conforms_in_godot(replayed, path):
    assert_trace_conforms(replayed, path)


def test_a_wrong_expectation_is_reported(run_godot, tmp_path):
    trace = next(t for t in (json.loads(path.read_text()) for path in TRACES)       # one that expects state and firings
                 if any(s.get("expect", {}).get("state") for s in t["steps"])
                 and any(s.get("expect", {}).get("fired") for s in t["steps"]))
    step = next(s for s in trace["steps"] if "expect" in s and s["expect"].get("state"))
    key = next(iter(step["expect"]["state"]))
    step["expect"]["state"][key] = "nonsense"
    fired = next(s for s in trace["steps"] if "expect" in s and s["expect"].get("fired"))
    fired["expect"]["fired"] = fired["expect"]["fired"] + ["a_rule_that_never_ran"]
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(trace))
    result = replay_traces(wrong, TRACES[0])
    assert "SCRIPT ERROR" not in result.stdout + result.stderr, result.stdout + result.stderr
    assert result.returncode == 1, result.stdout + result.stderr
    lines = [json.loads(line.removeprefix("trace: ")) for line in result.stdout.splitlines()
             if line.startswith("trace: ")]
    assert len(lines) == 2                              # one line per file, the sound one too
    assert any(key in problem and "nonsense" in problem for problem in lines[0]["problems"])
    assert any("a_rule_that_never_ran" in problem for problem in lines[0]["problems"])
    assert lines[1]["problems"] == []
