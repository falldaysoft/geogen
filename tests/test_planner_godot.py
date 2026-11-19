"""The Godot planner (runtime/godot/scripts/npc_planner.gd) replays the conformance traces the
reference planner is held to (geogen/planner.py, tests/test_planner.py)."""

import json
from pathlib import Path

import pytest
from helpers import assert_trace_conforms, replay_traces, replayed_fixture, trace_results

TRACES = sorted((Path(__file__).parent / "data" / "planner").glob("*.json"))


replayed = replayed_fixture(TRACES)


@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_trace_conforms_in_godot(replayed, path):
    assert_trace_conforms(replayed, path)


def test_a_wrong_expectation_is_reported(run_godot, tmp_path):
    trace = json.loads(next(p for p in TRACES if p.stem == "kitchen").read_text())
    trace["steps"][0]["expect"]["steps"] = ["kitchen/table/chair#0"]
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(trace))
    result = replay_traces(path, timeout=120)
    assert result.returncode == 1
    assert any("steps" in problem for problem in trace_results(result.stdout)["kitchen"])
