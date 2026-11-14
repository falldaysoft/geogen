"""The Godot planner (runtime/godot/scripts/npc_planner.gd) replays the conformance traces the
reference planner is held to (geogen/planner.py, tests/test_planner.py)."""

import json
import subprocess
from pathlib import Path

import pytest

TRACES = sorted((Path(__file__).parent / "data" / "planner").glob("*.json"))


def _results(output: str) -> dict[str, list[str]]:
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


@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_trace_conforms_in_godot(replayed, path):
    name = json.loads(path.read_text())["name"]
    assert name in replayed, f"no result for {name}: got {sorted(replayed)}"
    assert replayed[name] == []


def test_a_wrong_expectation_is_reported(run_godot, tmp_path):
    from conftest import GODOT_PROJECT, _godot_binary

    trace = json.loads(next(p for p in TRACES if p.stem == "kitchen").read_text())
    trace["steps"][0]["expect"]["steps"] = ["kitchen/table/chair#0"]
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(trace))
    result = subprocess.run([_godot_binary(), "--headless", "--path", str(GODOT_PROJECT), "--", f"--trace={path}"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 1
    assert any("steps" in problem for problem in _results(result.stdout)["kitchen"])
