"""The reference planner (geogen/planner.py) against its conformance traces (geogen.101.3)."""
import json
from pathlib import Path

import pytest

from geogen.planner import plan, run_trace, urgency

TRACES = sorted((Path(__file__).parent / "data" / "planner").glob("*.json"))


@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_trace_conforms(path):
    assert run_trace(path) == []


def test_traces_cover_the_planner():
    assert {p.stem for p in TRACES} >= {"kitchen", "missing_ingredient", "one_stove", "states", "urgency", "depth"}


def test_traces_are_what_make_traces_writes(tmp_path):
    import subprocess
    import sys

    subprocess.run([sys.executable, str(TRACES[0].parent / "make_traces.py"), str(tmp_path)], check=True)
    for path in TRACES:
        assert (tmp_path / path.name).read_text() == path.read_text(), f"{path.name} is stale: run make_traces.py"


def test_a_wrong_expectation_is_reported():
    trace = json.loads(next(p for p in TRACES if p.stem == "kitchen").read_text())
    trace["steps"][0]["expect"]["cost"] = 1.0
    assert any("cost" in problem for problem in run_trace(trace))


def test_a_plan_is_the_same_whatever_order_the_world_is_listed_in():
    trace = json.loads(next(p for p in TRACES if p.stem == "one_stove").read_text())
    world = trace["world"]["affordances"]
    snapshot = {"affordances": world, "actor": trace["actors"]["ana"]}
    assert plan(snapshot) == plan({**snapshot, "affordances": world[::-1]})


def test_urgency_curve():
    assert urgency(0.75) == 0.25 and urgency(1.0) == 0.0 and urgency(-1) == 1.0
    curve = {"urgent": 0.3, "boost": 4}
    assert urgency(0.5, curve) == 0.5                          # above the threshold: as without
    assert urgency(0.1, curve) == pytest.approx(0.9 + 3 * 0.2)  # below it the shortfall counts four times
    assert urgency(0.3, curve) == pytest.approx(urgency(0.3))    # continuous at the threshold


def test_definitions_declare_need_curves_consequences_and_planning():
    """`npcs/householder.yaml`: needs with `urgent` and `at_zero`, and `planner:` (geogen.101.2, 101.4)."""
    from geogen.npc import PLANNER_DEFAULTS, _need, _planner, load_actions, load_definition

    home = load_definition(Path(__file__).parent.parent / "assets" / "npcs" / "householder.yaml")
    assert home["planner"] == {"depth": 5, "patience": 60.0, "nodes": int(PLANNER_DEFAULTS["nodes"])}
    assert home["needs"]["hunger"] == {"initial": 0.3, "decay": 0.0015, "urgent": 0.35, "boost": 4.0}
    assert home["needs"]["rest"]["at_zero"] == {"action": "collapse", "recover": 0.3}
    resident = load_definition(Path(__file__).parent.parent / "assets" / "npcs" / "resident.yaml")
    assert "planner" not in resident and resident["needs"]["rest"] == {"initial": 0.7, "decay": 0.006}
    actions = load_actions()
    assert _planner(True, "t") == {"planner": {"depth": 5, "patience": 60.0, "nodes": 2000}} and _planner(None, "t") == {}
    for bad, message in (({"boost": 2}, "goes with urgent"), ({"urgent": 2}, "urgent is a level"),
                         ({"at_zero": "fly"}, "unknown action"), ({"at_zero": {"action": "collapse", "recover": 0}}, "recover"),
                         ({"colour": 1}, "keys are")):
        with pytest.raises(ValueError, match=message):
            _need(bad, actions, "t")
    with pytest.raises(ValueError, match="true or keys"):
        _planner({"deep": 3}, "t")


def test_household_things_declare_the_links_of_a_meal():
    """What the planner chains through, as the assets export it (geogen.101.5)."""
    from geogen.layout.loader import LayoutLoader

    assets = Path(__file__).parent.parent / "assets"

    def links(name):
        found = []
        for a in LayoutLoader().load(assets / f"{name}.yaml").meta["affordances"]:
            found.append({k: a[k] for k in ("requires", "takes", "gives") if k in a})
            found.extend(a.get("uses", {}).values())
        return found

    assert links("fridge") == [{"gives": {"raw_food": 1}}]
    cook = {"requires": {"holding": ["raw_food"]}, "takes": ["raw_food"], "gives": {"meal": 1}}
    assert links("microwave") == [cook]
    assert links("kitchen_range") == [{}, {**cook, "duration": [12.0, 18.0], "tags": ["household"]}]      # the chef's, and the `meal` use
    assert links("chair") == [{}, {"duration": [10.0, 15.0], "requires": {"holding": ["meal"]}, "takes": ["meal"],
                                   "gives": {"dirty_plate": 1}, "advertises": {"hunger": 0.7},
                                   "tags": ["seat", "table", "household"], "action": "eat"}]      # eaten at the table (geogen.105)
    assert links("kitchen_sink") == [{"requires": {"holding": ["dirty_plate"]}, "takes": ["dirty_plate"]}]
    assert links("shower") == [{}]
