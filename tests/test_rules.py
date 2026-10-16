"""The reference evaluator for state, events, rules and bindings (docs/rules.md)."""

import copy
import json
from pathlib import Path

import pytest

from geogen.rules import Machine, RuleError, binding_order, check_program, run_trace

TRACES = sorted((Path(__file__).parent / "data" / "rules").glob("*.json"))


@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_trace_conforms(path):
    assert run_trace(path) == []


def test_traces_cover_the_semantics():
    assert {p.stem for p in TRACES} >= {"lever_opens_gate", "bindings", "actors_and_once", "watched_conditions",
                                       "time", "cascades", "quest_go_there", "quest_change_the_world",
                                       "quest_timed"}


def test_a_wrong_expectation_is_reported():
    trace = next(t for t in (json.loads(path.read_text()) for path in TRACES)       # one that expects state and firings
                 if any(s.get("expect", {}).get("state") for s in t["steps"])
                 and any(s.get("expect", {}).get("fired") for s in t["steps"]))
    step = next(s for s in trace["steps"] if "expect" in s and s["expect"].get("state"))
    key = next(iter(step["expect"]["state"]))
    step["expect"]["state"][key] = "nonsense"
    assert any(key in problem for problem in run_trace(trace))


def program(**parts):
    return {"format": "geogen-rules", "version": 1, **parts}


def get(key, package="demo"):
    scope, _, name = key.partition(".")
    return {"get": scope, "package": package, "name": name}


NUMBER = {"type": "number", "default": 0}
FLAG = {"type": "bool", "default": False}


@pytest.mark.parametrize("parts, message", [
    (dict(variables={"world:demo.a": {"type": "colour", "default": 1}}), "type must be one of"),
    (dict(variables={"world:demo.a": {"type": "bool", "default": 3}}), "isn't a bool"),
    (dict(variables={"world:demo.a": {"type": "choice", "choices": ["x"], "default": "y"}}), "isn't a choice"),
    (dict(interactions={"demo:d#swing": {"initial": "ajar", "states": {"closed": {}}}}), "initial state"),
    (dict(interactions={"demo:d#swing": {"initial": "closed", "states": {"closed": {"next": "open"}}}}), "isn't one of its states"),
    (dict(variables={"world:demo.a": FLAG, "world:demo.b": FLAG},
          bindings=[{"set": "world:demo.a", "to": get("world.b")}, {"set": "world:demo.b", "to": get("world.a")}]),
     "bindings form a cycle"),
    (dict(variables={"world:demo.a": FLAG, "actor:demo.x": FLAG},
          bindings=[{"set": "world:demo.a", "to": get("actor.x")}]), "can't involve actor state"),
    (dict(interactions={"demo:d#swing": {"initial": "closed", "states": {"closed": {}}}},
          bindings=[{"set": "demo:d#swing", "to": "closed"}]), "can't be bound"),
    (dict(variables={"world:demo.a": FLAG}, bindings=[{"set": "world:demo.a", "to": True}],
          rules=[{"id": "r", "on": {"type": "start"}, "do": [{"set": "world:demo.a", "to": False}]}]), "which is bound"),
    (dict(rules=[{"id": "r", "on": {"type": "start"}, "do": [{"set": "world:demo.nope", "to": 1}]}]), "isn't declared"),
    (dict(variables={"world:demo.a": FLAG},
          rules=[{"id": "r", "on": {"type": "start"}, "when": get("world.nope"), "do": [{"set": "world:demo.a", "to": True}]}]),
     "reads 'world:demo.nope'"),
    (dict(variables={"actor:demo.x": FLAG},
          rules=[{"id": "r", "on": {"type": "clock", "name": "night"}, "do": [{"set": "actor:demo.x", "to": True}]}]),
     "says which actor"),
    (dict(variables={"actor:demo.x": FLAG, "world:demo.a": FLAG},
          rules=[{"id": "r", "when": get("world.a"), "do": [{"set": "actor:demo.x", "to": True}]}]), "says which actor"),
    (dict(variables={"world:demo.a": FLAG},
          rules=[{"id": "r", "on": {"type": "explode"}, "do": [{"set": "world:demo.a", "to": True}]}]), "unknown event type"),
    (dict(rules=[{"id": "r", "on": {"type": "start"}, "do": []}]), "does nothing"),
    (dict(rules=[{"id": "r", "do": [{"emit": "x"}]}]), "needs 'on'"),
    (dict(rules=[{"id": "r", "on": {"type": "start"}, "do": [{"explode": 1}]}]), "unknown effect"),
    (dict(interactions={"demo:d#swing": {"initial": "closed", "states": {"closed": {}}}},
          rules=[{"id": "r", "on": {"type": "start"}, "do": [{"set": "demo:d#swing", "to": "ajar"}]}]), "has no state 'ajar'"),
    (dict(variables={"world:demo.a": FLAG},
          rules=[{"id": "r", "on": {"type": "start"}, "when": {"get": "ref", "ref": "door", "name": "swing"},
                  "do": [{"set": "world:demo.a", "to": True}]}]), "unresolved"),
])
def test_check_program_reports(parts, message):
    problems = check_program(program(**parts))
    assert any(message in p for p in problems), problems
    with pytest.raises(RuleError):
        Machine(program(**parts))


def test_sound_program_has_no_problems_and_bindings_are_ordered():
    p = program(variables={"world:demo.a": NUMBER, "world:demo.b": NUMBER, "world:demo.c": NUMBER},
                bindings=[{"set": "world:demo.c", "to": get("world.b")},
                          {"set": "world:demo.b", "to": {"op": "+", "args": [get("world.a"), 1]}}])
    assert check_program(p) == []
    assert [b["set"] for b in binding_order(p)] == ["world:demo.b", "world:demo.c"]
    machine = Machine(p)
    machine.set("world:demo.a", 4)
    assert machine.get("world:demo.c") == 5


def test_values_are_typed():
    machine = Machine(program(variables={
        "world:demo.n": NUMBER, "world:demo.mode": {"type": "choice", "choices": ["a", "b"], "default": "a"},
        "actor:demo.flag": FLAG}), {"player": {"kind": "player"}})
    with pytest.raises(RuleError, match="is a number"):
        machine.set("world:demo.n", "three")
    with pytest.raises(RuleError, match=r"choice \(a, b\)"):
        machine.set("world:demo.mode", "c")
    with pytest.raises(RuleError, match="no actor"):
        machine.set("actor:demo.flag", True)
    with pytest.raises(RuleError, match="whose"):
        machine.get("actor:demo.flag")
    with pytest.raises(RuleError, match="isn't declared"):
        machine.get("world:demo.nope")
    machine.set("actor:demo.flag", 1, "player")           # a bool takes truthiness
    assert machine.get("actor:demo.flag", "player") is True


def test_save_keeps_world_and_actor_state_apart_and_only_what_changed():
    trace = json.loads((TRACES[0].parent / "actors_and_once.json").read_text())
    machine = Machine(trace["program"], trace["actors"])
    machine.post({"type": "enter", "subject": "demo:gallery", "actor": "player"})
    saved = machine.save()
    assert saved["format"] == "geogen-save" and json.loads(json.dumps(saved)) == saved
    assert saved["world"] == {"world:demo.visitors": 1}
    assert saved["actors"]["player"] == {"once:discover": True, "actor:demo.seen_gallery": True, "actor:demo.visits": 1}
    assert saved["actors"]["npc:a"] == {}
    other = Machine(copy.deepcopy(trace["program"]), trace["actors"])
    other.load(saved)
    assert other.save() == saved
    with pytest.raises(RuleError, match="not a geogen-save"):
        other.load({"format": "something"})


def test_watches_that_trigger_each_other_stop():
    flip = {"op": "not", "args": [get("world.a")]}
    machine = Machine(program(variables={"world:demo.a": FLAG}, rules=[
        {"id": "up", "when": flip, "do": [{"set": "world:demo.a", "to": True}]},
        {"id": "down", "when": get("world.a"), "do": [{"set": "world:demo.a", "to": False}]}]))
    assert any("kept triggering" in e for e in machine.errors)
