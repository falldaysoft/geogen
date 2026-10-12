"""The save format (docs/rules.md "Saving", docs/schema/geogen-save.v1.schema.json): what
geogen.rules.Machine.save writes and takes up again."""

import json
from pathlib import Path

import pytest

from geogen import rules
from geogen.rules import Machine, RuleError

SCHEMA = Path(__file__).parent.parent / "docs/schema/geogen-save.v1.schema.json"
TRACES = sorted((Path(__file__).parent / "data" / "rules").glob("*.json"))

DOOR = "demo:door#swing"
PROGRAM = {
    "format": "geogen-rules", "version": 1,
    "variables": {"demo:lamp#lit": {"type": "bool", "default": False},
                  "demo:plate#weight": {"type": "number", "default": 0},
                  "world:demo.power": {"type": "bool", "default": True},
                  "actor:demo.visits": {"type": "number", "default": 0}},
    "interactions": {DOOR: {"initial": "closed", "duration": 1.0, "states": {
        "closed": {"next": "open"}, "open": {"next": "closed", "then": "closed", "after": 5}}}},
    "bindings": [{"set": "demo:lamp#lit", "to": {"get": "world", "package": "demo", "name": "power"}}],
    "rules": [
        {"id": "visit", "on": {"type": "enter", "subject": "demo:hall"}, "once": True,
         "do": [{"set": "actor:demo.visits", "to": 1}]},
        {"id": "later", "on": {"type": "enter", "subject": "demo:hall", "actor": "player"}, "after": 30,
         "do": [{"set": "world:demo.power", "to": False}]}],
}
ACTORS = {"player": {"kind": "player"}, "demo:keeper": {"kind": "npc"}}


def played():
    machine = Machine(PROGRAM, ACTORS)
    machine.set("demo:plate#weight", 4)
    machine.use(DOOR, "player")
    for actor in ACTORS:
        machine.post({"type": "enter", "subject": "demo:hall", "actor": actor})
    machine.advance(0.25)
    assert not machine.errors
    return machine


def validate(save):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(save, json.loads(SCHEMA.read_text()))


def test_sections_are_flat_and_keys_name_their_owner():
    saved = played().save()
    assert set(saved) == {"format", "version", "now", "clock", "world", "actors", "actor_info", "timers"}
    assert (saved["format"], saved["version"], saved["now"]) == ("geogen-save", 1, 0.25)
    # an interaction's state is instance state like a declared variable's; the bound lamp isn't stored
    assert saved["world"] == {"demo:plate#weight": 4, DOOR: "open"}
    assert saved["actors"] == {"player": {"actor:demo.visits": 1, "once:visit": True},
                               "demo:keeper": {"actor:demo.visits": 1, "once:visit": True}}
    assert [(t["kind"], t["at"]) for t in saved["timers"]] == [("arrive", 1.0), ("rule", 30.0)]
    assert json.loads(json.dumps(saved)) == saved
    validate(saved)


def test_only_what_differs_from_the_defaults_is_stored():
    machine = played()
    machine.set("demo:plate#weight", 0)
    machine.set("actor:demo.visits", 0, "demo:keeper")
    machine.advance(8)                                  # the door arrives, dwells and shuts itself
    saved = machine.save()
    assert saved["world"] == {} and saved["actors"]["demo:keeper"] == {"once:visit": True}
    assert [t["kind"] for t in saved["timers"]] == ["rule"]
    validate(saved)


def test_a_save_round_trips_with_its_timers():
    machine = played()
    saved = json.loads(json.dumps(machine.save()))
    other = Machine(PROGRAM, ACTORS)
    other.load(saved)
    assert other.save() == saved and other.fired == [] and other.signals == []
    assert other.get(DOOR) == "open" and other.get("demo:lamp#lit") is True          # the binding, recomputed
    assert other.get("actor:demo.visits", "demo:keeper") == 1
    for m in (machine, other):
        m.advance(40)
    assert other.save() == machine.save()
    assert other.get("world:demo.power") is False and other.get("demo:lamp#lit") is False and other.get(DOOR) == "closed"


def test_a_save_may_leave_sections_out():
    saved = played().save()
    del saved["actor_info"]
    validate(saved)
    del saved["timers"]                                 # not a save anyone writes, but it loads
    other = Machine(PROGRAM, ACTORS)
    other.load(saved)
    assert other.actor_info["player"]["kind"] == "player" and other.actor_info["demo:keeper"]["kind"] == "npc"
    assert other.timers == [] and other.get("actor:demo.visits", "player") == 1
    with pytest.raises(RuleError, match="not a geogen-save"):
        other.load({"demo:door#swing": {"state": "open"}})          # a flat save from before the format
    with pytest.raises(RuleError, match="version 2"):
        other.load({**saved, "version": 2})


@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_saves_made_by_the_traces_match_the_schema(path, monkeypatch):
    saves = []
    save = Machine.save
    monkeypatch.setattr(Machine, "save", lambda self: saves.append(save(self)) or saves[-1])
    assert rules.run_trace(path) == []
    for saved in saves:
        validate(saved)


def test_the_schema_rejects_what_isnt_a_save():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    good = played().save()
    for bad in ({**good, "world": {"actor:demo.visits": 1}},                 # actor state in the world section
                {**good, "actors": {"player": {"world:demo.power": False}}},
                {**good, "world": {DOOR: {"state": "open", "locked": False}}},   # the old nested snapshot
                {**good, "timers": [{"at": 1.0, "seq": 1, "kind": "rule"}]},
                {k: v for k, v in good.items() if k != "world"},
                {DOOR: {"state": "open"}}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
