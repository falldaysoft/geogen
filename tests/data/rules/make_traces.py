"""Writes the conformance traces beside this file (conditions are compiled from text, so the
trees are right). Run after editing a trace here: python tests/data/rules/make_traces.py

Each trace is replayed by geogen.rules.run_trace (tests/test_rules.py) and, in time, by the
Godot runtime: both must agree with what the steps expect. Semantics: docs/rules.md.
"""
import json, sys
from pathlib import Path
from geogen import conditions

PKG = "demo"

def c(text, **ids):
    """Compile with refs resolved by hand: thing 'x' -> 'demo:x'."""
    tree = conditions.compile_condition(text)
    for n in conditions.walk(tree):
        if n.get("get") == "ref":
            n.update(get="instance", id=f"{PKG}:{n.pop('ref')}")
        elif n.get("get") in ("world", "actor") and n["name"] not in conditions.ACTOR_BUILTINS:
            n["package"] = PKG
        elif "select" in n:
            n["over"] = [f"{PKG}:{x}" for x in ids[n.pop("select")]]
    conditions.validate(tree)
    return tree

def program(**parts):
    return {"format": "geogen-rules", "version": 1, **parts}

TOGGLE = {"off": {"next": "on"}, "on": {"next": "off"}}
traces = {}

traces["lever_opens_gate"] = {
    "about": "Wiring with no player: a lever's signal opens a gate; pulling it back closes it.",
    "program": program(
        interactions={
            "demo:lever#pull": {"initial": "up", "duration": 0.5, "states": {
                "up": {"next": "down", "emit": "released"}, "down": {"next": "up", "emit": "pulled"}}},
            "demo:gate#gate": {"initial": "closed", "duration": 2.0, "states": {
                "closed": {"emit": "shut"}, "open": {"emit": "opened"}}}},
        rules=[
            {"id": "open", "on": {"type": "signal", "subject": "demo:lever", "name": "pulled"},
             "do": [{"set": "demo:gate#gate", "to": "open"}]},
            {"id": "close", "on": {"type": "signal", "subject": "demo:lever", "name": "released"},
             "do": [{"set": "demo:gate#gate", "to": "closed"}]}]),
    "actors": {"npc:keeper": {"kind": "npc"}},
    "steps": [
        {"use": "demo:gate#gate", "used": False},                       # no 'next': only rules move it
        {"use": "demo:lever#pull", "actor": "npc:keeper", "used": True},
        {"expect": {"state": {"demo:lever#pull": "down", "demo:gate#gate": "closed"}, "signals": [], "fired": []}},
        {"advance": 0.6},                                                 # the lever arrives and signals
        {"expect": {"state": {"demo:gate#gate": "open"}, "signals": [["demo:lever", "pulled"]], "fired": ["open"]}},
        {"advance": 2.1},
        {"expect": {"signals": [["demo:gate", "opened"]], "fired": []}},
        {"use": "demo:lever#pull"},
        {"advance": 0.2},
        {"use": "demo:lever#pull"},                                       # reversed before arriving: no 'released'
        {"advance": 1.0},
        {"expect": {"state": {"demo:lever#pull": "down", "demo:gate#gate": "open"},
                    "signals": [["demo:lever", "pulled"]], "fired": ["open"]}},
        {"use": "demo:lever#pull"},
        {"advance": 3.0},
        {"expect": {"state": {"demo:gate#gate": "closed"},
                    "signals": [["demo:lever", "released"], ["demo:gate", "shut"]], "fired": ["close"], "errors": 0}},
    ]}

traces["bindings"] = {
    "about": "Standing relations: a lamp lit while a plate is occupied and the power is on; a binding reading a binding.",
    "program": program(
        variables={
            "demo:plate#occupants": {"type": "number", "default": 0},
            "world:demo.power": {"type": "bool", "default": True},
            "demo:lamp#lit": {"type": "bool", "default": False},
            "world:demo.dark": {"type": "bool", "default": True}},
        bindings=[
            {"set": "world:demo.dark", "to": c("not lamp.lit")},          # listed first, evaluated second
            {"set": "demo:lamp#lit", "to": c("plate.occupants > 0 and world.power")}],
        rules=[
            {"id": "step_on", "on": {"type": "enter", "subject": "demo:plate"},
             "do": [{"set": "demo:plate#occupants", "to": c("plate.occupants + 1")}]},
            {"id": "step_off", "on": {"type": "leave", "subject": "demo:plate"},
             "do": [{"set": "demo:plate#occupants", "to": c("max(plate.occupants - 1, 0)")}]},
            {"id": "lamp_came_on", "on": {"type": "changed", "key": "demo:lamp#lit", "to": True},
             "do": [{"emit": "lit", "subject": "demo:lamp"}]}]),
    "actors": {"player": {"kind": "player"}, "npc:a": {"kind": "npc"}},
    "steps": [
        {"expect": {"state": {"demo:lamp#lit": False, "world:demo.dark": True}}},
        {"event": {"type": "enter", "subject": "demo:plate", "actor": "npc:a"}},
        {"expect": {"state": {"demo:lamp#lit": True, "world:demo.dark": False},
                    "signals": [["demo:lamp", "lit"]], "fired": ["step_on", "lamp_came_on"]}},
        {"event": {"type": "enter", "subject": "demo:plate", "actor": "player"}},
        {"event": {"type": "leave", "subject": "demo:plate", "actor": "npc:a"}},
        {"expect": {"state": {"demo:plate#occupants": 1, "demo:lamp#lit": True}, "signals": []}},
        {"set": "world:demo.power", "to": False},
        {"expect": {"state": {"demo:lamp#lit": False, "world:demo.dark": True}}},
        {"save_load": True},                                              # bindings are recomputed, not saved
        {"expect": {"state": {"demo:lamp#lit": False, "demo:plate#occupants": 1}, "signals": [], "fired": []}},
        {"set": "world:demo.power", "to": True},
        {"expect": {"state": {"demo:lamp#lit": True}, "signals": [["demo:lamp", "lit"]], "errors": 0}},
    ]}

traces["actors_and_once"] = {
    "about": "Actor state is one value per actor; 'once' is per actor; events turn into lasting state.",
    "program": program(
        variables={
            "actor:demo.seen_gallery": {"type": "bool", "default": False},
            "actor:demo.visits": {"type": "number", "default": 0},
            "world:demo.visitors": {"type": "number", "default": 0}},
        rules=[
            {"id": "discover", "on": {"type": "enter", "subject": "demo:gallery"}, "once": True,
             "do": [{"set": "actor:demo.seen_gallery", "to": True},
                    {"set": "world:demo.visitors", "to": c("world.visitors + 1")}]},
            {"id": "count", "on": {"type": "enter", "subject": "demo:gallery"},
             "do": [{"set": "actor:demo.visits", "to": c("actor.visits + 1")}]},
            {"id": "players_only", "on": {"type": "enter", "subject": "demo:gallery", "actor": "player"},
             "when": c("actor.visits >= 2"), "do": [{"emit": "regular", "subject": "demo:gallery"}]}]),
    "actors": {"player": {"kind": "player"}, "npc:a": {"kind": "npc"}},
    "steps": [
        {"event": {"type": "enter", "subject": "demo:gallery", "actor": "player"}},
        {"event": {"type": "enter", "subject": "demo:gallery", "actor": "npc:a"}},
        {"event": {"type": "enter", "subject": "demo:gallery", "actor": "npc:a"}},
        {"expect": {"state": {"world:demo.visitors": 2},
                    "actors": {"player": {"actor:demo.seen_gallery": True, "actor:demo.visits": 1},
                               "npc:a": {"actor:demo.seen_gallery": True, "actor:demo.visits": 2}},
                    "fired": ["discover", "count", "discover", "count", "count"], "signals": []}},
        {"save_load": True},
        {"event": {"type": "enter", "subject": "demo:gallery", "actor": "player"}},
        {"expect": {"state": {"world:demo.visitors": 2}, "actors": {"player": {"actor:demo.visits": 2}},
                    "fired": ["count", "players_only"], "signals": [["demo:gallery", "regular"]]}},
        {"event": {"type": "enter", "subject": "demo:elsewhere", "actor": "player"}},
        {"event": {"type": "enter", "subject": "demo:gallery"}},        # nobody: actor effects are errors
        {"expect": {"state": {"world:demo.visitors": 3}, "fired": ["discover", "count"], "errors": 2}},
    ]}

lights = ["l1", "l2", "l3"]
traces["watched_conditions"] = {
    "about": "Rules with no event fire when their condition becomes true: every light off (what a quest objective compiles to), per actor when the condition reads actor state.",
    "program": program(
        variables={
            "world:demo.dark_count": {"type": "number", "default": 0},
            "actor:demo.asked": {"type": "bool", "default": False},
            "actor:demo.done": {"type": "bool", "default": False}},
        interactions={f"demo:{l}#switch": {"initial": "on", "duration": 0.1, "states": TOGGLE} for l in lights},
        rules=[
            {"id": "all_dark", "when": c('all("L", it.switch == "off")', L=lights),
             "do": [{"set": "world:demo.dark_count", "to": c("world.dark_count + 1")}]},
            {"id": "objective", "when": c('actor.asked and count("L", it.switch == "off") == 3', L=lights),
             "do": [{"set": "actor:demo.done", "to": True}]}]),
    "actors": {"player": {"kind": "player"}, "npc:a": {"kind": "npc"}},
    "steps": [
        {"set": "actor:demo.asked", "to": True, "actor": "npc:a"},
        {"use": "demo:l1#switch", "actor": "npc:a"},
        {"use": "demo:l2#switch", "actor": "npc:a"},
        {"expect": {"state": {"world:demo.dark_count": 0}, "fired": []}},
        {"use": "demo:l3#switch", "actor": "player"},
        {"expect": {"state": {"world:demo.dark_count": 1},
                    "actors": {"npc:a": {"actor:demo.done": True}, "player": {"actor:demo.done": False}},
                    "fired": ["all_dark", "objective"]}},
        {"use": "demo:l2#switch"},
        {"use": "demo:l2#switch"},                                        # true again: fires again
        {"expect": {"state": {"world:demo.dark_count": 2}, "fired": ["all_dark", "objective"]}},
        {"save_load": True},
        {"set": "actor:demo.asked", "to": True, "actor": "player"},     # already dark: true for the player now
        {"expect": {"state": {"world:demo.dark_count": 2}, "actors": {"player": {"actor:demo.done": True}},
                    "fired": ["objective"], "errors": 0}},
    ]}

traces["time"] = {
    "about": "Delayed rules, a door that closes itself, locks, the clock, and a save in the middle of it all.",
    "program": program(
        variables={
            "world:demo.alarm": {"type": "bool", "default": False},
            "world:demo.open_hours": {"type": "bool", "default": False},
            "demo:lamp#lit": {"type": "bool", "default": False}},
        interactions={
            "demo:door#swing": {"initial": "closed", "duration": 1.0, "locked": True, "states": {
                "closed": {"next": "open"}, "open": {"next": "closed", "then": "closed", "after": 6, "emit": "opened"}}},
            "demo:bell#ring": {"initial": "rest", "duration": 0.2, "states": {
                "rest": {"next": "struck"}, "struck": {"then": "rest", "emit": "rang"}}}},
        bindings=[{"set": "world:demo.open_hours", "to": c('clock.time >= "09:00" and clock.time < "17:30"')}],
        rules=[
            {"id": "unlock", "on": {"type": "changed", "key": "world:demo.open_hours"},
             "do": [{"set": "demo:door#swing_locked", "to": c("not world.open_hours")}]},
            {"id": "alarm", "on": {"type": "signal", "subject": "demo:door", "name": "opened"}, "after": 3,
             "do": [{"set": "world:demo.alarm", "to": True}]},
            {"id": "dusk", "on": {"type": "clock", "name": "night"}, "do": [{"set": "demo:lamp#lit", "to": True}]},
            {"id": "dawn", "on": {"type": "clock", "name": "day"}, "do": [{"set": "demo:lamp#lit", "to": False}]}]),
    "actors": {"player": {"kind": "player"}},
    "steps": [
        {"clock": {"time": "08:00"}},
        {"use": "demo:door#swing", "actor": "player", "used": False},
        {"expect": {"signals": [["demo:door", "locked"]], "state": {"demo:door#swing": "closed"}}},
        {"clock": {"time": "09:00"}},
        {"expect": {"state": {"demo:door#swing_locked": False}, "fired": ["unlock"]}},
        {"use": "demo:door#swing", "actor": "player", "used": True},
        {"advance": 1.5},
        {"expect": {"signals": [["demo:door", "opened"]], "fired": ["alarm"], "state": {"world:demo.alarm": False}}},
        {"save_load": True},                                              # timers survive a save
        {"advance": 3.0},
        {"expect": {"state": {"world:demo.alarm": True, "demo:door#swing": "open"}}},
        {"advance": 3.0},                                                 # 6 s after arriving it shuts itself
        {"expect": {"state": {"demo:door#swing": "closed"}, "signals": [], "fired": []}},
        {"use": "demo:bell#ring"},
        {"advance": 1.0},                                                 # struck -> rest at once (momentary)
        {"expect": {"state": {"demo:bell#ring": "rest"}, "signals": [["demo:bell", "rang"]]}},
        {"clock": {"time": "21:00", "night": True}},
        {"expect": {"state": {"demo:lamp#lit": True, "demo:door#swing_locked": True}, "fired": ["dusk", "unlock"]}},      # the event's rules, then what bindings changed
        {"clock": {"time": "07:00", "night": False, "day": 1}},
        {"expect": {"state": {"demo:lamp#lit": False}, "fired": ["dawn"], "errors": 0}},
    ]}

traces["cascades"] = {
    "about": "Rules trigger rules in a fixed order; a loop is cut off after 16 events and reported.",
    "program": program(
        variables={k: {"type": "number", "default": 0} for k in
                   ("world:demo.a", "world:demo.b", "world:demo.c", "world:demo.ping", "world:demo.order")},
        rules=[
            {"id": "a_to_b", "on": {"type": "changed", "key": "world:demo.a"},
             "do": [{"set": "world:demo.b", "to": c("world.a * 2")}, {"set": "world:demo.order", "to": c("world.order * 10 + 1")}]},
            {"id": "a_to_c", "on": {"type": "changed", "key": "world:demo.a"},
             "do": [{"set": "world:demo.c", "to": c("world.b + 1")}, {"set": "world:demo.order", "to": c("world.order * 10 + 2")}]},
            {"id": "b_seen", "on": {"type": "changed", "key": "world:demo.b"},
             "do": [{"set": "world:demo.order", "to": c("world.order * 10 + 3")}]},
            {"id": "loop", "on": {"type": "changed", "key": "world:demo.ping"},
             "do": [{"set": "world:demo.ping", "to": c("world.ping + 1")}]}]),
    "steps": [
        {"set": "world:demo.a", "to": 5},
        # Both rules for the event run, in program order, before the events they caused: c sees the new b.
        {"expect": {"state": {"world:demo.b": 10, "world:demo.c": 11, "world:demo.order": 123},
                    "fired": ["a_to_b", "a_to_c", "b_seen"], "errors": 0}},
        {"set": "world:demo.ping", "to": 1},
        {"expect": {"state": {"world:demo.ping": 17}, "errors": 1}},
    ]}

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
for name, trace in traces.items():
    (out / f"{name}.json").write_text(json.dumps({"name": name, **trace}, indent=1) + "\n")
print(len(traces))
