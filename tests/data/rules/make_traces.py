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
TOGGLE_SHUT = {"shut": {"next": "open"}, "open": {"next": "shut"}}
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
        {"save_load": {"without": ["actor_info"]}},                       # the host still knows who its actors are
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

traces["actor_id"] = {
    "about": "actor.id is the actor's stable ID: a watch that gives one particular NPC something (an NPC's own quest).",
    "program": program(
        variables={"actor:demo.chosen": {"type": "bool", "default": False}},
        rules=[{"id": "give_b", "when": c('actor.id == "npc:b"'), "do": [{"set": "actor:demo.chosen", "to": True}]}]),
    "actors": {"player": {"kind": "player"}, "npc:a": {"kind": "npc"}, "npc:b": {"kind": "npc"}},
    "steps": [
        {"advance": 0.1},
        {"expect": {"actors": {"npc:b": {"actor:demo.chosen": True}, "npc:a": {"actor:demo.chosen": False},
                               "player": {"actor:demo.chosen": False}}, "errors": 0}},
    ]}

held = {"get": "actor", "package": "demo", "name": "item_parcel"}
traces["items"] = {
    "about": "Carried items are actor state counted per actor (geogen.18): give adds, take never goes below none, has() reads it.",
    "program": program(
        variables={"actor:demo.item_parcel": {"type": "number", "default": 0},
                   "actor:demo.delivered": {"type": "bool", "default": False}},
        rules=[{"id": "give", "on": {"type": "signal", "subject": "demo:shelf", "name": "taken"},
                "do": [{"set": "actor:demo.item_parcel", "to": {"op": "+", "args": [held, 1]}}]},
               {"id": "take", "on": {"type": "signal", "subject": "demo:guest", "name": "greeted"},
                "when": c('has("parcel")'),
                "do": [{"set": "actor:demo.item_parcel", "to": {"op": "max", "args": [{"op": "-", "args": [held, 2]}, 0]}},
                       {"set": "actor:demo.delivered", "to": True}]}]),
    "actors": {"player": {"kind": "player"}, "npc:a": {"kind": "npc"}},
    "steps": [
        {"event": {"type": "signal", "subject": "demo:guest", "name": "greeted", "actor": "player"}},
        {"expect": {"actors": {"player": {"actor:demo.delivered": False}}, "fired": []}},
        {"event": {"type": "signal", "subject": "demo:shelf", "name": "taken", "actor": "player"}},
        {"expect": {"actors": {"player": {"actor:demo.item_parcel": 1}, "npc:a": {"actor:demo.item_parcel": 0}},
                    "fired": ["give"]}},
        {"event": {"type": "signal", "subject": "demo:guest", "name": "greeted", "actor": "player"}},
        {"expect": {"actors": {"player": {"actor:demo.item_parcel": 0, "actor:demo.delivered": True}},
                    "fired": ["take"], "errors": 0}},
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

traces["when"] = {
    "about": "An interaction's `when`: nobody can use the hatch until the power is on, and only someone "
             "carrying the key may use the vault; rules drive both whatever their conditions say.",
    "program": program(
        variables={"world:demo.power": {"type": "bool", "default": False},
                   "actor:demo.has_key": {"type": "bool", "default": False}},
        interactions={
            "demo:hatch#lid": {"initial": "shut", "duration": 0.5, "when": c("world.power"), "states": TOGGLE_SHUT},
            "demo:vault#door": {"initial": "shut", "duration": 0.5, "when": c("actor.has_key"), "states": TOGGLE_SHUT}},
        rules=[{"id": "blackout_opens", "on": {"type": "changed", "key": "world:demo.power", "to": False},
                "do": [{"set": "demo:hatch#lid", "to": "open"}]}]),
    "actors": {"player": {"kind": "player"}, "npc:guard": {"kind": "npc"}},
    "steps": [
        {"use": "demo:hatch#lid", "actor": "player", "used": False},       # no power: refused, no event
        {"expect": {"state": {"demo:hatch#lid": "shut"}, "signals": [], "fired": []}},
        {"set": "world:demo.power", "to": True},
        {"use": "demo:hatch#lid", "actor": "player", "used": True},
        {"advance": 0.6},
        {"expect": {"state": {"demo:hatch#lid": "open"}}},
        {"use": "demo:hatch#lid", "actor": "player", "used": True},
        {"advance": 0.6},
        {"set": "world:demo.power", "to": False},                          # a rule opens it anyway
        {"expect": {"state": {"demo:hatch#lid": "open"}, "fired": ["blackout_opens"]}},
        {"set": "actor:demo.has_key", "to": True, "actor": "npc:guard"},
        {"use": "demo:vault#door", "actor": "player", "used": False},      # actor state: per actor
        {"use": "demo:vault#door", "actor": "npc:guard", "used": True},
        {"advance": 0.6},
        {"expect": {"state": {"demo:vault#door": "open"}, "errors": 0}},
    ]}

traces["feedback"] = {
    "about": "Feedback (notify, play) tells a rule's actor something; it is neither state nor an event, "
             "and nothing reacts to it. A use an interaction's `when` stops says its `refused` text, with the refused sound; a "
             "locked one says nothing (rules hear `locked`). A delayed rule tells its actor when it runs.",
    "program": program(
        variables={"world:demo.power": {"type": "bool", "default": False}},
        interactions={
            "demo:jar#lid": {"initial": "shut", "duration": 0.5, "states": TOGGLE_SHUT},
            "demo:jar#spread": {"initial": "ready", "duration": 0.5, "when": c('jar.lid == "open"'),
                                "refused": "The jar is closed", "states": {"ready": {"next": "ready"}}},
            "demo:hatch#lid": {"initial": "shut", "duration": 0.5, "when": c("world.power"), "states": TOGGLE_SHUT},
            "demo:safe#door": {"initial": "shut", "duration": 0.5, "locked": True, "states": TOGGLE_SHUT}},
        rules=[{"id": "opened", "on": {"type": "use", "subject": "demo:jar", "name": "lid"},
                "do": [{"notify": "You open the jar"}, {"play": "click"}]},
               {"id": "locked", "on": {"type": "signal", "subject": "demo:safe", "name": "locked"},
                "do": [{"notify": "It won't budge"}, {"play": "refused"}]},
               {"id": "powered", "on": {"type": "changed", "key": "world:demo.power", "to": True}, "after": 1,
                "do": [{"notify": "The lights hum"}]}]),
    "actors": {"player": {"kind": "player"}, "npc:cook": {"kind": "npc"}},
    "steps": [
        {"use": "demo:jar#spread", "actor": "player", "used": False},
        {"expect": {"feedback": [["notify", "The jar is closed", "player"], ["play", "refused", "player"]],
                    "signals": [], "fired": []}},
        {"use": "demo:hatch#lid", "actor": "npc:cook", "used": False},      # no refused text: nothing said
        {"use": "demo:jar#lid", "actor": "npc:cook", "used": True},
        {"expect": {"feedback": [["notify", "You open the jar", "npc:cook"], ["play", "click", "npc:cook"]]}},
        {"advance": 0.6},
        {"use": "demo:jar#spread", "actor": "player", "used": True},
        {"use": "demo:safe#door", "actor": "player", "used": False},
        {"expect": {"feedback": [["notify", "It won't budge", "player"], ["play", "refused", "player"]],
                    "signals": [["demo:safe", "locked"]], "fired": ["locked"]}},
        {"set": "world:demo.power", "to": True, "actor": "npc:cook"},
        {"save_load": True},                                                # feedback is never saved
        {"expect": {"feedback": [], "fired": ["powered"]}},
        {"advance": 1.0},
        {"expect": {"feedback": [["notify", "The lights hum", "npc:cook"]], "errors": 0}},
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


# ---- quests: compiled by geogen.quests, so a runtime that passes these runs quests (docs/quests.md) ----

from geogen.quests import compile_quests
from geogen.state import Declarations

traces["event_data"] = {
    "about": "Rules read the data of the event they answer (event.<field>): a bounty adds gold, a leak takes lives, "
             "a delayed rule keeps its event's data, emit carries data and forward passes it on.",
    "program": program(
        variables={"world:demo.gold": {"type": "number", "default": 0},
                   "world:demo.lives": {"type": "number", "default": 3},
                   "world:demo.last": {"type": "text", "default": ""},
                   "demo:lamp#lit": {"type": "bool", "default": False}},
        rules=[
            {"id": "bounty", "on": {"type": "signal", "subject": "demo:gate", "name": "killed"},
             "when": c("event.value > 0"),
             "do": [{"set": "world:demo.gold", "to": c("world.gold + event.value")},
                    {"set": "world:demo.last", "to": c("event.item")}]},
            {"id": "leak", "on": {"type": "signal", "subject": "demo:gate", "name": "leaked"}, "after": 1.0,
             "do": [{"set": "world:demo.lives", "to": c("max(0, world.lives - event.harm)")}]},
            {"id": "relay", "on": {"type": "signal", "subject": "demo:gate", "name": "leaked"},
             "do": [{"emit": "alarm", "subject": "demo:core", "forward": True,
                     "data": {"double": c("event.harm * 2")}}]},
            {"id": "lit", "on": {"type": "changed", "key": "demo:lamp#lit"},
             "do": [{"set": "world:demo.last", "to": c('"lamp on" if event.to else "lamp off"')}]},
            {"id": "missing", "on": {"type": "signal", "subject": "demo:gate", "name": "odd"},
             "do": [{"set": "world:demo.gold", "to": c("event.value")}]}]),
    "actors": {"player": {"kind": "player"}},
    "steps": [
        {"event": {"type": "signal", "subject": "demo:gate", "name": "killed", "data": {"value": 5, "item": "grunt"}}},
        {"event": {"type": "signal", "subject": "demo:gate", "name": "killed", "data": {"value": 0, "item": "dud"}}},
        {"expect": {"state": {"world:demo.gold": 5, "world:demo.last": "grunt"}, "fired": ["bounty"]}},
        {"event": {"type": "signal", "subject": "demo:gate", "name": "leaked", "data": {"harm": 2}}},
        {"expect": {"state": {"world:demo.lives": 3}, "fired": ["leak", "relay"],
                    "signal_data": [["demo:gate", "leaked", {"harm": 2}], ["demo:core", "alarm", {"harm": 2, "double": 4}]]}},
        {"save_load": True},                                              # the pending rule keeps its data
        {"advance": 1.0},
        {"expect": {"state": {"world:demo.lives": 1}, "fired": []}},
        {"set": "demo:lamp#lit", "to": True},
        {"expect": {"state": {"world:demo.last": "lamp on"}, "fired": ["lit"]}},
        {"event": {"type": "signal", "subject": "demo:gate", "name": "odd"}},    # no data: reported, nothing set
        {"expect": {"state": {"world:demo.gold": 5}, "fired": ["missing"], "errors": 1}},
    ],
}


def quest_trace(about, quests, actors, steps, interactions=None, variables=None, selectors=None, rules=()):
    declarations = Declarations(PKG)
    declarations.interactions = dict(interactions or {})
    declarations.variables = dict(variables or {})
    compiled = compile_quests(quests, declarations, selectors=selectors)
    rules = [{**r, "do": [e for effect in r["do"] for e in (compiled.effect(effect) if "set" not in effect and "emit" not in effect else [effect])]}
             for r in rules]
    return {"about": about, "quests": compiled.quests, "program": compiled.program(rules), "actors": actors, "steps": steps}

def q(quest, objective=None, suffix=""):
    return f"actor:{PKG}.quest_{quest}" + (f"_{objective}" if objective else "") + suffix

SWITCH = {"initial": "on", "duration": 0.2, "states": {"on": {"next": "off"}, "off": {"next": "on"}}}

traces["quest_go_there"] = quest_trace(
    "Quests that start with the game: reach a place, and reach it three times. Only the holder's own arrival counts.",
    [{"id": "view", "title": "The view from the top", "start": "auto",
      "objectives": [{"id": "gallery", "text": "Climb to the gallery", "done": {"enter": "gallery"}}]},
     {"id": "regular", "title": "A regular", "start": "auto",
      "objectives": [{"id": "visits", "text": "Visit the gallery three times", "done": {"enter": "gallery", "count": 3}}]}],
    {"player": {"kind": "player"}, "npc:keeper": {"kind": "npc"}},
    [
        {"expect": {"actors": {"player": {q("view"): "active", q("regular"): "active"},
                               "npc:keeper": {q("view"): "available"}},        # an NPC can't hold it: never started
                    "fired": ["quest:view:start", "quest:regular:start"], "errors": 0}},
        {"event": {"type": "enter", "subject": "demo:gallery", "actor": "npc:keeper"}},
        {"expect": {"actors": {"player": {q("view"): "active", q("view", "gallery"): False}}, "fired": []}},
        {"save_load": True},
        {"event": {"type": "enter", "subject": "demo:gallery", "actor": "player"}},
        {"expect": {"actors": {"player": {q("view"): "complete", q("view", "gallery"): True,
                                          q("regular"): "active", q("regular", "visits", "_count"): 1}},
                    "fired": ["quest:view:gallery", "quest:regular:visits"]}},
        {"event": {"type": "leave", "subject": "demo:gallery", "actor": "player"}},
        {"event": {"type": "enter", "subject": "demo:gallery", "actor": "player"}},
        {"save_load": True},
        {"event": {"type": "enter", "subject": "demo:gallery", "actor": "player"}},
        {"expect": {"actors": {"player": {q("regular"): "complete", q("regular", "visits", "_count"): 3,
                                          q("view"): "complete"}},
                    "fired": ["quest:regular:visits", "quest:regular:visits"]}},
        {"event": {"type": "enter", "subject": "demo:gallery", "actor": "player"}},     # finished: counts no more
        {"expect": {"actors": {"player": {q("regular", "visits", "_count"): 3}}, "fired": [], "errors": 0}},
    ])

traces["quest_change_the_world"] = quest_trace(
    "A quest given by a rule when the player greets an NPC: a condition over every light, then an objective that waits for it, an optional one, a reward.",
    [{"id": "lights_out", "title": "Lights out",
      "objectives": [
          {"id": "lights", "text": "Switch off every light", "done": 'all("[tag=switch.light]", it.switch == "off")'},
          {"id": "bed", "text": "Go to bed", "after": "lights", "done": {"enter": "bedroom"}},
          {"id": "window", "text": "Close the window", "optional": True, "done": 'window.swing == "closed"'}],
      "rewards": [{"set": "actor.slept_well", "to": True}, {"emit": "rested", "of": "resident"}]}],
    {"player": {"kind": "player"}, "npc:visitor": {"kind": "npc"}},
    [
        {"use": "demo:hall#switch", "actor": "player"},                   # before the quest: the world changes anyway
        {"event": {"type": "enter", "subject": "demo:bedroom", "actor": "player"}},
        # the resident gives it to the player who greets them, not to another NPC who does
        {"event": {"type": "signal", "subject": "demo:resident", "name": "greeted", "actor": "npc:visitor"}},
        {"expect": {"actors": {"player": {q("lights_out"): "available"}, "npc:visitor": {q("lights_out"): "available"}},
                    "fired": []}},
        {"event": {"type": "signal", "subject": "demo:resident", "name": "greeted", "actor": "player"}},
        {"expect": {"actors": {"player": {q("lights_out"): "active", q("lights_out", "lights"): False}}, "fired": ["give"]}},
        {"event": {"type": "enter", "subject": "demo:bedroom", "actor": "player"}},    # too early: it waits on the lights
        {"use": "demo:kitchen#switch", "actor": "player"},
        {"expect": {"actors": {"player": {q("lights_out", "bed"): False, q("lights_out", "lights"): False}}, "fired": []}},
        {"use": "demo:lounge#switch", "actor": "player"},
        {"expect": {"actors": {"player": {q("lights_out", "lights"): True}}, "fired": ["quest:lights_out:lights"]}},
        {"use": "demo:hall#switch", "actor": "player"},                   # back on: done stays done
        {"save_load": True},
        {"expect": {"actors": {"player": {q("lights_out", "lights"): True, q("lights_out"): "active"}}, "fired": []}},
        {"event": {"type": "enter", "subject": "demo:bedroom", "actor": "player"}},
        {"expect": {"actors": {"player": {q("lights_out"): "complete", q("lights_out", "bed"): True,
                                          "actor:demo.slept_well": True}},
                    "signals": [["demo:resident", "rested"]],
                    "fired": ["quest:lights_out:bed", "quest:lights_out:rewards"]}},
        {"use": "demo:window#swing", "actor": "player"},                  # optional, and too late
        {"event": {"type": "signal", "subject": "demo:resident", "name": "greeted", "actor": "player"}},
        {"expect": {"actors": {"player": {q("lights_out"): "complete", q("lights_out", "window"): False}},
                    "fired": ["give"], "errors": 0}},
    ],
    interactions={**{f"demo:{room}#switch": SWITCH for room in ("hall", "kitchen", "lounge")},
                  "demo:window#swing": {"initial": "open", "duration": 0.5,
                                        "states": {"open": {"next": "closed"}, "closed": {"next": "open"}}}},
    variables={"actor:demo.slept_well": {"type": "bool", "default": False}},
    selectors={"[tag=switch.light]": ["demo:hall", "demo:kitchen", "demo:lounge"]},
    rules=[{"id": "give", "on": {"type": "signal", "subject": "demo:resident", "name": "greeted", "actor": "player"},
            "do": [{"start_quest": "lights_out"}]}])

traces["quest_timed"] = quest_trace(
    "A quest anyone can hold, open from a clock time, that fails at another: one actor makes it and one doesn't.",
    [{"id": "catch_train", "title": "Catch the 17:05", "holder": "any", "start": "auto",
      "available": 'clock.time >= "17:00"',
      "objectives": [{"id": "board", "text": "Be on the platform while the doors are open",
                      "done": {"enter": "platform", "when": 'train.doors == "open"'}}],
      "fail": 'clock.time >= "17:10"'}],
    {"player": {"kind": "player"}, "npc:late": {"kind": "npc"}},
    [
        {"expect": {"actors": {"player": {q("catch_train"): "unavailable"}, "npc:late": {q("catch_train"): "unavailable"}},
                    "fired": []}},
        {"event": {"type": "enter", "subject": "demo:platform", "actor": "player"}},
        {"clock": {"time": "17:00"}},
        {"expect": {"actors": {"player": {q("catch_train"): "active"}, "npc:late": {q("catch_train"): "active"}},
                    "fired": ["quest:catch_train:available", "quest:catch_train:available",
                              "quest:catch_train:start", "quest:catch_train:start"]}},
        {"event": {"type": "leave", "subject": "demo:platform", "actor": "player"}},
        {"event": {"type": "enter", "subject": "demo:platform", "actor": "player"}},      # the doors are shut
        {"expect": {"actors": {"player": {q("catch_train"): "active"}}, "fired": []}},
        {"set": "demo:train#doors", "to": "open"},
        {"event": {"type": "leave", "subject": "demo:platform", "actor": "player"}},
        {"event": {"type": "enter", "subject": "demo:platform", "actor": "player"}},
        {"expect": {"actors": {"player": {q("catch_train"): "complete"}, "npc:late": {q("catch_train"): "active"}},
                    "fired": ["quest:catch_train:board"]}},
        {"clock": {"time": "17:10"}},
        {"expect": {"actors": {"player": {q("catch_train"): "complete"}, "npc:late": {q("catch_train"): "failed"}},
                    "fired": ["quest:catch_train:fail"]}},
        {"event": {"type": "enter", "subject": "demo:platform", "actor": "npc:late"}},      # failed is final
        {"clock": {"time": "16:00"}},
        {"expect": {"actors": {"npc:late": {q("catch_train"): "failed", q("catch_train", "board"): False}},
                    "fired": [], "errors": 0}},
    ],
    interactions={"demo:train#doors": {"initial": "closed", "duration": 1.0, "states": {"closed": {}, "open": {}}}})

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
for name, trace in traces.items():
    (out / f"{name}.json").write_text(json.dumps({"name": name, **trace}, indent=1) + "\n")
print(len(traces))
