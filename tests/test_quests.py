"""Quests (geogen/quests.py): the quest kind, its compilation to state and rules, and export checks."""

import json
from pathlib import Path

import pytest

from geogen.chunks import export_chunks
from geogen.export import export_scene, manifest_path
from geogen.layout.composer import SceneComposer
from geogen.quests import STATES, QuestError, collect, compile_quests, load_quest
from geogen.rules import Machine, check_program
from geogen.state import Declarations
from test_state import GATE, LAMP, LEVER

SCHEMA = Path(__file__).parent.parent / "docs/schema/geogen-quests.v1.schema.json"
STATE_SCHEMA = SCHEMA.with_name("geogen-state.v1.schema.json")
ACTORS = {"player": {"kind": "player"}, "npc:ann": {"kind": "npc"}}
SWITCH = {"initial": "on", "duration": 0.2, "states": {"on": {"next": "off"}, "off": {"next": "on"}}}
LIGHTS = ["demo:hall", "demo:kitchen"]


def declarations():
    found = Declarations("demo")
    found.interactions = {f"{thing}#switch": SWITCH for thing in LIGHTS}
    found.interactions["demo:gate#gate"] = {"initial": "closed", "duration": 1.0,
                                           "states": {"closed": {}, "open": {"emit": "opened"}}}
    found.variables = {"actor:demo.tired": {"type": "bool", "default": False},
                       "world:demo.alarm": {"type": "bool", "default": False},
                       "actor:demo.coins": {"type": "number", "default": 0}}
    return found


def quest(**parts):
    return {"id": "lights_out", "title": "Lights out",
            "objectives": [{"id": "lights", "text": "Switch off every light",
                            "done": 'all("[tag=light]", it.switch == "off")'},
                           {"id": "bed", "text": "Go to bed", "done": {"enter": "bedroom"}}], **parts}


def run(*quests, rules=(), actors=ACTORS):
    """(compiled quests, a machine running them). ``rules`` may use the quest effects."""
    plain = [rule for rule in rules if all("set" in effect for effect in rule["do"])]
    compiled = compile_quests(list(quests), declarations(), selectors={"[tag=light]": LIGHTS}, rules=plain)
    rules = [rule if rule in plain else {**rule, "do": [e for effect in rule["do"] for e in compiled.effect(effect)]}
             for rule in rules]
    return compiled, Machine(compiled.program(rules), actors)


RING = {"id": "ring", "on": {"type": "signal", "subject": "demo:bell", "name": "rung"},
        "do": [{"set": "world:demo.alarm", "to": True}]}
GIVE = {"id": "give", "on": {"type": "signal", "subject": "demo:ann", "name": "greeted"},
        "do": [{"start_quest": "lights_out"}]}


def greet(machine, actor="player"):
    machine.post({"type": "signal", "subject": "demo:ann", "name": "greeted", "actor": actor})


def enter(machine, place, actor="player"):
    machine.post({"type": "enter", "subject": f"demo:{place}", "actor": actor})


def states(report, name="lights_out"):
    return {o["id"]: o["state"] for o in report[name]["objectives"]}


def test_available_to_active_to_complete():
    quests, machine = run(quest(), rules=[GIVE])
    assert quests.status(machine, "lights_out", "player") == "available"
    enter(machine, "bedroom")                           # nothing counts before it starts
    greet(machine)
    report = quests.report(machine, "player")
    assert report["lights_out"]["status"] == "active" and report["lights_out"]["title"] == "Lights out"
    assert states(report) == {"lights": "open", "bed": "open"}
    assert report["lights_out"]["objectives"][0]["progress"] == [0, 2]

    machine.use("demo:hall#switch", "player")
    assert quests.report(machine, "player")["lights_out"]["objectives"][0]["progress"] == [1, 2]
    enter(machine, "bedroom")                           # unordered: this one can come first
    assert states(quests.report(machine, "player")) == {"lights": "open", "bed": "done"}
    machine.use("demo:kitchen#switch", "player")
    assert quests.status(machine, "lights_out", "player") == "complete"
    machine.use("demo:hall#switch", "player")           # undoing it afterwards changes nothing
    greet(machine)
    assert quests.status(machine, "lights_out", "player") == "complete"
    assert states(quests.report(machine, "player")) == {"lights": "done", "bed": "done"}
    # progress is the actor's own: nobody else moved
    assert quests.status(machine, "lights_out", "npc:ann") == "available"
    assert machine.errors == []


def test_available_to_active_to_failed_and_failed_is_final():
    quests, machine = run(quest(fail="world.alarm"), rules=[
        GIVE, {"id": "alarm", "on": {"type": "signal", "subject": "demo:bell", "name": "rung"},
               "do": [{"fail_quest": "lights_out"}]}])
    machine.set("world:demo.alarm", True)               # not active yet: nothing to fail
    assert quests.status(machine, "lights_out", "player") == "available"
    machine.set("world:demo.alarm", False)
    greet(machine)
    machine.set("world:demo.alarm", True)
    assert quests.status(machine, "lights_out", "player") == "failed"
    enter(machine, "bedroom")
    greet(machine)
    assert quests.status(machine, "lights_out", "player") == "failed"
    assert states(quests.report(machine, "player")) == {"lights": "missed", "bed": "missed"}   # open when it failed

    _, other = run(quest(), rules=[GIVE, {"id": "alarm", "on": {"type": "signal", "subject": "demo:bell", "name": "rung"},
                                          "do": [{"fail_quest": "lights_out"}]}])
    greet(other)
    other.post({"type": "signal", "subject": "demo:bell", "name": "rung", "actor": "player"})
    assert other.get("actor:demo.quest_lights_out", "player") == "failed"


def test_the_last_objective_beats_a_fail_at_the_same_moment():
    done_and_failed = quest(objectives=[{"id": "alarm", "text": "Raise the alarm", "done": "world.alarm"}],
                            fail="world.alarm", start="auto")
    quests, machine = run(done_and_failed, rules=[RING])
    machine.post({"type": "signal", "subject": "demo:bell", "name": "rung"})
    assert quests.status(machine, "lights_out", "player") == "complete"
    assert [rule for rule, _ in machine.fired] == ["quest:lights_out:start", "ring", "quest:lights_out:alarm",
                                                  "quest:lights_out:fail"]


def test_ordered_objectives_wait_and_optional_ones_dont_block():
    ordered = quest(ordered=True, start="auto", objectives=[
        {"id": "lights", "text": "Lights", "done": 'all("[tag=light]", it.switch == "off")'},
        {"id": "snack", "text": "Have a snack", "optional": True, "done": {"enter": "kitchen"}},
        {"id": "bed", "text": "Bed", "done": {"enter": "bedroom"}}])
    quests, machine = run(ordered)
    assert quests.quests["lights_out"]["objectives"][2]["after"] == ["lights"]     # not the optional one
    assert states(quests.report(machine, "player")) == {"lights": "open", "snack": "waiting", "bed": "waiting"}
    enter(machine, "bedroom")
    enter(machine, "kitchen")
    assert states(quests.report(machine, "player")) == {"lights": "open", "snack": "waiting", "bed": "waiting"}
    for light in LIGHTS:
        machine.use(f"{light}#switch", "player")
    assert states(quests.report(machine, "player")) == {"lights": "done", "snack": "open", "bed": "open"}
    enter(machine, "bedroom")
    assert quests.status(machine, "lights_out", "player") == "complete"
    enter(machine, "kitchen")                           # the quest is over
    assert states(quests.report(machine, "player"))["snack"] == "waiting"


def test_auto_start_waits_until_available_and_respects_the_holder():
    night = quest(start="auto", holder="npc", available="clock.night")
    quests, machine = run(night)
    assert STATES[0] == quests.status(machine, "lights_out", "npc:ann") == "unavailable"
    machine.set_clock(night=True)
    assert quests.status(machine, "lights_out", "npc:ann") == "active"
    assert quests.status(machine, "lights_out", "player") == "unavailable"     # not a holder
    machine.set_clock(night=False)
    assert quests.status(machine, "lights_out", "npc:ann") == "active"         # started: no going back

    quests, machine = run(quest(available="clock.night"), rules=[GIVE])
    greet(machine)                                      # given too early
    assert quests.status(machine, "lights_out", "player") == "unavailable"
    machine.set_clock(night=True)
    assert quests.status(machine, "lights_out", "player") == "available"
    machine.set_clock(night=False)
    assert quests.status(machine, "lights_out", "player") == "unavailable"
    machine.set_clock(night=True)
    greet(machine, "npc:ann")                           # the holder is the player (the default)
    assert quests.status(machine, "lights_out", "npc:ann") == "unavailable"
    greet(machine)
    assert quests.status(machine, "lights_out", "player") == "active"


def test_rewards_chain_quests_and_complete_objective_respects_order():
    first = quest(ordered=True, rewards=[{"set": "actor.coins", "to": "actor.coins + 5"}, {"start_quest": "sleep"}])
    second = {"id": "sleep", "title": "Sleep", "objectives": [
        {"id": "tired", "text": "Get tired", "done": "actor.coins >= 10"}],
        "rewards": [{"set": "actor.tired", "to": False}, {"emit": "rested"}]}
    skip = {"id": "skip", "on": {"type": "signal", "subject": "demo:ann", "name": "nudged"},
            "do": [{"complete_objective": "lights_out.bed"}, {"complete_objective": "lights_out.lights"}]}
    yawn = {"id": "yawn", "on": {"type": "signal", "subject": "demo:ann", "name": "yawned"},
            "do": [{"complete_objective": "sleep.tired"}]}
    quests, machine = run(first, second, rules=[GIVE, skip, yawn])

    def nudge(name):
        machine.post({"type": "signal", "subject": "demo:ann", "name": name, "actor": "player"})

    nudge("nudged")                                     # not started
    assert quests.status(machine, "lights_out", "player") == "available"
    greet(machine)
    nudge("nudged")                                     # bed waits on lights, which is done second: only lights
    assert states(quests.report(machine, "player")) == {"lights": "done", "bed": "open"}
    nudge("nudged")
    assert quests.status(machine, "lights_out", "player") == "complete"
    assert machine.get("actor:demo.coins", "player") == 5
    assert quests.status(machine, "sleep", "player") == "active"
    nudge("yawned")
    assert quests.status(machine, "sleep", "player") == "complete"
    assert ["demo:", "rested", "player"] in machine.signals and machine.errors == []


def test_progress_survives_a_save():
    quests, machine = run(quest(), rules=[GIVE])
    greet(machine)
    enter(machine, "bedroom")
    saved = json.loads(json.dumps(machine.save()))
    assert saved["actors"]["player"]["actor:demo.quest_lights_out"] == "active"
    assert not saved["world"]                           # quest progress is the actor's
    other = Machine(machine.program, ACTORS)
    other.load(saved)
    assert states(quests.report(other, "player")) == {"lights": "open", "bed": "done"}
    for light in LIGHTS:
        other.use(f"{light}#switch", "player")
    assert quests.status(other, "lights_out", "player") == "complete"


def objective(done, **parts):
    return quest(objectives=[{"id": "only", "text": "Do it", "done": done, **parts}])


@pytest.mark.parametrize("spec, message", [
    (quest(tittle="x"), r"unknown key 'tittle'. Did you mean 'title'\?"),
    ({**quest(), "title": ""}, "a quest needs a title"),
    (quest(holder="monster"), "holder is one of player, npc, any"),
    (quest(start="soon"), "start is one of given, auto"),
    (quest(objectives=[]), "a quest needs objectives"),
    (objective("world.alarm", optional=True), "every objective is optional"),
    (objective("world.alarm", txet="x"), "unknown key 'txet'"),
    (quest(objectives=[{"id": "a", "done": "world.alarm"}]), "an objective needs text"),
    (quest(objectives=[{"id": "a", "text": "A"}]), "an objective needs done"),
    (objective("world.alarm", after="bedd"), "this quest has no objective 'bedd'"),
    (objective("world.alarm", after="only"), "objective 'only' waits on itself: only -> only"),
    (quest(objectives=[{"id": "a", "text": "A", "after": "b", "done": "world.alarm"},
                       {"id": "b", "text": "B", "after": ["a"], "done": "world.alarm"}]),
     "objective 'a' waits on itself: a -> b -> a"),
    (objective("actor.quest_lights_out_only"), "waits on itself: its condition reads its own done state"),
    (objective("world.alarm and"), "can't read condition"),
    (objective("world.alarrm"), r"no world state 'alarrm' is declared in 'demo'. Did you mean 'alarm'\?"),
    (objective("world.tired"), "'tired' is actor state, not world state"),
    (objective('hall.switch == "of"'), r"'hall.switch' is never 'of'. Did you mean 'off'\? It is one of: on, off"),
    (objective('gate.gate == "open"'), "nothing moves 'gate.gate' to 'open': it starts 'closed'"),
    (objective("1 + 1 == 2"), "reads no state, so nothing can make it true"),
    (objective("actor.tired"), r"nothing changes what it reads \(actor.tired\)"),
    (objective({"signal": "gate.boom"}), "'demo:gate' never raises 'boom'. It raises: opened"),
    (objective({"signal": "gate"}), "write signal: <thing>.<name>"),
    (objective({"use": "gate"}), "can't be used"),
    (objective({"use": "hall.swich"}), r"no interaction 'swich' to use. Did you mean 'switch'\?"),
    (objective({"changed": "world.alarm", "to": 3}), "is a bool: it can't change to 3"),
    (objective({"changed": "world.alarm"}), "nothing changes 'world.alarm'"),
    (objective({"clock": "night"}), "The clock is state, not an event"),
    (objective({"enter": "bedroom", "leave": "bedroom"}), "an event is one of"),
    (objective({"enter": "bedroom", "count": 0}), "count is a whole number of times"),
    (objective("world.alarm", marker={"radius": 3}), "a marker is hidden, a reference or {at, area}"),
    (quest(fail="world.alarrm"), "fail: no world state 'alarrm'"),
    (quest(rewards=[{"set": "world.alarm"}]), "rewards.0: set needs to"),
    (quest(rewards=[{"set": "world.alarm", "to": 3}]), "is a bool: it can't be 3"),
    (quest(rewards=[{"start_quest": "nope"}]), "no quest 'nope'"),
    (quest(rewards=[{"complete_objective": "lights_out.nope"}]), "has no objective 'nope'"),
    (quest(rewards=[{"explode": 1}]), "an effect is one of set, emit, input, start_quest"),
])
def test_bad_quests_say_what_is_wrong(spec, message):
    with pytest.raises(QuestError, match=message):
        compile_quests([spec], declarations(), selectors={"[tag=light]": LIGHTS})


def test_every_problem_is_listed_and_names_clash():
    bad = quest(objectives=[{"id": "a", "text": "A", "done": "world.nope"},
                            {"id": "b", "text": "B", "done": {"signal": "gate"}}], fail="actor.missing")
    with pytest.raises(QuestError) as failed:
        compile_quests([bad], declarations())
    assert len(str(failed.value).splitlines()) == 3
    with pytest.raises(QuestError, match="contributed twice"):
        compile_quests([quest(), quest()], declarations(), selectors={"[tag=light]": LIGHTS})
    clash = [{"id": "a_b", "title": "One", "objectives": [{"id": "c", "text": "C", "done": "world.alarm"}]},
             {"id": "a", "title": "Two", "objectives": [{"id": "b", "text": "B", "done": "world.alarm"}]}]
    with pytest.raises(QuestError, match="quest state 'quest_a_b' is already declared"):
        compile_quests(clash, declarations())


def test_what_other_rules_set_can_be_an_objective():
    """A condition is only impossible if nothing at all sets what it reads."""
    spec = objective('gate.gate == "open" and world.alarm')
    with pytest.raises(QuestError, match="nothing moves 'gate.gate'"):
        compile_quests([spec], declarations())
    wiring = [{"id": "open", "on": {"type": "signal", "subject": "demo:lever", "name": "pulled"},
               "do": [{"set": "demo:gate#gate", "to": "open"}, {"set": "world:demo.alarm", "to": True}]}]
    quests = compile_quests([spec], declarations(), rules=wiring)
    assert check_program(quests.program(wiring)) == []
    assert quests.quests["lights_out"]["objectives"][0]["subjects"] == ["demo:gate"]


# ---- in a scene ----------------------------------------------------------------------------------

SCENE = """
name: yard
size: [12, 3, 12]
state:
  bell_rung: {scope: world, type: bool, default: false}
  thanked:   {scope: actor, type: bool, default: false}
quests: [quests/open_up.yaml]
slots:
  a: {position: [-3, 0, 0]}
  b: {position: [0, 0, 0]}
  c: {position: [3, 0, 0]}
place:
  lever: {asset: lever.yaml, slot: a}
  gate:  {asset: gate.yaml, slot: b}
  lamp:  {asset: lamp.yaml, slot: c}
places:
  porch: {box: [2, 2, 2], position: [0, 0, 4]}
"""

QUEST = """# Pull the lever, then stand on the porch.
kind: quest
version: 1
title: Open up
description: Let yourself in.
start: auto
objectives:
  - id: lever
    text: Pull the lever
    done: 'lever.pull == "down"'
  - id: porch
    text: Stand on the porch
    after: lever
    done: {enter: porch}
    marker: {at: gate, area: 3}
  - id: twice
    text: Pull it twice more
    optional: true
    done: {signal: lever.pulled, count: 2}
    marker: hidden
rewards:
  - {set: actor.thanked, to: true}
"""


@pytest.fixture
def assets(tmp_path):
    for name, text in (("lever", LEVER), ("gate", GATE), ("lamp", LAMP), ("yard", SCENE), ("quests/open_up", QUEST)):
        path = tmp_path / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    return tmp_path


def compose(assets):
    return SceneComposer(assets_dir=assets).compose(assets / "yard.yaml")


def rewrite(assets, name, old, new):
    path = assets / f"{name}.yaml"
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def test_a_scene_quest_resolves_and_runs(assets):
    assert load_quest(assets / "quests/open_up.yaml")["id"] == "open_up"
    quests = collect(compose(assets), "yard")
    entry = quests.quests["open_up"]
    assert entry["status"] == "actor:yard.quest_open_up" and entry["start"] == "auto"
    lever, porch, twice = entry["objectives"]
    assert lever["subjects"] == ["yard:lever"] and porch["after"] == ["lever"]
    assert porch["subjects"] == ["yard:gate"] and porch["marker"] == {"area": 3.0}
    assert twice["marker"] == {"hidden": True} and twice["subjects"] == [] and twice["count"]["of"] == 2
    assert quests.declarations.spec("actor:yard.quest_open_up")["choices"] == list(STATES)

    machine = Machine(quests.program(), {"player": {"kind": "player"}})
    assert quests.status(machine, "open_up", "player") == "active"
    machine.post({"type": "enter", "subject": "yard:porch", "actor": "player"})
    machine.use("yard:lever#pull", "player")
    machine.advance(0.6)                                # arrives: 'pulled', once
    assert quests.report(machine, "player")["open_up"]["objectives"][2]["progress"] == [1, 2]
    machine.post({"type": "enter", "subject": "yard:porch", "actor": "player"})
    assert quests.status(machine, "open_up", "player") == "complete"
    assert machine.get("actor:yard.thanked", "player") is True and machine.errors == []


def test_manifest_lists_quests_and_matches_schema(assets, tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    out = tmp_path / "out"
    path = export_scene(compose(assets), out / "yard.glb")
    manifest = json.loads(manifest_path(path).read_text())
    jsonschema.validate(manifest["quests"], json.loads(SCHEMA.read_text()))
    jsonschema.validate(manifest["state"], json.loads(STATE_SCHEMA.read_text()))
    assert manifest["quests"]["package"] == "yard" and list(manifest["quests"]["quests"]) == ["open_up"]
    quest_rules = [r for r in manifest["rules"]["rules"] if r["id"].startswith("quest:")]
    jsonschema.validate(quest_rules[0]["when"],
                        json.loads(SCHEMA.with_name("geogen-condition.v1.schema.json").read_text()))
    assert manifest["rules"]["format"] == "geogen-rules" and manifest["rules"]["bindings"] == []
    assert [r["id"] for r in quest_rules] == [
        "quest:open_up:start", "quest:open_up:lever", "quest:open_up:porch", "quest:open_up:twice",
        "quest:open_up:rewards"]
    # quest progress is declared state like any other
    assert {"actor:yard.quest_open_up", "actor:yard.quest_open_up_porch", "actor:yard.quest_open_up_twice_count",
            "actor:yard.thanked"} <= set(manifest["state"]["variables"])
    assert "Pull the lever" not in path.read_bytes().decode("latin-1")     # not node extras
    # what was exported runs as it is
    program = {"format": "geogen-rules", "version": 1, "variables": manifest["state"]["variables"],
               "interactions": manifest["state"]["interactions"], "rules": manifest["rules"]["rules"]}
    assert check_program(program) == []

    index = json.loads(export_chunks(compose(assets), out / "chunks", name="yard").read_text())
    assert index["quests"] == manifest["quests"] and index["rules"] == manifest["rules"]


@pytest.mark.parametrize("old, new, message", [
    ("lever.pull ==", "levr.pull ==", r"quests/open_up.yaml:10: objectives.0.done: .*'levr'.*Did you mean 'lever'\?"),
    ('"down"', '"dwon"', r"open_up.yaml:10: objectives.0.done: 'lever.pull' is never 'dwon'. Did you mean 'down'\?"),
    ("lever.pull ==", "lever.pulll ==", r"open_up.yaml:10: objectives.0.done: .*has no state 'pulll'"),
    ("lever.pull ==", "gate.gate ==", r"open_up.yaml:10: objectives.0.done: 'gate.gate' is never 'down'"),
    ('lever.pull == "down"', 'gate.gate == "open"',
     r"open_up.yaml:10: objectives.0.done: objective 'lever' can't be done: nothing moves 'gate.gate' to 'open'"),
    ("enter: porch", "enter: proch", r"open_up.yaml:14: objectives.1.done: .*'proch'.*Did you mean 'porch'\?"),
    ("enter: porch", "enter: lamp", r"open_up.yaml:14: objectives.1.done: 'lamp' is not a place or a room. Places here: porch"),
    ("lever.pulled, count", "lever.pushed, count",
     r"open_up.yaml:19: objectives.2.done: objective 'twice' can't be done: 'yard:lever' never raises 'pushed'"),
    ("at: gate", "at: gaet", r"open_up.yaml:15: objectives.1.marker: .*'gaet'"),
    ("actor.thanked", "world.thanked", r"open_up.yaml:22: rewards.0: 'thanked' is actor state, not world state"),
    ("start: auto", "start: auto\nfail: world.bell_rang",
     r"open_up.yaml:7: fail: no world state 'bell_rang' is declared in 'yard'. Did you mean 'bell_rung'\?"),
])
def test_bad_quests_fail_the_export_with_file_and_line(assets, tmp_path, old, new, message):
    rewrite(assets, "quests/open_up", old, new)
    with pytest.raises(QuestError, match=message):
        export_scene(compose(assets), tmp_path / "out" / "yard.glb")
    assert not (tmp_path / "out" / "yard.glb").exists()


def test_scene_quest_lists_are_checked(assets):
    rewrite(assets, "yard", "quests/open_up.yaml", "quests/open_pu.yaml")
    with pytest.raises(QuestError, match=r"yard.yaml:7: quests.0: no quest file 'quests/open_pu.yaml'. "
                                         r"Did you mean 'quests/open_up.yaml'\?"):
        compose(assets)
    rewrite(assets, "yard", "quests/open_pu.yaml", "lever.yaml")
    with pytest.raises(QuestError, match="expected kind 'quest' version 1"):
        compose(assets)


def test_island_has_the_go_there_quest(built_scene):
    quests = collect(built_scene("island"), "island")
    gallery = quests.quests["lighthouse_gallery"]["objectives"][0]
    assert gallery["subjects"] == ["island:lighthouse/gallery"]
    machine = Machine(quests.program(), {"player": {"kind": "player"}})
    machine.post({"type": "enter", "subject": "island:lighthouse/gallery", "actor": "player"})
    assert quests.status(machine, "lighthouse_gallery", "player") == "complete"


def test_cottage_has_the_change_the_world_quest(built_scene):
    """Given by greeting the resident, done when every light switch is off, counted on the way."""
    quests = collect(built_scene("cottage"), "cottage")
    switches = [f"cottage:{name}_switch" for name in ("dining", "sitting", "lamp")]
    lights = quests.quests["lights_out"]["objectives"][0]
    assert sorted(lights["subjects"]) == sorted(switches) and lights["progress"]["of"] == 3
    # the resident's greeting is a signal the export knows about
    assert quests.declarations.signals["cottage:resident#greeted"] == {"subject": "cottage:resident", "name": "greeted"}
    give = next(r for r in quests.wiring if r["id"] == "give_lights_out")
    assert give["on"] == {"type": "signal", "subject": "cottage:resident", "name": "greeted", "actor": "player"}

    machine = Machine(quests.program(), {"player": {"kind": "player"}, "cottage:resident": {"kind": "npc"}})

    def progress():
        (objective,) = quests.report(machine, "player")["lights_out"]["objectives"]
        return objective["state"], objective["progress"]

    def greet(actor):
        machine.post({"type": "signal", "subject": "cottage:resident", "name": "greeted", "actor": actor})

    assert quests.status(machine, "lights_out", "player") == "available" and progress() == ("waiting", [0, 3])
    machine.use(f"{switches[0]}#switch", "player")              # before the quest: the count moves anyway
    assert progress() == ("waiting", [1, 3])
    machine.use(f"{switches[0]}#switch", "player")
    greet("cottage:resident")                                   # not the player's greeting
    assert quests.status(machine, "lights_out", "player") == "available"
    greet("player")
    assert quests.status(machine, "lights_out", "player") == "active" and progress() == ("open", [0, 3])
    for count, switch in enumerate(switches, 1):
        assert quests.status(machine, "lights_out", "player") == "active"
        machine.use(f"{switch}#switch", "player")
        assert progress()[1] == [count, 3]
    assert quests.status(machine, "lights_out", "player") == "complete" and progress()[0] == "done"
    machine.use(f"{switches[1]}#switch", "player")              # back on: done stays done
    greet("player")
    assert quests.status(machine, "lights_out", "player") == "complete" and progress() == ("done", [2, 3])
    assert quests.status(machine, "lights_out", "cottage:resident") == "available" and machine.errors == []


def test_a_nested_cottage_contributes_no_quest(built_scene):
    """The cottage is part of bigger scenes (two on the island): there it gives no quest."""
    quests = collect(built_scene("island"), "island")
    assert list(quests.quests) == ["lighthouse_gallery"] and not quests.wiring
    # its residents can still be greeted, and rules of the bigger scene could listen
    assert {"island:cottage_a/resident#greeted", "island:cottage_b/resident#greeted"} <= set(quests.declarations.signals)


def test_level_crossing_has_the_timed_quest(built_scene):
    """Open from 08:00, done by being on the north platform while the train stands there, failed
    when that train leaves first. The train's `at_north` is the runtime's to set (its timetable)."""
    quests = collect(built_scene("level_crossing"), "level_crossing")
    train, platform = "level_crossing:commuter#at_north", "level_crossing:railway_main/station_north"
    assert quests.declarations.variables[train] == {"type": "bool", "default": False, "driven": "train"}
    assert "level_crossing:commuter#at_south" in quests.declarations.variables
    board = quests.quests["catch_train"]["objectives"][0]
    assert board["subjects"] == [platform]                              # the marker: the platform, not the train

    def play():
        machine = Machine(quests.program())
        machine.set_clock(time="07:50")                     # before anyone is there: it isn't open yet
        machine.add_actor("player", kind="player")
        machine.add_actor("npc:late", kind="npc")
        return machine, lambda actor="player": quests.status(machine, "catch_train", actor)

    def stand(machine, actor, on=True):
        machine.post({"type": "enter" if on else "leave", "subject": platform, "actor": actor})

    machine, status = play()                                # made it: waiting on the platform as it pulls in
    assert status() == "unavailable"
    stand(machine, "player")
    machine.set(train, True)                                # the 07:5x: before the quest opens, it doesn't count
    machine.set(train, False)
    machine.set_clock(time="08:00")
    assert status() == "active" and status("npc:late") == "unavailable"     # only the player holds it
    machine.set(train, True)
    assert status() == "complete"
    machine.set(train, False)                               # it leaves: nothing to fail
    assert status() == "complete" and machine.get("world:level_crossing.north_train_left") is True

    machine, status = play()                                # missed it: off the platform when it left
    machine.set_clock(time="08:00")
    stand(machine, "player")
    stand(machine, "player", on=False)
    machine.set(train, True)
    assert status() == "active"
    machine.set(train, False)
    assert status() == "failed"
    stand(machine, "player")
    machine.set(train, True)                                # the next one is too late
    assert status() == "failed" and machine.errors == []

    machine, status = play()                                # arriving while it stands there counts too
    machine.set_clock(time="08:00")
    machine.set(train, True)
    stand(machine, "player")
    assert status() == "complete"


def test_items_are_actor_state_given_and_taken_by_effects():
    """geogen.18.1: has("parcel") reads the actor's count of items/parcel.yaml, declared on first
    use; give / take compile to set effects (take never below none). Unknown items are errors."""
    fetch = {"id": "fetch", "title": "Fetch", "start": "auto",
             "objectives": [{"id": "get", "text": "Find the parcel", "done": 'has("parcel")'}],
             "rewards": [{"take": "parcel"}, {"take": "parcel"}, {"give": "rover", "count": 2}]}
    compiled, machine = run(fetch)
    assert compiled.items == {"parcel": {"title": "Parcel"}, "rover": {"title": "Rover"}}
    spec = compiled.declarations.spec("actor:demo.item_parcel")
    assert spec == {"type": "number", "default": 0}
    machine.set("actor:demo.item_parcel", 1, "player")
    assert machine.get("actor:demo.quest_fetch", "player") == "complete"
    assert machine.get("actor:demo.item_parcel", "player") == 0             # taken twice, held one
    assert machine.get("actor:demo.item_rover", "player") == 2
    assert machine.get("actor:demo.item_rover", "npc:ann") == 0             # one count per actor
    with pytest.raises(QuestError, match="no item 'parcle'.*parcel"):
        compile_quests([{**fetch, "rewards": [{"give": "parcle"}]}], declarations())
    with pytest.raises(QuestError, match="whole number"):
        compile_quests([{**fetch, "rewards": [{"give": "rover", "count": 0}]}], declarations())
