"""Wiring (geogen/wiring.py): the rules, wires and bindings scenes and assets write."""

import json

import pytest

from geogen.export import export_scene, manifest_path
from geogen.layout.composer import SceneComposer
from geogen.quests import QuestError, collect
from geogen.rules import Machine, check_program
from test_state import GATE, LAMP, LEVER

# A pressure plate that counts who stands on it: its own rules, relative to itself.
PLATE = """
name: plate
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
state:
  occupants: {type: number, default: 0}
parts:
  slab: {primitive: cube, size: [1.2, 0.05, 1.2], anchor: bottom_center, bevel: 0}
places:
  top: {around: slab, headroom: 2}
rules:
  - on: {enter: top}
    do: [{set: self.occupants, to: "self.occupants + 1"}]
  - id: step_off
    on: {leave: top}
    do: [{set: self.occupants, to: "max(self.occupants - 1, 0)"}]
"""

SCENE = """
name: yard
size: [12, 3, 12]
state:
  power:  {scope: world, type: bool, default: true}
  pulls:  {scope: actor, type: number, default: 0}
  dark:   {scope: world, type: bool, default: false}
quests: [quests/stand.yaml]
slots:
  a: {position: [-3, 0, 0]}
  b: {position: [0, 0, 0]}
  c: {position: [3, 0, 0]}
  d: {position: [3, 0, 3]}
  e: {position: [-3, 0, 3]}
place:
  lever:   {asset: lever.yaml, slot: a}
  gate:    {asset: gate.yaml, slot: b}
  lamp:    {asset: lamp.yaml, slot: c}
  plate:   {asset: plate.yaml, slot: d}
  plate_2: {asset: plate.yaml, slot: e}
wire:
  - {from: lever.pulled, to: gate.open}
  - {from: lever.released, to: gate.close, after: 1}
bind:
  lamp.lit: "plate.occupants > 0 and world.power"
rules:
  - id: count_pulls
    on: {use: lever, actor: player}
    do: [{set: actor.pulls, to: "actor.pulls + 1"}, {start_quest: stand}]
  - id: nightfall
    on: {clock: night}
    do: [{set: world.dark, to: true}]
  - id: crowded
    when: "plate.occupants >= 2"
    do: [{emit: crowded, of: plate}, {input: gate.lock}]
    once: true
"""

QUEST = """
kind: quest
version: 1
title: Stand in the light
objectives:
  - {id: lit, text: Light the lamp, done: "lamp.lit"}
  - {id: open, text: Open the gate, done: 'gate.gate == "open"'}
"""

ACTORS = {"player": {"kind": "player"}, "npc:ann": {"kind": "npc"}}


@pytest.fixture
def assets(tmp_path):
    for name, text in (("lever", LEVER), ("gate", GATE), ("lamp", LAMP), ("plate", PLATE), ("yard", SCENE),
                       ("quests/stand", QUEST)):
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


def test_wiring_compiles_to_rules_and_bindings(assets):
    found = collect(compose(assets), "yard")
    assert [r["id"] for r in found.wiring] == [
        "count_pulls", "nightfall", "crowded", "wire_1", "wire_2",
        "yard:plate#rule_1", "yard:plate#step_off", "yard:plate_2#rule_1", "yard:plate_2#step_off"]
    by_id = {r["id"]: r for r in found.wiring}
    # an asset's rules are each instance's own
    assert by_id["yard:plate_2#step_off"]["on"] == {"type": "leave", "subject": "yard:plate_2/top"}
    assert by_id["yard:plate_2#step_off"]["do"][0]["set"] == "yard:plate_2#occupants"
    assert by_id["wire_1"] == {"id": "wire_1", "on": {"type": "signal", "subject": "yard:lever", "name": "pulled"},
                                    "do": [{"set": "yard:gate#gate", "to": "open"}]}
    assert by_id["wire_2"]["after"] == 1.0
    assert by_id["count_pulls"]["on"] == {"type": "use", "subject": "yard:lever", "actor": "player"}
    assert by_id["crowded"]["once"] is True and "on" not in by_id["crowded"]
    assert [b["set"] for b in found.bindings] == ["yard:lamp#lit"]
    program = found.program()
    assert check_program(program) == []
    # declared signals first, then the wiring, then the quests it can give
    ids = [r["id"] for r in program["rules"]]
    assert ids.index("signal:yard:lamp#switched_on") < ids.index("count_pulls") < ids.index("quest:stand:lit")


def test_a_lever_opens_a_gate_a_plate_lights_a_lamp_and_a_rule_gives_a_quest(assets):
    """The no-player wiring archetype (geogen.3.6), in the reference evaluator."""
    found = collect(compose(assets), "yard")
    machine = Machine(found.program(), ACTORS)

    def step(kind, plate="plate", actor="npc:ann"):
        machine.post({"type": kind, "subject": f"yard:{plate}/top", "actor": actor})

    step("enter")                                       # an NPC on the plate: no player involved
    assert machine.get("yard:plate#occupants") == 1 and machine.get("yard:lamp#lit") is True
    assert ["yard:lamp", "switched_on", None] in machine.signals     # the lamp's own declared signal
    machine.set("world:yard.power", False)
    assert machine.get("yard:lamp#lit") is False
    machine.set("world:yard.power", True)
    step("leave")
    step("leave")                                       # never below zero
    assert machine.get("yard:plate#occupants") == 0 and machine.get("yard:lamp#lit") is False
    assert machine.get("yard:plate_2#occupants") == 0   # the other plate is another instance

    machine.use("yard:lever#pull", "npc:ann")
    assert machine.get("actor:yard.pulls", "npc:ann") == 0          # the rule is for the player
    machine.advance(0.6)
    assert machine.get("yard:gate#gate") == "open"
    machine.use("yard:lever#pull", "player")
    machine.advance(0.6)                                # released: the gate closes a second later
    assert machine.get("yard:gate#gate") == "open"
    machine.advance(1.0)
    assert machine.get("yard:gate#gate") == "closed"
    assert machine.get("actor:yard.pulls", "player") == 1
    assert found.status(machine, "stand", "player") == "active"     # given by the rule
    assert found.status(machine, "stand", "npc:ann") == "available"

    step("enter", actor="player")
    step("enter")
    assert ["yard:plate", "crowded", None] in machine.signals      # a watch over world state has no actor
    assert machine.get("yard:gate#gate_locked") is True
    machine.use("yard:lever#pull", "player")
    machine.advance(0.6)
    assert found.status(machine, "stand", "player") == "complete"
    machine.set_clock(night=True)
    assert machine.get("world:yard.dark") is True and machine.errors == []


def test_manifest_has_the_program(assets, tmp_path):
    path = export_scene(compose(assets), tmp_path / "out" / "yard.glb")
    manifest = json.loads(manifest_path(path).read_text())
    rules = manifest["rules"]
    assert rules["format"] == "geogen-rules" and rules["package"] == "yard"
    assert [b["set"] for b in rules["bindings"]] == ["yard:lamp#lit"]
    program = {**rules, "variables": manifest["state"]["variables"], "interactions": manifest["state"]["interactions"]}
    del program["package"]
    assert check_program(program) == []
    assert "count_pulls" not in path.read_bytes().decode("latin-1")     # not node extras


@pytest.mark.parametrize("name, old, new, message", [
    ("yard", "from: lever.pulled", "from: lever.pushed",
     r"yard.yaml:\d+: wire.0.from: 'yard:lever' has no signal 'pushed'. Did you mean 'pulled'\?"),
    ("yard", "to: gate.open", "to: gate.opne", r"yard.yaml:\d+: wire.0.to.0: 'yard:gate' has no input 'opne'"),
    ("yard", "to: gate.open}", "to: gate.open, value: 3}", r"wire.0.to.0: input 'open' sets a fixed value"),
    ("yard", "lamp.lit:", "lamp.lite:", r"yard.yaml:\d+: bind.lamp.lite: .*has no state 'lite'"),
    ("yard", '"plate.occupants > 0 and world.power"', '"actor.pulls > 0"', "bindings can't involve actor state"),
    ("yard", "on: {clock: night}", "on: {clock: noon}", r"rules.1.on: clock: night or day"),
    ("yard", "on: {clock: night}", "on: {tick: 1}", r"rules.1.on: unknown key 'tick'"),
    ("yard", "do: [{set: world.dark, to: true}]", "do: [{set: actor.pulls, to: 1}]",
     "sets actor state 'actor:yard.pulls', but nothing here says which actor"),
    ("yard", "{start_quest: stand}", "{start_quest: stnad}", r"rules.0.do.1: no quest 'stnad'. Did you mean 'stand'\?"),
    ("yard", "    once: true", "    once: true\n    often: true", r"rules.2: unknown keys \['often'\]"),
    # a rule that waits for a signal nothing raises
    ("yard", "on: {clock: night}", "on: {signal: lever.pushed}",
     r"yard.yaml:\d+: rules.1.on: 'yard:lever' never raises 'pushed'. Did you mean 'pulled'\?"),
    ("yard", "on: {clock: night}", "on: {signal: lamp.greeted}", r"rules.1.on: 'yard:lamp' never raises 'greeted'"),
    ("yard", '    when: "plate.occupants >= 2"\n', "", r"rules.2: a rule needs on: \(an event\) or when:"),
    ("plate", "on: {enter: top}", "on: {enter: tpo}", r"plate.yaml:\d+: rules.0.on: .*'tpo'"),
    ("plate", "set: self.occupants, to: \"self.occupants + 1\"", "set: self.ocupants, to: 1",
     r"plate.yaml:\d+: rules.0.do.0: .*has no state 'ocupants'"),
    # a quest objective that only wiring could cause stops being possible without it
    ("yard", "  - {from: lever.pulled, to: gate.open}\n", "",
     r"stand.yaml:\d+: objectives.1.done: objective 'open' can't be done: nothing moves 'gate.gate' to 'open'"),
])
def test_bad_wiring_fails_with_file_and_line(assets, name, old, new, message):
    rewrite(assets, name, old, new)
    with pytest.raises(QuestError, match=message):
        collect(compose(assets), "yard")


def test_if_leaves_rules_wires_and_quests_out_of_a_build(assets):
    """A scene that is also nested in bigger ones gates what it contributes on a param."""
    rewrite(assets, "yard", "name: yard\n", "name: yard\nparams:\n  quest: {default: 1}\n")
    rewrite(assets, "yard", "quests: [quests/stand.yaml]", 'quests: [{file: quests/stand.yaml, if: "{quest}"}]')
    rewrite(assets, "yard", "    on: {use: lever, actor: player}\n    do: [{set: actor.pulls, to: \"actor.pulls + 1\"}, "
                            "{start_quest: stand}]",
            '    if: "{quest}"\n    on: {use: lever, actor: player}\n    do: [{start_quest: stand}]')
    rewrite(assets, "yard", "{from: lever.released, to: gate.close, after: 1}",
            '{from: lever.released, to: gate.close, after: 1, if: "{quest}"}')
    path = assets / "yard.yaml"
    with_quest = collect(SceneComposer(assets_dir=assets).compose(path), "yard")
    assert list(with_quest.quests) == ["stand"]
    assert [r["id"] for r in with_quest.wiring[:4]] == ["count_pulls", "nightfall", "crowded", "wire_1"]
    assert "wire_2" in [r["id"] for r in with_quest.wiring]
    without = collect(SceneComposer(assets_dir=assets).compose(path, params={"quest": 0}), "yard")
    assert not without.quests and not [r for r in without.rules if r["id"].startswith("quest:")]
    ids = [r["id"] for r in without.wiring]
    assert "count_pulls" not in ids and "wire_2" not in ids and {"nightfall", "crowded", "wire_1"} <= set(ids)


GUARDED = ("  pull:\n    duration: 0.5\n",
           "  pull:\n    duration: 0.5\n    when: \"world.power and not self.handle_stuck\"\n    refused: No power\n")
STUCK = "bounds: geometry\n", "bounds: geometry\nstate:\n  handle_stuck: {type: bool, default: false}\n"


def test_an_interaction_only_works_when_its_condition_holds(assets, tmp_path):
    rewrite(assets, "lever", *GUARDED)
    rewrite(assets, "lever", *STUCK)
    found = collect(compose(assets), "yard")
    when = found.declarations.interactions["yard:lever#pull"]["when"]
    assert when == {"op": "and", "args": [{"get": "world", "package": "yard", "name": "power"},
                                          {"op": "not", "args": [{"get": "instance", "id": "yard:lever", "name": "handle_stuck"}]}]}
    machine = Machine(found.program(), ACTORS)
    machine.set("world:yard.power", False)
    assert machine.use("yard:lever#pull", "player") is False and machine.get("yard:lever#pull") == "up"
    machine.set("world:yard.power", True)
    assert machine.use("yard:lever#pull", "player") is True
    machine.set("yard:lever#handle_stuck", True)
    machine.advance(1.0)
    assert machine.use("yard:lever#pull", "player") is False and machine.errors == []

    machine.set("yard:lever#handle_stuck", False)
    machine.set("world:yard.power", False)
    machine.feedback.clear()
    assert machine.use("yard:lever#pull", "player") is False
    assert machine.feedback == [["notify", "No power", "player"], ["play", "refused", "player"]]

    manifest = json.loads(manifest_path(export_scene(compose(assets), tmp_path / "yard.glb")).read_text())
    assert manifest["state"]["interactions"]["yard:lever#pull"]["when"] == when
    assert manifest["state"]["interactions"]["yard:lever#pull"]["refused"] == "No power"


def test_rules_notify_and_play(assets):
    rewrite(assets, "yard", "do: [{set: world.dark, to: true}]",
            'do: [{set: world.dark, to: true}, {notify: "Night falls"}, {play: notice}]')
    found = collect(compose(assets), "yard")
    machine = Machine(found.program(), ACTORS)
    machine.set_clock(night=True)
    assert machine.feedback == [["notify", "Night falls", None], ["play", "notice", None]]


@pytest.mark.parametrize("old, new, message", [
    ("do: [{set: world.dark, to: true}]", "do: [{play: gong}]", r"rules.1.do.0: play: no sound 'gong'"),
    ("do: [{set: world.dark, to: true}]", 'do: [{notify: ""}]', r"rules.1.do.0: notify takes the text"),
])
def test_bad_feedback_fails_with_file_and_line(assets, old, new, message):
    rewrite(assets, "yard", old, new)
    with pytest.raises(QuestError, match=message):
        collect(compose(assets), "yard")


def test_refused_needs_a_when(assets):
    rewrite(assets, "lever", "  pull:\n    duration: 0.5\n", "  pull:\n    duration: 0.5\n    refused: Stuck\n")
    with pytest.raises(ValueError, match="refused is the text shown while its `when` is false"):
        compose(assets)


def test_a_bad_interaction_condition_names_the_interaction(assets):
    rewrite(assets, "lever", GUARDED[0], GUARDED[0] + '    when: "self.handle_stuk"\n')
    with pytest.raises(QuestError, match=r"yard:lever#pull: when: .*has no state 'handle_stuk'"):
        collect(compose(assets), "yard")


def test_godot_refuses_an_interaction_while_its_condition_is_false(assets, tmp_path, godot_runtime):
    rewrite(assets, "lever", *GUARDED)
    rewrite(assets, "lever", *STUCK)
    export_scene(compose(assets), tmp_path / "yard.glb")
    rt = godot_runtime("yard", tmp_path)
    rt.call("state.set", key="world:yard.power", value=False)
    lever = rt.call("interactions.list", asset="lever")[0]
    assert lever["prompt"] == "" and lever["refused"] == "No power"
    rt.call("interactions.use", asset="lever", actor="player", wait=True)
    assert rt.call("state.get", key="yard:lever#pull")["value"] == "up"
    rt.call("time.advance", seconds=0.2)
    toasts = rt.call("hud.notifications")
    assert [(t["kind"], t["text"]) for t in toasts["showing"] + toasts["queued"]] == [("notice", "No power")]
    assert toasts["sounds"] == ["refused"]
    rt.call("state.set", key="world:yard.power", value=True)
    rt.call("interactions.use", asset="lever", wait=True)
    assert rt.call("state.get", key="yard:lever#pull")["value"] == "down"


# A puzzle as a scene: instance state, a signal and an input of its own (geogen.40.4).
PUZZLE = """
name: puzzle
size: [4, 3, 4]
state:
  solved: {scope: instance, type: bool, default: false}
  tries:  {scope: world, type: number, default: 0}
signals:
  done:  {changed: solved, to: true}
  moved: {emit: pulled, of: lever}
inputs:
  reset: {set: solved, to: false}
place:
  lever: {asset: lever.yaml}
rules:
  - on: {signal: lever.pulled}
    do: [{set: self.solved, to: true}, {set: world.tries, to: "world.tries + 1"}]
"""

PUZZLES = """
name: puzzles
size: [12, 3, 12]
slots:
  a: {position: [-3, 0, 0]}
  b: {position: [3, 0, 0]}
  c: {position: [0, 0, 4]}
place:
  puzzle_a: {scene: puzzle.yaml, slot: a}
  puzzle_b: {scene: puzzle.yaml, slot: b, state: {solved: true}}
  gate:     {asset: gate.yaml, slot: c}
wire:
  - {from: puzzle_a.done, to: gate.open}
  - {from: gate.opened, to: puzzle_b.reset}
"""


@pytest.fixture
def puzzles(assets):
    (assets / "puzzle.yaml").write_text(PUZZLE)
    (assets / "puzzles.yaml").write_text(PUZZLES)
    return assets


def test_a_scene_placed_twice_is_two_puzzles(puzzles):
    found = collect(SceneComposer(assets_dir=puzzles).compose(puzzles / "puzzles.yaml"), "puzzles")
    decls = found.declarations
    assert decls.variables["puzzles:puzzle_a#solved"] == {"type": "bool", "default": False}
    assert decls.variables["puzzles:puzzle_b#solved"]["default"] is True     # the placement's state:
    assert "world:puzzles.tries" in decls.variables                          # world state is still shared
    assert decls.inputs["puzzles:puzzle_b#reset"] == {"set": "puzzles:puzzle_b#solved", "to": False}
    machine = Machine(found.program(), ACTORS)

    machine.use("puzzles:puzzle_a/lever#pull", "player")
    machine.advance(0.6)
    assert machine.get("puzzles:puzzle_a#solved") is True                    # `self` is this copy
    assert ["puzzles:puzzle_a", "done", "player"] in machine.signals          # with who solved it
    assert ["puzzles:puzzle_a", "moved", "player"] in machine.signals
    assert machine.get("puzzles:gate#gate") == "open"                        # the parent wired its signal
    assert machine.get("puzzles:puzzle_b#solved") is True
    machine.advance(2.1)                                                     # the gate arrives: b is reset
    assert machine.get("puzzles:puzzle_b#solved") is False
    assert machine.get("world:puzzles.tries") == 1 and machine.errors == []


def test_a_scene_exported_alone_has_its_own_instance_state(puzzles):
    found = collect(SceneComposer(assets_dir=puzzles).compose(puzzles / "puzzle.yaml"), "puzzle")
    machine = Machine(found.program(), ACTORS)
    machine.use("puzzle:lever#pull", "player")
    machine.advance(0.6)
    assert machine.get("puzzle:#solved") is True and machine.errors == []


@pytest.mark.parametrize("old, new, message", [
    ("scope: instance", "scope: placed", "or instance \\(one per placed copy"),
    ("{changed: solved, to: true}", "{changed: solve, to: true}", "has no state 'solve'"),
])
def test_bad_scene_declarations_fail(puzzles, old, new, message):
    from geogen.state import StateError

    rewrite(puzzles, "puzzle", old, new)
    with pytest.raises((StateError, QuestError), match=message):
        collect(SceneComposer(assets_dir=puzzles).compose(puzzles / "puzzles.yaml"), "puzzles")


def test_exported_puzzles_validate(puzzles, tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    from pathlib import Path

    schema = json.loads((Path(__file__).parent.parent / "docs/schema/geogen-state.v1.schema.json").read_text())
    for name in ("puzzles", "puzzle"):
        root = SceneComposer(assets_dir=puzzles).compose(puzzles / f"{name}.yaml")
        manifest = json.loads(manifest_path(export_scene(root, tmp_path / name / f"{name}.glb")).read_text())
        jsonschema.validate(manifest["state"], schema)
    assert "puzzle:#solved" in manifest["state"]["variables"]


def test_reset_puts_a_puzzle_back_as_it_started(puzzles):
    """reset: a thing (and what is in it) back to its defaults; reset_quest: a quest (geogen.40.5)."""
    rewrite(puzzles, "puzzles", "wire:\n", """rules:
  - id: retry
    on: {signal: puzzle_b.done}
    after: 1
    do: [{reset: puzzle_a}, {reset_quest: stand}]
  - id: everything
    on: {clock: night}
    do: [{reset: "[tag=gate]"}]
quests: [quests/stand.yaml]
wire:
""")
    (puzzles / "puzzles.yaml").write_text((puzzles / "puzzles.yaml").read_text().replace(
        "  gate:     {asset: gate.yaml, slot: c}", "  gate:     {asset: gate.yaml, slot: c, tags: [gate]}\n"
        "  lamp:     {asset: lamp.yaml, slot: c}\nbind:\n  lamp.lit: \"puzzle_a.solved\""))
    found = collect(SceneComposer(assets_dir=puzzles).compose(puzzles / "puzzles.yaml"), "puzzles")
    retry = next(r for r in found.wiring if r["id"] == "retry")
    keys = [e["set"] for e in retry["do"]]
    assert {"puzzles:puzzle_a#solved", "puzzles:puzzle_a/lever#pull", "puzzles:puzzle_a/lever#pull_locked"} <= set(keys)
    assert "puzzles:puzzle_b#solved" not in keys and "actor:puzzles.quest_stand" in keys
    assert all("_reset" not in e for r in found.wiring for e in r["do"])

    machine = Machine(found.program(), ACTORS)
    machine.set("actor:puzzles.quest_stand", "active", "player")
    machine.use("puzzles:puzzle_a/lever#pull", "player")
    machine.advance(0.6)
    assert machine.get("puzzles:puzzle_a#solved") is True and machine.get("puzzles:lamp#lit") is True
    machine.advance(2.1)                                                     # the gate opens: b is reset
    assert machine.get("puzzles:puzzle_b#solved") is False
    machine.set("puzzles:puzzle_b#solved", True, "player")                  # b done: a is reset a second later
    machine.advance(1.0)
    assert machine.get("puzzles:puzzle_a#solved") is False and machine.get("puzzles:puzzle_a/lever#pull") == "up"
    assert machine.get("puzzles:lamp#lit") is False                          # bound: it follows, never set
    assert machine.get("actor:puzzles.quest_stand", "player") == "available"
    machine.set("puzzles:gate#gate", "open")
    machine.set_clock(night=True)
    assert machine.get("puzzles:gate#gate") == "closed" and machine.errors == []


def test_rules_read_event_data_and_emit_it(assets):
    """event.<field> reads what the event carries; emit gives a signal data a rule can read."""
    rewrite(assets, "yard", "    once: true\n", "    once: true\n"
            "  - id: shout\n    on: {signal: lever.pulled}\n"
            "    do: [{emit: rang, of: lamp, data: {loud: 3, twice: \"actor.pulls * 2\"}}]\n"
            "  - id: hear\n    on: {signal: lamp.rang}\n    when: \"event.loud > 2\"\n"
            "    do: [{set: actor.pulls, to: \"event.twice + event.loud\"}]\n"
            "  - id: dimmed\n    on: {changed: lamp.lit}\n    do: [{set: world.dark, to: \"not event.to\"}]\n")
    found = collect(compose(assets), "yard")
    by_id = {r["id"]: r for r in found.wiring}
    assert by_id["shout"]["do"][0]["data"]["loud"] == 3
    machine = Machine(found.program(), ACTORS)
    machine.use("yard:lever#pull", "player")
    machine.advance(2)
    assert machine.get("actor:yard.pulls", "player") == 1 * 2 + 3     # pulls was 1 when the lever signalled
    assert machine.errors == []


@pytest.mark.parametrize("old, new, message", [
    # a field the event never carries
    ("do: [{set: world.dark, to: true}]", 'do: [{set: world.dark, to: "event.to"}]',
     r"rules.1: event.to: a clock event here carries no data"),
    ('"plate.occupants >= 2"', '"event.value >= 2"', r"rules.2: event.value: only a rule with on: \(an event\)"),
    ("to: gate.open}", "to: gate.open, when: \"event.speed > 1\"}", r"wire.0: event.speed: a signal event here carries no data"),
])
def test_bad_event_reads_fail_with_file_and_line(assets, old, new, message):
    rewrite(assets, "yard", old, new)
    with pytest.raises(QuestError, match=message):
        collect(compose(assets), "yard")
