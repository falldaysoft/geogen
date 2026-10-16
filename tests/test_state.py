"""State declarations (geogen/state.py): instance, world and actor state, asset signals and inputs."""

import json
from pathlib import Path

import pytest

from geogen import conditions
from geogen.chunks import export_chunks
from geogen.export import export_scene, manifest_path
from geogen.layout.composer import SceneComposer
from geogen.rules import Machine
from geogen.state import Declarations, StateError, collect

SCHEMA = Path(__file__).parent.parent / "docs/schema/geogen-state.v1.schema.json"

# A lever: instance state is its `pull` interaction; it offers two signals.
LEVER = """
name: lever
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
parts:
  base:   {primitive: cube, size: [0.3, 0.1, 0.3], anchor: bottom_center}
  handle: {primitive: cube, size: [0.05, 0.6, 0.05], anchor: bottom_center, offset: [0, 0.1, 0]}
interactions:
  pull:
    duration: 0.5
    states:
      up:   {next: down, emit: released}
      down: {next: up, emit: pulled}
    motions:
      - {parts: [handle], rotate: x, pivot: [0, 0.1, 0], values: {down: 60}}
signals:
  pulled:   {emit: pulled}
  released: {emit: released}
"""

# A gate only rules can move, with inputs to open and close it.
GATE = """
name: gate
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
parts:
  leaf: {primitive: cube, size: [2, 2, 0.1], anchor: bottom_center}
interactions:
  gate:
    targets: [leaf]
    duration: 2
    lock: {key: gate_key, locked: false}
    states:
      closed: {}
      open:   {emit: opened}
    motions:
      - {parts: [leaf], translate: y, values: {open: 2}}
signals:
  opened: {emit: opened}
  moved:  {changed: gate}
inputs:
  open:   {set: gate, to: open}
  close:  {set: gate, to: closed}
  lock:   {set: gate_locked, to: true}
"""

# A lamp with declared instance state, a signal off it and an input that takes a value.
LAMP = """
name: lamp
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
params:
  lit: {default: 0}
state:
  lit:    {type: bool, default: false}
  colour: {type: choice, choices: [warm, cool]}
  level:  {type: number, default: "{lit + 2}"}
  label:  {type: text, default: lamp}
parts:
  post: {primitive: cube, size: [0.1, 2, 0.1], anchor: bottom_center}
signals:
  switched_on: {changed: lit, to: true}
inputs:
  light:  {set: lit}
  colour: {set: colour}
"""

SCENE = """
name: wiring
size: [12, 3, 12]
state:
  power:      {scope: world, type: bool, default: true}
  visits:     {scope: actor, type: number, default: 0}
  has_ticket: {scope: actor, type: bool, default: false}
slots:
  a: {position: [-3, 0, 0]}
  b: {position: [0, 0, 0]}
  c: {position: [3, 0, 0]}
  d: {position: [3, 0, 3]}
place:
  lever:  {asset: lever.yaml, slot: a}
  gate:   {asset: gate.yaml, slot: b}
  lamp:   {asset: lamp.yaml, slot: c}
  lamp_2: {asset: lamp.yaml, slot: d, state: {lit: true, colour: cool}}
"""


@pytest.fixture
def assets(tmp_path):
    for name, text in (("lever", LEVER), ("gate", GATE), ("lamp", LAMP), ("wiring", SCENE)):
        (tmp_path / f"{name}.yaml").write_text(text)
    return tmp_path


def compose(assets, name="wiring"):
    return SceneComposer(assets_dir=assets).compose(assets / f"{name}.yaml")


def rewrite(assets, name, old, new):
    path = assets / f"{name}.yaml"
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def test_declarations_have_explicit_owners(assets):
    found = collect(compose(assets), "wiring")
    assert found.variables == {
        "world:wiring.power": {"type": "bool", "default": True},
        "actor:wiring.visits": {"type": "number", "default": 0},
        "actor:wiring.has_ticket": {"type": "bool", "default": False},
        "wiring:lamp#lit": {"type": "bool", "default": False},
        "wiring:lamp#colour": {"type": "choice", "choices": ["warm", "cool"], "default": "warm"},
        "wiring:lamp#level": {"type": "number", "default": 2},
        "wiring:lamp#label": {"type": "text", "default": "lamp"},
        # a placement's state: starts that instance elsewhere
        "wiring:lamp_2#lit": {"type": "bool", "default": True},
        "wiring:lamp_2#colour": {"type": "choice", "choices": ["warm", "cool"], "default": "cool"},
        "wiring:lamp_2#level": {"type": "number", "default": 2},
        "wiring:lamp_2#label": {"type": "text", "default": "lamp"},
    }
    # interactions are instance state too
    assert found.interactions["wiring:lever#pull"] == {
        "initial": "up", "duration": 0.5,
        "states": {"up": {"next": "down", "emit": "released"}, "down": {"next": "up", "emit": "pulled"}}}
    assert found.spec("wiring:gate#gate") == {"type": "choice", "choices": ["closed", "open"], "default": "closed"}
    assert found.spec("wiring:gate#gate_locked") == {"type": "bool", "default": False}
    assert sorted(found.names("wiring:gate")) == ["gate", "gate_locked"]


def test_signals_and_inputs_resolve_to_ids(assets):
    found = collect(compose(assets), "wiring")
    # an interaction's own emit needs no source; anything else says what raises it
    assert found.signal("wiring:lever", "pulled") == {"subject": "wiring:lever", "name": "pulled"}
    assert found.signal("wiring:gate", "moved") == {
        "subject": "wiring:gate", "name": "moved", "from": {"type": "changed", "key": "wiring:gate#gate"}}
    assert found.signal("wiring:lamp_2", "switched_on")["from"] == {
        "type": "changed", "key": "wiring:lamp_2#lit", "to": True}
    assert found.input("wiring:gate", "open") == {"set": "wiring:gate#gate", "to": "open"}
    assert found.input("wiring:gate", "lock") == {"set": "wiring:gate#gate_locked", "to": True}
    assert found.input("wiring:lamp", "light") == {"set": "wiring:lamp#lit"}      # the driver gives the value
    with pytest.raises(StateError, match=r"no signal 'pull'. Did you mean 'pulled'\? Its signals: pulled, released"):
        found.signal("wiring:lever", "pull")
    with pytest.raises(StateError, match="declares no inputs"):
        found.input("wiring:lever", "open")


def test_manifest_lists_declarations_and_matches_schema(assets, tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    out = tmp_path / "out"
    path = export_scene(compose(assets), out / "wiring.glb")
    state = json.loads(manifest_path(path).read_text())["state"]
    jsonschema.validate(state, json.loads(SCHEMA.read_text()))
    assert state["format"] == "geogen-state" and state["package"] == "wiring"
    assert {"world:wiring.power", "actor:wiring.visits", "wiring:lamp#lit"} <= set(state["variables"])
    assert set(state["interactions"]) == {"wiring:lever#pull", "wiring:gate#gate"}
    assert "wiring:lever#pulled" in state["signals"] and "wiring:gate#open" in state["inputs"]
    # the declarations themselves aren't node extras
    assert "declares" not in path.read_bytes().decode("latin-1")
    # another package's declarations can be read back from its manifest
    assert Declarations.from_json(state).input("wiring:gate", "close") == {"set": "wiring:gate#gate", "to": "closed"}

    index = json.loads(export_chunks(compose(assets), out / "chunks", name="wiring").read_text())
    assert index["state"] == state


def test_scene_without_declarations_still_lists_its_interactions(tmp_path):
    scene = SceneComposer().compose_string("name: s\nsize: [4, 3, 4]\nplace:\n  door: {asset: door.yaml}\n")
    found = collect(scene, "s")
    assert found.spec("s:door#swing")["choices"] == ["closed", "open"]
    assert not found.variables and not found.signals


def test_declarations_run_as_a_rules_program(assets):
    """Wiring by hand (geogen.3.6 compiles it): a signal's rule drives an input, and a declared
    signal with a source is raised by the machine."""
    found = collect(compose(assets), "wiring")
    lever, gate = found.signal("wiring:lever", "pulled"), found.input("wiring:gate", "open")
    machine = Machine(found.program(rules=[
        {"id": "open_gate", "on": {"type": "signal", "subject": lever["subject"], "name": lever["name"]},
         "do": [gate]}]))
    machine.use("wiring:lever#pull")
    machine.advance(0.6)
    assert machine.get("wiring:gate#gate") == "open"
    assert ["wiring:gate", "moved", None] in machine.signals
    machine.set("wiring:lamp#lit", True)
    assert ["wiring:lamp", "switched_on", None] in machine.signals


def test_conditions_see_declared_state(assets):
    scene = compose(assets)
    found = collect(scene, "wiring")
    tree = conditions.resolve(conditions.compile_condition(
        "lamp.lit and gate.gate_locked == false and world.power and actor.visits > 2"), scene, "wiring")
    assert found.check_condition(tree) == []
    with pytest.raises(conditions.ConditionError, match="has no state 'lite' Did you mean 'lit'"):
        conditions.resolve(conditions.compile_condition("lamp.lite"), scene, "wiring")
    wrong = conditions.resolve(conditions.compile_condition("world.visits > 2 or actor.tickets"), scene, "wiring")
    assert found.check_condition(wrong) == [
        "'visits' is actor state, not world state: write actor.visits",
        "no actor state 'tickets' is declared in 'wiring'. Did you mean 'has_ticket'? Declared: has_ticket, visits"]


@pytest.mark.parametrize("name, old, new, message", [
    ("lever", "pulled:   {emit: pulled}", "pulled:   {emit: pushed}",
     r"lever.yaml:18: signals.pulled: no interaction of this asset emits 'pushed'. Did you mean 'pulled'\? "
     r"It emits: pulled, released"),
    ("gate", "open:   {set: gate, to: open}", "open:   {set: gates, to: open}",
     r"gate.yaml:22: inputs.open: this asset has no state 'gates'. Did you mean 'gate'\? It has: gate, gate_locked"),
    ("gate", "open:   {set: gate, to: open}", "open:   {set: gate, to: ajar}",
     r"inputs.open: 'gate' is a choice of closed, open: can't be 'ajar'"),
    ("lamp", "switched_on: {changed: lit, to: true}", "switched_on: {changed: power}",
     r"signals.switched_on: this asset has no state 'power'.*'power' is world state, which belongs to the scene"),
    ("lamp", "light:  {set: lit}", "light:  {set: lit, of: bulb}",
     r"inputs.light: 'bulb': no 'bulb' in"),
    ("lamp", "light:  {set: lit}", "light:  {set: lit, of: post}", r"inputs.light: 'post' is a plain part"),
    ("lamp", "lit:    {type: bool, default: false}", "lit:    {type: bool, default: false}\n  post_2: {type: bool, default: 3}",
     r"lamp.yaml:\d+: state.post_2: default 3 isn't a bool"),
    ("lamp", "lit:    {type: bool, default: false}", "lit:    {type: string, default: x}",
     r"state.lit: type must be one of bool, number, text, choice, got 'string'. Did you mean 'text'\?"),
    ("lamp", "lit:    {type: bool, default: false}", "lit:    {type: bool, default: false, scope: instance}",
     r"state.lit: an asset's state belongs to each placed instance, or is world or actor state it uses"),
    ("lamp", "light:  {set: lit}", "light:  {sets: lit}", r"inputs.light: unknown key 'sets'. Did you mean 'set'\?"),
    ("lamp", "switched_on: {changed: lit, to: true}", "switched_on: {changed: lit, emit: x}", "a signal has one of"),
    ("wiring", "power:      {scope: world, type: bool, default: true}", "power:      {type: bool, default: true}",
     r"wiring.yaml:5: state.power: a scene's state needs scope: world"),
    ("wiring", "state: {lit: true, colour: cool}", "state: {lit: true, colour: blue}",
     r"state of lamp.yaml: 'colour' is a choice of warm, cool: can't be 'blue'"),
])
def test_bad_declarations_fail_the_export(assets, tmp_path, name, old, new, message):
    rewrite(assets, name, old, new)
    with pytest.raises(ValueError, match=message):
        export_scene(compose(assets), tmp_path / "out" / "wiring.glb")
    assert not (tmp_path / "out" / "wiring.glb").exists()


def test_every_problem_is_reported_once(assets):
    rewrite(assets, "lamp", "light:  {set: lit}", "light:  {set: lite}")
    rewrite(assets, "lever", "pulled:   {emit: pulled}", "pulled:   {emit: pushed}")
    with pytest.raises(StateError) as error:
        collect(compose(assets), "wiring")
    lines = str(error.value).splitlines()
    assert len(lines) == 2                      # two lamps, one mistake
    assert "signals.pulled" in lines[0] and "inputs.light" in lines[1]


def test_nested_scenes_share_the_package(assets):
    (assets / "outer.yaml").write_text("""
name: outer
size: [30, 3, 30]
state:
  power: {scope: world, type: bool, default: true}
place:
  inner: {scene: wiring.yaml}
""")
    found = collect(compose(assets, "outer"), "outer")
    assert "world:outer.power" in found.variables and "actor:outer.visits" in found.variables
    assert "outer:inner/lamp#lit" in found.variables
    rewrite(assets, "outer", "{scope: world, type: bool, default: true}", "{scope: actor, type: bool, default: true}")
    with pytest.raises(StateError, match=r"wiring.yaml:5: state.power: 'power' is world state here and actor state "
                                         r"at .*outer.yaml:5: state.power"):
        collect(compose(assets, "outer"), "outer")


def test_an_asset_may_use_the_packages_state(assets):
    """An asset declaring world state it uses builds alone; placed, it is the scene's variable
    (the scene's default wins), and a different type is a clash."""
    rewrite(assets, "lamp", "lit:    {type: bool, default: false}",
            "lit:    {type: bool, default: false}\n  power: {scope: world, type: bool, default: false}")
    found = collect(compose(assets), "wiring")
    assert found.variables["world:wiring.power"] == {"type": "bool", "default": True}
    rewrite(assets, "lamp", "power: {scope: world, type: bool, default: false}",
            "power: {scope: world, type: number, default: 0}")
    with pytest.raises(StateError, match=r"state.power: world state 'power' is declared differently"):
        collect(compose(assets), "wiring")


def test_declare_adds_variables():
    found = Declarations("demo")
    found.declare("actor:demo.quest_status", {"type": "choice", "choices": ["available", "active"], "default": "available"})
    with pytest.raises(StateError, match="already declared"):
        found.declare("actor:demo.quest_status", {"type": "bool", "default": False})
    with pytest.raises(StateError, match="needs a type"):
        found.declare("world:demo.x", {"type": "bool", "default": 1})
