"""What several test modules share (import as ``from helpers import ...``; tests/ is on sys.path)."""

import json
import struct
from pathlib import Path

ASSETS = Path(__file__).parent.parent / "assets"


def gltf_json(path) -> dict:
    """The JSON chunk of a ``.glb``."""
    data = Path(path).read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


_validators: dict[str, object] = {}


def schema_validate(instance, schema: dict) -> None:
    """``jsonschema.validate(instance, schema)``, with the validator kept per schema.

    ``jsonschema.validate`` checks the schema itself against its metaschema on every call, which
    for the extras schema costs over a second: a loop over a scene's nodes spent minutes on it."""
    import jsonschema

    key = json.dumps(schema, sort_keys=True)
    validator = _validators.get(key)
    if validator is None:
        cls = jsonschema.validators.validator_for(schema)
        cls.check_schema(schema)
        validator = _validators[key] = cls(schema)
    error = jsonschema.exceptions.best_match(validator.iter_errors(instance))
    if error is not None:
        raise error


def mesh_volume(mesh) -> float:
    """The signed volume a closed mesh encloses."""
    import numpy as np

    v = mesh.vertices[mesh.faces]
    return np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6


# ---- YAML fixtures: small assets and scenes that several modules build on --------------------------

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

PLACES_PLATE = """
name: plate
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
parts:
  slab: {primitive: cube, size: [1.2, 0.05, 1.2], anchor: bottom_center, bevel: 0}
places:
  top: {around: slab, headroom: 2}
signals:
  occupied: {enter: top}
  vacated:  {leave: top}
"""

# A freestanding arch: two posts, a lintel, and a walk-in trigger between them.
PORTAL = """
name: portal_test
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
travel: {on: enter, prompt: Step through}
parts:
  left_post:  {primitive: cube, size: [0.2, 2.4, 0.3], anchor: bottom_center, offset: [-0.7, 0, 0]}
  right_post: {primitive: cube, size: [0.2, 2.4, 0.3], anchor: bottom_center, offset: [0.7, 0, 0]}
  lintel:     {primitive: cube, size: [1.6, 0.2, 0.3], anchor: bottom_center, offset: [0, 2.4, 0]}
  travel_volume: {primitive: cube, size: [1.2, 2.2, 0.3], anchor: bottom_center}
"""

SMALL = """
name: small_building
building:
  storeys:
    - floorplan: { generate: hotel_lobby, length: 30, depth: 15, wall_height: 3.6, finishes: false }
    - floorplan: { generate: hotel_corridor, length: 30, depth: 15, finishes: false }
      repeat: 2
  roof: { parapet: 1.0 }
"""

STRIP = """
name: strip
city:
  seed: 3
  blocks: [5, 1]
  block_size: [30, 32]
  avenues: { ew: [0] }
  lot_width: [11, 14]
  buildings:
    residential: [{ recipe: detached_house, storeys: 1, interior: full }]
    commercial: [{ recipe: shop_row, storeys: 2, interior: lobby }]
spawns:
  start: { position: [-75, 0, -20], facing: east }
"""

# ---- the HUD tests' camp ------------------------------------------------------------------------

# Three levers, a gate and a plate. `levers` starts with the game and counts the levers pulled;
# `errand` is given by using the first lever and fails when the alarm goes.
HUD_CAMP = """
name: camp
size: [12, 3, 12]
state:
  alarm: {scope: world, type: bool, default: false}
quests: [quests/levers.yaml, quests/errand.yaml]
slots:
  a: {position: [-3, 0, 0]}
  b: {position: [0, 0, 0]}
  c: {position: [3, 0, 0]}
  d: {position: [0, 0, -3]}
  e: {position: [3, 0, 3]}
place:
  lever:   {asset: lever.yaml, slot: a, tags: [lever]}
  lever_2: {asset: lever.yaml, slot: b, tags: [lever]}
  lever_3: {asset: lever.yaml, slot: c, tags: [lever]}
  gate:    {asset: gate.yaml, slot: d}
  plate:   {asset: plate.yaml, slot: e}
wire:
  - {from: lever_3.released, to: gate.open}
rules:
  - id: give_errand
    on: {use: lever, actor: player}
    do: [{start_quest: errand}]
spawns:
  start: {position: [0, 0, 5], facing: south}
"""

HUD_LEVERS = """
kind: quest
version: 1
title: Throw the switches
start: auto
objectives:
  - {id: all, text: Pull every lever, done: 'all("[tag=lever]", it.pull == "down")'}
  - {id: gate, text: Open the gate, after: all, done: 'gate.gate == "open"'}
  - {id: plate, text: Stand on the plate, optional: true, done: {enter: plate/top}}
"""

HUD_ERRAND = """
kind: quest
version: 1
title: An errand with a title long enough that it has to wrap in the tracker
fail: "world.alarm"
objectives:
  - {id: stand, text: Stand on the pressure plate in the far corner of the camp, done: {enter: plate/top}}
"""

HUD_LONG_TITLE = "An errand with a title long enough that it has to wrap in the tracker"

# ---- the content package tests' two scenes and their package ---------------------------------------

# Two scenes that know nothing of each other: a lever in one, a gate, a lamp and a plate in the other.
CONTENT_CAMP = """
name: camp
size: [8, 3, 8]
state:
  fire: {scope: world, type: bool, default: true}
slots:
  a: {position: [0, 0, 2]}
place:
  lever: {asset: lever.yaml, slot: a}
spawns:
  start: {position: [0, 0, 3.2], facing: north}
"""

CONTENT_ANNEX = """
name: annex
size: [10, 3, 10]
slots:
  a: {position: [0, 0, -3]}
  b: {position: [3, 0, 0]}
  c: {position: [-3, 0, 0]}
place:
  gate:  {asset: gate.yaml, slot: a}
  lamp:  {asset: lamp.yaml, slot: b}
  plate: {asset: plate.yaml, slot: c}
spawns:
  start: {position: [0, 0, 3], facing: south}
"""

# The package: pulling the camp's lever is remembered, and opens the annex's gate and lights its lamp.
CONTENT_ERRAND = """
kind: package
version: 1
title: The errand
description: Pull the lever at camp, then stand on the plate in the annex.
scenes: [camp, annex]
state:
  lever_pulled: {scope: world, type: bool, default: false}
  pulls:        {scope: actor, type: number, default: 0}
rules:
  - id: note_pull
    on: {signal: "camp:lever.pulled"}
    do: [{set: world.lever_pulled, to: true}]
  - id: count_pull
    on: {use: "camp:lever", actor: player}
    when: 'world("camp").fire'
    do: [{set: actor.pulls, to: "actor.pulls + 1"}]
  - id: open_gate
    when: "world.lever_pulled"
    do: [{input: "annex:gate.open"}]
bind:
  'ref("annex:lamp").lit': "world.lever_pulled"
quests: [quests/errand.yaml]
"""

CONTENT_QUEST = """
kind: quest
version: 1
title: Run the errand
start: auto
ordered: true
objectives:
  - {id: pull, text: Pull the lever at camp, done: {signal: "camp:lever.pulled"}}
  - {id: plate, text: Stand on the plate in the annex, done: {enter: "annex:plate/top"}}
"""

CONTENT_FILES = (("lever", LEVER), ("gate", GATE), ("lamp", LAMP), ("plate", PLATE), ("camp", CONTENT_CAMP), ("annex", CONTENT_ANNEX),
         ("packages/errand", CONTENT_ERRAND), ("quests/errand", CONTENT_QUEST))


def write_assets(assets):
    """Write the content tests' files into the directory ``assets``."""
    for name, text in CONTENT_FILES:
        path = assets / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    return assets


def deploy(assets, out, packages=("errand",)):
    """Export both scenes and the packages into ``out``, as --export-catalogue does."""
    from geogen.catalogue import Catalogue, CatalogueEntry, export_catalogue
    from geogen.layout.composer import SceneComposer

    catalogue = Catalogue("camp", {"camp": CatalogueEntry("camp"), "annex": CatalogueEntry("annex")},
                          {name: assets / "packages" / f"{name}.yaml" for name in packages})
    export_catalogue(catalogue, lambda name: SceneComposer(assets_dir=assets).compose(assets / f"{name}.yaml"),
                     out, log=lambda *_: None)
    return catalogue
