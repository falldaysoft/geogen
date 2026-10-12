# State, events, rules and bindings

This is the contract for the layer under object wiring, quests, dialogue and
inventory. It is declarative: content is data, and a runtime is anything that
obeys the rules below. `geogen/rules.py` is the reference implementation
(`Machine`), and the traces in `tests/data/rules/` say what must happen for
given inputs. The Godot runtime is one implementation
(`runtime/godot/scripts/machine.gd`) and passes the same traces
(`tests/test_rules_godot.py`).

This document covers what is exported and how it behaves; "Authoring" below is
the YAML that compiles to it.

## State

A variable has a type (`bool`, `number`, `text`, or `choice` with `choices`), a
default, and exactly one owner. Its key says which:

| Key | Owner | Example |
|---|---|---|
| `<thing id>#<name>` | one placed instance, by stable ID | `cottage:house/lamp#lit` |
| `world:<package>.<name>` | the package: one shared value | `world:town.bridge_down` |
| `actor:<package>.<name>` | each actor: one value per player or NPC | `actor:town.has_ticket` |

- There is no default owner. Content always says which it means.
- An **actor** is the player or an NPC, by ID. Reading or writing actor state
  needs to know which actor; doing so with none is an error.
- An **interaction** is instance state: `<thing id>#<interaction>` is a `choice`
  over its states, and `<thing id>#<interaction>_locked` is a `bool`.
- Only values that differ from the default are stored.

## Declarations

Every variable is declared, and where it is declared says who owns it. An
asset's `state:` belongs to each placed instance; a scene's `state:` must give
each variable `scope: world` or `scope: actor`.

```yaml
state:                                      # in an asset: one value per instance
  lit: {type: bool, default: false}
state:                                      # in a scene: the package's
  bridge_down: {scope: world, type: bool, default: false}
  has_ticket:  {scope: actor, type: bool, default: false}
```

A nested scene's state belongs to the package being exported, so two scenes
declaring one name must agree. A placement's `state: {lit: true}` starts that
instance with another value.

Assets also declare what they offer to wiring, by names relative to themselves
(`of: <reference>` for something inside the asset), so every instance wires
itself:

```yaml
signals:
  pulled:   {emit: pulled}              # an interaction arriving at a state with that emit
  switched: {changed: lit, to: true}    # a variable or interaction taking a value
  occupied: {enter: plate}              # an actor entering a place of the asset (leave: too)
inputs:
  open:  {set: gate, to: open}          # drive an interaction, or set a variable
  light: {set: lit}                     # no `to`: the driver gives the value
```

A place is a named trigger volume a scene or asset declares (`places:`, see
`geogen.places`): `plate` above is one of the asset's. `enter` and `leave` also
take a room of a floor plan; anything else is an error. A railway station's
platform is a place too (`railway_<line>/station_<name>`).

Some instance state is the runtime's to set, not a rule's: it is declared with
`driven: <what drives it>`. A train has `at_<station>` (a bool, `driven:
train`) for every station it calls at, true while a train of that service
stands there by its timetable. Rules and quests read it and react to it
changing (`{changed: commuter.at_north, to: false}`: the train left); export
counts it as state that changes.

Export collects all of it (`geogen.state.collect`) into the `state` section of
the manifest and chunk index (schema `docs/schema/geogen-state.v1.schema.json`):

```json
{"format": "geogen-state", "version": 1, "package": "demo",
 "variables": {"demo:lamp#lit": {"type": "bool", "default": false},
               "world:demo.bridge_down": {"type": "bool", "default": false}},
 "interactions": {"demo:lever#pull": {"initial": "up", "duration": 0.5, "states": {...}}},
 "signals": {"demo:lever#pulled": {"subject": "demo:lever", "name": "pulled"},
             "demo:lamp#switched": {"subject": "demo:lamp", "name": "switched",
                                    "from": {"type": "changed", "key": "demo:lamp#lit", "to": true}}},
 "inputs": {"demo:gate#open": {"set": "demo:gate#gate", "to": "open"}}}
```

- `interactions` lists every interaction of the scene, declared or not: they
  are instance state, and a runtime needs them for things that aren't loaded.
- A **signal** is the event `{type: signal, subject, name}`. One with `from` is
  raised whenever an event matching `from` is processed, exactly as a rule
  `{on: <from>, do: [{emit: name, subject}]}` placed before every other rule
  would (`Declarations.signal_rules`). One without is raised by the world: an
  interaction's own emit, or an NPC being greeted.
- An NPC that can be greeted (`attention.greet` in its definition, on by
  default) offers the signal `greeted` with no declaration: `signal` under
  `attention.greet` renames it. See "Events".
- An **input** is an effect: driving it sets `set` to `to`.

A declaration that clashes, or a signal or input naming state, an emit or a
thing that isn't there, fails the export with the file and line it was written
at and the near names.

## Interactions

An interaction's variable holds the state it is heading to, and changes the
moment it is used or driven. The parts take up to `duration` seconds to get
there; **arriving** is a separate moment.

- `use` by an actor moves it to the current state's `next`. With no `next`
  nothing happens. While locked it raises the signal `locked` and nothing else.
- A rule drives it by setting the variable to a state. This ignores `next` and
  the lock, so a gate with no `next` can only be moved by rules.
- On arrival, the state's `emit` is raised as a signal from the thing. A state
  with `then` moves on: at once, or `after` seconds later if it is still there.
- Reversing before arrival cancels the arrival: no signal from the state it
  never reached.

The reference machine takes exactly `duration` for every move. A real runtime
may arrive sooner on a short move, so traces leave slack round arrivals.

## Events

An event is `{type, subject, name, actor, data}`. `subject` is a stable ID,
`actor` an actor ID or nothing.

| Type | Meaning |
|---|---|
| `signal` | a thing raised `name`: an interaction arriving at a state with `emit`, an asset's declared signal, a rule's `emit`, or an NPC being greeted |
| `changed` | a variable took a new value: `key`, `data.from`, `data.to` |
| `use` | an actor used an interaction, before it moves |
| `enter`, `leave` | an actor entered or left a place or room (`subject`: its ID). An actor who sits or lies down there has not left |
| `travel` | an actor travelled to another scene or spawn |
| `clock` | dusk or dawn: `name` is `night` or `day`. Never has an actor |
| `start` | a new game began for an actor |

**Greeting an NPC** (the player pressing E on one) is the signal `greeted`
from the NPC, with the greeter as the actor: `{type: signal, subject:
"cottage:resident", name: "greeted", actor: "player"}`. It is raised whenever
the NPC is there to be greeted, in any pose and in the middle of any action
(seated, it waves from its chair): not while it is away, and not for a second
press within its greeting's `debounce` (half a second). This is how an NPC gives a quest: a rule
`{on: {signal: resident.greeted, actor: player}, do: [{start_quest: ...}]}`.
There is no separate "talk" event until dialogue needs one.

An event caused by a rule carries the actor of the event that triggered the
rule. So "the player pulled the lever, which opened the gate" keeps the player
as the actor all the way down.

## Rules

```json
{"id": "open_gate",
 "on": {"type": "signal", "subject": "demo:lever", "name": "pulled"},
 "when": <condition>, "do": [<effect>, ...], "after": 0, "once": false}
```

- `on` matches an event by `type`, and by `subject` (an ID or a list), `name`,
  `key` and `to` (the new value of a `changed` event) when given. `actor` filters
  by who: `player`, `npc`, `none`, or anyone (the default).
- `when` is a compiled condition (`geogen.conditions`), evaluated for the
  event's actor at the moment the event is processed.
- Effects run in order. `{"set": key, "to": <condition tree>}` writes a value;
  `{"emit": name, "subject": id}` raises a signal.
- `after: s` delays the effects. `when` is checked at the trigger, not again
  when the effects run.
- `once: true` lets the rule fire once per actor, or once in all for events
  with no actor.

**A rule with no `on` watches its condition** and fires each time the condition
becomes true. If the condition reads actor state it is watched separately for
every actor, and fires for that actor. This is what "done when every light is
off" compiles to. A condition that is already true at the start fires once.

## Bindings

`{"set": key, "to": <condition tree>}` is a standing relation: the variable
always equals the expression. A bound variable is never set by a rule and never
saved; it is recomputed from what it reads. Bindings can read other bindings but
not form a cycle, and cannot involve actor state.

Use a binding when something should *be* so while a condition holds (a lamp lit
while a plate is occupied) and a rule when something should *happen* (a gate
opens when a lever is pulled).

## Processing order

Inputs are handled one at a time, each to completion:

1. The event goes on a queue.
2. Take the first event. Every rule whose `on` matches and whose `when` holds
   runs, in program order. Each sees the state the rules before it left.
3. Changes and signals those rules caused join the end of the queue. They are
   processed after every rule for the current event has run.
4. Bindings are recomputed, in dependency order. A changed binding queues a
   `changed` event.
5. Repeat from 2 until the queue is empty.
6. Watched conditions are all judged against that settled state; the ones that
   became true then fire in program order. If that queued anything, go to 2.

Setting a variable to the value it already has is not a change and queues
nothing.

**Limits.** An event more than 16 steps removed from the input is dropped, and
so is a 17th round of watches. Either is reported as an error, not a crash.
An effect that fails (a wrong type, actor state with no actor) is reported and
skipped; the rule's other effects still run.

## Time

- Timers (arrivals, `after` dwells, delayed rules) fire in time order, each as
  its own input. Ties go to the one scheduled first.
- The clock offers `time` ("HH:MM" text, which orders correctly as text), `hour`,
  `night` and `day`. Bindings and watches that read it are re-evaluated when the
  minute changes.

## Saving

A save holds the stored world values, the stored values of every actor, the
clock and the pending timers (schema `docs/schema/geogen-save.v1.schema.json`;
`Machine.save` / `Machine.load`):

```json
{"format": "geogen-save", "version": 1, "now": 12.5,
 "clock": {"time": "21:10", "hour": 21.17, "night": true, "day": 0},
 "world": {"yard:gate#gate": "open", "yard:gate#gate_locked": true,
           "yard:plate#occupants": 1, "world:yard.power": false, "watch:crowded": true},
 "actors": {"player": {"actor:yard.pulls": 2, "once:discover": true},
            "yard:steward": {}},
 "actor_info": {"player": {"kind": "player", "room": ""}, "yard:steward": {"kind": "npc", "room": ""}},
 "timers": [{"at": 13.0, "seq": 4, "kind": "rule", "rule": "close_later", "actor": "player"}]}
```

- World and actor state are separate sections, and each is flat: a key names
  its owner, so instance state (`<thing id>#<name>`) and world variables sit
  side by side in `world`, and every actor has its own `actor:` values. A
  single-player game writes all of it to one file.
- An interaction is instance state like any other: `<thing id>#<interaction>`
  is the state it is in or heading to, `<thing id>#<interaction>_locked` its lock.
- Only values that differ from their defaults are stored. A key a save doesn't
  have is at its default, so loading puts everything else back as it started.
- `once` and watch latches are kept as hidden values (`once:<rule>`,
  `watch:<rule>`) with the state they belong to.
- Bindings are never saved. Loading recomputes them without raising events:
  nothing changed while saved.
- `timers` are what is still to happen (`rule`: a delayed rule; `arrive` and
  `dwell`: an interaction on its way or waiting to move on), on `now`'s scale.
- `actor_info` (kind, room) may be left out: actors a save doesn't describe stay
  as the host knows them.

## What export must reject

`rules.check_program` reports all of these:

- a variable, state or type that isn't declared, or a default of the wrong type
- a binding cycle, a binding on an interaction, a binding involving actor state
- a rule that sets a bound variable
- actor state written or read by a rule whose trigger never has an actor (a
  `clock` event, `actor: none`, a watched condition over world state only)
- an unresolved reference or selector in a condition
- a rule with neither `on` nor `when`, or with no effects

## Authoring

Scenes and assets write rules in YAML (`geogen.wiring`), compiled at export
with the scene's quests into the manifest's and chunk index's `rules` section:
this program without its variables and interactions, which `state` holds.

```yaml
rules:
  - id: open_gate                       # optional
    on: {signal: lever.pulled}          # an event; without one, `when` is watched
    when: "world.power"
    do:
      - {set: gate.gate, to: open}      # <thing>.<state> | world.<name> | actor.<name>
      - {input: lamp.light, value: true}
      - {emit: opened, of: gate}
      - {start_quest: lights_out}       # also fail_quest, complete_objective (quests.md)
    after: 2
    once: true
    if: "{quests}"                      # build time: see below
wire:
  - {from: lever.pulled, to: gate.open}     # a declared signal drives a declared input
bind:
  lamp.lit: "plate.occupants > 0 and world.power"
```

- **Events:** `{enter | leave: <place or room>}`, `{signal: <thing>.<name>}`,
  `{use: <thing>[.<interaction>]}`, `{changed: <state>, to}`, `{start: true}`,
  `{clock: night | day}`, `{travel: true | <scene>}`, each with an optional
  `actor: player | npc | none | any`.
- **In a scene** names are references from that scene. **In an asset** they are
  relative to it and every placed instance gets its own copy: `self` is the
  instance (`self.lit`, `{signal: self.pulled}`), other names are things inside it.
- **Order in the program:** declared signals, then wiring (scene order), then the
  quests, so a rule that gives a quest runs before that quest's objectives.
- A rule's id is its `id`, or `rule_<n>` / `wire_<n>`; below the package's own
  scene it is prefixed with the owner's ID (`yard:plate#step_off`).
- **`if:`** on a rule or a wire is a param expression (`"{quest}"`), resolved
  when the file is loaded: when it is 0, false or "" the entry is left out of
  the build. A rule's `when:` is its condition at run time, hence the other
  word. A `quests:` entry takes it too (`{file: quests/name.yaml, if: "{quest}"}`).
  A scene that is also nested in bigger ones uses it to give its quest only
  when it is the scene being played: the cottage's `quest` param.
- Mistakes fail the export with file and line: an unknown signal or one the
  thing never raises (`{signal: lever.pushed}`), an unknown input, state or
  quest, an event that isn't one, and everything `check_program` rejects.

Not built: `when:` on an interaction (a condition to use it).

## Content packages

A scene's own `state:`, `rules:` and `quests:` are its implicit package, named
after the scene. Content that belongs to no one scene (a quest from the hub to
the island and back) is a **content package**: a `kind: package` file,
`assets/packages/<name>.yaml` (`geogen.content`).

```yaml
kind: package
version: 1
id: errand                        # default: the file name; never a scene's name
title: The errand                 # optional, with description
scenes: [camp, annex]             # the catalogue scenes it runs in and may reach into
state:                            # its own variables: scope world | actor, as in a scene
  lever_pulled: {scope: world, type: bool, default: false}
rules:                            # rules, wire and bind, as in "Authoring"
  - on: {signal: "camp:lever.pulled"}
    do: [{set: world.lever_pulled, to: true}]
  - when: "world.lever_pulled"
    do: [{input: "annex:gate.open"}]
bind:
  'ref("annex:lamp").lit': "world.lever_pulled"
quests: [quests/errand.yaml]      # kind: quest files (quests.md)
```

- **Names.** Everything in a scene is named with its scene: `camp:lever`,
  `annex:plate/top`, in a condition `ref("camp:lever").pull == "down"` and
  `all("camp:[tag=light]", ...)`. `world.x` and `actor.x` are the package's own
  variables (`world:errand.x`); `world("camp").fire` reads a scene's.
- **One scene at a time.** Only one scene is loaded, so a rule, binding or
  objective may touch one scene, plus the package's own state: that is what
  crosses. One that touches two is an export error; split it into a rule that
  sets package state in the first scene and one that reads it in the second.
  A binding can't bind the package's own state to a scene's (it would only hold
  while that scene is loaded): set it with a rule.
- **Deploying.** The catalogue lists them (`packages: [errand]` in
  `assets/runtime_scenes.yaml`). `--export-catalogue`, and every
  `--export-godot`, compiles each against the *exported* scenes it names:
  references resolve on the scene's reference index (`<scene>.refs.json`), and
  conditions, inputs and signals are checked against its manifest's `state` and
  `rules`, exactly as the scene's own rules are. A package whose scenes aren't
  all exported yet waits. Mistakes are reported with file and line, and the
  package is not written:

  ```
  packages/errand.yaml:12: rules.0.on: 'camp:levr.pulled': no 'levr' in 'camp'. Did you mean 'lever'? It has: lever
  ```

The export is `<name>.content.json` beside `catalogue.json` (format
`geogen-content`, schema `docs/schema/geogen-content.v1.schema.json`):

```json
{"format": "geogen-content", "version": 1, "package": "errand", "scenes": ["camp", "annex"],
 "variables": {"world:errand.lever_pulled": {"type": "bool", "default": false}},
 "bindings": [{"set": "annex:lamp#lit", "to": {...}, "scene": "annex"}],
 "rules": [{"id": "errand:rule_1", "scene": "camp",
            "on": {"type": "signal", "subject": "camp:lever", "name": "pulled"},
            "do": [{"set": "world:errand.lever_pulled", "to": true}]},
           {"id": "errand:rule_2", "scene": "annex", "when": {...},
            "do": [{"set": "annex:gate#gate", "to": "open"}]}],
 "quests": {"format": "geogen-quests", ...}}
```

`scene` is the one scene an entry touches; without it the entry only reads and
writes the package's own state. Rule ids start `<package>:`.

**What a runtime does** (`content.apply`, `carried` and `resume` are the
reference; the rules semantics above don't change):

1. It reads the deployed packages once.
2. When a scene loads, for each package that lists it: the package's
   `variables` join the scene's program, and its `rules` and `bindings` with no
   `scene` or with that scene are appended after the scene's own. Its quests
   are listed with the scene's.
3. When the scene unloads, the package's values are kept apart from the
   scene's: its `world:<package>.` and `actor:<package>.` values and its rules'
   `once:` and `watch:` latches. The next scene's program starts with them, so
   state set in one scene is read in the next.
4. The watch latch of a rule with a `scene` is not kept: that scene's things
   start over when it loads again, so the rule's condition is judged afresh
   (the gate above opens on every visit). `once` holds for the whole game.
5. Arriving in a scene settles the watches for the actors there, as a new
   actor does.

A save made in a scene holds the packages' values with that scene's. Pending
delayed effects (`after`) don't survive leaving the scene, as for a scene's own
rules.

Not built: packages that place geometry into a scene, a package that runs
where a scene it names isn't deployed (soft dependencies), a rule that reads
two scenes at once, signals and inputs declared by a package, and markers for
a package's quests (an objective in another scene should point at the way
there: geogen.15.5). An NPC's package state is kept by its stable ID, so it
means something only in the NPC's own scene.

## Traces

A trace is `{program, actors, steps}`. Steps are inputs (`use`, `event`, `set`,
`advance`, `clock`, `save_load`) and `expect` blocks that check `state`,
`actors`, and exactly which `signals` and rules (`fired`) happened since the
last one. `geogen.rules.run_trace` replays one and returns the mismatches. A `save_load`
step is not an input: nothing fires because of it (`{"save_load": {"without": [section]}}`
drops sections of the save before loading it).

Edit traces in `tests/data/rules/make_traces.py`, which compiles the conditions,
and run it to rewrite the JSON.

| Trace | Covers |
|---|---|
| `lever_opens_gate` | wiring with no player, arrival signals, reversing mid-move |
| `bindings` | standing relations, a binding reading a binding, save and load |
| `actors_and_once` | one value per actor, `once`, events with no actor |
| `watched_conditions` | conditions that fire when they become true, per actor |
| `time` | delayed rules, a self-closing door, locks, the clock, timers in a save |
| `cascades` | the order rules run in, and a loop being cut off |
| `quest_go_there` | quests ([quests.md](quests.md)) that start with the game; an event objective, a counted one, a save mid-quest |
| `quest_change_the_world` | a quest given by a rule when the player (not an NPC) greets someone; a condition over a selector, `after`, an optional objective, rewards |
| `quest_timed` | availability and failure from the clock; one actor completes and another fails |
