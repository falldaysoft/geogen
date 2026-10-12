# State, events, rules and bindings

This is the contract for the layer under object wiring, quests, dialogue and
inventory. It is declarative: content is data, and a runtime is anything that
obeys the rules below. `geogen/rules.py` is the reference implementation
(`Machine`), and the traces in `tests/data/rules/` say what must happen for
given inputs. The Godot runtime is one implementation and must pass the same
traces.

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
take a room of a floor plan; anything else is an error.

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
  would (`Declarations.signal_rules`). One without is an interaction's own emit.
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
| `signal` | a thing raised `name`: an interaction arriving at a state with `emit`, an asset's declared signal, or a rule's `emit` |
| `changed` | a variable took a new value: `key`, `data.from`, `data.to` |
| `use` | an actor used an interaction, before it moves |
| `enter`, `leave` | an actor entered or left a place or room (`subject`: its ID). An actor who sits or lies down there has not left |
| `travel` | an actor travelled to another scene or spawn |
| `clock` | dusk or dawn: `name` is `night` or `day`. Never has an actor |
| `start` | a new game began for an actor |

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
clock and the pending timers. World and actor state are separate sections.
`once` and watch latches are kept as hidden values (`once:<rule>`,
`watch:<rule>`) with the state they belong to. Loading recomputes bindings
without raising events: nothing changed while saved.

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
- Mistakes fail the export with file and line: an unknown signal, input, state
  or quest, an event that isn't one, and everything `check_program` rejects.

Not built: `when:` on an interaction (a condition to use it), and rules that
reach into another package (geogen.3.10).

## Traces

A trace is `{program, actors, steps}`. Steps are inputs (`use`, `event`, `set`,
`advance`, `clock`, `save_load`) and `expect` blocks that check `state`,
`actors`, and exactly which `signals` and rules (`fired`) happened since the
last one. `geogen.rules.run_trace` replays one and returns the mismatches. A `save_load`
step is not an input: nothing fires because of it.

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
| `quest_change_the_world` | a quest given by a rule; a condition over a selector, `after`, an optional objective, rewards |
| `quest_timed` | availability and failure from the clock; one actor completes and another fails |
