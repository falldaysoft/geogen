# State, events, rules and bindings

This is the contract for the layer under object wiring, quests, dialogue and
inventory. It is declarative: content is data, and a runtime is anything that
obeys the rules below. `geogen/rules.py` is the reference implementation
(`Machine`), and the traces in `tests/data/rules/` say what must happen for
given inputs. The Godot runtime is one implementation and must pass the same
traces.

Authoring syntax (the YAML that compiles to this) belongs to the loaders; this
document covers what is exported and how it behaves.

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
| `enter`, `leave` | an actor entered or left a place or room (`subject`) |
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

## Traces

A trace is `{program, actors, steps}`. Steps are inputs (`use`, `event`, `set`,
`advance`, `clock`, `save_load`) and `expect` blocks that check `state`,
`actors`, and exactly which `signals` and rules (`fired`) happened since the
last one. `geogen.rules.run_trace` replays one and returns the mismatches.

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
