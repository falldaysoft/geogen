# Quests

This is what the quest system is being built towards, and how far it has got.
The epic is geogen.15; the layer it stands on (state, events, rules) is
geogen.3, specified in [rules.md](rules.md).

## The idea

A quest is a task with progress and an end, held by an actor: the player or an
NPC. It is data. A quest file is thin sugar that compiles, at export, to
actor state and rules of the general rules layer. A runtime that implements
that layer runs quests with no quest logic of its own: it only reports what the
state says (a console line, a HUD tracker, a marker).

That gives three properties we want to keep:

- **One semantics.** `geogen.rules.Machine` is the reference. The Godot runtime
  must pass the same traces (`tests/data/rules/`), quests included.
- **Checked at export.** A quest that names a thing that isn't there, reads
  state nothing declares, or has an objective nothing can cause fails the
  export with the file and line, before anyone plays it.
- **No dependency on dialogue or inventory.** They add ways to start a quest
  and things an objective can ask for. Quests work without them.

Quests also serve as tutorials, guided tours, discovery and achievements,
scripted sequences, and playtest scripts: a bot that follows a quest's markers
proves the scene it runs through still works.

## Authoring

A quest is a `kind: quest` file in `assets/quests/`. A scene lists the ones it
contributes with `quests: [quests/lighthouse_gallery.yaml]`; they belong to the
package being exported, and references in them resolve from that scene.

```yaml
kind: quest
version: 1
id: lights_out                 # default: the file name
title: Lights out
description: Switch off every light before bed.
holder: player                 # player (default) | npc | any: who can hold it
start: given                   # given (default): a 'start quest' effect; auto: as soon as it is available
available: "clock.night"       # optional: it can't start until this holds
ordered: false                 # true: each objective waits for the one before it
objectives:
  - id: lights
    text: Switch off every light
    done: 'all("house/[tag=light_switch]", it.switch == "off")'
  - id: bed
    text: Go to bed
    after: lights              # an id or a list; unordered objectives can be done in any order
    done: {enter: house/bedroom}
  - id: window
    text: Close the window
    optional: true
    done: 'window.swing == "closed"'
fail: 'clock.time >= "06:00" and not clock.night'
rewards:
  - {set: actor.slept_well, to: true}
```

**`done`** is either a condition or an event.

- A condition (text, the syntax of `geogen.conditions`) is watched while the
  objective is open and latches the moment it is true. Undoing it afterwards
  does not undo the objective.
- An event is one of `{enter: <place or room>}`, `{leave: ...}`,
  `{signal: <thing>.<name>}`, `{use: <thing>[.<interaction>]}` or
  `{changed: <state>, to: <value>}`, with an optional `when:` condition and
  `count: n` (it must happen n times). The event must be the holder's own:
  someone else entering the gallery does nothing.

**Effects** that other content uses to drive a quest: `{start_quest: id}`,
`{fail_quest: id}`, `{complete_objective: quest.objective}`. A rule on any event
can give a quest (E on an NPC, entering a place, another quest's rewards).
`start: auto` covers the tutorial and intro case.

**Markers** come from each objective's subject: the place or thing of its
event, or the things its condition reads. `marker: hidden` suppresses it,
`marker: <reference>` points somewhere else, `marker: {at: <reference>, area: 6}`
shows a ring on the ground and not the exact spot. A subject in another scene
is marked at the travel portal that leads there.

## Lifecycle

```
unavailable <-> available -> active -> complete
                                    \-> failed
```

- `unavailable` exists only for quests with an `available:` condition; they
  move between the two as it changes. Others start `available`.
- `active` is entered by a start effect, or at once for `start: auto`.
- An objective is **open** while the quest is active, everything in its `after`
  is done and it isn't done itself. Only open objectives can be done.
- The quest is `complete` the moment its last required objective is done.
  `rewards` run then. Optional objectives can't be done afterwards.
- `fail` is watched while active. If the last objective and the fail condition
  come true at the same moment, the quest completes.
- `complete` and `failed` are final. Repeatable quests are not designed yet.

## What a quest compiles to

For quest `q` in package `pkg`, all per actor:

| State | Type | Meaning |
|---|---|---|
| `actor:pkg.quest_q` | choice | the lifecycle state above |
| `actor:pkg.quest_q_<objective>` | bool | the objective's done latch |
| `actor:pkg.quest_q_<objective>_count` | number | times its event has happened (with `count:`) |

And rules, in this order, each id starting `quest:q:`:

| Rule | Trigger | Does |
|---|---|---|
| `available`, `unavailable` | watch `available:` | moves between the two |
| `start` | watch: available, and the actor is a holder | `start: auto` only: activates |
| `<objective>` | watch its condition, or `on` its event, while open | latches it; completes the quest if that was the last |
| `fail` | watch `fail:` while active | fails it |
| `rewards` | the status changing to `complete` | the reward effects |

Every write is conditional on the state it expects (`"failed" if status ==
"active" else status`), so two rules firing in one round can't overrule each
other and effects from outside can't skip a step.

Because progress is ordinary actor state, it is saved with the actor, survives
scene switches with them, and an NPC holds a quest exactly as the player does.

## Export

`geogen.quests.collect(root, package)` compiles every quest of a built scene
and raises `QuestError` listing each problem as `file:line: key: message`:

- unknown keys, a missing title or text, ids that clash
- `after` naming an objective that isn't there, or a cycle (an objective that
  waits on itself, directly or round a loop)
- a reference or selector that doesn't resolve, with near names
- state that isn't declared or is read with the wrong owner; a choice compared
  with a value it can't have
- **an objective nothing can cause**: a condition that reads no live state, or
  only state that no use of an interaction and no rule ever changes (a declared
  input counts once something is wired to it); a state it needs
  that can't be reached; an event from a thing that never raises it

The manifest and chunk index get a `quests` section (`geogen-quests` v1,
`docs/schema/geogen-quests.v1.schema.json`): per quest its title, holder, the
status key, and per objective its text, latch key, `after`, subjects and
progress (`{value: <condition tree>, of: n}`, derived from an `all(...)` so a
tracker can say "3 of 5"), plus the compiled `rules`. The quest variables join
the manifest's `state` section.

## Runtime (not built yet)

- **Console** (15.2): `quest event: {...}` on every quest and objective
  transition, `--quests` and the control API's `quests.list` dump every deployed
  quest and its state per actor. Markers are printed.
- **HUD** (15.10 to 15.13): a tracker for the active quest with progress
  counts, toasts for started / objective done / complete / failed, world
  markers clamped to the screen edge, and a quest log screen.
- **NPC holders** (15.8): the NPC planner turns the open objective's subject
  into a goal with its existing action vocabulary: go to a place, use an
  interaction or affordance.
- **Playtest bot** (15.9): `--playtest=quests` walks every quest with the real
  player body and reports the ones it can't finish.

## Archetypes: the acceptance tests

Each is a real quest in a showcase scene, passing in the Python evaluator and
in Godot.

| Quest | Scene | Exercises |
|---|---|---|
| Go there: climb to the lighthouse gallery (15.3) | island | places, actor state, a marker, `start: auto` |
| Change the world: switch off every light (15.4) | cottage | conditions over interaction state, a selector, progress, a quest given by an NPC |
| Ordered, across scenes: hub to the jetty and back (15.5) | hub + island | travel events, state surviving a scene switch, markers at portals |
| Timed, can fail: be on the platform before the train leaves (15.6) | level_crossing | clock conditions, `failed`, events from trains |
| Fetch and deliver: the room key and the parcel (15.7) | hotel_showcase | items, an NPC giver, giving to an NPC |
| An NPC does one: the resident switches off the lights (15.8) | cottage | NPC holders |

## Where it stands

| Piece | Task | State |
|---|---|---|
| Conditions, compiled for the runtime | 3.1 | done |
| Semantics and reference evaluator | 3.8 | done |
| State declarations in the manifest | 3.2 | done |
| Named places with enter / leave events | 3.9 | done |
| Quest kind, compilation, export checks | 15.1 | done |
| Event bus in Godot | 3.5 | open |
| State store and save format in Godot | 3.3, 3.4 | open |
| Rules authoring (`rules:`, `wire:`, `bind:`) and the Godot interpreter | 3.6 | open: until then nothing in YAML can give a quest, so authored quests are `start: auto` |
| Content packages (quests that span scenes) | 3.10 | open |
| Runtime quests, HUD, archetypes | 15.2 to 15.13 | open |
| Items for fetch quests | 18 | open |

Not designed: repeatable and daily quests, quests shared by a party, quest
chains beyond one quest's rewards starting another, branching outcomes.
