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
package being exported, and references in them resolve from that scene. A
quest belongs to one placed scene, so a scene that is also nested in others
(the cottage, twice on the island) lists its quest as
`{file: quests/lights_out.yaml, if: "{quest}"}` and the scenes that nest it
pass `quest: 0`.

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
    done: 'all("house/[tag=switch.light]", it.switch == "off")'
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
can give a quest (entering a place, another quest's rewards). An NPC gives one
when the player greets it (E): the NPC raises the signal `greeted` with the
greeter as the actor ([rules.md](rules.md), "Events"), and a scene rule starts
the quest:

```yaml
rules:
  - id: give_lights_out
    on: {signal: resident.greeted, actor: player}
    do: [{start_quest: lights_out}]
```

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
tracker can say "3 of 5"), plus `markers`: the world position of every
subject, so a marker shows for things in chunks that aren't loaded. The rules
that run the quests are in the manifest's `rules` section with the scene's own
(`quest:<id>:...`, after the wiring that can give them). The quest variables join
the manifest's `state` section.

## Runtime

The console and the HUD's quest tracker, notifications, log and markers are built (`runtime/godot/README.md`, "Rules, state
and quests" and "HUD"); the rest is planned.

- **Console** (15.2): `quest event: {...}` on every quest and objective
  transition, `--quests` and the control API's `quests.list` dump every deployed
  quest and its state per actor, with `progress: [n, of]` evaluated now for an
  objective that counts ("2 of 3 lights"). Markers are printed.
- **Tracker** (15.10): the HUD (`scripts/hud.gd`, a layer later panels join) has a
  quest tracker in the top-right corner: the tracked quest's title, its open
  and done objectives and their progress counts ("3 of 5"), read from the same
  report as `quests.list` and redrawn on quest and state events. Q tracks the
  next active quest; a finished one stays up a few seconds marked complete or
  failed. The control API's `hud.quests` returns what is displayed.
- **Notifications** (15.11): a HUD panel in the bottom-left corner shows a toast
  with a chime for each of the player's quest events: started, objective
  complete, quest complete, quest failed. Several at once queue and appear one
  by one. Restored and resumed state raises none. The control API's
  `hud.notifications` returns what is showing and what is queued.
- **Quest log** (`scripts/hud_quest_log.gd`): J opens a screen listing active,
  completed and failed quests of every package loaded this session, with each
  one's objectives; Enter makes the tracker follow the selected one. The
  control API's `hud.log` returns what it holds.
- **Markers** (15.12, `scripts/hud_markers.gd`): an indicator with the distance
  over each subject of the tracked quest's open objectives, clamped to the
  screen's edge with an arrow when it is off screen or behind; a ring on the
  ground, off-centre, for `{area}`; nothing for `hidden`; and for a quest of
  another scene, the travel point that goes straight there. The control API's
  `hud.markers` returns what is displayed.
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
| Change the world: switch off every light (15.4) | cottage (`quests/lights_out.yaml`) | conditions over interaction state, a selector, progress, a quest given by an NPC |
| Ordered, across scenes: hub to the jetty and back (15.5) | hub + island | travel events, state surviving a scene switch, markers at portals |
| Timed, can fail: be on the platform before the train leaves (15.6) | level_crossing | clock conditions, `failed`, events from trains |
| Fetch and deliver: the room key and the parcel (15.7) | hotel_showcase | items, an NPC giver, giving to an NPC |
| An NPC does one: the resident switches off the lights (15.8) | cottage | NPC holders |

## A timed quest

`quests/catch_train.yaml` (scene `level_crossing`) is the pattern for "be there
in time". Nothing in it is a timer: it reads the world.

- `available: 'clock.time >= "08:00"'` opens it, and `start: auto` starts it then.
- `done: 'actor.on_north_platform and commuter.at_north'` is a condition, so it
  holds whether the player or the train arrives first. `commuter.at_north` is
  the train's own state, set by the runtime from its timetable. The scene keeps
  `actor.on_north_platform` with two rules on entering and leaving the station
  (conditions have no "is in this place" term yet).
- `fail: 'world.north_train_left'`: a scene rule sets that when the train leaves
  after 08:00. An event with no actor (a train leaving) can't write actor
  state, so it notes the fact in world state and each holder's quest reads it.

A fixed limit after an event is a delayed rule instead: the quest yard's dash
has `{on: {enter: start_line}, after: 6, do: [{fail_quest: yard_dash}]}`.

## Trying them

The quest yard (`assets/scenes/quest_yard.yaml`, the Quests portal in the hub)
is a plaza with three quests chosen to exercise the tracker:

| Quest | How it starts | Shows |
|---|---|---|
| The round | with the game | objectives that wait for the one before, an optional one |
| Lamps out | greet the steward (E) | a count ("2 of 3"), then an objective that appears when it is done |
| The dash | step onto the start square | a quest that fails: a scene rule on the same event, `after: 6`, does `fail_quest` |

```bash
python -m geogen.main -s quest_yard --export-godot
godot --path runtime/godot -- --scene quest_yard
```

Several are active at once, so Q cycles the tracker. Progress lasts for the
session (restart to play them again). `tests/test_quest_yard.py` plays all
three in the evaluator and in Godot.

## Where it stands

| Piece | Task | State |
|---|---|---|
| Conditions, compiled for the runtime | 3.1 | done |
| Semantics and reference evaluator | 3.8 | done |
| State declarations in the manifest | 3.2 | done |
| Named places with enter / leave events | 3.9 | done |
| Quest kind, compilation, export checks | 15.1 | done |
| Event bus in Godot (`runtime/godot/scripts/events.gd`) | 3.5 | done |
| State store in Godot; progress in saves | 3.3, 3.4 | done (`scripts/rules.gd`); the save file's own format and schema are open |
| Rules authoring (`rules:`, `wire:`, `bind:`: `geogen.wiring`) | 3.6 | done: a scene rule gives a quest with `{start_quest: id}` |
| The rules interpreter in Godot (`scripts/machine.gd`, passes the traces) | 3.3, 3.6 | done |
| Content packages (quests that span scenes) | 3.10 | open |
| Runtime quests: `quest event:`, `--quests`, `quests.list`, markers as data | 15.2 | done |
| The lighthouse quest, the first archetype, in Godot | 15.3 | done: walked from the spawn to the gallery with the real body, and the reference evaluator, fed the runtime's place events, ends in the same state (`tests/test_quests_runtime.py`) |
| The lights-out quest, given by greeting the cottage's resident | 15.4 | runs in the evaluator (`tests/test_quests.py`) and in Godot over the control API (`tests/test_quests_runtime.py`): greeted, three real switches, the count rising, complete |
| HUD layer and quest tracker (`scripts/hud.gd`, `scripts/hud_quest_tracker.gd`, `hud.quests`) | 15.10 | done (`tests/test_hud_runtime.py`) |
| Catch the train: opens at a clock time, fails when the train leaves | 15.6 | runs in the evaluator (`tests/test_quests.py`) and in Godot, both outcomes (`tests/test_quests_runtime.py`): `quests/catch_train.yaml` in `level_crossing` |
| Quest notifications (`scripts/hud_notifications.gd`, `hud.notifications`) | 15.11 | done (`tests/test_hud_runtime.py`) |
| Quest log screen (`scripts/hud_quest_log.gd`, `hud.log`) | 15.13 | done (`tests/test_hud_log_runtime.py`) |
| World markers (`scripts/hud_markers.gd`, `hud.markers`) | 15.12 | done (`tests/test_hud_markers_runtime.py`); a direct portal only, no routes over several scenes |
| Other archetypes, NPC holders, playtest bot | 15.5, 15.7 to 15.9 | open |
| Items for fetch quests | 18 | open |

Not designed: repeatable and daily quests, quests shared by a party, quest
chains beyond one quest's rewards starting another, branching outcomes.
