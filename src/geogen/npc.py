"""NPCs as data: definitions, actions, poses, portals and home regions.

Everything an NPC does is declared and resolved here; the runtime
(``runtime/godot/scripts/npc.gd``) only interprets it:

- ``assets/npcs/actions.yaml`` (``kind: npc_actions``): what each action is,
  as steps from a fixed vocabulary (go_to, face, pose, wait, use), plus
  default approach offsets and durations per action.
- ``assets/npcs/<name>.yaml`` (``kind: npc``): a body asset, speed, needs,
  preferences, activities, scoring weights and flags.
- Assets advertise ``affordances:`` (see ``layout.loader._affordance``) and
  doors a ``portal:`` (an opening an NPC path may cross, with the interaction
  that opens it).
- Body assets declare ``poses:`` (root transforms relative to an anchor).

A scene places an NPC like an asset (``place: {resident: {npc: npcs/resident.yaml,
on: house.floor, home: house.floor}}``). The placed node carries
``meta.type = "npc"`` and ``meta.npc`` (the resolved definition, with the
actions and poses it needs inlined), and its body asset as a child, so NPCs
travel with the model or chunk they're placed in.
"""

from __future__ import annotations

import copy
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from . import roots
from .core.node import SceneNode
from .layout.yaml_utils import load_kind, safe_load_path
from .fields import known_keys

ASSETS_DIR = roots.CORE
STIMULI_KIND = "npc_stimuli"
# Perception (geogen.99): what an NPC notices of the stimuli around it (npcs/stimuli.yaml), and
# what it does then. Opt-in: a definition without `senses:` notices nothing and costs nothing.
STIMULUS_EVENTS = {"impact", "changed"}     # bus event types a stimulus can be (runtime/godot/scripts/events.gd)
# Hearing: a stimulus carries `range` x its loudness (x its strength); with something in the way
# (a wall, a shut door) only `muffled` of that. Sight: within `range` and a `cone` of degrees
# ahead, with a clear line. (attention.look's range and cone are for looking at a person who
# comes near; these are for noticing anything, so they are the definition's own.)
SENSES_DEFAULTS = {"hearing": {"range": 10.0, "muffled": 0.3}, "sight": {"range": 8.0, "cone": 150.0}}
REACT_KEYS = {
    "look": "seconds (or [lo, hi]) the head turns to where it came from (default [0.8, 1.5])",
    "interrupt": "true: cut short what it is doing, if that action is `interruptible` (actions.yaml), and "
                 "decide again, so a want about what happened is seen to now",
    "action": "an action (actions.yaml) run on top of what it is doing when it notices, on its feet with empty "
              "hands, facing where it came from (its `anchor`): a start (`startle`); seated or lying, just the look",
    "go": "true (the `look_into` action) or an action: go and see where something it only heard came from, at its "
          "next decision if that comes within a few seconds (with `interrupt`, now); not for a prop it remembers "
          "(memory's search sees to that), and once per stimulus and subject in GO_AGAIN s",
}
GO_DEFAULT = "look_into"
# Memory (geogen.99.2): with `memory:`, an NPC knows of the props it has seen (where, and what they
# rested on then) or heard land (a stimulus it reacts to: only where), each kept `forget` s from
# when it last perceived it; it looks round every `every` s. Its wants about props are then about
# what it remembers, and a thing it only heard, or that isn't where it was, it goes to look for
# first (the `search` action). A spot it can see where something it remembers no longer is
# corrects the memory. Needs senses.
MEMORY_DEFAULTS = {"forget": 300.0, "every": 0.5, "search": "look_for"}

NPC_KIND = "npc"
ACTIONS_KIND = "npc_actions"
VERSION = 1

STEP_KEYS = {"go_to", "face", "pose", "wait", "use", "vanish", "play", "use_item", "pick_up", "put_down", "find",
             "set_out", "show", "do"}
# `do: <motion>` runs a motion file (assets/motions/) on the body and waits for it to end.
# `roles: {<motion role>: subject | place | anchor | interaction}` binds its roles to what the action
# is about, `params: {<param>: value}` sets its params (`duration`: drawn from the action's range), and
# `keep: true` starts it and goes on: it runs until the option ends. A body with no rig skips the step.
DO_ROLES = {"subject", "place", "anchor", "interaction"}
# What is carried shows (geogen.105): `set_out: <item>` puts the carried item's body on what is in
# front of the NPC (a table) until the action ends or the item is used up; `show: {<item>: <state>}`
# gives that body (in the hand or set out) a state of its own interaction: a meal eaten from full
# to half to empty.
# `find: [sit, lie]`: the nearest free affordance of those types this NPC may use (in its home, not
# another's, not reserved, nothing lying on it) becomes the action's anchor, approach and heading,
# and `pose: found` takes its pose; finding none, the action goes on where it is (pose: found does
# nothing). `prefer: {tag: factor}` scales how far a thing with that tag counts (0.4: as if 40% as
# far); its own things count half as far and each distance gets a seeded +-40%. Reading a book in
# the armchair or in bed (geogen.104).
FIND_TYPES = ("sit", "lie")
STEP_OPTIONS = {"at", "if", "hold", "loop", "close", "look", "prefer", "part", "roles", "params", "keep"}
# `look:` on any step: where the head turns from that step on, until a later step says otherwise
# or the action ends (the action's subject, place or anchor; none: nowhere in particular).
LOOK_TARGETS = {"subject", "place", "anchor", "partner", "none"}
GO_TO_TARGETS = {"approach", "anchor", "random", "nearby", "near", "far", "exit", "aside", "visit", "group",
                 "subject", "place", "partner"}
# What a body's hands get to and what it can carry (physics props: pick_up / put_down). A prop
# over `two_hands` m on its longest side is carried in front of the body in both hands; a
# smaller one in one hand, at the side.
REACH_DEFAULTS = {"distance": 1.0, "heights": [0.0, 1.5], "stand": 0.5}
CARRY_DEFAULTS = {"mass": 10.0, "size": 0.6, "two_hands": 0.2}
# What a body can jump (geogen.100.6), m: `up` onto something at most this much higher, `down` off
# something at most this much higher, `across` as far as this level. An NPC with `jump:` reaches
# spots off the navmesh (perches: a sofa's seat) by a jump from where it can walk to, and jumps
# down off them again; its body plays `jump_up` / `jump_down` (one-shot clips with a `leap`).
JUMP_DEFAULTS = {"up": 0.8, "down": 1.4, "across": 1.0}
ITEM_SELECTOR = re.compile(r"^\[tag=([A-Za-z_][\w.]*)\]$")   # use_item: "[tag=food]": the first such item carried
FACE_TARGETS = {"anchor", "portal", "player", "partner"}
POSE_AT = {"anchor", "approach"}

NPC_KEYS = {"kind", "version", "body", "speed", "turn_speed", "radius", "needs", "preferences",
            "activities", "scoring", "home_margin", "flags", "seed", "affordance_tags", "wander", "routine", "attention",
            "reach", "carry", "jump", "wants", "senses", "reacts", "memory", "planner", "offers", "relations", "skips", "life"}
# A need (geogen.101.2): 0 (desperate) .. 1 (satisfied).
NEED_KEYS = {
    "initial": "where it starts, 0..1 (default 0.5)",
    "decay": "how much it falls per second",
    "urgent": "below this level its shortfall counts `boost` times more (planner.urgency): a need that has "
              "got low wins over ones that are merely not full",
    "boost": "how much more (default 4)",
    "at_zero": "{action, recover}: what happens when it runs out: the action is run where the NPC stands "
               "(collapse: it sleeps on the floor) and then gives `recover` (default 0.3) of the need back",
}
# `life:` (geogen.111): what stays changed about a person. `practice` grows by what they finish doing;
# `chapters` are the stretches of their life, in order, each left when its `until` holds.
LIFE_KEYS = {"practice", "chapters"}
PRACTICE_KEYS = {
    "from": "the tags of the things that count as doing this",
    "per_use": "how much each one finished adds, of 1 (default 0.05)",
}
CHAPTER_KEYS = {
    "name": "what this stretch of their life is called",
    "story": "the line written in their story on entering this chapter",
    "hopes": "what they are hoping for during it, in words",
    "pursues": "{tag: worth}: things with the tag are worth this much to them whatever their needs",
    "until": "{practice: {name: level}, knows: {person: level}, world: {interaction: state}}: all of these hold, "
             "and the next chapter starts",
    "becomes": "{preferences, needs: {need: {keys}}, home: {rects: [[x0, z0, x1, z1], ...]}}: how they change on "
               "entering, for good (each rect, in world metres, joins their home and must touch it)",
    "tends": "{interaction: state}: every interaction of that name goes to that state on entering",
}
LIFE_UNTIL = {"practice", "knows", "world"}
LIFE_BECOMES = {"preferences", "needs", "home"}
# `planner:` (geogen.101.4; true for these): the NPC decides by plans, chains of affordances found
# through what they require, take and give (geogen/planner.py), instead of one affordance at a time.
PLANNER_DEFAULTS = {"depth": 5.0, "patience": 60.0, "nodes": 2000.0}
# What others may do with this NPC (geogen.101.6): an affordance a person offers. Another NPC near
# enough (this one within its home) takes it as an option `with/<this NPC>/<name>`: both are held
# for each other while it lasts, the one who chose it walks up, and both get what it advertises.
OFFER_KEYS = {
    "action": "the action the one who takes it up runs (default: the offer's name): its `go_to: partner` walks "
              "up to this NPC, `wait: duration` is the time together",
    "answer": "the action this NPC runs meanwhile when it is on its feet (it waits for the other, turned to "
              "them: `wait: partner`); seated, it stays seated and looks at them",
    "advertises": "{need: amount}: what doing it adds to the needs, of both",
    "duration": "[lo, hi] s together",
    "weight": "multiplies its score (default 1)",
}
# How well an NPC knows the others, 0..1 each: `household` for those of its placement's `household`,
# `stranger` for anyone else; doing something together adds `grow`. An offer's score is multiplied by
# 0.5 + how well it knows the one who offers, and a seat beside someone it knows `friend` or better
# is liked as one beside a groupmate is.
RELATION_DEFAULTS = {"household": 0.5, "stranger": 0.1, "grow": 0.02, "friend": 0.3}
# A want (geogen.97): a condition on the world the NPC puts right with whatever is there.
WANT_KEYS = {
    "of": "what it is about: prop (a physics prop with `carry`), or a reference to things with an interaction "
          "(`[tag=door]`, `hall/switch`): resolved in the scene the NPC is placed in",
    "interaction": "of a reference: which of the things' interactions (default: the only one a thing has)",
    "when": "which subjects are a problem. A prop: {resting_on: floor} (floor: anything actors walk on). A thing: "
            "{state: <state>} it shows that state, and with room: empty, nobody is in the room it is in",
    "to": "of a reference: the state the action brings the thing to",
    "action": "the action that puts it right: run with the subject and a chosen place (put_away), or with the "
              "thing as its anchor and `to` as the state `use: goal` works for (set)",
    "place": "{prefer: {tag: multiplier}, quality: {name: weight}, height, distance, off_height, noise, tried, "
             "radius}: how it picks among the spots that hold the subject within its reach",
    "weight": "utility of putting one subject right (default 0.5), beside what `advertises` adds",
    "advertises": "{need: amount}: what doing it adds to the needs",
    "reluctance": "utility lost each time this NPC has picked the subject up (default 0.1), whatever for: once "
                  "that uses up `weight` it leaves the subject be, so wants that undo each other end",
    "forget": "seconds over which one handling fades from its reluctance (default 600)",
}
WANT_DEFAULTS = {"weight": 0.5, "reluctance": 0.1, "forget": 600.0}
WANT_PROP = "prop"
WANT_WHEN = {"resting_on": ("floor",)}          # of: prop
WANT_ROOM = ("empty",)                          # of a reference: when.room
# How a spot is scored: its tags' `prefer` multipliers, less `distance` per metre to walk there,
# less `off_height` per metre above or below the `height` (over its feet) it likes to put things
# at, less `tried` for a spot it has already used for this subject, plus `noise` x random. It
# looks `radius` m round itself where it has no home region.
# `quality: {name: weight}` (or a name, or a list of names: weight 1 each) adds weight x how much a
# spot has of that quality, 0..1 (PLACE_QUALITIES; any other name is one a thing is tagged with).
PLACE_QUALITIES = {
    "high": "how far above the floor it is (1 at 2 m)",
    "enclosed": "how shut in: the share of the ways out of it, sideways and up, that are blocked within 0.6 m",
    "hidden": "how hard to see: the share of the places to stand 2 m around it with no line of sight to it",
    "warm": "in the sun (by the time of day), or on or near a thing tagged quality.warm",
    "soft": "on a thing tagged quality.soft (a sofa, a bed)",
}
QUALITY_TAG = "quality."            # a thing's tag `quality.<name>` gives every spot on it that quality
PLACE_DEFAULTS = {"height": 0.9, "distance": 0.1, "off_height": 0.3, "noise": 0.1, "tried": 1.0, "radius": 6.0}
CHARACTER_KIND = "character"
# Placement keys a character file may give defaults for (the scene's placement wins).
CHARACTER_PLACEMENT_KEYS = ("home", "work", "owns", "prefers", "visits", "quests", "seed", "household")
CHARACTER_KEYS = {"name", "npc", "body", *CHARACTER_PLACEMENT_KEYS,
                  *(NPC_KEYS - {"kind", "version", "body", "seed"})}
# An NPC's own activities (things it can always do, besides the affordances around it).
ACTIVITY_KEYS = {
    "action": "the action it runs (default: the activity's name)",
    "advertises": "{need: amount}: what doing it adds to the needs",
    "duration": "[lo, hi] s: the action's `wait: duration`",
    "weight": "multiplies its score (default 1): how much more or less often it wins",
    "cooldown": "s after doing it before it is an option again (a rare trip)",
    "visit": "one of the placement's `visits`: the region its `go_to: visit` goes to (the dining room)",
}
ROUTINE_KEYS = {"from", "to", "away", "activities", "preferences", "name", "at", "only", "hold", "quests"}
# A routine quest's `repeat`: once (left out), every day the block starts, as soon as it is done,
# or when it is done and a condition holds.
ROUTINE_REPEATS = ("daily", "always")
ROUTINE_OWNS = "owns"               # in a block's `only`: the things the NPC owns, whatever their tags
ROUTINE_AT = ("home", "work")        # where a block's options are: the home region, or the placement's `work`
SCORING_DEFAULTS = {"distance": 0.02, "recency": 0.4, "memory": 120.0, "noise": 0.05, "retry": 30.0}
POSE_KEYS = {"offset", "rotation", "scale"}
# How an NPC notices the player (runtime/godot/scripts/npc.gd): turn its head toward them
# (``look``: within ``range`` m and a ``cone`` of degrees ahead, head/neck turned up to ``yaw`` /
# ``pitch`` degrees, eased at ``speed``), greet them when they press E on it (``greet``: the
# action it runs when standing, the player's prompt, and the ``signal`` the NPC raises with the
# greeter as the actor, which rules listen to. The signal is raised in any pose and mid-action;
# only a second press within ``debounce`` s is ignored. ``cooldown`` is the seconds before it
# waves again, and ``acknowledge`` how long it keeps looking at the greeter), and step
# ``step`` m aside when it stands within ``distance`` m of the player walking at it (``yield``).
ATTENTION_DEFAULTS = {
    # Busy (a `play` step's looping clip: work at a counter) it only glances up: for `busy_glance`
    # s, every `busy_every` [lo, hi] s.
    "look": {"range": 4.0, "cone": 150.0, "yaw": 70.0, "pitch": 25.0, "speed": 4.0,
             "busy_every": [5.0, 12.0], "busy_glance": 1.2},
    "greet": {"action": "greet", "prompt": "Say hello", "cooldown": 8.0, "signal": "greeted",
              "debounce": 0.5, "acknowledge": 3.0},
    "yield": {"distance": 1.0, "step": 0.9, "action": "step_aside"},
}
PORTAL_KEYS = {"interaction", "open", "closed", "center", "normal", "width", "height", "depth", "clearance",
               "handles", "swing"}
PORTAL_SIDES = ("front", "back")     # +normal, -normal


def _range(value: Any, what: str) -> list[float]:
    """``5`` -> [5, 5]; ``[lo, hi]`` -> [lo, hi] (seconds)."""
    if isinstance(value, (int, float)):
        return [float(value), float(value)]
    if isinstance(value, (list, tuple)) and len(value) == 2 and value[0] <= value[1]:
        return [float(value[0]), float(value[1])]
    raise ValueError(f"{what}: expected seconds or [lo, hi], got {value!r}")


def _vec3(value: Any, what: str) -> list[float]:
    if not (isinstance(value, (list, tuple)) and len(value) == 3):
        raise ValueError(f"{what}: expected [x, y, z], got {value!r}")
    return [float(v) for v in value]


def _check_do(step: dict[str, Any], where: str) -> None:
    """A ``do:`` step names a motion there is, binds every role it needs and only params it has."""
    from .motion import load_motion, motion_names

    name = step["do"]
    if name not in motion_names():
        raise ValueError(f"{where}: do: no motion {name!r} in assets/motions/")
    motion = load_motion(name)
    roles = step.get("roles") or {}
    if not isinstance(roles, dict) or set(roles.values()) - DO_ROLES:
        raise ValueError(f"{where}: roles is {{<motion role>: one of {sorted(DO_ROLES)}}}, got {roles!r}")
    declared = motion.get("roles") or {}
    if set(roles) - set(declared):
        raise ValueError(f"{where}: motion {name!r} has no role {sorted(set(roles) - set(declared))} "
                         f"(it has {sorted(declared)})")
    missing = [r for r, spec in declared.items() if r not in roles and not (spec or {}).get("optional")]
    if missing:
        raise ValueError(f"{where}: motion {name!r} needs its roles {missing} bound")
    params = step.get("params") or {}
    if not isinstance(params, dict) or set(params) - set(motion.get("params") or {}) - {"duration"}:
        raise ValueError(f"{where}: motion {name!r} has no params {sorted(set(params) - set(motion.get('params') or {}))}")
    if step.get("keep", False) not in (True, False):
        raise ValueError(f"{where}: keep is true or false")


def _check_step(step: Any, where: str) -> dict[str, Any]:
    if not isinstance(step, dict):
        raise ValueError(f"{where}: a step must be a mapping, got {step!r}")
    verbs = STEP_KEYS & set(step)
    unknown = set(step) - STEP_KEYS - STEP_OPTIONS
    if len(verbs) != 1 or unknown:
        raise ValueError(f"{where}: a step has exactly one of {sorted(STEP_KEYS)} "
                         f"(options {sorted(STEP_OPTIONS)}), got {sorted(step)}")
    verb = next(iter(verbs))
    arg = step[verb]
    if verb == "do":
        _check_do(step, where)
    elif {"roles", "params", "keep"} & set(step):
        raise ValueError(f"{where}: roles, params and keep are a do step's options, got {sorted(step)}")
    if verb == "go_to" and arg not in GO_TO_TARGETS:
        raise ValueError(f"{where}: go_to must be one of {sorted(GO_TO_TARGETS)}, got {arg!r}")
    if verb == "pick_up" and arg != "subject":
        raise ValueError(f"{where}: pick_up takes the action's subject (pick_up: subject), got {arg!r}")
    if verb == "find" and not (isinstance(arg, list) and arg and set(arg) <= set(FIND_TYPES)):
        raise ValueError(f"{where}: find takes a list of affordance types to look for, of {list(FIND_TYPES)}, got {arg!r}")
    if "part" in step and (verb != "wait" or step["wait"] != "duration" or not isinstance(step["part"], (int, float))
                           or not 0 < step["part"] <= 1):
        raise ValueError(f"{where}: part is a `wait: duration` step's share (0..1] of the duration, drawn once for "
                         f"the action (eating in stages), got {step!r}")
    if "prefer" in step and (verb != "find" or not isinstance(step["prefer"], dict)
                             or not all(isinstance(v, (int, float)) and v > 0 for v in step["prefer"].values())):
        raise ValueError(f"{where}: prefer is a find step's {{tag: factor}} on how far a thing with that tag counts "
                         f"(below 1: preferred), got {step.get('prefer')!r}")
    if verb == "set_out" and not (isinstance(arg, str) and arg.isidentifier()):
        raise ValueError(f"{where}: set_out names an item the NPC carries (items/<id>.yaml), got {arg!r}")
    if verb == "show" and not (isinstance(arg, dict) and arg and all(isinstance(k, str) and isinstance(v, str)
                                                                       for k, v in arg.items())):
        raise ValueError(f"{where}: show is {{<item>: <state of its body's interaction>}}, got {arg!r}")
    if verb == "put_down" and arg != "place":
        raise ValueError(f"{where}: put_down sets it at the action's place (put_down: place), got {arg!r}")
    if verb == "face" and arg not in FACE_TARGETS:
        raise ValueError(f"{where}: face must be one of {sorted(FACE_TARGETS)}, got {arg!r}")
    if "hold" in step and (verb != "pose" or not isinstance(step["hold"], str) or not step["hold"]):
        raise ValueError(f"{where}: hold names the body pose to hold in place of a pose step's own, got {step!r}")
    if "close" in step and (verb != "go_to" or step["close"] is not True):
        raise ValueError(f"{where}: close: true is a go_to step's option (step up to the target off the navmesh)")
    if "loop" in step and verb != "play":
        raise ValueError(f"{where}: loop is a play step's option (how long to loop its clip), got {step!r}")
    if "look" in step and step["look"] not in LOOK_TARGETS:
        raise ValueError(f"{where}: look must be one of {sorted(LOOK_TARGETS)}, got {step['look']!r}")
    if "at" in step and step["at"] not in POSE_AT:
        raise ValueError(f"{where}: at must be one of {sorted(POSE_AT)}, got {step['at']!r}")
    out = dict(step)
    if verb == "vanish" and arg is not True:
        raise ValueError(f"{where}: vanish: true")
    if verb == "use_item" and not (isinstance(arg, str) and (arg.isidentifier() or ITEM_SELECTOR.match(arg))):
        raise ValueError(f"{where}: use_item names an item (items/<id>.yaml) or \"[tag=<tag>]\" (the first such "
                         f"item carried), got {arg!r}")
    if verb == "wait" and arg not in ("duration", "partner"):
        out["wait"] = _range(arg, f"{where}: wait")
    if "loop" in step and step["loop"] != "duration":
        out["loop"] = _range(step["loop"], f"{where}: loop")
    return out


def parse_actions(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate an ``npc_actions`` document; returns {action: {steps, approach?, duration?}}."""
    if data.get("kind") != ACTIONS_KIND or int(data.get("version", 0)) != VERSION:
        raise ValueError(f"npc actions: expected kind '{ACTIONS_KIND}' version {VERSION}")
    actions = {}
    for name, spec in (data.get("actions") or {}).items():
        known_keys(spec, {"steps", "approach", "duration", "interruptible"}, f"action '{name}'")
        if spec.get("interruptible", False) not in (True, False):
            raise ValueError(f"action '{name}': interruptible is true or false")
        steps = [_check_step(s, f"action '{name}' step {i}") for i, s in enumerate(spec.get("steps") or [])]
        if not steps:
            raise ValueError(f"action '{name}': needs steps")
        action: dict[str, Any] = {"steps": steps}
        if "approach" in spec:
            action["approach"] = _vec3(spec["approach"], f"action '{name}': approach")
        if "duration" in spec:
            action["duration"] = _range(spec["duration"], f"action '{name}': duration")
        if spec.get("interruptible"):
            action["interruptible"] = True      # something noticed may cut it short (`reacts: interrupt`)
        actions[name] = action
    return actions


def parse_stimuli(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate an ``npc_stimuli`` document; returns {stimulus: {event, loudness, of?}}."""
    if data.get("kind") != STIMULI_KIND or int(data.get("version", 0)) != VERSION:
        raise ValueError(f"npc stimuli: expected kind '{STIMULI_KIND}' version {VERSION}")
    stimuli = {}
    for name, spec in (data.get("stimuli") or {}).items():
        where = f"stimulus '{name}'"
        if not isinstance(spec, dict) or set(spec) - {"event", "of", "loudness"}:
            raise ValueError(f"{where}: keys are event, of, loudness, got {spec!r}")
        if spec.get("event") not in STIMULUS_EVENTS:
            raise ValueError(f"{where}: event is one of {sorted(STIMULUS_EVENTS)}, got {spec.get('event')!r}")
        loudness = float(spec.get("loudness", 1.0))
        if loudness < 0:
            raise ValueError(f"{where}: loudness can't be negative")
        stimuli[str(name)] = {"event": spec["event"], "loudness": loudness}
        if "of" in spec:
            if not isinstance(spec["of"], str) or not spec["of"]:
                raise ValueError(f"{where}: of is a tag the event's subject carries, got {spec['of']!r}")
            stimuli[str(name)]["of"] = spec["of"]
    return stimuli


def parse_memory(memory: Any, senses: Any, actions: dict, where: str) -> dict[str, Any]:
    """``memory:`` -> {memory: {forget, every, search}} (``MEMORY_DEFAULTS``; ``true`` for them), or {}."""
    if memory in (None, False):
        return {}
    if senses in (None, False):
        raise ValueError(f"{where}: memory needs senses (senses: true for the defaults)")
    memory = {} if memory is True else memory
    if not isinstance(memory, dict) or set(memory) - set(MEMORY_DEFAULTS):
        raise ValueError(f"{where}: memory is {{{', '.join(MEMORY_DEFAULTS)}}} or true, got {memory!r}")
    out: dict[str, Any] = {k: float(memory.get(k, v)) for k, v in MEMORY_DEFAULTS.items() if k != "search"}
    if out["forget"] <= 0 or out["every"] <= 0:
        raise ValueError(f"{where}: memory.forget and memory.every are seconds (> 0)")
    out["search"] = str(memory.get("search", MEMORY_DEFAULTS["search"]))
    if out["search"] not in actions:
        raise ValueError(f"{where}: memory.search: unknown action {out['search']!r}")
    return {"memory": out}


def _layers(name: str) -> tuple[Path, ...]:
    """Every content root's ``npcs/<name>``, the core's first: a project adds to the core's, and
    what it names again replaces the core's."""
    return tuple(root / "npcs" / name for root in reversed(roots.roots()) if (root / "npcs" / name).is_file())


@lru_cache(maxsize=8)
def _layered(paths: tuple[Path, ...], parse: Any) -> dict[str, dict[str, Any]]:
    return {name: entry for path in paths for name, entry in parse(safe_load_path(path) or {}).items()}


def load_stimuli(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    return _layered((Path(path),) if path is not None else _layers("stimuli.yaml"), parse_stimuli)


def parse_senses(senses: Any, reacts: Any, where: str, actions: dict | None = None) -> dict[str, Any]:
    """``senses:`` and ``reacts:`` -> {senses: {hearing, sight}, reacts: {stimulus: {event, loudness,
    of?, look, interrupt, action?, go?}}}, or {} without either. ``senses: true`` is the defaults; ``reacts``
    names stimuli from npcs/stimuli.yaml (inlined, like actions) and needs senses."""
    if senses in (None, False) and not reacts:
        return {}
    if senses in (None, False):
        raise ValueError(f"{where}: reacts needs senses (senses: true for the defaults)")
    senses = {} if senses is True else senses
    if not isinstance(senses, dict) or set(senses) - set(SENSES_DEFAULTS):
        raise ValueError(f"{where}: senses is {{{', '.join(SENSES_DEFAULTS)}}} or true, got {senses!r}")
    out_senses = {}
    for part, defaults in SENSES_DEFAULTS.items():
        spec = senses.get(part, True)
        if spec is False:
            continue                    # deaf, or blind
        spec = {} if spec is True else spec
        if not isinstance(spec, dict) or set(spec) - set(defaults):
            raise ValueError(f"{where}: senses.{part}: keys are {sorted(defaults)}, got {spec!r}")
        out_senses[part] = {k: float(spec.get(k, v)) for k, v in defaults.items()}
    stimuli = load_stimuli()
    out_reacts = {}
    if reacts is not None and not isinstance(reacts, dict):
        raise ValueError(f"{where}: reacts is {{stimulus: {{look, interrupt}}}}, got {reacts!r}")
    for name, spec in (reacts or {}).items():
        at = f"{where}: reacts.{name}"
        if name not in stimuli:
            raise ValueError(f"{at}: unknown stimulus (npcs/stimuli.yaml has {sorted(stimuli)})")
        spec = {} if spec in (True, None) else spec
        if not isinstance(spec, dict) or set(spec) - set(REACT_KEYS):
            raise ValueError(f"{at}: keys are {sorted(REACT_KEYS)}, got {spec!r}")
        if spec.get("interrupt", False) not in (True, False):
            raise ValueError(f"{at}: interrupt is true or false")
        out_reacts[str(name)] = {**stimuli[name], "look": _range(spec.get("look", [0.8, 1.5]), f"{at}: look"),
                                 "interrupt": bool(spec.get("interrupt", False))}
        for key in ("action", "go"):
            value = spec.get(key)
            if value in (None, False):
                continue
            if key == "go" and value is True:
                value = GO_DEFAULT
            if not isinstance(value, str) or not value:
                raise ValueError(f"{at}: {key} is an action name{' or true' if key == 'go' else ''}, got {value!r}")
            if actions is not None and value not in actions:
                raise ValueError(f"{at}: {key}: unknown action {value!r}")
            out_reacts[str(name)][key] = value
    return {"senses": out_senses, "reacts": out_reacts}


def load_actions(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    return _layered((Path(path),) if path is not None else _layers("actions.yaml"), parse_actions)


def parse_poses(spec: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """A body asset's ``poses:`` -> {name: {offset, rotation (deg), scale}} (all present)."""
    poses = {}
    for name, pose in (spec or {}).items():
        pose = pose or {}
        known_keys(pose, POSE_KEYS, f"pose '{name}'")
        poses[name] = {
            "offset": _vec3(pose.get("offset", [0, 0, 0]), f"pose '{name}': offset"),
            "rotation": _vec3(pose.get("rotation", [0, 0, 0]), f"pose '{name}': rotation"),
            "scale": _vec3(pose.get("scale", [1, 1, 1]), f"pose '{name}': scale"),
        }
    if poses and "stand" not in poses:
        raise ValueError("poses: a body must define 'stand'")
    return poses


def parse_portal(spec: dict[str, Any], interactions: list, parts: dict | None = None, root: Any = None) -> dict[str, Any]:
    """A door's ``portal:``: the opening an NPC path may cross (asset frame).

    ``{interaction: swing, open: open, closed: closed, center: [x, y, z],
    normal: [0, 0, 1], width, height, depth, clearance}``: the opening is
    ``width`` x ``height`` centred on ``center``, ``depth`` thick along
    ``normal``; NPCs stop ``clearance`` beyond either face to operate it
    (clear of the swinging leaf). ``handles: {front: <part>, back: <part>}`` names the handle on
    each side (front: +normal), exported as its middle in the asset frame as the door stands
    when built; ``swing: front | back`` is the side the leaf opens toward. An NPC reaches for
    the handle on its side to open the door, and on the side the leaf swings away from stands
    close enough to hold it as it opens.
    """
    known_keys(spec, PORTAL_KEYS, "portal")
    name = spec.get("interaction")
    found = next((i for i in interactions if i.name == name), None)
    if found is None:
        raise ValueError(f"portal: no interaction {name!r} on this asset ({[i.name for i in interactions]})")
    portal = {
        "interaction": name,
        "open": str(spec.get("open", "open")),
        "closed": str(spec.get("closed", "closed")),
        "center": _vec3(spec.get("center", [0, 1, 0]), "portal: center"),
        "normal": _vec3(spec.get("normal", [0, 0, 1]), "portal: normal"),
        "width": float(spec.get("width", 0.9)),
        "height": float(spec.get("height", 2.0)),
        "depth": float(spec.get("depth", 0.3)),
        "clearance": float(spec.get("clearance", 0.9)),
    }
    for state in (portal["open"], portal["closed"]):
        if state not in found.states:
            raise ValueError(f"portal: interaction {name!r} has no state {state!r}")
    if "swing" in spec:
        if spec["swing"] not in PORTAL_SIDES:
            raise ValueError(f"portal: swing is the side the leaf opens toward ({' | '.join(PORTAL_SIDES)}), "
                             f"got {spec['swing']!r}")
        portal["swing"] = spec["swing"]
    handles = spec.get("handles") or {}
    if not isinstance(handles, dict) or set(handles) - set(PORTAL_SIDES):
        raise ValueError(f"portal: handles is {{front: <part>, back: <part>}}, got {handles!r}")
    if handles:
        portal["handles"] = {}
        moving = {id(n) for m in found.motions for n in m.parts} if hasattr(found, "motions") else set()
        for side, part in handles.items():
            node = (parts or {}).get(part)
            bounds = node.mesh_bounds(root) if node is not None and root is not None else None
            if bounds is None:
                raise ValueError(f"portal: handles.{side}: no part {part!r} with a mesh on this asset")
            if moving and id(node) not in moving:
                raise ValueError(f"portal: handles.{side}: '{part}' doesn't move with {name!r}")
            portal["handles"][side] = [round(float(v), 4) for v in (bounds[0] + bounds[1]) / 2]
    return portal


def approach_point(action: str, position: np.ndarray, yaw_deg: float,
                   override: Any = None) -> list[float]:
    """Where an actor stands to start ``action`` at an anchor (asset frame).

    ``override`` is the affordance's own ``approach`` (a distance in front
    of the anchor, or [x, y, z] in the anchor frame); else the action's
    default. y is dropped to the anchor's floor level (y = 0 of the asset);
    the runtime snaps it onto the navmesh.
    """
    if override is None:
        offset = load_actions().get(action, {}).get("approach", [0.0, 0.0, 0.0])
    elif isinstance(override, (int, float)):
        offset = [0.0, 0.0, float(override)]
    else:
        offset = _vec3(override, "affordance approach")
    yaw = np.radians(yaw_deg)
    right = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    forward = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
    p = position + offset[0] * right + offset[2] * forward
    return [round(float(p[0]), 4), 0.0, round(float(p[2]), 4)]


def load_definition(path: str | Path) -> dict[str, Any]:
    """Load and validate an NPC definition (``kind: npc``). Body and home are resolved later."""
    return _definition(load_kind(path, NPC_KIND, VERSION, NPC_KEYS), path)


def load_character(path: str | Path, assets_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """A named person (``kind: character``): the NPC definition of its role (``npc:``) with the
    character's own keys on top, and the placement defaults it brings (``home``, ``work``, ``owns``,
    ``prefers``, ``quests``, ``seed``), which a scene's placement overrides.

    Role keys that are mappings (needs, preferences, activities, ...) merge key by key; the rest
    replace. A ``body`` with an ``asset`` or ``archetype`` replaces the role's; one with only
    ``params`` adds them to the role's body (the role's archetype, a few params fixed).
    """
    data = load_kind(path, CHARACTER_KIND, VERSION, CHARACTER_KEYS)
    for key in ("name", "npc"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"{path}: a character needs `{key}`")
    role_path = roots.find(data["npc"], assets_dir)
    role = load_kind(role_path, NPC_KIND, VERSION, NPC_KEYS)
    merged = dict(role)
    for key in set(data) & (NPC_KEYS - {"kind", "version", "body", "seed"}):
        value = data[key]
        merged[key] = {**(role.get(key) or {}), **value} if isinstance(value, dict) and isinstance(role.get(key), dict) \
            else value
    body = data.get("body")
    if isinstance(body, str):
        body = {"asset": body}
    if body is not None:
        if not isinstance(body, dict):
            raise ValueError(f"{path}: body is {{asset|archetype, params}} or {{params}}, got {body!r}")
        if {"asset", "archetype"} & set(body):
            merged["body"] = body
        else:
            base = role.get("body") or {"asset": "characters/capsule.yaml"}
            base = {"asset": base} if isinstance(base, str) else base
            merged["body"] = {**base, "params": {**(base.get("params") or {}), **(body.get("params") or {})}}
    definition = _definition(merged, path)
    definition["definition"] = Path(role_path).stem
    definition["character"] = Path(path).stem
    definition["name"] = data["name"]
    defaults = {k: data[k] for k in CHARACTER_PLACEMENT_KEYS if k in data}
    return definition, defaults


def _definition(data: dict[str, Any], path: str | Path) -> dict[str, Any]:
    actions = load_actions()
    needs = {}
    for need, spec in (data.get("needs") or {}).items():
        needs[need] = _need(spec or {}, actions, f"{path}: need {need}")
    activities = {}
    for name, spec in (data.get("activities") or {}).items():
        action = spec.get("action", name)
        if action not in actions:
            raise ValueError(f"{path}: activity {name}: unknown action {action!r}")
        known_keys(spec, ACTIVITY_KEYS, f"{path}: activity {name}")
        activity = {"action": action, "advertises": _advertises(spec.get("advertises"), needs, f"{path}: {name}")}
        if "duration" in spec:
            activity["duration"] = _range(spec["duration"], f"{path}: {name}: duration")
        for key in ("weight", "cooldown"):
            if key in spec:
                activity[key] = float(spec[key])
                if activity[key] < 0:
                    raise ValueError(f"{path}: activity {name}: {key} can't be negative")
        if "visit" in spec:
            activity["visit"] = str(spec["visit"])
        activities[name] = activity
    scoring = {**SCORING_DEFAULTS, **{k: float(v) for k, v in (data.get("scoring") or {}).items()}}
    if set(scoring) - set(SCORING_DEFAULTS):
        raise ValueError(f"{path}: scoring keys are {sorted(SCORING_DEFAULTS)}")
    attention = parse_attention(data.get("attention", True), actions, f"{path}: attention")
    body = data.get("body") or {"asset": "characters/capsule.yaml"}
    if isinstance(body, str):
        body = {"asset": body}
    if not ({"asset", "archetype"} & set(body)) or set(body) - {"asset", "archetype", "params"}:
        raise ValueError(f"{path}: body is {{asset|archetype, params}}, got {body!r}")
    return {
        "definition": Path(path).stem,
        "body": body,
        "speed": float(data.get("speed", 1.2)),
        "turn_speed": float(data.get("turn_speed", 360.0)),
        "seed": int(data.get("seed", 0)),
        "needs": needs,
        "preferences": {str(k): float(v) for k, v in (data.get("preferences") or {}).items()},
        "activities": activities,
        "scoring": scoring,
        "home_margin": float(data.get("home_margin", 1.0)),
        "flags": {str(k): bool(v) for k, v in (data.get("flags") or {}).items()},
        # Only affordances carrying one of these tags (e.g. [outside]: a passer-by doesn't sit
        # in people's kitchens); empty = all of them.
        "affordance_tags": [str(t) for t in data.get("affordance_tags") or []],
        # Affordances carrying one of these tags are left to someone else.
        **({"skips": [str(t) for t in data["skips"]]} if data.get("skips") else {}),
        # Where `go_to: random` goes: the home region, or points on surfaces with these tags.
        "wander": {"tags": [str(t) for t in (data.get("wander") or {}).get("tags", [])]},
        "routine": parse_routine(data.get("routine") or [], activities, f"{path}: routine"),
        "attention": attention,
        "reach": _limits(data.get("reach"), REACH_DEFAULTS, f"{path}: reach"),
        "carry": _limits(data.get("carry"), CARRY_DEFAULTS, f"{path}: carry"),
        **({"jump": _limits(data["jump"], JUMP_DEFAULTS, f"{path}: jump")} if data.get("jump") is not None else {}),
        "wants": parse_wants(data.get("wants"), actions, needs, f"{path}: wants"),
        **parse_senses(data.get("senses"), data.get("reacts"), str(path), actions),
        **parse_memory(data.get("memory"), data.get("senses"), actions, str(path)),
        **_planner(data.get("planner"), f"{path}: planner"),
        **_social(data.get("offers"), data.get("relations"), actions, needs, str(path)),
        **_life(data.get("life"), needs, f"{path}: life"),
    }


def _need(spec: dict[str, Any], actions: dict, where: str) -> dict[str, Any]:
    if not isinstance(spec, dict) or set(spec) - set(NEED_KEYS):
        raise ValueError(f"{where}: keys are {sorted(NEED_KEYS)}, got {spec!r}")
    need: dict[str, Any] = {"initial": float(spec.get("initial", 0.5)), "decay": float(spec.get("decay", 0.0))}
    if not 0.0 <= need["initial"] <= 1.0:
        raise ValueError(f"{where}: initial must be 0..1")
    if "urgent" in spec:
        need["urgent"] = float(spec["urgent"])
        need["boost"] = float(spec.get("boost", 4.0))
        if not 0.0 < need["urgent"] <= 1.0 or need["boost"] < 1.0:
            raise ValueError(f"{where}: urgent is a level in 0..1 and boost at least 1")
    elif "boost" in spec:
        raise ValueError(f"{where}: boost goes with urgent (the level below which it applies)")
    if "at_zero" in spec:
        zero = spec["at_zero"]
        zero = {"action": zero} if isinstance(zero, str) else zero
        if not isinstance(zero, dict) or "action" not in zero or set(zero) - {"action", "recover"}:
            raise ValueError(f"{where}: at_zero is an action or {{action, recover}}, got {zero!r}")
        if zero["action"] not in actions:
            raise ValueError(f"{where}: at_zero: unknown action {zero['action']!r}")
        need["at_zero"] = {"action": str(zero["action"]), "recover": float(zero.get("recover", 0.3))}
        if not 0.0 < need["at_zero"]["recover"] <= 1.0:
            raise ValueError(f"{where}: at_zero.recover is in 0..1 (and more than 0, or it never ends)")
    return need


def _life(spec: Any, needs: dict, where: str) -> dict[str, Any]:
    """``life:`` (geogen.111), checked and with its numbers as floats; {} without one."""
    if spec is None:
        return {}
    if not isinstance(spec, dict) or set(spec) - LIFE_KEYS or not spec.get("chapters"):
        raise ValueError(f"{where}: life is {{practice, chapters}} with at least one chapter, got {spec!r}")
    practice = {}
    for name, entry in (spec.get("practice") or {}).items():
        if not isinstance(entry, dict) or set(entry) - set(PRACTICE_KEYS) or not entry.get("from"):
            raise ValueError(f"{where}: practice {name}: keys are {sorted(PRACTICE_KEYS)}, with `from`, got {entry!r}")
        practice[str(name)] = {"from": [str(t) for t in entry["from"]], "per_use": float(entry.get("per_use", 0.05))}
    chapters = []
    for i, chapter in enumerate(spec["chapters"]):
        at = f"{where}: chapter {i}"
        if not isinstance(chapter, dict) or set(chapter) - set(CHAPTER_KEYS) or "name" not in chapter:
            raise ValueError(f"{at}: keys are {sorted(CHAPTER_KEYS)}, with `name`, got {chapter!r}")
        out: dict[str, Any] = {"name": str(chapter["name"])}
        for key in ("story", "hopes"):
            if key in chapter:
                out[key] = str(chapter[key])
        if chapter.get("pursues"):
            out["pursues"] = {str(k): float(v) for k, v in chapter["pursues"].items()}
        until = chapter.get("until") or {}
        if set(until) - LIFE_UNTIL:
            raise ValueError(f"{at}: until keys are {sorted(LIFE_UNTIL)}, got {sorted(until)}")
        for name in until.get("practice") or {}:
            if name not in practice:
                raise ValueError(f"{at}: until practice {name!r} isn't one of {sorted(practice)}")
        if until:
            out["until"] = {
                **{k: {str(n): float(v) for n, v in until[k].items()} for k in ("practice", "knows") if k in until},
                **({"world": {str(n): str(v) for n, v in until["world"].items()}} if "world" in until else {}),
            }
        becomes = chapter.get("becomes") or {}
        if set(becomes) - LIFE_BECOMES:
            raise ValueError(f"{at}: becomes keys are {sorted(LIFE_BECOMES)}, got {sorted(becomes)}")
        changed: dict[str, Any] = {}
        if "preferences" in becomes:
            changed["preferences"] = {str(k): float(v) for k, v in becomes["preferences"].items()}
        if "needs" in becomes:
            changed["needs"] = {}
            for need, keys in becomes["needs"].items():
                if need not in needs:
                    raise ValueError(f"{at}: becomes needs {need!r}: they have no such need")
                changed["needs"][str(need)] = {k: v for k, v in _need(keys, {}, f"{at}: {need}").items() if k in keys}
        if "home" in becomes:
            rects = becomes["home"].get("rects") if isinstance(becomes["home"], dict) else None
            if not isinstance(rects, list) or not rects or any(not isinstance(r, list) or len(r) != 4 for r in rects):
                raise ValueError(f"{at}: becomes home is {{rects: [[x0, z0, x1, z1], ...]}}, got {becomes['home']!r}")
            changed["home"] = {"rects": [[float(v) for v in r] for r in rects]}
        if changed:
            out["becomes"] = changed
        if chapter.get("tends"):
            out["tends"] = {str(k): str(v) for k, v in chapter["tends"].items()}
        chapters.append(out)
    return {"life": {"practice": practice, "chapters": chapters}}


def _planner(spec: Any, where: str) -> dict[str, Any]:
    if spec is None or spec is False:
        return {}
    spec = {} if spec is True else spec
    if not isinstance(spec, dict) or set(spec) - set(PLANNER_DEFAULTS) - {"hands"}:
        raise ValueError(f"{where}: true or keys {sorted({*PLANNER_DEFAULTS, 'hands'})}, got {spec!r}")
    out = {k: float(spec.get(k, v)) for k, v in PLANNER_DEFAULTS.items()}
    if out["depth"] < 1 or out["nodes"] < 1 or out["patience"] < 0:
        raise ValueError(f"{where}: depth and nodes are at least 1, patience isn't negative")
    planner = {"depth": int(out["depth"]), "patience": out["patience"], "nodes": int(out["nodes"])}
    if "hands" in spec:             # carries this many things at most, and shares a job with others (geogen.110.1)
        if not isinstance(spec["hands"], int) or spec["hands"] < 1:
            raise ValueError(f"{where}: hands is a whole number of at least 1, got {spec['hands']!r}")
        planner["hands"] = spec["hands"]
    return {"planner": planner}


def _social(offers: Any, relations: Any, actions: dict, needs: dict, where: str) -> dict[str, Any]:
    """``offers:`` and ``relations:`` (geogen.101.6) -> {offers: {name: {action, answer, advertises, duration,
    weight}}, relations: {household, stranger, grow, friend}}; {} for an NPC with neither."""
    if not offers and relations is None:
        return {}
    if not isinstance(offers or {}, dict) or not isinstance(relations or {}, dict):
        raise ValueError(f"{where}: offers is {{name: {{{', '.join(OFFER_KEYS)}}}}}, relations {{{', '.join(RELATION_DEFAULTS)}}}")
    known_keys(relations or {}, RELATION_DEFAULTS, f"{where}: relations")
    known = {**RELATION_DEFAULTS, **{k: float(v) for k, v in (relations or {}).items()}}
    if any(not 0.0 <= known[k] <= 1.0 for k in known):
        raise ValueError(f"{where}: relations are 0..1, got {relations!r}")
    out = {}
    for name, spec in (offers or {}).items():
        spec = spec or {}
        at = f"{where}: offer {name}"
        if not isinstance(spec, dict) or set(spec) - set(OFFER_KEYS):
            raise ValueError(f"{at}: keys are {sorted(OFFER_KEYS)}, got {spec!r}")
        action, answer = str(spec.get("action", name)), str(spec.get("answer", ""))
        if action not in actions or (answer and answer not in actions):
            raise ValueError(f"{at}: unknown action {action if action not in actions else answer!r}")
        verbs = [(k, v) for step in actions[action]["steps"] for k, v in step.items()]
        if ("go_to", "partner") not in verbs:
            raise ValueError(f"{at}: action {action!r} has no `go_to: partner` step")
        if answer and ("wait", "partner") not in [(k, v) for step in actions[answer]["steps"] for k, v in step.items()]:
            raise ValueError(f"{at}: answer {answer!r} has no `wait: partner` step")
        weight = float(spec.get("weight", 1.0))
        if weight < 0:
            raise ValueError(f"{at}: weight can't be negative")
        out[str(name)] = {"action": action, "answer": answer, "weight": weight,
                          "advertises": _advertises(spec.get("advertises"), needs, at),
                          "duration": _range(spec.get("duration", actions[action].get("duration", [8, 16])), f"{at}: duration")}
    return {"offers": out, "relations": known}


def parse_wants(spec: Any, actions: dict, needs: dict, where: str) -> dict[str, Any]:
    """``wants:`` -> {name: {of, when, action, weight, reluctance, forget, advertises, and place (of: prop) or
    to, interaction (of: a reference)}} (see ``WANT_KEYS``)."""
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: expected {{name: {{of, when, action, ...}}}}, got {spec!r}")
    wants = {}
    for name, want in spec.items():
        at = f"{where}: {name}"
        if not isinstance(want, dict) or set(want) - set(WANT_KEYS):
            raise ValueError(f"{at}: keys are {sorted(WANT_KEYS)}, got {want!r}")
        of = want.get("of")
        if not isinstance(of, str) or not of:
            raise ValueError(f"{at}: of is `prop` or a reference to things with an interaction, got {of!r}")
        when = want.get("when")
        action = want.get("action")
        if action not in actions:
            raise ValueError(f"{at}: unknown action {action!r}")
        place = want.get("place") or {}
        if of == WANT_PROP:
            if not isinstance(when, dict) or not when or set(when) - set(WANT_WHEN):
                known = ", ".join(f"{k}: {' | '.join(v)}" for k, v in WANT_WHEN.items())
                raise ValueError(f"{at}: when is {{{known}}}, got {when!r}")
            for key, value in when.items():
                if value not in WANT_WHEN[key]:
                    raise ValueError(f"{at}: when.{key} is one of {list(WANT_WHEN[key])}, got {value!r}")
            if set(want) & {"to", "interaction"}:
                raise ValueError(f"{at}: `to` and `interaction` belong to a want about things (of: a reference)")
            if not isinstance(place, dict) or set(place) - set(PLACE_DEFAULTS) - {"prefer", "quality"}:
                raise ValueError(f"{at}: place keys are {sorted([*PLACE_DEFAULTS, 'prefer', 'quality'])}, got {place!r}")
            quality = place.get("quality") or {}
            if isinstance(quality, str):
                quality = [quality]
            if isinstance(quality, list):
                quality = dict.fromkeys(quality, 1.0)
            if not isinstance(quality, dict) or not all(
                    isinstance(k, str) and k.isidentifier() and isinstance(v, (int, float)) for k, v in quality.items()):
                raise ValueError(f"{at}: place.quality is {{name: weight}}, a name or a list of names "
                                 f"({', '.join(PLACE_QUALITIES)}, or one things are tagged with), got {place.get('quality')!r}")
        else:
            if not isinstance(when, dict) or not isinstance(when.get("state"), str) or set(when) - {"state", "room"}:
                raise ValueError(f"{at}: when is {{state: <state>, room: empty}}, got {when!r}")
            if "room" in when and when["room"] not in WANT_ROOM:
                raise ValueError(f"{at}: when.room is one of {list(WANT_ROOM)}, got {when['room']!r}")
            if not isinstance(want.get("to"), str) or not want["to"] or want["to"] == when["state"]:
                raise ValueError(f"{at}: to is the state the action brings the thing to (not the one it is a "
                                 f"problem in), got {want.get('to')!r}")
            if "interaction" in want and not (isinstance(want["interaction"], str) and want["interaction"]):
                raise ValueError(f"{at}: interaction names one of the things' interactions, got {want['interaction']!r}")
            if "place" in want:
                raise ValueError(f"{at}: place belongs to a want about props (of: prop)")
        numbers = {k: float(want.get(k, v)) for k, v in WANT_DEFAULTS.items()}
        if numbers["weight"] < 0 or numbers["reluctance"] < 0 or numbers["forget"] <= 0:
            raise ValueError(f"{at}: weight and reluctance can't be negative, and forget is seconds (> 0)")
        wants[str(name)] = {"of": of, "when": dict(when), "action": action, **numbers,
                            "advertises": _advertises(want.get("advertises"), needs, at)}
        if of == WANT_PROP:
            wants[str(name)]["place"] = {**{k: float(place.get(k, v)) for k, v in PLACE_DEFAULTS.items()},
                                         "prefer": {str(k): float(v) for k, v in (place.get("prefer") or {}).items()},
                                         "quality": {k: float(v) for k, v in quality.items()}}
        else:
            wants[str(name)].update({k: want[k] for k in ("to", "interaction") if k in want})
    return wants


def resolve_wants(node: SceneNode, refs: Any, keys: list) -> None:
    """The things an NPC's wants are about, where ``of`` is a reference: every node it names in
    the scene that has the want's interaction (its only one, when the want names none), kept
    as the nodes until export writes their stable IDs (``meta.npc_want_subjects``). A selector
    that names nothing here is a want with nothing to do in this scene."""
    found: dict[str, list] = {}
    for name, want in (node.meta["npc"].get("wants") or {}).items():
        if want["of"] == WANT_PROP:
            continue
        at = [*keys, "wants", name, "of"]
        subjects = []
        for thing in refs.nodes(want["of"], at, optional=True):
            names = [i.name for i in getattr(thing, "interactions", None) or []]
            which = want.get("interaction") or (names[0] if len(names) == 1 else "")
            if which not in names:
                continue                # a door's leaf, a thing with other interactions
            state_names = set(next(i for i in thing.interactions if i.name == which).states)
            for state in (want["when"]["state"], want["to"]):
                if state not in state_names:
                    raise refs.fail(f"npc want {name}: {thing.name}.{which} has no state '{state}' "
                                    f"(states: {sorted(state_names)})", at)
            subjects.append((thing, which))
        found[name] = subjects
    if found:
        node.meta["npc_want_subjects"] = found


def want_extras(node: SceneNode, npc: dict[str, Any]) -> dict[str, Any]:
    """``npc.wants`` with ``meta.npc_want_subjects`` as stable IDs: each want about things gets
    ``subjects: [{subject, interaction}]``, what the runtime looks up."""
    found = node.meta.get("npc_want_subjects") or {}
    if not found:
        return {}
    wants = {name: dict(want) for name, want in (npc.get("wants") or {}).items()}
    for name, subjects in found.items():
        wants[name]["subjects"] = [{"subject": thing.meta["uid"], "interaction": which}
                                   for thing, which in subjects if "uid" in thing.meta]
    return {"wants": wants}


def _limits(spec: Any, defaults: dict[str, Any], where: str) -> dict[str, Any]:
    """A body-limits block (``reach:``, ``carry:``) over its defaults; numbers, or [lo, hi] ranges."""
    if spec is None:
        spec = {}
    if not isinstance(spec, dict) or set(spec) - set(defaults):
        raise ValueError(f"{where}: keys are {sorted(defaults)}, got {spec!r}")
    out: dict[str, Any] = {}
    for key, default in defaults.items():
        value = spec.get(key, default)
        out[key] = _range(value, f"{where}: {key}") if isinstance(default, list) else float(value)
        if not isinstance(default, list) and out[key] <= 0:
            raise ValueError(f"{where}: {key} must be positive, got {value!r}")
    return out


def parse_attention(value: Any, actions: dict, where: str) -> dict[str, Any]:
    """``attention:`` -> {look, greet, yield} with defaults; ``false`` (or a part ``false``) turns
    that behaviour off (the part is omitted)."""
    if value is False:
        return {}
    value = {} if value is True or value is None else value
    if not isinstance(value, dict) or set(value) - set(ATTENTION_DEFAULTS):
        raise ValueError(f"{where}: expected {{{', '.join(ATTENTION_DEFAULTS)}}} or false, got {value!r}")
    out = {}
    for part, defaults in ATTENTION_DEFAULTS.items():
        spec = value.get(part, True)
        if spec is False:
            continue
        spec = {} if spec is True else spec
        known_keys(spec, defaults, f"{where}: {part}")
        merged = {**defaults, **spec}
        for k, v in merged.items():
            merged[k] = v if isinstance(defaults[k], str) else (_range(v, f"{where}: {part}.{k}")
                                                                if isinstance(defaults[k], list) else float(v))
        if "action" in merged and merged["action"] not in actions:
            raise ValueError(f"{where}: {part}: unknown action {merged['action']!r}")
        out[part] = merged
    return out


def parse_time(value: Any, where: str) -> float:
    """"HH:MM" (or hours as a number) -> hours in [0, 24)."""
    if isinstance(value, (int, float)):
        hours = float(value)
    else:
        text = str(value)
        try:
            hh, _, mm = text.partition(":")
            minutes = int(mm) if mm else 0
            if not 0 <= minutes < 60:
                raise ValueError(text)
            hours = int(hh) + minutes / 60.0
        except ValueError:
            raise ValueError(f"{where}: times are 'HH:MM', got {value!r}") from None
    if not 0.0 <= hours <= 24.0:
        raise ValueError(f"{where}: time out of range: {value!r}")
    return round(hours % 24.0, 4)


def parse_routine(blocks: list, activities: dict, where: str) -> list[dict[str, Any]]:
    """Time-of-day blocks that bias behaviour (the runtime's world clock picks the active ones)::

        routine:
          - { from: "12:00", to: "14:00", activities: { pause: 2 }, preferences: { seat: 2 } }
          - { from: "22:30", to: "07:00", away: true }    # wraps midnight; off to an entrance
          - { from: "23:00", to: "07:00", only: [bed], hold: true }   # to bed, and stays there
          - { from: "09:00", to: "17:00", quests: [{ file: quests/prep.yaml, repeat: daily }] }

    ``activities`` / ``preferences`` multiply the scores of the NPC's own activities / of
    affordances with those tags; ``away`` sends the NPC off (the ``leave`` action) until the
    block ends; ``at: work`` keeps its options to the affordances inside the placement's
    ``work`` (a reference), wherever that is from home. ``only`` keeps the options to affordances
    with one of those tags (``owns``: the things the NPC owns), none of its own activities, and
    ``hold`` keeps it on the one it takes until the block ends (in its bed all night) instead of
    the affordance's duration. With nothing in ``only`` to take, it carries on as without them.
    ``quests`` are quest files (or ``{file, repeat}``) the NPC is given when the block starts and
    works on only while it lasts; ``repeat`` is ``daily`` (started over each day the block
    starts), ``always`` (again as soon as it is done, within the block) or ``{if: <condition>}``
    (again once it is done and the condition holds). The scene's export compiles them
    (``quests.collect``).
    """
    out = []
    for i, block in enumerate(blocks):
        if not isinstance(block, dict) or set(block) - ROUTINE_KEYS or not {"from", "to"} <= set(block):
            raise ValueError(f"{where}[{i}]: needs from/to (keys {sorted(ROUTINE_KEYS)}), got {block!r}")
        if block.get("at", "home") not in ROUTINE_AT:
            raise ValueError(f"{where}[{i}]: at is one of {', '.join(ROUTINE_AT)}, got {block['at']!r}")
        only = block.get("only", [])
        if not isinstance(only, list) or not all(isinstance(t, str) and t for t in only):
            raise ValueError(f"{where}[{i}]: only is a list of affordance tags (or {ROUTINE_OWNS!r}), got {only!r}")
        quests = []
        for j, entry in enumerate(block.get("quests") or []):
            entry = {"file": entry} if isinstance(entry, str) else entry
            repeat = entry.get("repeat") if isinstance(entry, dict) else None
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str) or set(entry) - {"file", "repeat"} \
                    or not (repeat is None or repeat in ROUTINE_REPEATS
                            or isinstance(repeat, dict) and set(repeat) == {"if"} and isinstance(repeat["if"], str)):
                raise ValueError(f"{where}[{i}].quests[{j}]: a quest file or {{file, repeat: "
                                 f"{' | '.join(ROUTINE_REPEATS)} | {{if: <condition>}}}}, got {entry!r}")
            quests.append({"file": entry["file"], **({"repeat": repeat} if repeat is not None else {})})
        acts = {str(k): float(v) for k, v in (block.get("activities") or {}).items()}
        unknown = set(acts) - set(activities)
        if unknown:
            raise ValueError(f"{where}[{i}]: unknown activities {sorted(unknown)}")
        out.append({"name": str(block.get("name", i)), "from": parse_time(block["from"], f"{where}[{i}]"),
                    "to": parse_time(block["to"], f"{where}[{i}]"), "away": bool(block.get("away", False)),
                    "activities": acts,
                    "preferences": {str(k): float(v) for k, v in (block.get("preferences") or {}).items()},
                    **({"at": block["at"]} if block.get("at", "home") != "home" else {}),
                    **({"only": list(only)} if only else {}),
                    **({"hold": True} if block.get("hold", False) else {}),
                    **({"quests": quests} if quests else {})})
    return out


POPULATION_KEYS = {
    "npc": "the NPC definition (npcs/<name>.yaml); one with an archetype body makes a different person each",
    "count": "how many a place: n, or [lo, hi] drawn per place",
    "on": "<places>.<surface>: a reference or selector to what they live on ([tag=house]/house.floor, "
          "plot_*/house.floor); each one matched gets its own count, and is its NPCs' home",
    "spacing": "m between them where they start (default 1.2)",
    "seed": "makes another draw (default: from the scene's and the entry's names)",
}


def population_placements(spec: Any, refs: Any, scene: str, keys: list) -> dict[str, dict[str, Any]]:
    """A scene's ``population:`` as NPC crowd placements: per entry, per place its ``on``
    matches, a scatter of ``count`` NPCs on that surface whose home it is. Deterministic: the
    draws are seeded by the scene's and entry's names (or the entry's ``seed``) and the place."""
    import random
    import zlib

    from .refs import path_of

    if not isinstance(spec, dict):
        raise refs.fail("population is {name: {npc, count, on, spacing, seed}}", keys)
    out: dict[str, dict[str, Any]] = {}
    for name, entry in spec.items():
        at = [*keys, name]
        if not isinstance(entry, dict) or set(entry) - set(POPULATION_KEYS) or not {"npc", "on"} <= set(entry):
            raise refs.fail(f"a population entry is {{{', '.join(POPULATION_KEYS)}}} with npc and on, "
                            f"got {entry!r}", at)
        count = entry.get("count", 1)
        lo, hi = (count, count) if isinstance(count, int) else tuple(count) if isinstance(count, list) else (None, None)
        if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (lo, hi)) or lo > hi:
            raise refs.fail(f"population count is a whole number or [lo, hi], got {count!r}", [*at, "count"])
        text = str(entry["on"])
        where, _, surface = text.rpartition(".")
        if not where or not surface:
            raise refs.fail(f"population on is <places>.<surface>, got {text!r}", [*at, "on"])
        places = refs.nodes(where, [*at, "on"])
        seed = entry.get("seed", zlib.crc32(f"{scene}/{name}".encode()))
        for i, place in enumerate(places):
            rng = random.Random(f"{seed}:{i}")
            n = rng.randint(lo, hi)
            if n == 0:
                continue
            on = f"{path_of(place, refs.root)}.{surface}"
            refs.surface(on, [*at, "on"])                    # a surface every one of them has
            out[name if len(places) == 1 else f"{name}_{i}"] = {
                "npc": entry["npc"], "home": on,
                "scatter": {"on": on, "count": n, "spacing": float(entry.get("spacing", 1.2)),
                            "seed": rng.randrange(1 << 30), "radius": 0.4}}
    return out


OWNERSHIP = {
    "owns": "references: the things only this NPC uses (its armchair, its bed); it also likes them best. "
            "Several NPCs may own one thing (a couple their bed): only they use it",
    "prefers": "references: things it likes best, still free for others",
    "work": "a reference: where its routine's `at: work` blocks happen (an office, a shop, a room)",
}
OWN_BONUS = 1.6                      # how much more an NPC likes what it owns (score multiplier)
PREFER_BONUS = 1.3


def resolve_ownership(node: SceneNode, placement: dict[str, Any], refs: Any, keys: list) -> None:
    """An NPC placement's ``owns`` / ``prefers`` (lists of references, selectors too) and ``work``
    (one reference), kept as the nodes until export writes their stable IDs (``meta.npc_owns``)."""
    found: dict[str, list] = {}
    for key in ("owns", "prefers"):
        listed = placement.get(key) or []
        if not isinstance(listed, list):
            raise refs.fail(f"npc {key}: a list of references, got {listed!r}", [*keys, key])
        found[key] = [n for i, text in enumerate(listed) for n in refs.nodes(text, [*keys, key, i])]
    if "work" in placement:
        found["work"] = [refs.node(placement["work"], [*keys, "work"])]
    if any(found.values()):
        node.meta["npc_owns"] = found


def declare_quests(node: SceneNode, placement_quests: list, source: Any, assets_dir: Path) -> None:
    """An NPC's quests: its placement's ``quests`` (given at the start) and its routine blocks'
    (given while a block lasts; ``meta.npc_routine_quests``: ``{id, from, to, repeat?}`` per
    quest and block, for ``quests.collect``). In the exported routine a block's ``quests`` are
    the quests' ids."""
    from .quests import META, declare_scene

    routine = node.meta["npc"].get("routine", [])   # type: ignore[union-attr]
    files = [q["file"] for block in routine for q in block.get("quests", [])]
    if not placement_quests and not files:
        return
    declare_scene(node, {"quests": [*placement_quests, *files]}, source, assets_dir)
    ids = {e["source"]: str(e["spec"].get("id") or Path(e["source"]).stem) for e in node.meta.get(META) or []}
    found, blocks = [], []
    for block in routine:
        listed = []
        for q in block.get("quests", []):
            quest_id = ids[str(roots.find(q["file"], assets_dir))]
            listed.append(quest_id)
            found.append({"id": quest_id, "from": block["from"], "to": block["to"],
                          **({"repeat": q["repeat"]} if "repeat" in q else {})})
        blocks.append({**block, "quests": listed} if listed else block)
    if found:                       # (a copy: placed copies of one NPC may share the definition)
        node.meta["npc"] = {**node.meta["npc"], "routine": blocks}   # type: ignore[dict-item]
        node.meta["npc_routine_quests"] = found


CHARACTER_ACTOR = "character:{}"


def actor_id(node: SceneNode, uid: str) -> str:
    """The rules actor an NPC is: a named person (``meta.npc.character``) is ``character:<name>``
    wherever it is placed, so its quests and the state content packages keep for it follow it
    from scene to scene; any other NPC is its placement (``uid``, its stable ID). Runtimes use the
    same rule (``WorldLoader.actor_id``)."""
    character = (node.meta.get("npc") or {}).get("character")
    return CHARACTER_ACTOR.format(character) if character else uid


# Groups (geogen.87.3): people who came together. `together` is their need to be near each
# other (it refills within `radius` of a groupmate, in the runtime, and decays apart), `rejoin`
# the activity that walks back (members to the leader, the leader to its nearest member).
GROUP_DEFAULTS = {"radius": 4.0, "spread": 1.2}
GROUP_KEYS = {"size", "radius", "spread"}
TOGETHER = {"initial": 1.0, "decay": 0.02}
REJOIN = {"member": 0.9, "leader": 0.4}      # what rejoining advertises of `together`


def parse_group(spec: Any, where: str) -> dict[str, Any]:
    """A placement's ``group: {size: n | [lo, hi], radius, spread}``: how many people each scattered
    copy is (the leader counts), how near counts as together (m) and how far from the leader the
    others start (m)."""
    if not isinstance(spec, dict) or "size" not in spec or set(spec) - GROUP_KEYS:
        raise ValueError(f"{where}: group is {{size: n | [lo, hi], radius, spread}}, got {spec!r}")
    size = spec["size"]
    lo, hi = (size, size) if isinstance(size, (int, float)) else _range(size, f"{where}: size")
    if not 1 <= lo <= hi:
        raise ValueError(f"{where}: group size is at least 1 (the leader), got {size!r}")
    out = {"size": [int(lo), int(hi)]}
    for key, default in GROUP_DEFAULTS.items():
        out[key] = float(spec.get(key, default))
        if out[key] <= 0:
            raise ValueError(f"{where}: group {key} must be positive")
    return out


def join_group(node: SceneNode, leader: SceneNode, group: dict[str, Any]) -> None:
    """Make an NPC node one of ``leader``'s group (the leader joins its own first): the build-only
    ``meta.npc_group`` (exported as the leader's stable ID, :func:`group_extras`), and the
    ``together`` need and ``rejoin`` activity in its definition unless it has its own."""
    node.meta["npc_group"] = {"leader": leader, "radius": group["radius"]}
    npc: dict[str, Any] = node.meta["npc"]  # type: ignore[assignment]
    npc["needs"].setdefault("together", dict(TOGETHER))
    role = "leader" if node is leader else "member"
    npc["activities"].setdefault("rejoin", {"action": "rejoin", "advertises": {"together": REJOIN[role]}})


def group_extras(node: SceneNode) -> dict[str, Any]:
    """``meta.npc_group`` as ``npc.group``: {leader: stable ID, radius}; {} for no group."""
    group = node.meta.get("npc_group")
    if not group or "uid" not in group["leader"].meta:
        return {}
    return {"group": {"leader": group["leader"].meta["uid"], "radius": group["radius"]}}


def ownership_extras(node: SceneNode) -> dict[str, Any]:
    """``meta.npc_owns`` as the stable IDs the runtime matches affordances by: {owns, prefers, work}."""
    out: dict[str, Any] = {}
    for key, nodes in (node.meta.get("npc_owns") or {}).items():
        ids = [n.meta["uid"] for n in nodes if "uid" in n.meta]
        if ids:
            out[key] = ids[0] if key == "work" else ids
    return out


def _advertises(spec: Any, needs: dict | None, where: str) -> dict[str, float]:
    out = {str(k): float(v) for k, v in (spec or {}).items()}
    if needs is not None:
        unknown = set(out) - set(needs)
        if unknown:
            raise ValueError(f"{where}: advertises unknown needs {sorted(unknown)} (needs: {sorted(needs)})")
    return out


def build_npc(definition: dict[str, Any], loader, assets_dir: Path, seed: int | None = None) -> SceneNode:
    """The NPC prototype: a node (meta.type npc, meta.npc) with the body asset as a child named 'body'.

    The body is ``{asset, params}`` or ``{archetype, params}`` (a character
    archetype; ``seed`` picks the person, and also seeds the brain).
    """
    body_spec = definition["body"]
    if "archetype" in body_spec:
        from .characters import load_character

        body = load_character(loader, assets_dir, body_spec["archetype"], seed or 0, body_spec.get("params"))
    else:
        body = loader.load(roots.find(body_spec["asset"], assets_dir), params=body_spec.get("params"))
    body.name = "body"
    poses = body.meta.get("poses") or {"stand": parse_poses({"stand": {}})["stand"]}
    npc = copy.deepcopy(definition)
    npc["body"] = {**body_spec, "node": "body", "poses": poses}
    if seed is not None:
        npc["seed"] = int(seed)
    if "character" in body.meta:
        npc["body"]["character"] = body.meta["character"]
    # Collision capsule from the body's bounds (a humanoid is narrower than its arm span).
    size = np.asarray(body.size, dtype=np.float64)
    npc["height"] = round(float(size[1]), 4)
    npc["radius"] = round(float(min(size[0], size[2]) / 2), 4)
    # The whole action table (about 3 KB as JSON): besides its own activities an NPC runs whatever
    # action an affordance near it names, and the scene's affordances aren't known here.
    npc["actions"] = load_actions()
    node = SceneNode(name=definition["definition"], tags=["npc"], meta={"type": "npc", "npc": npc})
    node.add_child(body)
    return node


def resolve_home(node: SceneNode, spec: Any, refs: Any, keys: list | None = None) -> None:
    """Record the NPC's home region in its own frame: ``meta.npc.home = {polygon: [[x, z], ...], y}``.

    ``spec`` is ``<object>.<surface>`` (a horizontal surface such as a
    house floor; the object is a reference resolved by ``refs``, a
    ``geogen.refs.SceneRefs``, and ``keys`` the key path it was written at) or ``{rect: [x0, z0, x1, z1], y: <floor height>}`` in the scene
    frame (``y`` defaults to 0; set it for an upper storey). Without one, the
    NPC's home is a 6 m square around where it is placed.
    """
    npc: dict[str, Any] = node.meta["npc"]  # type: ignore[assignment]
    if spec is None:
        npc["home"] = {"polygon": [[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]], "y": 0.0}
        return
    npc["home"] = _region(node, spec, refs, list(keys or ["home"]), "home")


def resolve_visits(node: SceneNode, spec: Any, refs: Any, keys: list) -> None:
    """An NPC placement's ``visits: {name: <object>.<surface> | {rect, y}}``: places outside its
    home that its activities with ``visit: <name>`` make short trips to, recorded like the home
    (``meta.npc.visits = {name: {polygon, y}}``, in the NPC's frame). An activity whose visit a
    placement doesn't give isn't an option there (a scene without the chef's dining room)."""
    npc: dict[str, Any] = node.meta["npc"]  # type: ignore[assignment]
    spec = spec or {}
    if not isinstance(spec, dict):
        raise refs.fail(f"npc visits: {{name: <object>.<surface> | {{rect, y}}}}, got {spec!r}", keys)
    if spec:
        npc["visits"] = {str(k): _region(node, v, refs, [*keys, k], "visit") for k, v in spec.items()}


def _region(node: SceneNode, spec: Any, refs: Any, keys: list, what: str) -> dict[str, Any]:
    """A home or visit region, ``<object>.<surface>`` or ``{rect, y}``, as {polygon, y} in the NPC's frame."""
    to_local = np.linalg.inv(node.transform.to_matrix())
    if isinstance(spec, dict) and "rect" in spec:
        x0, z0, x1, z1 = (float(v) for v in spec["rect"])
        y = float(spec.get("y", 0.0))
        corners = [np.array([x, y, z]) for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))]
    elif isinstance(spec, str) and "." in spec:
        target, surface = refs.surface(spec, keys)
        if abs(surface.normal[1]) < 0.9:
            raise refs.fail(f"npc {what}: {spec} is not a horizontal surface", keys, spec)
        to_scene = np.linalg.inv(refs.root.world_transform()) @ target.world_transform()
        corners = []
        for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
            local = surface.resolve(u, v).translation
            corners.append((to_scene @ np.append(local, 1.0))[:3])
    else:
        raise ValueError(f"npc {what}: expected <object>.<surface> or {{rect: [...]}}, got {spec!r}")
    pts = np.array([(to_local @ np.append(c, 1.0))[:3] for c in corners])
    return {"polygon": [[round(float(p[0]), 4), round(float(p[2]), 4)] for p in pts],
            "y": round(float(pts[:, 1].mean()), 4)}
