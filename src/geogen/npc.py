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
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from . import roots
from .core.node import SceneNode
from .layout.yaml_utils import load_kind, safe_load_path

ASSETS_DIR = roots.CORE
ACTIONS_PATH = ASSETS_DIR / "npcs" / "actions.yaml"

NPC_KIND = "npc"
ACTIONS_KIND = "npc_actions"
VERSION = 1

STEP_KEYS = {"go_to", "face", "pose", "wait", "use", "vanish", "play"}
STEP_OPTIONS = {"at", "if", "hold", "loop", "close"}
GO_TO_TARGETS = {"approach", "anchor", "random", "near", "far", "exit", "aside"}
FACE_TARGETS = {"anchor", "portal", "player"}
POSE_AT = {"anchor", "approach"}

NPC_KEYS = {"kind", "version", "body", "speed", "turn_speed", "radius", "needs", "preferences",
            "activities", "scoring", "home_margin", "flags", "seed", "affordance_tags", "wander", "routine", "attention"}
CHARACTER_KIND = "character"
# Placement keys a character file may give defaults for (the scene's placement wins).
CHARACTER_PLACEMENT_KEYS = ("home", "work", "owns", "prefers", "quests", "seed")
CHARACTER_KEYS = {"name", "npc", "body", *CHARACTER_PLACEMENT_KEYS,
                  *(NPC_KEYS - {"kind", "version", "body", "seed"})}
ROUTINE_KEYS = {"from", "to", "away", "activities", "preferences", "name", "at", "only", "hold"}
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
    if verb == "go_to" and arg not in GO_TO_TARGETS:
        raise ValueError(f"{where}: go_to must be one of {sorted(GO_TO_TARGETS)}, got {arg!r}")
    if verb == "face" and arg not in FACE_TARGETS:
        raise ValueError(f"{where}: face must be one of {sorted(FACE_TARGETS)}, got {arg!r}")
    if "hold" in step and (verb != "pose" or not isinstance(step["hold"], str) or not step["hold"]):
        raise ValueError(f"{where}: hold names the body pose to hold in place of a pose step's own, got {step!r}")
    if "close" in step and (verb != "go_to" or step["close"] is not True):
        raise ValueError(f"{where}: close: true is a go_to step's option (step up to the target off the navmesh)")
    if "loop" in step and verb != "play":
        raise ValueError(f"{where}: loop is a play step's option (how long to loop its clip), got {step!r}")
    if "at" in step and step["at"] not in POSE_AT:
        raise ValueError(f"{where}: at must be one of {sorted(POSE_AT)}, got {step['at']!r}")
    out = dict(step)
    if verb == "vanish" and arg is not True:
        raise ValueError(f"{where}: vanish: true")
    if verb == "wait" and arg != "duration":
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
        unknown = set(spec) - {"steps", "approach", "duration"}
        if unknown:
            raise ValueError(f"action '{name}': unknown keys {sorted(unknown)}")
        steps = [_check_step(s, f"action '{name}' step {i}") for i, s in enumerate(spec.get("steps") or [])]
        if not steps:
            raise ValueError(f"action '{name}': needs steps")
        action: dict[str, Any] = {"steps": steps}
        if "approach" in spec:
            action["approach"] = _vec3(spec["approach"], f"action '{name}': approach")
        if "duration" in spec:
            action["duration"] = _range(spec["duration"], f"action '{name}': duration")
        actions[name] = action
    return actions


@lru_cache(maxsize=4)
def load_actions(path: str | Path = ACTIONS_PATH) -> dict[str, dict[str, Any]]:
    return parse_actions(safe_load_path(path) or {})


def parse_poses(spec: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """A body asset's ``poses:`` -> {name: {offset, rotation (deg), scale}} (all present)."""
    poses = {}
    for name, pose in (spec or {}).items():
        pose = pose or {}
        unknown = set(pose) - POSE_KEYS
        if unknown:
            raise ValueError(f"pose '{name}': unknown keys {sorted(unknown)}; known: {sorted(POSE_KEYS)}")
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
    unknown = set(spec) - PORTAL_KEYS
    if unknown:
        raise ValueError(f"portal: unknown keys {sorted(unknown)}; known: {sorted(PORTAL_KEYS)}")
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
        needs[need] = {"initial": float(spec.get("initial", 0.5)), "decay": float(spec.get("decay", 0.0))}
        if not 0.0 <= needs[need]["initial"] <= 1.0:
            raise ValueError(f"{path}: need {need}: initial must be 0..1")
    activities = {}
    for name, spec in (data.get("activities") or {}).items():
        action = spec.get("action", name)
        if action not in actions:
            raise ValueError(f"{path}: activity {name}: unknown action {action!r}")
        activity = {"action": action, "advertises": _advertises(spec.get("advertises"), needs, f"{path}: {name}")}
        if "duration" in spec:
            activity["duration"] = _range(spec["duration"], f"{path}: {name}: duration")
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
        # Where `go_to: random` goes: the home region, or points on surfaces with these tags.
        "wander": {"tags": [str(t) for t in (data.get("wander") or {}).get("tags", [])]},
        "routine": parse_routine(data.get("routine") or [], activities, f"{path}: routine"),
        "attention": attention,
    }


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
        unknown = set(spec) - set(defaults)
        if unknown:
            raise ValueError(f"{where}: {part}: unknown keys {sorted(unknown)}")
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
            hours = int(hh) + (int(mm) if mm else 0) / 60.0
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

    ``activities`` / ``preferences`` multiply the scores of the NPC's own activities / of
    affordances with those tags; ``away`` sends the NPC off (the ``leave`` action) until the
    block ends; ``at: work`` keeps its options to the affordances inside the placement's
    ``work`` (a reference), wherever that is from home. ``only`` keeps the options to affordances
    with one of those tags (``owns``: the things the NPC owns), none of its own activities, and
    ``hold`` keeps it on the one it takes until the block ends (in its bed all night) instead of
    the affordance's duration. With nothing in ``only`` to take, it carries on as without them.
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
                    **({"hold": True} if block.get("hold", False) else {})})
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
    "owns": "references: the things only this NPC uses (its armchair, its bed); it also likes them best",
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
    # Only the actions this NPC can run: its activities, anything an
    # affordance may name, and 'pass' for doors.
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
    to_local = np.linalg.inv(node.transform.to_matrix())
    if spec is None:
        npc["home"] = {"polygon": [[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]], "y": 0.0}
        return
    if isinstance(spec, dict) and "rect" in spec:
        x0, z0, x1, z1 = (float(v) for v in spec["rect"])
        y = float(spec.get("y", 0.0))
        corners = [np.array([x, y, z]) for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))]
    elif isinstance(spec, str) and "." in spec:
        keys = list(keys or ["home"])
        target, surface = refs.surface(spec, keys)
        if abs(surface.normal[1]) < 0.9:
            raise refs.fail(f"npc home: {spec} is not a horizontal surface", keys, spec)
        to_scene = np.linalg.inv(refs.root.world_transform()) @ target.world_transform()
        corners = []
        for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
            local = surface.resolve(u, v).translation
            corners.append((to_scene @ np.append(local, 1.0))[:3])
    else:
        raise ValueError(f"npc home: expected <object>.<surface> or {{rect: [...]}}, got {spec!r}")
    pts = np.array([(to_local @ np.append(c, 1.0))[:3] for c in corners])
    npc["home"] = {"polygon": [[round(float(p[0]), 4), round(float(p[2]), 4)] for p in pts],
                   "y": round(float(pts[:, 1].mean()), 4)}
