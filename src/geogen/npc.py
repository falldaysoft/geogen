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

from .core.node import SceneNode
from .layout.yaml_utils import safe_load_path

ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"
ACTIONS_PATH = ASSETS_DIR / "npcs" / "actions.yaml"

NPC_KIND = "npc"
ACTIONS_KIND = "npc_actions"
VERSION = 1

STEP_KEYS = {"go_to", "face", "pose", "wait", "use"}
STEP_OPTIONS = {"at", "if"}
GO_TO_TARGETS = {"approach", "anchor", "random", "near", "far"}
FACE_TARGETS = {"anchor", "portal"}
POSE_AT = {"anchor", "approach"}

NPC_KEYS = {"kind", "version", "body", "speed", "turn_speed", "radius", "needs", "preferences",
            "activities", "scoring", "home_margin", "flags", "seed"}
SCORING_DEFAULTS = {"distance": 0.02, "recency": 0.4, "memory": 120.0, "noise": 0.05, "retry": 30.0}
POSE_KEYS = {"offset", "rotation", "scale"}
PORTAL_KEYS = {"interaction", "open", "closed", "center", "normal", "width", "height", "depth", "clearance"}


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
    if "at" in step and step["at"] not in POSE_AT:
        raise ValueError(f"{where}: at must be one of {sorted(POSE_AT)}, got {step['at']!r}")
    out = dict(step)
    if verb == "wait" and arg != "duration":
        out["wait"] = _range(arg, f"{where}: wait")
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


def parse_portal(spec: dict[str, Any], interactions: list) -> dict[str, Any]:
    """A door's ``portal:``: the opening an NPC path may cross (asset frame).

    ``{interaction: swing, open: open, closed: closed, center: [x, y, z],
    normal: [0, 0, 1], width, height, depth, clearance}``: the opening is
    ``width`` x ``height`` centred on ``center``, ``depth`` thick along
    ``normal``; NPCs stop ``clearance`` beyond either face to operate it
    (clear of the swinging leaf).
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
    data = safe_load_path(path) or {}
    if data.get("kind") != NPC_KIND or int(data.get("version", 0)) != VERSION:
        raise ValueError(f"{path}: expected kind '{NPC_KIND}' version {VERSION}")
    unknown = set(data) - NPC_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; known: {sorted(NPC_KEYS)}")
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
    body = data.get("body") or {"asset": "characters/capsule.yaml"}
    if isinstance(body, str):
        body = {"asset": body}
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
    }


def _advertises(spec: Any, needs: dict | None, where: str) -> dict[str, float]:
    out = {str(k): float(v) for k, v in (spec or {}).items()}
    if needs is not None:
        unknown = set(out) - set(needs)
        if unknown:
            raise ValueError(f"{where}: advertises unknown needs {sorted(unknown)} (needs: {sorted(needs)})")
    return out


def build_npc(definition: dict[str, Any], loader, assets_dir: Path) -> SceneNode:
    """The NPC prototype: a node (meta.type npc, meta.npc) with the body asset as a child named 'body'."""
    body_spec = definition["body"]
    body = loader.load(Path(assets_dir) / body_spec["asset"], params=body_spec.get("params"))
    body.name = "body"
    poses = body.meta.get("poses") or {"stand": parse_poses({"stand": {}})["stand"]}
    npc = copy.deepcopy(definition)
    npc["body"] = {**body_spec, "node": "body", "poses": poses}
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


def resolve_home(node: SceneNode, spec: Any, loaded: dict[str, SceneNode]) -> None:
    """Record the NPC's home region in its own frame: ``meta.npc.home = {polygon: [[x, z], ...], y}``.

    ``spec`` is ``<object>.<surface>`` (a horizontal surface such as a
    house floor) or ``{rect: [x0, z0, x1, z1]}`` in the scene frame. Without
    one, the NPC's home is a 6 m square around where it is placed.
    """
    npc: dict[str, Any] = node.meta["npc"]  # type: ignore[assignment]
    to_local = np.linalg.inv(node.transform.to_matrix())
    if spec is None:
        npc["home"] = {"polygon": [[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]], "y": 0.0}
        return
    if isinstance(spec, dict) and "rect" in spec:
        x0, z0, x1, z1 = (float(v) for v in spec["rect"])
        corners = [np.array([x, 0.0, z]) for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))]
    elif isinstance(spec, str) and "." in spec:
        target_name, surface_name = spec.split(".", 1)
        target = loaded.get(target_name)
        if target is None:
            raise ValueError(f"npc home: unknown object {target_name!r} (known: {sorted(loaded)})")
        surface = target.surfaces.get(surface_name)
        if surface is None:
            raise ValueError(f"npc home: {target_name} has no surface {surface_name!r} "
                             f"(surfaces: {target.list_surfaces()})")
        if abs(surface.normal[1]) < 0.9:
            raise ValueError(f"npc home: {spec} is not a horizontal surface")
        to_scene = target.transform.to_matrix()
        corners = []
        for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
            local = surface.resolve(u, v).translation
            corners.append((to_scene @ np.append(local, 1.0))[:3])
    else:
        raise ValueError(f"npc home: expected <object>.<surface> or {{rect: [...]}}, got {spec!r}")
    pts = np.array([(to_local @ np.append(c, 1.0))[:3] for c in corners])
    npc["home"] = {"polygon": [[round(float(p[0]), 4), round(float(p[2]), 4)] for p in pts],
                   "y": round(float(pts[:, 1].mean()), 4)}
