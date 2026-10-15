"""Physics props: assets the runtime simulates as rigid bodies.

An asset with a ``physics:`` block is one dynamic body in the runtime: it falls, is pushed
and, with ``carry``, can be picked up and carried. The build records it as ``meta.physics``
(exported in extras): ``mass`` (kg), ``friction``, ``bounce``, ``carry`` and the collision
box (``size`` and ``center`` in the asset's frame, from its geometry).

``spawn: "<thing>.<attachment>"`` makes the prop one a machine hands out (a dropper's
cube). It gets a ``spawn`` interaction, ``stowed | live`` (``initial``, default
``stowed``): stowed, it is hidden and out of the simulation; live, the runtime puts it at
that attachment, at rest, and lets it go. Rules drive the state (it has no ``next``), and
the runtime stows a live prop that falls out of the world, so a rule on
``{changed: cube.spawn, to: stowed}`` can hand out another. The attachment is found as a
``move`` motion's is, from the asset's parent scope outward (usually a text param the
scene fills: ``params: {from: dropper.spawn_at}``), and is recorded as ``spawn_at`` in the
asset's frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.node import SceneNode

KEYS = {
    "mass": "kg (default 10)",
    "friction": "0-1 (default 0.8)",
    "bounce": "0-1 (default 0)",
    "carry": "the player can pick it up and carry it (default false)",
    "spawn": "\"<thing>.<attachment>\": handed out there by rules (a `spawn` interaction, stowed | live)",
    "initial": "the spawn interaction's first state: stowed (default) or live",
}
SPAWN = "spawn"
STATES = ("stowed", "live")


def parse_physics(spec: Any, root: SceneNode, part_nodes: dict[str, SceneNode], name: str) -> dict[str, Any]:
    """Normalise an asset's ``physics:`` block; the box comes from the asset's geometry."""
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError(f"{name}: physics is {{{', '.join(KEYS)}}}, got {spec!r}")
    unknown = set(spec) - set(KEYS)
    if unknown:
        raise ValueError(f"{name}: physics: unknown keys {sorted(unknown)}; known: {sorted(KEYS)}")
    out: dict[str, Any] = {
        "mass": float(spec.get("mass", 10.0)),
        "friction": float(spec.get("friction", 0.8)),
        "bounce": float(spec.get("bounce", 0.0)),
        "carry": bool(spec.get("carry", False)),
    }
    if out["mass"] <= 0:
        raise ValueError(f"{name}: physics mass must be positive")
    box = _box(root)
    if box is None:
        raise ValueError(f"{name}: physics needs geometry to make a body of")
    lo, hi = box
    out["size"] = [round(float(v), 6) for v in hi - lo]
    out["center"] = [round(float(v), 6) for v in (lo + hi) / 2]
    spawn = spec.get("spawn")
    if spawn is not None and str(spawn):
        initial = str(spec.get("initial", "stowed"))
        if initial not in STATES:
            raise ValueError(f"{name}: physics initial is stowed or live, got {initial!r}")
        if any(i.name == SPAWN for i in root.interactions):
            raise ValueError(f"{name}: physics spawn adds the '{SPAWN}' interaction, which the asset already has")
        out["spawn"] = str(spawn)
        root.interactions.append(_spawn_interaction(root, part_nodes, initial))
    return out


def _box(root: SceneNode) -> tuple[np.ndarray, np.ndarray] | None:
    inv = np.linalg.inv(root.world_transform())
    pts = [(inv @ n.world_transform() @ np.c_[n.mesh.vertices, np.ones(len(n.mesh.vertices))].T).T[:, :3]
           for n in root.iter_nodes() if n.mesh is not None and len(n.mesh.vertices)]
    if not pts:
        return None
    allp = np.vstack(pts)
    return allp.min(axis=0), allp.max(axis=0)


def _spawn_interaction(root: SceneNode, part_nodes: dict[str, SceneNode], initial: str):
    from .layout.interactions import Interaction, Motion, State, apply_state

    tops = [node for node in part_nodes.values() if node.parent is root]
    motion = Motion(tops, "visible", np.zeros(3), np.zeros(3), {"stowed": 0.0, "live": 1.0})
    interaction = Interaction(name=SPAWN, states={s: State() for s in STATES}, initial=initial,
                              motions=[motion], targets=[], duration=0.0)
    apply_state(root, interaction, initial)
    return interaction


def resolve_spawns(root: SceneNode, final: bool = False) -> None:
    """Record ``spawn_at`` (asset frame) for every prop under ``root`` whose ``spawn``
    attachment can be found; ``final`` (the outermost scene) makes a missing one an error."""
    from . import refs
    from .layout.interactions import _attachment_point

    missing = []
    for node in root.iter_nodes():
        physics = node.meta.get("physics")
        if not isinstance(physics, dict) or "spawn" not in physics or "spawn_at" in physics:
            continue
        where = _attachment_point(node, physics["spawn"], refs)
        if where is None:
            missing.append(f"{node.name}: physics spawn '{physics['spawn']}' isn't a placed thing's "
                           f"attachment (<thing>.<attachment>) from here")
            continue
        local = np.linalg.inv(node.world_transform()) @ np.r_[where, 1.0]
        node.meta["physics"] = {**physics, "spawn_at": [round(float(v), 6) for v in local[:3]]}
    if final and missing:
        raise ValueError("\n".join(missing))
