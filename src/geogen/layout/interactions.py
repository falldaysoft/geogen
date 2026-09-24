"""Declarative interactions: small state machines that move parts of an asset.

YAML (asset level, ``{expr}`` params work as everywhere else)::

    interactions:
      swing:
        targets: [leaf, knob]      # parts the player aims at (default: moving parts)
        initial: closed            # default: first state
        duration: 0.9              # seconds for the widest move (constant speed)
        states:
          closed: { prompt: Open,  next: open }
          open:   { prompt: Close, next: closed }
        motions:
          - parts: [leaf, knob]
            rotate: y              # x|y|z|-x|... or [x, y, z], asset frame
            pivot: [-0.4, 0, -0.1] # metres in the asset frame, or <part>.<anchor>
            values: { open: 95 }   # degrees (rotate) or metres (translate) per state

State keys: ``next`` (the state ``use`` heads to when this is the target
state; absent = not usable, e.g. a latched button), ``then`` (auto-advance on
arrival, for momentary buttons), ``emit`` (event name raised on arrival),
``prompt`` (UI text). ``use`` applies ``next`` of the *target* state, so a
moving door reverses mid-swing.

Exported node transforms show the ``initial`` state; runtimes recover each
part's rest pose as ``M(initial_value)^-1 * exported`` (see export.py and the
Godot runtime's interaction.gd).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.node import SceneNode

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


@dataclass
class State:
    next: str | None = None
    then: str | None = None
    emit: str | None = None
    prompt: str | None = None

    def to_dict(self) -> dict[str, str]:
        return {k: v for k, v in (("next", self.next), ("then", self.then),
                                  ("emit", self.emit), ("prompt", self.prompt)) if v is not None}


@dataclass
class Motion:
    """Move ``parts`` about ``pivot``/along ``axis`` (asset frame) by a value per state."""

    parts: list[SceneNode]
    kind: str                       # "rotate" (degrees) | "translate" (metres)
    axis: np.ndarray
    pivot: np.ndarray
    values: dict[str, float]
    # Value the parts are currently posed at (0 = as authored).
    applied: float = 0.0

    def matrix(self, value: float) -> np.ndarray:
        """Asset-frame transform for ``value``."""
        m = np.eye(4)
        if self.kind == "translate":
            m[:3, 3] = self.axis * value
            return m
        a = self.axis / np.linalg.norm(self.axis)
        t = np.radians(value)
        k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        r = np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)
        m[:3, :3] = r
        m[:3, 3] = self.pivot - r @ self.pivot
        return m


@dataclass
class Interaction:
    name: str
    states: dict[str, State]
    motions: list[Motion]
    targets: list[SceneNode] = field(default_factory=list)
    initial: str = ""
    duration: float = 0.9

    def value(self, motion: Motion, state: str) -> float:
        return float(motion.values.get(state, 0.0))

    def to_extras(self, name_of: Callable[[SceneNode], str]) -> dict[str, Any]:
        """JSON for extras.geogen.interactions[name], nodes by exported name."""
        return {
            "targets": [name_of(n) for n in self.targets],
            "initial": self.initial,
            "duration": self.duration,
            "states": {k: s.to_dict() for k, s in self.states.items()},
            "motions": [{
                "nodes": [name_of(n) for n in m.parts],
                "type": m.kind,
                "axis": [float(v) for v in m.axis],
                "pivot": [round(float(v), 6) for v in m.pivot],
                "values": {s: self.value(m, s) for s in self.states},
            } for m in self.motions],
        }


def apply_state(root: SceneNode, interaction: Interaction, state: str) -> None:
    """Pose ``interaction``'s parts for ``state`` (parts are authored at value 0)."""
    root_world = root.world_transform()
    root_inv = np.linalg.inv(root_world)
    for motion in interaction.motions:
        value = interaction.value(motion, state)
        m = motion.matrix(value) @ np.linalg.inv(motion.matrix(motion.applied))
        motion.applied = value
        if np.allclose(m, np.eye(4)):
            continue
        for part in motion.parts:
            in_root = root_inv @ part.world_transform()
            new_world = root_world @ m @ in_root
            parent_world = part.parent.world_transform() if part.parent is not None else np.eye(4)
            part.transform = _to_transform(np.linalg.inv(parent_world) @ new_world, part.transform.scale)


def _to_transform(matrix: np.ndarray, scale: np.ndarray):
    """Rigid 4x4 (+ known scale) -> Transform with XYZ Euler rotation."""
    from ..core.transform import Transform

    r = matrix[:3, :3] / np.where(scale == 0, 1, scale)[None, :]
    # R = Rz @ Ry @ Rx for XYZ Euler (see Transform.to_matrix).
    ry = np.arcsin(np.clip(-r[2, 0], -1, 1))
    if abs(np.cos(ry)) > 1e-9:
        rx = np.arctan2(r[2, 1], r[2, 2])
        rz = np.arctan2(r[1, 0], r[0, 0])
    else:
        rx = np.arctan2(-r[1, 2], r[1, 1])
        rz = 0.0
    return Transform(translation=matrix[:3, 3].copy(), rotation=np.array([rx, ry, rz]), scale=scale.copy())


def _axis(spec: Any) -> np.ndarray:
    if isinstance(spec, str):
        sign = -1.0 if spec.startswith("-") else 1.0
        key = spec.lstrip("+-")
        if key not in _AXES:
            raise ValueError(f"axis must be x|y|z (optionally signed) or [x, y, z], got {spec!r}")
        return sign * np.array(_AXES[key])
    axis = np.asarray(spec, dtype=np.float64)
    if axis.shape != (3,) or np.linalg.norm(axis) == 0:
        raise ValueError(f"axis must be a non-zero [x, y, z], got {spec!r}")
    return axis / np.linalg.norm(axis)


def parse_interactions(data: dict[str, Any], root: SceneNode, parts: dict[str, SceneNode]) -> list[Interaction]:
    """Build interactions from an asset's ``interactions:`` block (parts already placed)."""
    from .anchors import resolve_anchor

    root_inv = np.linalg.inv(root.world_transform())
    result = []
    for name, spec in (data or {}).items():
        states_spec = spec.get("states") or {}
        if not states_spec:
            raise ValueError(f"Interaction '{name}' needs states:")
        states = {}
        for sname, sdef in states_spec.items():
            sdef = sdef or {}
            unknown = set(sdef) - {"next", "then", "emit", "prompt"}
            if unknown:
                raise ValueError(f"Interaction '{name}' state '{sname}': unknown keys {sorted(unknown)}")
            states[sname] = State(**{k: (str(v) if v is not None else None) for k, v in sdef.items()})
        for sname, state in states.items():
            for ref in (state.next, state.then):
                if ref is not None and ref not in states:
                    raise ValueError(f"Interaction '{name}' state '{sname}' refers to unknown state '{ref}'")

        def part(pname: str) -> SceneNode:
            if pname not in parts:
                raise ValueError(f"Interaction '{name}' references unknown part '{pname}'."
                                 f" Parts: {sorted(parts)}")
            return parts[pname]

        motions = []
        for mspec in spec.get("motions", []):
            if ("rotate" in mspec) == ("translate" in mspec):
                raise ValueError(f"Interaction '{name}': each motion needs exactly one of rotate/translate")
            kind = "rotate" if "rotate" in mspec else "translate"
            pivot_spec = mspec.get("pivot", [0, 0, 0])
            if isinstance(pivot_spec, str):
                pname, _, anchor = pivot_spec.partition(".")
                node = part(pname)
                size = node.size if node.size is not None else np.zeros(3)
                # Anchors are bottom-centre based; part meshes are centred on the part origin.
                local = resolve_anchor(anchor or "center", size) - np.array([0.0, size[1] / 2, 0.0])
                pivot = (root_inv @ node.world_transform() @ np.r_[local, 1.0])[:3]
            else:
                pivot = np.asarray(pivot_spec, dtype=np.float64)
            values = {str(k): float(v) for k, v in (mspec.get("values") or {}).items()}
            for sname in values:
                if sname not in states:
                    raise ValueError(f"Interaction '{name}' motion value for unknown state '{sname}'")
            motions.append(Motion([part(p) for p in mspec.get("parts", [])], kind,
                                  _axis(mspec[kind]), pivot, values))
        if not motions:
            raise ValueError(f"Interaction '{name}' needs at least one motion")

        targets = [part(p) for p in spec.get("targets", [])] or [p for m in motions for p in m.parts]
        initial = str(spec.get("initial", next(iter(states))))
        if initial not in states:
            raise ValueError(f"Interaction '{name}' initial state '{initial}' is not a state")
        result.append(Interaction(name, states, motions, targets, initial, float(spec.get("duration", 0.9))))
    return result
