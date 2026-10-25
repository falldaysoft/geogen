"""Mechanisms: assets whose parts form several rigid bodies joined by joints, some motorised.

A ``physics:`` block makes a whole asset one prop (geogen.physics). A ``mechanism:`` block
instead names groups of parts that the runtime simulates as separate rigid bodies, each
held to the asset's fixed parts, or to another body, by a joint: a carousel's platform on a
motorised hinge, a Ferris wheel's gondolas swinging on free hinges from the wheel, a drop
tower's carriage on a slide. Parts in no body stay static scenery. YAML (asset level,
``{expr}`` params work as everywhere else)::

    mechanism:
      bodies:
        wheel:
          parts: [rim, spoke]            # part names (an arrayed part: all its copies)
          mass: 800                      # kg (default 100); friction, bounce, damping as physics:
          joint:
            type: hinge                  # hinge (turns about axis) | slide (moves along axis) | pin (a ball joint)
            to: frame                    # another body (default: the asset's fixed parts, i.e. the world)
            axis: z                      # hinge / slide: x|y|z (signed) or [x, y, z], asset frame
            pivot: hub.center            # [x, y, z] in the asset frame, or <part>.<anchor>
            limits: [-30, 30]            # degrees (hinge) or metres (slide); absent: free
            motor: {speed: 12, force: 5000}   # a motor at speed deg/s or m/s (interactions set it; see below)
        gondola:
          parts: [hanger, cabin]         # arrayed parts (array: {count: 8, ...})
          array: true                    # one body per copy: gondola_0 .. gondola_7, of hanger_0, cabin_0, ...
          joint: {type: hinge, to: wheel, axis: z, pivot: hanger.top}   # the copy's own hanger
        link:
          parts: [ring]                  # a chain: each copy pinned to the one before, the first to the frame
          array: true
          joint: {type: pin, to: previous, pivot: ring.top}

An interaction drives a motor with a ``drive`` motion (geogen.layout.interactions)::

    motions:
      - {body: wheel, drive: {running: 12, stopped: 0}, free: [coasting]}

whose values are the motor's target speed per state (ramped over the interaction's
``duration``, like any motion), ``0`` a brake, and ``free`` lists states in which the motor
is off (a drop tower's fall). A body's ``motor.speed`` is only a default for the drive's
values. Affordances may name a ``body`` so a seated player rides with it.

Build: ``meta.mechanism`` on the asset, exported in extras (bodies' parts by exported node
name). Runtime (``runtime/godot/scripts/mechanisms.gd``): each body is a RigidBody3D with
convex shapes from its parts' colliders (concave shapes can't be dynamic: ``collider: mesh``
on a body part becomes a hull), each joint a Godot joint with the two bodies excluded from
colliding with each other; standing on a body carries the player with it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.node import SceneNode
from .layout.interactions import _axis

KEYS = {"bodies": "{name: {parts, mass, friction, bounce, damping, joint}}: the rigid bodies"}
BODY_KEYS = {
    "parts": "the asset's parts that make up this body (an arrayed part: all its copies)",
    "array": "true: the parts are arrayed (the same count); one body per copy, <name>_<i>, of the parts' copies, "
             "its joint pivot <part>.<anchor> taken from the copy",
    "mass": "kg (default 100)",
    "friction": "0-1 (default 0.8)",
    "bounce": "0-1 (default 0)",
    "damping": "{linear, angular}: how fast it slows, per second (default 0, 0)",
    "joint": "{type: hinge | slide | pin, to, axis, pivot, limits, motor}: what holds it (absent: a free body)",
}
JOINT_KEYS = {
    "type": "hinge (turns about axis) | slide (moves along axis) | pin (a ball joint at pivot)",
    "to": "the body it is joined to (default: the asset's fixed parts); in an arrayed body, `previous`: the copy before "
          "(the first to the fixed parts), a chain",
    "axis": "hinge / slide: x|y|z (signed) or [x, y, z] in the asset frame (an arrayed body's turns with its copy)",
    "pivot": "[x, y, z] in the asset frame, or <part>.<point> (an attachment point of the part, top / front / ..., "
             "or an anchor, top_center; default: the asset origin)",
    "limits": "[lo, hi]: degrees (hinge) or metres (slide) the joint may move from rest (absent: free)",
    "motor": "{speed: deg/s | m/s, force: N·m | N}: a motor an interaction's drive motion runs",
}
JOINT_TYPES = ("hinge", "slide", "pin")
DEFAULT_FORCE = 10000.0


def parse_mechanism(spec: Any, root: SceneNode, parts: dict[str, SceneNode], name: str,
                    arrays: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Normalise an asset's ``mechanism:`` block (parts already placed) and check the
    interactions' drive motions against it."""
    from .layout.anchors import resolve_anchor

    if not isinstance(spec, dict) or set(spec) - set(KEYS) or not isinstance(spec.get("bodies"), dict) or not spec["bodies"]:
        raise ValueError(f"{name}: mechanism is {{bodies: {{<name>: {{{', '.join(BODY_KEYS)}}}}}}}, got {spec!r}")
    root_inv = np.linalg.inv(root.world_transform())
    bodies: dict[str, dict[str, Any]] = {}
    owner: dict[str, str] = {}
    for bname, bspec in spec["bodies"].items():
        bname = str(bname)
        where = f"{name}: mechanism body '{bname}'"
        if not isinstance(bspec, dict):
            raise ValueError(f"{where} is {{{', '.join(BODY_KEYS)}}}, got {bspec!r}")
        unknown = set(bspec) - set(BODY_KEYS)
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}; known: {sorted(BODY_KEYS)}")
        listed = bspec.get("parts")
        listed = [listed] if isinstance(listed, str) else [str(p) for p in (listed or [])]
        if not listed:
            raise ValueError(f"{where} needs parts: [...]")
        # One body of every listed part (and an arrayed part's copies), or with `array`, a body
        # per copy: the listed parts must all be arrayed the same number of times.
        groups: list[tuple[str, list[str], str | None]] = []
        if bspec.get("array"):
            counts = {len((arrays or {}).get(p, [])) for p in listed}
            if len(counts) != 1 or not next(iter(counts)):
                raise ValueError(f"{where}: array: true needs every part arrayed, with the same count; got {listed}")
            for i in range(next(iter(counts))):
                groups.append((f"{bname}_{i}", [f"{p}_{i}" for p in listed], f"_{i}"))
        else:
            groups.append((bname, [c for p in listed for c in (arrays or {}).get(p, [p])], None))
        for cname, copies, suffix in groups:
            nodes = []
            for copy in copies:
                if copy not in parts:
                    raise ValueError(f"{where}: unknown part '{copy}'. Parts: {sorted(parts)}")
                if copy in owner:
                    raise ValueError(f"{where}: part '{copy}' is already in body '{owner[copy]}'")
                owner[copy] = cname
                node = parts[copy]
                if node.mesh is None:
                    raise ValueError(f"{where}: part '{copy}' has no geometry to make a body of")
                # A dynamic body can't be concave: its parts collide as hulls (or the box they ask for).
                if str(node.meta.get("collider", "auto")) in ("auto", "mesh"):
                    node.meta["collider"] = "hull"
                nodes.append(copy)
            body: dict[str, Any] = {
                "nodes": nodes,
                "mass": float(bspec.get("mass", 100.0)),
                "friction": float(bspec.get("friction", 0.8)),
                "bounce": float(bspec.get("bounce", 0.0)),
            }
            if body["mass"] <= 0:
                raise ValueError(f"{where}: mass must be positive")
            damping = bspec.get("damping")
            if damping is not None:
                if not isinstance(damping, dict) or set(damping) - {"linear", "angular"}:
                    raise ValueError(f"{where}: damping is {{linear, angular}}, got {damping!r}")
                body["damping"] = {k: float(damping.get(k, 0.0)) for k in ("linear", "angular")}
                if min(body["damping"].values()) < 0:
                    raise ValueError(f"{where}: damping can't be negative")
            if bspec.get("joint") is not None:
                body["joint"] = _parse_joint(bspec["joint"], where, parts, root_inv, resolve_anchor, suffix)
                turn = parts[copies[0]].meta.get("array_turn") if suffix else None
                if turn and "axis" in body["joint"]:
                    # A polar copy's joint turns with it: the swing chairs' hinges stay tangential.
                    body["joint"]["axis"] = [round(float(v), 6) for v in _turned(body["joint"]["axis"], turn)]
                if body["joint"].get("to") == "previous":
                    if suffix is None:
                        raise ValueError(f"{where}: joint to: previous needs array: true (a chain of copies)")
                    index = int(suffix[1:])
                    if index == 0:
                        del body["joint"]["to"]                 # the first link hangs from the fixed parts
                    else:
                        body["joint"]["to"] = f"{bname}_{index - 1}"
            bodies[cname] = body
    for bname, body in bodies.items():
        to = body.get("joint", {}).get("to")
        if to is not None and to not in bodies:
            raise ValueError(f"{name}: mechanism body '{bname}' is joined to unknown body '{to}'; bodies: {sorted(bodies)}")
        if to == bname:
            raise ValueError(f"{name}: mechanism body '{bname}' can't be joined to itself")
    for interaction in root.interactions:
        for motion in interaction.motions:
            if motion.kind != "drive":
                continue
            if motion.body not in bodies:
                raise ValueError(f"{name}: interaction '{interaction.name}' drives unknown body '{motion.body}'; "
                                 f"bodies: {sorted(bodies)}")
            joint = bodies[motion.body].get("joint")
            if joint is None or joint["type"] == "pin":
                raise ValueError(f"{name}: interaction '{interaction.name}' drives body '{motion.body}', which has "
                                 f"no hinge or slide joint to put a motor on")
            joint.setdefault("motor", {"speed": 0.0, "force": DEFAULT_FORCE})
    return {"bodies": bodies}


def _turned(axis, turn: dict[str, Any]) -> np.ndarray:
    """``axis`` rotated by a polar array copy's ``turn`` ({axis: x|y|z, angle: degrees})."""
    a = np.radians(turn["angle"])
    c, s = np.cos(a), np.sin(a)
    x, y, z = (float(v) for v in axis)
    if turn["axis"] == "y":
        return np.array([c * x + s * z, y, -s * x + c * z])
    if turn["axis"] == "z":
        return np.array([c * x - s * y, s * x + c * y, z])
    return np.array([x, c * y - s * z, s * y + c * z])


def _parse_joint(jspec: Any, where: str, parts: dict[str, SceneNode], root_inv: np.ndarray, resolve_anchor,
                 suffix: str | None = None) -> dict[str, Any]:
    """``suffix``: an arrayed body's copy (``_3``): a ``<part>.<anchor>`` pivot names that copy's part."""
    if not isinstance(jspec, dict):
        raise ValueError(f"{where}: joint is {{{', '.join(JOINT_KEYS)}}}, got {jspec!r}")
    unknown = set(jspec) - set(JOINT_KEYS)
    if unknown:
        raise ValueError(f"{where}: joint: unknown keys {sorted(unknown)}; known: {sorted(JOINT_KEYS)}")
    kind = str(jspec.get("type", "hinge"))
    if kind not in JOINT_TYPES:
        raise ValueError(f"{where}: joint type is {', '.join(JOINT_TYPES)}, got {kind!r}")
    out: dict[str, Any] = {"type": kind}
    if jspec.get("to") is not None:
        out["to"] = str(jspec["to"])
    pivot_spec = jspec.get("pivot", [0.0, 0.0, 0.0])
    if isinstance(pivot_spec, str):
        pname, _, anchor = pivot_spec.partition(".")
        if suffix and pname not in parts and f"{pname}{suffix}" in parts:
            pname = f"{pname}{suffix}"
        if pname not in parts:
            raise ValueError(f"{where}: joint pivot names unknown part '{pname}'. Parts: {sorted(parts)}")
        node = parts[pname]
        point = node.attachments.get(anchor) if anchor else None
        if point is not None and node.size is not None:   # the primitive's own point (top, front, ...)
            # Attachment points are bottom-centre based; part meshes are centred on the part origin.
            local = point.resolve(node.size).translation - np.array([0.0, node.size[1] / 2, 0.0])
            pivot = (root_inv @ node.world_transform() @ np.r_[local, 1.0])[:3]
        else:
            size = node.size if node.size is not None else np.zeros(3)
            try:
                # Anchors are bottom-centre based; part meshes are centred on the part origin.
                local = resolve_anchor(anchor or "center", size) - np.array([0.0, size[1] / 2, 0.0])
            except ValueError:
                raise ValueError(f"{where}: joint pivot '{pivot_spec}': '{pname}' has no attachment point or "
                                 f"anchor '{anchor}' (points: {sorted(node.attachments)})") from None
            pivot = (root_inv @ node.world_transform() @ np.r_[local, 1.0])[:3]
    else:
        pivot = np.asarray(pivot_spec, dtype=np.float64)
        if pivot.shape != (3,):
            raise ValueError(f"{where}: joint pivot is [x, y, z] or <part>.<anchor>, got {pivot_spec!r}")
    out["pivot"] = [round(float(v), 6) for v in pivot]
    if kind != "pin":
        try:
            out["axis"] = [round(float(v), 6) for v in _axis(jspec.get("axis", "y"))]
        except ValueError as e:
            raise ValueError(f"{where}: joint {e}") from None
        limits = jspec.get("limits")
        if limits is not None:
            if not isinstance(limits, (list, tuple)) or len(limits) != 2 or float(limits[0]) > float(limits[1]):
                raise ValueError(f"{where}: joint limits are [lo, hi], got {limits!r}")
            out["limits"] = [float(limits[0]), float(limits[1])]
        motor = jspec.get("motor")
        if motor is not None:
            if not isinstance(motor, dict) or set(motor) - {"speed", "force"}:
                raise ValueError(f"{where}: joint motor is {{speed, force}}, got {motor!r}")
            out["motor"] = {"speed": float(motor.get("speed", 0.0)), "force": float(motor.get("force", DEFAULT_FORCE))}
            if out["motor"]["force"] <= 0:
                raise ValueError(f"{where}: joint motor force must be positive")
    elif jspec.get("axis") is not None or jspec.get("limits") is not None or jspec.get("motor") is not None:
        raise ValueError(f"{where}: a pin joint has no axis, limits or motor")
    return out


def exported_mechanism(mechanism: dict[str, Any], node: SceneNode, name_of) -> dict[str, Any]:
    """The extras form: bodies' parts by exported node name."""
    by_name: dict[str, SceneNode] = {}
    for n in node.iter_nodes(include_self=False):
        by_name.setdefault(n.name, n)
    bodies = {}
    for bname, body in mechanism["bodies"].items():
        nodes = [name_of(by_name[p]) for p in body["nodes"] if p in by_name]
        bodies[bname] = {**body, "nodes": nodes}
    return {"bodies": bodies}
