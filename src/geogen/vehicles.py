"""Vehicles as data: the ``vehicle:`` block of a vehicle asset.

A vehicle asset (``assets/vehicles/*.yaml``) is an ordinary parametric asset
built along +Z (the front) with its origin on the ground at the centre of
the wheelbase. Its ``vehicle:`` block tells runtimes how to move it::

    vehicle:
      class: car                     # car | van | bus | loco | carriage
      wheels: [wheel_fl, wheel_fr, wheel_rl, wheel_rr]   # parts that spin (about local X)
      steer: [wheel_fl, wheel_fr]    # parts that turn with the steering (optional)
      paint: [body]                  # parts tinted per instance (the `paint` param)
      lamps: { head: [headlights], tail: [taillights], brake: [taillights] }
      bogies: [bogie_front, bogie_rear]   # rail vehicles
      couplers: [coupler_front, coupler_rear]
      max_speed: 16                  # m/s
      turn_radius: 5.5               # m, kerb to kerb

``parse_vehicle`` resolves it to ``meta.vehicle``: part names, each wheel's
measured radius and centre, the wheelbase and track, and the clearance box
(``[length, width, height]`` from the geometry unless given).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.node import SceneNode

CLASSES = {"car", "van", "bus", "truck", "loco", "carriage"}
KEYS = {"class", "wheels", "steer", "paint", "lamps", "bogies", "couplers", "max_speed", "turn_radius",
        "clearance", "accel", "decel", "sound"}
SOUND_DEFAULTS = {
    "engine": {"base": 45.0, "per_speed": 4.0, "volume_db": -10.0, "roughness": 0.35},
    "horn": {"tones": [370.0, 440.0], "seconds": 1.2, "volume_db": 0.0},
}
LAMP_KINDS = {"head", "tail", "brake", "indicator"}
DEFAULTS = {"car": (16.0, 5.5), "van": (14.0, 6.5), "bus": (12.0, 9.0), "truck": (12.0, 9.0),
            "loco": (30.0, 150.0), "carriage": (30.0, 150.0)}


def _points(node: SceneNode, root: SceneNode) -> np.ndarray | None:
    to_root = np.linalg.inv(root.world_transform())
    pts = []
    for n in node.iter_nodes():
        if n.mesh is None or not len(n.mesh.vertices):
            continue
        m = to_root @ n.world_transform()
        pts.append((m @ np.c_[n.mesh.vertices, np.ones(len(n.mesh.vertices))].T).T[:, :3])
    return np.concatenate(pts) if pts else None


def _part(name: str, parts: dict[str, SceneNode], what: str) -> SceneNode:
    node = parts.get(name)
    if node is None:
        raise ValueError(f"vehicle: {what} names unknown part {name!r} (parts: {sorted(parts)})")
    return node


def parse_sound(spec: dict[str, Any]) -> dict[str, Any]:
    """Sound hooks a runtime synthesises (no audio files)::

        sound:
          engine: { base: 45, per_speed: 4, volume_db: -10, roughness: 0.35 }  # Hz at rest, Hz per m/s
          horn: { tones: [370, 440], seconds: 1.2, volume_db: 0 }              # trains sound it before crossings

    ``engine: true`` / ``horn: true`` take the defaults.
    """
    unknown = set(spec) - set(SOUND_DEFAULTS)
    if unknown:
        raise ValueError(f"vehicle sound: kinds are {sorted(SOUND_DEFAULTS)}, got {sorted(unknown)}")
    out = {}
    for kind, value in spec.items():
        value = {} if value is True else dict(value or {})
        extra = set(value) - set(SOUND_DEFAULTS[kind])
        if extra:
            raise ValueError(f"vehicle sound.{kind}: keys are {sorted(SOUND_DEFAULTS[kind])}, got {sorted(extra)}")
        merged = {**SOUND_DEFAULTS[kind], **value}
        out[kind] = {k: ([float(x) for x in v] if isinstance(v, list) else float(v)) for k, v in merged.items()}
    return out


def parse_vehicle(spec: dict[str, Any], root: SceneNode, parts: dict[str, SceneNode]) -> dict[str, Any]:
    unknown = set(spec) - KEYS
    if unknown:
        raise ValueError(f"vehicle: unknown keys {sorted(unknown)}; known: {sorted(KEYS)}")
    cls = spec.get("class", "car")
    if cls not in CLASSES:
        raise ValueError(f"vehicle: class must be one of {sorted(CLASSES)}, got {cls!r}")
    out: dict[str, Any] = {"class": cls}

    wheels = []
    for name in spec.get("wheels") or []:
        pts = _points(_part(name, parts, "wheels"), root)
        if pts is None:
            continue
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        wheels.append({"part": name, "center": [round(float(v), 4) for v in (lo + hi) / 2],
                       "radius": round(float(hi[1] - lo[1]) / 2, 4)})
    out["wheels"] = wheels
    if wheels:
        z = [w["center"][2] for w in wheels]
        x = [w["center"][0] for w in wheels]
        out["wheelbase"] = round(max(z) - min(z), 4)
        out["track"] = round(max(x) - min(x), 4)
    for key in ("steer", "paint", "bogies", "couplers"):
        names = [str(n) for n in spec.get(key) or []]
        for n in names:
            _part(n, parts, key)
        if names:
            out[key] = names
    # Rail: where each bogie pivots (it turns on its own to follow the track).
    centers = []
    for name in out.get("bogies", []):
        pts = _points(parts[name], root)
        if pts is not None:
            centers.append(round(float((pts[:, 2].min() + pts[:, 2].max()) / 2), 4))
    if centers:
        out["bogie_centers"] = centers
    lamps = {}
    for kind, names in (spec.get("lamps") or {}).items():
        if kind not in LAMP_KINDS:
            raise ValueError(f"vehicle: lamp kinds are {sorted(LAMP_KINDS)}, got {kind!r}")
        names = [names] if isinstance(names, str) else [str(n) for n in names]
        for n in names:
            _part(n, parts, f"lamps.{kind}")
        lamps[kind] = names
    if lamps:
        out["lamps"] = lamps

    pts = _points(root, root)
    if "clearance" in spec:
        out["clearance"] = [float(v) for v in spec["clearance"]]
    elif pts is not None:
        ext = pts.max(axis=0) - pts.min(axis=0)
        out["clearance"] = [round(float(ext[2]), 3), round(float(ext[0]), 3), round(float(ext[1]), 3)]
    if pts is not None:
        out["front"] = round(float(pts[:, 2].max()), 4)     # bumper positions along +Z
        out["rear"] = round(float(pts[:, 2].min()), 4)
    if "sound" in spec:
        out["sound"] = parse_sound(spec["sound"])
    max_speed, turn_radius = DEFAULTS[cls]
    out["max_speed"] = float(spec.get("max_speed", max_speed))
    out["turn_radius"] = float(spec.get("turn_radius", turn_radius))
    for key, default in (("accel", 2.0), ("decel", 4.0)):
        out[key] = float(spec.get(key, default))
    return out
