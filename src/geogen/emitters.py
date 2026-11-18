"""Emitters: what a part gives off, described physically (``emitter:``) and what it sounds like (``sound:``).

Both are keys of a part and neither names an engine feature: a runtime maps them to its own
particle system and audio (Godot: ``runtime/godot/scripts/world_emitters.gd``)::

    parts:
      spout:
        primitive: cylinder
        size: [0.1, 0.2, 0.1]
        emitter:
          at: top             # an attachment point of the part, or [x, y, z] m from its centre
          rate: 150           # particles a second
          lifetime: 1.2       # seconds each one lives
          speed: [4, 4.6]     # m/s as it leaves (a range: each picks one)
          direction: [0, 1, 0]  # in the part's frame
          spread: 6           # degrees either side of `direction`
          gravity: 9.81       # m/s² downward; negative rises (smoke)
          size: [0.03, 0.05]  # metres across, at birth and at death
          color: [0.8, 0.9, 1.0]
          material: droplet   # droplet | soft | streak | glow
        sound: {band: [400, 5000], volume_db: -14, range: 18}

A part with either has a bool ``running`` (``state.emitter_state``), started from the block's
``running:``, that switches both: bind it (``bind: {spout.running: "self.valve == 'open'"}``) or
set it from any rule, exactly like a light fixture's ``lit``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.node import SceneNode
from .layout.anchors import resolve_anchor

MATERIALS = ("droplet", "soft", "streak", "glow")
# More live particles than this from one emitter is a mistake (or wants a texture instead).
MAX_PARTICLES = 6000

EMITTER_DEFAULTS: dict[str, Any] = {
    "at": "top", "rate": 20.0, "lifetime": 1.0, "speed": 1.0, "direction": [0.0, 1.0, 0.0], "spread": 10.0,
    "gravity": 0.0, "wind": [0.0, 0.0, 0.0], "drag": 0.0, "size": 0.1, "color": [1.0, 1.0, 1.0], "opacity": 1.0,
    "material": "soft", "area": [0.0, 0.0, 0.0], "running": True,
}
SOUND_DEFAULTS: dict[str, Any] = {
    "at": "top", "band": [300.0, 4000.0], "hum": 0.0, "flutter": 0.2, "crackle": 0.0, "volume_db": -12.0,
    "range": 15.0, "running": True,
}


def _known(spec: Any, defaults: dict[str, Any], key: str, part: str) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"part '{part}': {key} is a mapping of {sorted(defaults)}, got {spec!r}")
    unknown = set(spec) - set(defaults)
    if unknown:
        raise ValueError(f"part '{part}': {key}: unknown keys {sorted(unknown)}; known: {sorted(defaults)}")
    return {**defaults, **spec}


def _vector(value: Any, what: str, length: int = 3) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{what} is {length} numbers, got {value!r}")
    return [float(v) for v in value]


def _span(value: Any, what: str) -> list[float]:
    """A number, or a [first, second] pair, as a pair."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value), float(value)]
    return _vector(value, what, 2)


def _offset(at: Any, node: SceneNode, what: str) -> list[float]:
    """Where ``at`` is in the part's frame: an attachment point's name, or metres from its centre."""
    if not isinstance(at, str):
        return _vector(at, f"{what}: at")
    point = node.attachments.get(at)
    if point is None:
        raise ValueError(f"{what}: at: no attachment point '{at}'; the part has {sorted(node.attachments)}")
    size = np.asarray(node.size, dtype=np.float64)
    # Attachment points resolve from the box's bottom centre; a part's geometry is centred on its origin.
    position = point.resolve(size).translation - resolve_anchor("center", size)
    return [round(float(v), 6) for v in position]


def parse_emitter(spec: Any, node: SceneNode, part: str) -> dict[str, Any]:
    """Normalise a part's ``emitter:`` block into what runtimes read (``extras.geogen.emitter``)."""
    where = f"part '{part}': emitter"
    e = _known(spec, EMITTER_DEFAULTS, "emitter", part)
    if e["material"] not in MATERIALS:
        raise ValueError(f"{where}: material must be one of {', '.join(MATERIALS)}, got {e['material']!r}")
    rate, lifetime = float(e["rate"]), float(e["lifetime"])
    if rate <= 0 or lifetime <= 0:
        raise ValueError(f"{where}: rate and lifetime must be above 0")
    if rate * lifetime > MAX_PARTICLES:
        raise ValueError(f"{where}: rate x lifetime is {rate * lifetime:.0f} live particles; the limit is {MAX_PARTICLES}")
    direction = np.array(_vector(e["direction"], f"{where}: direction"))
    if np.linalg.norm(direction) < 1e-9:
        raise ValueError(f"{where}: direction has no length")
    color = _vector(e["color"], f"{where}: color")
    if not 0.0 <= float(e["opacity"]) <= 1.0:
        raise ValueError(f"{where}: opacity must be 0 to 1")
    if not isinstance(e["running"], bool):
        raise ValueError(f"{where}: running is true or false")
    return {
        "offset": _offset(e["at"], node, where), "rate": rate, "lifetime": lifetime,
        "speed": _span(e["speed"], f"{where}: speed"),
        "direction": [float(v) for v in direction / np.linalg.norm(direction)],
        "spread": float(e["spread"]), "gravity": float(e["gravity"]), "wind": _vector(e["wind"], f"{where}: wind"),
        "drag": float(e["drag"]), "size": _span(e["size"], f"{where}: size"), "color": color,
        "opacity": float(e["opacity"]), "material": e["material"], "area": _vector(e["area"], f"{where}: area"),
        "running": e["running"],
    }


def parse_sound(spec: Any, node: SceneNode, part: str) -> dict[str, Any]:
    """Normalise a part's ``sound:`` block (``extras.geogen.sound``): a synthesised loop, no audio files."""
    where = f"part '{part}': sound"
    s = _known(spec, SOUND_DEFAULTS, "sound", part)
    band = _vector(s["band"], f"{where}: band", 2)
    if not 0 < band[0] < band[1]:
        raise ValueError(f"{where}: band is [low, high] Hz, got {band}")
    for key in ("flutter", "crackle"):
        if not 0.0 <= float(s[key]) <= 1.0:
            raise ValueError(f"{where}: {key} must be 0 to 1")
    if float(s["range"]) <= 0:
        raise ValueError(f"{where}: range must be above 0")
    if not isinstance(s["running"], bool):
        raise ValueError(f"{where}: running is true or false")
    return {"offset": _offset(s["at"], node, where), "band": band, "hum": float(s["hum"]),
            "flutter": float(s["flutter"]), "crackle": float(s["crackle"]), "volume_db": float(s["volume_db"]),
            "range": float(s["range"]), "running": s["running"]}


def apply_part(node: SceneNode, part_def: dict[str, Any], part: str) -> None:
    """Record a part's ``emitter:`` / ``sound:`` on its node; they must agree on how it starts."""
    if "emitter" in part_def:
        node.meta["emitter"] = parse_emitter(part_def["emitter"], node, part)
    if "sound" in part_def:
        node.meta["sound"] = parse_sound(part_def["sound"], node, part)
    if "emitter" in node.meta and "sound" in node.meta and node.meta["emitter"]["running"] != node.meta["sound"]["running"]:
        raise ValueError(f"part '{part}': emitter and sound share one `running` state, so they must start alike")


def initially_running(node: SceneNode) -> bool | None:
    """How ``node``'s emitter and sound start; None if it has neither."""
    block = node.meta.get("emitter") or node.meta.get("sound")
    return bool(block["running"]) if isinstance(block, dict) else None
