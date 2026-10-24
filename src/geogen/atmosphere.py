"""A scene's ``atmosphere:``: the lighting mood the runtime plays it in (geogen.84.13).

The runtime's one environment (sky, sun, fog, grading) is tuned for a clear afternoon; every
scene looked like the same postcard. A scene names a preset and overrides what it likes::

    atmosphere: {preset: grim, time: "16:00", fog: 0.025}

The presets are resolved here, at build time, into plain numbers (``PRESETS[name]`` overlaid by
the overrides) and written to the manifest's ``atmosphere`` (``export.gameplay_summary``): a
runtime applies values and knows no preset names. The Godot runtime reads them in
``clock.gd`` (``GeogenClock.set_atmosphere``): the static ones once per load (fog, grading,
glow), the rest into its hourly sun, sky and moon. ``time`` sets the world clock when the scene
first loads, unless ``--time`` was given.
"""

from __future__ import annotations

import re
from typing import Any

from .core.node import SceneNode

META = "atmosphere"

KEYS: dict[str, str] = {
    "preset": "clear (the runtime's default look) | overcast | grim | storm: the base the rest overrides",
    "time": "HH:MM: the world clock when the scene first loads (the runtime's --time wins)",
    "fog": "fog density per metre (clear 0.004; the far shore dissolves at 0.02)",
    "haze": "0..1: how far the fog climbs into the sky and greys the distance (0: the horizon stays crisp)",
    "saturation": "colour grading, 1 as rendered (0.7 drains the cheer out of grass and sand)",
    "contrast": "colour grading, 1 as rendered",
    "sun": "the sun's energy at noon, 1 full",
    "sun_color": "[r, g, b] 0..1: the sun's colour high in the sky (it is always warm near the horizon)",
    "sun_height": "degrees: the sun's elevation at noon (58 clear; lower stretches shadows all day)",
    "sky": "{top, horizon, ground: [r, g, b]}: the day sky, overhead, at the horizon and below it",
    "moon": "moonlight energy at night (0.14 clear; 0.45 keeps shapes readable)",
    "night": "0..1: the ambient light left at night (0.35 clear: near black)",
    "glow": "0..1: bloom around bright things (0 off): a lit fire blooms in the dark",
}
COLOR_KEYS = ("sun_color",)
SKY_KEYS = ("top", "horizon", "ground")
RANGES: dict[str, tuple[float, float]] = {
    "fog": (0.0, 0.2), "haze": (0.0, 1.0), "saturation": (0.0, 2.0), "contrast": (0.5, 2.0), "sun": (0.0, 3.0),
    "sun_height": (5.0, 90.0), "moon": (0.0, 2.0), "night": (0.0, 1.0), "glow": (0.0, 1.0),
}

# Every preset is complete: the runtime's own look is ``clear`` (runtime/godot/scenes/main.tscn
# and clock.gd's constants), so a scene without an atmosphere block looks as it always did.
PRESETS: dict[str, dict[str, Any]] = {
    "clear": {
        "fog": 0.004, "haze": 0.0, "saturation": 1.0, "contrast": 1.0,
        "sun": 1.0, "sun_color": [1.0, 0.98, 0.94], "sun_height": 58.0,
        "sky": {"top": [0.385, 0.454, 0.55], "horizon": [0.66, 0.72, 0.8], "ground": [0.36, 0.39, 0.35]},
        "moon": 0.14, "night": 0.35, "glow": 0.0,
    },
    "overcast": {
        "fog": 0.012, "haze": 0.4, "saturation": 0.85, "contrast": 1.0,
        "sun": 0.7, "sun_color": [0.95, 0.95, 0.95], "sun_height": 50.0,
        "sky": {"top": [0.45, 0.48, 0.52], "horizon": [0.68, 0.69, 0.7], "ground": [0.33, 0.34, 0.33]},
        "moon": 0.2, "night": 0.4, "glow": 0.0,
    },
    "grim": {
        "fog": 0.02, "haze": 0.6, "saturation": 0.68, "contrast": 1.06,
        "sun": 0.55, "sun_color": [0.9, 0.9, 0.92], "sun_height": 42.0,
        "sky": {"top": [0.34, 0.37, 0.41], "horizon": [0.58, 0.6, 0.62], "ground": [0.3, 0.31, 0.3]},
        "moon": 0.45, "night": 0.55, "glow": 0.5,
    },
    "storm": {
        "fog": 0.035, "haze": 0.8, "saturation": 0.6, "contrast": 1.08,
        "sun": 0.35, "sun_color": [0.8, 0.82, 0.86], "sun_height": 38.0,
        "sky": {"top": [0.22, 0.24, 0.27], "horizon": [0.42, 0.44, 0.46], "ground": [0.22, 0.23, 0.22]},
        "moon": 0.3, "night": 0.5, "glow": 0.4,
    },
}
_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class AtmosphereError(ValueError):
    """An ``atmosphere:`` block that isn't the shape above."""


def _color(value: Any, where: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3 \
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0 for v in value):
        raise AtmosphereError(f"{where}: a colour is [r, g, b] in 0..1, got {value!r}")
    return [float(v) for v in value]


def resolve(spec: dict[str, Any], where: str = "atmosphere") -> dict[str, Any]:
    """The block's preset overlaid by its overrides, checked, as the manifest carries it."""
    if not isinstance(spec, dict) or set(spec) - set(KEYS):
        raise AtmosphereError(f"{where}: atmosphere is {{{', '.join(KEYS)}}}, got {spec!r}")
    preset = spec.get("preset", "clear")
    if preset not in PRESETS:
        raise AtmosphereError(f"{where}: preset is one of {', '.join(PRESETS)}, got {preset!r}")
    out: dict[str, Any] = {"preset": preset, **{k: (dict(v) if isinstance(v, dict) else v)
                                                for k, v in PRESETS[preset].items()}}
    for key, value in spec.items():
        if key == "preset":
            continue
        if key == "time":
            if not isinstance(value, str) or not _TIME.match(value):
                raise AtmosphereError(f"{where}.time: HH:MM, got {value!r}")
            out["time"] = value
        elif key in COLOR_KEYS:
            out[key] = _color(value, f"{where}.{key}")
        elif key == "sky":
            if not isinstance(value, dict) or set(value) - set(SKY_KEYS):
                raise AtmosphereError(f"{where}.sky: {{{', '.join(SKY_KEYS)}}}, got {value!r}")
            out["sky"] = {**out["sky"], **{k: _color(v, f"{where}.sky.{k}") for k, v in value.items()}}
        else:
            lo, hi = RANGES[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not lo <= value <= hi:
                raise AtmosphereError(f"{where}.{key}: a number in {lo:g}..{hi:g}, got {value!r}")
            out[key] = float(value)
    return out


def declare(root: SceneNode, data: dict[str, Any], source: Any) -> None:
    """Record a scene's ``atmosphere:`` on its root, resolved (build-only meta; the manifest's
    ``atmosphere``)."""
    spec = data.get(META)
    if spec is None:
        return
    root.meta[META] = resolve(spec, f"{source}: atmosphere" if source is not None else "atmosphere")
