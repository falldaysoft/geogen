"""Motion QA (geogen.103.9): every motion scenario played by the extremes of every archetype.

A scenario (``motion_review``) is written for one person; the people who will do it in a game are
drawn from archetypes, and the smallest and largest of them are where a reach falls short, feet
slide, a seat doesn't fit or an arm goes through the body. ``sweep`` runs every scenario with each
archetype's extremes in place of its actor and reports, per run:

- the rig's own numbers (``motion_review.metrics``): ``contact_error_mm`` (hands and feet against
  their goals once a beat arrives: the reach), ``foot_slide_mm``, ``pops``, ``cloth_mm``;
- clipping sampled through the motion (``clipping.measure_actor`` every ``EVERY`` frames, and on
  the frame each beat arrives): ``shell_poke``, ``hands``, ``feet_skin``, as ``{max_mm, count, t}``
  at the worst frame;
- ``props``: how deep the body, clothes and hair go into the scene's furniture (a seat, the TV),
  leaving out what the person holds.

``failures(report)`` holds them against ``LIMITS``; ``tests/test_motion_qa.py`` runs the sweep
(the shortest and tallest people of two archetypes in the default tier, every archetype's
extremes with ``--all``).

Extremes (``extremes``): an archetype draws its body from ``preset`` choices; the people its seeds
give are grouped by the preset they drew, and the first seed of the shortest, the tallest and the
heaviest (``build``) are the extremes. Searching seeds keeps everything else the archetype draws
together (a variant's outfit with its gender), which overriding the preset would not.

    python -m geogen.motion_qa                                   # every scenario x every archetype
    python -m geogen.motion_qa press_tv --archetypes older_woman young_man --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import roots

ARCHETYPES_DIR = "characters/archetypes"
SEEDS = 200              # seeds searched for each preset an archetype draws
EVERY = 10               # frames between clipping samples (plus each beat's arrival)
CLIP_KEYS = ("shell_poke", "hands", "feet_skin")

# What a run may show, in mm (``failures``). Reach: a hand on a button or a grip ends within
# REACH of it. The clipping limits are what the reference body already shows standing and sitting
# (``clipping.measure``: a few mm of hand in the thigh it rests on). A hand on a button or a table
# may press a little way into it (``props_hands``); nothing else goes into the furniture.
LIMITS = {
    "contact_error_mm": 10.0,
    "foot_slide_mm": 5.0,
    "pops": 0,
    "cloth_mm": 6.0,
    "shell_poke": 6.0,
    "hands": 12.0,
    "props": 5.0,
    "props_hands": 8.0,
}


# --- the people ----------------------------------------------------------------------------------

def archetype_names(assets_dir: Path | None = None) -> list[str]:
    return sorted(p.stem for p in roots.glob(f"{ARCHETYPES_DIR}/*.yaml", assets_dir or roots.CORE))


def _presets(assets_dir: Path) -> tuple[dict, dict]:
    from .layout.yaml_utils import safe_load_path

    spec = safe_load_path(roots.find("characters/humanoid.yaml", assets_dir))
    defaults = {k: (v.get("default") if isinstance(v, dict) else v) for k, v in (spec.get("params") or {}).items()}
    return spec.get("presets") or {}, defaults


def _size(preset, presets: dict, defaults: dict, params: dict) -> tuple[float, float]:
    """(height, build) a preset list gives, explicit params winning as the loader has it."""
    merged = dict(defaults)
    for name in [preset] if isinstance(preset, str) else list(preset or []):
        merged.update(presets.get(name) or {})
    for k in ("height", "build"):
        if k in params and not isinstance(params[k], (dict, list)):
            merged[k] = params[k]
    return float(merged.get("height") or 1.72), float(merged.get("build") or 0.5)


def extremes(archetype: str, assets_dir: Path | None = None, seeds: int = SEEDS) -> dict[str, dict]:
    """``{short|tall|heavy: {archetype, seed, preset, height, build}}``: the first seed of the
    shortest, tallest and heaviest preset the archetype draws (one entry when they coincide,
    labels joined: ``short+heavy``)."""
    from .characters import resolve_character

    assets_dir = Path(assets_dir or roots.CORE)
    path = roots.find(f"{ARCHETYPES_DIR}/{archetype}.yaml", assets_dir)
    presets, defaults = _presets(assets_dir)
    found: dict[tuple, dict] = {}
    for seed in range(seeds):
        _, params = resolve_character(path, seed)
        preset = params.get("preset")
        key = tuple([preset] if isinstance(preset, str) else preset or ())
        if key not in found:
            height, build = _size(preset, presets, defaults, params)
            found[key] = {"archetype": f"{ARCHETYPES_DIR}/{archetype}.yaml", "seed": seed,
                          "preset": "+".join(key) or "default", "height": height, "build": build}
    people = list(found.values())
    picks = {"short": min(people, key=lambda p: (p["height"], p["build"])),
             "tall": max(people, key=lambda p: (p["height"], -p["build"])),
             "heavy": max(people, key=lambda p: (p["build"], p["height"]))}
    out: dict[str, dict] = {}
    for label, person in picks.items():
        same = next((k for k, v in out.items() if v is person), None)
        if same is not None:
            out[f"{same}+{label}"] = out.pop(same)
        else:
            out[label] = person
    return out


# --- one run -------------------------------------------------------------------------------------

class _Clipping:
    """Clipping through a motion: ``clipping.measure_actor`` every ``every`` frames and when a beat
    arrives, and the furniture the body goes into every frame it is sampled."""

    def __init__(self, every: int = EVERY) -> None:
        self.every = every
        self.rest: dict | None = None
        self.worst: dict[str, dict] = {}
        self.arrivals = 0
        self.frames = 0

    def __call__(self, rig) -> None:
        from .clipping import rest_meshes

        if self.rest is None:
            self.rest = rest_meshes(rig.actor)          # a fresh instance stands at rest
            self.props = _Props(rig)
        n = len(rig.samples)
        arrived = sum(1 for e in rig.trace if e["event"] in ("attach", "release", "emit", "posture", "end"))
        if n == 0 or (n % self.every and arrived == self.arrivals):
            return
        self.arrivals = arrived
        self.measure(rig)

    def measure(self, rig) -> None:
        from .clipping import measure_actor

        self.frames += 1
        t = rig.samples[-1]["t"] if rig.samples else 0.0
        found = measure_actor(rig.actor, self.rest)
        found.update(self.props.depth(rig))
        for key, value in found.items():
            if not isinstance(value, dict) or "max_mm" not in value:
                continue
            if key not in self.worst or value["max_mm"] > self.worst[key]["max_mm"]:
                self.worst[key] = {**value, "t": t}


HAND_PARTS = ("Hand", "Thumb", "Index", "Middle", "Ring", "Little")


class _Props:
    """How deep the person goes into the scene's other meshes: body, clothes and hair vertices
    inside a prop (winding number), leaving out what the rig holds, the person themself and other
    people.
    ``props_hands`` is the hands (a finger on a button or a table top), ``props`` the rest (a
    thigh in a table, a seat). ``where``: ``<bone or part> in <prop node>`` at the deepest."""

    def __init__(self, rig) -> None:
        from .generators.humanoid import dominant_bones

        actor = rig.actor
        mine = {id(n) for n in actor.iter_nodes()}
        root = rig.scene
        # Other people (a two-person motion's partner) aren't props: their meshes are at rest, not
        # posed by their rig, and a hug touches them on purpose.
        if root is not None:
            for body in root.iter_nodes():
                if (body.meta.get("humanoid") or {}).get("rig") is not None:
                    mine |= {id(n) for n in body.iter_nodes()}
        self.props = [n for n in root.iter_nodes() if n.mesh is not None and len(n.mesh.faces)
                      and id(n) not in mine and not n.meta.get("build_only")] if root is not None else []
        body = next((n for n in actor.iter_nodes() if n.name == "body" and n.skin is not None), None)
        self.bones = dominant_bones(body.mesh, [j.name for j in body.skin.joints]) if body is not None else None

    def depth(self, rig) -> dict:
        from .clipping import TOLERANCE, _meshes, distance, winding

        held = {id(n) for h in rig.attached.values() for n in h.node.iter_nodes()}
        parts = [(k, m.vertices) for k, m in _meshes(rig.actor).items() if k != "body_closed"]
        empty = {"count": 0, "max_mm": 0.0}
        if not parts:
            return {"props": dict(empty), "props_hands": dict(empty)}
        pts = np.concatenate([v for _, v in parts])
        label = np.concatenate([np.asarray(self.bones, dtype=object) if k == "body" and self.bones is not None
                                else np.full(len(v), k, dtype=object) for k, v in parts])
        hand = np.array([any(str(b).removeprefix("Left").removeprefix("Right").startswith(h) for h in HAND_PARTS)
                         for b in label])
        found = {"props": [0.0, 0, None], "props_hands": [0.0, 0, None]}
        lo, hi = pts.min(axis=0) - 0.01, pts.max(axis=0) + 0.01
        for node in self.props:
            if id(node) in held:
                continue
            m = node.world_mesh()
            vlo, vhi = m.vertices.min(axis=0), m.vertices.max(axis=0)
            if np.any(vlo > hi) or np.any(vhi < lo):
                continue
            near = np.flatnonzero(np.all((pts >= vlo) & (pts <= vhi), axis=1))
            if not len(near):
                continue
            tris = m.vertices[m.faces]
            inside = near[winding(pts[near], tris) > 0.5]
            if not len(inside):
                continue
            d = distance(pts[inside], tris)
            for key, rows in (("props_hands", hand[inside]), ("props", ~hand[inside])):
                bad = d[rows & (d > TOLERANCE)]
                found[key][1] += len(bad)
                if len(bad) and float(bad.max()) > found[key][0]:
                    k = int(np.flatnonzero(rows)[np.argmax(d[rows])])
                    found[key][0], found[key][2] = float(bad.max()), f"{label[inside[k]]} in {node.name}"
        out = {}
        for key, (worst, count, where) in found.items():
            out[key] = {"count": int(count), "max_mm": round(worst * 1000, 1)}
            if where:
                out[key]["where"] = where
        return out


def run(scenario: str, person: dict | None = None, assets_dir: Path | None = None,
        every: int = EVERY) -> dict[str, Any]:
    """One scenario played by ``person`` (an ``extremes`` entry; None: the scenario's own actor):
    the rig's metrics, the worst clipping frame by frame, and the furniture the body went into."""
    from .motion_review import run_scenario

    probe = _Clipping(every)
    actor = {"archetype": person["archetype"], "seed": person["seed"]} if person else None
    result = run_scenario(scenario, assets_dir, actor=actor, probe=probe)
    probe.measure(result.rig)                          # where it ends
    m = result.metrics
    out: dict[str, Any] = {
        "contact_error_mm": m["contact_error_mm"],
        "foot_slide_mm": m["foot_slide_mm"],
        "pops": m["pops"],
        "peak_joint_speed": m["peak_joint_speed"],
        "clipping_frames": probe.frames,
    }
    if "cloth_mm" in m:
        out["cloth_mm"] = m["cloth_mm"]["mm"]
    out.update(probe.worst)
    if person:
        out["person"] = {k: person[k] for k in ("archetype", "seed", "preset", "height", "build")}
    return out


def sweep(scenarios: list[str] | None = None, archetypes: list[str] | None = None,
          labels: tuple[str, ...] = ("short", "tall", "heavy"), assets_dir: Path | None = None,
          every: int = EVERY) -> dict[str, dict]:
    """``{"<scenario>/<archetype>/<label>": run(...)}`` for every scenario and the archetypes'
    extremes whose label (``short+heavy`` counts for both) is in ``labels``."""
    from .motion_review import scenario_names

    assets_dir = Path(assets_dir or roots.CORE)
    scenarios = scenarios or scenario_names(assets_dir)
    people = {}
    for a in archetypes or archetype_names(assets_dir):
        for label, person in extremes(a, assets_dir).items():
            if set(label.split("+")) & set(labels):
                people[f"{a}/{label}"] = person
    return {f"{s}/{who}": run(s, person, assets_dir, every) for s in scenarios for who, person in people.items()}


def failures(report: dict[str, dict], limits: dict | None = None) -> list[str]:
    """One line per number over its limit."""
    limits = {**LIMITS, **(limits or {})}
    out = []
    for key, r in report.items():
        for name, limit in limits.items():
            value = r.get(name)
            if isinstance(value, dict):
                value = value.get("max_mm")
            if value is not None and value > limit:
                where = f" at {r[name]['t']}s" if isinstance(r.get(name), dict) and "t" in r[name] else ""
                out.append(f"{key}: {name} {value} > {limit}{where}")
        for name in CLIP_KEYS[2:]:
            if (r.get(name) or {}).get("count"):
                out.append(f"{key}: {name} {r[name]['count']} vertices")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m geogen.motion_qa", description=__doc__.split("\n\n")[0])
    ap.add_argument("scenarios", nargs="*", help="scenario names (default: all)")
    ap.add_argument("--archetypes", nargs="*", help="archetype names (default: all)")
    ap.add_argument("--labels", default="short,tall,heavy", help="which extremes (short,tall,heavy)")
    ap.add_argument("--every", type=int, default=EVERY, help="frames between clipping samples")
    ap.add_argument("--json", action="store_true", help="print the whole report as JSON")
    args = ap.parse_args(argv)
    report = sweep(args.scenarios or None, args.archetypes or None, tuple(args.labels.split(",")),
                   every=args.every)
    if args.json:
        print(json.dumps(report, indent=1, default=float))
    else:
        for key, r in report.items():
            p = r.get("person") or {}
            clip = " ".join(f"{k} {r[k]['max_mm']}" for k in (*CLIP_KEYS, "props", "props_hands") if k in r)
            print(f"{key:58} {p.get('preset', ''):18} reach {r['contact_error_mm']:5} slide "
                  f"{r['foot_slide_mm']:4} pops {r['pops']} cloth {r.get('cloth_mm', '-')} | {clip}")
    bad = failures(report)
    for line in bad:
        print("FAIL", line)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
