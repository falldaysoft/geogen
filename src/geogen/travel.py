"""Travel: declarative teleports between runtime scenes, or within one.

An asset declares a default ``travel:`` block at its top level, and a scene
placement's ``travel:`` overrides its keys::

    travel:
      scene: hotel            # a catalogue scene (assets/runtime_scenes.yaml); omitted = this scene
      spawn: arrival          # a spawn in the target (default: its first): its name, or its
                              # full authored path (street/hotel/entrance_spawn, see geogen.refs)
      prompt: Enter the hotel # UI text (default "Go to <scene>" / "Go")
      on: use                 # use: press E on it; enter: walk into its trigger volume
      target: frame           # on: use only: the part aimed at (default the whole asset)

The node gets ``meta.travel = {scene?, spawn?, prompt, on, volume?}``,
exported as ``extras.geogen.travel``:

- ``on: use`` adds a ``travel`` interaction (``ready`` -> ``going``, which
  emits ``travel`` and returns to ``ready``), so the runtime's usual aim and
  E prompt drive it;
- ``on: enter`` records ``volume: {center, size}``, a box in the asset frame:
  the asset's ``travel_volume`` part (always taken out of the geometry) or,
  without one, the asset's bounds.

The runtime unloads the current world, loads the target and puts the player
at the spawn (``runtime/godot/scripts/main.gd``). ``check_travel`` validates
exported targets against the catalogue, resolving each spawn in the target's
reference index (``<name>.refs.json``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.node import SceneNode

TRAVEL_KEYS = {"scene", "spawn", "prompt", "on", "target"}
TRIGGERS = ("use", "enter")
VOLUME_PART = "travel_volume"


def parse_travel(spec: dict[str, Any] | None, where: str = "") -> dict[str, Any]:
    """Validate a ``travel:`` block; returns its keys (strings) without defaults applied."""
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: travel must be a mapping, got {spec!r}")
    unknown = set(spec) - TRAVEL_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown travel keys {sorted(unknown)} (known: {sorted(TRAVEL_KEYS)})")
    out = {k: str(v) for k, v in spec.items() if v is not None}
    if out.get("on", "use") not in TRIGGERS:
        raise ValueError(f"{where}: travel on must be one of {TRIGGERS}, got {out['on']!r}")
    return out


def apply_travel(root: SceneNode, spec: dict[str, Any], where: str = "") -> None:
    """Make ``root`` a travel point: ``meta.travel`` plus its interaction or trigger volume.

    ``spec`` is merged over the travel the asset already declares (``root``'s
    own ``meta.travel``), so placements override only the keys they give.
    """
    merged = {**root.meta.get("travel_spec", {}), **parse_travel(spec, where or root.name)}
    root.meta["travel_spec"] = merged       # what was declared (not exported): later overrides merge over it
    if where:
        root.meta["travel_source"] = where  # for export diagnostics (not exported)
    on = merged.get("on", "use")
    scene = merged.get("scene")
    travel: dict[str, Any] = {"on": on,
                              "prompt": merged.get("prompt") or (f"Go to {scene.replace('_', ' ')}" if scene else "Go")}
    if scene:
        travel["scene"] = scene
    if merged.get("spawn"):
        travel["spawn"] = merged["spawn"]
    if merged.get("target"):
        travel["target"] = merged["target"]
    root.interactions = [i for i in root.interactions if i.name != "travel"]
    volume = (root.meta.get("travel") or {}).get("volume")    # already taken from a part / the bounds
    volume_part = root.find(VOLUME_PART)
    if volume_part is not None:
        volume = _volume(volume_part.mesh_bounds(root))
        volume_part.parent.remove_child(volume_part)           # a trigger, not geometry
    if volume is None and on == "enter":
        volume = _volume(root.mesh_bounds(root))
        if volume is None:
            raise ValueError(f"{where or root.name}: travel on: enter needs geometry or a '{VOLUME_PART}' part")
    if volume is not None:
        travel["volume"] = volume
    if on == "use":
        root.interactions.append(_travel_interaction(root, travel, where))
    root.meta["travel"] = travel


def _volume(box) -> dict[str, list[float]] | None:
    if box is None:
        return None
    lo, hi = box
    return {"center": [round(float(v), 6) for v in (lo + hi) / 2], "size": [round(float(v), 6) for v in hi - lo]}


def _travel_interaction(root: SceneNode, travel: dict[str, Any], where: str):
    from .layout.interactions import Interaction, State

    target = root
    if travel.get("target"):
        target = root.find(travel["target"])
        if target is None:
            raise ValueError(f"{where or root.name}: travel target part '{travel['target']}' not found")
    return Interaction("travel", {"ready": State(next="going", prompt=travel["prompt"]),
                                  "going": State(then="ready", emit="travel")},
                       [], [target], "ready", 0.1)


def travel_summary(root: SceneNode) -> list[dict[str, Any]]:
    """Every travel point under ``root`` for the manifest: {node, scene?, spawn?, on}."""
    out = []
    for node in root.iter_nodes():
        travel = node.meta.get("travel")
        if isinstance(travel, dict):
            out.append({"node": node.name, **{k: node.meta[k] for k in ("uid",) if k in node.meta},
                        **{k: travel[k] for k in ("scene", "spawn", "on") if k in travel},
                        **({"source": node.meta["travel_source"]} if node.meta.get("travel_source") else {})})
    return out


def check_travel(catalogue, out_dir: str | Path) -> list[str]:
    """Problems with the travel points of the catalogue's exported scenes in ``out_dir``.

    Every target must be a catalogue scene; when that scene is exported, the
    spawn must resolve in it (same-scene travel: in the scene itself): a
    plain name is any spawn of that name, a path is one spawn's full authored
    path. Targets that aren't exported yet can't be checked for their spawn.
    Each problem starts with where the travel was written, when known.
    """
    import json

    from . import refs
    from .catalogue import entry_manifest

    out_dir = Path(out_dir)
    manifests: dict[str, dict] = {}
    trees: dict[str, SceneNode | None] = {}
    for name, entry in catalogue.entries.items():
        path = out_dir / entry_manifest(entry)
        if path.exists():
            manifests[name] = json.loads(path.read_text())
            trees[name] = refs.load_index(path)
    problems = []
    for name, manifest in manifests.items():
        for point in manifest.get("travel", []):
            target = point.get("scene", name)
            where = point.get("source") or f"{name}: travel point '{point.get('uid') or point.get('node')}'"
            if target not in catalogue.entries:
                near = refs._near(target, catalogue.entries)
                problems.append(f"{where}: goes to '{target}', which isn't in the catalogue.{near}")
                continue
            spawn = point.get("spawn")
            if spawn is None or target not in manifests:
                continue
            problem = _check_spawn(target, str(spawn), manifests[target], trees.get(target))
            if problem:
                problems.append(f"{where}: {problem}")
    return problems


def _check_spawn(target: str, spawn: str, manifest: dict, tree: SceneNode | None) -> str | None:
    """Why ``spawn`` doesn't name a spawn of the exported ``target`` (None if it does)."""
    from . import refs

    names = [s.get("name") for s in manifest.get("spawns", [])]
    uids = [s.get("uid") for s in manifest.get("spawns", []) if s.get("uid")]
    if spawn in names or f"{target}:{spawn}" in uids:
        return None
    paths = [u.partition(":")[2] for u in uids]
    if tree is not None and any(c in spawn for c in "/[*#:"):
        try:
            found = [n for n in refs.select(tree, spawn, package=target, optional=False)
                     if n.meta.get("type") == "spawn"]
        except refs.RefError as exc:
            return f"spawn {exc.message}"
        if len(found) == 1:
            full = str(found[0].meta["uid"]).partition(":")[2]
            return f"spawn '{spawn}' in '{target}' should be written as its full path '{full}'"
        if found:
            return (f"spawn '{spawn}' matches {len(found)} spawns in '{target}': "
                    + ", ".join(sorted(str(n.meta["uid"]).partition(":")[2] for n in found)[:6]))
        return f"'{spawn}' in '{target}' isn't a spawn. Spawns: {refs._listed(paths or names)}"
    return (f"goes to spawn '{spawn}' in '{target}', which has {refs._listed(paths or names, 20) if names else 'no spawns'}."
            f"{refs._near(spawn, [*names, *paths])}")
