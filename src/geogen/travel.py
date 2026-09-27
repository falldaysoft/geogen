"""Travel: declarative teleports between runtime scenes, or within one.

An asset declares a default ``travel:`` block at its top level, and a scene
placement's ``travel:`` overrides its keys::

    travel:
      scene: hotel            # a catalogue scene (assets/runtime_scenes.yaml); omitted = this scene
      spawn: arrival          # a named spawn in the target (default: its first)
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
exported targets against the catalogue.
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
        volume = _volume(_local_box(root, volume_part))
        volume_part.parent.remove_child(volume_part)           # a trigger, not geometry
    if volume is None and on == "enter":
        volume = _volume(_local_box(root))
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


def _local_box(root: SceneNode, node: SceneNode | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Bounds of ``node``'s subtree (default: all of ``root``) in ``root``'s frame, or None if empty."""
    to_root = np.linalg.inv(root.world_transform())
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for n in (node or root).iter_nodes():
        if n.mesh is None or len(n.mesh.vertices) == 0:
            continue
        m = to_root @ n.world_transform()
        pts = n.mesh.vertices @ m[:3, :3].T + m[:3, 3]
        lo, hi = np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0))
    return (lo, hi) if np.all(np.isfinite(lo)) else None


def travel_summary(root: SceneNode) -> list[dict[str, Any]]:
    """Every travel point under ``root`` for the manifest: {node, scene?, spawn?, on}."""
    out = []
    for node in root.iter_nodes():
        travel = node.meta.get("travel")
        if isinstance(travel, dict):
            out.append({"node": node.name, **{k: travel[k] for k in ("scene", "spawn", "on") if k in travel}})
    return out


def check_travel(catalogue, out_dir: str | Path) -> list[str]:
    """Problems with the travel points of the catalogue's exported scenes in ``out_dir``.

    Every target must be a catalogue scene; when that scene is exported, the
    named spawn must exist in it (same-scene travel: in the scene itself).
    Targets that aren't exported yet can't be checked for their spawn.
    """
    import json

    from .catalogue import entry_manifest

    out_dir = Path(out_dir)
    manifests: dict[str, dict] = {}
    for name, entry in catalogue.entries.items():
        path = out_dir / entry_manifest(entry)
        if path.exists():
            manifests[name] = json.loads(path.read_text())
    problems = []
    for name, manifest in manifests.items():
        for point in manifest.get("travel", []):
            target = point.get("scene", name)
            where = f"{name}: travel point '{point.get('node')}'"
            if target not in catalogue.entries:
                problems.append(f"{where} goes to '{target}', which isn't in the catalogue")
                continue
            spawn = point.get("spawn")
            if spawn is None or target not in manifests:
                continue
            spawns = [s.get("name") for s in manifests[target].get("spawns", [])]
            if spawn not in spawns:
                problems.append(f"{where} goes to spawn '{spawn}' in '{target}', which has {spawns or 'no spawns'}")
    return problems
