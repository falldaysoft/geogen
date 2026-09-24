"""Chunked export for large scenes: per-block/building files, exterior LODs, interiors on demand.

``export_chunks(root, out_dir)`` splits a scene into streamable pieces:

- **units**: the outermost nodes tagged ``city.block`` or ``building`` (a
  city exports one unit per block; a lone building is one unit);
- ``<unit>.glb``: the unit's exterior, placed in world space;
- ``<unit>_lod.glb``: a cheap distant version of that exterior: parts
  smaller than ``lod_min_size`` dropped, the rest decimated to
  ``lod_ratio`` and merged per material;
- ``<unit>_<building>_interior.glb``: each building's rooms (finishes,
  furniture, fixtures, room volumes) and interior doors, for engines to
  load when the player enters;
- ``base.glb``: everything outside the units (streets, street furniture);
- ``textures/``: every image once, named by content hash, referenced by URI
  from the chunk GLBs;
- ``<name>.chunks.json``: the index (format ``geogen-chunks`` v1): files,
  world bounds and interiors per chunk, plus the manifest fields (player,
  rooms, spawns) for the whole scene.

Meshes stay shared between instances inside each file (see
``export.to_trimesh_scene``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .core.mesh import Mesh
from .core.node import SceneNode
from .core.transform import Transform

CHUNKS_FORMAT = "geogen-chunks"
CHUNKS_VERSION = 1
UNIT_TAGS = ("city.block", "building")


def _is_unit(node: SceneNode) -> bool:
    return any(t in node.tags for t in UNIT_TAGS)


def find_units(root: SceneNode) -> list[SceneNode]:
    """Outermost city blocks / buildings under (or at) ``root``."""
    units: list[SceneNode] = []

    def visit(node: SceneNode) -> None:
        if _is_unit(node):
            units.append(node)
            return
        for child in node.children:
            visit(child)

    visit(root)
    return units


def _detached_copy(node: SceneNode) -> SceneNode:
    """Copy ``node``'s subtree (sharing meshes) as a root placed at its world transform."""
    out = node.instance()
    out.transform = Transform.from_matrix(node.world_transform())
    return out


def _strip_units(node: SceneNode, units: set[int]) -> SceneNode | None:
    """A copy of ``node`` without the unit subtrees (None if ``node`` is a unit)."""
    if id(node) in units:
        return None
    if not any(id(n) in units for n in node.iter_nodes()):
        return node.instance()
    shell = node.copy(deep=False)
    shell.children = []
    for child in node.children:
        stripped = _strip_units(child, units)
        if stripped is not None:
            stripped.transform = child.transform.copy()
            shell.add_child(stripped)
    return shell


def _is_interior(node: SceneNode) -> bool:
    room = node.meta.get("room")
    return (isinstance(room, dict) and node.meta.get("type") != "room_volume") or "door.interior" in node.tags


def split_interiors(exterior: SceneNode) -> list[tuple[str, SceneNode]]:
    """Detach every building's rooms and interior doors from ``exterior`` (in place).

    Returns ``(building name, interior root)`` pairs; interior roots are in
    ``exterior``'s parent frame (world, for a detached unit copy).
    """
    buildings = [n for n in exterior.iter_nodes() if "building" in n.tags]
    out = []
    for building in buildings:
        parts = []

        def collect(node: SceneNode) -> None:
            for child in list(node.children):
                if _is_interior(child):
                    parts.append(child)
                elif "building" not in child.tags:     # nested buildings get their own interior
                    collect(child)

        collect(building)
        if not parts:
            continue
        interior = SceneNode(name=f"{building.name}_interior", tags=["interior"])
        interior.meta["interior"] = {"building": building.name}
        for part in parts:
            matrix = part.world_transform()
            part.parent.children.remove(part)
            part.parent = None
            part.transform = Transform.from_matrix(matrix)
            interior.add_child(part)
        out.append((building.name, interior))
    return out


def exterior_lod(exterior: SceneNode, ratio: float = 0.25, min_size: float = 0.4) -> SceneNode:
    """Merged, decimated stand-in for a unit's exterior (world space, one mesh per material)."""
    from .core import meshops

    by_material: dict[int, list[Mesh]] = {}
    materials: dict[int, object] = {}
    decimated: dict[int, Mesh] = {}
    for node in exterior.iter_nodes():
        mesh = node.mesh
        if mesh is None or not len(mesh.faces) or node.meta.get("type") in ("collider", "room_volume"):
            continue
        extent = np.ptp(mesh.vertices, axis=0)
        if extent.max() < min_size or node.meta.get("collider") == "none" and extent.max() < 2 * min_size:
            continue
        if id(mesh) not in decimated:
            decimated[id(mesh)] = meshops.decimate(mesh, ratio) if len(mesh.faces) > 64 else mesh
        world = decimated[id(mesh)].transform(node.world_transform())
        key = id(mesh.material)
        materials[key] = mesh.material
        by_material.setdefault(key, []).append(world)
    root = SceneNode(name=f"{exterior.name}_lod", tags=["lod.exterior"])
    root.meta["lod"] = {"ratio": ratio, "min_size": min_size}
    for k, (key, meshes) in enumerate(by_material.items()):
        merged = Mesh.merge(meshes)
        merged.material = materials[key]
        part = SceneNode(name=f"{exterior.name}_lod_{k}", mesh=merged)
        part.meta["collider"] = "none"
        root.add_child(part)
    return root


def _bounds(node: SceneNode) -> list[list[float]] | None:
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for n in node.iter_nodes():
        if n.mesh is None or not len(n.mesh.vertices):
            continue
        v = n.mesh.vertices
        w = (n.world_transform() @ np.c_[v, np.ones(len(v))].T).T[:, :3]
        lo, hi = np.minimum(lo, w.min(axis=0)), np.maximum(hi, w.max(axis=0))
    if not np.isfinite(lo).all():
        return None
    return [[round(float(v), 3) for v in lo], [round(float(v), 3) for v in hi]]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def export_chunks(root: SceneNode, out_dir: str | Path, name: str | None = None, player=None,
                  lod_ratio: float = 0.25, lod_min_size: float = 0.4, interiors: bool = True) -> Path:
    """Export ``root`` as chunk files plus an index (see module docstring); returns the index path."""
    from .export import export_scene, gameplay_summary
    from .player import load_player_spec

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or root.name
    units = find_units(root)
    chunks = []
    used: set[str] = set()

    def filename(stem: str) -> str:
        stem, candidate, i = _safe(stem), _safe(stem), 1
        while candidate in used:
            i += 1
            candidate = f"{stem}_{i}"
        used.add(candidate)
        return candidate

    def write(node: SceneNode, stem: str) -> str:
        path = export_scene(node, out_dir / f"{stem}.glb", player=player, manifest=False,
                            textures_dir=out_dir / "textures")
        return path.name

    base = _strip_units(root, {id(u) for u in units}) if units and units[0] is not root else None
    if base is not None and any(n.mesh is not None for n in base.iter_nodes()):
        stem = filename("base")
        chunks.append({"name": "base", "file": write(base, stem), "bounds": _bounds(base), "interiors": []})

    for unit in units:
        stem = filename(unit.name)
        exterior = _detached_copy(unit)
        entry = {"name": unit.name, "tags": list(unit.tags), "interiors": []}
        if interiors:
            for building, interior in split_interiors(exterior):
                interior_stem = filename(f"{stem}_{building}_interior")
                entry["interiors"].append({"building": building, "file": write(interior, interior_stem),
                                           "bounds": _bounds(interior)})
        entry["file"] = write(exterior, stem)
        entry["bounds"] = _bounds(exterior)
        lod = exterior_lod(exterior, lod_ratio, lod_min_size)
        if lod.children:
            entry["lod"] = write(lod, filename(f"{stem}_lod"))
        chunks.append(entry)

    index = {
        "format": CHUNKS_FORMAT,
        "version": CHUNKS_VERSION,
        "name": name,
        "units": "m",
        "up": "+Y",
        "player": (player or load_player_spec()).to_dict(),
        **gameplay_summary(root),
        "chunks": chunks,
    }
    path = out_dir / f"{_safe(name)}.chunks.json"
    previous = _index_files(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(index, indent=2) + "\n")
    tmp.replace(path)      # written last and atomically: runtimes watch the index
    for stale in previous - _index_files(path):
        (out_dir / stale).unlink(missing_ok=True)
    return path


def _index_files(index_path: Path) -> set[str]:
    """Chunk files an index lists (empty if there is no readable index)."""
    try:
        index = json.loads(index_path.read_text())
    except (OSError, ValueError):
        return set()
    files = set()
    for chunk in index.get("chunks", []):
        files |= {chunk.get("file"), chunk.get("lod")} | {i.get("file") for i in chunk.get("interiors", [])}
    return {f for f in files if f}
