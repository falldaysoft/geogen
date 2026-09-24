"""Export scene graphs to game-engine formats (glTF/GLB, OBJ).

glTF keeps the node hierarchy with local transforms, so an exported house
still has separately addressable doors and windows. Meshes carry explicit
crease-angle normals, and PBR metallic-roughness materials with base colour,
metallic-roughness, normal and occlusion textures. Metric UVs are converted
to texture space using each material's ``tile_size`` so textures repeat at
the right real-world scale (samplers use REPEAT wrapping).

glTF exports also write ``<name>.manifest.json`` alongside the model: the
contract the Godot runtime reads (units, up axis, model file, player spec,
rooms, spawns).

Gameplay metadata (schema: docs/schema/geogen-extras.v1.schema.json) rides
along in glTF node extras as ``extras.geogen`` -- tags, room ids, trigger
volumes, spawn points, walkable flags and the collider chosen for each mesh.
Colliders are extra child nodes named with Godot's import suffixes, so the
editor importer builds physics with no plugin: ``<name>-convcolonly`` (box or
convex hull) and ``<name>-colonly`` (trimesh). Runtimes that don't know the
suffixes should skip nodes whose extras say ``type: collider``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from .core import meshops
from .core.mesh import Mesh
from .core.node import SceneNode
from .materials.material import Material
from .player import PlayerSpec, load_player_spec

FORMATS = {".glb", ".gltf", ".obj"}
MANIFEST_FORMAT = "geogen-manifest"
MANIFEST_VERSION = 1
EXTRAS_VERSION = 1
EXTRAS_SCHEMA = "docs/schema/geogen-extras.v1.schema.json"

# Colliders smaller than this in every dimension are skipped by ``auto``.
MIN_COLLIDER_SIZE = 0.03
COLLIDER_SUFFIX = {"box": "-convcolonly", "hull": "-convcolonly", "mesh": "-colonly"}


def _pbr_material(material: Material, cache: dict[int, trimesh.visual.material.PBRMaterial]):
    key = id(material)
    if key not in cache:
        images = material.gltf_images()
        cache[key] = trimesh.visual.material.PBRMaterial(
            name=material.name,
            baseColorTexture=images["base_color"],
            metallicRoughnessTexture=images["metallic_roughness"],
            normalTexture=images.get("normal"),
            occlusionTexture=images.get("occlusion"),
            metallicFactor=1.0,
            roughnessFactor=1.0,
        )
    return cache[key]


def to_trimesh(mesh: Mesh, cache: dict | None = None) -> trimesh.Trimesh:
    """Convert a geogen Mesh to a trimesh with normals, texture-space UVs and PBR material."""
    cache = {} if cache is None else cache
    mesh = meshops.ensure_normals(mesh)
    visual = None
    if mesh.material is not None and mesh.uvs is not None:
        uv = mesh.uvs * np.asarray(mesh.material.texture_uv_scale)
        visual = trimesh.visual.TextureVisuals(uv=uv, material=_pbr_material(mesh.material, cache))
    tm = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_normals=mesh.normals,
        visual=visual,
        process=False,
    )
    return tm


def resolve_collider(node: SceneNode) -> str:
    """Collider for a mesh node: its ``collider`` meta, or an ``auto`` choice.

    ``auto`` skips tiny parts, uses a box when the mesh fills its bounds, a
    convex hull when it is nearly convex, and the triangle mesh otherwise
    (e.g. walls with openings, hollow shells).
    """
    kind = str(node.meta.get("collider", "auto"))
    if kind != "auto" or node.mesh is None:
        return kind
    mesh = node.mesh
    extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    if extent.max() < MIN_COLLIDER_SIZE:
        return "none"
    if extent.min() < 1e-4:
        return "mesh"  # flat (plane) geometry has no volume to compare
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    volume = abs(tm.volume)
    if volume >= 0.97 * float(np.prod(extent)):
        return "box"
    try:
        hull_volume = tm.convex_hull.volume
    except Exception:  # degenerate input: fall back to the exact mesh
        return "mesh"
    return "hull" if hull_volume > 0 and volume >= 0.9 * hull_volume else "mesh"


def collider_mesh(mesh: Mesh, kind: str) -> trimesh.Trimesh:
    """Collision geometry (in the mesh's own frame) for ``kind`` box|hull|mesh."""
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    if kind == "box":
        return trimesh.creation.box(bounds=tm.bounds)
    if kind == "hull":
        return tm.convex_hull
    return trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)


def node_extras(node: SceneNode, collider: str | None = None) -> dict:
    """``extras`` for one node: {"geogen": {...}} or {} when there is nothing to say."""
    data: dict = {}
    if node.tags:
        data["tags"] = list(node.tags)
    for key, value in node.meta.items():
        if key != "collider":
            data[key] = value.tolist() if hasattr(value, "tolist") else value
    if collider is not None:
        data["collider"] = collider
    if not data:
        return {}
    return {"geogen": {"version": EXTRAS_VERSION, **data}}


def to_trimesh_scene(root: SceneNode, colliders: bool = True) -> trimesh.Scene:
    """Build a trimesh Scene mirroring the SceneNode hierarchy.

    Nodes carry ``extras.geogen``; with ``colliders`` each mesh node that gets
    a collider has a ``<name>-colonly`` / ``<name>-convcolonly`` child.
    """
    scene = trimesh.Scene(base_frame="world")
    cache: dict = {}
    used: set[str] = {"world"}

    def unique(name: str) -> str:
        candidate, i = name, 1
        while candidate in used:
            i += 1
            candidate = f"{name}_{i}"
        used.add(candidate)
        return candidate

    def visit(node: SceneNode, parent_name: str) -> None:
        name = unique(node.name)
        matrix = node.transform.to_matrix()
        has_mesh = node.mesh is not None and len(node.mesh.faces) > 0
        collider = resolve_collider(node) if has_mesh else None
        extras = node_extras(node, collider)
        if has_mesh:
            scene.add_geometry(
                to_trimesh(node.mesh, cache),
                node_name=name,
                geom_name=name,
                parent_node_name=parent_name,
                transform=matrix,
                metadata=extras or None,
            )
            if colliders and collider in COLLIDER_SUFFIX:
                col_name = unique(f"{name}{COLLIDER_SUFFIX[collider]}")
                scene.add_geometry(
                    collider_mesh(node.mesh, collider),
                    node_name=col_name,
                    geom_name=col_name,
                    parent_node_name=name,
                    metadata={"geogen": {"version": EXTRAS_VERSION, "type": "collider", "shape": collider}},
                )
        else:
            scene.graph.update(frame_from=parent_name, frame_to=name, matrix=matrix,
                               **({"metadata": extras} if extras else {}))
        for child in node.children:
            visit(child, name)

    visit(root, "world")
    return scene


def manifest_path(model_path: str | Path) -> Path:
    """Where the manifest for an exported model lives (``chair.glb`` -> ``chair.manifest.json``)."""
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}.manifest.json")


def gameplay_summary(root: SceneNode) -> dict[str, list[dict]]:
    """Rooms and spawns in world space, for the manifest.

    Rooms come from ``room_volume`` nodes (centre, size, and the volume's
    world rotation about Y); spawns from ``spawn`` nodes (position and
    ``forward``, the node's +Z axis: the direction the player should face).
    """
    rooms, spawns = [], []
    for node in root.iter_nodes():
        kind = node.meta.get("type")
        if kind not in ("room_volume", "spawn"):
            continue
        world = node.world_transform()
        position = [round(float(v), 6) for v in world[:3, 3]]
        forward = world[:3, 2] / max(np.linalg.norm(world[:3, 2]), 1e-12)
        if kind == "room_volume":
            room = dict(node.meta.get("room", {}))
            rooms.append({
                **room,
                "node": node.name,
                "center": position,
                "size": [float(v) for v in node.meta.get("size", [0, 0, 0])],
                "yaw_deg": round(float(np.degrees(np.arctan2(world[0, 2], world[2, 2]))), 6),
            })
        else:
            spawns.append({"name": node.name, "position": position,
                           "forward": [round(float(v), 6) for v in forward]})
    return {"rooms": rooms, "spawns": spawns}


def write_manifest(model_path: str | Path, player: PlayerSpec | None = None,
                   root: SceneNode | None = None) -> Path:
    """Write the runtime manifest next to an exported glTF model."""
    model_path = Path(model_path)
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "name": model_path.stem,
        "model": model_path.name,
        "units": "m",
        "up": "+Y",
        "extras": {"key": "geogen", "version": EXTRAS_VERSION, "schema": EXTRAS_SCHEMA},
        "player": (player or load_player_spec()).to_dict(),
        **(gameplay_summary(root) if root is not None else {"rooms": [], "spawns": []}),
    }
    out = manifest_path(model_path)
    # Written last and atomically: runtimes watch the manifest to know an export finished.
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(out)
    return out


def export_scene(root: SceneNode, path: str | Path, player: PlayerSpec | None = None) -> Path:
    """Export ``root`` to ``path``; the format is chosen by the file extension.

    glTF/GLB exports also get a manifest (see ``write_manifest``); ``player``
    overrides the project's default player spec in it.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        raise ValueError(f"Unsupported export format '{suffix}'. Use one of {sorted(FORMATS)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # OBJ has no node extras or hierarchy, so collider meshes would just be clutter.
    scene = to_trimesh_scene(root, colliders=suffix != ".obj")
    if suffix == ".obj":
        # OBJ has no hierarchy: bake world transforms. trimesh writes the
        # .mtl and texture images alongside the .obj.
        scene.export(str(path), file_type="obj", include_normals=True, include_texture=True)
    elif suffix == ".glb":
        # Write-then-rename so a runtime watching the file never reads half a GLB.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(scene.export(file_type="glb"))
        tmp.replace(path)
        write_manifest(path, player, root)
    else:
        scene.export(str(path), file_type=suffix[1:])
        write_manifest(path, player, root)
    return path
