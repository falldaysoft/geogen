"""Export scene graphs to game-engine formats (glTF/GLB, OBJ).

glTF keeps the node hierarchy with local transforms, so an exported house
still has separately addressable doors and windows. Meshes carry explicit
crease-angle normals, and PBR metallic-roughness materials with base colour,
metallic-roughness, normal and occlusion textures. Metric UVs are converted
to texture space using each material's ``tile_size`` so textures repeat at
the right real-world scale (samplers use REPEAT wrapping).

glTF exports also write ``<name>.manifest.json`` alongside the model: the
contract the Godot runtime reads (units, up axis, model file, player spec).
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


def to_trimesh_scene(root: SceneNode) -> trimesh.Scene:
    """Build a trimesh Scene mirroring the SceneNode hierarchy."""
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
        if node.mesh is not None and len(node.mesh.faces):
            scene.add_geometry(
                to_trimesh(node.mesh, cache),
                node_name=name,
                geom_name=name,
                parent_node_name=parent_name,
                transform=matrix,
            )
        else:
            scene.graph.update(frame_from=parent_name, frame_to=name, matrix=matrix)
        for child in node.children:
            visit(child, name)

    visit(root, "world")
    return scene


def manifest_path(model_path: str | Path) -> Path:
    """Where the manifest for an exported model lives (``chair.glb`` -> ``chair.manifest.json``)."""
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}.manifest.json")


def write_manifest(model_path: str | Path, player: PlayerSpec | None = None) -> Path:
    """Write the runtime manifest next to an exported glTF model."""
    model_path = Path(model_path)
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "name": model_path.stem,
        "model": model_path.name,
        "units": "m",
        "up": "+Y",
        "player": (player or load_player_spec()).to_dict(),
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
    scene = to_trimesh_scene(root)
    if suffix == ".obj":
        # OBJ has no hierarchy: bake world transforms. trimesh writes the
        # .mtl and texture images alongside the .obj.
        scene.export(str(path), file_type="obj", include_normals=True, include_texture=True)
    elif suffix == ".glb":
        # Write-then-rename so a runtime watching the file never reads half a GLB.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(scene.export(file_type="glb"))
        tmp.replace(path)
        write_manifest(path, player)
    else:
        scene.export(str(path), file_type=suffix[1:])
        write_manifest(path, player)
    return path
