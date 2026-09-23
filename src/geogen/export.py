"""Export scene graphs to game-engine formats (glTF/GLB, OBJ).

glTF keeps the node hierarchy with local transforms, so an exported house
still has separately addressable doors and windows. Meshes carry explicit
crease-angle normals, and PBR metallic-roughness materials with base colour,
metallic-roughness, normal and occlusion textures. Metric UVs are converted
to texture space using each material's ``tile_size`` so textures repeat at
the right real-world scale (samplers use REPEAT wrapping).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .core import meshops
from .core.mesh import Mesh
from .core.node import SceneNode
from .materials.material import Material

FORMATS = {".glb", ".gltf", ".obj"}


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


def export_scene(root: SceneNode, path: str | Path) -> Path:
    """Export ``root`` to ``path``; the format is chosen by the file extension."""
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
    else:
        scene.export(str(path), file_type=suffix[1:])
    return path
