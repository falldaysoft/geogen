"""Mesh class for geometry data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import trimesh
    from ..materials.material import Material


class Mesh:
    """Container for mesh geometry data.

    Stores vertices, faces, and optional normals/UVs as numpy arrays.
    Can convert to/from trimesh for rendering and export.

    Multi-material meshes carry ``face_materials`` (one index per face) into
    ``materials``; a ``None`` entry falls back to ``material``. ``groups()``
    splits them for renderers and exporters (one glTF primitive each).
    ``colors`` are optional per-vertex RGBA (0-1) tints, exported as COLOR_0.
    """

    def __init__(
        self,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int64],
        normals: NDArray[np.float64] | None = None,
        uvs: NDArray[np.float64] | None = None,
        material: Material | None = None,
        face_materials: NDArray[np.int64] | None = None,
        materials: list[Material | None] | None = None,
        colors: NDArray[np.float64] | None = None,
    ) -> None:
        """Create a mesh from geometry data.

        Args:
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of triangle indices
            normals: Optional Nx3 array of vertex normals
            uvs: Optional Nx2 array of texture coordinates
            material: Optional material for texturing
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.normals = (
            np.asarray(normals, dtype=np.float64) if normals is not None else None
        )
        self.uvs = np.asarray(uvs, dtype=np.float64) if uvs is not None else None
        self.material = material
        self.face_materials = np.asarray(face_materials, dtype=np.int64) if face_materials is not None else None
        self.materials = list(materials) if materials is not None else None
        self.colors = np.asarray(colors, dtype=np.float64) if colors is not None else None

        self._trimesh_cache: trimesh.Trimesh | None = None

    @property
    def multi_material(self) -> bool:
        return self.face_materials is not None and self.materials is not None and len(self.materials) > 1

    def face_material(self, index: int) -> Material | None:
        """Effective material of material slot ``index``."""
        if self.materials is None or not (0 <= index < len(self.materials)):
            return self.material
        return self.materials[index] if self.materials[index] is not None else self.material

    def effective_face_materials(self) -> tuple[NDArray[np.int64], list[Material | None]]:
        """(slot per face, material per slot), for single- and multi-material meshes alike."""
        if self.face_materials is None or self.materials is None:
            return np.zeros(len(self.faces), dtype=np.int64), [self.material]
        return self.face_materials, [self.face_material(i) for i in range(len(self.materials))]

    def with_attributes_of(self, source: Mesh, vertex_index=None, face_keep=None, face_repeat: int = 1) -> Mesh:
        """Carry ``source``'s material slots and colours onto this derived mesh.

        ``vertex_index`` maps this mesh's vertices to source vertices (for
        colours); ``face_keep`` is a mask/index of source faces kept, in order;
        ``face_repeat`` tiles face slots (subdivision makes faces in blocks).
        """
        self.material = source.material if self.material is None else self.material
        if source.face_materials is not None:
            slots = source.face_materials if face_keep is None else source.face_materials[face_keep]
            self.face_materials = np.tile(slots, face_repeat)
            self.materials = list(source.materials) if source.materials is not None else None
        if source.colors is not None and vertex_index is not None:
            self.colors = source.colors[vertex_index]
        return self

    def groups(self) -> list[tuple[Material | None, Mesh]]:
        """One (material, sub-mesh) per material slot in use (vertices compacted)."""
        if not self.multi_material:
            return [(self.material, self)]
        out = []
        for slot in np.unique(self.face_materials):
            faces = self.faces[self.face_materials == slot]
            used, remap = np.unique(faces, return_inverse=True)
            sub = Mesh(vertices=self.vertices[used], faces=remap.reshape(-1, 3),
                       normals=self.normals[used] if self.normals is not None else None,
                       uvs=self.uvs[used] if self.uvs is not None else None,
                       material=self.face_material(int(slot)),
                       colors=self.colors[used] if self.colors is not None else None)
            out.append((sub.material, sub))
        return out

    @property
    def vertex_count(self) -> int:
        """Number of vertices in the mesh."""
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        """Number of faces (triangles) in the mesh."""
        return len(self.faces)

    def to_trimesh(self, apply_material: bool = True) -> trimesh.Trimesh:
        """Convert to a trimesh.Trimesh object for rendering/export.

        Args:
            apply_material: If True and mesh has material+UVs, apply texture

        Returns:
            trimesh.Trimesh object with optional texture applied
        """
        import trimesh as tm

        if self._trimesh_cache is not None:
            return self._trimesh_cache

        mesh = tm.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            process=False,  # Don't modify our geometry
        )

        if self.normals is not None:
            mesh.vertex_normals = self.normals

        # Apply material texture if available
        if apply_material and self.material is not None and self.uvs is not None:
            texture_image = self.material.get_texture()
            mesh.visual = tm.visual.TextureVisuals(
                uv=self.uvs,
                image=texture_image,
            )

        self._trimesh_cache = mesh
        return mesh

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh) -> Mesh:
        """Create a Mesh from a trimesh.Trimesh object."""
        return cls(
            vertices=np.array(mesh.vertices),
            faces=np.array(mesh.faces),
            normals=np.array(mesh.vertex_normals) if mesh.vertex_normals is not None else None,
        )

    def transform(self, matrix: NDArray[np.float64]) -> Mesh:
        """Apply a 4x4 transformation matrix, returning a new mesh.

        Args:
            matrix: 4x4 transformation matrix

        Returns:
            New Mesh with transformed vertices and normals
        """
        # Transform vertices (homogeneous coordinates)
        ones = np.ones((len(self.vertices), 1))
        homogeneous = np.hstack([self.vertices, ones])
        transformed = (matrix @ homogeneous.T).T
        new_vertices = transformed[:, :3]

        # Transform normals (use inverse transpose of upper-left 3x3)
        new_normals = None
        if self.normals is not None:
            normal_matrix = np.linalg.inv(matrix[:3, :3]).T
            new_normals = (normal_matrix @ self.normals.T).T
            # Renormalize
            norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
            new_normals = np.divide(
                new_normals, norms, where=norms != 0, out=new_normals
            )

        return Mesh(
            vertices=new_vertices,
            faces=self.faces.copy(),
            normals=new_normals,
            uvs=self.uvs.copy() if self.uvs is not None else None,
            material=self.material,  # Material is preserved through transform
            face_materials=self.face_materials, materials=self.materials, colors=self.colors,
        )

    def copy(self) -> Mesh:
        """Create a deep copy of this mesh."""
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            uvs=self.uvs.copy() if self.uvs is not None else None,
            material=self.material,  # Material reference is shared (not deep copied)
            face_materials=self.face_materials.copy() if self.face_materials is not None else None,
            materials=self.materials,
            colors=self.colors.copy() if self.colors is not None else None,
        )

    @staticmethod
    def merge(meshes: list[Mesh]) -> Mesh:
        """Merge multiple meshes into a single mesh.

        Meshes with different materials become material groups of the
        result (``face_materials``); if all share one material (or none),
        the result has that single material.

        Args:
            meshes: List of Mesh objects to merge

        Returns:
            New Mesh containing all geometry
        """
        if not meshes:
            return Mesh(
                vertices=np.empty((0, 3)),
                faces=np.empty((0, 3), dtype=np.int64),
            )

        all_vertices = []
        all_faces = []
        all_normals = []
        all_uvs = []
        vertex_offset = 0
        has_normals = all(m.normals is not None for m in meshes)
        has_uvs = all(m.uvs is not None for m in meshes)

        for mesh in meshes:
            all_vertices.append(mesh.vertices)
            all_faces.append(mesh.faces + vertex_offset)
            if has_normals:
                all_normals.append(mesh.normals)
            if has_uvs:
                all_uvs.append(mesh.uvs)
            vertex_offset += len(mesh.vertices)

        # Material slots: the distinct effective materials, in order of appearance.
        slots: list = []
        face_slots = []
        for mesh in meshes:
            per_face, mats = mesh.effective_face_materials()
            local = []
            for m in mats:
                index = next((i for i, existing in enumerate(slots) if existing is m), None)
                if index is None:
                    slots.append(m)
                    index = len(slots) - 1
                local.append(index)
            face_slots.append(np.asarray(local, dtype=np.int64)[per_face] if len(per_face) else per_face)
        colors = None
        if any(m.colors is not None for m in meshes):
            colors = np.vstack([m.colors if m.colors is not None else np.ones((len(m.vertices), 4))
                                for m in meshes])
        grouped = len(slots) > 1
        return Mesh(
            vertices=np.vstack(all_vertices),
            faces=np.vstack(all_faces),
            normals=np.vstack(all_normals) if has_normals else None,
            uvs=np.vstack(all_uvs) if has_uvs else None,
            material=slots[0] if slots else None,
            face_materials=np.concatenate(face_slots) if grouped else None,
            materials=slots if grouped else None,
            colors=colors,
        )
