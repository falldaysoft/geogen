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
    """

    def __init__(
        self,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int64],
        normals: NDArray[np.float64] | None = None,
        uvs: NDArray[np.float64] | None = None,
        material: Material | None = None,
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

        self._trimesh_cache: trimesh.Trimesh | None = None

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
        )

    def copy(self) -> Mesh:
        """Create a deep copy of this mesh."""
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            uvs=self.uvs.copy() if self.uvs is not None else None,
            material=self.material,  # Material reference is shared (not deep copied)
        )

    @staticmethod
    def merge(meshes: list[Mesh]) -> Mesh:
        """Merge multiple meshes into a single mesh.

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

        return Mesh(
            vertices=np.vstack(all_vertices),
            faces=np.vstack(all_faces),
            normals=np.vstack(all_normals) if has_normals else None,
            uvs=np.vstack(all_uvs) if has_uvs else None,
        )
