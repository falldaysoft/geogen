"""Constructive solid geometry (union / difference / intersection) via manifold3d.

Inputs must be closed meshes (the generators produce these). UVs are carried
through the boolean as vertex properties, so faces created by a cutter keep
the cutter's metric UVs; normals are recomputed with a crease angle
afterwards because a subtracted cutter's normals face the wrong way.
"""

from __future__ import annotations

import numpy as np

from . import meshops
from .mesh import Mesh


class CSGError(ValueError):
    """Raised when an operand isn't a valid closed manifold."""


def _to_manifold(mesh: Mesh):
    import manifold3d

    uvs = mesh.uvs if mesh.uvs is not None else np.zeros((len(mesh.vertices), 2))
    props = np.hstack([mesh.vertices, uvs]).astype(np.float32)
    mm = manifold3d.Mesh(vert_properties=np.ascontiguousarray(props),
                         tri_verts=np.ascontiguousarray(mesh.faces.astype(np.uint32)))
    mm.merge()  # stitch vertices that share a position but differ in UV
    result = manifold3d.Manifold(mm)
    if result.status() != manifold3d.Error.NoError:
        raise CSGError(f"mesh is not a closed manifold ({result.status().name})")
    return result


def _from_manifold(man, material, crease_angle: float) -> Mesh:
    out = man.to_mesh()
    props = np.asarray(out.vert_properties, dtype=np.float64)
    faces = np.asarray(out.tri_verts, dtype=np.int64)
    mesh = Mesh(props[:, :3], faces, uvs=props[:, 3:5], material=material)
    if len(faces) == 0:
        return mesh
    return meshops.compute_normals(mesh, crease_angle)


def _check(mesh: Mesh, role: str) -> None:
    report = meshops.validate(mesh)
    if not report.watertight:
        raise CSGError(f"{role} is not closed ({report.boundary_edges} open edges)")


def difference(target: Mesh, *cutters: Mesh, crease_angle: float = 40.0) -> Mesh:
    """Subtract every cutter from ``target``. The result keeps target's material."""
    if not cutters:
        return target
    _check(target, "CSG target")
    result = _to_manifold(target)
    for i, cutter in enumerate(cutters):
        _check(cutter, f"CSG cutter #{i}")
        result = result - _to_manifold(cutter)
    return _from_manifold(result, target.material, crease_angle)


def union(*meshes: Mesh, crease_angle: float = 40.0) -> Mesh:
    """Merge solids into one watertight mesh (internal faces removed)."""
    if not meshes:
        raise ValueError("union needs at least one mesh")
    result = None
    for i, mesh in enumerate(meshes):
        _check(mesh, f"CSG operand #{i}")
        m = _to_manifold(mesh)
        result = m if result is None else result + m
    return _from_manifold(result, meshes[0].material, crease_angle)


def intersection(a: Mesh, b: Mesh, crease_angle: float = 40.0) -> Mesh:
    _check(a, "CSG operand #0")
    _check(b, "CSG operand #1")
    return _from_manifold(_to_manifold(a) ^ _to_manifold(b), a.material, crease_angle)
