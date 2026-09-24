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


def _to_manifold(mesh: Mesh, slots: np.ndarray | None = None):
    """Manifold with properties (x, y, z, u, v, material slot).

    ``slots`` (one per face) become a per-vertex property; vertices are
    split where faces of different slots meet so each triangle reads its own.
    """
    import manifold3d

    uvs = mesh.uvs if mesh.uvs is not None else np.zeros((len(mesh.vertices), 2))
    verts, faces = mesh.vertices, mesh.faces
    slot_prop = np.zeros(len(verts))
    if slots is not None:
        corner = faces.reshape(-1)
        key = np.column_stack([corner, np.repeat(slots, 3)])
        uniq, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
        verts, uvs = verts[uniq[:, 0]], uvs[uniq[:, 0]]
        slot_prop = uniq[:, 1].astype(np.float64)
        faces = inverse.reshape(-1, 3)
    props = np.column_stack([verts, uvs, slot_prop]).astype(np.float32)
    mm = manifold3d.Mesh(vert_properties=np.ascontiguousarray(props),
                         tri_verts=np.ascontiguousarray(faces.astype(np.uint32)))
    mm.merge()  # stitch vertices that share a position but differ in UV
    result = manifold3d.Manifold(mm)
    if result.status() != manifold3d.Error.NoError:
        raise CSGError(f"mesh is not a closed manifold ({result.status().name})")
    return result


def _from_manifold(man, material, crease_angle: float, materials: list | None = None) -> Mesh:
    out = man.to_mesh()
    props = np.asarray(out.vert_properties, dtype=np.float64)
    faces = np.asarray(out.tri_verts, dtype=np.int64)
    mesh = Mesh(props[:, :3], faces, uvs=props[:, 3:5], material=material)
    if materials is not None and len(materials) > 1 and len(faces):
        mesh.face_materials = np.rint(props[faces[:, 0], 5]).astype(np.int64)
        mesh.materials = list(materials)
    if len(faces) == 0:
        return mesh
    return meshops.compute_normals(mesh, crease_angle)


def _check(mesh: Mesh, role: str) -> None:
    report = meshops.validate(mesh)
    if not report.watertight:
        raise CSGError(f"{role} is not closed ({report.boundary_edges} open edges)")


def _slots(meshes: list[Mesh]) -> tuple[list, list[np.ndarray | None]]:
    """Shared material slots for several operands, and each operand's per-face slots.

    With a single material overall the per-face slots are None (nothing to track).
    """
    slots: list = []
    per_mesh = []
    for mesh in meshes:
        per_face, mats = mesh.effective_face_materials()
        local = []
        for m in mats:
            index = next((i for i, existing in enumerate(slots) if existing is m), None)
            if index is None:
                slots.append(m)
                index = len(slots) - 1
            local.append(index)
        per_mesh.append(np.asarray(local, dtype=np.int64)[per_face])
    if len(slots) <= 1:
        return slots, [None] * len(meshes)
    return slots, per_mesh


def difference(target: Mesh, *cutters: Mesh, crease_angle: float = 40.0) -> Mesh:
    """Subtract every cutter from ``target``. The result keeps target's material(s);
    faces cut by a cutter take the target's primary material."""
    if not cutters:
        return target
    _check(target, "CSG target")
    slots, (target_slots,) = _slots([target])
    result = _to_manifold(target, target_slots)
    for i, cutter in enumerate(cutters):
        _check(cutter, f"CSG cutter #{i}")
        cut_slots = np.zeros(len(cutter.faces), dtype=np.int64) if target_slots is not None else None
        result = result - _to_manifold(cutter, cut_slots)
    return _from_manifold(result, target.material, crease_angle, slots if target_slots is not None else None)


def union(*meshes: Mesh, crease_angle: float = 40.0) -> Mesh:
    """Merge solids into one watertight mesh (internal faces removed); faces keep
    their operand's material (material groups when they differ)."""
    if not meshes:
        raise ValueError("union needs at least one mesh")
    slots, per_mesh = _slots(list(meshes))
    result = None
    for i, (mesh, face_slots) in enumerate(zip(meshes, per_mesh)):
        _check(mesh, f"CSG operand #{i}")
        m = _to_manifold(mesh, face_slots)
        result = m if result is None else result + m
    return _from_manifold(result, slots[0] if slots else None, crease_angle,
                          slots if per_mesh[0] is not None else None)


def intersection(a: Mesh, b: Mesh, crease_angle: float = 40.0) -> Mesh:
    _check(a, "CSG operand #0")
    _check(b, "CSG operand #1")
    slots, (sa, sb) = _slots([a, b])
    return _from_manifold(_to_manifold(a, sa) ^ _to_manifold(b, sb), slots[0] if slots else a.material,
                          crease_angle, slots if sa is not None else None)
