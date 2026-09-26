"""Constructive solid geometry (union / difference / intersection) via manifold3d.

Inputs must be closed meshes (the generators produce these). UVs are carried
through the boolean as vertex properties, so faces created by a cutter keep
the cutter's metric UVs; normals are recomputed with a crease angle
afterwards because a subtracted cutter's normals face the wrong way.

Skin weights ride along too, as one dense weight column per joint the
operands use (so new vertices interpolate them), and come back as the top
four influences. Vertices left without weights (from an unskinned operand)
take their nearest skinned vertex's.
"""

from __future__ import annotations

import numpy as np

from . import meshops
from .mesh import Mesh


class CSGError(ValueError):
    """Raised when an operand isn't a valid closed manifold."""


def _joint_columns(meshes) -> np.ndarray | None:
    """Sorted joint ids used by any operand (the dense weight columns), or None if none is skinned."""
    used = [np.unique(m.joints[m.weights > 0]) for m in meshes if m.joints is not None and m.weights is not None]
    return np.unique(np.concatenate(used)).astype(np.int64) if used else None


def _dense_weights(mesh: Mesh, columns: np.ndarray) -> np.ndarray:
    dense = np.zeros((len(mesh.vertices), len(columns)))
    if mesh.joints is not None and mesh.weights is not None:
        col = np.searchsorted(columns, mesh.joints)
        rows = np.repeat(np.arange(len(mesh.vertices)), mesh.joints.shape[1])
        mask = (mesh.weights > 0).reshape(-1)
        np.add.at(dense, (rows[mask], col.reshape(-1)[mask]), mesh.weights.reshape(-1)[mask])
    return dense


def _sparse_weights(vertices: np.ndarray, dense: np.ndarray, columns: np.ndarray):
    """Dense weight columns -> (joints, weights) top four, normalised; empty rows copy their nearest neighbour."""
    from scipy.spatial import cKDTree

    from .skin import normalize_weights

    dense = np.clip(dense, 0.0, None)
    empty = dense.sum(axis=1) < 1e-6
    if empty.any() and not empty.all():
        _, nearest = cKDTree(vertices[~empty]).query(vertices[empty])
        dense[empty] = dense[~empty][nearest]
    # Vertices at one position (split by UV seams or material) came from different operands and
    # may carry different weights; they must move together or posing tears the surface open.
    _, group = np.unique(np.round(vertices / 1e-6).astype(np.int64), axis=0, return_inverse=True)
    group = group.reshape(-1)
    sums = np.zeros((group.max() + 1, dense.shape[1]))
    np.add.at(sums, group, dense / np.maximum(dense.sum(axis=1, keepdims=True), 1e-12))
    dense = sums[group]
    index = np.tile(columns, (len(dense), 1))
    return normalize_weights(index, dense)


def _to_manifold(mesh: Mesh, slots: np.ndarray | None = None, columns: np.ndarray | None = None):
    """Manifold with properties (x, y, z, u, v, material slot[, weight per joint column]).

    ``slots`` (one per face) become a per-vertex property; vertices are
    split where faces of different slots meet so each triangle reads its own.
    """
    import manifold3d

    uvs = mesh.uvs if mesh.uvs is not None else np.zeros((len(mesh.vertices), 2))
    verts, faces = mesh.vertices, mesh.faces
    weights = _dense_weights(mesh, columns) if columns is not None else np.zeros((len(verts), 0))
    slot_prop = np.zeros(len(verts))
    if slots is not None:
        corner = faces.reshape(-1)
        key = np.column_stack([corner, np.repeat(slots, 3)])
        uniq, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
        verts, uvs, weights = verts[uniq[:, 0]], uvs[uniq[:, 0]], weights[uniq[:, 0]]
        slot_prop = uniq[:, 1].astype(np.float64)
        faces = inverse.reshape(-1, 3)
    props = np.column_stack([verts, uvs, slot_prop, weights]).astype(np.float32)
    mm = manifold3d.Mesh(vert_properties=np.ascontiguousarray(props),
                         tri_verts=np.ascontiguousarray(faces.astype(np.uint32)))
    mm.merge()  # stitch vertices that share a position but differ in UV
    result = manifold3d.Manifold(mm)
    if result.status() != manifold3d.Error.NoError:
        raise CSGError(f"mesh is not a closed manifold ({result.status().name})")
    return result


def _from_manifold(man, material, crease_angle: float, materials: list | None = None,
                   columns: np.ndarray | None = None) -> Mesh:
    out = man.to_mesh()
    props = np.asarray(out.vert_properties, dtype=np.float64)
    faces = np.asarray(out.tri_verts, dtype=np.int64)
    mesh = Mesh(props[:, :3], faces, uvs=props[:, 3:5], material=material)
    if columns is not None and len(props):
        mesh.joints, mesh.weights = _sparse_weights(props[:, :3], props[:, 6:6 + len(columns)], columns)
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
    columns = _joint_columns([target])
    result = _to_manifold(target, target_slots, columns)
    for i, cutter in enumerate(cutters):
        _check(cutter, f"CSG cutter #{i}")
        cut_slots = np.zeros(len(cutter.faces), dtype=np.int64) if target_slots is not None else None
        result = result - _to_manifold(cutter, cut_slots, columns)
    return _from_manifold(result, target.material, crease_angle, slots if target_slots is not None else None,
                          columns)


def union(*meshes: Mesh, crease_angle: float = 40.0) -> Mesh:
    """Merge solids into one watertight mesh (internal faces removed); faces keep
    their operand's material (material groups when they differ)."""
    if not meshes:
        raise ValueError("union needs at least one mesh")
    slots, per_mesh = _slots(list(meshes))
    columns = _joint_columns(meshes)
    result = None
    for i, (mesh, face_slots) in enumerate(zip(meshes, per_mesh)):
        _check(mesh, f"CSG operand #{i}")
        m = _to_manifold(mesh, face_slots, columns)
        result = m if result is None else result + m
    return _from_manifold(result, slots[0] if slots else None, crease_angle,
                          slots if per_mesh[0] is not None else None, columns)


def intersection(a: Mesh, b: Mesh, crease_angle: float = 40.0) -> Mesh:
    _check(a, "CSG operand #0")
    _check(b, "CSG operand #1")
    slots, (sa, sb) = _slots([a, b])
    columns = _joint_columns([a, b])
    return _from_manifold(_to_manifold(a, sa, columns) ^ _to_manifold(b, sb, columns),
                          slots[0] if slots else a.material, crease_angle, slots if sa is not None else None, columns)
