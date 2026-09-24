"""Mesh processing operations: welding, normals, tangents and validation.

All functions are pure: they return new Mesh objects (or reports) and never
mutate their input. Operations are vectorised with numpy so they stay usable
on terrain-sized meshes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .mesh import Mesh

DEFAULT_CREASE_ANGLE = 30.0  # degrees


def face_normals(vertices: NDArray[np.float64], faces: NDArray[np.int64]) -> tuple[NDArray, NDArray]:
    """Return (unit face normals, face areas) for a triangle mesh."""
    v0 = vertices[faces[:, 0]]
    cross = np.cross(vertices[faces[:, 1]] - v0, vertices[faces[:, 2]] - v0)
    length = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    ok = length > 1e-20
    normals[ok] = cross[ok] / length[ok, None]
    return normals, length * 0.5


def _quantize(values: NDArray[np.float64], tol: float) -> NDArray[np.int64]:
    return np.round(values / tol).astype(np.int64)


def _position_ids(vertices: NDArray[np.float64], tol: float) -> NDArray[np.int64]:
    """Map each vertex to an id shared by all vertices at the same position."""
    _, inverse = np.unique(_quantize(vertices, tol), axis=0, return_inverse=True)
    return inverse.reshape(-1)


def weld_vertices(mesh: Mesh, tol: float = 1e-6) -> Mesh:
    """Merge vertices that share position, normal and UV (within ``tol``).

    Vertices that differ in normal or UV (hard edges, UV seams) are kept
    separate so shading and texturing are preserved.
    """
    keys = [_quantize(mesh.vertices, tol)]
    if mesh.normals is not None:
        keys.append(_quantize(mesh.normals, 1e-4))
    if mesh.uvs is not None:
        keys.append(_quantize(mesh.uvs, 1e-5))
    key = np.hstack(keys)

    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    faces = inverse[mesh.faces]
    # Drop faces that collapsed to a line or point.
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])

    return Mesh(
        vertices=mesh.vertices[first],
        faces=faces[keep],
        normals=mesh.normals[first] if mesh.normals is not None else None,
        uvs=mesh.uvs[first] if mesh.uvs is not None else None,
        material=mesh.material,
    )


def compute_normals(
    mesh: Mesh,
    crease_angle: float = DEFAULT_CREASE_ANGLE,
    tol: float = 1e-6,
) -> Mesh:
    """Compute vertex normals with crease-angle smoothing.

    Faces meeting at a vertex share a smoothed normal when the angle between
    their face normals is at most ``crease_angle`` degrees; sharper edges get
    split vertices so they render as hard edges. Smoothing ignores UV seams
    (vertices are grouped by position), so a seam never shows as a crease.

    ``crease_angle=0`` gives flat shading, ``180`` gives fully smooth.
    """
    verts, faces = mesh.vertices, mesh.faces
    n_faces = len(faces)
    if n_faces == 0:
        return mesh.copy()

    fn, area = face_normals(verts, faces)
    weighted = fn * area[:, None]

    # CSR incidence: welded position id -> incident faces.
    pos_id = _position_ids(verts, tol)
    corner_pos = pos_id[faces].reshape(-1)  # (3F,)
    corner_face = np.repeat(np.arange(n_faces), 3)
    order = np.argsort(corner_pos, kind="stable")
    sorted_pos = corner_pos[order]
    sorted_face = corner_face[order]
    n_pos = int(pos_id.max()) + 1
    starts = np.searchsorted(sorted_pos, np.arange(n_pos))
    counts = np.bincount(sorted_pos, minlength=n_pos)

    # For each corner (face f, position p), enumerate every face g incident to p.
    corner_counts = counts[corner_pos]
    pair_corner = np.repeat(np.arange(3 * n_faces), corner_counts)
    offsets = np.arange(len(pair_corner)) - np.repeat(np.cumsum(corner_counts) - corner_counts, corner_counts)
    pair_face = sorted_face[np.repeat(starts[corner_pos], corner_counts) + offsets]

    cos_limit = np.cos(np.radians(np.clip(crease_angle, 0.0, 180.0))) - 1e-6
    own_normal = fn[corner_face[pair_corner]]
    include = np.einsum("ij,ij->i", own_normal, fn[pair_face]) >= cos_limit
    # A face always contributes to its own corners (covers degenerate faces).
    include |= pair_face == corner_face[pair_corner]

    corner_normals = np.zeros((3 * n_faces, 3))
    np.add.at(corner_normals, pair_corner[include], weighted[pair_face[include]])
    length = np.linalg.norm(corner_normals, axis=1)
    bad = length < 1e-20
    corner_normals[~bad] /= length[~bad, None]
    corner_normals[bad] = fn[corner_face[bad]]

    # Split original vertices wherever their corners got different normals.
    corner_vert = faces.reshape(-1)
    key = np.hstack([corner_vert[:, None], _quantize(corner_normals, 1e-4)])
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    src = corner_vert[first]

    return Mesh(
        vertices=verts[src],
        faces=inverse.reshape(-1, 3),
        normals=corner_normals[first],
        uvs=mesh.uvs[src] if mesh.uvs is not None else None,
        material=mesh.material,
    )


def ensure_normals(mesh: Mesh, crease_angle: float = DEFAULT_CREASE_ANGLE) -> Mesh:
    """Return ``mesh`` unchanged if it has normals, otherwise compute them."""
    if mesh.normals is not None and len(mesh.normals) == len(mesh.vertices):
        return mesh
    return compute_normals(mesh, crease_angle)


def compute_tangents(mesh: Mesh) -> NDArray[np.float64]:
    """Compute per-vertex tangents (Nx4, xyz + handedness w) for normal mapping.

    Requires normals and UVs. Tangents are accumulated per triangle from UV
    derivatives, Gram-Schmidt orthogonalised against the normal, and w stores
    the bitangent sign as in glTF.
    """
    if mesh.normals is None or mesh.uvs is None:
        raise ValueError("compute_tangents requires normals and UVs")
    v, f, uv, n = mesh.vertices, mesh.faces, mesh.uvs, mesh.normals

    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    d1 = uv[f[:, 1]] - uv[f[:, 0]]
    d2 = uv[f[:, 2]] - uv[f[:, 0]]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-20, 1.0 / np.where(det == 0, 1, det), 0.0)
    t = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    b = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]

    tan = np.zeros_like(v)
    bit = np.zeros_like(v)
    for k in range(3):
        np.add.at(tan, f[:, k], t)
        np.add.at(bit, f[:, k], b)

    # Gram-Schmidt against the normal.
    tan -= n * np.einsum("ij,ij->i", n, tan)[:, None]
    length = np.linalg.norm(tan, axis=1)
    bad = length < 1e-12
    if bad.any():
        # Fallback: any vector perpendicular to the normal.
        helper = np.where(np.abs(n[bad, 0:1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        tan[bad] = np.cross(n[bad], helper)
        length[bad] = np.linalg.norm(tan[bad], axis=1)
    tan /= np.maximum(length, 1e-20)[:, None]
    w = np.where(np.einsum("ij,ij->i", np.cross(n, tan), bit) < 0.0, -1.0, 1.0)
    return np.hstack([tan, w[:, None]])


@dataclass
class MeshReport:
    """Result of :func:`validate`."""

    vertex_count: int
    face_count: int
    degenerate_faces: int = 0
    nan_values: int = 0
    unused_vertices: int = 0
    boundary_edges: int = 0
    non_manifold_edges: int = 0
    inconsistent_winding_edges: int = 0
    bad_normals: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def watertight(self) -> bool:
        return self.boundary_edges == 0 and self.non_manifold_edges == 0

    @property
    def ok(self) -> bool:
        return not self.issues


def validate(mesh: Mesh, tol: float = 1e-6, area_eps: float = 1e-12) -> MeshReport:
    """Check a mesh for common defects.

    Topology checks (boundary / non-manifold / winding) are done on welded
    positions so UV and normal seams don't count as open edges.
    ``issues`` lists only hard errors (NaNs, degenerate faces, bad indices,
    bad normals); openness is reported but not treated as an error since
    many parts are intentionally open.
    """
    v, f = mesh.vertices, mesh.faces
    report = MeshReport(vertex_count=len(v), face_count=len(f))
    if len(f) and (f.min() < 0 or f.max() >= len(v)):
        report.issues.append("face indices out of range")
        return report

    report.nan_values = int(np.count_nonzero(~np.isfinite(v)))
    if mesh.normals is not None:
        report.nan_values += int(np.count_nonzero(~np.isfinite(mesh.normals)))
        lengths = np.linalg.norm(mesh.normals, axis=1)
        report.bad_normals = int(np.count_nonzero(np.abs(lengths - 1.0) > 1e-3))
    if mesh.uvs is not None:
        report.nan_values += int(np.count_nonzero(~np.isfinite(mesh.uvs)))

    if len(f):
        _, area = face_normals(v, f)
        report.degenerate_faces = int(np.count_nonzero(area <= area_eps))
        report.unused_vertices = len(v) - len(np.unique(f))

        pid = _position_ids(v, tol)
        pf = pid[f]
        directed = np.concatenate([pf[:, [0, 1]], pf[:, [1, 2]], pf[:, [2, 0]]])
        directed = directed[directed[:, 0] != directed[:, 1]]
        undirected = np.sort(directed, axis=1)
        uniq, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
        report.boundary_edges = int(np.count_nonzero(counts == 1))
        report.non_manifold_edges = int(np.count_nonzero(counts > 2))
        # Consistently wound manifold edges are traversed once in each direction.
        forward = (directed[:, 0] < directed[:, 1]).astype(np.int64)
        fwd_count = np.bincount(inverse.reshape(-1), weights=forward, minlength=len(uniq))
        two = counts == 2
        report.inconsistent_winding_edges = int(np.count_nonzero(two & (fwd_count != 1)))

    if report.nan_values:
        report.issues.append(f"{report.nan_values} non-finite values")
    if report.degenerate_faces:
        report.issues.append(f"{report.degenerate_faces} degenerate faces")
    if report.bad_normals:
        report.issues.append(f"{report.bad_normals} non-unit normals")
    if report.inconsistent_winding_edges:
        report.issues.append(f"{report.inconsistent_winding_edges} edges with inconsistent winding")
    return report


def decimate(mesh: Mesh, ratio: float, crease_angle: float = 40.0) -> Mesh:
    """Reduce ``mesh`` to about ``ratio`` of its triangles, keeping UVs and hard edges.

    Uses manifold3d's edge-collapse simplification (UV seams and material
    boundaries are preserved because UVs ride along as vertex properties),
    binary-searching the collapse tolerance for the target triangle count.
    Meshes that aren't closed manifolds are returned unchanged.
    """
    from .csg import CSGError, _from_manifold, _to_manifold

    if ratio >= 1.0 or len(mesh.faces) < 8:
        return mesh
    try:
        man = _to_manifold(mesh)
    except CSGError:
        return mesh
    target = max(4, int(len(mesh.faces) * ratio))
    extent = float(np.ptp(mesh.vertices, axis=0).max()) or 1.0
    lo, hi = 0.0, extent * 0.25
    best = man
    for _ in range(18):
        mid = (lo + hi) / 2
        candidate = man.simplify(mid)
        tris = candidate.num_tri()
        if tris > target:
            lo = mid
        else:
            hi, best = mid, candidate
        if abs(tris - target) <= max(2, target * 0.05):
            best = candidate
            break
    if best.num_tri() == 0:
        return mesh
    return _from_manifold(best, mesh.material, crease_angle)
