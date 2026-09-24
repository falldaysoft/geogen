"""Subdivision surfaces and noise displacement for organic forms.

- ``subdivide(mesh, levels, crease_angle)``: Loop subdivision of a triangle
  cage. Edges sharper than ``crease_angle`` (and open boundaries) are tagged
  as creases and keep their line (crease rules for edge and vertex points),
  so a box cage can become a pillow with a crisp seam or a fully smooth
  pebble (``crease_angle=None``). Creases turning by more than 60 degrees,
  and vertices where three or more meet, are corners and stay put.
- ``displace(mesh, amplitude, scale, octaves, seed)``: moves every vertex
  along its smoothed normal by fractal 3D gradient noise with features of
  ``scale`` metres. Vertices sharing a position move together, so closed
  meshes stay closed.

Both work on the position-welded mesh and drop UVs and normals; callers
re-project metric UVs (``uvmap.box_project``) and recompute normals.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .mesh import Mesh
from .meshops import _position_ids, face_normals


def _weld_positions(mesh: Mesh, tol: float = 1e-6) -> tuple[NDArray, NDArray]:
    ids = _position_ids(mesh.vertices, tol)
    n = int(ids.max()) + 1 if len(ids) else 0
    verts = np.zeros((n, 3))
    verts[ids] = mesh.vertices
    faces = ids[mesh.faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[keep]
    used, faces = np.unique(faces, return_inverse=True)   # drop unreferenced positions
    return verts[used], faces.reshape(-1, 3)


def _edges(faces: NDArray) -> tuple[NDArray, NDArray]:
    """Unique sorted edges and, per face corner edge (f, k: v_k -> v_k+1), its edge index."""
    half = np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).reshape(-1, 2)
    edges, inverse = np.unique(np.sort(half, axis=1), axis=0, return_inverse=True)
    return edges, inverse.reshape(-1, 3)


def _subdivide_once(verts: NDArray, faces: NDArray, sharp: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """One Loop step. ``sharp`` is a bool per edge of ``_edges(faces)``."""
    edges, face_edge = _edges(faces)
    n_v, n_e = len(verts), len(edges)

    # Opposite vertex of each face corner edge: v_{k+2}.
    opposite = np.roll(faces, -2, axis=1).reshape(-1)
    fe = face_edge.reshape(-1)
    face_count = np.bincount(fe, minlength=n_e)
    sharp = sharp | (face_count != 2)   # boundaries and non-manifold edges act as creases

    # Edge points.
    opp_sum = np.zeros((n_e, 3))
    np.add.at(opp_sum, fe, verts[opposite])
    a, b = verts[edges[:, 0]], verts[edges[:, 1]]
    edge_pts = np.where(sharp[:, None], (a + b) / 2, 3 / 8 * (a + b) + 1 / 8 * opp_sum)

    # Vertex points.
    valence = np.bincount(edges.reshape(-1), minlength=n_v)
    nbr_sum = np.zeros((n_v, 3))
    np.add.at(nbr_sum, edges[:, 0], b)
    np.add.at(nbr_sum, edges[:, 1], a)
    n = np.maximum(valence, 1)
    beta = np.where(n == 3, 3 / 16, 3 / (8 * n))
    smooth = (1 - n * beta)[:, None] * verts + beta[:, None] * nbr_sum

    sharp_count = np.bincount(edges[sharp].reshape(-1), minlength=n_v)
    sharp_sum = np.zeros((n_v, 3))
    np.add.at(sharp_sum, edges[sharp, 0], b[sharp])
    np.add.at(sharp_sum, edges[sharp, 1], a[sharp])
    crease = 3 / 4 * verts + 1 / 8 * sharp_sum
    new_verts = np.where((sharp_count == 2)[:, None], crease, smooth)
    # A crease that turns by more than 60 degrees (e.g. a square's open border) is a corner.
    d = b[sharp] - a[sharp]
    d /= np.maximum(np.linalg.norm(d, axis=1), 1e-20)[:, None]
    turn = np.zeros((n_v, 3))
    np.add.at(turn, edges[sharp, 0], d)
    np.add.at(turn, edges[sharp, 1], -d)
    corner = (sharp_count > 2) | ((sharp_count == 2) & (np.einsum("ij,ij->i", turn, turn) > 1.0))
    new_verts[corner] = verts[corner]           # corners stay put
    # (sharp_count == 1, a dart, uses the smooth rule.)

    # Faces: (v0, v1, v2) with edge points m0 = v0v1, m1 = v1v2, m2 = v2v0.
    m = face_edge + n_v
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    m0, m1, m2 = m[:, 0], m[:, 1], m[:, 2]
    new_faces = np.concatenate([
        np.stack([v0, m0, m2], axis=1),
        np.stack([v1, m1, m0], axis=1),
        np.stack([v2, m2, m1], axis=1),
        np.stack([m0, m1, m2], axis=1),
    ])

    # Child edges of a sharp edge stay sharp; interior edges are smooth.
    new_edges, _ = _edges(new_faces)
    sharp_keys = np.concatenate([
        np.sort(np.stack([edges[sharp, 0], np.flatnonzero(sharp) + n_v], axis=1), axis=1),
        np.sort(np.stack([edges[sharp, 1], np.flatnonzero(sharp) + n_v], axis=1), axis=1),
    ])
    new_sharp = _rows_in(new_edges, sharp_keys)
    return np.vstack([new_verts, edge_pts]), new_faces, new_sharp


def _rows_in(rows: NDArray, keys: NDArray) -> NDArray:
    if not len(keys):
        return np.zeros(len(rows), dtype=bool)
    width = int(max(rows.max(), keys.max())) + 1
    return np.isin(rows[:, 0] * width + rows[:, 1], keys[:, 0] * width + keys[:, 1])


def subdivide(mesh: Mesh, levels: int = 1, crease_angle: float | None = None) -> Mesh:
    """Loop-subdivide ``mesh`` ``levels`` times (each level quadruples the triangles).

    Edges whose dihedral angle exceeds ``crease_angle`` degrees stay creased;
    ``None`` smooths everything (open boundaries are always creases).
    """
    verts, faces = _weld_positions(mesh)
    edges, face_edge = _edges(faces)
    sharp = np.zeros(len(edges), dtype=bool)
    if crease_angle is not None and len(faces):
        fn, _ = face_normals(verts, faces)
        fe = face_edge.reshape(-1)
        face_of = np.repeat(np.arange(len(faces)), 3)
        order = np.argsort(fe, kind="stable")
        fe_sorted, faces_sorted = fe[order], face_of[order]
        # Manifold edges: exactly two consecutive entries.
        pair = np.flatnonzero((fe_sorted[1:] == fe_sorted[:-1]))
        cos = np.einsum("ij,ij->i", fn[faces_sorted[pair]], fn[faces_sorted[pair + 1]])
        sharp[fe_sorted[pair]] = cos < np.cos(np.radians(crease_angle))
    for _ in range(max(0, int(levels))):
        verts, faces, sharp = _subdivide_once(verts, faces, sharp)
    return Mesh(vertices=verts, faces=faces.astype(np.int64), material=mesh.material)


# --- noise ---------------------------------------------------------------------------------------

_GRADIENTS = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1], [1, 0, -1],
                       [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], dtype=np.float64)


def gradient_noise3(points: NDArray, seed: int = 0) -> NDArray:
    """3D Perlin gradient noise at ``points`` (N, 3); roughly in [-1, 1], 0 at lattice points."""
    perm = np.random.default_rng(seed).permutation(256)
    perm = np.concatenate([perm, perm])
    p = np.asarray(points, dtype=np.float64)
    cell = np.floor(p).astype(np.int64)
    f = p - cell
    i, j, k = (cell[:, 0] & 255), (cell[:, 1] & 255), (cell[:, 2] & 255)

    def grad(di, dj, dk):
        h = perm[perm[perm[i + di] + j + dj] + k + dk] % 12
        return np.einsum("ij,ij->i", _GRADIENTS[h], f - np.array([di, dj, dk]))

    u = f * f * f * (f * (f * 6 - 15) + 10)
    lerp = lambda a, b, t: a + t * (b - a)  # noqa: E731
    x00 = lerp(grad(0, 0, 0), grad(1, 0, 0), u[:, 0])
    x10 = lerp(grad(0, 1, 0), grad(1, 1, 0), u[:, 0])
    x01 = lerp(grad(0, 0, 1), grad(1, 0, 1), u[:, 0])
    x11 = lerp(grad(0, 1, 1), grad(1, 1, 1), u[:, 0])
    return lerp(lerp(x00, x10, u[:, 1]), lerp(x01, x11, u[:, 1]), u[:, 2])


def fractal_noise3(points: NDArray, octaves: int = 3, seed: int = 0, lacunarity: float = 2.0,
                   gain: float = 0.5) -> NDArray:
    """Summed octaves of ``gradient_noise3``, normalised to roughly [-1, 1]."""
    total = np.zeros(len(points))
    amp, freq, norm = 1.0, 1.0, 0.0
    # Offset each octave so lattice zeros don't line up.
    for o in range(max(1, int(octaves))):
        total += amp * gradient_noise3(np.asarray(points) * freq + 17.31 * o, seed + o)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm * 1.6   # single-octave Perlin rarely exceeds ~0.6


def displace(mesh: Mesh, amplitude: float, scale: float = 0.3, octaves: int = 3, seed: int = 0,
             ridged: bool = False) -> Mesh:
    """Push vertices along their smoothed normals by fractal noise.

    ``amplitude`` is the peak offset in metres, ``scale`` the feature size in
    metres. ``ridged`` folds the noise (``1 - 2|n|``) for chipped, rocky
    ridges instead of soft lumps.
    """
    verts, faces = _weld_positions(mesh)
    fn, area = face_normals(verts, faces)
    normals = np.zeros_like(verts)
    for k in range(3):
        np.add.at(normals, faces[:, k], fn * area[:, None])
    length = np.linalg.norm(normals, axis=1)
    normals[length > 1e-20] /= length[length > 1e-20, None]
    noise = fractal_noise3(verts / max(float(scale), 1e-6), octaves, int(seed))
    if ridged:
        noise = 1.0 - 2.0 * np.abs(np.clip(noise, -1, 1))
    return Mesh(vertices=verts + normals * (float(amplitude) * noise)[:, None], faces=faces.astype(np.int64),
                material=mesh.material)
