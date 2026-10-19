"""Sculpted humanoid heads: a signed-distance head, polygonised and simplified.

The head is a smooth union of simple shapes (cranium, jaw, chin, cheekbones, brow,
nose, lips, ears, a neck stub) minus soft eye sockets, all in the head's frame:
x across, y up from the eye line, z forward, sizes scaled by the head size. Surface
nets on a 4 mm grid (evaluated only near the surface) make a closed mesh, which
manifold3d's simplifier brings down to the head's triangle budget, keeping detail
where the curvature is (nose, lips, ears) and little over the smooth cranium.

Feature positions follow the painted face (``textures/character.FaceTextureGenerator``):
eyes at ``eye_v`` and ``eye_spacing`` of the face box, the mouth at ``mouth_v``, so
the paint and the sockets, nose and lips agree.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.mesh import Mesh

# Feature shape params (1 = the neutral face) and their defaults.
SHAPE_DEFAULTS = {"jaw": 1.0, "chin": 1.0, "nose": 1.0, "brow": 1.0, "cheeks": 1.0, "lips": 1.0, "ears": 1.0}


def _ellipsoid(p: NDArray, center, radii) -> NDArray:
    """Approximate signed distance to an axis-aligned ellipsoid (negative inside)."""
    q = (p - np.asarray(center)) / np.asarray(radii)
    k0 = np.sqrt((q * q).sum(axis=-1))
    q = q / np.asarray(radii)
    k1 = np.sqrt((q * q).sum(axis=-1))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def _capsule(p: NDArray, a, b, r0: float, r1: float) -> NDArray:
    """Signed distance to a capsule from ``a`` (radius ``r0``) to ``b`` (radius ``r1``), linearly tapered."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ab = b - a
    t = np.clip(((p - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1) - (r0 + (r1 - r0) * t)


def _smin(a: NDArray, b: NDArray, k: float) -> NDArray:
    """Polynomial smooth minimum: a union blended over ``k`` metres."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def _smax(a: NDArray, b: NDArray, k: float) -> NDArray:
    """Smooth maximum: ``a`` minus the solid ``-b``, blended over ``k``."""
    return -_smin(-a, -b, k)


class HeadShape:
    """The signed-distance head for one set of params, in the head frame (metres, unscaled)."""

    def __init__(self, size: float = 1.0, eye_spacing: float = 0.072, mouth_drop: float = 0.084,
                 neck_radius: float = 0.048, neck_drop: float = 0.16, **shape: float) -> None:
        self.s = float(size)
        self.eye_x = eye_spacing / 2 / self.s       # in unscaled head units
        self.mouth_y = -mouth_drop / self.s
        self.neck_radius = neck_radius / self.s
        self.neck_drop = neck_drop / self.s
        unknown = set(shape) - set(SHAPE_DEFAULTS)
        if unknown:
            raise ValueError(f"head: unknown shape params {sorted(unknown)} (known: {sorted(SHAPE_DEFAULTS)})")
        self.p = {**SHAPE_DEFAULTS, **{k: float(v) for k, v in shape.items()}}

    def __call__(self, points: NDArray) -> NDArray:
        points = np.asarray(points, dtype=np.float64)
        p = points.reshape(-1, 3) / self.s
        f = self.p
        ex, my = self.eye_x, self.mouth_y
        jaw = f["jaw"]

        def add(center, radii, k, shape=None, cut=False):
            """Blend an ellipsoid (or ``shape``, bounded by ``radii`` round ``center``) in over ``k``,
            or carve it out with ``cut``: evaluated only inside its bounds, where it can matter."""
            center, radii = np.asarray(center, dtype=np.float64), np.asarray(radii, dtype=np.float64)
            near = np.all(np.abs(p - center) < radii + k, axis=1)
            if not near.any():
                return
            q = p[near]
            other = shape(q) if shape is not None else _ellipsoid(q, center, radii)
            d[near] = _smax(d[near], -other, k) if cut else _smin(d[near], other, k)

        # Cranium and the face's lower mass: the jaw narrows to the chin and tucks in behind.
        d = _ellipsoid(p, [0, 0.026, -0.006], [0.084, 0.103, 0.101])
        add([0, my + 0.008, 0.012], [0.07 * jaw, 0.064, 0.082], 0.045)
        add([0, my - 0.032, 0.05 + 0.006 * f["chin"]], [0.026 * (0.8 + 0.2 * jaw), 0.022, 0.03 + 0.006 * f["chin"]],
            0.025)
        # Cheekbones, brow ridge.
        for side in (-1, 1):
            add([side * 0.044, -0.026, 0.044], [0.032, 0.026, 0.034 * f["cheeks"]], 0.045)
        add([0, 0.026, 0.062 + 0.004 * f["brow"]], [0.064, 0.018, 0.03], 0.04)
        # Soft eye sockets under the brow, where the painted eyes sit.
        for side in (-1, 1):
            add([side * ex, 0.004, 0.104], [0.024, 0.016, 0.012], 0.016, cut=True)
        # Nose: a bridge widening to a rounded tip, with the wings either side.
        n = f["nose"]
        tip = np.array([0, -0.04 * (0.85 + 0.15 * n), 0.098 + 0.006 * n])

        def nose(q):
            d = _capsule(q, [0, 0.008, 0.082], tip, 0.0075, 0.009 * (0.8 + 0.2 * n))
            d = _smin(d, _ellipsoid(q, tip, [0.012 * n, 0.0105, 0.0105]), 0.008)
            for side in (-1, 1):
                d = _smin(d, _ellipsoid(q, [side * 0.0115 * n, tip[1] - 0.001, tip[2] - 0.013],
                                        [0.008, 0.007, 0.007]), 0.007)
            return d

        add([0, (0.01 + tip[1]) / 2, 0.096], [0.026 * n, (0.01 - tip[1]) / 2 + 0.016, 0.03], 0.014, nose)
        # Lips, upper and lower (the painted mouth draws the line between them).
        lips = f["lips"]
        add([0, my + 0.006, 0.086], [0.022, 0.007, 0.01 * lips], 0.01)
        add([0, my - 0.008, 0.083], [0.019, 0.008, 0.01 * lips], 0.01)
        # Ears: flattened shells hugging the sides, tipped back a little.
        e = f["ears"]
        for side in (-1, 1):
            center = np.array([side * 0.08, -0.012, -0.014])

            def ear(q, center=center):
                q = q - center
                c, s = np.cos(0.25), np.sin(0.25)
                q = np.stack([q[:, 0], c * q[:, 1] - s * q[:, 2], s * q[:, 1] + c * q[:, 2]], axis=-1)
                return _ellipsoid(q, [0, 0, 0], [0.013 * e, 0.03 * e, 0.018 * e])

            add(center, [0.015 * e, 0.034 * e, 0.034 * e], 0.008, ear)
        # The neck, blending into the underside of the jaw.
        d = _smin(d, _capsule(p, [0, -self.neck_drop, -0.012], [0, my - 0.01, -0.016],
                              self.neck_radius, self.neck_radius * 1.05), 0.03)
        return (d * self.s).reshape(points.shape[:-1])


_AXES = ((0, 1, 2), (1, 2, 0), (2, 0, 1))      # (edge axis, then the two axes of its quad, cyclic)


def surface_nets(sdf, lo, hi, step: float) -> Mesh:
    """Polygonise the solid ``sdf`` < 0 inside the box ``lo``..``hi`` (naive surface nets).

    One vertex per grid cell the surface crosses, at the mean of its edge crossings; one quad
    per crossed grid edge. The solid is clipped to the box
    so the mesh is closed. ``sdf`` is evaluated only near the surface: a grid twice as coarse
    finds the band, and points well outside it keep the coarse value's sign."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    counts = np.ceil((hi - lo) / step).astype(int) + 1
    axes = [lo[a] + step * np.arange(counts[a]) for a in range(3)]
    margin = 1.5 * step

    def clipped(points: NDArray) -> NDArray:
        box = np.max(np.maximum(lo + margin - points, points - (hi - margin)), axis=-1)
        return np.maximum(sdf(points), box)

    coarse_axes = [ax[::2] for ax in axes]
    coarse = clipped(np.stack(np.meshgrid(*coarse_axes, indexing="ij"), axis=-1))
    values = np.repeat(np.repeat(np.repeat(coarse, 2, 0), 2, 1), 2, 2)[:counts[0], :counts[1], :counts[2]]
    band = np.abs(values) < 2.0 * step          # a coarse node is at most sqrt(3) fine steps away
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    values[band] = clipped(grid[band])
    inside = values < 0

    cell_shape = tuple(counts - 1)
    flat_count = int(np.prod(cell_shape))
    sums = np.zeros((flat_count, 3))
    hits = np.zeros(flat_count)
    crossings = []
    for a, b, c in _AXES:
        n = [slice(None)] * 3
        n0, n1 = list(n), list(n)
        n0[a], n1[a] = slice(0, -1), slice(1, None)
        f0, f1 = values[tuple(n0)], values[tuple(n1)]
        edge = np.argwhere(inside[tuple(n0)] != inside[tuple(n1)])
        t = (f0[tuple(edge.T)] / (f0[tuple(edge.T)] - f1[tuple(edge.T)]))[:, None]
        point = grid[tuple(edge.T)] + t * step * np.eye(3)[a]
        for db in (-1, 0):
            for dc in (-1, 0):
                cell = edge.copy()
                cell[:, b] += db
                cell[:, c] += dc
                ok = np.all((cell >= 0) & (cell < np.array(cell_shape)), axis=1)
                flat = np.ravel_multi_index(tuple(cell[ok].T), cell_shape)
                for axis in range(3):
                    sums[:, axis] += np.bincount(flat, point[ok][:, axis], minlength=flat_count)
                hits += np.bincount(flat, minlength=flat_count)
        crossings.append((a, b, c, edge, inside[tuple(n0)][tuple(edge.T)]))
    sums, hits = sums.reshape(*cell_shape, 3), hits.reshape(cell_shape)
    active = hits > 0
    index = np.full(cell_shape, -1, dtype=np.int64)
    index[active] = np.arange(int(active.sum()))
    vertices = sums[active] / hits[active][:, None]

    quads = []
    for a, b, c, edge, low_inside in crossings:
        corners = []
        for db, dc in ((-1, -1), (0, -1), (0, 0), (-1, 0)):     # normal e_b x e_c = +e_a
            cell = edge.copy()
            cell[:, b] += db
            cell[:, c] += dc
            corners.append(cell)
        ok = np.all([np.all((cell >= 0) & (cell < np.array(cell_shape)), axis=1) for cell in corners], axis=0)
        quad = np.stack([index[tuple(cell[ok].T)] for cell in corners], axis=1)
        # Outward is from inside to outside along the edge: +e_a when its low end is inside.
        quads.append(np.where(low_inside[ok][:, None], quad, quad[:, ::-1]))
    quads = np.concatenate(quads)

    # Split each quad along its shorter diagonal.
    q = vertices[quads]
    short = np.linalg.norm(q[:, 0] - q[:, 2], axis=1) <= np.linalg.norm(q[:, 1] - q[:, 3], axis=1)
    faces = np.where(short[:, None, None],
                     np.stack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=1),
                     np.stack([quads[:, [0, 1, 3]], quads[:, [1, 2, 3]]], axis=1)).reshape(-1, 3)
    return Mesh(vertices, faces)


def build_head(shape: HeadShape, origin, step: float = 0.004, tolerance: float = 0.002) -> Mesh:
    """A closed head mesh from ``shape``, its head frame at ``origin`` (body frame, rest pose),
    simplified to within ``tolerance`` metres of the sculpt (both scaled by the head size).

    manifold3d's import merges the odd pinched vertex surface nets leave, so the result is
    always a closed manifold."""
    from ..core import csg

    s = shape.s
    lo = np.array([-0.115, -shape.neck_drop - 0.01, -0.13]) * s
    hi = np.array([0.115, 0.145, 0.15]) * s
    net = surface_nets(shape, lo, hi, step * s)
    simple = csg._to_manifold(net).simplify(tolerance * s).to_mesh()
    # Weld: manifold keeps a vertex split where its properties differ, which relax would tear.
    vertices = np.asarray(simple.vert_properties)[:, :3].astype(np.float64)
    index = np.arange(len(vertices))
    index[np.asarray(simple.merge_from_vert, dtype=np.int64)] = np.asarray(simple.merge_to_vert, dtype=np.int64)
    used, faces = np.unique(index[np.asarray(simple.tri_verts).astype(np.int64)], return_inverse=True)
    faces = faces.reshape(-1, 3)
    vertices = relax(shape, vertices[used], faces)
    return Mesh(vertices + np.asarray(origin, dtype=np.float64), faces)


def _face_normals(v: NDArray, faces: NDArray) -> NDArray:
    n = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)


def relax(sdf, vertices: NDArray, faces: NDArray, iterations: int = 8, rate: float = 0.3) -> NDArray:
    """Even out a simplified mesh's slivers: move each vertex toward its neighbours' mean in
    its tangent plane, then back onto the surface. Topology (and so where the simplifier spent
    its triangles) is kept; the triangles get rounder, which is what smooth shading needs. A
    vertex whose move would turn one of its faces over (or nearly) stays where it was."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    n = len(vertices)
    degree = np.bincount(edges[:, 0], minlength=n).astype(np.float64)
    v = vertices.copy()
    for _ in range(iterations):
        mean = np.stack([np.bincount(edges[:, 0], v[edges[:, 1], a], minlength=n) for a in range(3)], axis=1)
        mean /= np.maximum(degree, 1)[:, None]
        face_normal = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
        corners = faces.reshape(-1)
        normal = np.stack([np.bincount(corners, np.repeat(face_normal[:, a], 3), minlength=n) for a in range(3)],
                          axis=1)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-15)
        move = mean - v
        move -= (move * normal).sum(axis=1, keepdims=True) * normal
        moved = v + rate * move
        moved -= sdf(moved)[:, None] * normal                 # back onto the surface
        for _ in range(4):
            bad = (_face_normals(moved, faces) * _face_normals(v, faces)).sum(axis=1) < 0.5
            if not bad.any():
                break
            stuck = np.unique(faces[bad])
            moved[stuck] = v[stuck]
        v = moved
    for _ in range(2):                                     # exactly onto the surface
        v = v - sdf(v)[:, None] * sdf_normals(sdf, v)
    return v


def sdf_normals(sdf, points: NDArray, eps: float = 0.0008) -> NDArray:
    """Unit outward normals of ``sdf`` at ``points`` (central differences): smooth shading that
    doesn't depend on how the simplified mesh happens to be triangulated."""
    grad = np.stack([sdf(points + eps * np.eye(3)[a]) - sdf(points - eps * np.eye(3)[a]) for a in range(3)], axis=1)
    return grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
