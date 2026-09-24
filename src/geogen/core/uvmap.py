"""UV projection helpers producing metric texture coordinates.

Geogen UVs are in metres: a UV distance of 1.0 equals one metre of surface.
Materials declare how many metres one texture repeat covers (``tile_size``),
so texel density is identical on every object regardless of its size.

Projections run in the mesh's local frame, so moving a part never makes its
texture swim; rotating or scaling a part transforms its texture with it.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .meshops import face_normals

# For each dominant axis, the (u, v) axes of the projection plane. Chosen so
# vertical faces have v = +Y and the texture reads left-to-right from outside.
_BOX_AXES = {
    (0, 1): (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),   # +X
    (0, -1): (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),   # -X
    (1, 1): (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),   # +Y
    (1, -1): (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),   # -Y
    (2, 1): (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),    # +Z
    (2, -1): (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # -Z
}


def _split_corners(mesh: Mesh, corner_uv: np.ndarray) -> Mesh:
    """Rebuild ``mesh`` so each vertex has one UV, splitting where corners disagree."""
    corner_vert = mesh.faces.reshape(-1)
    key = np.hstack([corner_vert[:, None], np.round(corner_uv / 1e-6).astype(np.int64)])
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    src = corner_vert[first]
    return Mesh(
        vertices=mesh.vertices[src],
        faces=inverse.reshape(-1, 3),
        normals=mesh.normals[src] if mesh.normals is not None else None,
        uvs=corner_uv[first],
        material=mesh.material,
    )


def box_project(mesh: Mesh, origin: np.ndarray | None = None, directions: np.ndarray | None = None) -> Mesh:
    """Assign metric UVs by projecting each face onto its dominant axis plane.

    Faces are grouped by the sign and axis of their largest normal
    component (like a cube map); vertices shared between groups are split.
    ``origin`` shifts the projection so textures start at a chosen corner.
    ``directions`` (one vector per face) replaces the face normals for
    grouping, e.g. the normals of a smooth base shape so a noisy surface gets
    clean seams instead of speckled islands.
    """
    if len(mesh.faces) == 0:
        return mesh.copy()
    fn = np.asarray(directions, dtype=np.float64) if directions is not None else \
        face_normals(mesh.vertices, mesh.faces)[0]
    axis = np.argmax(np.abs(fn), axis=1)
    sign = np.where(fn[np.arange(len(fn)), axis] >= 0, 1, -1)
    points = mesh.vertices[mesh.faces] - (origin if origin is not None else 0.0)  # (F, 3, 3)

    corner_uv = np.zeros((len(mesh.faces), 3, 2))
    for (ax, sg), (u_axis, v_axis) in _BOX_AXES.items():
        sel = (axis == ax) & (sign == sg)
        if sel.any():
            corner_uv[sel, :, 0] = points[sel] @ u_axis
            corner_uv[sel, :, 1] = points[sel] @ v_axis
    return _split_corners(mesh, corner_uv.reshape(-1, 2))


def planar_project(mesh: Mesh, u_axis=(1.0, 0.0, 0.0), v_axis=(0.0, 0.0, 1.0)) -> Mesh:
    """Assign metric UVs by projecting every vertex onto one plane."""
    uv = np.column_stack([mesh.vertices @ np.asarray(u_axis), mesh.vertices @ np.asarray(v_axis)])
    out = mesh.copy()
    out.uvs = uv
    return out


def cylindrical_project(mesh: Mesh, radius: float | None = None) -> Mesh:
    """Assign metric UVs by wrapping around the Y axis.

    ``u`` is arc length at ``radius`` (default: mean radial distance) and
    ``v`` is height. Faces straddling the wrap seam get split vertices.
    """
    r_xz = np.linalg.norm(mesh.vertices[:, [0, 2]], axis=1)
    radius = float(radius if radius is not None else max(r_xz.mean(), 1e-6))
    angle = np.arctan2(mesh.vertices[:, 0], mesh.vertices[:, 2])  # 0 at +Z
    corner_angle = angle[mesh.faces]
    # Unwrap per face so no triangle spans the -pi/pi seam.
    ref = corner_angle[:, :1]
    corner_angle = ref + (corner_angle - ref + np.pi) % (2 * np.pi) - np.pi
    corner_uv = np.stack([corner_angle * radius, mesh.vertices[mesh.faces][..., 1]], axis=-1)
    return _split_corners(mesh, corner_uv.reshape(-1, 2))


def texel_density(mesh: Mesh) -> float:
    """Median ratio of UV area to surface area (1.0 for metric UVs)."""
    if mesh.uvs is None or len(mesh.faces) == 0:
        return 0.0
    _, area = face_normals(mesh.vertices, mesh.faces)
    uv = mesh.uvs[mesh.faces]
    d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    uv_area = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) * 0.5
    ok = area > 1e-12
    if not ok.any():
        return 0.0
    return float(np.median(uv_area[ok] / area[ok]))


def face_planar_project(mesh: Mesh, origin: np.ndarray | None = None) -> Mesh:
    """Assign metric UVs in each face's own plane: u horizontal, v up the slope.

    Ideal for roofs and ramps: shingle or plank rows stay parallel to the
    eaves with no stretching on steep pitches (box projection would stretch
    by 1/cos(pitch)). Horizontal faces fall back to X/Z.
    """
    if len(mesh.faces) == 0:
        return mesh.copy()
    fn, _ = face_normals(mesh.vertices, mesh.faces)
    up = np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(up, fn)
    length = np.linalg.norm(u_axis, axis=1, keepdims=True)
    flat = length[:, 0] < 1e-6
    u_axis = np.where(flat[:, None], [[1.0, 0.0, 0.0]], u_axis / np.maximum(length, 1e-12))
    v_axis = np.cross(fn, u_axis)
    v_axis[flat] = np.where(fn[flat, 1:2] >= 0, [[0.0, 0.0, -1.0]], [[0.0, 0.0, 1.0]])
    points = mesh.vertices[mesh.faces] - (origin if origin is not None else 0.0)
    corner_uv = np.stack([
        np.einsum("fcj,fj->fc", points, u_axis),
        np.einsum("fcj,fj->fc", points, v_axis),
    ], axis=-1)
    return _split_corners(mesh, corner_uv.reshape(-1, 2))
