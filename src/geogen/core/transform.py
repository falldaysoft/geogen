"""Transform class for 3D transformations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray


@dataclass
class Transform:
    """Represents a 3D transformation with translation, rotation, and scale.

    Rotation is stored as Euler angles (XYZ order) in radians.
    """

    translation: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    rotation: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    scale: NDArray[np.float64] = field(
        default_factory=lambda: np.ones(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.translation = np.asarray(self.translation, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.scale = np.asarray(self.scale, dtype=np.float64)

    def to_matrix(self) -> NDArray[np.float64]:
        """Convert to a 4x4 transformation matrix.

        Order: Scale -> Rotate -> Translate (standard game engine order)
        """
        # R = Rz @ Ry @ Rx (XYZ Euler), written out: this runs for every node of every posed frame.
        rx, ry, rz = self.rotation.tolist()
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        kx, ky, kz = self.scale.tolist()
        tx, ty, tz = self.translation.tolist()
        # Combine: T * R * S
        return np.array([
            [cz * cy * kx, (cz * sy * sx - sz * cx) * ky, (cz * sy * cx + sz * sx) * kz, tx],
            [sz * cy * kx, (sz * sy * sx + cz * cx) * ky, (sz * sy * cx - cz * sx) * kz, ty],
            [-sy * kx, cy * sx * ky, cy * cx * kz, tz],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @classmethod
    def from_matrix(cls, matrix: NDArray[np.float64]) -> Self:
        """Create Transform from a 4x4 transformation matrix.

        Note: Assumes the matrix was created with uniform or axis-aligned scaling.
        """
        (m00, m01, m02, tx), (m10, m11, m12, ty), (m20, m21, m22, tz) = np.asarray(matrix)[:3].tolist()
        # Scale: the length of each basis vector; rotation: the columns normalised.
        kx = math.sqrt(m00 * m00 + m10 * m10 + m20 * m20)
        ky = math.sqrt(m01 * m01 + m11 * m11 + m21 * m21)
        kz = math.sqrt(m02 * m02 + m12 * m12 + m22 * m22)
        ix, iy, iz = 1.0 / (kx or 1.0), 1.0 / (ky or 1.0), 1.0 / (kz or 1.0)
        return cls(
            translation=np.array([tx, ty, tz]),
            rotation=euler_xyz([[m00 * ix, m01 * iy, m02 * iz], [m10 * ix, m11 * iy, m12 * iz],
                                [m20 * ix, m21 * iy, m22 * iz]]),
            scale=np.array([kx, ky, kz]),
        )

    def copy(self) -> Self:
        """Create a deep copy of this transform."""
        return Transform(
            translation=self.translation.copy(),
            rotation=self.rotation.copy(),
            scale=self.scale.copy(),
        )

    @staticmethod
    def identity() -> Transform:
        """Create an identity transform."""
        return Transform()

    def __matmul__(self, other: Transform) -> Transform:
        """Combine two transforms via matrix multiplication."""
        combined = self.to_matrix() @ other.to_matrix()
        return Transform.from_matrix(combined)


def euler_xyz(rot) -> NDArray[np.float64]:
    """XYZ Euler angles (radians, as ``Transform.rotation``) of a rotation matrix (3x3, no scale)."""
    (r00, r01, r02), (r10, _, _), (r20, r21, r22) = np.asarray(rot, dtype=np.float64).tolist()
    if abs(r20) < 0.9999:
        ry = math.asin(-r20)
        rx = math.atan2(r21, r22)
        rz = math.atan2(r10, r00)
    else:
        # Gimbal lock
        rz = 0.0
        if r20 < 0:
            ry = math.pi / 2
            rx = math.atan2(r01, r02)
        else:
            ry = -math.pi / 2
            rx = math.atan2(-r01, -r02)
    return np.array([rx, ry, rz])


def quat_from_matrix(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation matrix (3x3, no scale) -> unit quaternion (x, y, z, w), glTF order."""
    r = np.asarray(r, dtype=np.float64).tolist()        # plain floats: this runs ~10k times a body
    t = r[0][0] + r[1][1] + r[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s, 0.25 * s]
    else:
        diag = (r[0][0], r[1][1], r[2][2])
        i = diag.index(max(diag))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + r[i][i] - r[j][j] - r[k][k]) * 2
        q = [0.0, 0.0, 0.0, (r[k][j] - r[j][k]) / s]
        q[i] = 0.25 * s
        q[j] = (r[j][i] + r[i][j]) / s
        q[k] = (r[k][i] + r[i][k]) / s
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return np.array([q[0] / n, q[1] / n, q[2] / n, q[3] / n])


def matrix_from_quat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64).tolist()
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quats_from_matrices(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """``quat_from_matrix`` over a stack of rotation matrices (..., 3, 3) -> (..., 4)."""
    r = np.asarray(r, dtype=np.float64)
    shape = r.shape[:-2]
    r = r.reshape(-1, 3, 3)
    q = np.empty((len(r), 4))
    t = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    pos = t > 0
    if pos.any():
        m = r[pos]
        s = np.sqrt(t[pos] + 1.0) * 2
        q[pos] = np.stack([(m[:, 2, 1] - m[:, 1, 2]) / s, (m[:, 0, 2] - m[:, 2, 0]) / s,
                           (m[:, 1, 0] - m[:, 0, 1]) / s, 0.25 * s], axis=1)
    big = np.argmax(np.diagonal(r, axis1=1, axis2=2), axis=1)
    for i in range(3):
        rows = ~pos & (big == i)
        if not rows.any():
            continue
        m = r[rows]
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + m[:, i, i] - m[:, j, j] - m[:, k, k]) * 2
        q[rows, 3] = (m[:, k, j] - m[:, j, k]) / s
        q[rows, i] = 0.25 * s
        q[rows, j] = (m[:, j, i] + m[:, i, j]) / s
        q[rows, k] = (m[:, k, i] + m[:, i, k]) / s
    q /= np.sqrt(np.einsum("ij,ij->i", q, q))[:, None]
    return q.reshape(*shape, 4)


def matrices_from_quats(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """``matrix_from_quat`` over a stack of quaternions (..., 4) -> (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.sqrt(np.einsum("...i,...i->...", q, q))[..., None]
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def quat_axis_angle(axis, angle: float) -> NDArray[np.float64]:
    """Quaternion (x, y, z, w) rotating ``angle`` radians about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.array([*(axis * np.sin(angle / 2)), np.cos(angle / 2)])


def slerp(a: NDArray[np.float64], b: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    """Spherical interpolation between unit quaternions (shortest arc), as glTF LINEAR rotation does."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + (b - a) * t
        return q / math.sqrt(float(q @ q))
    theta = math.acos(d)
    return (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)
