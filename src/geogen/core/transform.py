"""Transform class for 3D transformations."""

from __future__ import annotations

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
        # Scale matrix
        s = np.diag([*self.scale, 1.0])

        # Rotation matrices (XYZ Euler order)
        rx, ry, rz = self.rotation

        cos_x, sin_x = np.cos(rx), np.sin(rx)
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        cos_z, sin_z = np.cos(rz), np.sin(rz)

        rot_x = np.array([
            [1, 0, 0, 0],
            [0, cos_x, -sin_x, 0],
            [0, sin_x, cos_x, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        rot_y = np.array([
            [cos_y, 0, sin_y, 0],
            [0, 1, 0, 0],
            [-sin_y, 0, cos_y, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        rot_z = np.array([
            [cos_z, -sin_z, 0, 0],
            [sin_z, cos_z, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Combined rotation (Z * Y * X for XYZ Euler)
        r = rot_z @ rot_y @ rot_x

        # Translation matrix
        t = np.eye(4, dtype=np.float64)
        t[:3, 3] = self.translation

        # Combine: T * R * S
        return t @ r @ s

    @classmethod
    def from_matrix(cls, matrix: NDArray[np.float64]) -> Self:
        """Create Transform from a 4x4 transformation matrix.

        Note: Assumes the matrix was created with uniform or axis-aligned scaling.
        """
        # Extract translation
        translation = matrix[:3, 3].copy()

        # Extract scale (length of each basis vector)
        scale = np.array([
            np.linalg.norm(matrix[:3, 0]),
            np.linalg.norm(matrix[:3, 1]),
            np.linalg.norm(matrix[:3, 2]),
        ], dtype=np.float64)

        # Extract rotation matrix (normalize columns)
        rot = matrix[:3, :3].copy()
        rot[:, 0] /= scale[0] if scale[0] != 0 else 1
        rot[:, 1] /= scale[1] if scale[1] != 0 else 1
        rot[:, 2] /= scale[2] if scale[2] != 0 else 1

        # Extract Euler angles (XYZ order)
        # Using standard decomposition
        if abs(rot[2, 0]) < 0.9999:
            ry = np.arcsin(-rot[2, 0])
            rx = np.arctan2(rot[2, 1], rot[2, 2])
            rz = np.arctan2(rot[1, 0], rot[0, 0])
        else:
            # Gimbal lock
            rz = 0.0
            if rot[2, 0] < 0:
                ry = np.pi / 2
                rx = np.arctan2(rot[0, 1], rot[0, 2])
            else:
                ry = -np.pi / 2
                rx = np.arctan2(-rot[0, 1], -rot[0, 2])

        return cls(
            translation=translation,
            rotation=np.array([rx, ry, rz], dtype=np.float64),
            scale=scale,
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


def quat_from_matrix(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation matrix (3x3, no scale) -> unit quaternion (x, y, z, w), glTF order."""
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def matrix_from_quat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_axis_angle(axis, angle: float) -> NDArray[np.float64]:
    """Quaternion (x, y, z, w) rotating ``angle`` radians about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.array([*(axis * np.sin(angle / 2)), np.cos(angle / 2)])


def slerp(a: NDArray[np.float64], b: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    """Spherical interpolation between unit quaternions (shortest arc), as glTF LINEAR rotation does."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + (b - a) * t
        return q / np.linalg.norm(q)
    theta = np.arccos(d)
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)
