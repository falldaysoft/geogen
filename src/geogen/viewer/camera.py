"""Orbit camera maths for the interactive viewer (pure numpy, no GL)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Named presets: (yaw degrees around +Y from +Z, pitch degrees above horizon)
PRESETS: dict[str, tuple[float, float]] = {
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "right": (90.0, 0.0),
    "left": (-90.0, 0.0),
    "top": (0.0, 89.9),
    "iso": (35.0, 25.0),
}


@dataclass
class OrbitCamera:
    """Camera orbiting a target point, with distance-relative pan and zoom."""

    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 35.0
    pitch: float = 25.0
    distance: float = 5.0
    fov: float = 45.0
    aspect: float = 1.0
    scene_radius: float = 1.0

    def eye(self) -> np.ndarray:
        yaw, pitch = np.radians(self.yaw), np.radians(self.pitch)
        direction = np.array([
            np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            np.cos(yaw) * np.cos(pitch),
        ])
        return self.target + direction * self.distance

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (right, up, forward) unit vectors in world space."""
        forward = self.target - self.eye()
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:  # looking straight down/up
            yaw = np.radians(self.yaw)
            right = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return right, up, forward

    def view_matrix(self) -> np.ndarray:
        right, up, forward = self.basis()
        eye = self.eye()
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = right, up, -forward
        view[:3, 3] = -view[:3, :3] @ eye
        return view

    def clip_planes(self) -> tuple[float, float]:
        """Near/far planes that keep the whole scene visible at any distance."""
        far = self.distance + self.scene_radius * 4 + 10.0
        near = max(far / 20000.0, min(0.01 * self.distance, 0.1))
        return near, far

    def projection_matrix(self) -> np.ndarray:
        near, far = self.clip_planes()
        f = 1.0 / np.tan(np.radians(self.fov) / 2)
        return np.array([
            [f / self.aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ])

    # --- interaction -----------------------------------------------------

    def orbit(self, dx: float, dy: float) -> None:
        """Rotate by screen-space pixel deltas."""
        self.yaw -= dx * 0.4
        self.pitch = float(np.clip(self.pitch + dy * 0.4, -89.9, 89.9))

    def pan(self, dx: float, dy: float, viewport_height: float) -> None:
        """Move the target in the view plane so the scene tracks the cursor."""
        right, up, _ = self.basis()
        world_per_pixel = 2 * self.distance * np.tan(np.radians(self.fov) / 2) / max(viewport_height, 1)
        self.target = self.target - right * dx * world_per_pixel + up * dy * world_per_pixel

    def zoom(self, steps: float, toward: np.ndarray | None = None) -> None:
        """Dolly by ``steps`` wheel notches; optionally move toward a world point."""
        factor = 0.85 ** steps
        new_distance = float(np.clip(self.distance * factor, 1e-3, self.scene_radius * 50 + 100))
        if toward is not None:
            # Keep the point under the cursor fixed while zooming.
            self.target = toward + (self.target - toward) * (new_distance / self.distance)
        self.distance = new_distance

    def frame(self, bounds: np.ndarray, keep_angles: bool = True) -> None:
        """Centre on ``bounds`` ([[min], [max]]) and fit it to the viewport."""
        bounds = np.asarray(bounds, dtype=np.float64)
        self.target = bounds.mean(axis=0)
        radius = max(float(np.linalg.norm(bounds[1] - bounds[0])) / 2, 1e-3)
        half_fov = np.radians(self.fov) / 2
        fit = min(np.tan(half_fov), np.tan(half_fov) * self.aspect)
        self.distance = radius / np.sin(np.arctan(fit)) * 1.05
        if not keep_angles:
            self.yaw, self.pitch = PRESETS["iso"]

    def set_preset(self, name: str) -> None:
        self.yaw, self.pitch = PRESETS[name]

    def ray(self, x: float, y: float, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
        """World-space ray (origin, direction) through pixel (x, y) from the top-left."""
        ndc_x = 2 * x / width - 1
        ndc_y = 1 - 2 * y / height
        tan_half = np.tan(np.radians(self.fov) / 2)
        right, up, forward = self.basis()
        direction = forward + right * ndc_x * tan_half * self.aspect + up * ndc_y * tan_half
        return self.eye(), direction / np.linalg.norm(direction)


def ray_mesh_intersect(
    origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> float | None:
    """Distance along the ray to the nearest triangle hit (Möller-Trumbore), or None."""
    if len(faces) == 0:
        return None
    v0 = vertices[faces[:, 0]]
    e1 = vertices[faces[:, 1]] - v0
    e2 = vertices[faces[:, 2]] - v0
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    t_vec = origin - v0
    u = np.einsum("ij,ij->i", t_vec, p) * inv
    q = np.cross(t_vec, e1)
    v = (q @ direction) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
    if not hit.any():
        return None
    return float(t[hit].min())
