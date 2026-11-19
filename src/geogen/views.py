"""Camera views, framing and the sun shared by the offscreen renderer (``render.py``) and the
interactive viewer (``viewer/``), so ``-r`` and ``--viewer-screenshot`` show a scene from the
same side, fitted the same way and lit from the same direction. Pure numpy, no GL."""

from __future__ import annotations

import numpy as np

# View name -> (azimuth degrees around +Y from +Z, elevation degrees above the horizon)
VIEWS: dict[str, tuple[float, float]] = {
    "iso": (35.0, 25.0),
    "front": (0.0, 5.0),
    "front_high": (0.0, 30.0),
    "side": (90.0, 5.0),
    "right": (90.0, 5.0),
    "left": (-90.0, 5.0),
    "back": (180.0, 5.0),
    "top": (0.0, 89.0),
    "iso_back": (215.0, 25.0),
}

FIT_MARGIN = 1.02                       # bounding sphere to viewport, times this
DEFAULT_COLOR = (0.7, 0.7, 0.75, 1.0)   # meshes without a material
SUN_DIRECTION = (-0.45, -1.0, -0.6)     # the way sunlight travels: from above, in front, to the right
GROUND_GAP = 1e-3                       # the ground plane sits this far under the scene


def view_direction(azimuth: float, elevation: float) -> np.ndarray:
    """Unit vector from the target toward the camera for angles in degrees (see ``VIEWS``)."""
    az, el = np.radians(azimuth), np.radians(elevation)
    return np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])


def fit_distance(radius: float, fov: float, aspect: float, margin: float = FIT_MARGIN) -> float:
    """Camera distance at which a sphere of ``radius`` fills a ``fov`` (degrees, vertical) view."""
    half_fov = np.radians(fov) / 2
    fit = min(np.tan(half_fov), np.tan(half_fov) * aspect)
    return float(radius / np.sin(np.arctan(fit)) * margin)


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Camera-to-world pose looking from ``eye`` at ``target`` (camera looks down -Z).
    Its inverse is the view matrix."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.999:
        up = np.array([0.0, 0.0, -1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def ground_quad(bounds: np.ndarray, half: float) -> np.ndarray:
    """Corners (4x3, counter-clockwise from above) of a square ground plane ``2 * half`` wide
    centred under ``bounds`` (``[[min], [max]]``), ``GROUND_GAP`` below its lowest point."""
    c = np.asarray(bounds, dtype=np.float64).mean(axis=0)
    y = float(bounds[0][1]) - GROUND_GAP
    return np.array([[c[0] - half, y, c[2] - half], [c[0] + half, y, c[2] - half],
                     [c[0] + half, y, c[2] + half], [c[0] - half, y, c[2] + half]])
