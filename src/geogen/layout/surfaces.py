"""Surfaces: 2D regions on assets where objects can be placed.

Where an `AttachmentPoint` is a single 0-dimensional connector (e.g. "top of
the post"), a `Surface` is a 2D region with its own coordinate system — a
wall, a floor, a plot of ground. You address positions on a surface with
(u, v) where both default to fractional (0-1) across the surface's extent.

Key operations:
- `resolve(u, v, depth=0, container_size=None)` returns a local-space
  Transform whose translation sits on the surface and whose rotation faces
  outward along the surface normal.
- u/v accept either a float (fractional, 0-1) or a dict with "abs" key
  (absolute meters along that axis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.transform import Transform


@dataclass
class Surface:
    """A 2D region on an asset, with its own (u, v) coordinate system.

    Coordinates:
        origin: local-space position of the (u=0, v=0) corner
        u_axis: unit vector pointing along the +u direction
        v_axis: unit vector pointing along the +v direction
        normal: unit vector pointing outward from the surface
        u_extent: size along the u axis (meters)
        v_extent: size along the v axis (meters)

    All vectors are in the *local* space of the node that owns the surface.
    Surfaces are combined with the node's world transform at resolution time.
    """

    name: str
    origin: NDArray[np.float64]
    u_axis: NDArray[np.float64]
    v_axis: NDArray[np.float64]
    normal: NDArray[np.float64]
    u_extent: float
    v_extent: float

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64)
        self.u_axis = _unit(np.asarray(self.u_axis, dtype=np.float64))
        self.v_axis = _unit(np.asarray(self.v_axis, dtype=np.float64))
        self.normal = _unit(np.asarray(self.normal, dtype=np.float64))

    def resolve(
        self,
        u: float | str | dict[str, Any] = 0.5,
        v: float | str | dict[str, Any] = 0.5,
        depth: float | str | dict[str, Any] = 0.0,
    ) -> Transform:
        """Resolve a (u, v, depth) coordinate to a local-space Transform.

        u, v, depth may each be:
            - a float: fractional if between 0 and 1 is ambiguous, so we
              treat all bare floats as FRACTIONAL (0-1) for u and v.
              For depth, bare floats are absolute meters (along the normal).
            - a dict `{"abs": 0.5}`: absolute meters along the axis
            - a dict `{"frac": 0.25}`: explicit fractional

        The returned transform's translation is the world (local-space) point
        on the surface; its rotation is a Y-axis rotation aligning +Z with
        the surface normal (so objects placed face *outward*).
        """
        u_offset = _axis_offset(u, self.u_extent, default_absolute=False)
        v_offset = _axis_offset(v, self.v_extent, default_absolute=False)
        d_offset = _axis_offset(depth, 1.0, default_absolute=True)

        position = (
            self.origin
            + u_offset * self.u_axis
            + v_offset * self.v_axis
            + d_offset * self.normal
        )

        # Rotation around Y that points +Z to the surface normal (flattened to XZ).
        y_rotation = float(np.arctan2(self.normal[0], self.normal[2]))

        return Transform(
            translation=position,
            rotation=np.array([0.0, y_rotation, 0.0], dtype=np.float64),
        )


def _unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a vector; returns zero vector unchanged."""
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def _axis_offset(
    value: float | str | dict[str, Any],
    extent: float,
    default_absolute: bool,
) -> float:
    """Convert a u/v/depth spec into an absolute offset in meters.

    Args:
        value: fractional float, dict with {"abs": x} or {"frac": x}
        extent: the surface's extent along this axis (used for fractional)
        default_absolute: if True, bare floats are absolute meters;
            if False (default for u/v), bare floats are fractional
    """
    if isinstance(value, dict):
        if "abs" in value:
            return float(value["abs"])
        if "frac" in value:
            return float(value["frac"]) * extent
        raise ValueError(
            f"Surface coordinate dict must have 'abs' or 'frac', got {list(value)}"
        )
    # Plain scalar
    v = float(value)
    if default_absolute:
        return v
    return v * extent
