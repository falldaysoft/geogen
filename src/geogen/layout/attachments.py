"""Attachment point system for connecting objects together."""

import difflib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..core.transform import Transform
from .anchors import Anchor, resolve_anchor


FacingDirection = Literal["center", "outward", "north", "south", "east", "west"]

# Yaw (radians about +Y) that turns an object's +Z front to each compass facing.
CARDINAL_YAW = {"north": 0.0, "south": np.pi, "east": np.pi / 2, "west": -np.pi / 2}
FACINGS = ("center", "outward", *CARDINAL_YAW)


def facing_to_yaw(facing: str, position: NDArray[np.float64]) -> float:
    """The Y rotation (radians) for a ``facing:`` value at ``position``.

    ``center`` faces the container's origin and ``outward`` away from it; the rest are compass
    directions. An unknown value raises rather than quietly facing north.
    """
    if facing == "center":
        return float(np.arctan2(-position[0], -position[2]))
    if facing == "outward":
        return float(np.arctan2(position[0], position[2]))
    if facing in CARDINAL_YAW:
        return float(CARDINAL_YAW[facing])
    near = difflib.get_close_matches(str(facing), FACINGS, n=1, cutoff=0.5)
    raise ValueError(f"Unknown facing {facing!r}: expected one of {', '.join(FACINGS)}."
                     + (f" Did you mean '{near[0]}'?" if near else ""))


OffsetMode = Literal["fractional", "absolute"]


@dataclass
class AttachmentPoint:
    """A named point where another object can be attached.

    Attachment points define both position and orientation. The position is
    specified using the anchor system plus an optional offset. The orientation
    can be specified either as an explicit Y rotation angle or semantically
    using the 'facing' direction.

    Attributes:
        name: Unique name for this attachment point
        anchor: Base anchor position within the container
        offset: Additional offset [x, y, z]. Interpretation depends on offset_mode.
        facing: Semantic direction the attached object should face
        rotation: Explicit Y rotation in degrees (overrides facing if set)
        offset_mode: "fractional" multiplies offset by container_size (default),
                     "absolute" uses offset directly in world units
    """

    name: str
    anchor: Anchor | str
    offset: NDArray[np.float64] | None = None
    facing: FacingDirection = "center"
    rotation: float | None = None  # degrees, Y-axis rotation
    offset_mode: OffsetMode = "absolute"

    def __post_init__(self) -> None:
        if self.offset is None:
            self.offset = np.zeros(3, dtype=np.float64)
        else:
            self.offset = np.asarray(self.offset, dtype=np.float64)

    def resolve(self, container_size: NDArray[np.float64]) -> Transform:
        """Resolve this attachment point to a Transform.

        Args:
            container_size: Size of the container [width, height, depth]

        Returns:
            Transform with position and rotation for the attachment
        """
        # Calculate position from anchor + offset
        anchor_pos = resolve_anchor(self.anchor, container_size)
        offset = self.offset if self.offset is not None else np.zeros(3, dtype=np.float64)
        if self.offset_mode == "fractional":
            offset_world = offset * container_size
        else:
            offset_world = offset
        position = anchor_pos + offset_world

        # Calculate rotation
        y_rotation = self._compute_rotation(position)

        return Transform(
            translation=position,
            rotation=np.array([0.0, y_rotation, 0.0], dtype=np.float64),
        )

    def _compute_rotation(self, position: NDArray[np.float64]) -> float:
        """Compute the Y-axis rotation based on facing direction or explicit angle.

        Args:
            position: The world position of the attachment point

        Returns:
            Rotation angle in radians around Y axis
        """
        if self.rotation is not None:
            # Explicit rotation in degrees, convert to radians
            return np.deg2rad(self.rotation)

        return facing_to_yaw(self.facing, position)


def parse_attachment(name: str, data: dict) -> AttachmentPoint:
    """Parse an attachment point from YAML data.

    Args:
        name: Name of the attachment point
        data: Dictionary with anchor, offset, facing, rotation fields

    Returns:
        AttachmentPoint instance
    """
    anchor = data.get("anchor", "center")
    offset = data.get("offset")
    if offset is not None:
        offset = np.array(offset, dtype=np.float64)

    facing = data.get("facing", "center")
    rotation = data.get("rotation")

    return AttachmentPoint(
        name=name,
        anchor=anchor,
        offset=offset,
        facing=facing,
        rotation=rotation,
    )
