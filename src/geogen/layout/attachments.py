"""Attachment point system for connecting objects together."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..core.transform import Transform
from .anchors import Anchor, resolve_anchor


FacingDirection = Literal["center", "outward", "north", "south", "east", "west"]


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
        offset: Additional offset in fractions of container size [x, y, z]
        facing: Semantic direction the attached object should face
        rotation: Explicit Y rotation in degrees (overrides facing if set)
    """

    name: str
    anchor: Anchor | str
    offset: NDArray[np.float64] | None = None
    facing: FacingDirection = "center"
    rotation: float | None = None  # degrees, Y-axis rotation

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
        offset_world = self.offset * container_size
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

        # Semantic facing directions
        if self.facing == "center":
            # Face toward the center (0, y, 0)
            # Calculate angle from position to center
            dx = -position[0]  # direction to center x
            dz = -position[2]  # direction to center z
            return np.arctan2(dx, dz)

        elif self.facing == "outward":
            # Face away from center
            dx = position[0]
            dz = position[2]
            return np.arctan2(dx, dz)

        elif self.facing == "north":
            return 0.0  # +Z direction

        elif self.facing == "south":
            return np.pi  # -Z direction

        elif self.facing == "east":
            return np.pi / 2  # +X direction

        elif self.facing == "west":
            return -np.pi / 2  # -X direction

        return 0.0


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
