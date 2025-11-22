"""Anchor point system for positioning objects within containers."""

from enum import Enum
import numpy as np


class Anchor(Enum):
    """Named anchor points within a container's bounding box.

    Anchors are defined in normalized coordinates (0-1) where:
    - X: 0 = left, 1 = right
    - Y: 0 = bottom, 1 = top
    - Z: 0 = back, 1 = front
    """
    # Center points
    CENTER = "center"
    BOTTOM_CENTER = "bottom_center"
    TOP_CENTER = "top_center"

    # Bottom corners
    BOTTOM_FRONT_LEFT = "bottom_front_left"
    BOTTOM_FRONT_RIGHT = "bottom_front_right"
    BOTTOM_BACK_LEFT = "bottom_back_left"
    BOTTOM_BACK_RIGHT = "bottom_back_right"

    # Top corners
    TOP_FRONT_LEFT = "top_front_left"
    TOP_FRONT_RIGHT = "top_front_right"
    TOP_BACK_LEFT = "top_back_left"
    TOP_BACK_RIGHT = "top_back_right"

    # Edge centers (useful for backs, sides)
    TOP_BACK_CENTER = "top_back_center"
    TOP_FRONT_CENTER = "top_front_center"
    BOTTOM_BACK_CENTER = "bottom_back_center"
    BOTTOM_FRONT_CENTER = "bottom_front_center"

    # Side centers
    LEFT_CENTER = "left_center"
    RIGHT_CENTER = "right_center"
    FRONT_CENTER = "front_center"
    BACK_CENTER = "back_center"


# Mapping from anchor to normalized coordinates (x, y, z)
# X: 0=left, 1=right | Y: 0=bottom, 1=top | Z: 0=back, 1=front
ANCHOR_POSITIONS: dict[Anchor, tuple[float, float, float]] = {
    # Centers
    Anchor.CENTER: (0.5, 0.5, 0.5),
    Anchor.BOTTOM_CENTER: (0.5, 0.0, 0.5),
    Anchor.TOP_CENTER: (0.5, 1.0, 0.5),

    # Bottom corners
    Anchor.BOTTOM_FRONT_LEFT: (0.0, 0.0, 1.0),
    Anchor.BOTTOM_FRONT_RIGHT: (1.0, 0.0, 1.0),
    Anchor.BOTTOM_BACK_LEFT: (0.0, 0.0, 0.0),
    Anchor.BOTTOM_BACK_RIGHT: (1.0, 0.0, 0.0),

    # Top corners
    Anchor.TOP_FRONT_LEFT: (0.0, 1.0, 1.0),
    Anchor.TOP_FRONT_RIGHT: (1.0, 1.0, 1.0),
    Anchor.TOP_BACK_LEFT: (0.0, 1.0, 0.0),
    Anchor.TOP_BACK_RIGHT: (1.0, 1.0, 0.0),

    # Edge centers
    Anchor.TOP_BACK_CENTER: (0.5, 1.0, 0.0),
    Anchor.TOP_FRONT_CENTER: (0.5, 1.0, 1.0),
    Anchor.BOTTOM_BACK_CENTER: (0.5, 0.0, 0.0),
    Anchor.BOTTOM_FRONT_CENTER: (0.5, 0.0, 1.0),

    # Side centers
    Anchor.LEFT_CENTER: (0.0, 0.5, 0.5),
    Anchor.RIGHT_CENTER: (1.0, 0.5, 0.5),
    Anchor.FRONT_CENTER: (0.5, 0.5, 1.0),
    Anchor.BACK_CENTER: (0.5, 0.5, 0.0),
}


def resolve_anchor(anchor: Anchor | str, container_size: np.ndarray) -> np.ndarray:
    """Convert an anchor point to world coordinates within a container.

    Args:
        anchor: The anchor point (enum or string name)
        container_size: The size of the container [width, height, depth]

    Returns:
        World coordinates [x, y, z] relative to container's bottom_center origin
    """
    if isinstance(anchor, str):
        anchor = Anchor(anchor)

    norm_pos = np.array(ANCHOR_POSITIONS[anchor])

    # Container origin is bottom_center, so we need to adjust x and z
    # X: -width/2 to +width/2 (norm 0->1 maps to -0.5->+0.5 of width)
    # Y: 0 to height (norm 0->1 maps directly)
    # Z: -depth/2 to +depth/2 (norm 0->1 maps to -0.5->+0.5 of depth)
    world_pos = np.array([
        (norm_pos[0] - 0.5) * container_size[0],  # X: centered
        norm_pos[1] * container_size[1],           # Y: bottom is 0
        (norm_pos[2] - 0.5) * container_size[2],  # Z: centered
    ])

    return world_pos
