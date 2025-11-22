"""Dining set scene: table with 4 chairs."""

from pathlib import Path

from ..core.node import SceneNode
from ..layout import SceneComposer


def create_dining_set_scene() -> SceneNode:
    """Create a scene with a table and 4 chairs.

    Returns:
        A SceneNode containing the dining set geometry.
    """
    assets_dir = Path(__file__).parent.parent.parent.parent / "assets"
    composer = SceneComposer(assets_dir)
    return composer.compose(assets_dir / "dining_set.yaml")
