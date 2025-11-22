"""Chair scene."""

from pathlib import Path

from ..core.node import SceneNode
from ..layout import LayoutLoader


def create_chair_scene() -> SceneNode:
    """Create a scene containing just the chair.

    Returns:
        A SceneNode containing the chair geometry.
    """
    root = SceneNode("root")

    assets_dir = Path(__file__).parent.parent.parent.parent / "assets"
    loader = LayoutLoader()
    chair = loader.load(assets_dir / "chair.yaml")
    root.add_child(chair)

    return root
