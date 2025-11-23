"""Room scene."""

from pathlib import Path

from ..core.node import SceneNode
from ..layout import LayoutLoader


def create_room_scene() -> SceneNode:
    """Create a scene containing just the room.

    Returns:
        A SceneNode containing the room geometry.
    """
    root = SceneNode("root")

    assets_dir = Path(__file__).parent.parent.parent.parent / "assets"
    loader = LayoutLoader()
    room = loader.load(assets_dir / "room.yaml")
    root.add_child(room)

    return root
