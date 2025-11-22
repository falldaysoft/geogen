"""Table scene."""

from pathlib import Path

from ..core.node import SceneNode
from ..layout import LayoutLoader


def create_table_scene() -> SceneNode:
    """Create a scene containing just the table.

    Returns:
        A SceneNode containing the table geometry.
    """
    root = SceneNode("root")

    assets_dir = Path(__file__).parent.parent.parent.parent / "assets"
    loader = LayoutLoader()
    table = loader.load(assets_dir / "table.yaml")
    root.add_child(table)

    return root
