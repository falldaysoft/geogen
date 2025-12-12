"""Street scene with various street furniture."""

from pathlib import Path

from ..core.node import SceneNode
from ..layout import SceneComposer


def create_street_scene() -> SceneNode:
    """Create a scene with street furniture objects.

    Loads the street scene from YAML, which includes a fire hydrant,
    mailbox, trashcan, bench, and street lamp arranged along a sidewalk.

    Returns:
        A SceneNode containing the street furniture.
    """
    assets_dir = Path(__file__).parent.parent.parent.parent / "assets"
    composer = SceneComposer(assets_dir)
    return composer.compose(assets_dir / "scenes" / "street.yaml")
