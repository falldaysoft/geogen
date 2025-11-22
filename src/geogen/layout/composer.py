"""Scene composer for assembling multi-asset scenes from YAML definitions."""

from pathlib import Path
from typing import Any

import yaml

from ..core.node import SceneNode
from .loader import LayoutLoader


class SceneComposer:
    """Composes scenes by loading and connecting multiple assets.

    The composer reads a composition YAML file that specifies which assets
    to load and how to connect them via attachment points.

    YAML format:
        name: scene_name

        compose:
          base_object:
            asset: path/to/asset.yaml

          attached_objects:
            asset: path/to/other.yaml
            attach_to: base_object
            at: [attachment_name1, attachment_name2, ...]

    Example:
        name: dining_set

        compose:
          table:
            asset: table.yaml

          chairs:
            asset: chair.yaml
            attach_to: table
            at: [seat_1, seat_2, seat_3, seat_4]
    """

    def __init__(self, assets_dir: str | Path | None = None) -> None:
        """Initialize the composer.

        Args:
            assets_dir: Base directory for asset files. Defaults to 'assets/' relative to project.
        """
        self._loader = LayoutLoader()
        if assets_dir is None:
            # Default to project's assets directory
            self._assets_dir = Path(__file__).parent.parent.parent.parent / "assets"
        else:
            self._assets_dir = Path(assets_dir)

    def compose(self, path: str | Path) -> SceneNode:
        """Load and compose a scene from a YAML file.

        Args:
            path: Path to the composition YAML file

        Returns:
            Root SceneNode of the composed scene
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return self._build_scene(data)

    def compose_string(self, yaml_string: str) -> SceneNode:
        """Compose a scene from a YAML string.

        Args:
            yaml_string: YAML content as a string

        Returns:
            Root SceneNode of the composed scene
        """
        data = yaml.safe_load(yaml_string)
        return self._build_scene(data)

    def _build_scene(self, data: dict[str, Any]) -> SceneNode:
        """Build scene from parsed YAML data."""
        name = data.get("name", "composed_scene")
        root = SceneNode(name)

        compose_data = data.get("compose", {})

        # First pass: load all base assets (those without attach_to)
        loaded_assets: dict[str, SceneNode] = {}

        for obj_name, obj_def in compose_data.items():
            if "attach_to" not in obj_def:
                asset_path = self._assets_dir / obj_def["asset"]
                node = self._loader.load(asset_path)
                node.name = obj_name
                root.add_child(node)
                loaded_assets[obj_name] = node

        # Second pass: attach objects to their targets
        for obj_name, obj_def in compose_data.items():
            if "attach_to" in obj_def:
                target_name = obj_def["attach_to"]
                target = loaded_assets.get(target_name)
                if target is None:
                    raise ValueError(
                        f"Cannot attach '{obj_name}': target '{target_name}' not found"
                    )

                asset_path = self._assets_dir / obj_def["asset"]
                attachment_names = obj_def.get("at", [])

                if isinstance(attachment_names, str):
                    attachment_names = [attachment_names]

                for i, attach_name in enumerate(attachment_names):
                    # Load a fresh copy for each attachment
                    node = self._loader.load(asset_path)
                    node.name = f"{obj_name}_{i}" if len(attachment_names) > 1 else obj_name

                    # Get the attachment transform
                    attach_transform = target.get_attachment(attach_name)
                    if attach_transform is None:
                        raise ValueError(
                            f"Attachment point '{attach_name}' not found on '{target_name}'"
                        )

                    node.transform = attach_transform
                    root.add_child(node)

        return root
