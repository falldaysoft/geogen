"""Scene composer for assembling multi-asset scenes from YAML definitions."""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..core.node import SceneNode
from ..core.transform import Transform
from .loader import LayoutLoader
from .anchors import resolve_anchor
from .attachments import parse_attachment
from .validation import validate_scene_yaml, pre_validate_scene_references

logger = logging.getLogger("geogen.layout")


class SceneComposer:
    """Composes scenes by loading and connecting multiple assets.

    The composer reads a composition YAML file that specifies which assets
    to load and how to position them. Supports three positioning modes:

    1. **Slots**: Named positions defined in the scene with semantic meaning.
       Objects are placed at slots by name.

    2. **Attachments**: Objects attached to other objects' attachment points
       (e.g., chairs attached to table's seat points).

    3. **Nested scenes**: Composed scenes can reference other composed scenes,
       enabling hierarchical composition (street -> house -> dining_set).

    YAML format:
        name: scene_name
        size: [width, height, depth]  # Optional, for slot positioning

        # Define named slots for semantic positioning
        slots:
          slot_name:
            anchor: bottom_center      # Base anchor within scene bounds
            offset: [x, y, z]          # Offset in scene units (optional)
            facing: center|outward|north|south|east|west  # Direction (optional)

        # Place objects into the scene
        place:
          object_name:
            asset: path/to/asset.yaml  # Primitive asset
            # OR
            scene: path/to/scene.yaml  # Composed scene (nested)

            slot: slot_name            # Place at a named slot
            # OR
            attach_to: other_object    # Attach to another object
            at: attachment_point       # At this attachment point (or list of points)

    Example - Street scene:
        name: street
        size: [20, 0, 10]

        slots:
          sidewalk_lamp:
            anchor: bottom_front_left
            offset: [0.1, 0, 0.1]
          sidewalk_bench:
            anchor: bottom_front_center
            offset: [0, 0, 0.1]
            facing: south
          plot_left:
            anchor: bottom_back_left
            offset: [0.25, 0, 0.25]
          plot_right:
            anchor: bottom_back_right
            offset: [-0.25, 0, 0.25]

        place:
          lamp:
            asset: street_lamp.yaml
            slot: sidewalk_lamp

          bench:
            asset: bench.yaml
            slot: sidewalk_bench

          house_left:
            scene: house.yaml
            slot: plot_left

          house_right:
            scene: house.yaml
            slot: plot_right
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
        logger.debug("Composing scene: %s", path)
        with open(path) as f:
            data = yaml.safe_load(f)

        # Validate YAML structure
        for warning in validate_scene_yaml(data):
            warnings.warn(warning, stacklevel=2)

        # Pre-validate references
        ref_errors = pre_validate_scene_references(data)
        if ref_errors:
            raise ValueError(
                f"Invalid references in {path.name}:\n  " + "\n  ".join(ref_errors)
            )

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

        # Get scene size for slot positioning and propagate to root node
        size = np.array(data.get("size", [1.0, 1.0, 1.0]), dtype=np.float64)
        root.size = size
        logger.debug("Building scene: %s (size=%s)", name, size)

        # Parse slot definitions
        slots = self._parse_slots(data.get("slots", {}), size)

        # Handle old 'compose' format for backwards compatibility
        compose_data = data.get("compose", {})
        place_data = data.get("place", {})

        # Merge compose into place for unified handling
        all_placements = {**compose_data, **place_data}

        # First pass: load all objects that don't depend on others
        loaded_objects: dict[str, SceneNode] = {}

        for obj_name, obj_def in all_placements.items():
            if "attach_to" not in obj_def:
                node = self._load_object(obj_def)
                node.name = obj_name

                # Position the object
                if "slot" in obj_def:
                    slot_name = obj_def["slot"]
                    if slot_name not in slots:
                        import difflib
                        available = sorted(slots.keys())
                        suggestion = difflib.get_close_matches(slot_name, available, n=1, cutoff=0.5)
                        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
                        raise ValueError(
                            f"Slot '{slot_name}' not defined.{hint}"
                            f" Available slots: {available}"
                        )
                    logger.debug("Placing '%s' at slot '%s'", obj_name, slot_name)
                    node.transform = slots[slot_name]

                root.add_child(node)
                loaded_objects[obj_name] = node

        # Second pass: attach objects to their targets
        for obj_name, obj_def in all_placements.items():
            if "attach_to" in obj_def:
                target_name = obj_def["attach_to"]
                target = loaded_objects.get(target_name)
                if target is None:
                    import difflib
                    available = sorted(loaded_objects.keys())
                    suggestion = difflib.get_close_matches(target_name, available, n=1, cutoff=0.5)
                    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
                    raise ValueError(
                        f"Cannot attach '{obj_name}': target '{target_name}' not found.{hint}"
                        f" Available objects: {available}"
                    )

                attachment_names = obj_def.get("at", [])
                if isinstance(attachment_names, str):
                    attachment_names = [attachment_names]

                for i, attach_name in enumerate(attachment_names):
                    node = self._load_object(obj_def)
                    node.name = f"{obj_name}_{i}" if len(attachment_names) > 1 else obj_name

                    attach_transform = target.get_attachment(attach_name)
                    if attach_transform is None:
                        import difflib
                        available = target.list_attachments()
                        suggestion = difflib.get_close_matches(attach_name, available, n=1, cutoff=0.5)
                        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
                        raise ValueError(
                            f"Attachment point '{attach_name}' not found on '{target_name}'.{hint}"
                            f" Available: {available}"
                        )
                    logger.debug("Attaching '%s' to '%s' at '%s'", obj_name, target_name, attach_name)

                    node.transform = attach_transform
                    root.add_child(node)

        # Parse explicit attachment points for the composed scene
        attachments_data = data.get("attachments", {})
        for attach_name, attach_def in attachments_data.items():
            attachment = parse_attachment(attach_name, attach_def)
            root.attachments[attach_name] = attachment

        return root

    def _load_object(self, obj_def: dict[str, Any]) -> SceneNode:
        """Load an object from asset or scene definition."""
        if "asset" in obj_def:
            asset_path = self._assets_dir / obj_def["asset"]
            return self._loader.load(asset_path)
        elif "scene" in obj_def:
            scene_path = self._assets_dir / obj_def["scene"]
            return self.compose(scene_path)
        else:
            raise ValueError("Object must have 'asset' or 'scene' specified")

    def _parse_slots(
        self, slots_data: dict[str, Any], size: np.ndarray
    ) -> dict[str, Transform]:
        """Parse slot definitions into transforms.

        Slots can specify position in two ways:
        1. anchor + offset: offset is in ABSOLUTE units (meters), not fractions
        2. position: direct [x, y, z] coordinates relative to scene center
        """
        slots = {}

        for slot_name, slot_def in slots_data.items():
            facing = slot_def.get("facing", "north")

            if "position" in slot_def:
                # Direct position in local coordinates (meters from center)
                position = np.array(slot_def["position"], dtype=np.float64)
            else:
                # Anchor-based positioning
                anchor = slot_def.get("anchor", "center")
                offset = np.array(slot_def.get("offset", [0, 0, 0]), dtype=np.float64)

                # Resolve anchor to position
                position = resolve_anchor(anchor, size)
                # Apply offset in ABSOLUTE units (meters), not fractions
                position += offset

            # Compute rotation from facing direction
            rotation = self._facing_to_rotation(facing, position)

            slots[slot_name] = Transform(
                translation=position,
                rotation=np.array([0.0, rotation, 0.0], dtype=np.float64),
            )

        return slots

    def _facing_to_rotation(self, facing: str, position: np.ndarray) -> float:
        """Convert facing direction to Y-axis rotation in radians."""
        if facing == "center":
            # Face toward origin
            dx = -position[0]
            dz = -position[2]
            return float(np.arctan2(dx, dz))
        elif facing == "outward":
            # Face away from origin
            dx = position[0]
            dz = position[2]
            return float(np.arctan2(dx, dz))
        elif facing == "north":
            return 0.0
        elif facing == "south":
            return float(np.pi)
        elif facing == "east":
            return float(np.pi / 2)
        elif facing == "west":
            return float(-np.pi / 2)
        else:
            return 0.0
