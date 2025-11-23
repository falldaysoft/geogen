"""YAML loader for composite object definitions."""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..core.node import SceneNode
from ..core.transform import Transform
from ..generators.primitives import CubeGenerator, CylinderGenerator, SphereGenerator, ConeGenerator
from ..generators.room import RoomGenerator, Opening
from ..materials.loader import MaterialLoader
from .anchors import Anchor, resolve_anchor
from .attachments import parse_attachment


# Registry of available primitive generators
PRIMITIVE_REGISTRY = {
    "cube": CubeGenerator,
    "cylinder": CylinderGenerator,
    "sphere": SphereGenerator,
    "cone": ConeGenerator,
    "room": RoomGenerator,
}


class LayoutLoader:
    """Loads composite object definitions from YAML files.

    YAML format:
        name: object_name
        origin: bottom_center  # where the object's origin is
        size: [width, height, depth]  # bounding box in world units

        parts:
          part_name:
            primitive: cube|cylinder|sphere|cone
            size: [x, y, z]  # fractions of parent size (0-1)
            anchor: anchor_name  # where in parent to position
            offset: [x, y, z]  # offset from anchor in fractions of parent size
            material: material_name  # optional material reference
    """

    def __init__(self) -> None:
        """Initialize the layout loader with a material loader."""
        self._material_loader = MaterialLoader()

    def load(self, path: str | Path) -> SceneNode:
        """Load a composite object definition from a YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            SceneNode hierarchy representing the composite object
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return self._build_hierarchy(data)

    def load_string(self, yaml_string: str) -> SceneNode:
        """Load a composite object definition from a YAML string.

        Args:
            yaml_string: YAML content as a string

        Returns:
            SceneNode hierarchy representing the composite object
        """
        data = yaml.safe_load(yaml_string)
        return self._build_hierarchy(data)

    def _build_hierarchy(self, data: dict[str, Any]) -> SceneNode:
        """Build scene hierarchy from parsed YAML data."""
        name = data.get("name", "composite")
        container_size = np.array(data["size"], dtype=np.float64)

        root = SceneNode(name)
        root.size = container_size

        # Check if this is a room definition (has 'room' key with openings)
        room_config = data.get("room")
        if room_config is not None:
            room_node = self._create_room_node(name, container_size, room_config)
            root.add_child(room_node)

        parts = data.get("parts", {})
        for part_name, part_def in parts.items():
            part_node = self._create_part(part_name, part_def, container_size)
            root.add_child(part_node)

        # Parse attachment points
        attachments_data = data.get("attachments", {})
        for attach_name, attach_def in attachments_data.items():
            attachment = parse_attachment(attach_name, attach_def)
            root.attachments[attach_name] = attachment

        return root

    def _create_room_node(
        self, name: str, size: np.ndarray, room_config: dict[str, Any]
    ) -> SceneNode:
        """Create a room node from room configuration.

        Args:
            name: Name for the room node
            size: Room size [width, height, depth]
            room_config: Room configuration dict from YAML

        Returns:
            SceneNode containing the room mesh
        """
        generator = self._create_room_generator(size, room_config)
        return generator.to_node(f"{name}_geometry")

    def _create_part(
        self, name: str, part_def: dict[str, Any], container_size: np.ndarray
    ) -> SceneNode:
        """Create a part node from its definition.

        Args:
            name: Name for the part node
            part_def: Part definition dict from YAML
            container_size: Size of the parent container [width, height, depth]

        Returns:
            SceneNode for the part with correct transform
        """
        primitive_type = part_def["primitive"]
        generator_class = PRIMITIVE_REGISTRY.get(primitive_type)
        if generator_class is None:
            raise ValueError(f"Unknown primitive type: {primitive_type}")

        # Calculate actual size from fractional size
        frac_size = np.array(part_def["size"], dtype=np.float64)
        actual_size = frac_size * container_size

        # Create the appropriate generator with computed size
        generator = self._create_generator(primitive_type, actual_size)
        node = generator.to_node(name)

        # Apply material if specified
        material_name = part_def.get("material")
        if material_name is not None and node.mesh is not None:
            try:
                material = self._material_loader.load(material_name)
                node.mesh.material = material
            except FileNotFoundError:
                # Material not found, continue without it
                pass

        # Calculate position from anchor + offset
        anchor_name = part_def.get("anchor", "center")
        anchor_pos = resolve_anchor(anchor_name, container_size)

        offset = np.array(part_def.get("offset", [0, 0, 0]), dtype=np.float64)
        # Offset is in fractions of container size
        offset_world = offset * container_size

        # Position is anchor + offset
        # But primitives are centered at origin, so for Y we need to account
        # for the primitive's own height (shift up by half its height)
        position = anchor_pos + offset_world

        # Primitives are centered at origin. For bottom_center origin containers,
        # a part at anchor "bottom_center" with the primitive's center should
        # be shifted up by half its height to sit on the floor.
        # We add half the part's Y size to lift it so its bottom is at anchor.
        position[1] += actual_size[1] / 2

        node.transform.translation = position

        return node

    def _create_generator(
        self, primitive_type: str, size: np.ndarray, extra_config: dict[str, Any] | None = None
    ):
        """Create a generator instance with the given size.

        Args:
            primitive_type: Type of primitive (cube, cylinder, etc.)
            size: Actual size [width, height, depth]
            extra_config: Additional configuration for complex generators (e.g., room)

        Returns:
            Generator instance configured with the size
        """
        if primitive_type == "cube":
            return CubeGenerator(size_x=size[0], size_y=size[1], size_z=size[2])
        elif primitive_type == "cylinder":
            # Cylinder uses radius (half of x/z) and height
            radius = min(size[0], size[2]) / 2
            return CylinderGenerator(radius=radius, height=size[1])
        elif primitive_type == "sphere":
            # Sphere uses radius (half of smallest dimension)
            radius = min(size) / 2
            return SphereGenerator(radius=radius)
        elif primitive_type == "cone":
            # Cone uses radius and height
            radius = min(size[0], size[2]) / 2
            return ConeGenerator(radius=radius, height=size[1])
        elif primitive_type == "room":
            return self._create_room_generator(size, extra_config or {})
        else:
            raise ValueError(f"Unknown primitive type: {primitive_type}")

    def _create_room_generator(
        self, size: np.ndarray, config: dict[str, Any]
    ) -> RoomGenerator:
        """Create a room generator with the given configuration.

        Args:
            size: Room size [width, height, depth]
            config: Room configuration from YAML

        Returns:
            RoomGenerator instance
        """
        openings = []
        for opening_def in config.get("openings", []):
            openings.append(Opening(
                wall=opening_def["wall"],
                position=opening_def.get("position", 0.5),
                bottom=opening_def.get("bottom", 0.0),
                width=opening_def.get("width", 0.2),
                height=opening_def.get("height", 0.8),
            ))

        return RoomGenerator(
            size_x=size[0],
            size_y=size[1],
            size_z=size[2],
            wall_thickness=config.get("wall_thickness", 0.1),
            has_floor=config.get("has_floor", True),
            has_ceiling=config.get("has_ceiling", True),
            openings=openings,
        )
