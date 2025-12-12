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
from .attachments import parse_attachment, AttachmentPoint


# Registry of available primitive generators
PRIMITIVE_REGISTRY = {
    "cube": CubeGenerator,
    "cylinder": CylinderGenerator,
    "sphere": SphereGenerator,
    "cone": ConeGenerator,
    "room": RoomGenerator,
}


def get_primitive_attachment_points(primitive_type: str, size: np.ndarray) -> dict[str, AttachmentPoint]:
    """Generate standard attachment points for a primitive based on its type and size.

    For cylinders/cones: top, bottom, and radial points (left, right, front, back)
    For spheres: top, bottom, and radial points at equator
    For cubes: center of each face

    Args:
        primitive_type: Type of primitive (cube, cylinder, sphere, cone)
        size: Actual size [width, height, depth]

    Returns:
        Dictionary of attachment point name -> AttachmentPoint
    """
    attachments = {}
    half_x = size[0] / 2
    half_y = size[1] / 2
    half_z = size[2] / 2

    if primitive_type == "cylinder" or primitive_type == "cone":
        radius = min(size[0], size[2]) / 2
        # Top and bottom (along Y axis)
        attachments["top"] = AttachmentPoint(
            name="top",
            anchor="center",
            offset=np.array([0, half_y, 0]),
            facing="north",
        )
        attachments["bottom"] = AttachmentPoint(
            name="bottom",
            anchor="center",
            offset=np.array([0, -half_y, 0]),
            facing="north",
        )
        # Radial points at mid-height
        attachments["left"] = AttachmentPoint(
            name="left",
            anchor="center",
            offset=np.array([-radius, 0, 0]),
            facing="west",
        )
        attachments["right"] = AttachmentPoint(
            name="right",
            anchor="center",
            offset=np.array([radius, 0, 0]),
            facing="east",
        )
        attachments["front"] = AttachmentPoint(
            name="front",
            anchor="center",
            offset=np.array([0, 0, radius]),
            facing="south",
        )
        attachments["back"] = AttachmentPoint(
            name="back",
            anchor="center",
            offset=np.array([0, 0, -radius]),
            facing="north",
        )

    elif primitive_type == "sphere":
        radius_x = size[0] / 2
        radius_y = size[1] / 2
        radius_z = size[2] / 2
        # Top and bottom
        attachments["top"] = AttachmentPoint(
            name="top",
            anchor="center",
            offset=np.array([0, radius_y, 0]),
            facing="north",
        )
        attachments["bottom"] = AttachmentPoint(
            name="bottom",
            anchor="center",
            offset=np.array([0, -radius_y, 0]),
            facing="north",
        )
        # Equator points
        attachments["left"] = AttachmentPoint(
            name="left",
            anchor="center",
            offset=np.array([-radius_x, 0, 0]),
            facing="west",
        )
        attachments["right"] = AttachmentPoint(
            name="right",
            anchor="center",
            offset=np.array([radius_x, 0, 0]),
            facing="east",
        )
        attachments["front"] = AttachmentPoint(
            name="front",
            anchor="center",
            offset=np.array([0, 0, radius_z]),
            facing="south",
        )
        attachments["back"] = AttachmentPoint(
            name="back",
            anchor="center",
            offset=np.array([0, 0, -radius_z]),
            facing="north",
        )

    elif primitive_type == "cube":
        # Face centers
        attachments["top"] = AttachmentPoint(
            name="top",
            anchor="center",
            offset=np.array([0, half_y, 0]),
            facing="north",
        )
        attachments["bottom"] = AttachmentPoint(
            name="bottom",
            anchor="center",
            offset=np.array([0, -half_y, 0]),
            facing="north",
        )
        attachments["left"] = AttachmentPoint(
            name="left",
            anchor="center",
            offset=np.array([-half_x, 0, 0]),
            facing="west",
        )
        attachments["right"] = AttachmentPoint(
            name="right",
            anchor="center",
            offset=np.array([half_x, 0, 0]),
            facing="east",
        )
        attachments["front"] = AttachmentPoint(
            name="front",
            anchor="center",
            offset=np.array([0, 0, half_z]),
            facing="south",
        )
        attachments["back"] = AttachmentPoint(
            name="back",
            anchor="center",
            offset=np.array([0, 0, -half_z]),
            facing="north",
        )

    return attachments


class LayoutLoader:
    """Loads composite object definitions from YAML files.

    YAML format supports two modes:

    1. Coordinate-based (legacy):
        parts:
          part_name:
            primitive: cube|cylinder|sphere|cone
            size: [x, y, z]  # fractions of parent size (0-1)
            anchor: anchor_name
            offset: [x, y, z]

    2. Hierarchical attachment (preferred):
        parts:
          base_part:
            primitive: cylinder
            size: [x, y, z]

          child_part:
            primitive: sphere
            size: [x, y, z]
            attach_to: base_part
            at: top  # attachment point name (top, bottom, left, right, front, back)
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

        # First pass: create all parts and track which have attachments
        part_nodes: dict[str, SceneNode] = {}
        part_sizes: dict[str, np.ndarray] = {}
        part_types: dict[str, str] = {}

        for part_name, part_def in parts.items():
            primitive_type = part_def["primitive"]
            frac_size = np.array(part_def["size"], dtype=np.float64)
            actual_size = frac_size * container_size

            generator = self._create_generator(primitive_type, actual_size)
            node = generator.to_node(part_name)

            # Apply material if specified
            material_name = part_def.get("material")
            if material_name is not None and node.mesh is not None:
                try:
                    material = self._material_loader.load(material_name)
                    node.mesh.material = material
                except FileNotFoundError:
                    pass

            # Store the node's actual size for attachment calculations
            node.size = actual_size

            # Generate automatic attachment points for this primitive
            auto_attachments = get_primitive_attachment_points(primitive_type, actual_size)
            for attach_name, attach_point in auto_attachments.items():
                node.attachments[attach_name] = attach_point

            part_nodes[part_name] = node
            part_sizes[part_name] = actual_size
            part_types[part_name] = primitive_type

        # Second pass: position parts (either by attachment or coordinate)
        for part_name, part_def in parts.items():
            node = part_nodes[part_name]
            actual_size = part_sizes[part_name]

            if "attach_to" in part_def:
                # Hierarchical attachment mode
                parent_name = part_def["attach_to"]
                attach_point = part_def.get("at", "top")
                child_attach = part_def.get("from", "bottom")  # Which point on child to attach

                if parent_name not in part_nodes:
                    raise ValueError(f"Part '{part_name}' cannot attach to unknown part '{parent_name}'")

                parent_node = part_nodes[parent_name]

                # Get the attachment point on the parent
                if attach_point not in parent_node.attachments:
                    raise ValueError(f"Attachment point '{attach_point}' not found on '{parent_name}'")

                parent_attach = parent_node.attachments[attach_point]
                parent_offset = parent_attach.offset.copy()

                # Get the child's attachment point offset (to align properly)
                child_offset = np.zeros(3)
                if child_attach in node.attachments:
                    child_offset = node.attachments[child_attach].offset.copy()

                # If the child has rotation, we need to rotate the child offset
                # because the attachment point moves with the rotation
                rotation = part_def.get("rotation")
                if rotation is not None:
                    rotation_rad = np.radians(np.array(rotation, dtype=np.float64))
                    node.transform.rotation = rotation_rad
                    # Rotate the child offset to match the applied rotation
                    child_offset = self._rotate_point(child_offset, rotation_rad)

                # Position child so its attachment point aligns with parent's attachment point
                # Both offsets are relative to each primitive's center (origin)
                node.transform.translation = parent_offset - child_offset

                # Add as child of the parent node (hierarchical)
                parent_node.add_child(node)

            else:
                # Coordinate-based positioning (legacy/root mode)
                # For parts with anchor "bottom_center", this places the part's bottom at that anchor
                anchor_name = part_def.get("anchor", "center")
                anchor_pos = resolve_anchor(anchor_name, container_size)

                offset = np.array(part_def.get("offset", [0, 0, 0]), dtype=np.float64)
                offset_world = offset * container_size

                position = anchor_pos + offset_world

                # If anchor is a "bottom" anchor, shift up so bottom of part is at anchor
                if "bottom" in anchor_name:
                    position[1] += actual_size[1] / 2

                node.transform.translation = position

                # Apply rotation if specified
                rotation = part_def.get("rotation")
                if rotation is not None:
                    rotation_rad = np.radians(np.array(rotation, dtype=np.float64))
                    node.transform.rotation = rotation_rad

                root.add_child(node)

        # Parse explicit attachment points for the composite object
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
            SceneNode containing the room mesh(es)
        """
        generator = self._create_room_generator(size, room_config)

        # Check if materials are specified for room surfaces
        materials_config = room_config.get("materials", {})
        if materials_config:
            # Use composite node with separate materials
            floor_material = None
            wall_material = None
            ceiling_material = None

            if "floor" in materials_config:
                try:
                    floor_material = self._material_loader.load(materials_config["floor"])
                except FileNotFoundError:
                    pass

            if "walls" in materials_config:
                try:
                    wall_material = self._material_loader.load(materials_config["walls"])
                except FileNotFoundError:
                    pass

            if "ceiling" in materials_config:
                try:
                    ceiling_material = self._material_loader.load(materials_config["ceiling"])
                except FileNotFoundError:
                    pass

            return generator.to_composite_node(
                name=f"{name}_geometry",
                floor_material=floor_material,
                wall_material=wall_material,
                ceiling_material=ceiling_material,
            )
        else:
            # No materials, use single merged mesh
            return generator.to_node(f"{name}_geometry")

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
            # Use the minimum dimension as the base radius
            # Scaling will be handled by transforming the mesh
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

    def _rotate_point(self, point: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Rotate a point by XYZ Euler angles.

        Args:
            point: The point to rotate [x, y, z]
            rotation: Euler angles in radians [rx, ry, rz]

        Returns:
            Rotated point
        """
        from scipy.spatial.transform import Rotation
        r = Rotation.from_euler('xyz', rotation)
        return r.apply(point)

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
