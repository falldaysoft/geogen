"""YAML loader for composite object definitions."""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..core.node import SceneNode
from ..core.profile import Shape, polyline_from_spec, shape_from_spec
from ..generators.primitives import (
    ConeGenerator,
    CubeGenerator,
    CylinderGenerator,
    EllipsoidGenerator,
    PlaneGenerator,
    SphereGenerator,
)
from ..generators.architecture import PrismGenerator, RoofGenerator
from ..generators.profiles import _AXIS_FRAMES, ExtrudeGenerator, LatheGenerator
from ..generators.round_shapes import BevelledCylinderGenerator, CapsuleGenerator, TorusGenerator
from ..generators.stairs import StairsGenerator
from ..generators.sweep import SweepGenerator
from ..generators.room import RoomGenerator, Opening
from ..materials.loader import MaterialLoader
from .anchors import resolve_anchor
from .attachments import parse_attachment
from .expressions import resolve_params, resolve_value
from .validation import validate_asset_yaml, pre_validate_asset_references
from .yaml_utils import safe_load, safe_load_path

logger = logging.getLogger("geogen.layout")


# Collision shapes a part can request (``collider:``); see export.py.
COLLIDER_TYPES = {"auto", "none", "box", "hull", "mesh"}

# Registry of available primitive generators
PRIMITIVE_REGISTRY = {
    "cube": CubeGenerator,
    "cylinder": CylinderGenerator,
    "sphere": SphereGenerator,
    "cone": ConeGenerator,
    "plane": PlaneGenerator,
    "room": RoomGenerator,
    "ellipsoid": EllipsoidGenerator,
    "extrude": ExtrudeGenerator,
    "lathe": LatheGenerator,
    "roof": RoofGenerator,
    "prism": PrismGenerator,
    "sweep": SweepGenerator,
    "stairs": StairsGenerator,
    "torus": TorusGenerator,
    "capsule": CapsuleGenerator,
}


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

    def load(
        self, path: str | Path, params: dict[str, float] | None = None
    ) -> SceneNode:
        """Load a composite object definition from a YAML file.

        Args:
            path: Path to the YAML file
            params: Optional parameter overrides for parametric assets

        Returns:
            SceneNode hierarchy representing the composite object
        """
        path = Path(path)
        logger.debug("Loading asset: %s", path)
        data = safe_load_path(path)

        # Validate YAML structure (pre-resolve; catches unknown keys)
        for warning in validate_asset_yaml(data):
            warnings.warn(warning, stacklevel=2)

        # Pre-validate references
        ref_errors = pre_validate_asset_references(data)
        if ref_errors:
            raise ValueError(
                f"Invalid references in {path.name}:\n  " + "\n  ".join(ref_errors)
            )

        data = self._resolve_params(data, params)
        return self._build_hierarchy(data)

    def load_string(
        self, yaml_string: str, params: dict[str, float] | None = None
    ) -> SceneNode:
        """Load a composite object definition from a YAML string.

        Args:
            yaml_string: YAML content as a string
            params: Optional parameter overrides for parametric assets

        Returns:
            SceneNode hierarchy representing the composite object
        """
        data = safe_load(yaml_string)
        data = self._resolve_params(data, params)
        return self._build_hierarchy(data)

    def _resolve_params(
        self, data: dict[str, Any], overrides: dict[str, float] | None
    ) -> dict[str, Any]:
        """Resolve `params:` + `{…}` interpolations throughout the YAML tree.

        The `params:` block itself and the `name` field are not interpolated
        (names are metadata, params are input). Everything else is walked
        recursively.
        """
        declared = data.get("params")
        resolved_params = resolve_params(declared, overrides)

        out: dict[str, Any] = {}
        for key, value in data.items():
            if key in ("params", "name"):
                out[key] = value
                continue
            out[key] = resolve_value(value, resolved_params)
        return out

    def _build_hierarchy(self, data: dict[str, Any]) -> SceneNode:
        """Build scene hierarchy from parsed YAML data."""
        name = data.get("name", "composite")
        if "floorplan" in data:
            return self._build_floorplan(name, data)
        if "building" in data:
            from ..generators.building import build_building

            root = build_building(data["building"], name, self._material_loader,
                                  assets_dir=Path(__file__).parents[3] / "assets", layout_loader=self)
            root.tags = [*root.tags, *data.get("tags", [])]
            return root
        container_size = np.array(data["size"], dtype=np.float64)
        logger.debug("Building hierarchy: %s (size=%s)", name, container_size)

        root = SceneNode(name)
        root.size = container_size
        root.tags = list(data.get("tags", []))
        # Furniture: footprint (x, z) and the free space needed around it
        # (front = +Z) for furnishing solvers and navigation.
        if "clearance" in data:
            clearance = data["clearance"] or {}
            unknown = set(clearance) - {"front", "back", "left", "right"}
            if unknown:
                raise ValueError(f"'{name}': clearance sides must be front/back/left/right, got {sorted(unknown)}")
            root.meta["footprint"] = [float(container_size[0]), float(container_size[2])]
            root.meta["clearance"] = {k: float(v) for k, v in clearance.items()}

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
            if "size" in part_def:
                frac_size = np.array(part_def["size"], dtype=np.float64)
                actual_size = frac_size * container_size
            else:  # sweeps: the path defines the size
                actual_size = np.zeros(3)

            generator = self._create_generator(primitive_type, actual_size, part_def)
            node = generator.to_node(part_name)
            if "size" not in part_def and node.mesh is not None:
                actual_size = node.mesh.vertices.max(axis=0) - node.mesh.vertices.min(axis=0)

            # Apply material if specified
            material_name = part_def.get("material")
            if material_name is not None and node.mesh is not None:
                try:
                    material = self._material_loader.load(material_name)
                    node.mesh.material = material
                except FileNotFoundError:
                    warnings.warn(
                        f"Material '{material_name}' not found for part '{part_name}'",
                        stacklevel=2,
                    )

            # Store the node's actual size for attachment calculations
            node.size = actual_size
            node.tags = [*node.tags, *[t for t in part_def.get("tags", []) if t not in node.tags]]
            collider = part_def.get("collider")
            if collider is not None:
                if collider not in COLLIDER_TYPES:
                    raise ValueError(
                        f"Part '{part_name}' collider must be one of {sorted(COLLIDER_TYPES)}, got {collider!r}"
                    )
                node.meta["collider"] = collider
            if part_def.get("walkable"):
                node.meta["walkable"] = True

            # Generate automatic attachment points from the generator
            auto_attachments = generator.get_attachment_points(actual_size)
            for attach_name, attach_point in auto_attachments.items():
                node.attachments[attach_name] = attach_point

            # Generate automatic surfaces from the generator
            auto_surfaces = generator.get_surfaces(actual_size)
            for surf_name, surf in auto_surfaces.items():
                surf.source = node
                node.surfaces[surf_name] = surf

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
                    import difflib
                    suggestion = difflib.get_close_matches(parent_name, part_nodes.keys(), n=1, cutoff=0.5)
                    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
                    raise ValueError(
                        f"Part '{part_name}' cannot attach to unknown part '{parent_name}'.{hint}"
                        f" Available parts: {sorted(part_nodes.keys())}"
                    )

                parent_node = part_nodes[parent_name]
                logger.debug("Attaching '%s' to '%s' at '%s'", part_name, parent_name, attach_point)

                # Get the attachment point on the parent
                if attach_point not in parent_node.attachments:
                    import difflib
                    available = sorted(parent_node.attachments.keys())
                    suggestion = difflib.get_close_matches(attach_point, available, n=1, cutoff=0.5)
                    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
                    raise ValueError(
                        f"Attachment point '{attach_point}' not found on '{parent_name}'.{hint}"
                        f" Available: {available}"
                    )

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

                # Apply additional offset if specified (in world units)
                extra_offset = part_def.get("offset")
                if extra_offset is not None:
                    node.transform.translation += np.array(extra_offset, dtype=np.float64)

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

                # Uncentred sweeps keep their path in the asset's own frame.
                if part_def["primitive"] == "sweep" and part_def.get("center") is False:
                    position = offset_world

                node.transform.translation = position

                # Apply rotation if specified
                rotation = part_def.get("rotation")
                if rotation is not None:
                    rotation_rad = np.radians(np.array(rotation, dtype=np.float64))
                    node.transform.rotation = rotation_rad

                root.add_child(node)

        self._apply_booleans(parts, part_nodes, root)

        # Parse explicit attachment points for the composite object
        attachments_data = data.get("attachments", {})
        for attach_name, attach_def in attachments_data.items():
            attachment = parse_attachment(attach_name, attach_def)
            root.attachments[attach_name] = attachment

        # Parse surface re-exports ("surfaces:" at asset root).
        # Each entry re-exports a part's surface under a (possibly renamed)
        # name on the root node, so scenes can address house.north_wall
        # rather than house.shell.north_wall.
        self._resolve_surface_exports(data.get("surfaces", {}), root, part_nodes)

        if data.get("interactions"):
            from .interactions import apply_state, parse_interactions

            root.interactions = parse_interactions(data["interactions"], root, part_nodes)
            for interaction in root.interactions:
                apply_state(root, interaction, interaction.initial)

        return root

    def _resolve_surface_exports(
        self,
        surfaces_data: dict[str, Any],
        root: SceneNode,
        part_nodes: dict[str, SceneNode],
    ) -> None:
        """Copy surfaces from part nodes onto the root node.

        Transforms the surface into the root's local frame so
        root.get_surface() resolves correctly. YAML format:

            surfaces:
              north_wall: { from: shell.north_wall }
              floor: { from: shell.floor }
              front_wall: { from: walls.front, cut: [lining] }  # openings also cut 'lining'
              back_wall:                                          # and get plastered reveals
                from: walls.back
                cut: [lining]
                reveal: { material: wall_plaster, thickness: 0.015 }
        """
        from .surfaces import Surface

        for export_name, spec in surfaces_data.items():
            if not isinstance(spec, dict) or "from" not in spec:
                raise ValueError(
                    f"Surface export '{export_name}' must specify 'from: <part>.<surface>'"
                )
            source = spec["from"]
            if "." not in source:
                raise ValueError(
                    f"Surface export '{export_name}' source must be 'part.surface', got {source!r}"
                )
            part_name, surface_name = source.split(".", 1)
            if part_name not in part_nodes:
                raise ValueError(
                    f"Surface export '{export_name}' references unknown part '{part_name}'."
                    f" Available parts: {sorted(part_nodes)}"
                )
            part_node = part_nodes[part_name]
            if surface_name not in part_node.surfaces:
                raise ValueError(
                    f"Surface export '{export_name}' references unknown surface"
                    f" '{surface_name}' on part '{part_name}'."
                    f" Available: {sorted(part_node.surfaces)}"
                )
            src = part_node.surfaces[surface_name]

            also_cut = []
            for cut_name in spec.get("cut", []):
                if cut_name not in part_nodes:
                    raise ValueError(
                        f"Surface export '{export_name}' cuts unknown part '{cut_name}'."
                        f" Available parts: {sorted(part_nodes)}"
                    )
                also_cut.append(part_nodes[cut_name])

            reveal = None
            if "reveal" in spec:
                reveal_spec = spec["reveal"]
                if not isinstance(reveal_spec, dict) or "material" not in reveal_spec:
                    raise ValueError(
                        f"Surface export '{export_name}' reveal must be"
                        " {material: <name>, thickness: <m>}"
                    )
                reveal = {
                    "material": self._material_loader.load(reveal_spec["material"]),
                    "thickness": float(reveal_spec.get("thickness", 0.015)),
                }

            # Transform surface from part's local frame into root's local frame.
            # Surfaces store positions (origin) and directions (axes, normal).
            part_matrix = part_node.transform.to_matrix()
            rotation = part_matrix[:3, :3]
            translation = part_matrix[:3, 3]

            root.surfaces[export_name] = Surface(
                name=export_name,
                origin=rotation @ src.origin + translation,
                u_axis=rotation @ src.u_axis,
                v_axis=rotation @ src.v_axis,
                normal=rotation @ src.normal,
                u_extent=src.u_extent,
                v_extent=src.v_extent,
                source=part_node,
                also_cut=also_cut,
                reveal=reveal,
            )

    def _build_floorplan(self, name: str, data: dict[str, Any]) -> SceneNode:
        """Build an asset whose geometry is a ``floorplan:`` (rooms + walls).

        The plan is centred on its bounds at floor level (origin bottom_center).
        Surfaces are exported at the root as ``<room>.<surface>``, e.g.
        ``on: suite.bedroom.north_wall`` or ``suite.bedroom.south_exterior``.
        """
        from ..generators.floorplan import FloorPlan

        if data.get("parts"):
            raise ValueError(f"'{name}': a floorplan asset can't also define parts")
        plan = FloorPlan.from_spec(data["floorplan"])
        root = plan.build(name, self._material_loader)
        root.tags = [*root.tags, *data.get("tags", [])]
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
                    warnings.warn(
                        f"Floor material '{materials_config['floor']}' not found",
                        stacklevel=2,
                    )

            if "walls" in materials_config:
                try:
                    wall_material = self._material_loader.load(materials_config["walls"])
                except FileNotFoundError:
                    warnings.warn(
                        f"Wall material '{materials_config['walls']}' not found",
                        stacklevel=2,
                    )

            if "ceiling" in materials_config:
                try:
                    ceiling_material = self._material_loader.load(materials_config["ceiling"])
                except FileNotFoundError:
                    warnings.warn(
                        f"Ceiling material '{materials_config['ceiling']}' not found",
                        stacklevel=2,
                    )

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
            config = extra_config or {}
            return CubeGenerator(
                size_x=size[0], size_y=size[1], size_z=size[2],
                bevel=config.get("bevel", 0.02),
                bevel_segments=config.get("bevel_segments", 2),
            )
        elif primitive_type == "cylinder":
            # Cylinder uses radius (half of x/z) and height
            radius = min(size[0], size[2]) / 2
            bevel = float((extra_config or {}).get("bevel", 0.0))
            if bevel > 0:  # rounded rims (cylinders are sharp by default)
                return BevelledCylinderGenerator(radius=radius, height=size[1], bevel=bevel,
                                                 bevel_segments=int((extra_config or {}).get("bevel_segments", 4)))
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
        elif primitive_type == "plane":
            # Plane uses width (X) and depth (Z), ignores Y
            # Get subdivisions from extra_config if available
            subdivisions_x = (extra_config or {}).get("subdivisions_x", 1)
            subdivisions_z = (extra_config or {}).get("subdivisions_z", 1)
            return PlaneGenerator(
                size_x=size[0],
                size_z=size[2],
                subdivisions_x=subdivisions_x,
                subdivisions_z=subdivisions_z,
            )
        elif primitive_type == "room":
            return self._create_room_generator(size, extra_config or {})
        elif primitive_type == "ellipsoid":
            return EllipsoidGenerator(size_x=size[0], size_y=size[1], size_z=size[2])
        elif primitive_type == "extrude":
            return self._create_extrude_generator(size, extra_config or {})
        elif primitive_type == "lathe":
            return self._create_lathe_generator(size, extra_config or {})
        elif primitive_type == "sweep":
            return self._create_sweep_generator(extra_config or {})
        elif primitive_type == "torus":
            return TorusGenerator(size_x=size[0], size_y=size[1], size_z=size[2],
                                  tube=float((extra_config or {}).get("tube", 0.0)))
        elif primitive_type == "capsule":
            return CapsuleGenerator(size_x=size[0], size_y=size[1], size_z=size[2])
        elif primitive_type == "stairs":
            config = extra_config or {}
            keys = ("style", "rise", "width", "max_riser", "tread", "turn", "landing_at", "waist", "railing",
                    "rail_height", "railing_material")
            args = {k: config[k] for k in keys if k in config}
            for k in ("rise", "width", "max_riser", "tread", "landing_at", "waist", "rail_height"):
                if k in args:
                    args[k] = float(args[k])
            return StairsGenerator(**args)
        elif primitive_type == "roof":
            config = extra_config or {}
            return RoofGenerator(
                width=size[0], height=size[1], depth=size[2],
                style=config.get("style", "gable"),
                overhang=float(config.get("overhang", 0.35)),
                thickness=float(config.get("thickness", 0.12)),
                ridge_axis=config.get("ridge_axis", "auto"),
                ridge_cap=bool(config.get("ridge_cap", True)),
            )
        elif primitive_type == "prism":
            return PrismGenerator(
                width=size[0], height=size[1], depth=size[2],
                apex=(extra_config or {}).get("apex", "center"),
            )
        else:
            import difflib
            known = list(PRIMITIVE_REGISTRY.keys())
            suggestion = difflib.get_close_matches(primitive_type, known, n=1, cutoff=0.5)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            raise ValueError(
                f"Unknown primitive type: '{primitive_type}'.{hint}"
                f" Available: {known}"
            )

    @staticmethod
    def _apply_booleans(parts: dict[str, Any], part_nodes: dict[str, SceneNode], root: SceneNode) -> None:
        """Cut ``subtract:`` parts out of their targets, then drop cutter parts.

        Cutters are positioned like any other part (anchor/offset or
        attach_to), so openings can be laid out semantically; the boolean
        happens in the target's local frame.

        Parts with ``cut_host: true`` are cutters for the *host* this asset
        gets placed on (e.g. a window's opening): they are removed from the
        asset and stored on ``root.host_cutters`` for the scene composer.
        Parts with ``reveal: true`` (or ``reveal: {bottom: false}``) mark the
        box behind the frame whose sides the host lines when its surface
        declares a ``reveal`` finish; they go to ``root.host_reveals``.
        """
        from ..core import csg

        for part_name, part_def in parts.items():
            cutter_names = part_def.get("subtract")
            if not cutter_names:
                continue
            if isinstance(cutter_names, str):
                cutter_names = [cutter_names]
            target = part_nodes[part_name]
            to_local = np.linalg.inv(target.world_transform())
            cutters = []
            for name in cutter_names:
                if name not in part_nodes:
                    raise ValueError(f"Part '{part_name}' subtracts unknown part '{name}'")
                cutter = part_nodes[name]
                cutters.append(cutter.mesh.transform(to_local @ cutter.world_transform()))
            try:
                target.mesh = csg.difference(target.mesh, *cutters)
            except csg.CSGError as exc:
                raise ValueError(f"Boolean on part '{part_name}' failed: {exc}") from exc

        root_inv = np.linalg.inv(root.world_transform())
        for part_name, part_def in parts.items():
            if part_def.get("cut_host"):
                node = part_nodes[part_name]
                root.host_cutters.append(node.mesh.transform(root_inv @ node.world_transform()))
            reveal = part_def.get("reveal")
            if reveal:
                node = part_nodes[part_name]
                line_bottom = bool(reveal.get("bottom", True)) if isinstance(reveal, dict) else True
                root.host_reveals.append((node.mesh.transform(root_inv @ node.world_transform()), line_bottom))
            if part_def.get("cutter") or part_def.get("cut_host") or reveal:
                node = part_nodes[part_name]
                if node.children:
                    raise ValueError(f"Cutter part '{part_name}' cannot have attached children")
                if node.parent is not None:
                    node.parent.remove_child(node)

    @staticmethod
    def _create_extrude_generator(size: np.ndarray, config: dict[str, Any]) -> ExtrudeGenerator:
        """Extrude ``config['shape']`` so it fills the part's size.

        With ``fit: stretch`` (default) the profile is scaled to the part's
        extent in the profile plane; with ``fit: none`` the profile is taken
        in metres. Either way it is centred on the part origin.
        """
        if "shape" not in config:
            raise ValueError("extrude parts require a 'shape' (e.g. shape: {rect: [1, 1], radius: 0.1})")
        axis = config.get("axis", "y")
        if axis not in _AXIS_FRAMES:
            raise ValueError(f"extrude axis must be one of {sorted(_AXIS_FRAMES)}, got '{axis}'")
        u, v, a = _AXIS_FRAMES[axis]
        shape = shape_from_spec(config["shape"])
        lo, hi = shape.bounds
        center = (lo + hi) / 2
        scale = np.ones(2)
        if config.get("fit", "stretch") == "stretch":
            target = np.array([float(np.abs(u) @ size), float(np.abs(v) @ size)])
            scale = target / np.maximum(hi - lo, 1e-12)
        shape = Shape((shape.outer - center) * scale, [(h - center) * scale for h in shape.holes])
        return ExtrudeGenerator(
            shape=shape,
            depth=float(np.abs(a) @ size),
            axis=axis,
            bevel=float(config.get("bevel", 0.0)),
            bevel_segments=int(config.get("bevel_segments", 3)),
            crease_angle=float(config.get("crease_angle", 40.0)),
            caps=bool(config.get("caps", True)),
        )

    @staticmethod
    def _create_sweep_generator(config: dict[str, Any]) -> SweepGenerator:
        """Sweep ``config['profile']`` (a shape spec, metres) along ``config['path']``.

        ``path`` is a list of [x, y, z] points or ``{spline: [...], samples: n}``
        (Catmull-Rom through the points). The mesh is centred on its bounds
        like other primitives unless ``center: false``, in which case the path
        is in the asset's frame (anchor ignored, ``offset`` still applies).
        """
        from ..core.profile import catmull_rom

        if "profile" not in config or "path" not in config:
            raise ValueError("sweep parts require 'profile' (a shape) and 'path' ([[x, y, z], ...])")
        closed = bool(config.get("closed", False))
        path_spec = config["path"]
        if isinstance(path_spec, dict):
            if "spline" not in path_spec:
                raise ValueError("sweep path mapping must be {spline: [[x, y, z], ...], samples: n}")
            path = catmull_rom(np.asarray(path_spec["spline"], dtype=np.float64),
                               int(path_spec.get("samples", 8)), closed=closed)
        else:
            path = np.asarray(path_spec, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError(f"sweep path must be a list of [x, y, z] points, got shape {path.shape}")
        scale = config.get("scale", [1.0, 1.0])
        scale = (float(scale), float(scale)) if isinstance(scale, (int, float)) else tuple(float(v) for v in scale)
        return SweepGenerator(
            profile=shape_from_spec(config["profile"]),
            path=path,
            closed=closed,
            up=tuple(float(v) for v in config.get("up", (0.0, 1.0, 0.0))),
            twist=float(config.get("twist", 0.0)),
            scale=scale,
            crease_angle=float(config.get("crease_angle", 40.0)),
            center=bool(config.get("center", True)),
        )

    @staticmethod
    def _create_lathe_generator(size: np.ndarray, config: dict[str, Any]) -> LatheGenerator:
        """Revolve ``config['profile']`` ((r, y) points) to fill the part's size."""
        if "profile" not in config:
            raise ValueError("lathe parts require a 'profile' list of [radius, y] points")
        prof = polyline_from_spec(config["profile"]).copy()
        if config.get("fit", "stretch") == "stretch":
            radius = min(size[0], size[2]) / 2
            prof[:, 0] *= radius / max(prof[:, 0].max(), 1e-12)
            y_range = max(np.ptp(prof[:, 1]), 1e-12)
            prof[:, 1] = (prof[:, 1] - prof[:, 1].min()) * (size[1] / y_range)
        prof[:, 1] -= (prof[:, 1].min() + prof[:, 1].max()) / 2
        return LatheGenerator(
            profile=prof,
            segments=int(config.get("segments", 48)),
            sweep=float(config.get("sweep", 360.0)),
            cap_bottom=bool(config.get("cap_bottom", True)),
            cap_top=bool(config.get("cap_top", True)),
            crease_angle=float(config.get("crease_angle", 40.0)),
        )

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
