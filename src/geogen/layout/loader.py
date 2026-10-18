"""YAML loader for composite object definitions."""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .. import roots
from ..core.node import SceneNode
from ..core.profile import Shape, polyline_from_spec, shape_from_spec
from ..generators.crops import CropGenerator
from ..generators.nature import RockGenerator, TreeGenerator
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
from ..generators.text import TextGenerator
from ..generators.paths import PathsGenerator
from ..generators.fence import FenceGenerator
from ..generators.terrain import MATERIAL_SLOTS as TERRAIN_SLOTS
from ..generators.terrain import TerrainGenerator
from ..generators.round_shapes import BevelledCylinderGenerator, CapsuleGenerator, TorusGenerator
from ..generators.stairs import StairsGenerator
from ..generators.sweep import SweepGenerator
from ..materials.loader import MaterialLoader
from .anchors import resolve_anchor
from .attachments import parse_attachment
from .expressions import enabled, resolve_document
from .validation import suggest, validate_asset_yaml, pre_validate_asset_references
from .yaml_utils import safe_load, safe_load_path

logger = logging.getLogger("geogen.layout")


# Primitives whose points are already in the asset's frame: placed at their offset, not anchored.
_IN_ASSET_FRAME = ("paths", "fence", "terrain")

# Collision shapes a part can request (``collider:``); see export.py.
COLLIDER_TYPES = {"auto", "none", "box", "hull", "mesh"}

# Registry of available primitive generators
PRIMITIVE_REGISTRY = {
    "cube": CubeGenerator,
    "cylinder": CylinderGenerator,
    "sphere": SphereGenerator,
    "cone": ConeGenerator,
    "plane": PlaneGenerator,
    "ellipsoid": EllipsoidGenerator,
    "extrude": ExtrudeGenerator,
    "lathe": LatheGenerator,
    "roof": RoofGenerator,
    "prism": PrismGenerator,
    "sweep": SweepGenerator,
    "stairs": StairsGenerator,
    "torus": TorusGenerator,
    "capsule": CapsuleGenerator,
    "tree": TreeGenerator,
    "crop": CropGenerator,
    "rock": RockGenerator,
    "text": TextGenerator,
    "paths": PathsGenerator,
    "fence": FenceGenerator,
    "terrain": TerrainGenerator,
    "water": ExtrudeGenerator,
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

    def __init__(self, detail: float = 1.0, material_loader: MaterialLoader | None = None) -> None:
        """Initialize the layout loader with a material loader (a new one by default).

        ``detail`` scales the tessellation of curved primitives (segments,
        rings, tube/cap/bevel steps): 0.5 halves them for a cheap level of
        detail, 2 doubles them for close-ups.
        """
        self.material_loader = material_loader or MaterialLoader()
        self.detail = float(detail)
        self._prototypes: dict[tuple[str, str], SceneNode] = {}

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
        # Identical loads (same file and params) return instances of one
        # prototype, so repeated furniture shares meshes (see SceneNode.instance).
        key = (str(Path(path).resolve()), _params_key(params))
        prototype = self._prototypes.get(key)
        if prototype is None:
            prototype = self._load(Path(path), params)
            self._prototypes[key] = prototype
        return prototype.instance()

    def _load(self, path: Path, params: dict[str, float] | None) -> SceneNode:
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
        root = self._build_hierarchy(data)
        _add_places(root, data, path.name)
        from ..state import declare_asset

        declare_asset(root, data, path)
        from ..wiring import declare as declare_wiring

        declare_wiring(root, data, path, asset=True)
        if data.get("readable"):
            from ..dialogue import parse_readable

            parse_readable(data["readable"], root, path.name)
        if data.get("pickup"):
            from ..inventory import parse_pickup


            parse_pickup(data["pickup"], root, path.name, roots.root_of(path), path)
        # Where it came from, for --inspect (not exported).
        root.meta["source"] = {"asset": _asset_ref(path), "params": dict(params or {})}
        return root

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
        root = self._build_hierarchy(data)
        _add_places(root, data, str(data.get("name", "asset")))
        return root

    def _resolve_params(
        self, data: dict[str, Any], overrides: dict[str, float] | None
    ) -> dict[str, Any]:
        """Resolve `params:`, presets, derived values and `{…}` interpolations (expressions.resolve_document)."""
        out, resolved_params, _ = resolve_document(data, overrides)
        if "body" in data:
            out["_params"] = resolved_params      # garments resolve against the body's params
        return out

    def build(self, data: dict[str, Any]) -> SceneNode:
        """Build an asset from an already-resolved document (parts, floorplan, building or body)."""
        return self._build_hierarchy(data)

    def _build_hierarchy(self, data: dict[str, Any]) -> SceneNode:
        """Build scene hierarchy from parsed YAML data."""
        name = data.get("name", "composite")
        if "floorplan" in data:
            return self._build_floorplan(name, data)
        if "body" in data:
            from ..generators.humanoid import build_humanoid, fit_seated_pose
            from ..npc import parse_poses

            root_poses = parse_poses(data["poses"]) if data.get("poses") else None
            root = build_humanoid(data["body"], name, self.material_loader,
                                  assets_dir=roots.CORE, params=data.get("_params"),
                                  root_poses=root_poses)
            root.tags = [*root.tags, *data.get("tags", [])]
            if data.get("preset"):
                root.meta["preset"] = data["preset"]
            if root_poses:
                root.meta["poses"] = root_poses
                fit_seated_pose(root, root.meta["poses"])
            if data.get("affordances"):
                root.meta["affordances"] = [_affordance(a, root) for a in data["affordances"]]
            return root
        if "building" in data:
            from ..generators.building import build_building

            root = build_building(data["building"], name, self.material_loader,
                                  assets_dir=roots.CORE, layout_loader=self)
            root.tags = [*root.tags, *data.get("tags", [])]
            return root
        container_size = np.array(data["size"], dtype=np.float64)
        logger.debug("Building hierarchy: %s (size=%s)", name, container_size)

        root = SceneNode(name)
        root.size = container_size
        root.tags = list(data.get("tags", []))
        # `when: "{expr}"` drops a part (and parts attached to it) when it resolves to 0 / false.
        kept = _enabled_parts(data.get("parts", {}))
        arrays = _array_copies(kept)
        parts = _expand_arrays(kept)

        # First pass: build every part; second: position each (attached, or anchored in the container).
        part_nodes: dict[str, SceneNode] = {}
        part_sizes: dict[str, np.ndarray] = {}
        for part_name, part_def in parts.items():
            part_nodes[part_name], part_sizes[part_name] = self._build_part(part_name, part_def, container_size)
        for part_name, part_def in parts.items():
            self._place_part(part_name, part_def, part_nodes, part_sizes[part_name], root, container_size)

        self._apply_booleans(parts, part_nodes, root)
        _apply_tints(parts, part_nodes)

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

        interactions = dict(data.get("interactions") or {})
        interactions.update(_joint_interactions(parts))
        if interactions:
            from .interactions import apply_state, parse_interactions

            root.interactions = parse_interactions(interactions, root, part_nodes, arrays)
            for interaction in root.interactions:
                apply_state(root, interaction, interaction.initial)

        for key, handler in _ROOT_META:
            if data.get(key) or (key == "clearance" and key in data):   # `clearance: {}` still records the footprint
                handler(root, data[key] or {}, part_nodes, name)

        if data.get("bounds") == "geometry":
            # The container is only a unit for part sizes (e.g. [scale, scale,
            # scale] with parts in metres): report the real extent instead.
            extent = _geometry_extent(root)
            if extent is not None:
                root.size = extent
        return root

    def _build_part(self, part_name: str, part_def: dict[str, Any],
                    container_size: np.ndarray) -> tuple[SceneNode, np.ndarray]:
        """One part's node (mesh, material, tags, collider, attachments, surfaces) and its size in metres."""
        primitive_type = part_def["primitive"]
        if "size" in part_def:
            frac_size = np.array(part_def["size"], dtype=np.float64)
            actual_size = frac_size * container_size
        else:  # sweeps: the path defines the size
            actual_size = np.zeros(3)

        generator = self._create_generator(primitive_type, actual_size, part_def)
        _cap_segments(generator, primitive_type, actual_size)
        if self.detail != 1.0:
            _apply_detail(generator, self.detail)
        node = generator.to_node(part_name)
        if node.mesh is not None and ("subdivide" in part_def or "displace" in part_def):
            node.mesh = _smooth_part(node.mesh, part_def, self.detail, part_name)
        if "size" not in part_def and node.mesh is not None:
            actual_size = node.mesh.vertices.max(axis=0) - node.mesh.vertices.min(axis=0)
        post = _POST_BUILD.get(primitive_type)
        if post is not None:
            post(self, node, generator, part_def, actual_size)

        # Apply material if specified
        material_name = part_def.get("material")
        if material_name is not None and node.mesh is not None:
            try:
                material = self.material_loader.load(material_name)
                node.mesh.material = material
                # Generated child meshes (a tree's branches) inherit the part's material.
                for child in node.iter_nodes():
                    if child.mesh is not None and child.mesh.material is None:
                        child.mesh.material = material
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
        return node, actual_size

    def _place_part(self, part_name: str, part_def: dict[str, Any], part_nodes: dict[str, SceneNode],
                    actual_size: np.ndarray, root: SceneNode, container_size: np.ndarray) -> None:
        """Position a built part: on another part's attachment point (``attach_to`` + ``at`` /
        ``from``), or at an anchor of the container plus an ``offset`` in fractions of it."""
        node = part_nodes[part_name]
        if "attach_to" in part_def:
            # Hierarchical attachment mode
            parent_name = part_def["attach_to"]
            attach_point = part_def.get("at", "top")
            child_attach = part_def.get("from", "bottom")  # Which point on child to attach

            if parent_name not in part_nodes:
                hint = suggest(parent_name, part_nodes.keys())
                raise ValueError(
                    f"Part '{part_name}' cannot attach to unknown part '{parent_name}'.{hint}"
                    f" Available parts: {sorted(part_nodes.keys())}"
                )

            parent_node = part_nodes[parent_name]
            logger.debug("Attaching '%s' to '%s' at '%s'", part_name, parent_name, attach_point)

            # Get the attachment point on the parent
            if attach_point not in parent_node.attachments:
                available = sorted(parent_node.attachments.keys())
                hint = suggest(attach_point, available)
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

            # Uncentred sweeps and path networks keep their points in the asset's own frame.
            if (part_def["primitive"] == "sweep" and part_def.get("center") is False) \
                    or part_def["primitive"] in ("paths", "fence", "terrain"):
                position = offset_world

            node.transform.translation = position

            # Apply rotation if specified
            rotation = part_def.get("rotation")
            if rotation is not None:
                rotation_rad = np.radians(np.array(rotation, dtype=np.float64))
                node.transform.rotation = rotation_rad

            root.add_child(node)

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
              front_inside: { from: lining_void.front, flip: true }  # seen from the other side
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
                    "material": self.material_loader.load(reveal_spec["material"]),
                    "thickness": float(reveal_spec.get("thickness", 0.015)),
                }

            # Transform surface from part's local frame into root's local frame.
            # Surfaces store positions (origin) and directions (axes, normal).
            part_matrix = part_node.transform.to_matrix()
            rotation = part_matrix[:3, :3]
            translation = part_matrix[:3, 3]

            # A flipped surface is the same region seen from behind: it faces
            # the other way and u still runs left to right for whoever faces it.
            origin, u_axis, normal = src.origin, src.u_axis, src.normal
            if spec.get("flip"):
                origin = origin + src.u_extent * u_axis
                u_axis, normal = -u_axis, -normal

            root.surfaces[export_name] = Surface(
                name=export_name,
                origin=rotation @ origin + translation,
                u_axis=rotation @ u_axis,
                v_axis=rotation @ src.v_axis,
                normal=rotation @ normal,
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
        root = plan.build(name, self.material_loader)
        root.tags = [*root.tags, *data.get("tags", [])]
        return root

    def _create_generator(
        self, primitive_type: str, size: np.ndarray, extra_config: dict[str, Any] | None = None
    ):
        """The generator for one part: ``_BUILDERS[primitive_type](loader, size, config)``.

        ``size`` is the part's size in metres; ``extra_config`` the part's YAML keys.
        """
        builder = _BUILDERS.get(primitive_type)
        if builder is None:
            raise ValueError(f"Unknown primitive type: '{primitive_type}'.{suggest(primitive_type, list(_BUILDERS))}"
                             f" Available: {list(_BUILDERS)}")
        return builder(self, size, extra_config or {})

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
            level=bool(config.get("level", False)),
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
        prof = _simplify_profile(prof)
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

def _joint_interactions(parts: dict[str, Any]) -> dict[str, Any]:
    """Expand part ``joint:`` shorthands into interactions.

    ``joint: {type: hinge, pivot: left, limits: [0, 100]}`` or
    ``joint: {type: slide, axis: z, range: [0, 0.4]}`` on a part makes a
    two-state (closed/open) interaction named after the part that moves it
    (and its ``with: [other parts]``). ``pivot`` is an anchor on the part
    (``left`` = its -X face centre, ``right``, ``top``, ``bottom``, ...) or
    ``[x, y, z]`` in the asset frame. Without ``axis``, side and top/bottom
    pivots pick the axis that swings the part out toward +Z (its front).
    """
    anchors = {"left": "left_center", "right": "right_center", "front": "front_center", "back": "back_center",
               "top": "top_center", "bottom": "bottom_center", "center": "center"}
    result: dict[str, Any] = {}
    for name, part in parts.items():
        joint = part.get("joint") if isinstance(part, dict) else None
        if not joint:
            continue
        kind = joint.get("type", "hinge")
        if kind not in ("hinge", "slide"):
            raise ValueError(f"Part '{name}' joint type must be hinge or slide, got {kind!r}")
        lo, hi = joint.get("limits", [0, 90]) if kind == "hinge" else joint.get("range", [0, 0.3])
        pivot = joint.get("pivot", "left")
        # Default hinge axes swing the part out toward +Z (its front) about a side pivot.
        default_axis = {"left": "-y", "right": "y", "top": "x", "bottom": "-x"}.get(pivot, "y") \
            if kind == "hinge" else "z"
        motion: dict[str, Any] = {
            "parts": [name, *joint.get("with", [])],
            ("rotate" if kind == "hinge" else "translate"): joint.get("axis", default_axis),
            "values": {"closed": float(lo), "open": float(hi)},
        }
        if kind == "hinge":
            motion["pivot"] = f"{name}.{anchors.get(pivot, pivot)}" if isinstance(pivot, str) else pivot
        result[joint.get("name", name)] = {
            "targets": [name, *joint.get("with", [])],
            "initial": joint.get("initial", "closed"),
            "duration": float(joint.get("duration", 0.8 if kind == "hinge" else 0.4)),
            "states": {"closed": {"prompt": joint.get("open_prompt", "Open"), "next": "open"},
                       "open": {"prompt": joint.get("close_prompt", "Close"), "next": "closed",
                                **({"then": "closed", "after": float(joint["auto_close"])}
                                   if joint.get("auto_close") else {})}},
            "motions": [motion],
            **({"lock": joint["lock"]} if joint.get("lock") else {}),
        }
    return result


def _geometry_extent(root: SceneNode) -> np.ndarray | None:
    """Size of the axis-aligned box around all meshes, in ``root``'s frame."""
    inv = np.linalg.inv(root.world_transform())
    pts = [(inv @ n.world_transform() @ np.c_[n.mesh.vertices, np.ones(len(n.mesh.vertices))].T).T[:, :3]
           for n in root.iter_nodes() if n.mesh is not None and len(n.mesh.vertices)]
    if not pts:
        return None
    allp = np.vstack(pts)
    return allp.max(axis=0) - allp.min(axis=0)


LIGHT_DEFAULTS = {"type": "omni", "color": [1.0, 0.93, 0.82], "energy": 1.0, "range": 5.0, "offset": [0.0, 0.0, 0.0],
                  "auto": None}
LIGHT_AUTO = {None, "night"}     # night: on after dark (runtimes' world clock), no switch


def light_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalise an asset's ``light:`` block (omni light; ``offset`` metres from the asset origin)."""
    unknown = set(spec) - set(LIGHT_DEFAULTS)
    if unknown:
        raise ValueError(f"light: unknown keys {sorted(unknown)}; known: {sorted(LIGHT_DEFAULTS)}")
    light = {**LIGHT_DEFAULTS, **spec}
    if light["type"] != "omni":
        raise ValueError("light: only type omni is supported")
    offset = light["offset"]
    light["offset"] = [0.0, float(offset), 0.0] if isinstance(offset, (int, float)) else [float(v) for v in offset]
    light["color"] = [float(c) for c in light["color"]]
    light["energy"], light["range"] = float(light["energy"]), float(light["range"])
    if light["auto"] not in LIGHT_AUTO:
        raise ValueError(f"light: auto must be one of {sorted(a for a in LIGHT_AUTO if a)}, got {light['auto']!r}")
    if light["auto"] is None:
        del light["auto"]
    return light


AFFORDANCE_TYPES = ("sit", "lie", "use", "stand", "look")
AFFORDANCE_KEYS = {"type", "at", "facing", "height", "prompt", "action", "approach", "duration",
                   "advertises", "tags", "slots", "interaction", "depth"}


def _apply_tints(parts: dict[str, Any], part_nodes: dict[str, SceneNode]) -> None:
    """``tint:`` is a vertex colour over the part's material (one light neutral material,
    many colours: car paint); generated children are tinted too. Runs after CSG."""
    from ..materials.loader import resolve_tint

    for name, part_def in parts.items():
        node = part_nodes.get(name)
        if part_def.get("tint") is None or node is None:
            continue
        rgba = [*resolve_tint(part_def["tint"]), 1.0]
        for child in node.iter_nodes():
            if child.mesh is not None:
                child.mesh.colors = np.tile(rgba, (len(child.mesh.vertices), 1))


MAX_SEAT_DEPTH = 0.9   # m: geometry further in front of a seat anchor is something else


def _seat_depth(root: SceneNode, position: np.ndarray, yaw_deg: float) -> float | None:
    """How far in front of a seat anchor the seat's front edge is: the furthest geometry at the
    seat's height (within 4 cm) and near the sitter's centre line, in the anchor's frame."""
    to_root = np.linalg.inv(root.world_transform())
    yaw = np.radians(yaw_deg)
    fwd = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
    right = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    best = None
    for n in root.iter_nodes():
        if n.mesh is None or not len(n.mesh.vertices):
            continue
        m = to_root @ n.world_transform()
        w = (m @ np.c_[n.mesh.vertices, np.ones(len(n.mesh.vertices))].T).T[:, :3] - position
        # Points across every face (a long cushion's vertices are all at its ends).
        tri = w[n.mesh.faces]
        g = np.linspace(0.0, 1.0, 6)
        uu, vv = (a.ravel() for a in np.meshgrid(g, g))
        keep = uu + vv <= 1.0
        uu, vv = uu[keep], vv[keep]
        v = (tri[:, None, 0] + (tri[:, None, 1] - tri[:, None, 0]) * uu[None, :, None]
             + (tri[:, None, 2] - tri[:, None, 0]) * vv[None, :, None]).reshape(-1, 3)
        # Only this seat: another seat facing it at the same height (a picnic table's far bench) is
        # further ahead than any seat is deep.
        near = (np.abs(v[:, 1]) < 0.04) & (np.abs(v @ right) < 0.3) & (v @ fwd < MAX_SEAT_DEPTH)
        if near.any():
            z = float((v[near] @ fwd).max())
            best = z if best is None else max(best, z)
    return best


def _array_copies(parts: dict[str, Any]) -> dict[str, list[str]]:
    """{arrayed part: the names of its copies}."""
    return {name: [f"{name}_{i}" for i in range(max(int(round(float(p["array"].get("count", 1)))), 0))]
            for name, p in parts.items() if p.get("array") is not None}


def _expand_arrays(parts: dict[str, Any]) -> dict[str, Any]:
    """``array: {count, step: [x, y, z]}`` repeats a part as ``<part>_0 .. <part>_<n-1>``,
    each ``step`` further along (the units of ``offset``: metres in a unit container).
    ``count`` may be an expression (rounded; 0 drops the part). Other parts can't attach to
    an arrayed part; a ``subtract:`` naming an arrayed cutter takes all its copies, and so does
    an interaction's ``parts`` / ``targets``."""
    out: dict[str, Any] = {}
    for name, part in parts.items():
        spec = part.get("array")
        if spec is None:
            out[name] = part
            continue
        unknown = set(spec) - {"count", "step"}
        if unknown:
            raise ValueError(f"Part '{name}' array: unknown keys {sorted(unknown)} (count, step)")
        if any(p.get("attach_to") == name for p in parts.values()):
            raise ValueError(f"Part '{name}' is arrayed: other parts can't attach_to it")
        count = int(round(float(spec.get("count", 1))))
        step = np.array(spec.get("step", [0, 0, 0]), dtype=np.float64)
        base = np.array(part.get("offset", [0, 0, 0]), dtype=np.float64)
        for i in range(max(count, 0)):
            copy = {k: v for k, v in part.items() if k != "array"}
            copy["offset"] = (base + step * i).tolist()
            out[f"{name}_{i}"] = copy
    # A subtract list naming an arrayed cutter means all of its copies.
    copies = _array_copies(parts)
    for part in out.values():
        if copies and part.get("subtract"):
            part["subtract"] = [c for s_ in part["subtract"] for c in copies.get(s_, [s_])]
    return out


def _enabled_parts(parts: dict[str, Any]) -> dict[str, Any]:
    """Parts whose ``when:`` is truthy (default), minus parts attached to dropped ones."""
    kept = {k: v for k, v in (parts or {}).items() if enabled(v.get("when", True))}
    changed = True
    while changed:
        changed = False
        for k, v in list(kept.items()):
            if v.get("attach_to") is not None and v["attach_to"] not in kept:
                del kept[k]
                changed = True
    out = {}
    for k, v in kept.items():
        v = {kk: vv for kk, vv in v.items() if kk != "when"}
        if "subtract" in v:
            v["subtract"] = [c for c in v["subtract"] if c in kept]
        out[k] = v
    return out


def _add_places(root: SceneNode, data: dict[str, Any], where: str) -> None:
    """An asset's ``places:``: trigger volumes in the asset frame (geogen.places)."""
    if not data.get("places"):
        return
    from .. import refs
    from ..places import build_places
    from .schema import PLACE_KEYS

    def around(place: str, ref: str) -> SceneNode:
        try:
            return refs.resolve(root, ref)
        except refs.RefError as exc:
            raise ValueError(f"{where}: place '{place}': around: {exc.message}") from None

    build_places(root, data["places"], where, resolve=around, known=set(PLACE_KEYS) - {"anchor", "offset"})


def _affordance(spec: dict[str, Any], root: SceneNode) -> dict[str, Any]:
    """Something an actor can do at an asset (the player and NPCs share these).

    ``{type: sit, at: <attachment> | [x, y, z], facing: <deg>, height: <m>}``:
    position/yaw are in the asset frame; for attachments the attachment's own
    facing is used unless ``facing`` (degrees about +Y, 0 = +Z) is given.
    ``height`` is the seat/mattress height the actor's hips rest at.

    For NPCs (see geogen.npc and assets/npcs/actions.yaml): ``action``
    (default: the type), ``approach`` (where to stand first: metres in front
    of the anchor or [x, y, z] in its frame; default from the action),
    ``duration`` (seconds or [lo, hi]), ``advertises`` ({need: amount}),
    ``tags``, ``slots`` (actors at once, default 1) and ``interaction`` (the
    asset interaction a ``use:`` step drives).
    """
    from ..npc import _range, approach_point, load_actions

    unknown = set(spec) - AFFORDANCE_KEYS
    if unknown:
        raise ValueError(f"affordance: unknown keys {sorted(unknown)}; known: {sorted(AFFORDANCE_KEYS)}")
    kind = spec.get("type")
    if kind not in AFFORDANCE_TYPES:
        raise ValueError(f"affordance type must be one of {AFFORDANCE_TYPES}, got {kind!r}")
    at = spec.get("at", [0.0, 0.0, 0.0])
    yaw = float(spec["facing"]) if "facing" in spec else 0.0
    if isinstance(at, str):
        transform = root.get_attachment(at)
        if transform is None:
            raise ValueError(f"affordance at unknown attachment '{at}'. Available: {root.list_attachments()}")
        m = np.linalg.inv(root.world_transform()) @ transform.to_matrix()
        position = m[:3, 3]
        if "facing" not in spec:
            yaw = float(np.degrees(np.arctan2(m[0, 2], m[2, 2])))
    else:
        position = np.asarray(at, dtype=np.float64)
    out = {"type": kind, "position": [round(float(v), 4) for v in position], "yaw": round(yaw, 3)}
    if kind == "sit":
        depth = spec.get("depth", _seat_depth(root, position, yaw))
        if depth is not None:
            out["depth"] = round(float(depth), 4)
    for key in ("height", "prompt"):
        if key in spec:
            out[key] = float(spec[key]) if key == "height" else str(spec[key])
    action = str(spec.get("action", kind))
    if action not in load_actions():
        raise ValueError(f"affordance action {action!r} is not in assets/npcs/actions.yaml")
    out["action"] = action
    out["approach"] = approach_point(action, position, yaw, spec.get("approach"))
    if "duration" in spec:
        out["duration"] = _range(spec["duration"], "affordance duration")
    if spec.get("advertises"):
        out["advertises"] = {str(k): float(v) for k, v in spec["advertises"].items()}
    if spec.get("tags"):
        out["tags"] = [str(t) for t in spec["tags"]]
    out["slots"] = int(spec.get("slots", 1))
    if "interaction" in spec:
        name = str(spec["interaction"])
        if not any(i.name == name for i in root.interactions):
            raise ValueError(f"affordance interaction {name!r} not on this asset")
        out["interaction"] = name
    return out


# Tessellation attributes scaled by LayoutLoader(detail=...) (minimums keep shapes valid).
_DETAIL_ATTRS = {"segments": 6, "rings": 4, "tube_segments": 6, "cap_segments": 3}


# Radial tessellation follows size: about one segment per SEGMENT_EDGE of circumference
# (never fewer than MIN_SEGMENTS), so a 6 cm chair leg isn't turned with 40 facets.
SEGMENT_EDGE = 0.025
MIN_SEGMENTS = 12
_ROUND_PRIMITIVES = {"cylinder", "cone", "sphere", "ellipsoid", "lathe", "torus", "capsule"}


def _simplify_profile(prof: np.ndarray) -> np.ndarray:
    """Drop lathe profile points that don't change the silhouette (finely sampled splines):
    Douglas-Peucker to 2% of the radius (thin spindles keep their swell), at most 2 mm."""
    from shapely.geometry import LineString

    if len(prof) < 4:
        return prof
    tol = min(0.02 * float(np.max(prof[:, 0])), 0.002)
    out = np.array(LineString(prof).simplify(tol, preserve_topology=False).coords)
    return out if len(out) >= 2 else prof


def _cap_segments(generator, primitive_type: str, size: np.ndarray) -> None:
    if primitive_type not in _ROUND_PRIMITIVES or not isinstance(getattr(generator, "segments", None), int):
        return
    if primitive_type == "lathe":
        radius = float(np.max(generator.profile[:, 0]))
    elif primitive_type == "sphere":
        radius = float(np.min(size)) / 2
    else:
        radius = float(max(size[0], size[2], size[1] if primitive_type == "ellipsoid" else 0.0)) / 2
    cap = max(MIN_SEGMENTS, int(np.ceil(2 * np.pi * radius / SEGMENT_EDGE)))
    if generator.segments > cap:
        generator.segments = cap
        if isinstance(getattr(generator, "rings", None), int):
            generator.rings = max(4, min(generator.rings, cap // 2))


def _apply_detail(generator, detail: float) -> None:
    for attr, minimum in _DETAIL_ATTRS.items():
        value = getattr(generator, attr, None)
        if isinstance(value, int):
            setattr(generator, attr, max(minimum, int(round(value * detail))))



def _smooth_part(mesh, part_def: dict, detail: float, part_name: str):
    """Apply a part's ``subdivide:`` / ``displace:`` (see core/subdiv.py), then metric UVs and normals.

    ``subdivide: 2`` or ``{levels: 2, crease: 60}`` (edges sharper than
    ``crease`` degrees stay creased; omitted = fully smooth); the smoothed
    mesh is stretched back to fill the part's box. ``detail``
    adds or removes levels (x2 detail = +1 level). ``displace: {amplitude,
    scale, octaves, seed, ridged}`` pushes the surface along its normals.
    """
    from ..core import meshops, subdiv, uvmap

    material = mesh.material
    sub = part_def.get("subdivide")
    crease = None
    if sub is not None:
        spec = sub if isinstance(sub, dict) else {"levels": sub}
        unknown = set(spec) - {"levels", "crease"}
        if unknown:
            raise ValueError(f"Part '{part_name}' subdivide: unknown keys {sorted(unknown)} (levels, crease)")
        crease = float(spec["crease"]) if spec.get("crease") is not None else None
        levels = int(spec.get("levels", 1))
        if detail != 1.0 and levels > 0:
            levels += int(round(np.log2(max(detail, 1e-3))))
        lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
        mesh = subdiv.subdivide(mesh, max(0, min(levels, 5)), crease)
        # Smoothing shrinks the cage; stretch the result back to the part's box.
        new_lo, new_hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
        span = np.where(new_hi - new_lo > 1e-9, new_hi - new_lo, 1.0)
        mesh.vertices = lo + (mesh.vertices - new_lo) * np.where(hi - lo > 1e-9, (hi - lo) / span, 1.0)
    disp = part_def.get("displace")
    directions = None
    if disp is not None:
        unknown = set(disp) - {"amplitude", "scale", "octaves", "seed", "ridged"}
        if unknown:
            raise ValueError(
                f"Part '{part_name}' displace: unknown keys {sorted(unknown)} (amplitude, scale, octaves, seed, ridged)"
            )
        # Project UVs by the undisplaced shape's normals: clean seams, no noise islands.
        base = subdiv.subdivide(mesh, 0)   # position-welded, same faces as the displaced result
        directions = meshops.face_normals(base.vertices, base.faces)[0]
        mesh = subdiv.displace(base, float(disp.get("amplitude", 0.02)), float(disp.get("scale", 0.3)),
                               int(disp.get("octaves", 3)), int(round(float(disp.get("seed", 0)))),
                               bool(disp.get("ridged", False)))
        if len(directions) != len(mesh.faces):
            directions = None
    mesh = meshops.compute_normals(uvmap.box_project(mesh, directions=directions), crease if crease is not None else 75.0)
    mesh.material = material
    return mesh


def _asset_ref(path: Path) -> str:
    """``path`` as a scene names it: relative to the content root it is in (``vehicles/car.yaml``;
    a scene folder's own file relative to the folder)."""

    root = roots.root_of(path)
    return Path(path).resolve().relative_to(root.resolve()).as_posix() if root is not None else str(path)


def _params_key(params: dict | None) -> str:
    import json

    return json.dumps(params or {}, sort_keys=True, default=str)


# --- primitive builders: (loader, part size in metres, part YAML) -> generator -----------------

def _build_cube(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    return CubeGenerator(
        size_x=size[0], size_y=size[1], size_z=size[2],
        bevel=config.get("bevel", 0.02),
        # Small parts (slats, posts, rails) get one step per 45 degrees: the extra
        # rounding isn't visible at that size and costs ~3x the triangles.
        bevel_segments=config.get("bevel_segments", 1 if float(np.min(size)) < 0.12 else 2),
    )


def _build_cylinder(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    radius = min(size[0], size[2]) / 2
    bevel = float(config.get("bevel", 0.0))
    if bevel > 0:  # rounded rims (cylinders are sharp by default)
        return BevelledCylinderGenerator(radius=radius, height=size[1], bevel=bevel,
                                         bevel_segments=int(config.get("bevel_segments", 4)))
    return CylinderGenerator(radius=radius, height=size[1])


def _build_plane(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    # Width (X) and depth (Z); Y is ignored.
    return PlaneGenerator(size_x=size[0], size_z=size[2], subdivisions_x=config.get("subdivisions_x", 1),
                          subdivisions_z=config.get("subdivisions_z", 1))


def _build_paths(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    return PathsGenerator(paths=list(config.get("paths") or []), width=float(config.get("width", 1.8)),
                          thickness=float(config.get("thickness", 0.04)),
                          bevel=float(config.get("bevel", 0.01)), detail=loader.detail)


def _build_water(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    return loader._create_extrude_generator(size, {**config, "fit": config.get("fit", "none"), "axis": "y"})


def _build_terrain(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    extent = config.get("extent") or [size[0], size[2]]
    return TerrainGenerator(
        size=(float(extent[0]), float(extent[1])), resolution=float(config.get("resolution", 1.0)),
        height=float(config.get("height", 12.0)), sea_level=float(config.get("sea_level", 0.0)),
        depth=float(config.get("depth", 4.0)), seed=int(config.get("seed", 0)),
        scale=float(config.get("scale", 40.0)), octaves=int(config.get("octaves", 4)),
        falloff=dict(config.get("falloff") or {}), pads=list(config.get("pads") or []),
        island=bool(config.get("island", True)), paths=list(config.get("paths") or []))


def _cast(config: dict[str, Any], keys: dict[str, Any]) -> dict[str, Any]:
    """The given ``keys`` present in ``config``, each converted with its type."""
    return {k: cast(config[k]) for k, cast in keys.items() if k in config}


def _build_fence(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    return FenceGenerator(path=config.get("path") or [], closed=bool(config.get("closed", False)),
                          gaps=[[float(a), float(b)] for a, b in config.get("gaps") or []],
                          **_cast(config, {"height": float, "post": float, "post_spacing": float,
                                           "picket": float, "spacing": float, "rail": float}))


def _build_text(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    text = config.get("text", "")
    if isinstance(text, float) and text.is_integer():
        text = int(text)            # "{floor}" resolves to a number
    return TextGenerator(
        text=str(text), font=str(config.get("font", "sans")),
        height=float(config.get("height", 0.2)), depth=float(config.get("depth", 0.02)),
        align=str(config.get("align", "center")), line_spacing=float(config.get("line_spacing", 1.0)),
        tracking=float(config.get("tracking", 0.0)), bevel=float(config.get("bevel", 0.0)),
        detail=loader.detail,
    )


def _build_stairs(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    return StairsGenerator(**_cast(config, {
        "style": str, "rise": float, "width": float, "max_riser": float, "tread": float, "turn": lambda v: v,
        "landing_at": float, "waist": float, "railing": lambda v: v, "rail_height": float,
        "railing_material": lambda v: v}))


def _build_roof(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    return RoofGenerator(width=size[0], height=size[1], depth=size[2], style=config.get("style", "gable"),
                         overhang=float(config.get("overhang", 0.35)), thickness=float(config.get("thickness", 0.12)),
                         ridge_axis=config.get("ridge_axis", "auto"), ridge_cap=bool(config.get("ridge_cap", True)))


def _seed_of(config: dict[str, Any]) -> int:
    return int(round(float(config.get("seed", 0))))


def _build_tree(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    return TreeGenerator(width=size[0], height=size[1], depth=size[2], seed=_seed_of(config), **_cast(config, {
        "style": str, "trunk_height": float, "trunk_radius": float, "attractors": int, "step": float,
        "leaf_size": float, "segments": int, "foliage_material": str, "fronds": int}))


def _build_crop(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    return CropGenerator(width=size[0], height=size[1], depth=size[2], seed=_seed_of(config),
                         **_cast(config, {"kind": str, "growth": float, "spacing": float}))


def _build_rock(loader: LayoutLoader, size: np.ndarray, config: dict[str, Any]):
    args = _cast(config, {"points": int, "levels": int, "roughness": float,
                          "crease": lambda v: None if v is None else float(v)})
    return RockGenerator(width=size[0], height=size[1], depth=size[2], seed=_seed_of(config), **args)


_BUILDERS: dict[str, Any] = {
    "cube": _build_cube,
    "cylinder": _build_cylinder,
    # The sphere keeps the smallest side as its radius; scaling the node stretches it.
    "sphere": lambda loader, size, config: SphereGenerator(radius=min(size) / 2),
    "cone": lambda loader, size, config: ConeGenerator(radius=min(size[0], size[2]) / 2, height=size[1]),
    "plane": _build_plane,
    "ellipsoid": lambda loader, size, config: EllipsoidGenerator(size_x=size[0], size_y=size[1], size_z=size[2]),
    "extrude": lambda loader, size, config: loader._create_extrude_generator(size, config),
    "lathe": lambda loader, size, config: loader._create_lathe_generator(size, config),
    "sweep": lambda loader, size, config: loader._create_sweep_generator(config),
    "paths": _build_paths,
    "water": _build_water,
    "terrain": _build_terrain,
    "fence": _build_fence,
    "text": _build_text,
    "torus": lambda loader, size, config: TorusGenerator(size_x=size[0], size_y=size[1], size_z=size[2],
                                                         tube=float(config.get("tube", 0.0))),
    "capsule": lambda loader, size, config: CapsuleGenerator(size_x=size[0], size_y=size[1], size_z=size[2]),
    "stairs": _build_stairs,
    "roof": _build_roof,
    "tree": _build_tree,
    "crop": _build_crop,
    "rock": _build_rock,
    "prism": lambda loader, size, config: PrismGenerator(width=size[0], height=size[1], depth=size[2],
                                                         apex=config.get("apex", "center")),
}


# --- per-primitive work after the generator builds the node ----------------------------------

def _post_water(loader: LayoutLoader, node: SceneNode, generator: Any, part_def: dict[str, Any],
                size: np.ndarray) -> None:
    from ..generators.water import add_barrier, water_meta

    node.meta["water"] = water_meta(part_def)
    node.meta["collider"] = "none"
    part_def.setdefault("material", "water")
    if part_def.get("barrier"):
        add_barrier(node, generator.shape, float(part_def["barrier"]), float(size[1]))


def _post_terrain(loader: LayoutLoader, node: SceneNode, generator: Any, part_def: dict[str, Any],
                  size: np.ndarray) -> None:
    node.meta["terrain"] = generator.params()
    part_def.setdefault("walkable", True)
    slots = {"sand": "sand", "grass": "grass", "rock": "rock", "path": "gravel", **(part_def.get("materials") or {})}
    node.mesh.materials = [loader.material_loader.load(slots[k]) for k in TERRAIN_SLOTS]
    ribbons = generator.path_mesh()
    if ribbons is not None:
        ribbons.material = loader.material_loader.load(slots["path"])
        path_node = SceneNode("paths", mesh=ribbons, tags=["terrain.path"])
        path_node.meta["collider"] = "none"      # you walk on the terrain beneath
        node.add_child(path_node)
    if part_def.get("shore"):
        from ..generators.terrain import add_shore

        add_shore(node, generator, dict(part_def["shore"]))


_POST_BUILD: dict[str, Any] = {"water": _post_water, "terrain": _post_terrain}


# --- asset-level keys that set the root's meta: (key, handler(root, value, part_nodes, name)) ----
# Run in this order once the parts, booleans, surfaces and interactions exist.

def _meta_light(root: SceneNode, spec: Any, parts: dict[str, SceneNode], name: str) -> None:
    root.meta["light"] = light_spec(spec)


def _meta_switch(root: SceneNode, spec: Any, parts: dict[str, SceneNode], name: str) -> None:
    if not isinstance(spec, dict) or set(spec) != {"light"}:
        raise ValueError(f"{name}: switch is {{light: <the light fixture's name>}}, got {spec!r}")
    if str(spec["light"]):          # "": a switch that isn't wired to anything
        root.meta["switch"] = {"light": str(spec["light"])}


def _meta_container(root: SceneNode, spec: Any, parts: dict[str, SceneNode], name: str) -> None:
    root.meta["container"] = {"capacity": int(spec.get("capacity", 1)),
                              **({"part": str(spec["part"])} if "part" in spec else {})}


def _meta_clearance(root: SceneNode, clearance: Any, parts: dict[str, SceneNode], name: str) -> None:
    # Furniture: footprint (x, z) and the free space needed around it (front = +Z)
    # for furnishing solvers and navigation.
    unknown = set(clearance) - {"front", "back", "left", "right"}
    if unknown:
        raise ValueError(f"'{name}': clearance sides must be front/back/left/right, got {sorted(unknown)}")
    root.meta["footprint"] = [float(root.size[0]), float(root.size[2])]
    root.meta["clearance"] = {k: float(v) for k, v in clearance.items()}


def _meta_affordances(root: SceneNode, spec: Any, parts: dict[str, SceneNode], name: str) -> None:
    root.meta["affordances"] = [_affordance(a, root) for a in spec]


def _meta_poses(root: SceneNode, spec: Any, parts: dict[str, SceneNode], name: str) -> None:
    from ..npc import parse_poses

    root.meta["poses"] = parse_poses(spec)


def _meta_portal(root: SceneNode, spec: Any, parts: dict[str, SceneNode], name: str) -> None:
    from ..npc import parse_portal

    root.meta["portal"] = parse_portal(spec, root.interactions)


def _meta_travel(root: SceneNode, spec: Any, parts: dict[str, SceneNode], name: str) -> None:
    from ..travel import apply_travel

    apply_travel(root, spec, name)


def _meta_vehicle(root: SceneNode, spec: Any, parts: dict[str, SceneNode], name: str) -> None:
    from ..vehicles import parse_vehicle

    root.meta["vehicle"] = parse_vehicle(spec, root, parts)
    root.meta["type"] = "vehicle"


def _meta_physics(root: SceneNode, spec: Any, parts: dict[str, SceneNode], name: str) -> None:
    from ..physics import parse_physics

    root.meta["physics"] = parse_physics(spec, root, parts, name)


def _meta_source(root: SceneNode, spec: Any, parts: dict[str, SceneNode], name: str) -> None:
    from ..flows import parse_source

    root.meta.setdefault("flows", {})["source"] = parse_source(spec, root, name)


def _meta_track(root: SceneNode, spec: Any, parts: dict[str, SceneNode], name: str) -> None:
    from ..flows import parse_track

    root.meta.setdefault("flows", {})["track"] = parse_track(spec, name)


def _meta_effector(root: SceneNode, spec: Any, parts: dict[str, SceneNode], name: str) -> None:
    from ..flows import parse_effector

    root.meta.setdefault("flows", {})["effector"] = parse_effector(spec, root, parts, name)


_ROOT_META: list[tuple[str, Any]] = [
    ("light", _meta_light), ("switch", _meta_switch), ("container", _meta_container),
    ("clearance", _meta_clearance), ("affordances", _meta_affordances), ("poses", _meta_poses),
    ("portal", _meta_portal), ("travel", _meta_travel), ("vehicle", _meta_vehicle), ("physics", _meta_physics),
    ("source", _meta_source), ("track", _meta_track), ("effector", _meta_effector),
]
