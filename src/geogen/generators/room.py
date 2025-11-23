"""Room generator with doors and windows."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.mesh import Mesh
from ..core.node import SceneNode
from .base import MeshGenerator


@dataclass
class Opening:
    """Defines a door or window opening in a wall.

    Attributes:
        wall: Which wall the opening is on ('north', 'south', 'east', 'west')
        position: Horizontal position along wall as fraction (0-1, left to right when facing wall)
        bottom: Bottom of opening as fraction of wall height (0 for doors)
        width: Width of opening as fraction of wall width
        height: Height of opening as fraction of wall height
    """
    wall: str
    position: float
    bottom: float
    width: float
    height: float


@dataclass
class RoomGenerator(MeshGenerator):
    """Generates a room with walls, floor, ceiling, and openings for doors/windows.

    The room is centered at the origin with the floor at y=0.
    Walls have thickness and can have rectangular openings cut out.

    Attributes:
        size_x: Width of the room interior (X axis)
        size_y: Height of the room (Y axis)
        size_z: Depth of the room interior (Z axis)
        wall_thickness: Thickness of the walls
        has_floor: Whether to generate a floor
        has_ceiling: Whether to generate a ceiling
        openings: List of door/window openings
    """
    size_x: float = 4.0
    size_y: float = 2.5
    size_z: float = 4.0
    wall_thickness: float = 0.1
    has_floor: bool = True
    has_ceiling: bool = True
    openings: list[Opening] = field(default_factory=list)

    def generate(self) -> Mesh:
        """Generate the room mesh."""
        meshes = []

        # Generate floor
        if self.has_floor:
            floor = self._make_floor()
            meshes.append(floor)

        # Generate ceiling
        if self.has_ceiling:
            ceiling = self._make_ceiling()
            meshes.append(ceiling)

        # Generate walls with openings
        for wall_name in ['north', 'south', 'east', 'west']:
            wall_openings = [o for o in self.openings if o.wall == wall_name]
            wall = self._make_wall(wall_name, wall_openings)
            meshes.append(wall)

        return Mesh.merge(meshes)

    def _make_floor(self) -> Mesh:
        """Generate the floor as a flat quad."""
        hx = self.size_x / 2
        hz = self.size_z / 2

        vertices = [
            [-hx, 0, -hz],
            [-hx, 0, hz],
            [hx, 0, hz],
            [hx, 0, -hz],
        ]

        # UVs map to room dimensions
        uvs = [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
        ]

        # CCW winding for upward normal (+Y)
        faces = [
            [0, 1, 2],
            [0, 2, 3],
        ]

        return Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
        )

    def _make_ceiling(self) -> Mesh:
        """Generate the ceiling as a flat quad visible from inside (normal points -Y)."""
        hx = self.size_x / 2
        hz = self.size_z / 2
        y = self.size_y

        # For normal to point -Y, we need CCW winding when viewed from below
        # Looking up at ceiling from inside: we see back-left, back-right, front-right, front-left
        vertices = [
            [-hx, y, -hz],  # 0: back left
            [hx, y, -hz],   # 1: back right
            [hx, y, hz],    # 2: front right
            [-hx, y, hz],   # 3: front left
        ]

        uvs = [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ]

        # Reverse winding for -Y normal
        faces = [
            [0, 2, 3],
            [0, 1, 2],
        ]

        return Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
        )

    def _make_wall(self, wall_name: str, openings: list[Opening]) -> Mesh:
        """Generate a wall with optional openings cut out.

        Args:
            wall_name: 'north', 'south', 'east', or 'west'
            openings: List of openings in this wall

        Returns:
            Mesh for the wall
        """
        # Get wall dimensions
        # North/South walls span the full width and include corner regions
        # East/West walls fit between North/South walls (shorter by wall_thickness on each end)
        if wall_name in ('north', 'south'):
            wall_width = self.size_x + 2 * self.wall_thickness  # Full width including corners
            wall_depth = self.wall_thickness
        else:  # east, west
            wall_width = self.size_z  # Fit between north/south walls
            wall_depth = self.wall_thickness

        wall_height = self.size_y

        # Sort openings by position for proper subdivision
        openings = sorted(openings, key=lambda o: o.position)

        # Generate wall faces by subdividing around openings
        # All walls need inner face pointing -Z (reversed winding) because:
        # - North (no rotation): -Z stays -Z, pointing into room from +Z position
        # - South (180° Y): -Z becomes +Z, pointing into room from -Z position
        # - East (-90° Y): -Z becomes -X, pointing into room from +X position
        # - West (90° Y): -Z becomes +X, pointing into room from -X position
        reverse_inner_winding = True

        if not openings:
            # Simple solid wall
            mesh = self._make_solid_wall_panel(wall_width, wall_height, wall_depth, reverse_inner_winding)
        else:
            # Wall with openings - create panels around them
            mesh = self._make_wall_with_openings(wall_width, wall_height, wall_depth, openings, reverse_inner_winding)

        # Transform wall to correct position and orientation
        mesh = self._position_wall(mesh, wall_name)

        return mesh

    def _make_solid_wall_panel(self, width: float, height: float, depth: float, reverse_inner_winding: bool = False) -> Mesh:
        """Create a solid rectangular wall panel centered at origin.

        The wall extends from -width/2 to +width/2 on X,
        from 0 to height on Y, and from -depth/2 to +depth/2 on Z.

        Args:
            reverse_inner_winding: If True, inner face normal points -Z; if False, points +Z
        """
        hw = width / 2
        hd = depth / 2

        vertices = []
        uvs = []
        faces = []

        # Inner face (at -Z position)
        base_idx = len(vertices)
        vertices.extend([
            [-hw, 0, -hd],
            [hw, 0, -hd],
            [hw, height, -hd],
            [-hw, height, -hd],
        ])
        uvs.extend([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ])
        if reverse_inner_winding:
            # Normal points -Z
            faces.extend([
                [base_idx, base_idx + 2, base_idx + 1],
                [base_idx, base_idx + 3, base_idx + 2],
            ])
        else:
            # Normal points +Z
            faces.extend([
                [base_idx, base_idx + 1, base_idx + 2],
                [base_idx, base_idx + 2, base_idx + 3],
            ])

        # Outer face (+Z direction)
        base_idx = len(vertices)
        vertices.extend([
            [hw, 0, hd],
            [-hw, 0, hd],
            [-hw, height, hd],
            [hw, height, hd],
        ])
        uvs.extend([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ])
        faces.extend([
            [base_idx, base_idx + 1, base_idx + 2],
            [base_idx, base_idx + 2, base_idx + 3],
        ])

        # Top face (visible from above, normal points +Y)
        base_idx = len(vertices)
        vertices.extend([
            [-hw, height, hd],   # 0: front left
            [hw, height, hd],    # 1: front right
            [hw, height, -hd],   # 2: back right
            [-hw, height, -hd],  # 3: back left
        ])
        uvs.extend([
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 0],
        ])
        # CCW when viewed from +Y (above) = normal points +Y
        faces.extend([
            [base_idx, base_idx + 1, base_idx + 2],
            [base_idx, base_idx + 2, base_idx + 3],
        ])

        # Left side face (-X) - normal points -X (outward from wall)
        base_idx = len(vertices)
        vertices.extend([
            [-hw, 0, -hd],      # back bottom
            [-hw, 0, hd],       # front bottom
            [-hw, height, hd],  # front top
            [-hw, height, -hd], # back top
        ])
        uvs.extend([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ])
        faces.extend([
            [base_idx, base_idx + 1, base_idx + 2],
            [base_idx, base_idx + 2, base_idx + 3],
        ])

        # Right side face (+X) - normal points +X (outward from wall)
        base_idx = len(vertices)
        vertices.extend([
            [hw, 0, hd],       # front bottom
            [hw, 0, -hd],      # back bottom
            [hw, height, -hd], # back top
            [hw, height, hd],  # front top
        ])
        uvs.extend([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ])
        faces.extend([
            [base_idx, base_idx + 1, base_idx + 2],
            [base_idx, base_idx + 2, base_idx + 3],
        ])

        return Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
        )

    def _make_wall_with_openings(
        self, width: float, height: float, depth: float, openings: list[Opening], reverse_inner_winding: bool = False
    ) -> Mesh:
        """Create a wall with rectangular openings.

        Generates panels around each opening:
        - Panels to the left and right of openings
        - Panels above and below openings
        - Inner faces of the opening (reveals)

        Args:
            reverse_inner_winding: If True, inner face normal points -Z; if False, points +Z
        """
        meshes = []
        hw = width / 2
        hd = depth / 2

        # Convert openings to absolute coordinates
        # Each opening: (left_x, right_x, bottom_y, top_y)
        abs_openings = []
        for o in openings:
            o_width = o.width * width
            o_height = o.height * height
            center_x = (o.position - 0.5) * width  # Convert 0-1 to -hw to +hw
            left_x = center_x - o_width / 2
            right_x = center_x + o_width / 2
            bottom_y = o.bottom * height
            top_y = bottom_y + o_height
            abs_openings.append((left_x, right_x, bottom_y, top_y))

        # Generate inner wall face with holes
        # We'll create a grid of panels around the openings
        inner_panels = self._subdivide_wall_face(
            -hw, hw, 0, height, abs_openings, -hd
        )

        for panel in inner_panels:
            mesh = self._make_face_panel(panel, -hd, reverse_inner_winding)
            meshes.append(mesh)

        # Generate outer wall face with same holes
        outer_panels = self._subdivide_wall_face(
            -hw, hw, 0, height, abs_openings, hd
        )

        for panel in outer_panels:
            mesh = self._make_face_panel(panel, hd, reverse_winding=False)
            meshes.append(mesh)

        # Generate top face of wall
        meshes.append(self._make_wall_top(width, height, depth))

        # Generate side faces (left and right edges of wall)
        meshes.append(self._make_wall_left_side(width, height, depth))
        meshes.append(self._make_wall_right_side(width, height, depth))

        # Generate reveals (inner faces of openings)
        for left_x, right_x, bottom_y, top_y in abs_openings:
            reveals = self._make_opening_reveals(
                left_x, right_x, bottom_y, top_y, depth
            )
            meshes.extend(reveals)

        return Mesh.merge(meshes)

    def _subdivide_wall_face(
        self,
        left: float,
        right: float,
        bottom: float,
        top: float,
        openings: list[tuple[float, float, float, float]],
        z: float,
    ) -> list[tuple[float, float, float, float]]:
        """Subdivide a rectangular face around openings.

        Returns list of (left, right, bottom, top) tuples for solid panels.
        """
        if not openings:
            return [(left, right, bottom, top)]

        panels = []

        # Collect all unique X and Y coordinates
        x_coords = sorted(set([left, right] + [o[0] for o in openings] + [o[1] for o in openings]))
        y_coords = sorted(set([bottom, top] + [o[2] for o in openings] + [o[3] for o in openings]))

        # Create grid cells and check if each is inside an opening
        for i in range(len(x_coords) - 1):
            for j in range(len(y_coords) - 1):
                cell_left = x_coords[i]
                cell_right = x_coords[i + 1]
                cell_bottom = y_coords[j]
                cell_top = y_coords[j + 1]

                # Check if this cell is inside any opening
                cell_center_x = (cell_left + cell_right) / 2
                cell_center_y = (cell_bottom + cell_top) / 2

                inside_opening = False
                for o_left, o_right, o_bottom, o_top in openings:
                    if (o_left < cell_center_x < o_right and
                        o_bottom < cell_center_y < o_top):
                        inside_opening = True
                        break

                if not inside_opening:
                    panels.append((cell_left, cell_right, cell_bottom, cell_top))

        return panels

    def _make_face_panel(
        self,
        bounds: tuple[float, float, float, float],
        z: float,
        reverse_winding: bool,
    ) -> Mesh:
        """Create a single rectangular panel at given Z position.

        Args:
            bounds: (left, right, bottom, top) bounds of the panel
            z: Z position of the panel
            reverse_winding: If True, normal points -Z; if False, normal points +Z
        """
        left, right, bottom, top = bounds

        # Vertices in standard order
        vertices = [
            [left, bottom, z],
            [right, bottom, z],
            [right, top, z],
            [left, top, z],
        ]

        uvs = [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ]

        if reverse_winding:
            # Normal points -Z
            faces = [
                [0, 2, 1],
                [0, 3, 2],
            ]
        else:
            # Normal points +Z
            faces = [
                [0, 1, 2],
                [0, 2, 3],
            ]

        return Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
        )

    def _make_wall_top(self, width: float, height: float, depth: float) -> Mesh:
        """Create the top face of the wall (visible from above, normal +Y)."""
        hw = width / 2
        hd = depth / 2

        # CCW when viewed from above (+Y) = normal points +Y
        vertices = [
            [-hw, height, hd],   # front left
            [hw, height, hd],    # front right
            [hw, height, -hd],   # back right
            [-hw, height, -hd],  # back left
        ]
        uvs = [[0, 1], [1, 1], [1, 0], [0, 0]]
        faces = [[0, 1, 2], [0, 2, 3]]

        return Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
        )

    def _make_wall_left_side(self, width: float, height: float, depth: float) -> Mesh:
        """Create the left side face of the wall (normal points -X, outward)."""
        hw = width / 2
        hd = depth / 2

        # For normal pointing -X, CCW when viewed from -X direction
        vertices = [
            [-hw, 0, -hd],     # 0: back bottom
            [-hw, 0, hd],      # 1: front bottom
            [-hw, height, hd], # 2: front top
            [-hw, height, -hd],# 3: back top
        ]
        uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
        faces = [[0, 1, 2], [0, 2, 3]]

        return Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
        )

    def _make_wall_right_side(self, width: float, height: float, depth: float) -> Mesh:
        """Create the right side face of the wall (normal points +X, outward)."""
        hw = width / 2
        hd = depth / 2

        # For normal pointing +X, CCW when viewed from +X direction
        vertices = [
            [hw, 0, hd],      # 0: front bottom
            [hw, 0, -hd],     # 1: back bottom
            [hw, height, -hd],# 2: back top
            [hw, height, hd], # 3: front top
        ]
        uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
        faces = [[0, 1, 2], [0, 2, 3]]

        return Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
        )

    def _make_opening_reveals(
        self,
        left_x: float,
        right_x: float,
        bottom_y: float,
        top_y: float,
        depth: float,
    ) -> list[Mesh]:
        """Create the inner faces of an opening (the reveals).

        Creates double-sided reveals so they're visible from both inside
        and outside the room. Each reveal has faces on both sides.
        """
        meshes = []
        hd = depth / 2

        # Left reveal - double sided
        vertices = [
            [left_x, bottom_y, -hd],  # 0: back bottom
            [left_x, bottom_y, hd],   # 1: front bottom
            [left_x, top_y, hd],      # 2: front top
            [left_x, top_y, -hd],     # 3: back top
        ]
        uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
        # Both sides: facing +X and facing -X
        faces = [
            [0, 1, 2], [0, 2, 3],  # facing +X
            [0, 2, 1], [0, 3, 2],  # facing -X
        ]
        meshes.append(Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
        ))

        # Right reveal - double sided
        vertices = [
            [right_x, bottom_y, -hd],  # 0: back bottom
            [right_x, bottom_y, hd],   # 1: front bottom
            [right_x, top_y, hd],      # 2: front top
            [right_x, top_y, -hd],     # 3: back top
        ]
        uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
        faces = [
            [0, 2, 1], [0, 3, 2],  # facing -X
            [0, 1, 2], [0, 2, 3],  # facing +X
        ]
        meshes.append(Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
        ))

        # Top reveal - double sided
        vertices = [
            [left_x, top_y, -hd],   # 0: left back
            [right_x, top_y, -hd],  # 1: right back
            [right_x, top_y, hd],   # 2: right front
            [left_x, top_y, hd],    # 3: left front
        ]
        uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
        faces = [
            [0, 1, 2], [0, 2, 3],  # facing -Y (down)
            [0, 2, 1], [0, 3, 2],  # facing +Y (up)
        ]
        meshes.append(Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
        ))

        # Bottom reveal - double sided, only if opening doesn't start at floor
        if bottom_y > 0.001:
            vertices = [
                [left_x, bottom_y, -hd],   # 0: left back
                [right_x, bottom_y, -hd],  # 1: right back
                [right_x, bottom_y, hd],   # 2: right front
                [left_x, bottom_y, hd],    # 3: left front
            ]
            uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
            faces = [
                [0, 2, 1], [0, 3, 2],  # facing +Y (up)
                [0, 1, 2], [0, 2, 3],  # facing -Y (down)
            ]
            meshes.append(Mesh(
                vertices=np.array(vertices, dtype=np.float64),
                faces=np.array(faces, dtype=np.int64),
                uvs=np.array(uvs, dtype=np.float64),
            ))

        return meshes

    def _position_wall(self, mesh: Mesh, wall_name: str) -> Mesh:
        """Transform a wall mesh to its correct position.

        Walls are generated centered at origin along X, from 0 to height on Y.
        This transforms them to their final positions:
        - north: at +Z edge, facing -Z (into room)
        - south: at -Z edge, facing +Z (into room)
        - east: at +X edge, facing -X (into room)
        - west: at -X edge, facing +X (into room)
        """
        hx = self.size_x / 2
        hz = self.size_z / 2

        if wall_name == 'north':
            # Wall at +Z, no rotation needed, just translate
            matrix = np.eye(4)
            matrix[2, 3] = hz  # Move to +Z

        elif wall_name == 'south':
            # Wall at -Z, rotate 180 degrees around Y
            matrix = np.array([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, -hz],
                [0, 0, 0, 1],
            ], dtype=np.float64)

        elif wall_name == 'east':
            # Wall at +X, rotate -90 degrees around Y
            matrix = np.array([
                [0, 0, 1, hx],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
            ], dtype=np.float64)

        elif wall_name == 'west':
            # Wall at -X, rotate 90 degrees around Y
            matrix = np.array([
                [0, 0, -1, -hx],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ], dtype=np.float64)
        else:
            raise ValueError(f"Unknown wall name: {wall_name}")

        return mesh.transform(matrix)

    def to_node(self, name: str | None = None) -> SceneNode:
        """Generate geometry and wrap it in a SceneNode."""
        node_name = name or "room"
        return SceneNode(name=node_name, mesh=self.generate())
