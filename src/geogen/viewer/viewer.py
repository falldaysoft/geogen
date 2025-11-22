"""Simple viewer for displaying geometry using trimesh."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

from ..core.mesh import Mesh
from ..core.node import SceneNode

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Viewer:
    """A simple viewer for displaying geometry.

    Uses trimesh's built-in viewer (pyglet-based) to display meshes and scenes.
    """

    def __init__(
        self,
        root: SceneNode,
        color: NDArray[np.float64] | tuple[float, ...] | None = None,
    ) -> None:
        """Initialize the viewer with a scene.

        Args:
            root: The root SceneNode to display
            color: Optional default color for all meshes
        """
        self._scene = trimesh.Scene()
        self._root = root
        self._add_scene_node(root, color)
        self._set_default_camera()

    def _set_default_camera(self) -> None:
        """Set up default camera position - zoomed out and rotated 20 degrees."""
        angle = np.radians(20)
        distance = 5.0
        cam_pos = np.array([np.sin(angle) * distance, 2.0, np.cos(angle) * distance])
        target = np.array([0.0, 0.4, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        # Build look-at matrix
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        camera_transform = np.eye(4)
        camera_transform[:3, 0] = right
        camera_transform[:3, 1] = up
        camera_transform[:3, 2] = -forward
        camera_transform[:3, 3] = cam_pos

        self._scene.camera_transform = camera_transform

    def _add_mesh(
        self,
        mesh: Mesh,
        name: str | None = None,
        color: NDArray[np.float64] | tuple[float, ...] | None = None,
        transform: NDArray[np.float64] | None = None,
    ) -> str:
        """Add a mesh to the scene.

        Args:
            mesh: The Mesh to add
            name: Optional name for the geometry
            color: Optional RGBA color (0-1 range) or RGB tuple (fallback if no material)
            transform: Optional 4x4 transformation matrix

        Returns:
            The name assigned to the geometry in the scene
        """
        # to_trimesh() will apply material texture if available
        tm_mesh = mesh.to_trimesh()

        # Only apply color if mesh doesn't have a material with texture
        has_texture = (
            mesh.material is not None
            and mesh.uvs is not None
            and hasattr(tm_mesh.visual, 'uv')
        )

        if color is not None and not has_texture:
            color_array = np.asarray(color)
            if len(color_array) == 3:
                color_array = np.append(color_array, 1.0)
            # Set face colors
            tm_mesh.visual.face_colors = (color_array * 255).astype(np.uint8)

        geom_name = name or f"mesh_{len(self._scene.geometry)}"
        self._scene.add_geometry(tm_mesh, node_name=geom_name, transform=transform)

        return geom_name

    def _add_scene_node(
        self,
        node: SceneNode,
        color: NDArray[np.float64] | tuple[float, ...] | None = None,
    ) -> list[str]:
        """Add a SceneNode hierarchy to the viewer.

        Args:
            node: The root SceneNode to add
            color: Optional default color for all meshes

        Returns:
            List of names assigned to the geometries
        """
        names = []

        for scene_node, world_mesh in node.iter_meshes():
            # Generate unique name using hierarchy path to avoid collisions
            path_parts = []
            current = scene_node
            while current is not None:
                path_parts.append(current.name)
                current = current.parent
            unique_name = "/".join(reversed(path_parts))

            name = self._add_mesh(
                mesh=world_mesh,
                name=unique_name,
                color=color,
            )
            names.append(name)

        return names

    def show(self, **kwargs) -> None:
        """Display the scene in an interactive viewer window.

        Args:
            **kwargs: Additional arguments passed to trimesh.Scene.show()
        """
        if len(self._scene.geometry) == 0:
            print("Warning: No geometry to display")
            return

        self._scene.show(**kwargs)

    @property
    def scene(self) -> trimesh.Scene:
        """Get the underlying trimesh Scene."""
        return self._scene


def show_mesh(mesh: Mesh, **kwargs) -> None:
    """Convenience function to quickly display a single mesh.

    Args:
        mesh: The Mesh to display
        **kwargs: Additional arguments passed to the viewer
    """
    node = SceneNode("root")
    node.mesh = mesh
    viewer = Viewer(node)
    viewer.show(**kwargs)


def show_node(node: SceneNode, **kwargs) -> None:
    """Convenience function to quickly display a SceneNode hierarchy.

    Args:
        node: The root SceneNode to display
        **kwargs: Additional arguments passed to the viewer
    """
    viewer = Viewer(node)
    viewer.show(**kwargs)
