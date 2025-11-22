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

    def __init__(self) -> None:
        """Initialize the viewer."""
        self._scene = trimesh.Scene()

    def add_mesh(
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
            color: Optional RGBA color (0-1 range) or RGB tuple
            transform: Optional 4x4 transformation matrix

        Returns:
            The name assigned to the geometry in the scene
        """
        tm_mesh = mesh.to_trimesh()

        if color is not None:
            color_array = np.asarray(color)
            if len(color_array) == 3:
                color_array = np.append(color_array, 1.0)
            # Set face colors
            tm_mesh.visual.face_colors = (color_array * 255).astype(np.uint8)

        geom_name = name or f"mesh_{len(self._scene.geometry)}"
        self._scene.add_geometry(tm_mesh, node_name=geom_name, transform=transform)

        return geom_name

    def add_scene_node(
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
            name = self.add_mesh(
                mesh=world_mesh,
                name=scene_node.name,
                color=color,
            )
            names.append(name)

        return names

    def clear(self) -> None:
        """Remove all geometry from the scene."""
        self._scene = trimesh.Scene()

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
    viewer = Viewer()
    viewer.add_mesh(mesh)
    viewer.show(**kwargs)


def show_node(node: SceneNode, **kwargs) -> None:
    """Convenience function to quickly display a SceneNode hierarchy.

    Args:
        node: The root SceneNode to display
        **kwargs: Additional arguments passed to the viewer
    """
    viewer = Viewer()
    viewer.add_scene_node(node)
    viewer.show(**kwargs)
