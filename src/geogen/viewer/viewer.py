"""Simple viewer for displaying geometry using trimesh."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import trimesh

from ..core.mesh import Mesh
from ..core.node import SceneNode

if TYPE_CHECKING:
    from numpy.typing import NDArray


SceneFactory = Callable[[], SceneNode]


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


class InteractiveViewer:
    """Interactive viewer with a menu to select different scenes.

    Press number keys (1-9) to switch between scenes.
    Press 'H' to show help/menu.
    """

    def __init__(
        self,
        scenes: dict[str, SceneFactory],
        default_scene: str | None = None,
        color: NDArray[np.float64] | tuple[float, ...] | None = None,
    ) -> None:
        """Initialize the interactive viewer.

        Args:
            scenes: Dictionary mapping scene names to factory functions
            default_scene: Name of scene to show initially (first if not specified)
            color: Default color for meshes without materials
        """
        self._scenes = scenes
        self._scene_names = list(scenes.keys())
        self._color = color
        self._current_index = 0

        if default_scene and default_scene in scenes:
            self._current_index = self._scene_names.index(default_scene)

        self._trimesh_scene: trimesh.Scene | None = None
        self._viewer = None

    def _build_trimesh_scene(self, root: SceneNode) -> trimesh.Scene:
        """Build a trimesh scene from a SceneNode hierarchy."""
        scene = trimesh.Scene()

        for scene_node, world_mesh in root.iter_meshes():
            # Generate unique name
            path_parts = []
            current = scene_node
            while current is not None:
                path_parts.append(current.name)
                current = current.parent
            unique_name = "/".join(reversed(path_parts))

            tm_mesh = world_mesh.to_trimesh()

            # Apply color if no texture
            has_texture = (
                world_mesh.material is not None
                and world_mesh.uvs is not None
                and hasattr(tm_mesh.visual, "uv")
            )

            if self._color is not None and not has_texture:
                color_array = np.asarray(self._color)
                if len(color_array) == 3:
                    color_array = np.append(color_array, 1.0)
                tm_mesh.visual.face_colors = (color_array * 255).astype(np.uint8)

            scene.add_geometry(tm_mesh, node_name=unique_name)

        return scene

    def _set_camera(self, scene: trimesh.Scene) -> None:
        """Set default camera position."""
        angle = np.radians(20)
        distance = 5.0
        cam_pos = np.array([np.sin(angle) * distance, 2.0, np.cos(angle) * distance])
        target = np.array([0.0, 0.4, 0.0])
        up = np.array([0.0, 1.0, 0.0])

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

        scene.camera_transform = camera_transform

    def _load_scene(self, index: int) -> None:
        """Load a scene by index."""
        name = self._scene_names[index]
        factory = self._scenes[name]
        root = factory()

        # Clear existing geometry
        if self._trimesh_scene is not None:
            geom_names = list(self._trimesh_scene.geometry.keys())
            for geom_name in geom_names:
                self._trimesh_scene.delete_geometry(geom_name)

            # Add new geometry
            new_scene = self._build_trimesh_scene(root)
            for geom_name, geom in new_scene.geometry.items():
                self._trimesh_scene.add_geometry(geom, node_name=geom_name)

        self._current_index = index
        print(f"\n>>> Switched to: {name}")

    def _print_menu(self) -> None:
        """Print the scene selection menu."""
        print("\n" + "=" * 40)
        print("Scene Selection Menu")
        print("=" * 40)
        for i, name in enumerate(self._scene_names):
            marker = " *" if i == self._current_index else ""
            key = i + 1 if i < 9 else 0
            print(f"  [{key}] {name}{marker}")
        print("-" * 40)
        print("  [H] Show this menu")
        print("  [Q] Quit")
        print("=" * 40)

    def show(self) -> None:
        """Display the interactive viewer."""
        # Import pyglet here to avoid issues if not installed
        try:
            import pyglet
            from trimesh.viewer import SceneViewer
        except ImportError as e:
            print(f"Error: {e}")
            print("Install pyglet with: pip install 'pyglet<2'")
            return

        # Build initial scene
        initial_name = self._scene_names[self._current_index]
        initial_root = self._scenes[initial_name]()
        self._trimesh_scene = self._build_trimesh_scene(initial_root)
        self._set_camera(self._trimesh_scene)

        # Print initial menu
        print("\nGeogen Interactive Viewer")
        print("Press number keys to switch scenes, H for help")
        self._print_menu()

        # Create viewer - we need to customize key handling
        viewer = self

        class CustomSceneViewer(SceneViewer):
            def on_key_press(inner_self, symbol, modifiers):
                # Number keys 1-9 for scene selection
                if pyglet.window.key._1 <= symbol <= pyglet.window.key._9:
                    index = symbol - pyglet.window.key._1
                    if index < len(viewer._scene_names):
                        viewer._load_scene(index)
                        inner_self._update_vertex_list()
                        return

                # 0 key for 10th scene
                if symbol == pyglet.window.key._0:
                    if len(viewer._scene_names) >= 10:
                        viewer._load_scene(9)
                        inner_self._update_vertex_list()
                        return

                # H for help/menu
                if symbol == pyglet.window.key.H:
                    viewer._print_menu()
                    return

                # Let parent handle other keys (Q to quit, etc.)
                super().on_key_press(symbol, modifiers)

        self._viewer = CustomSceneViewer(
            self._trimesh_scene,
            start_loop=False,
            callback=lambda scene: None,
        )

        pyglet.app.run()
