"""PyQt6-based viewer with OpenGL rendering and UI controls."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable

import numpy as np
from OpenGL import GL
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.node import SceneNode

if TYPE_CHECKING:
    from numpy.typing import NDArray

SceneFactory = Callable[[], SceneNode]


class GLWidget(QOpenGLWidget):
    """OpenGL widget for rendering 3D meshes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._meshes: list[dict] = []  # List of mesh data for rendering
        self._pending_scene: SceneNode | None = None  # Scene to load when GL is ready
        self._gl_initialized = False
        self._rotation_x = 20.0
        self._rotation_y = 20.0
        self._zoom = 5.0
        self._pan_x = 0.0
        self._pan_y = -0.4
        self._last_pos = None
        self._default_color = (0.7, 0.7, 0.8, 1.0)

    def set_scene(self, root: SceneNode) -> None:
        """Load a scene for rendering."""
        if self._gl_initialized:
            self._load_scene_data(root)
        else:
            # Defer until GL is ready
            self._pending_scene = root

    def _load_scene_data(self, root: SceneNode) -> None:
        """Actually load scene data (call only after GL init)."""
        self._meshes.clear()

        for scene_node, world_mesh in root.iter_meshes():
            vertices = world_mesh.vertices.astype(np.float32)
            normals = world_mesh.normals.astype(np.float32) if world_mesh.normals is not None else None
            faces = world_mesh.faces.astype(np.uint32)

            # Get color from material or use default
            color = self._default_color
            texture_id = None
            uvs = None

            if world_mesh.material is not None:
                # Create texture from material
                try:
                    tex_data = world_mesh.material.get_texture_array()
                    if tex_data is not None and len(tex_data.shape) >= 2:
                        texture_id = self._create_texture(tex_data)
                        uvs = world_mesh.uvs.astype(np.float32) if world_mesh.uvs is not None else None
                except Exception:
                    pass  # Fall back to default color

            self._meshes.append({
                "vertices": vertices,
                "normals": normals,
                "faces": faces,
                "color": color,
                "texture_id": texture_id,
                "uvs": uvs,
                "name": scene_node.name,
            })

        self.update()

    def _create_texture(self, image: NDArray) -> int:
        """Create an OpenGL texture from image data."""
        texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)

        # Handle different image formats
        if len(image.shape) == 2:
            # Grayscale
            internal_format = GL.GL_RED
            tex_format = GL.GL_RED
            data = image.astype(np.uint8)
        elif image.shape[2] == 3:
            internal_format = GL.GL_RGB
            tex_format = GL.GL_RGB
            data = image.astype(np.uint8)
        else:
            internal_format = GL.GL_RGBA
            tex_format = GL.GL_RGBA
            data = image.astype(np.uint8)

        # Flip vertically for OpenGL
        data = np.flipud(data).copy()

        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, internal_format,
            data.shape[1], data.shape[0], 0,
            tex_format, GL.GL_UNSIGNED_BYTE, data
        )

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)

        return texture_id

    def initializeGL(self) -> None:
        """Initialize OpenGL state."""
        GL.glClearColor(0.2, 0.2, 0.2, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)

        # Set up light
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])

        self._gl_initialized = True

        # Load any pending scene
        if self._pending_scene is not None:
            self._load_scene_data(self._pending_scene)
            self._pending_scene = None

    def resizeGL(self, w: int, h: int) -> None:
        """Handle window resize."""
        GL.glViewport(0, 0, w, h)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()

        aspect = w / h if h > 0 else 1.0
        # Perspective projection
        fov = 45.0
        near = 0.1
        far = 100.0
        top = near * np.tan(np.radians(fov) / 2)
        bottom = -top
        right = top * aspect
        left = -right
        GL.glFrustum(left, right, bottom, top, near, far)

        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paintGL(self) -> None:
        """Render the scene."""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()

        # Camera transform
        GL.glTranslatef(-self._pan_x, -self._pan_y, -self._zoom)
        GL.glRotatef(self._rotation_x, 1, 0, 0)
        GL.glRotatef(self._rotation_y, 0, 1, 0)

        # Render each mesh
        for mesh_data in self._meshes:
            self._render_mesh(mesh_data)

    def _render_mesh(self, mesh_data: dict) -> None:
        """Render a single mesh."""
        vertices = mesh_data["vertices"]
        normals = mesh_data["normals"]
        faces = mesh_data["faces"]
        color = mesh_data["color"]
        texture_id = mesh_data["texture_id"]
        uvs = mesh_data["uvs"]

        if texture_id is not None and uvs is not None:
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
            GL.glColor4f(1.0, 1.0, 1.0, 1.0)  # White to show texture colors
        else:
            GL.glDisable(GL.GL_TEXTURE_2D)
            GL.glColor4f(*color)

        GL.glBegin(GL.GL_TRIANGLES)
        for face in faces:
            for i, vertex_idx in enumerate(face):
                if normals is not None:
                    GL.glNormal3fv(normals[vertex_idx])
                if texture_id is not None and uvs is not None:
                    GL.glTexCoord2fv(uvs[vertex_idx])
                GL.glVertex3fv(vertices[vertex_idx])
        GL.glEnd()

        GL.glDisable(GL.GL_TEXTURE_2D)

    def mousePressEvent(self, event) -> None:
        """Handle mouse press."""
        self._last_pos = event.position()

    def mouseMoveEvent(self, event) -> None:
        """Handle mouse drag for rotation and pan."""
        if self._last_pos is None:
            return

        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()

        if event.buttons() & Qt.MouseButton.LeftButton:
            # Rotate
            self._rotation_y += dx * 0.5
            self._rotation_x += dy * 0.5
        elif event.buttons() & Qt.MouseButton.RightButton:
            # Pan
            self._pan_x -= dx * 0.01
            self._pan_y += dy * 0.01

        self._last_pos = pos
        self.update()

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()
        self._zoom -= delta * 0.005
        self._zoom = max(1.0, min(20.0, self._zoom))
        self.update()


class ViewerWindow(QMainWindow):
    """Main viewer window with scene selection UI."""

    def __init__(
        self,
        scenes: dict[str, SceneFactory],
        default_scene: str | None = None,
    ) -> None:
        super().__init__()
        self._scenes = scenes
        self._scene_names = list(scenes.keys())
        self._current_scene: SceneNode | None = None

        self.setWindowTitle("Geogen Viewer")
        self.resize(1200, 800)

        self._setup_ui()

        # Load initial scene
        initial_index = 0
        if default_scene and default_scene in scenes:
            initial_index = self._scene_names.index(default_scene)
        self._scene_combo.setCurrentIndex(initial_index)
        self._load_scene(initial_index)

    def _setup_ui(self) -> None:
        """Set up the UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel - controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Scene selector
        scene_label = QLabel("Scene:")
        left_layout.addWidget(scene_label)

        self._scene_combo = QComboBox()
        self._scene_combo.addItems(self._scene_names)
        self._scene_combo.currentIndexChanged.connect(self._load_scene)
        left_layout.addWidget(self._scene_combo)

        # Node tree
        tree_label = QLabel("Scene Nodes:")
        left_layout.addWidget(tree_label)

        self._node_tree = QTreeWidget()
        self._node_tree.setHeaderHidden(True)
        left_layout.addWidget(self._node_tree)

        left_panel.setMinimumWidth(200)
        left_panel.setMaximumWidth(300)

        # Right panel - GL view
        self._gl_widget = GLWidget()

        splitter.addWidget(left_panel)
        splitter.addWidget(self._gl_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _load_scene(self, index: int) -> None:
        """Load a scene by index."""
        if index < 0 or index >= len(self._scene_names):
            return

        name = self._scene_names[index]
        factory = self._scenes[name]
        self._current_scene = factory()

        # Update GL widget
        self._gl_widget.set_scene(self._current_scene)

        # Update node tree
        self._update_node_tree()

    def _update_node_tree(self) -> None:
        """Update the node tree widget."""
        self._node_tree.clear()

        if self._current_scene is None:
            return

        def add_node(parent_item: QTreeWidgetItem | None, node: SceneNode) -> None:
            text = node.name
            if node.mesh:
                text += f" ({node.mesh.face_count} faces)"

            if parent_item is None:
                item = QTreeWidgetItem(self._node_tree, [text])
            else:
                item = QTreeWidgetItem(parent_item, [text])

            for child in node.children:
                add_node(item, child)

            item.setExpanded(True)

        add_node(None, self._current_scene)


def run_viewer(
    scenes: dict[str, SceneFactory],
    default_scene: str | None = None,
) -> None:
    """Run the Qt viewer application."""
    # Set up OpenGL format
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)  # Anti-aliasing
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = ViewerWindow(scenes, default_scene)
    window.show()

    # Force initial paint after window is shown
    QTimer.singleShot(100, window._gl_widget.update)

    sys.exit(app.exec())
