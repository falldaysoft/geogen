"""PyQt6-based viewer with modern OpenGL rendering and UI controls."""

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
from ..lighting import SceneLighting

if TYPE_CHECKING:
    from numpy.typing import NDArray

SceneFactory = Callable[[], SceneNode]


class GLWidget(QOpenGLWidget):
    """OpenGL widget for rendering 3D meshes with modern shaders."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._meshes: list[dict] = []
        self._pending_scene: SceneNode | None = None
        self._gl_initialized = False
        self._use_modern_gl = True

        # Camera
        self._rotation_x = 20.0
        self._rotation_y = 20.0
        self._zoom = 5.0
        self._pan_x = 0.0
        self._pan_y = -0.4
        self._last_pos = None

        # Rendering
        self._default_color = (0.7, 0.7, 0.8, 1.0)
        self._shader = None
        self._lighting = SceneLighting.room_lighting()
        self._aspect = 1.0
        self._use_pbr = False

    def set_lighting(self, lighting: SceneLighting) -> None:
        """Set the scene lighting configuration."""
        self._lighting = lighting
        self.update()

    def set_scene(self, root: SceneNode) -> None:
        """Load a scene for rendering."""
        if self._gl_initialized:
            self._load_scene_data(root)
        else:
            self._pending_scene = root

    def _load_scene_data(self, root: SceneNode) -> None:
        """Load scene data into GPU buffers."""
        # Clean up old mesh data
        for mesh_data in self._meshes:
            self._delete_mesh_buffers(mesh_data)
        self._meshes.clear()

        for scene_node, world_mesh in root.iter_meshes():
            vertices = world_mesh.vertices.astype(np.float32)
            normals = (
                world_mesh.normals.astype(np.float32)
                if world_mesh.normals is not None
                else self._compute_flat_normals(vertices, world_mesh.faces)
            )
            faces = world_mesh.faces.astype(np.uint32)
            uvs = (
                world_mesh.uvs.astype(np.float32)
                if world_mesh.uvs is not None
                else np.zeros((len(vertices), 2), dtype=np.float32)
            )

            # Get material properties
            color = self._default_color
            texture_id = None
            normal_id = None
            roughness_id = None
            ao_id = None
            roughness = 0.5
            metallic = 0.0
            normal_strength = 1.0
            ao_strength = 1.0

            if world_mesh.material is not None:
                mat = world_mesh.material
                roughness = mat.roughness
                metallic = mat.metallic
                normal_strength = mat.normal_strength
                ao_strength = mat.ao_strength

                try:
                    # Albedo texture
                    tex_data = mat.get_texture_array()
                    if tex_data is not None and len(tex_data.shape) >= 2:
                        texture_id = self._create_texture(tex_data)

                    # Normal map
                    normal_data = mat.get_normal_array()
                    if normal_data is not None and len(normal_data.shape) >= 2:
                        normal_id = self._create_texture(normal_data)

                    # AO map
                    ao_data = mat.get_ao_array()
                    if ao_data is not None:
                        ao_id = self._create_texture(ao_data)
                except Exception:
                    pass

            # Create VAO/VBO for modern rendering
            mesh_data = {
                "vertices": vertices,
                "normals": normals,
                "uvs": uvs,
                "faces": faces,
                "color": color,
                "texture_id": texture_id,
                "normal_id": normal_id,
                "roughness_id": roughness_id,
                "ao_id": ao_id,
                "roughness": roughness,
                "metallic": metallic,
                "normal_strength": normal_strength,
                "ao_strength": ao_strength,
                "name": scene_node.name,
                "vao": None,
                "vbo_vertices": None,
                "vbo_normals": None,
                "vbo_uvs": None,
                "ebo": None,
            }

            if self._use_modern_gl:
                self._create_mesh_buffers(mesh_data)

            self._meshes.append(mesh_data)

        self.update()

    def _compute_flat_normals(
        self, vertices: NDArray, faces: NDArray
    ) -> NDArray[np.float32]:
        """Compute flat normals when mesh has none."""
        normals = np.zeros_like(vertices)
        for face in faces:
            v0, v1, v2 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            normals[face] += normal
        # Normalize accumulated normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (normals / norms).astype(np.float32)

    def _create_mesh_buffers(self, mesh_data: dict) -> None:
        """Create VAO and VBOs for a mesh."""
        vertices = mesh_data["vertices"]
        normals = mesh_data["normals"]
        uvs = mesh_data["uvs"]
        faces = mesh_data["faces"]

        # Create VAO
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)

        # Vertex positions (location 0)
        vbo_vertices = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_vertices)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        # Normals (location 1)
        vbo_normals = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_normals)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, normals.nbytes, normals, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(1)

        # UVs (location 2)
        vbo_uvs = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_uvs)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(2)

        # Element buffer (indices)
        ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL.GL_STATIC_DRAW
        )

        GL.glBindVertexArray(0)

        mesh_data["vao"] = vao
        mesh_data["vbo_vertices"] = vbo_vertices
        mesh_data["vbo_normals"] = vbo_normals
        mesh_data["vbo_uvs"] = vbo_uvs
        mesh_data["ebo"] = ebo

    def _delete_mesh_buffers(self, mesh_data: dict) -> None:
        """Delete VAO and VBOs for a mesh."""
        if mesh_data.get("vao"):
            GL.glDeleteVertexArrays(1, [mesh_data["vao"]])
        for key in ["vbo_vertices", "vbo_normals", "vbo_uvs", "ebo"]:
            if mesh_data.get(key):
                GL.glDeleteBuffers(1, [mesh_data[key]])
        for tex_key in ["texture_id", "normal_id", "roughness_id", "ao_id"]:
            if mesh_data.get(tex_key):
                GL.glDeleteTextures(1, [mesh_data[tex_key]])

    def _create_texture(self, image: NDArray) -> int:
        """Create an OpenGL texture from image data."""
        texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)

        if len(image.shape) == 2:
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
            GL.GL_TEXTURE_2D,
            0,
            internal_format,
            data.shape[1],
            data.shape[0],
            0,
            tex_format,
            GL.GL_UNSIGNED_BYTE,
            data,
        )

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)

        return texture_id

    def _init_shaders(self) -> bool:
        """Initialize shader program."""
        try:
            from .shaders import ShaderCompiler

            # Try PBR shader first, fall back to Blinn-Phong
            try:
                self._shader = ShaderCompiler.load("basic.vert", "pbr.frag")
                self._use_pbr = True
            except Exception:
                self._shader = ShaderCompiler.load("basic.vert", "blinn_phong.frag")
                self._use_pbr = False
            return True
        except Exception as e:
            print(f"Shader compilation failed, falling back to legacy: {e}")
            self._use_modern_gl = False
            return False

    def initializeGL(self) -> None:
        """Initialize OpenGL state."""
        GL.glClearColor(0.2, 0.2, 0.2, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)

        # Try to initialize modern shaders
        if self._use_modern_gl:
            self._use_modern_gl = self._init_shaders()

        # Legacy fallback setup
        if not self._use_modern_gl:
            GL.glEnable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_LIGHT0)
            GL.glEnable(GL.GL_COLOR_MATERIAL)
            GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1.0, 1.0, 1.0, 0.0])
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])

        self._gl_initialized = True

        if self._pending_scene is not None:
            self._load_scene_data(self._pending_scene)
            self._pending_scene = None

    def resizeGL(self, w: int, h: int) -> None:
        """Handle window resize."""
        GL.glViewport(0, 0, w, h)
        self._aspect = w / h if h > 0 else 1.0

        if not self._use_modern_gl:
            # Legacy matrix setup
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            fov = 45.0
            near = 0.1
            far = 100.0
            top = near * np.tan(np.radians(fov) / 2)
            bottom = -top
            right = top * self._aspect
            left = -right
            GL.glFrustum(left, right, bottom, top, near, far)
            GL.glMatrixMode(GL.GL_MODELVIEW)

    def _get_view_matrix(self) -> NDArray[np.float32]:
        """Compute the view matrix from camera parameters."""
        # Translation
        trans = np.eye(4, dtype=np.float32)
        trans[0, 3] = -self._pan_x
        trans[1, 3] = -self._pan_y
        trans[2, 3] = -self._zoom

        # Rotation X
        rx = np.radians(self._rotation_x)
        rot_x = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(rx), -np.sin(rx), 0],
                [0, np.sin(rx), np.cos(rx), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Rotation Y
        ry = np.radians(self._rotation_y)
        rot_y = np.array(
            [
                [np.cos(ry), 0, np.sin(ry), 0],
                [0, 1, 0, 0],
                [-np.sin(ry), 0, np.cos(ry), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        return trans @ rot_x @ rot_y

    def _get_projection_matrix(self) -> NDArray[np.float32]:
        """Compute the perspective projection matrix."""
        fov = np.radians(45.0)
        near = 0.1
        far = 100.0
        f = 1.0 / np.tan(fov / 2)

        return np.array(
            [
                [f / self._aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    def _get_camera_position(self) -> NDArray[np.float32]:
        """Get camera position in world space."""
        # Inverse of view transform
        view = self._get_view_matrix()
        inv_view = np.linalg.inv(view)
        return inv_view[:3, 3].astype(np.float32)

    def paintGL(self) -> None:
        """Render the scene."""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        if self._use_modern_gl and self._shader:
            self._render_modern()
        else:
            self._render_legacy()

    def _render_modern(self) -> None:
        """Render using modern shaders."""
        self._shader.use()

        # Set matrices
        model = np.eye(4, dtype=np.float32)
        view = self._get_view_matrix()
        projection = self._get_projection_matrix()
        normal_matrix = np.linalg.inv(model[:3, :3]).T

        self._shader.set_uniform("uModel", model)
        self._shader.set_uniform("uView", view)
        self._shader.set_uniform("uProjection", projection)
        self._shader.set_uniform("uNormalMatrix", normal_matrix)
        self._shader.set_uniform("uCameraPos", self._get_camera_position())

        # Set lighting uniforms
        light_data = self._lighting.get_shader_data()
        self._shader.set_uniform("uAmbientColor", light_data["uAmbientColor"])
        self._shader.set_uniform("uLightCount", light_data["uLightCount"])

        for i in range(light_data["uLightCount"]):
            self._shader.set_uniform(f"uLightTypes[{i}]", light_data["uLightTypes"][i])
            self._shader.set_uniform(
                f"uLightPositions[{i}]", light_data["uLightPositions"][i]
            )
            self._shader.set_uniform(f"uLightColors[{i}]", light_data["uLightColors"][i])
            self._shader.set_uniform(
                f"uLightIntensities[{i}]", light_data["uLightIntensities"][i]
            )

        # Render meshes
        for mesh_data in self._meshes:
            self._render_mesh_modern(mesh_data)

        GL.glUseProgram(0)

    def _render_mesh_modern(self, mesh_data: dict) -> None:
        """Render a single mesh using shaders."""
        if mesh_data["vao"] is None:
            return

        # Set material uniforms - handle both PBR and Blinn-Phong shaders
        has_albedo = mesh_data["texture_id"] is not None
        has_normal = mesh_data.get("normal_id") is not None
        has_roughness = mesh_data.get("roughness_id") is not None
        has_ao = mesh_data.get("ao_id") is not None

        # Common uniforms
        self._shader.set_uniform("uBaseColor", mesh_data["color"])

        if getattr(self, "_use_pbr", False):
            # PBR shader uniforms
            self._shader.set_uniform("uHasAlbedoMap", has_albedo)
            self._shader.set_uniform("uHasNormalMap", has_normal)
            self._shader.set_uniform("uHasRoughnessMap", has_roughness)
            self._shader.set_uniform("uHasAOMap", has_ao)
            self._shader.set_uniform("uRoughness", mesh_data.get("roughness", 0.5))
            self._shader.set_uniform("uMetallic", mesh_data.get("metallic", 0.0))
            self._shader.set_uniform("uNormalStrength", mesh_data.get("normal_strength", 1.0))
            self._shader.set_uniform("uAOStrength", mesh_data.get("ao_strength", 1.0))
        else:
            # Blinn-Phong shader uniforms
            self._shader.set_uniform("uHasTexture", has_albedo)
            # Convert roughness to shininess for Blinn-Phong
            roughness = mesh_data.get("roughness", 0.5)
            shininess = 1.0 - roughness
            self._shader.set_uniform("uShininess", shininess)

        # Bind textures
        tex_unit = 0

        if has_albedo:
            GL.glActiveTexture(GL.GL_TEXTURE0 + tex_unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, mesh_data["texture_id"])
            self._shader.set_uniform("uAlbedoMap", tex_unit)
            tex_unit += 1

        if has_normal and getattr(self, "_use_pbr", False):
            GL.glActiveTexture(GL.GL_TEXTURE0 + tex_unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, mesh_data["normal_id"])
            self._shader.set_uniform("uNormalMap", tex_unit)
            tex_unit += 1

        if has_roughness and getattr(self, "_use_pbr", False):
            GL.glActiveTexture(GL.GL_TEXTURE0 + tex_unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, mesh_data["roughness_id"])
            self._shader.set_uniform("uRoughnessMap", tex_unit)
            tex_unit += 1

        if has_ao and getattr(self, "_use_pbr", False):
            GL.glActiveTexture(GL.GL_TEXTURE0 + tex_unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, mesh_data["ao_id"])
            self._shader.set_uniform("uAOMap", tex_unit)

        GL.glBindVertexArray(mesh_data["vao"])
        GL.glDrawElements(
            GL.GL_TRIANGLES,
            len(mesh_data["faces"]) * 3,
            GL.GL_UNSIGNED_INT,
            None,
        )
        GL.glBindVertexArray(0)

    def _render_legacy(self) -> None:
        """Render using legacy fixed-function pipeline."""
        GL.glLoadIdentity()
        GL.glTranslatef(-self._pan_x, -self._pan_y, -self._zoom)
        GL.glRotatef(self._rotation_x, 1, 0, 0)
        GL.glRotatef(self._rotation_y, 0, 1, 0)

        for mesh_data in self._meshes:
            self._render_mesh_legacy(mesh_data)

    def _render_mesh_legacy(self, mesh_data: dict) -> None:
        """Render a single mesh using legacy OpenGL."""
        vertices = mesh_data["vertices"]
        normals = mesh_data["normals"]
        faces = mesh_data["faces"]
        color = mesh_data["color"]
        texture_id = mesh_data["texture_id"]
        uvs = mesh_data["uvs"]

        if texture_id is not None:
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
            GL.glColor4f(1.0, 1.0, 1.0, 1.0)
        else:
            GL.glDisable(GL.GL_TEXTURE_2D)
            GL.glColor4f(*color)

        GL.glBegin(GL.GL_TRIANGLES)
        for face in faces:
            for vertex_idx in face:
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
            self._rotation_y += dx * 0.5
            self._rotation_x += dy * 0.5
        elif event.buttons() & Qt.MouseButton.RightButton:
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

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        scene_label = QLabel("Scene:")
        left_layout.addWidget(scene_label)

        self._scene_combo = QComboBox()
        self._scene_combo.addItems(self._scene_names)
        self._scene_combo.currentIndexChanged.connect(self._load_scene)
        left_layout.addWidget(self._scene_combo)

        tree_label = QLabel("Scene Nodes:")
        left_layout.addWidget(tree_label)

        self._node_tree = QTreeWidget()
        self._node_tree.setHeaderHidden(True)
        left_layout.addWidget(self._node_tree)

        left_panel.setMinimumWidth(200)
        left_panel.setMaximumWidth(300)

        # GL view
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

        self._gl_widget.set_scene(self._current_scene)
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
    # Set up OpenGL format with core profile
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = ViewerWindow(scenes, default_scene)
    window.show()

    QTimer.singleShot(100, window._gl_widget.update)

    sys.exit(app.exec())
