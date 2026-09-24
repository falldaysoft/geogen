"""OpenGL viewport widget: PBR shading, shadows, debug display modes, picking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from OpenGL import GL
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QWidget

from ..core import meshops
from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..lighting import DirectionalLight, SceneLighting
from ..materials.material import Material
from .camera import OrbitCamera, ray_mesh_intersect

DISPLAY_MODES = ["Lit", "Clay", "Normals", "UV checker"]
SHADOW_SIZE = 4096
SELECTION_COLOR = (1.0, 0.6, 0.1)
DEFAULT_COLOR = (0.7, 0.7, 0.75, 1.0)


@dataclass
class _GpuMesh:
    node: SceneNode | None
    vertices: np.ndarray
    faces: np.ndarray
    bounds: np.ndarray
    material: Material | None
    vao: int = 0
    buffers: list[int] = field(default_factory=list)


@dataclass
class _GpuTextures:
    albedo: int | None = None
    normal: int | None = None
    roughness: int | None = None
    ao: int | None = None

    def ids(self) -> list[int]:
        return [t for t in (self.albedo, self.normal, self.roughness, self.ao) if t]


def _ortho(left, right, bottom, top, near, far) -> np.ndarray:
    m = np.eye(4)
    m[0, 0] = 2 / (right - left)
    m[1, 1] = 2 / (top - bottom)
    m[2, 2] = -2 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def _look_at_view(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.99 else np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


class GLView(QOpenGLWidget):
    """Interactive 3D viewport for a SceneNode hierarchy."""

    nodeClicked = pyqtSignal(object)  # SceneNode or None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.camera = OrbitCamera()
        self.lighting = SceneLighting.outdoor_lighting()

        # Display options
        self.display_mode = 0
        self.wireframe = False
        self.show_grid = True
        self.show_ground = True
        self.shadows = True
        self.exposure = 1.0

        self._root: SceneNode | None = None
        self._pending_root: SceneNode | None = None
        self._pending_reframe = True
        self._meshes: list[_GpuMesh] = []
        self._textures: dict[int, _GpuTextures] = {}
        self._bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        self._selected: set[int] = set()  # ids of selected nodes (incl. descendants)
        self._selected_root: SceneNode | None = None

        self._gl_ready = False
        self._scene_prog = None
        self._depth_prog = None
        self._line_prog = None
        self._grid_vao = 0
        self._grid_buffers: list[int] = []
        self._grid_counts = (0, 0)
        self._ground: _GpuMesh | None = None
        self._blank_tex = 0
        self._shadow_fbo = 0
        self._shadow_tex = 0
        self._light_space = np.eye(4)
        self._shadow_light = -1

        self._press_pos = None
        self._last_pos = None
        self._dragged = False

    # ------------------------------------------------------------------ scene

    def set_scene(self, root: SceneNode, reframe: bool = True) -> None:
        if not self._gl_ready:
            self._pending_root, self._pending_reframe = root, reframe
            return
        self.makeCurrent()
        self._upload_scene(root)
        if reframe:
            self.frame_all()
        self.doneCurrent()
        self.update()

    def scene_bounds(self) -> np.ndarray:
        return self._bounds.copy()

    def select(self, node: SceneNode | None) -> None:
        self._selected_root = node
        self._selected = {id(n) for n in node.iter_nodes()} if node is not None else set()
        self.update()

    def node_bounds(self, node: SceneNode) -> np.ndarray | None:
        ids = {id(n) for n in node.iter_nodes()}
        boxes = [m.bounds for m in self._meshes if id(m.node) in ids]
        if not boxes:
            return None
        stack = np.array(boxes)
        return np.array([stack[:, 0].min(axis=0), stack[:, 1].max(axis=0)])

    def frame_all(self) -> None:
        self.camera.frame(self._bounds)
        self.update()

    def frame_selected(self) -> None:
        bounds = self.node_bounds(self._selected_root) if self._selected_root is not None else None
        self.camera.frame(bounds if bounds is not None else self._bounds)
        self.update()

    def set_view(self, preset: str) -> None:
        self.camera.set_preset(preset)
        self.update()

    def _upload_scene(self, root: SceneNode) -> None:
        self._release_scene()
        self._root = root
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for node, world_mesh in root.iter_meshes():
            if len(world_mesh.faces) == 0:
                continue
            mesh = meshops.ensure_normals(world_mesh)
            gpu = _GpuMesh(
                node=node,
                vertices=mesh.vertices,
                faces=mesh.faces,
                bounds=np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)]),
                material=mesh.material,
            )
            gpu.vao, gpu.buffers = self._create_vao(mesh)

            if mesh.material is not None and id(mesh.material) not in self._textures:
                self._textures[id(mesh.material)] = self._upload_material(mesh.material)
            self._meshes.append(gpu)
            lo = np.minimum(lo, gpu.bounds[0])
            hi = np.maximum(hi, gpu.bounds[1])

        self._bounds = np.array([lo, hi]) if np.all(np.isfinite(lo)) else np.array([[-1.0] * 3, [1.0] * 3])
        self.camera.scene_radius = float(np.linalg.norm(self._bounds[1] - self._bounds[0])) / 2
        self._build_grid()
        self._build_ground()
        if self._selected_root is not None and not any(n is self._selected_root for n in root.iter_nodes()):
            self.select(None)

    @staticmethod
    def _create_vao(mesh) -> tuple[int, list[int]]:
        vertices = mesh.vertices.astype(np.float32)
        uvs = mesh.uvs.astype(np.float32) if mesh.uvs is not None else np.zeros((len(vertices), 2), np.float32)
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        buffers = []
        for location, data, width in ((0, vertices, 3), (1, mesh.normals.astype(np.float32), 3), (2, uvs, 2)):
            vbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(location, width, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(location)
            buffers.append(vbo)
        ebo = GL.glGenBuffers(1)
        indices = mesh.faces.astype(np.uint32)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
        buffers.append(ebo)
        GL.glBindVertexArray(0)
        return vao, buffers

    def _build_ground(self) -> None:
        """Large neutral plane just under the scene that catches shadows."""
        if self._ground is not None:
            GL.glDeleteVertexArrays(1, [self._ground.vao])
            GL.glDeleteBuffers(len(self._ground.buffers), self._ground.buffers)
        radius = max(self.camera.scene_radius, 0.5) * 6
        c = self._bounds.mean(axis=0)
        y = float(self._bounds[0][1]) - 2e-3
        v = np.array([[c[0] - radius, y, c[2] - radius], [c[0] + radius, y, c[2] - radius],
                      [c[0] + radius, y, c[2] + radius], [c[0] - radius, y, c[2] + radius]])
        mesh = Mesh(v, np.array([[0, 2, 1], [0, 3, 2]]), normals=np.tile([0.0, 1.0, 0.0], (4, 1)))
        self._ground = _GpuMesh(node=None, vertices=v, faces=mesh.faces, bounds=np.array([v.min(0), v.max(0)]),
                                material=None)
        self._ground.vao, self._ground.buffers = self._create_vao(mesh)

    def _release_scene(self) -> None:
        for m in self._meshes:
            GL.glDeleteVertexArrays(1, [m.vao])
            GL.glDeleteBuffers(len(m.buffers), m.buffers)
        for tex in self._textures.values():
            if tex.ids():
                GL.glDeleteTextures(len(tex.ids()), tex.ids())
        self._meshes.clear()
        self._textures.clear()

    def _upload_material(self, material: Material) -> _GpuTextures:
        textures = _GpuTextures()
        try:
            textures.albedo = self._create_texture(material.get_texture_array())
            normal = material.get_normal_array()
            if normal is not None:
                textures.normal = self._create_texture(normal)
            rough = material.get_roughness_array()
            if rough is not None:
                textures.roughness = self._create_texture(rough)
            ao = material.get_ao_array()
            if ao is not None:
                textures.ao = self._create_texture(ao)
        except Exception as exc:  # texture generation should never kill the viewer
            print(f"Texture generation failed for material '{material.name}': {exc}")
        return textures

    @staticmethod
    def _create_texture(image: np.ndarray) -> int:
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        data = np.ascontiguousarray(np.flipud(image.astype(np.uint8)))
        fmt = GL.GL_RED if data.ndim == 2 else {3: GL.GL_RGB, 4: GL.GL_RGBA}[data.shape[2]]
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, fmt, data.shape[1], data.shape[0], 0, fmt, GL.GL_UNSIGNED_BYTE, data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        try:  # anisotropic filtering keeps tiled ground textures crisp at grazing angles
            GL.glTexParameterf(GL.GL_TEXTURE_2D, 0x84FE, 8.0)  # GL_TEXTURE_MAX_ANISOTROPY
        except Exception:
            pass
        return tex

    def _build_grid(self) -> None:
        if self._grid_vao:
            GL.glDeleteVertexArrays(1, [self._grid_vao])
            GL.glDeleteBuffers(len(self._grid_buffers), self._grid_buffers)
        radius = max(self.camera.scene_radius, 0.5)
        step = 10 ** np.floor(np.log10(radius / 2))  # ~10-100 lines across the scene
        half = np.ceil(radius * 3 / step) * step
        y = float(self._bounds[0][1]) if self._root is not None else 0.0
        y = min(y, 0.0) if abs(y) < radius * 0.05 else y
        center = np.round(self._bounds.mean(axis=0) / step) * step

        lines, colors = [], []
        n = int(half / step)
        for i in range(-n, n + 1):
            major = i % 10 == 0
            c = (0.35, 0.35, 0.38, 0.55) if major else (0.45, 0.45, 0.48, 0.3)
            x = center[0] + i * step
            z = center[2] + i * step
            lines += [(x, y, center[2] - half), (x, y, center[2] + half)]
            lines += [(center[0] - half, y, z), (center[0] + half, y, z)]
            colors += [c] * 4
        grid_count = len(lines)
        axis_len = max(step * 2, radius * 0.25)
        for axis, color in ((0, (0.9, 0.2, 0.2, 1.0)), (1, (0.2, 0.8, 0.2, 1.0)), (2, (0.2, 0.4, 0.95, 1.0))):
            end = [0.0, y, 0.0]
            end[axis] += axis_len
            lines += [(0.0, y, 0.0), tuple(end)]
            colors += [color] * 2
        positions = np.array(lines, dtype=np.float32)
        cols = np.array(colors, dtype=np.float32)

        self._grid_vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._grid_vao)
        self._grid_buffers = list(GL.glGenBuffers(2))
        for location, data, width, vbo in ((0, positions, 3, self._grid_buffers[0]), (1, cols, 4, self._grid_buffers[1])):
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(location, width, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(location)
        GL.glBindVertexArray(0)
        self._grid_counts = (grid_count, len(lines) - grid_count)

    # --------------------------------------------------------------- GL setup

    def initializeGL(self) -> None:
        from .shaders import ShaderCompiler

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_MULTISAMPLE)
        self._scene_prog = ShaderCompiler.load("scene.vert", "scene.frag")
        self._depth_prog = ShaderCompiler.load("depth.vert", "depth.frag")
        self._line_prog = ShaderCompiler.load("line.vert", "line.frag")

        self._shadow_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._shadow_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT24, SHADOW_SIZE, SHADOW_SIZE, 0,
                        GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        for pname, value in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                             (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, pname, value)
        self._shadow_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._shadow_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self._shadow_tex, 0)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())

        self._blank_tex = self._create_texture(np.full((1, 1, 3), 255, dtype=np.uint8))
        self._gl_ready = True
        self._build_grid()
        if self._pending_root is not None:
            self._upload_scene(self._pending_root)
            if self._pending_reframe:
                self.camera.frame(self._bounds)
            self._pending_root = None

    def resizeGL(self, w: int, h: int) -> None:
        self.camera.aspect = w / h if h > 0 else 1.0

    # ---------------------------------------------------------------- drawing

    def _sun(self) -> tuple[int, np.ndarray | None]:
        for i, light in enumerate(self.lighting.lights[:4]):
            if isinstance(light, DirectionalLight):
                d = np.array(light.direction, dtype=np.float64)
                return i, d / np.linalg.norm(d)
        return -1, None

    def _render_shadow_map(self) -> None:
        index, direction = self._sun()
        self._shadow_light = index if (self.shadows and direction is not None) else -1
        if self._shadow_light < 0:
            return
        center = self._bounds.mean(axis=0)
        radius = max(self.camera.scene_radius, 0.1) * 1.05
        view = _look_at_view(center - direction * radius * 2, center)
        self._light_space = _ortho(-radius, radius, -radius, radius, 0.01, radius * 4) @ view

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._shadow_fbo)
        GL.glViewport(0, 0, SHADOW_SIZE, SHADOW_SIZE)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(2.0, 4.0)
        self._depth_prog.use()
        self._depth_prog.set_uniform("uLightSpace", self._light_space)
        for m in self._meshes:
            GL.glBindVertexArray(m.vao)
            GL.glDrawElements(GL.GL_TRIANGLES, m.faces.size, GL.GL_UNSIGNED_INT, None)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())

    def paintGL(self) -> None:
        if not self._gl_ready:
            return
        self._render_shadow_map()
        ratio = self.devicePixelRatioF()
        GL.glViewport(0, 0, int(self.width() * ratio), int(self.height() * ratio))
        GL.glClearColor(0.62, 0.74, 0.86, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix()
        eye = self.camera.eye().astype(np.float32)

        if self.wireframe:
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(1.0, 1.0)
        self._draw_meshes(view, proj, eye)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        self._line_prog.use()
        self._line_prog.set_uniform("uView", view)
        self._line_prog.set_uniform("uProjection", proj)
        self._line_prog.set_uniform("uCameraPos", eye)
        self._line_prog.set_uniform("uUseVertexColor", False)
        self._line_prog.set_uniform("uFadeDistance", 0.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        for m in self._meshes:
            selected = id(m.node) in self._selected
            if not (self.wireframe or selected):
                continue
            color = (*SELECTION_COLOR, 1.0) if selected else (0.05, 0.05, 0.08, 0.6)
            self._line_prog.set_uniform("uColor", color)
            GL.glBindVertexArray(m.vao)
            GL.glDrawElements(GL.GL_TRIANGLES, m.faces.size, GL.GL_UNSIGNED_INT, None)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        if self.show_grid and self._grid_vao:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDepthMask(GL.GL_FALSE)
            self._line_prog.set_uniform("uUseVertexColor", True)
            self._line_prog.set_uniform("uFadeDistance", float(max(self.camera.scene_radius, 0.5) * 3))
            GL.glBindVertexArray(self._grid_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self._grid_counts[0])
            self._line_prog.set_uniform("uFadeDistance", 0.0)
            GL.glDrawArrays(GL.GL_LINES, self._grid_counts[0], self._grid_counts[1])
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def _draw_meshes(self, view: np.ndarray, proj: np.ndarray, eye: np.ndarray) -> None:
        prog = self._scene_prog
        prog.use()
        prog.set_uniform("uView", view)
        prog.set_uniform("uProjection", proj)
        prog.set_uniform("uLightSpace", self._light_space)
        prog.set_uniform("uCameraPos", eye)
        prog.set_uniform("uDisplayMode", int(self.display_mode))
        prog.set_uniform("uShadowsEnabled", self._shadow_light >= 0)
        prog.set_uniform("uShadowLight", int(self._shadow_light))
        prog.set_uniform("uExposure", float(self.exposure))
        prog.set_uniform("uSkyColor", np.array([0.42, 0.48, 0.58]))
        prog.set_uniform("uGroundColor", np.array([0.2, 0.18, 0.16]))

        lights = self.lighting.get_shader_data()
        prog.set_uniform("uLightCount", int(lights["uLightCount"]))
        for i in range(lights["uLightCount"]):
            prog.set_uniform(f"uLightTypes[{i}]", int(lights["uLightTypes"][i]))
            prog.set_uniform(f"uLightPositions[{i}]", np.array(lights["uLightPositions"][i]))
            prog.set_uniform(f"uLightColors[{i}]", np.array(lights["uLightColors"][i]))
            prog.set_uniform(f"uLightIntensities[{i}]", float(lights["uLightIntensities"][i]))

        GL.glActiveTexture(GL.GL_TEXTURE4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._shadow_tex)
        prog.set_uniform("uShadowMap", 4)
        for unit, name in enumerate(("uAlbedoMap", "uNormalMap", "uRoughnessMap", "uAOMap")):
            prog.set_uniform(name, unit)

        # Opaque first, then alpha-blended (glass) without depth writes.
        ordered = sorted(self._meshes, key=lambda m: bool(m.material is not None and m.material.transparent))
        for m in ordered:
            mat = m.material
            tex = self._textures.get(id(mat)) if mat is not None else None
            transparent = mat is not None and mat.transparent
            if transparent:
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glDepthMask(GL.GL_FALSE)
            prog.set_uniform("uOpacity", float(mat.opacity) if mat else 1.0)
            prog.set_uniform("uEmissive", np.array(mat.emissive) * mat.emissive_strength if mat else np.zeros(3))
            prog.set_uniform("uBaseColor", DEFAULT_COLOR)
            prog.set_uniform("uRoughness", float(mat.roughness) if mat else 0.6)
            prog.set_uniform("uMetallic", float(mat.metallic) if mat else 0.0)
            prog.set_uniform("uNormalStrength", float(mat.normal_strength) if mat else 1.0)
            prog.set_uniform("uAOStrength", float(mat.ao_strength) if mat else 1.0)
            prog.set_uniform("uUVScale", np.array(mat.texture_uv_scale if mat else (1.0, 1.0)))
            highlight = (*SELECTION_COLOR, 0.18) if id(m.node) in self._selected else (0.0, 0.0, 0.0, 0.0)
            prog.set_uniform("uHighlight", highlight)
            for unit, (flag, tex_id) in enumerate((
                ("uHasAlbedoMap", tex.albedo if tex else None),
                ("uHasNormalMap", tex.normal if tex else None),
                ("uHasRoughnessMap", tex.roughness if tex else None),
                ("uHasAOMap", tex.ao if tex else None),
            )):
                prog.set_uniform(flag, bool(tex_id))
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id or self._blank_tex)
            GL.glBindVertexArray(m.vao)
            GL.glDrawElements(GL.GL_TRIANGLES, m.faces.size, GL.GL_UNSIGNED_INT, None)
            if transparent:
                GL.glDepthMask(GL.GL_TRUE)
                GL.glDisable(GL.GL_BLEND)

        if self.show_ground and self._ground is not None:
            prog.set_uniform("uBaseColor", (0.78, 0.78, 0.76, 1.0))
            prog.set_uniform("uOpacity", 1.0)
            prog.set_uniform("uEmissive", np.zeros(3))
            prog.set_uniform("uRoughness", 0.95)
            prog.set_uniform("uMetallic", 0.0)
            prog.set_uniform("uHighlight", (0.0, 0.0, 0.0, 0.0))
            if self.display_mode == 3:
                prog.set_uniform("uDisplayMode", 1)  # plain clay ground under the UV checker
            for flag in ("uHasAlbedoMap", "uHasNormalMap", "uHasRoughnessMap", "uHasAOMap"):
                prog.set_uniform(flag, False)
            GL.glBindVertexArray(self._ground.vao)
            GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)

    # ----------------------------------------------------------------- picking

    def pick(self, x: float, y: float) -> tuple[SceneNode | None, np.ndarray | None]:
        """Return the node and world point under widget pixel (x, y)."""
        origin, direction = self.camera.ray(x, y, self.width(), self.height())
        inv = 1.0 / np.where(np.abs(direction) < 1e-12, 1e-12, direction)
        best_t, best_node = np.inf, None
        for m in self._meshes:
            t1 = (m.bounds[0] - origin) * inv
            t2 = (m.bounds[1] - origin) * inv
            t_near = np.max(np.minimum(t1, t2))
            t_far = np.min(np.maximum(t1, t2))
            if t_far < max(t_near, 0.0) or t_near > best_t:
                continue
            t = ray_mesh_intersect(origin, direction, m.vertices, m.faces)
            if t is not None and t < best_t:
                best_t, best_node = t, m.node
        if best_node is None:
            return None, None
        return best_node, origin + direction * best_t

    # ------------------------------------------------------------------ input

    def mousePressEvent(self, event) -> None:
        self._press_pos = self._last_pos = event.position()
        self._dragged = False

    def mouseMoveEvent(self, event) -> None:
        if self._last_pos is None:
            return
        pos = event.position()
        dx, dy = pos.x() - self._last_pos.x(), pos.y() - self._last_pos.y()
        if (pos - self._press_pos).manhattanLength() > 3:
            self._dragged = True
        buttons = event.buttons()
        mods = event.modifiers()
        pan = (buttons & Qt.MouseButton.RightButton or buttons & Qt.MouseButton.MiddleButton
               or (buttons & Qt.MouseButton.LeftButton and mods & Qt.KeyboardModifier.ShiftModifier))
        if pan:
            self.camera.pan(dx, dy, self.height())
        elif buttons & Qt.MouseButton.LeftButton:
            self.camera.orbit(dx, dy)
        self._last_pos = pos
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if not self._dragged and event.button() == Qt.MouseButton.LeftButton:
            node, _ = self.pick(event.position().x(), event.position().y())
            self.nodeClicked.emit(node)
        self._last_pos = None

    def mouseDoubleClickEvent(self, event) -> None:
        node, point = self.pick(event.position().x(), event.position().y())
        if node is not None:
            self.nodeClicked.emit(node)
            self.frame_selected()

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        _, point = self.pick(event.position().x(), event.position().y())
        self.camera.zoom(steps, toward=point)
        self.update()
