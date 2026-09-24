"""Offscreen rendering of scene graphs with PBR materials and shadows.

Used by the CLI (``geogen -r``) and by tests/tools that need to look at
generated geometry. Builds a pyrender scene directly from SceneNode meshes so
that normals, UV tiling and PBR maps (normal, roughness, metallic) are all
honoured, and supports a multi-view contact sheet for inspecting an asset
from several sides at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyrender
from PIL import Image

from .core import meshops
from .core.mesh import Mesh
from .core.node import SceneNode
from .materials.material import Material

# pyrender 0.1.45 (latest release) still references np.infty, removed in NumPy 2.
if not hasattr(np, "infty"):
    np.infty = np.inf

DEFAULT_COLOR = (0.7, 0.7, 0.75, 1.0)
BACKGROUND = (0.55, 0.7, 0.85, 1.0)

# View name -> (azimuth degrees around +Y from +Z, elevation degrees)
VIEWS: dict[str, tuple[float, float]] = {
    "iso": (35.0, 25.0),
    "front": (0.0, 5.0),
    "side": (90.0, 5.0),
    "back": (180.0, 5.0),
    "top": (0.0, 89.0),
    "iso_back": (215.0, 25.0),
}


@dataclass
class RenderOptions:
    width: int = 1280
    height: int = 960
    fov: float = 45.0
    shadows: bool = True
    ground: bool = True
    camera: np.ndarray | None = None
    target: np.ndarray | None = None
    zoom: float = 1.0


_renderer: pyrender.OffscreenRenderer | None = None


def _get_renderer(width: int, height: int) -> pyrender.OffscreenRenderer:
    """Share one offscreen context (recreating it breaks pyglet on macOS)."""
    global _renderer
    if _renderer is None:
        _renderer = pyrender.OffscreenRenderer(width, height)
    else:
        _renderer.viewport_width = width
        _renderer.viewport_height = height
    return _renderer


def _texture(image: Image.Image, channels: str) -> pyrender.Texture:
    sampler = pyrender.Sampler(wrapS=pyrender.constants.GLTF.REPEAT, wrapT=pyrender.constants.GLTF.REPEAT)
    return pyrender.Texture(source=np.asarray(image), source_channels=channels, sampler=sampler)


class _MaterialCache:
    def __init__(self) -> None:
        self._cache: dict[int, pyrender.Material] = {}
        self._default = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=DEFAULT_COLOR, metallicFactor=0.0, roughnessFactor=0.6
        )

    def get(self, material: Material | None) -> pyrender.Material:
        if material is None:
            return self._default
        key = id(material)
        if key not in self._cache:
            self._cache[key] = self._build(material)
        return self._cache[key]

    @staticmethod
    def _build(material: Material) -> pyrender.Material:
        images = material.gltf_images()
        kwargs: dict = {
            "baseColorTexture": _texture(images["base_color"], "RGB"),
            "metallicRoughnessTexture": _texture(images["metallic_roughness"], "RGB"),
            "metallicFactor": 1.0,
            "roughnessFactor": 1.0,
        }
        if "normal" in images:
            kwargs["normalTexture"] = _texture(images["normal"], "RGB")
        if "occlusion" in images:
            kwargs["occlusionTexture"] = _texture(images["occlusion"], "RGB")
        return pyrender.MetallicRoughnessMaterial(name=material.name, **kwargs)


def _primitive(mesh: Mesh, materials: _MaterialCache) -> pyrender.Primitive:
    mesh = meshops.ensure_normals(mesh)
    texcoords = None
    tangents = None
    if mesh.uvs is not None:
        scale = np.array(mesh.material.texture_uv_scale if mesh.material is not None else (1.0, 1.0))
        texcoords = mesh.uvs * scale
        if mesh.material is not None:
            tangents = meshops.compute_tangents(Mesh(mesh.vertices, mesh.faces, mesh.normals, texcoords))
    material = materials.get(mesh.material if mesh.uvs is not None else None)
    return pyrender.Primitive(
        positions=mesh.vertices.astype(np.float32),
        normals=mesh.normals.astype(np.float32),
        tangents=tangents.astype(np.float32) if tangents is not None else None,
        texcoord_0=texcoords.astype(np.float32) if texcoords is not None else None,
        indices=mesh.faces.astype(np.uint32),
        material=material,
    )


def scene_bounds(root: SceneNode) -> np.ndarray:
    """Axis-aligned world bounds ``[[min], [max]]`` of all meshes under root."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for _, mesh in root.iter_meshes():
        if len(mesh.vertices):
            lo = np.minimum(lo, mesh.vertices.min(axis=0))
            hi = np.maximum(hi, mesh.vertices.max(axis=0))
    if not np.all(np.isfinite(lo)):
        return np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    return np.array([lo, hi])


def _ground_mesh(bounds: np.ndarray) -> pyrender.Mesh:
    center = bounds.mean(axis=0)
    extent = max(float(np.max(bounds[1] - bounds[0])), 1.0) * 10
    y = bounds[0][1] - 1e-3
    h = extent / 2
    verts = np.array([
        [center[0] - h, y, center[2] - h], [center[0] + h, y, center[2] - h],
        [center[0] + h, y, center[2] + h], [center[0] - h, y, center[2] + h],
    ], dtype=np.float32)
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.62, 0.62, 0.6, 1.0), metallicFactor=0.0, roughnessFactor=0.9
    )
    prim = pyrender.Primitive(
        positions=verts,
        normals=np.tile([0, 1, 0], (4, 1)).astype(np.float32),
        indices=np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32),
        material=mat,
    )
    return pyrender.Mesh([prim])


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Camera-to-world pose looking from ``eye`` at ``target`` (camera looks down -Z)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.999:
        up = np.array([0.0, 0.0, -1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _direction_pose(direction) -> np.ndarray:
    """Pose for a directional light shining along ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    return look_at(-d, np.zeros(3))


class SceneRenderer:
    """Renders one SceneNode from any number of viewpoints."""

    def __init__(self, root: SceneNode, options: RenderOptions | None = None) -> None:
        self.options = options or RenderOptions()
        self.bounds = scene_bounds(root)
        self.scene = pyrender.Scene(ambient_light=[0.25, 0.27, 0.3], bg_color=BACKGROUND)
        materials = _MaterialCache()
        for _, mesh in root.iter_meshes():
            if len(mesh.faces):
                self.scene.add(pyrender.Mesh([_primitive(mesh, materials)]))
        if self.options.ground:
            self.scene.add(_ground_mesh(self.bounds))

        # Warm sun casting shadows plus a cool, weaker sky fill from the opposite side.
        self.scene.add(
            pyrender.DirectionalLight(color=[1.0, 0.96, 0.9], intensity=3.5),
            pose=_direction_pose([-0.45, -1.0, -0.6]),
        )
        self._fill = pyrender.DirectionalLight(color=[0.8, 0.87, 1.0], intensity=1.2)
        self._fill_node = self.scene.add(self._fill, pose=np.eye(4))
        self._camera = pyrender.PerspectiveCamera(
            yfov=np.radians(self.options.fov), aspectRatio=self.options.width / self.options.height
        )
        self._camera_node = self.scene.add(self._camera, pose=np.eye(4))

    def camera_pose(self, view: str = "iso") -> np.ndarray:
        opts = self.options
        center = self.bounds.mean(axis=0)
        target = opts.target if opts.target is not None else center
        if opts.camera is not None:
            return look_at(np.asarray(opts.camera, dtype=np.float64), target)
        azimuth, elevation = VIEWS[view]
        az, el = np.radians(azimuth), np.radians(elevation)
        direction = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
        radius = float(np.linalg.norm(self.bounds[1] - self.bounds[0])) / 2
        half_fov = np.radians(opts.fov) / 2
        aspect = opts.width / opts.height
        fit = min(np.tan(half_fov), np.tan(half_fov) * aspect)
        distance = radius / np.sin(np.arctan(fit)) * 1.02 / opts.zoom
        up = (0.0, 0.0, -1.0) if elevation > 80 else (0.0, 1.0, 0.0)
        return look_at(target + direction * max(distance, 0.1), target, up)

    def render(self, view: str = "iso") -> Image.Image:
        opts = self.options
        pose = self.camera_pose(view)
        self.scene.set_pose(self._camera_node, pose)
        # Fill light comes from behind/above the camera so shadowed sides aren't black,
        # tilted off the view axis so flat surfaces facing the camera (a tabletop
        # in top view) don't mirror it straight back as a hotspot.
        cam_dir = -pose[:3, 2]
        cam_right = pose[:3, 0]
        self.scene.set_pose(self._fill_node, _direction_pose(cam_dir + cam_right * 0.8 + np.array([0.0, -0.5, 0.0])))
        flags = pyrender.RenderFlags.NONE
        if opts.shadows:
            flags |= pyrender.RenderFlags.SHADOWS_DIRECTIONAL
        renderer = _get_renderer(opts.width, opts.height)
        color, _ = renderer.render(self.scene, flags=flags)
        return Image.fromarray(color[..., :3])


CUTAWAY_TAGS = ("ceiling", "light.ceiling")
CUTAWAY_NAMES = ("ceiling", "roof", "gables", "chimney", "chimney_cap", "cornice")


def cutaway(root: SceneNode) -> SceneNode:
    """Copy of ``root`` without ceilings, roofs and ceiling fixtures (to look inside)."""
    copy = root.copy(deep=True)

    def prune(node: SceneNode) -> None:
        node.children = [c for c in node.children
                         if c.name not in CUTAWAY_NAMES and not any(t in CUTAWAY_TAGS for t in c.tags)]
        for child in node.children:
            child.parent = node
            prune(child)

    prune(copy)
    return copy


def render_scene(root: SceneNode, options: RenderOptions | None = None, view: str = "iso") -> Image.Image:
    return SceneRenderer(root, options).render(view)


def render_views(
    root: SceneNode,
    views: list[str] | None = None,
    options: RenderOptions | None = None,
    columns: int = 2,
) -> Image.Image:
    """Render several views and tile them into one labelled contact sheet."""
    from PIL import ImageDraw

    views = views or ["iso", "front", "side", "top"]
    renderer = SceneRenderer(root, options)
    images = [renderer.render(v) for v in views]
    w, h = images[0].size
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (w * columns, h * rows), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    for i, (view, img) in enumerate(zip(views, images)):
        x, y = (i % columns) * w, (i // columns) * h
        sheet.paste(img, (x, y))
        draw.text((x + 8, y + 6), view, fill=(20, 20, 20))
    return sheet
