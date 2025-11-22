"""Tests for all scenes - loads and renders each scene."""

import tempfile
from pathlib import Path

import numpy as np
import pyrender
import pytest
from PIL import Image

from geogen.scenes import create_chair_scene, create_table_scene
from geogen.viewer import Viewer

# All available scenes
SCENES = {
    "chair": create_chair_scene,
    "table": create_table_scene,
}

# Module-level renderer to avoid macOS pyglet context issues
_renderer = None
_renderer_size = (640, 480)


def get_renderer(width: int = 640, height: int = 480) -> pyrender.OffscreenRenderer:
    """Get or create a shared renderer."""
    global _renderer, _renderer_size
    if _renderer is None or _renderer_size != (width, height):
        if _renderer is not None:
            _renderer.delete()
        _renderer = pyrender.OffscreenRenderer(width, height)
        _renderer_size = (width, height)
    return _renderer


@pytest.fixture(scope="module", autouse=True)
def cleanup_renderer():
    """Clean up renderer after all tests in module."""
    yield
    global _renderer
    if _renderer is not None:
        _renderer.delete()
        _renderer = None


def render_scene(root, width: int = 640, height: int = 480) -> np.ndarray:
    """Render a scene to an image array."""
    viewer = Viewer(root, color=(0.7, 0.7, 0.8))

    pr_scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    for name, geom in viewer.scene.geometry.items():
        pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
        pr_scene.add(pr_mesh)

    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
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

    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = -forward
    camera_pose[:3, 3] = cam_pos
    pr_scene.add(camera, pose=camera_pose)

    # Add lighting
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    pr_scene.add(light, pose=camera_pose)

    # Render offscreen using shared renderer
    renderer = get_renderer(width, height)
    color, _ = renderer.render(pr_scene)

    return color


@pytest.mark.parametrize("scene_name,scene_factory", list(SCENES.items()))
def test_scene_loads(scene_name, scene_factory):
    """Test that each scene loads without errors."""
    root = scene_factory()
    assert root is not None
    assert root.name is not None

    # Verify scene has nodes
    nodes = list(root.iter_nodes())
    assert len(nodes) > 0, f"Scene '{scene_name}' should have at least one node"


@pytest.mark.parametrize("scene_name,scene_factory", list(SCENES.items()))
def test_scene_renders(scene_name, scene_factory):
    """Test that each scene renders without errors."""
    root = scene_factory()
    color = render_scene(root)

    # Verify we got a valid image
    assert color is not None
    assert color.shape == (480, 640, 3), f"Expected (480, 640, 3), got {color.shape}"
    assert color.dtype == np.uint8

    # Verify the image isn't completely black (rendering failed)
    assert color.max() > 0, f"Scene '{scene_name}' rendered as completely black"


@pytest.mark.parametrize("scene_name,scene_factory", list(SCENES.items()))
def test_scene_renders_to_file(scene_name, scene_factory):
    """Test that each scene can be saved to a file."""
    root = scene_factory()
    color = render_scene(root)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / f"{scene_name}.png"
        img = Image.fromarray(color)
        img.save(str(output_path))

        # Verify file was created and is non-empty
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify it's a valid image by re-loading it
        loaded = Image.open(output_path)
        assert loaded.size == (640, 480)
