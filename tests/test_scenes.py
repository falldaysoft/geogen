"""Tests for all scenes - loads and renders each scene."""

import numpy as np
import pyrender
import pytest

from geogen.registry import SceneRegistry
from geogen.render import _get_renderer
from geogen.scenes.nature import create_nature_scene
from geogen.viewer import Viewer


def _build_registry() -> SceneRegistry:
    """Build the full scene registry for testing."""
    registry = SceneRegistry()
    registry.discover()
    registry.register("nature", create_nature_scene)
    return registry


_registry = _build_registry()
SCENES = list(_registry.scenes.items())

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

    # One offscreen context per process: recreating it breaks pyglet on macOS.
    renderer = _get_renderer(width, height)
    color, _ = renderer.render(pr_scene)

    return color


@pytest.mark.parametrize("scene_name", [name for name, _ in SCENES])
def test_scene_renders(scene_name, built_scene):
    """Each scene converts to trimesh and renders to a non-black image.

    Loading and geometry checks live in test_asset_quality.py (sharing the build).
    """
    color = render_scene(built_scene(scene_name))
    assert color.shape == (480, 640, 3) and color.dtype == np.uint8
    assert color.max() > 0, f"Scene '{scene_name}' rendered as completely black"


def test_registry_discovers_all_yaml():
    """Test that the registry discovers all YAML files."""
    registry = SceneRegistry()
    registry.discover()

    # Should find all individual assets
    for asset in ["chair", "table", "bench", "fire_hydrant", "mailbox",
                  "street_lamp", "pine_tree", "maple_tree", "trashcan",
                  "house_simple", "house_peaked", "road", "sidewalk", "ground",
                  "rock_small", "rock_medium", "rock_large", "bush"]:
        assert asset in registry, f"Expected asset '{asset}' to be discovered"

    # Should find all composed scenes
    for scene in ["dining_set", "street", "street_side", "house_plot", "trees",
                  "room_with_furniture"]:
        assert scene in registry, f"Expected scene '{scene}' to be discovered"


def test_registry_register_python_scene():
    """Test that Python-coded scenes can be registered."""
    registry = SceneRegistry()
    registry.register("nature", create_nature_scene)
    assert "nature" in registry
    root = registry["nature"]()
    assert root is not None
