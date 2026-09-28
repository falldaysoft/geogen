"""Tests for all scenes - loads and renders each scene."""

import numpy as np
import pyrender
import pytest

from geogen.registry import SceneRegistry
from geogen.render import RenderOptions, _get_renderer
from geogen.render import render_scene as geogen_render
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
    """Render a scene with the offline renderer (instanced: a city's hundreds of trees share
    one upload) to an image array."""
    return np.asarray(geogen_render(root, RenderOptions(width=width, height=height, shadows=False)))


@pytest.mark.parametrize("scene_name", [name for name, _ in SCENES])
def test_scene_renders(scene_name, built_scene):
    """Each scene renders to a non-black image.

    Loading and geometry checks live in test_asset_quality.py (sharing the build).
    """
    color = render_scene(built_scene(scene_name))
    assert color.shape == (480, 640, 3) and color.dtype == np.uint8
    assert color.max() > 0, f"Scene '{scene_name}' rendered as completely black"


@pytest.mark.parametrize("scene_name", ["chair", "dining_set", "cottage"])
def test_viewer_converts_to_trimesh(scene_name, built_scene):
    """The legacy trimesh viewer still converts scenes (small ones: it copies every instance)."""
    viewer = Viewer(built_scene(scene_name), color=(0.7, 0.7, 0.8))
    assert viewer.scene.geometry and all(len(g.faces) for g in viewer.scene.geometry.values())
    mesh = pyrender.Mesh.from_trimesh(next(iter(viewer.scene.geometry.values())), smooth=False)
    renderer = _get_renderer(64, 48)
    scene = pyrender.Scene()
    scene.add(mesh)
    scene.add(pyrender.PerspectiveCamera(yfov=0.8), pose=np.eye(4))
    renderer.render(scene)


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
