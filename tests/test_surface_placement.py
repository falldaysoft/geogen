"""Tests for surface-based placement in the scene composer."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from geogen.layout import SceneComposer


@pytest.fixture
def tmp_assets():
    """Create a temp assets dir with a couple of assets and wire up a composer."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "box.yaml").write_text(
            """
name: box
params:
  width: { default: 2 }
  depth: { default: 2 }
  height: { default: 2 }
size: ["{width}", "{height}", "{depth}"]
parts:
  shell:
    primitive: cube
    size: [1, 1, 1]
    anchor: bottom_center
surfaces:
  front: { from: shell.front }
  top: { from: shell.top }
"""
        )
        (d / "marker.yaml").write_text(
            """
name: marker
size: [0.1, 0.1, 0.1]
parts:
  body:
    primitive: cube
    size: [1, 1, 1]
    anchor: bottom_center
"""
        )
        yield d


def test_place_on_object_surface(tmp_assets: Path):
    scene_path = tmp_assets / "scene.yaml"
    scene_path.write_text(
        """
name: test_scene
size: [10, 10, 10]
place:
  box:
    asset: box.yaml
  marker:
    asset: marker.yaml
    on: box.front
    at: { u: 0.5, v: 0.5 }
"""
    )
    composer = SceneComposer(assets_dir=tmp_assets)
    root = composer.compose(scene_path)
    marker = next(n for n in root.children if n.name == "marker")
    # box is 2x2x2, placed with bottom_center at origin → part center at (0,1,0).
    # front face center → (0, 1, 1).
    np.testing.assert_allclose(marker.transform.translation, [0.0, 1.0, 1.0])


def test_place_on_object_surface_fractional(tmp_assets: Path):
    scene_path = tmp_assets / "scene.yaml"
    scene_path.write_text(
        """
name: test_scene
size: [10, 10, 10]
place:
  box:
    asset: box.yaml
  marker:
    asset: marker.yaml
    on: box.front
    at: { u: 0, v: 0 }
"""
    )
    composer = SceneComposer(assets_dir=tmp_assets)
    root = composer.compose(scene_path)
    marker = next(n for n in root.children if n.name == "marker")
    # u=0, v=0 corner of front face: left-bottom → (-1, 0, 1) in part local,
    # part centered at (0, 1, 0), so world corner is (-1, 0, 1).
    np.testing.assert_allclose(marker.transform.translation, [-1.0, 0.0, 1.0])


def test_place_on_object_surface_absolute_units(tmp_assets: Path):
    scene_path = tmp_assets / "scene.yaml"
    scene_path.write_text(
        """
name: test_scene
size: [10, 10, 10]
place:
  box:
    asset: box.yaml
  marker:
    asset: marker.yaml
    on: box.front
    at:
      u: { abs: 0.5 }
      v: { abs: 0.25 }
"""
    )
    composer = SceneComposer(assets_dir=tmp_assets)
    root = composer.compose(scene_path)
    marker = next(n for n in root.children if n.name == "marker")
    # Absolute: u=0.5m from left → x=-1+0.5=-0.5; v=0.25m up → y=0+0.25=0.25
    np.testing.assert_allclose(marker.transform.translation, [-0.5, 0.25, 1.0])


def test_unknown_object_in_on_raises(tmp_assets: Path):
    scene_path = tmp_assets / "scene.yaml"
    scene_path.write_text(
        """
name: test_scene
size: [10, 10, 10]
place:
  marker:
    asset: marker.yaml
    on: nobody.front
    at: { u: 0.5, v: 0.5 }
"""
    )
    composer = SceneComposer(assets_dir=tmp_assets)
    with pytest.raises(ValueError, match="unknown object 'nobody'"):
        composer.compose(scene_path)


def test_unknown_surface_raises(tmp_assets: Path):
    scene_path = tmp_assets / "scene.yaml"
    scene_path.write_text(
        """
name: test_scene
size: [10, 10, 10]
place:
  box:
    asset: box.yaml
  marker:
    asset: marker.yaml
    on: box.ghost_wall
    at: { u: 0.5, v: 0.5 }
"""
    )
    composer = SceneComposer(assets_dir=tmp_assets)
    with pytest.raises(ValueError, match="'ghost_wall' not found on 'box'"):
        composer.compose(scene_path)
