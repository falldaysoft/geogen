"""Tests for the Surface type and surface-based placement."""

import numpy as np
import pytest

from geogen.core.node import SceneNode
from geogen.generators.primitives import CubeGenerator, PlaneGenerator
from geogen.generators.room import RoomGenerator
from geogen.layout import LayoutLoader
from geogen.layout.surfaces import Surface


def _make_wall_surface() -> Surface:
    """A south wall: the interior face of the -Z wall of a 4x3x2 room.

    - origin at the left-bottom of the wall
    - u runs along +X (left → right when viewed from inside the room)
    - v runs up (+Y)
    - normal points +Z (outward into the room interior)
    """
    return Surface(
        name="south_wall",
        origin=np.array([-2.0, 0.0, -1.0]),  # left-bottom corner
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_extent=4.0,
        v_extent=3.0,
    )


class TestSurface:
    def test_fractional_center(self):
        s = _make_wall_surface()
        t = s.resolve(u=0.5, v=0.5)
        np.testing.assert_allclose(t.translation, [0.0, 1.5, -1.0])

    def test_fractional_left_bottom(self):
        s = _make_wall_surface()
        t = s.resolve(u=0.0, v=0.0)
        np.testing.assert_allclose(t.translation, [-2.0, 0.0, -1.0])

    def test_fractional_right_top(self):
        s = _make_wall_surface()
        t = s.resolve(u=1.0, v=1.0)
        np.testing.assert_allclose(t.translation, [2.0, 3.0, -1.0])

    def test_absolute_meters(self):
        s = _make_wall_surface()
        # u=0.5m absolute, v=1.0m absolute
        t = s.resolve(u={"abs": 0.5}, v={"abs": 1.0})
        np.testing.assert_allclose(t.translation, [-2.0 + 0.5, 1.0, -1.0])

    def test_depth_moves_along_normal(self):
        s = _make_wall_surface()
        t = s.resolve(u=0.5, v=0.5, depth=0.3)
        # normal is +Z, so depth 0.3 moves to z = -1.0 + 0.3 = -0.7
        np.testing.assert_allclose(t.translation, [0.0, 1.5, -0.7])

    def test_rotation_faces_normal(self):
        s = _make_wall_surface()
        t = s.resolve(u=0.5, v=0.5)
        # normal is +Z, so Y rotation should be 0 (object's +Z aligns)
        assert t.rotation[1] == pytest.approx(0.0)

    def test_rotation_for_north_wall(self):
        # A wall whose normal points -Z (like the interior face of a +Z wall)
        s = Surface(
            name="n",
            origin=np.array([-2.0, 0.0, 1.0]),
            u_axis=np.array([-1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 1.0, 0.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            u_extent=4.0,
            v_extent=3.0,
        )
        t = s.resolve(u=0.5, v=0.5)
        # atan2(0, -1) = pi
        assert abs(t.rotation[1]) == pytest.approx(np.pi)


class TestSceneNodeSurface:
    def test_get_surface_returns_world_transform(self):
        node = SceneNode(name="room")
        node.size = np.array([4.0, 3.0, 2.0])
        node.surfaces["south_wall"] = _make_wall_surface()

        # Move the node to (10, 0, 0) in world space.
        node.transform.translation = np.array([10.0, 0.0, 0.0])

        t = node.get_surface("south_wall", u=0.5, v=0.5)
        assert t is not None
        # Surface center is local (0, 1.5, -1), plus node translation (10, 0, 0).
        np.testing.assert_allclose(t.translation, [10.0, 1.5, -1.0])

    def test_get_surface_missing_returns_none(self):
        node = SceneNode(name="room")
        assert node.get_surface("ghost") is None

    def test_list_surfaces(self):
        node = SceneNode(name="room")
        node.surfaces["a"] = _make_wall_surface()
        node.surfaces["b"] = _make_wall_surface()
        assert set(node.list_surfaces()) == {"a", "b"}


class TestPrimitiveSurfaces:
    def test_cube_has_six_surfaces(self):
        cube = CubeGenerator(size_x=2, size_y=3, size_z=4)
        surfaces = cube.get_surfaces(np.array([2.0, 3.0, 4.0]))
        assert set(surfaces) == {"front", "back", "left", "right", "top", "bottom"}

    def test_cube_front_center(self):
        # Cubes are centered at origin on all axes.
        cube = CubeGenerator(size_x=2, size_y=3, size_z=4)
        surfaces = cube.get_surfaces(np.array([2.0, 3.0, 4.0]))
        front = surfaces["front"]
        # Front face at z=+2, center at (0, 0, 2)
        t = front.resolve(u=0.5, v=0.5)
        np.testing.assert_allclose(t.translation, [0.0, 0.0, 2.0])
        # Normal points +Z so Y rotation is 0
        assert t.rotation[1] == pytest.approx(0.0)

    def test_cube_back_viewed_from_outside(self):
        cube = CubeGenerator(size_x=2, size_y=3, size_z=4)
        surfaces = cube.get_surfaces(np.array([2.0, 3.0, 4.0]))
        back = surfaces["back"]
        # u=0 should be at +X (my left as I stand at -Z looking at +Z)
        # v=0 is at -Y (bottom corner since cube is centered)
        t = back.resolve(u=0.0, v=0.0)
        np.testing.assert_allclose(t.translation, [1.0, -1.5, -2.0])
        # Normal is -Z, rotation atan2(0, -1) = pi
        assert abs(t.rotation[1]) == pytest.approx(np.pi)

    def test_plane_has_top_surface(self):
        plane = PlaneGenerator(size_x=5, size_z=3)
        surfaces = plane.get_surfaces(np.array([5.0, 0.0, 3.0]))
        assert "top" in surfaces
        t = surfaces["top"].resolve(u=0.5, v=0.5)
        np.testing.assert_allclose(t.translation, [0.0, 0.0, 0.0])

    def test_room_interior_surfaces(self):
        room = RoomGenerator(size_x=4, size_y=3, size_z=6)
        surfaces = room.get_surfaces(np.array([4.0, 3.0, 6.0]))
        assert set(surfaces) == {
            "north_wall", "south_wall", "east_wall", "west_wall",
            "floor", "ceiling",
        }

    def test_room_north_wall_normal_points_inward(self):
        room = RoomGenerator(size_x=4, size_y=3, size_z=6)
        surfaces = room.get_surfaces(np.array([4.0, 3.0, 6.0]))
        # North wall is at z = +3, normal should point -Z (into room)
        north = surfaces["north_wall"]
        np.testing.assert_allclose(north.normal, [0.0, 0.0, -1.0])
        # Center of the wall: (0, 1.5, 3)
        t = north.resolve(u=0.5, v=0.5)
        np.testing.assert_allclose(t.translation, [0.0, 1.5, 3.0])


class TestSurfaceExports:
    def test_export_from_part_makes_surface_available_on_root(self):
        loader = LayoutLoader()
        yaml_str = """
name: box
size: [2, 3, 4]
parts:
  shell:
    primitive: cube
    size: [1, 1, 1]
    anchor: bottom_center
surfaces:
  front: { from: shell.front }
"""
        node = loader.load_string(yaml_str)
        assert "front" in node.surfaces

    def test_exported_surface_accounts_for_part_offset(self):
        # A part placed at bottom_center is translated up by its half-height.
        # The re-exported front surface origin must account for that.
        loader = LayoutLoader()
        yaml_str = """
name: box
size: [2, 3, 4]
parts:
  shell:
    primitive: cube
    size: [1, 1, 1]
    anchor: bottom_center
surfaces:
  front: { from: shell.front }
"""
        node = loader.load_string(yaml_str)
        # The part is a 2x3x4 cube placed with bottom at y=0 → center at y=1.5.
        # Its front surface is at z=+2 in part-local space.
        # Part origin is (0, 1.5, 0) in root space, so exported front
        # surface origin should be at the part's (-1, -1.5, 2) + (0, 1.5, 0) = (-1, 0, 2).
        t = node.get_surface("front", u=0.0, v=0.0)
        assert t is not None
        np.testing.assert_allclose(t.translation, [-1.0, 0.0, 2.0])

    def test_bad_export_source_raises(self):
        loader = LayoutLoader()
        yaml_str = """
name: box
size: [1, 1, 1]
parts:
  shell:
    primitive: cube
    size: [1, 1, 1]
    anchor: bottom_center
surfaces:
  ghost: { from: notapart.north_wall }
"""
        with pytest.raises(ValueError, match="unknown part 'notapart'"):
            loader.load_string(yaml_str)
