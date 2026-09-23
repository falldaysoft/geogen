"""Tests for the viewer's orbit camera and picking maths (no GL required)."""

import numpy as np
import pytest

from geogen.generators.primitives import CubeGenerator
from geogen.viewer.camera import OrbitCamera, ray_mesh_intersect


def test_view_matrix_puts_target_in_front():
    cam = OrbitCamera(target=np.array([1.0, 2.0, 3.0]), distance=10.0)
    p = cam.view_matrix() @ np.array([1.0, 2.0, 3.0, 1.0])
    np.testing.assert_allclose(p[:3], [0, 0, -10], atol=1e-9)


@pytest.mark.parametrize("preset", ["front", "right", "top", "iso"])
def test_frame_fits_bounds(preset):
    cam = OrbitCamera(aspect=1.5)
    cam.set_preset(preset)
    bounds = np.array([[-5.0, 0.0, -2.0], [5.0, 3.0, 2.0]])
    cam.frame(bounds)
    corners = np.array([[x, y, z, 1.0] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
    clip = (cam.projection_matrix() @ cam.view_matrix() @ corners.T).T
    ndc = clip[:, :3] / clip[:, 3:4]
    assert np.all(np.abs(ndc[:, :2]) <= 1.0)
    assert np.all(np.abs(ndc[:, 2]) < 1.0)  # inside near/far


def test_zoom_toward_point_keeps_point_fixed_on_axis():
    cam = OrbitCamera(target=np.zeros(3), distance=10.0, scene_radius=5.0)
    point = cam.target.copy()
    cam.zoom(2, toward=point)
    assert cam.distance < 10.0
    np.testing.assert_allclose(cam.target, point)


def test_pan_moves_target_in_view_plane():
    cam = OrbitCamera(distance=10.0)
    right, up, forward = cam.basis()
    cam.pan(100, 0, viewport_height=800)
    assert abs(np.dot(cam.target, forward)) < 1e-9
    assert np.dot(cam.target, right) < 0  # dragging right moves the scene right


def test_center_ray_hits_cube():
    cam = OrbitCamera(target=np.zeros(3), distance=5.0, yaw=0, pitch=0, aspect=1.0)
    origin, direction = cam.ray(400, 300, 800, 600)
    cube = CubeGenerator(bevel=0).generate()
    t = ray_mesh_intersect(origin, direction, cube.vertices, cube.faces)
    assert t == pytest.approx(4.5)
    assert ray_mesh_intersect(origin, -direction, cube.vertices, cube.faces) is None
