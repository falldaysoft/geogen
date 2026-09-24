"""Sweep generator: profiles swept along 3D paths stay closed and keep their shape."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.core.profile import Shape, circle, rect
from geogen.generators.sweep import SweepGenerator
from geogen.layout import LayoutLoader

SQUARE = Shape(rect(0.1, 0.2, center=(0.05, 0.1)))  # 10 cm wide, 20 cm tall, to the right of the path


def _volume(mesh):
    v = mesh.vertices[mesh.faces]
    return np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6


@pytest.mark.parametrize("path, closed", [
    ([[0, 0, 0], [1, 0, 0]], False),
    ([[0, 0, 0], [1, 0, 0], [1, 0, 1]], False),
    ([[0, 0, 0], [2, 0, 0], [2, 0, 1.5], [0, 0, 1.5]], True),
    ([[np.cos(t), 0.2 * t, np.sin(t)] for t in np.linspace(0, 6, 40)], False),
])
def test_sweeps_are_closed_and_outward(path, closed):
    mesh = SweepGenerator(profile=SQUARE, path=np.array(path, dtype=float), closed=closed, center=False).generate()
    report = meshops.validate(mesh)
    assert report.watertight and report.inconsistent_winding_edges == 0
    assert _volume(mesh) > 0


def test_straight_sweep_volume_and_side():
    mesh = SweepGenerator(profile=SQUARE, path=np.array([[0.0, 0, 0], [1.0, 0, 0]]), center=False).generate()
    assert _volume(mesh) == pytest.approx(0.1 * 0.2 * 1.0)
    # Profile x is to the right of travel (+X heading -> +Z), y is up.
    assert mesh.vertices.min(axis=0) == pytest.approx([0, 0, 0])
    assert mesh.vertices.max(axis=0) == pytest.approx([1, 0.2, 0.1])


def test_mitred_room_perimeter_keeps_thickness():
    # Skirting around a 2 x 1.5 m room walked with the interior on the right.
    path = np.array([[0.0, 0, 0], [2, 0, 0], [2, 0, 1.5], [0, 0, 1.5]])
    mesh = SweepGenerator(profile=SQUARE, path=path, closed=True, center=False).generate()
    # A ring of 10 cm thick board: outer 2 x 1.5 minus inner 1.8 x 1.3, 20 cm tall.
    assert _volume(mesh) == pytest.approx((2 * 1.5 - 1.8 * 1.3) * 0.2, rel=1e-6)
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    assert lo == pytest.approx([0, 0, 0]) and hi == pytest.approx([2, 0.2, 1.5])


def test_rotation_minimising_frames_keep_level_paths_upright():
    # A wavy but level path: the profile's "up" must stay vertical all along.
    path = np.array([[x, 0.0, 0.3 * np.sin(x)] for x in np.linspace(0, 6, 60)])
    mesh = SweepGenerator(profile=SQUARE, path=path, center=False).generate()
    assert mesh.vertices[:, 1].min() == pytest.approx(0, abs=1e-9)
    assert mesh.vertices[:, 1].max() == pytest.approx(0.2, abs=1e-9)


def test_twist_and_scale():
    tube = Shape(circle(0.05, 16))
    path = np.array([[0.0, 0, 0], [0, 1, 0]])
    mesh = SweepGenerator(profile=tube, path=path, scale=(1.0, 0.5), twist=90, center=False).generate()
    assert meshops.validate(mesh).watertight
    top = mesh.vertices[mesh.vertices[:, 1] > 0.99]
    assert np.linalg.norm(top[:, [0, 2]], axis=1).max() == pytest.approx(0.025, abs=1e-3)


def test_yaml_sweep_part_with_spline_path():
    root = LayoutLoader().load("assets/handrail.yaml")
    rail = root.find("rail")
    assert meshops.validate(rail.mesh).watertight
    pts = (rail.world_transform() @ np.c_[rail.mesh.vertices, np.ones(len(rail.mesh.vertices))].T).T
    assert pts[:, 1].min() == pytest.approx(0, abs=0.03)       # feet on the floor
    assert pts[:, 1].max() == pytest.approx(0.9, abs=0.01)     # rail top at the height param
