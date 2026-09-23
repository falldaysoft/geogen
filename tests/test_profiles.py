"""Tests for 2D profiles and the extrude / lathe generators."""

import numpy as np
import pytest

from geogen.core import meshops, profile as P, uvmap
from geogen.generators.profiles import ExtrudeGenerator, LatheGenerator
from geogen.layout import LayoutLoader


def test_shape_orientation_normalised():
    shape = P.Shape(P.rect(2, 1)[::-1], [P.circle(0.2)])
    assert P.signed_area(shape.outer) > 0
    assert P.signed_area(shape.holes[0]) < 0
    assert shape.area == pytest.approx(2 - np.pi * 0.04, rel=0.02)


def test_triangulate_with_hole_covers_area():
    shape = P.Shape(P.rect(2, 2), [P.rect(1, 1)])
    pts, tris = shape.triangulate()
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    areas = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    assert np.all(areas > 0)  # CCW
    assert areas.sum() == pytest.approx(3.0)


def test_fillet_rounds_corners_within_bounds():
    loop = P.fillet(P.rect(1, 1), 0.2, segments=8)
    assert len(loop) == 4 * 9
    assert np.all(np.abs(loop) <= 0.5 + 1e-9)
    # Corner is cut: no point near the sharp corner remains.
    assert np.min(np.linalg.norm(loop - [0.5, 0.5], axis=1)) > 0.05


def test_fillet_clamps_oversized_radius():
    loop = P.fillet(P.rect(1, 0.2), 5.0)
    assert np.all(np.isfinite(loop))
    assert np.all(np.abs(loop[:, 1]) <= 0.1 + 1e-9)


def test_catmull_rom_passes_through_points():
    pts = np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=float)
    curve = P.catmull_rom(pts, samples=6)
    for p in pts:
        assert np.min(np.linalg.norm(curve - p, axis=1)) < 1e-9


def test_offset_and_difference():
    grown = P.offset(P.rect(1, 1), 0.1)
    assert len(grown) == 1 and np.ptp(grown[0][:, 0]) == pytest.approx(1.2)
    assert P.offset(P.rect(1, 1), -0.6) == []
    wall = P.Shape(P.rect(4, 3))
    cut = wall.difference(P.Shape(P.rect(1, 1.5)))
    assert len(cut) == 1 and len(cut[0].holes) == 1


@pytest.mark.parametrize("spec", [
    [[0, 0], [1, 0], [1, 1]],
    {"rect": [1, 2], "radius": 0.1},
    {"circle": 0.5, "segments": 12},
    {"ellipse": [1, 0.5]},
    {"ngon": 1, "sides": 6},
    {"polygon": [[0, 0], [2, 0], [2, 1], [0, 1]], "fillet": 0.2},
    {"spline": [[0, 0], [1, 0], [1, 1], [0, 1]]},
])
def test_loop_specs(spec):
    loop = P.loop_from_spec(spec)
    assert loop.ndim == 2 and loop.shape[1] == 2 and len(loop) >= 3


EXTRUDES = {
    "sharp": ExtrudeGenerator(P.Shape(P.rect(1.2, 0.8)), depth=0.04),
    "beveled": ExtrudeGenerator(P.Shape(P.rect(1.2, 0.8, radius=0.1)), depth=0.05, bevel=0.01),
    "frame_z": ExtrudeGenerator(P.Shape(P.rect(1, 1.4), [P.rect(0.8, 1.2)]), depth=0.06, axis="z", bevel=0.005),
    "disc_x": ExtrudeGenerator(P.Shape(P.circle(0.3, 48)), depth=0.1, axis="x", bevel=0.02),
    "chamfer": ExtrudeGenerator(P.Shape(P.regular_polygon(0.5, 5)), depth=0.2,
                                bevel=0.03, bevel_segments=1),
}


@pytest.mark.parametrize("name", EXTRUDES)
def test_extrude_is_closed_and_metric(name):
    gen = EXTRUDES[name]
    mesh = gen.generate()
    report = meshops.validate(mesh)
    assert report.ok, report.issues
    assert report.watertight
    np.testing.assert_allclose(np.ptp(mesh.vertices, axis=0), gen._size(), atol=1e-6)
    assert uvmap.texel_density(mesh) == pytest.approx(1.0, rel=0.2)
    # Outward facing: signed volume is positive.
    v = mesh.vertices[mesh.faces]
    volume = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6
    assert volume > 0


def test_extrude_sharp_corners_stay_hard():
    mesh = ExtrudeGenerator(P.Shape(P.rect(1, 1)), depth=1.0).generate()
    # A sharp box needs split vertices: 6 faces x 4 corners.
    assert len(mesh.vertices) >= 24


@pytest.mark.parametrize("prof", [
    np.array([[0.2, 0.0], [0.2, 1.0]]),                                    # tube with caps
    np.array([[0.0, 0.0], [0.3, 0.0], [0.3, 0.5], [0.1, 0.8], [0.0, 1.0]]),  # closed at both poles
    P.catmull_rom(np.array([[0.1, 0], [0.2, 0.3], [0.1, 0.6], [0.15, 1.0]]), 6),
])
def test_lathe_is_closed(prof):
    mesh = LatheGenerator(profile=prof, segments=24).generate()
    report = meshops.validate(mesh)
    assert report.ok, report.issues
    assert report.watertight
    v = mesh.vertices[mesh.faces]
    assert np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() > 0


def test_lathe_partial_sweep_is_open():
    mesh = LatheGenerator(profile=np.array([[0.2, 0.0], [0.2, 1.0]]), sweep=180).generate()
    assert meshops.validate(mesh).boundary_edges > 0


def test_yaml_extrude_and_lathe_fill_part_size():
    yaml = """
name: test
size: [2, 1, 1]
parts:
  slab:
    primitive: extrude
    shape: {rect: [1, 1], radius: 0.1}
    size: [1.0, 0.1, 0.5]
    anchor: bottom_center
  post:
    primitive: lathe
    profile: {spline: [[1, 0], [0.5, 0.5], [1, 1]]}
    size: [0.1, 0.8, 0.1]
    anchor: bottom_center
  egg:
    primitive: ellipsoid
    size: [0.5, 0.2, 0.3]
    anchor: bottom_center
"""
    root = LayoutLoader().load_string(yaml)
    sizes = {n.name: np.ptp(n.mesh.vertices, axis=0) for n in root.iter_nodes() if n.mesh is not None}
    np.testing.assert_allclose(sizes["slab"], [2.0, 0.1, 0.5], atol=1e-6)
    np.testing.assert_allclose(sizes["post"], [0.1, 0.8, 0.1], rtol=0.02)  # round: min(x, z)
    np.testing.assert_allclose(sizes["egg"], [1.0, 0.2, 0.3], rtol=0.02)
