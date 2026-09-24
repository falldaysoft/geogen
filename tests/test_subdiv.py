"""Loop subdivision, creases and noise displacement (core/subdiv.py) and their YAML hooks."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.core.subdiv import displace, fractal_noise3, subdivide
from geogen.generators.primitives import CubeGenerator
from geogen.layout.loader import LayoutLoader


def _cube():
    return CubeGenerator(size_x=1.0, size_y=1.0, size_z=1.0, bevel=0).generate()


def test_each_level_quadruples_triangles_and_stays_closed():
    mesh = _cube()
    for levels in (1, 2, 3):
        out = subdivide(mesh, levels)
        assert len(out.faces) == 12 * 4 ** levels
        report = meshops.validate(out)
        assert report.boundary_edges == 0 and report.non_manifold_edges == 0
        assert report.inconsistent_winding_edges == 0


def test_smooth_subdivision_rounds_the_cage():
    out = subdivide(_cube(), 3)
    extent = out.vertices.max(axis=0) - out.vertices.min(axis=0)
    assert np.all(extent < 0.95)            # corners pulled in
    radius = np.linalg.norm(out.vertices, axis=1)
    assert radius.max() / radius.min() < 1.4   # much rounder than a cube (sqrt 3)


def test_creases_keep_sharp_edges():
    out = subdivide(_cube(), 3, crease_angle=60)
    np.testing.assert_allclose(out.vertices.min(axis=0), -0.5, atol=1e-9)
    np.testing.assert_allclose(out.vertices.max(axis=0), 0.5, atol=1e-9)
    # Every vertex still lies on the cube's surface.
    assert np.allclose(np.abs(out.vertices).max(axis=1), 0.5)


def test_open_boundaries_act_as_creases():
    # A single quad: its border must stay on the original square outline.
    quad = _cube()
    top = quad.faces[np.all(quad.vertices[quad.faces][:, :, 1] > 0.49, axis=1)]
    from geogen.core.mesh import Mesh

    out = subdivide(Mesh(vertices=quad.vertices, faces=top), 2)
    assert len(out.vertices) == 25                     # unused cage vertices dropped
    report = meshops.validate(out)
    assert report.boundary_edges == 16
    # The square outline is kept exactly: corners stay, edge points stay on the edges.
    border = np.abs(out.vertices[:, [0, 2]]).max(axis=1) > 0.499
    assert border.sum() == 16
    np.testing.assert_allclose(out.vertices[:, 1], 0.5)


def test_displacement_is_deterministic_bounded_and_closed():
    base = subdivide(_cube(), 3)
    a = displace(base, 0.05, 0.3, 3, seed=4)
    b = displace(base, 0.05, 0.3, 3, seed=4)
    c = displace(base, 0.05, 0.3, 3, seed=5)
    np.testing.assert_array_equal(a.vertices, b.vertices)
    assert not np.allclose(a.vertices, c.vertices)
    report = meshops.validate(a)
    assert report.boundary_edges == 0 and report.degenerate_faces == 0
    # Moved, but by no more than ~the amplitude.
    welded = subdivide(base, 0)
    delta = np.linalg.norm(a.vertices - welded.vertices, axis=1)
    assert delta.max() <= 0.05 * 1.05 and delta.mean() > 0.005


def test_fractal_noise_range():
    pts = np.random.default_rng(0).uniform(-20, 20, (5000, 3))
    n = fractal_noise3(pts, octaves=4, seed=1)
    assert -1.05 <= n.min() < -0.3 and 0.3 < n.max() <= 1.05


def _write(tmp_path, body: str):
    path = tmp_path / "blob.yaml"
    path.write_text(body)
    return path


BLOB = """
name: blob
origin: bottom_center
size: [1, 0.6, 0.8]
parts:
  body:
    primitive: cube
    bevel: 0
    size: [1, 1, 1]
    anchor: bottom_center
    subdivide: {subdivide}
    displace: {{ amplitude: 0.03, scale: 0.2, seed: 2 }}
"""


def _body(node):
    return next(n for n in node.iter_nodes() if n.name == "body").mesh


def test_yaml_subdivide_fills_part_box_and_gets_metric_uvs(tmp_path):
    path = _write(tmp_path, BLOB.format(subdivide=3))
    mesh = _body(LayoutLoader().load(path))
    assert meshops.validate(mesh).ok
    extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    np.testing.assert_allclose(extent, [1, 0.6, 0.8], atol=0.07)   # box + displacement
    assert mesh.uvs is not None and mesh.normals is not None
    from geogen.core.uvmap import texel_density

    assert 0.75 < texel_density(mesh) <= 1.01   # box projection on curved faces: cos-weighted


def test_detail_changes_subdivision_levels(tmp_path):
    path = _write(tmp_path, BLOB.format(subdivide="{ levels: 3, crease: 60 }"))
    full = len(_body(LayoutLoader().load(path)).faces)
    low = len(_body(LayoutLoader(detail=0.5).load(path)).faces)
    assert full == 12 * 64 and low == 12 * 16


def test_unknown_subdivide_keys_raise(tmp_path):
    path = _write(tmp_path, BLOB.format(subdivide="{ level: 2 }"))
    with pytest.raises(ValueError, match="subdivide"):
        LayoutLoader().load(path)
