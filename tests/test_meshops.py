"""Tests for mesh processing operations and primitive mesh quality."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.core.mesh import Mesh
from geogen.generators.primitives import (
    ConeGenerator,
    CubeGenerator,
    CylinderGenerator,
    PlaneGenerator,
    SphereGenerator,
)


@pytest.mark.parametrize("bevel", [0.0, 0.01, 0.1, 0.5])
@pytest.mark.parametrize("segments", [1, 2, 4])
def test_rounded_box_is_closed_and_consistently_wound(bevel, segments):
    mesh = CubeGenerator(1.0, 2.0, 0.5, bevel=bevel, bevel_segments=segments).generate()
    report = meshops.validate(mesh)
    assert report.ok, report.issues
    assert report.watertight
    np.testing.assert_allclose(mesh.vertices.min(axis=0), [-0.5, -1.0, -0.25], atol=1e-9)
    np.testing.assert_allclose(mesh.vertices.max(axis=0), [0.5, 1.0, 0.25], atol=1e-9)


def test_rounded_box_faces_point_outward():
    mesh = CubeGenerator(bevel=0.2).generate()
    fn, _ = meshops.face_normals(mesh.vertices, mesh.faces)
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", fn, centroids) > 0)
    # Analytic vertex normals agree with the faces they belong to.
    for k in range(3):
        assert np.all(np.einsum("ij,ij->i", mesh.normals[mesh.faces[:, k]], fn) > 0.5)


@pytest.mark.parametrize(
    "generator",
    [SphereGenerator(), CylinderGenerator(), ConeGenerator(), PlaneGenerator()],
    ids=lambda g: type(g).__name__,
)
def test_primitives_validate(generator):
    report = meshops.validate(generator.generate())
    assert report.ok, report.issues


def _two_faced_wedge(angle_deg: float) -> Mesh:
    """Two triangles sharing an edge along X, folded by angle_deg."""
    a = np.radians(angle_deg)
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 0, 1],
        [0.5, np.sin(a), -np.cos(a)],
    ], dtype=np.float64)
    faces = np.array([[0, 2, 1], [0, 1, 3]], dtype=np.int64)
    return Mesh(verts, faces)


def test_compute_normals_splits_hard_edges():
    hard = meshops.compute_normals(_two_faced_wedge(90), crease_angle=30)
    assert len(hard.vertices) == 6  # shared edge vertices split
    soft = meshops.compute_normals(_two_faced_wedge(10), crease_angle=30)
    assert len(soft.vertices) == 4
    np.testing.assert_allclose(np.linalg.norm(soft.normals, axis=1), 1.0)


def test_compute_normals_smooth_across_uv_seam():
    sphere = SphereGenerator().generate()
    smoothed = meshops.compute_normals(sphere, crease_angle=45)
    # On a sphere every normal should point along its position.
    dirs = smoothed.vertices / np.linalg.norm(smoothed.vertices, axis=1, keepdims=True)
    assert np.all(np.einsum("ij,ij->i", dirs, smoothed.normals) > 0.98)


def test_weld_merges_duplicates_but_keeps_seams():
    mesh = CubeGenerator(bevel=0).generate()  # 24 verts, 4 per face with distinct normals
    welded = meshops.weld_vertices(mesh)
    assert len(welded.vertices) == 24
    stripped = Mesh(mesh.vertices, mesh.faces)
    assert len(meshops.weld_vertices(stripped).vertices) == 8


def test_tangents_are_unit_and_orthogonal():
    mesh = CubeGenerator(bevel=0.1).generate()
    tangents = meshops.compute_tangents(mesh)
    assert tangents.shape == (len(mesh.vertices), 4)
    np.testing.assert_allclose(np.linalg.norm(tangents[:, :3], axis=1), 1.0, atol=1e-6)
    assert np.all(np.abs(np.einsum("ij,ij->i", tangents[:, :3], mesh.normals)) < 1e-6)
    assert set(np.unique(tangents[:, 3])) <= {-1.0, 1.0}


def test_validate_detects_flipped_face():
    mesh = CubeGenerator(bevel=0).generate()
    faces = mesh.faces.copy()
    faces[0] = faces[0][::-1]
    report = meshops.validate(Mesh(mesh.vertices, faces))
    assert report.inconsistent_winding_edges > 0
    assert not report.ok
