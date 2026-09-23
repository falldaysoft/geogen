"""Metric UV tests: texel density must not depend on object size."""

import numpy as np
import pytest

from geogen.core import uvmap
from geogen.generators.primitives import (
    ConeGenerator,
    CubeGenerator,
    CylinderGenerator,
    PlaneGenerator,
    SphereGenerator,
)
from geogen.generators.room import RoomGenerator


@pytest.mark.parametrize(
    "generator",
    [
        CubeGenerator(0.3, 2.0, 5.0, bevel=0),
        CubeGenerator(4.0, 1.0, 0.2, bevel=0.05),
        CylinderGenerator(radius=0.2, height=3.0, segments=64),
        ConeGenerator(radius=1.0, height=2.0, segments=64),
        PlaneGenerator(size_x=7.0, size_z=3.0),
    ],
    ids=lambda g: type(g).__name__,
)
def test_primitive_uvs_are_metric(generator):
    density = uvmap.texel_density(generator.generate())
    assert density == pytest.approx(1.0, rel=0.1)


def test_sphere_uvs_are_metric_at_equator():
    mesh = SphereGenerator(radius=2.0, segments=64, rings=32).generate()
    tri = mesh.vertices[mesh.faces]
    equator = np.all(np.abs(tri[..., 1]) < 0.25, axis=1)
    from geogen.core.mesh import Mesh
    band = Mesh(mesh.vertices, mesh.faces[equator], uvs=mesh.uvs)
    assert uvmap.texel_density(band) == pytest.approx(1.0, rel=0.05)


def test_room_uvs_are_metric():
    for mesh in RoomGenerator(size_x=6, size_y=3, size_z=4).generate_parts().values():
        assert uvmap.texel_density(mesh) == pytest.approx(1.0, rel=0.05)


def test_box_project_splits_shared_vertices():
    cube = CubeGenerator(bevel=0).generate()
    cube.uvs = None
    projected = uvmap.box_project(cube)
    assert projected.uvs is not None
    assert uvmap.texel_density(projected) == pytest.approx(1.0)


def test_cylindrical_project_handles_seam():
    mesh = CylinderGenerator(radius=0.5, height=1.0, segments=32).generate()
    projected = uvmap.cylindrical_project(mesh)
    tri = projected.vertices[projected.faces]
    sides = np.ptp(tri[..., 1], axis=1) > 0.5  # skip caps
    uv = projected.uvs[projected.faces[sides]]
    # No triangle should span more than one segment's worth of arc.
    span = uv[..., 0].max(axis=1) - uv[..., 0].min(axis=1)
    assert span.max() < 2 * np.pi * 0.5 / 32 * 1.5
