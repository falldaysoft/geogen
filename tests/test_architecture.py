"""Tests for CSG, roofs, prisms and openings cut by placed windows/doors."""

import numpy as np
import pytest

from geogen.core import csg, meshops, uvmap
from geogen.generators.architecture import ROOF_STYLES, PrismGenerator, RoofGenerator
from geogen.generators.primitives import CubeGenerator, CylinderGenerator
from geogen.layout import LayoutLoader, SceneComposer


def _volume(mesh):
    v = mesh.vertices[mesh.faces]
    return np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6


def test_difference_removes_volume_and_stays_closed():
    wall = CubeGenerator(4, 3, 0.3, bevel=0).generate()
    hole = CubeGenerator(1, 1, 1, bevel=0).generate()
    result = csg.difference(wall, hole)
    report = meshops.validate(result)
    assert report.watertight and report.ok, report.issues
    assert _volume(result) == pytest.approx(4 * 3 * 0.3 - 1 * 1 * 0.3, rel=1e-4)
    assert uvmap.texel_density(result) == pytest.approx(1.0, rel=0.01)


def test_union_and_intersection():
    a = CubeGenerator(bevel=0).generate()
    b = CubeGenerator(bevel=0).generate().transform(np.array(
        [[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]))
    assert _volume(csg.union(a, b)) == pytest.approx(1.5, rel=1e-4)
    assert _volume(csg.intersection(a, b)) == pytest.approx(0.5, rel=1e-4)


def test_csg_rejects_open_mesh():
    cube = CubeGenerator(bevel=0).generate()
    from geogen.core.mesh import Mesh
    open_mesh = Mesh(cube.vertices, cube.faces[:-2])
    with pytest.raises(csg.CSGError):
        csg.difference(open_mesh, CylinderGenerator().generate())


@pytest.mark.parametrize("style", ROOF_STYLES)
@pytest.mark.parametrize("footprint", [(8.0, 6.0), (5.0, 9.0)])
def test_roofs_are_closed_and_cover_footprint(style, footprint):
    w, d = footprint
    mesh = RoofGenerator(width=w, height=2.0, depth=d, style=style, overhang=0.3).generate()
    report = meshops.validate(mesh)
    assert report.watertight and report.ok, report.issues
    extent = np.ptp(mesh.vertices, axis=0)
    assert extent[0] == pytest.approx(w + 0.6, abs=0.02)
    assert extent[2] == pytest.approx(d + 0.6, abs=0.02)
    assert uvmap.texel_density(mesh) == pytest.approx(1.0, rel=0.05)


def test_gable_roof_faces_slope_with_eaves_rows():
    mesh = RoofGenerator(width=8, height=2, depth=6, style="gable").generate()
    fn, area = meshops.face_normals(mesh.vertices, mesh.faces)
    sloped = (np.abs(fn[:, 1]) > 0.3) & (np.abs(fn[:, 1]) < 0.95) & (area > 1.0)
    uv = mesh.uvs[mesh.faces[sloped]]
    pos = mesh.vertices[mesh.faces[sloped]]
    # u runs along the ridge (X); v climbs the slope.
    du = uv[:, 1, 0] - uv[:, 0, 0]
    dx = pos[:, 1, 0] - pos[:, 0, 0]
    assert np.allclose(np.abs(du), np.abs(dx), atol=1e-6)


def test_prism_is_closed():
    mesh = PrismGenerator(8, 2, 6).generate()
    assert meshops.validate(mesh).watertight


def test_subtract_and_cutter_parts():
    root = LayoutLoader().load_string("""
name: box
size: [2, 2, 2]
parts:
  body:
    primitive: cube
    size: [1, 1, 1]
    anchor: bottom_center
    bevel: 0
    subtract: [hole]
  hole:
    primitive: cylinder
    size: [0.4, 1.2, 0.4]
    anchor: center
    cutter: true
""")
    names = [n.name for n in root.iter_nodes()]
    assert "hole" not in names
    body = root.find("body")
    assert meshops.validate(body.mesh).watertight
    assert _volume(body.mesh) < 8.0 - 0.9  # a ~1 m^3 cylinder was removed


def test_window_cuts_opening_into_host_wall():
    composer = SceneComposer()
    plain = composer.compose_string("""
name: s
place:
  house: { asset: house_peaked.yaml }
""")
    with_window = composer.compose_string("""
name: s
place:
  house: { asset: house_peaked.yaml }
  win:
    asset: window.yaml
    on: house.front_wall
    at: { u: 0.5, v: { abs: 1.0 } }
""")
    walls_before = plain.find("walls").mesh
    walls_after = with_window.find("walls").mesh
    assert meshops.validate(walls_after).watertight
    removed = _volume(walls_before) - _volume(walls_after)
    assert removed == pytest.approx(1.0 * 1.3 * 0.3, rel=0.02)  # window opening through a 0.3 m wall
    assert "opening" not in [n.name for n in with_window.iter_nodes()]

    # The interior lining behind the wall is cut by the same opening.
    lining_before = plain.find("lining").mesh
    lining_after = with_window.find("lining").mesh
    assert meshops.validate(lining_after).watertight
    removed = _volume(lining_before) - _volume(lining_after)
    assert removed == pytest.approx(1.0 * 1.3 * 0.015, rel=0.05)


def test_house_has_interior_finish():
    house = LayoutLoader().load("assets/house_peaked.yaml")
    materials = {n.name: n.mesh.material.name for n in house.iter_nodes() if n.mesh is not None and n.mesh.material}
    assert materials["floor"] == "hardwood_floor"
    assert materials["lining"] == "wall_plaster"
    assert materials["ceiling"] == "ceiling_white"
    assert "lining_void" not in materials


def test_openings_get_plastered_reveals():
    scene = SceneComposer().compose_string("""
name: s
place:
  house: { asset: house_peaked.yaml }
  win:
    asset: window.yaml
    on: house.front_wall
    at: { u: 0.5, v: { abs: 1.0 } }
""")
    reveal = scene.find("win_reveal")
    assert reveal is not None and reveal.mesh.material.name == "wall_plaster"
    lo, hi = reveal.mesh.vertices.min(axis=0), reveal.mesh.vertices.max(axis=0)
    # Sleeve spans the window opening (1.0 x 1.3 m) ...
    assert hi[0] - lo[0] == pytest.approx(1.0, abs=1e-6)
    assert hi[1] - lo[1] == pytest.approx(1.3, abs=1e-6)
    # ... from behind the frame to the lining's inner face, never outside.
    depth = 3.0  # house_peaked default depth 6 m, front wall face at z = +3
    assert hi[2] == pytest.approx(depth - 0.09 - 0.035, abs=1e-6)
    assert lo[2] == pytest.approx(depth - 0.3 - 0.015, abs=1e-6)
    assert meshops.validate(reveal.mesh).watertight
