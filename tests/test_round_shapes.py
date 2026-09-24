"""Torus, capsule and bevelled cylinder primitives."""

import numpy as np
import pytest

from geogen.core import meshops, uvmap
from geogen.generators.round_shapes import BevelledCylinderGenerator, CapsuleGenerator, TorusGenerator
from geogen.layout import LayoutLoader


@pytest.mark.parametrize("gen, extent", [
    (TorusGenerator(size_x=1.0, size_y=0.2, size_z=1.0), (1.0, 0.2, 1.0)),
    (TorusGenerator(size_x=0.8, size_y=0.3, size_z=1.2, tube=0.1), (0.8, 0.2, 0.8)),
    (CapsuleGenerator(size_x=0.5, size_y=1.5, size_z=0.5), (0.5, 1.5, 0.5)),
    (CapsuleGenerator(size_x=0.6, size_y=0.4, size_z=0.6), (0.4, 0.4, 0.4)),   # degenerates to a sphere
    (BevelledCylinderGenerator(radius=0.3, height=0.5, bevel=0.05), (0.6, 0.5, 0.6)),
])
def test_closed_and_sized(gen, extent):
    mesh = gen.generate()
    assert meshops.validate(mesh).watertight
    assert np.ptp(mesh.vertices, axis=0) == pytest.approx(extent, abs=2e-3)
    assert uvmap.texel_density(mesh) == pytest.approx(1.0, rel=0.35)


def test_torus_has_a_hole():
    mesh = TorusGenerator(size_x=1.0, size_y=0.2, size_z=1.0).generate()
    r = np.linalg.norm(mesh.vertices[:, [0, 2]], axis=1)
    assert r.min() == pytest.approx(0.3, abs=2e-3)


def test_yaml_primitives():
    root = LayoutLoader().load_string("""
name: t
size: [1, 1, 1]
parts:
  ring: { primitive: torus, size: [1, 0.2, 1], anchor: bottom_center }
  pill: { primitive: capsule, size: [0.3, 0.8, 0.3], anchor: bottom_center, offset: [0, 0.2, 0] }
  puck: { primitive: cylinder, size: [0.5, 0.1, 0.5], bevel: 0.02, anchor: bottom_center }
""")
    for name in ("ring", "pill", "puck"):
        assert meshops.validate(root.find(name).mesh).watertight
    assert "top" in root.find("pill").attachments
