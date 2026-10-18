"""The crop generator (generators/crops.py): closed, low-poly plants at every growth stage."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.generators.crops import KINDS, CropGenerator
from geogen.layout import LayoutLoader


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("growth", [0.0, 0.15, 0.5, 0.8, 1.0])
def test_every_stage_is_closed_and_within_budget(kind, growth):
    node = CropGenerator(width=1.1, height=0.6, depth=1.1, kind=kind, growth=growth, seed=3).to_node()
    report = meshops.validate(node.mesh)
    assert not report.issues and report.boundary_edges == 0
    assert len(node.mesh.faces) < 8000                    # a 1 m patch; fields instance many
    assert node.meta["collider"] == "none"
    lo, hi = node.mesh.vertices.min(axis=0), node.mesh.vertices.max(axis=0)
    assert lo[1] >= -0.3 - 0.11 and hi[1] <= 0.3 + 0.05   # stands on the box's bottom (a root below)


@pytest.mark.parametrize("kind", KINDS)
def test_plants_grow(kind):
    heights = [CropGenerator(kind=kind, growth=g, height=0.6).to_node().mesh.vertices[:, 1].max()
               for g in (0.1, 0.5, 1.0)]
    assert heights[0] < heights[1] <= heights[2] + 1e-9


def test_ripe_crops_change_colour():
    def materials(kind, growth):
        node = CropGenerator(kind=kind, growth=growth).to_node()
        return {m.name for m in node.mesh.materials or [node.mesh.material]}
    assert "wheat_straw" not in materials("wheat", 0.6) and "wheat_straw" in materials("wheat", 1.0)
    assert "pumpkin_skin" not in materials("pumpkin", 0.6) and "pumpkin_skin" in materials("pumpkin", 1.0)
    assert "carrot_root" in materials("carrot", 1.0)


def test_seed_varies_and_bad_params_raise():
    a = CropGenerator(kind="wheat", seed=1).to_node().mesh.vertices
    b = CropGenerator(kind="wheat", seed=2).to_node().mesh.vertices
    assert a.shape != b.shape or not np.allclose(a, b)
    with pytest.raises(ValueError, match="kind"):
        CropGenerator(kind="turnip")
    with pytest.raises(ValueError, match="growth"):
        CropGenerator(growth=1.5)


def test_crop_patch_asset_loads():
    patch = LayoutLoader().load("assets/crop_patch.yaml", params={"kind": "pumpkin", "growth": 0.9})
    assert patch.find("crop").mesh is not None
