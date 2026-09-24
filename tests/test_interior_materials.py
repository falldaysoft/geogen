"""Interior material pack: palette colours, PBR maps, and generator registration."""

import numpy as np
import pytest

from geogen.materials.loader import PAINT_PALETTE, MaterialLoader


def test_palette_colour_names_resolve():
    material = MaterialLoader().load("paint_sage")
    assert material.texture_generator.base_color == PAINT_PALETTE["sage"]


def test_unknown_palette_colour_raises():
    loader = MaterialLoader()
    with pytest.raises(ValueError, match="palette"):
        loader._parse_material({"texture": {"type": "painted_wall", "params": {"base_color": "puce"}}})


@pytest.mark.parametrize("name", ["fabric_linen", "carpet", "tile_white", "tile_subway", "marble"])
def test_interior_materials_have_pbr_detail(name):
    material = MaterialLoader().load(name)
    material.texture_size = (128, 128)
    images = material.gltf_images()
    assert "normal" in images
    assert np.asarray(images["base_color"]).std() > 1.0


def test_tile_grout_is_rougher_than_glaze():
    material = MaterialLoader().load("tile_white")
    material.texture_size = (256, 256)
    rough = np.asarray(material.get_roughness_map(), dtype=float)
    assert rough.max() - rough.min() > 100


@pytest.mark.parametrize("name", ["chrome", "mirror"])
def test_reflective_metals(name):
    material = MaterialLoader().load(name)
    assert material.metallic == 1.0
    assert material.roughness < 0.1
