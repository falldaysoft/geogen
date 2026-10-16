"""Every material texture must tile seamlessly, since metric UVs repeat it."""

from pathlib import Path

import numpy as np
import pytest

from geogen.check import seam_ratio as _seam_ratio
from geogen.materials.loader import MaterialLoader

MATERIALS = sorted(p.stem for p in (Path(__file__).parent.parent / "assets" / "materials").glob("*.yaml"))


def _maps(name):
    material = MaterialLoader().load(name)
    material.texture_size = (256, 256)
    return {"albedo": material.get_texture(), "normal": material.get_normal_map(),
            "roughness": material.get_roughness_map()}


@pytest.mark.parametrize("name", MATERIALS)
def test_maps_tile(name):
    for kind, image in _maps(name).items():
        if image is None:
            continue
        img = np.asarray(image)
        for axis in (0, 1):
            assert _seam_ratio(img, axis) < 1.1, f"{name} {kind} seam along axis {axis}"
