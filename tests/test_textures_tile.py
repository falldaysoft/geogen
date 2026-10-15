"""Every material texture must tile seamlessly, since metric UVs repeat it."""

from pathlib import Path

import numpy as np
import pytest

from geogen.check import seam_ratio as _seam_ratio
from geogen.materials.loader import MaterialLoader

MATERIALS = sorted(p.stem for p in (Path(__file__).parent.parent / "assets" / "materials").glob("*.yaml"))


@pytest.mark.parametrize("name", MATERIALS)
def test_albedo_tiles(name):
    material = MaterialLoader().load(name)
    material.texture_size = (256, 256)
    img = np.asarray(material.get_texture())
    for axis in (0, 1):
        assert _seam_ratio(img, axis) < 1.1, f"{name} seam along axis {axis}"
