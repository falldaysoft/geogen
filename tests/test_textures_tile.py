"""Every material texture must tile seamlessly, since metric UVs repeat it."""

from pathlib import Path

import numpy as np
import pytest

from geogen.materials.loader import MaterialLoader

MATERIALS = sorted(p.stem for p in (Path(__file__).parent.parent / "assets" / "materials").glob("*.yaml"))


def _seam_ratio(img: np.ndarray, axis: int) -> float:
    """Wrap-around edge difference relative to the largest interior row/column step.

    Patterned textures (plank gaps, mortar) legitimately have sharp steps,
    so the wrap edge only counts as a seam if it is sharper than any step
    inside the texture.
    """
    a = img.astype(np.float64)
    other = tuple(i for i in range(a.ndim) if i != axis)
    steps = np.abs(np.diff(a, axis=axis)).mean(axis=other)
    first = np.take(a, 0, axis=axis)
    last = np.take(a, -1, axis=axis)
    return float(np.abs(first - last).mean() / max(steps.max(), 1e-6))


@pytest.mark.parametrize("name", MATERIALS)
def test_albedo_tiles(name):
    material = MaterialLoader().load(name)
    material.texture_size = (256, 256)
    img = np.asarray(material.get_texture())
    for axis in (0, 1):
        assert _seam_ratio(img, axis) < 1.1, f"{name} seam along axis {axis}"
