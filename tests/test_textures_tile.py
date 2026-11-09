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


@pytest.mark.parametrize("degrees", [0, 90, 30, -45, 117])
def test_directional_noise_tiles_and_runs_along_its_angle(degrees):
    """Streaks at any angle: no step across the wrap bigger than one inside the tile, and the
    noise changes several times faster across the direction than along it."""
    from geogen.textures.noise import directional_noise

    noise = directional_noise(128, 128, along=4, across=32, angle=np.radians(degrees), seed=1)
    for axis in (0, 1):
        steps = np.abs(noise - np.roll(noise, 1, axis=axis))
        inside = np.delete(steps, 0, axis=axis).max()
        assert np.take(steps, 0, axis=axis).max() <= 1.25 * inside
    gy, gx = np.gradient(noise)
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    assert np.abs(-gx * s + gy * c).mean() > 3 * np.abs(gx * c + gy * s).mean()
