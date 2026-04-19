"""Tests for the parametric house_simple.yaml asset."""

from pathlib import Path

import numpy as np
import pytest

from geogen.layout import LayoutLoader


ASSETS_DIR = Path(__file__).parent.parent / "assets"


@pytest.fixture
def loader() -> LayoutLoader:
    return LayoutLoader()


def test_loads_at_default_params(loader: LayoutLoader):
    house = loader.load(ASSETS_DIR / "house_simple.yaml")
    assert house.size is not None
    np.testing.assert_allclose(house.size, [8, 4, 6])


def test_loads_at_overridden_params(loader: LayoutLoader):
    house = loader.load(
        ASSETS_DIR / "house_simple.yaml",
        params={"width": 12, "depth": 9, "height": 5},
    )
    assert house.size is not None
    np.testing.assert_allclose(house.size, [12, 5, 9])


def test_parts_present_at_overridden_params(loader: LayoutLoader):
    """All the expected parts should still exist at non-default sizes."""
    house = loader.load(
        ASSETS_DIR / "house_simple.yaml",
        params={"width": 12, "depth": 9},
    )
    names = {c.name for c in house.children}
    for expected in ["foundation", "walls", "door", "roof", "chimney",
                     "window_left", "window_right"]:
        assert expected in names, f"Expected part '{expected}' in children"


def test_surface_exports_available(loader: LayoutLoader):
    """The re-exported surfaces should resolve correctly at scene level."""
    house = loader.load(ASSETS_DIR / "house_simple.yaml")
    for surface_name in ["front_wall", "back_wall", "left_wall",
                         "right_wall", "roof"]:
        assert surface_name in house.surfaces, f"Missing surface {surface_name}"

    # Resolve a point on the front wall: u=0.5, v=0.5 should be inside the
    # bounding box of the house.
    t = house.get_surface("front_wall", u=0.5, v=0.5)
    assert t is not None
    # Front face is on +Z side; at default width=8 depth=6, 'walls' part is
    # 0.95 * container wide/deep → front wall z-coord is at roughly +2.85.
    assert t.translation[2] > 2.0
    assert t.translation[2] < 3.2


def test_unknown_param_override_errors(loader: LayoutLoader):
    from geogen.layout.expressions import ExpressionError
    with pytest.raises(ExpressionError, match="Unknown param override 'colour'"):
        loader.load(
            ASSETS_DIR / "house_simple.yaml",
            params={"colour": 1.0},
        )
