"""Per-placement params: scenes can resize the assets (and scenes) they place."""

import numpy as np
import pytest

from geogen.layout import SceneComposer

from test_architecture import _volume


def _cut_volume(yaml: str) -> float:
    scene = SceneComposer().compose_string(yaml)
    return _volume(scene.find("walls").mesh)


BASE = """
name: s
place:
  house: { asset: house_peaked.yaml }
"""


def test_placed_asset_params_resize_it_and_its_opening():
    plain = _cut_volume(BASE)
    default = _cut_volume(BASE + "  win: { asset: window.yaml, on: house.front_wall, at: {u: 0.5, v: {abs: 1.0}} }\n")
    wide = _cut_volume(BASE + "  win: { asset: window.yaml, params: {width: 1.6, height: 1.0},"
                              " on: house.front_wall, at: {u: 0.5, v: {abs: 1.0}} }\n")
    assert plain - default == pytest.approx(1.0 * 1.3 * 0.3, rel=0.02)
    assert plain - wide == pytest.approx(1.6 * 1.0 * 0.3, rel=0.02)


def test_house_params_from_scene():
    scene = SceneComposer().compose_string("""
name: s
place:
  house: { asset: house_peaked.yaml, params: { width: 11, depth: 7 } }
""")
    walls = scene.find("walls").mesh.vertices
    assert np.ptp(walls[:, 0]) == pytest.approx(11) and np.ptp(walls[:, 2]) == pytest.approx(7)


def test_unknown_param_raises():
    with pytest.raises(Exception, match="nknown"):
        SceneComposer().compose_string(BASE.replace("house_peaked.yaml }", "house_peaked.yaml, params: {wdth: 3} }"))


def test_scene_params_and_nested_scene_overrides(tmp_path):
    (tmp_path / "plot.yaml").write_text("""
name: plot
params:
  house_width: { default: 8 }
place:
  house: { asset: house_peaked.yaml, params: { width: "{house_width}" } }
""")
    composer = SceneComposer()
    default = composer.compose(tmp_path / "plot.yaml")
    assert np.ptp(default.find("walls").mesh.vertices[:, 0]) == pytest.approx(8)
    wide = composer.compose(tmp_path / "plot.yaml", params={"house_width": 12})
    assert np.ptp(wide.find("walls").mesh.vertices[:, 0]) == pytest.approx(12)
    # And from a parent scene via `scene:` + `params:`.
    (tmp_path / "street.yaml").write_text(f"""
name: street
place:
  plot: {{ scene: {tmp_path / 'plot.yaml'}, params: {{ house_width: 10 }} }}
""")
    street = composer.compose(tmp_path / "street.yaml")
    assert np.ptp(street.find("walls").mesh.vertices[:, 0]) == pytest.approx(10)
