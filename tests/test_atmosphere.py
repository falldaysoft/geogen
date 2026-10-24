"""A scene's ``atmosphere:`` (geogen.atmosphere, geogen.84.13): the lighting mood, a preset's values
overridden, resolved at build time and carried by the manifest for the runtime to apply."""

import pytest

from geogen import atmosphere
from geogen.export import gameplay_summary
from geogen.layout import SceneComposer


def test_presets_are_complete_and_clear_is_the_runtime_default():
    keys = set(atmosphere.KEYS) - {"preset", "time"}
    for name, preset in atmosphere.PRESETS.items():
        assert set(preset) == keys, name
    clear = atmosphere.PRESETS["clear"]
    assert clear["fog"] == 0.004 and clear["saturation"] == 1.0 and clear["sun"] == 1.0 and clear["glow"] == 0.0


def test_a_preset_is_overridden_by_what_the_block_says():
    out = atmosphere.resolve({"preset": "grim", "fog": 0.03, "sky": {"top": [0.1, 0.2, 0.3]}, "time": "16:00"})
    assert out["preset"] == "grim" and out["fog"] == 0.03 and out["time"] == "16:00"
    assert out["saturation"] == atmosphere.PRESETS["grim"]["saturation"]
    assert out["sky"]["top"] == [0.1, 0.2, 0.3] and out["sky"]["horizon"] == atmosphere.PRESETS["grim"]["sky"]["horizon"]
    assert atmosphere.PRESETS["grim"]["sky"]["top"] != [0.1, 0.2, 0.3]     # the preset itself is untouched
    assert atmosphere.resolve({})["preset"] == "clear"


@pytest.mark.parametrize("spec, says", [
    ({"preset": "noir"}, "preset is one of"),
    ({"mist": 1}, "atmosphere is {"),
    ({"time": "25:00"}, "HH:MM"),
    ({"fog": 5}, "0..0.2"),
    ({"saturation": True}, "a number"),
    ({"sun_color": [1, 2, 0]}, "a colour"),
    ({"sky": {"zenith": [0, 0, 0]}}, "sky: {"),
])
def test_the_block_is_checked(spec, says):
    with pytest.raises(atmosphere.AtmosphereError, match=says):
        atmosphere.resolve(spec)


def test_the_manifest_carries_the_resolved_atmosphere():
    composer = SceneComposer()
    root = composer.compose_string("name: moody\nsize: [4, 3, 4]\natmosphere: {preset: storm, moon: 0.8}\nplace: {}\n")
    summary = gameplay_summary(root)
    assert summary["atmosphere"]["preset"] == "storm" and summary["atmosphere"]["moon"] == 0.8
    assert summary["atmosphere"]["fog"] == atmosphere.PRESETS["storm"]["fog"]
    plain = composer.compose_string("name: plain\nsize: [4, 3, 4]\nplace: {}\n")
    assert "atmosphere" not in gameplay_summary(plain)
    with pytest.raises(atmosphere.AtmosphereError, match="preset is one of"):
        composer.compose_string("name: moody\nsize: [4, 3, 4]\natmosphere: {preset: noir}\nplace: {}\n")
