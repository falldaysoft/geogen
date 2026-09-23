"""Tests for the shared player spec (assets/player.yaml)."""

import pytest

from geogen.main import _build_registry
from geogen.player import PlayerSpec, load_player_spec


def test_default_spec_loads_from_assets():
    spec = load_player_spec()
    assert spec.radius == 0.3
    assert spec.height == 1.8
    assert spec.eye_height == 1.65
    assert spec.step_height == 0.3
    assert spec.max_slope_deg == 40
    assert (spec.door_min_width, spec.door_min_height) == (0.85, 2.0)
    assert spec.corridor_min_width == 1.2
    assert spec.reach == 1.5


def test_yaml_matches_dataclass_defaults():
    # The dataclass defaults are a fallback; keep them in sync with the YAML.
    assert load_player_spec() == PlayerSpec()


def test_round_trip_through_dict():
    spec = load_player_spec()
    assert PlayerSpec.from_dict(spec.to_dict()) == spec


def test_custom_spec_file(tmp_path):
    path = tmp_path / "small.yaml"
    path.write_text("kind: player_spec\nversion: 1\nradius: 0.2\nheight: 1.2\n"
                    "eye_height: 1.1\ndoor_min_height: 1.5\n")
    spec = load_player_spec(path)
    assert (spec.radius, spec.height, spec.eye_height) == (0.2, 1.2, 1.1)
    assert spec.step_height == 0.3  # unspecified keys keep defaults


@pytest.mark.parametrize("data, match", [
    ({"radius": 0.3, "wingspan": 2}, "unknown keys"),
    ({"kind": "chair"}, "expected kind"),
    ({"version": 2}, "unsupported version"),
    ({"radius": -0.1}, "positive"),
    ({"eye_height": 1.9}, "eye_height"),
    ({"door_min_height": 1.7}, "door_min"),
    ({"corridor_min_width": 0.5}, "corridor"),
    ({"max_slope_deg": 95}, "max_slope"),
])
def test_invalid_specs_rejected(data, match):
    with pytest.raises(ValueError, match=match):
        PlayerSpec.from_dict(data)


def test_player_yaml_is_not_a_scene():
    assert "player" not in _build_registry()



def test_godot_reads_manifest(tmp_path, run_godot):
    """The runtime's PlayerSpec parses exactly what the exporter wrote."""
    import json

    from geogen.export import export_scene, manifest_path

    # Non-default values so a silent fallback to defaults would fail.
    spec = PlayerSpec(radius=0.25, height=1.7, eye_height=1.55, step_height=0.25,
                      max_slope_deg=35, door_min_width=0.8, door_min_height=1.9,
                      corridor_min_width=1.0, reach=1.2)
    path = export_scene(_build_registry()["chair"](), tmp_path / "chair.glb", player=spec)
    out = run_godot("--quit-after=2", f"--generated={tmp_path}", f"--manifest={manifest_path(path)}")
    line = next(l for l in out.splitlines() if l.startswith("player spec: "))
    assert PlayerSpec.from_dict(json.loads(line.removeprefix("player spec: "))) == spec
