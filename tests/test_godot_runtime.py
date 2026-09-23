"""Walk the Godot runtime's first-person player into an exported cottage.

The cottage's front wall sits at z = 3.0 on a foundation reaching z = 3.05;
the door step (0.3 m, exactly the player's step height) runs z 2.895-3.345
and the closed door leaf's face is at z = 2.923. The player starts 3 m in
front and walks toward -Z.
"""

import json

import pytest

from geogen.export import export_scene
from geogen.main import _build_registry
from geogen.player import load_player_spec

RADIUS = load_player_spec().radius


@pytest.fixture(scope="module")
def cottage_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("generated")
    export_scene(_build_registry()["cottage"](), out / "cottage.glb")
    return out


def walk(run_godot, generated, spawn, seconds=4.0) -> dict:
    out = run_godot("--scene", "cottage", f"--generated={generated}",
                    f"--spawn={spawn}", f"--walk={seconds}")
    assert "geogen: loaded cottage.glb" in out
    line = next(l for l in out.splitlines() if l.startswith("walk result: "))
    return json.loads(line.removeprefix("walk result: "))


def test_open_ground_is_walkable(run_godot, cottage_dir):
    # Beside the house nothing blocks: 4 s at 4 m/s covers ~16 m.
    end = walk(run_godot, cottage_dir, "6,0,6.4")
    assert end["z"] < -8
    assert end["on_floor"]


def test_wall_blocks_player(run_godot, cottage_dir):
    end = walk(run_godot, cottage_dir, "-2,0,6.4")
    assert end["z"] == pytest.approx(3.05 + RADIUS, abs=0.05)  # stopped at the foundation
    assert end["y"] == pytest.approx(0, abs=0.02)


def test_player_climbs_door_step_and_stops_at_door(run_godot, cottage_dir):
    end = walk(run_godot, cottage_dir, "0,0,6.4")
    assert end["y"] == pytest.approx(0.3, abs=0.02)  # up on the step
    assert end["z"] == pytest.approx(2.923 + RADIUS, abs=0.05)  # against the closed leaf
    assert end["on_floor"]
