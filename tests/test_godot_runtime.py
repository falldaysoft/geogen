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


def walk(run_godot, generated, spawn, seconds=4.0, *extra) -> dict:
    out = run_godot("--scene", "cottage", f"--generated={generated}",
                    f"--spawn={spawn}", f"--walk={seconds}", *extra)
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


def test_opened_door_lets_player_into_house(run_godot, cottage_dir):
    # --use=door runs the door's swing interaction; walk once it has opened.
    end = walk(run_godot, cottage_dir, "0,0,6.4", 1.5, "--use=door", "--wait=1.2")
    assert end["z"] < 2.0  # through the doorway, inside the house
    assert end["y"] == pytest.approx(0.32, abs=0.02)  # on the floorboards
    assert end["on_floor"]


def test_aim_and_use_opens_door(run_godot, cottage_dir):
    # Standing within reach and looking at the leaf, "E" opens the door.
    out = run_godot("--scene", "cottage", f"--generated={cottage_dir}", "--spawn=0,0,3.9",
                    "--use=@aim", "--wait=1.3", "--walk=1.0")
    assert "used: door (swing)" in out
    assert '"state":"open"' in out
    end = json.loads(next(l for l in out.splitlines() if l.startswith("walk result: "))
                     .removeprefix("walk result: "))
    assert end["z"] < 2.0


@pytest.fixture(scope="module")
def suite_dir(tmp_path_factory):
    from geogen.layout import SceneComposer
    out = tmp_path_factory.mktemp("generated_suite")
    root = SceneComposer().compose_string("""
name: suite_level
place:
  suite: { asset: hotel_suite.yaml }
spawns:
  hallway: { position: [2.2, 0, 1.3], facing: west }
""")
    export_scene(root, out / "suite_level.glb")
    return out


def test_walk_through_floorplan_doors_into_room(run_godot, suite_dir):
    # From outside the entry door (east), through the corridor and the
    # bedroom door, to the bedroom's west wall (inner face x = -3.25).
    out = run_godot("--scene", "suite_level", f"--generated={suite_dir}",
                    "--spawn=4,0,1.3", "--yaw=90", "--walk=3")
    assert "3 rooms" in out
    end = json.loads(next(l for l in out.splitlines() if l.startswith("walk result: "))
                     .removeprefix("walk result: "))
    assert end["x"] == pytest.approx(-3.25 + RADIUS, abs=0.05)
    assert end["room"] == "bedroom"
    assert end["on_floor"]


def test_player_starts_at_manifest_spawn(run_godot, suite_dir):
    # The spawn faces west, so walking forward leads into the bedroom.
    out = run_godot("--scene", "suite_level", f"--generated={suite_dir}", "--walk=0.2")
    end = json.loads(next(l for l in out.splitlines() if l.startswith("walk result: "))
                     .removeprefix("walk result: "))
    assert end["x"] < 2.2 and end["z"] == pytest.approx(1.3, abs=0.05)
    assert end["room"] == "corridor"
