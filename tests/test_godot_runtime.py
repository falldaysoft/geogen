"""Walk the Godot runtime's first-person player into an exported cottage.

The cottage's front wall sits at z = 3.0 on a foundation reaching z = 3.05;
the door step (0.3 m, exactly the player's step height) runs z 2.895-3.345
and the closed door leaf's face is at z = 2.923. The player starts 3 m in
front and walks toward -Z.
"""

import json

import numpy as np

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


def test_all_exports_load_side_by_side(run_godot, tmp_path):
    # Without --scene every export loads; they must not overlap (the hotel
    # suite's walls used to sit inside the cottage) so the cottage stays
    # enterable. The player starts in front of the first model (cottage).
    from geogen.layout import LayoutLoader
    export_scene(_build_registry()["cottage"](), tmp_path / "cottage.glb")
    export_scene(LayoutLoader().load("assets/hotel_suite.yaml"), tmp_path / "hotel_suite.glb")
    out = run_godot(f"--generated={tmp_path}", "--use=door", "--wait=1.2", "--walk=1.5")
    assert "loaded cottage.glb" in out and "loaded hotel_suite.glb" in out
    end = json.loads(next(l for l in out.splitlines() if l.startswith("walk result: "))
                     .removeprefix("walk result: "))
    assert end["z"] < 2.0  # walked through the door into the cottage


def test_m1_furnished_hotel_room_walkthrough(run_godot, tmp_path):
    # Milestone M1: floor plan + archetype furnishing, exported with metadata.
    # From the entry spawn (facing west) the player walks through the
    # corridor and the bedroom door and is stopped by a nightstand.
    from geogen.layout import SceneComposer
    scene = SceneComposer().compose("assets/scenes/hotel_room_auto.yaml")
    nightstand = scene.find("nightstand_2").world_transform()[:3, 3]
    export_scene(scene, tmp_path / "hotel_room_auto.glb")
    out = run_godot("--scene", "hotel_room_auto", f"--generated={tmp_path}", "--walk=4")
    assert "3 rooms" in out
    end = json.loads(next(l for l in out.splitlines() if l.startswith("walk result: "))
                     .removeprefix("walk result: "))
    assert end["room"] == "bedroom"
    assert end["x"] == pytest.approx(nightstand[0] + 0.2 + RADIUS, abs=0.05)  # against its front


def _nav(run_godot, generated, scene, a, b) -> dict:
    out = run_godot("--scene", scene, f"--generated={generated}", f"--nav={a[0]},{a[1]}:{b[0]},{b[1]}")
    return json.loads(next(l for l in out.splitlines() if l.startswith("nav path: ")).removeprefix("nav path: "))


@pytest.fixture(scope="module")
def auto_room_dir(tmp_path_factory):
    from geogen.layout import SceneComposer
    out = tmp_path_factory.mktemp("generated_auto")
    export_scene(SceneComposer().compose("assets/scenes/hotel_room_auto.yaml"), out / "hotel_room_auto.glb")
    return out


def test_scene_builder_summary(run_godot, auto_room_dir):
    out = run_godot("--scene", "hotel_room_auto", f"--generated={auto_room_dir}", "--quit-after=2")
    line = next(l for l in out.splitlines() if "loaded hotel_room_auto.glb" in l)
    assert "3 rooms" in line and "1 spawns" in line
    assert "geogen: unknown" not in out and "extras.geogen version" not in out


def test_navigation_reaches_rooms_through_doorways(run_godot, auto_room_dir):
    # From the entry to open floor in the bedroom (foot of the bed) and into the bathroom.
    to_bedroom = _nav(run_godot, auto_room_dir, "hotel_room_auto", (4.2, 1.3), (-0.6, 0.6))
    assert to_bedroom["points"] and to_bedroom["end_gap"] < 0.5
    to_bathroom = _nav(run_godot, auto_room_dir, "hotel_room_auto", (4.2, 1.3), (2.6, -0.6))
    assert to_bathroom["points"] and to_bathroom["end_gap"] < 0.5
    # Paths go through the doorway gaps, never through walls: every corridor->bedroom
    # path crosses x = 1.0 (the shared wall) within the door's z span.
    pts = np.array(to_bedroom["points"])
    crossing = next(i for i in range(1, len(pts)) if pts[i - 1][0] >= 1.0 > pts[i][0])
    a, b = pts[crossing - 1], pts[crossing]
    z = a[2] + (b[2] - a[2]) * (a[0] - 1.0) / (a[0] - b[0])
    assert 0.85 < z < 1.75


def test_editor_import_plugin_builds_gameplay_nodes(run_godot, auto_room_dir, tmp_path):
    # Copy the runtime project, drop a geogen .glb in as a regular editor
    # asset and import it: the geogen addon must add room areas, spawn
    # markers and tag groups to the imported scene.
    import shutil
    import subprocess
    from conftest import GODOT_PROJECT, _godot_binary

    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(GODOT_PROJECT / "project.godot", project)
    for folder in ("addons", "scripts", "scenes"):
        shutil.copytree(GODOT_PROJECT / folder, project / folder)
    (project / "models").mkdir()
    shutil.copy(auto_room_dir / "hotel_room_auto.glb", project / "models")
    (project / "check.gd").write_text(
        "extends SceneTree\n"
        "func _initialize() -> void:\n"
        "\tvar root: Node = load(\"res://models/hotel_room_auto.glb\").instantiate()\n"
        "\tvar bed = root.find_child(\"bed\", true, false)\n"
        "\tprint(\"import check: %s\" % JSON.stringify({\n"
        "\t\t\"areas\": root.find_children(\"RoomArea\", \"\", true, false).size(),\n"
        "\t\t\"markers\": root.find_children(\"SpawnMarker\", \"\", true, false).size(),\n"
        "\t\t\"bed_groups\": bed.get_groups()}))\n"
        "\troot.free()\n"
        "\tquit()\n")
    godot = _godot_binary()
    # First pass registers the addon's classes; the second imports with the plugin active.
    for _ in range(2):
        (project / "models" / "hotel_room_auto.glb.import").unlink(missing_ok=True)
        subprocess.run([godot, "--headless", "--path", str(project), "--import"], capture_output=True, timeout=240)
    result = subprocess.run([godot, "--headless", "--path", str(project), "-s", "check.gd"],
                            capture_output=True, text=True, timeout=120)
    line = next(l for l in result.stdout.splitlines() if l.startswith("import check: "))
    check = json.loads(line.removeprefix("import check: "))
    assert check["areas"] == 3 and check["markers"] == 1
    assert {"furniture", "furniture.bed"} <= set(check["bed_groups"])


def test_player_climbs_stairs(run_godot, tmp_path):
    from geogen.layout import LayoutLoader
    export_scene(LayoutLoader().load("assets/staircase.yaml"), tmp_path / "staircase.glb")
    out = run_godot("--scene", "staircase", f"--generated={tmp_path}", "--spawn=0,0,3.2", "--walk=1.4")
    end = json.loads(next(l for l in out.splitlines() if l.startswith("walk result: "))
                     .removeprefix("walk result: "))
    assert end["y"] == pytest.approx(3.0, abs=0.02) and end["on_floor"]
