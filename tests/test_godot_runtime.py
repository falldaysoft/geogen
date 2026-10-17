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
from geogen.player import load_player_spec
from geogen.runtime_client import step

RADIUS = load_player_spec().radius


@pytest.fixture(scope="module")
def cottage_dir(scene_export):
    return scene_export("cottage")


def walk(godot_script, generated, spawn, seconds=4.0, before=()) -> dict:
    """Put the player at ``spawn`` facing -Z, run the ``before`` steps, walk forward; the end of the walk."""
    run = godot_script([step("player.teleport", pos=spawn, yaw=0), *before, step("player.move", seconds=seconds)],
                       "--scene=cottage", f"--generated={generated}")
    assert "geogen: loaded cottage.glb" in run.output
    return run["player.move"]


def test_open_ground_is_walkable(godot_script, cottage_dir):
    # Beside the house nothing blocks: 4 s at 4 m/s covers ~16 m.
    end = walk(godot_script, cottage_dir, [6, 0, 6.4])
    assert end["position"][2] < -8
    assert end["on_floor"]


def test_wall_blocks_player(godot_script, cottage_dir):
    end = walk(godot_script, cottage_dir, [-2, 0, 6.4])
    assert end["position"][2] == pytest.approx(3.05 + RADIUS, abs=0.05)  # stopped at the foundation
    assert end["position"][1] == pytest.approx(0, abs=0.02)
    assert end["blocked_by"] is not None          # the front wall


def test_player_climbs_door_step_and_stops_at_door(godot_script, cottage_dir):
    end = walk(godot_script, cottage_dir, [0, 0, 6.4])
    assert end["position"][1] == pytest.approx(0.3, abs=0.02)  # up on the step
    assert end["position"][2] == pytest.approx(2.923 + RADIUS, abs=0.05)  # against the closed leaf
    assert end["on_floor"]
    assert end["blocked_by"]["interaction"] == {"name": "swing", "state": "closed", "asset": "door"}


def test_opened_door_lets_player_into_house(godot_script, cottage_dir):
    # The door's swing interaction opens it; walk once it has.
    end = walk(godot_script, cottage_dir, [0, 0, 6.4], 1.5,
               [step("interactions.use", asset="door", actor="player"), step("wait", seconds=1.2)])
    assert end["position"][2] < 2.0  # through the doorway, inside the house
    assert end["position"][1] == pytest.approx(0.32, abs=0.02)  # on the floorboards
    assert end["on_floor"]


def test_aim_and_use_opens_door(godot_script, cottage_dir):
    # Standing within reach and looking at the leaf, "E" opens the door.
    run = godot_script([step("player.teleport", pos=[0, 0, 3.9], yaw=0), step("player.use", retry=2),
                        step("wait", seconds=1.3), step("interactions.list", asset="door"),
                        step("player.move", seconds=1.0)],
                       "--scene=cottage", f"--generated={cottage_dir}")
    assert run["player.use"] == {"kind": "interaction", "asset": "door", "interaction": "swing", "prompt": "Open"}
    assert run["interactions.list"][0]["state"] == "open"
    assert run["player.move"]["position"][2] < 2.0


def test_action_flags_are_script_steps(godot_script, cottage_dir):
    # --walk / --use / --status (CLAUDE.md's quick checks) expand to the same steps a script runs.
    run = godot_script((), "--scene=cottage", f"--generated={cottage_dir}", "--spawn=0,0,6.4", "--yaw=0",
                       "--use=door", "--wait=1.2", "--walk=1.5", "--status")
    assert [s["cmd"] for s in run.steps] == ["interactions.use", "wait", "player.move", "player.status"]
    assert run["player.move"]["position"][2] < 2.0 and run["player.status"]["room"] == run["player.move"]["room"]


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


def test_walk_through_floorplan_doors_into_room(godot_script, suite_dir):
    # From outside the entry door (east; the suite's door starts locked, so
    # restore a save with it open), through the corridor and the bedroom
    # door, to the bedroom's west wall (inner face x = -3.25).
    save = suite_dir / "entry_open.json"
    save.write_text(json.dumps({"entry/swing": {"state": "open", "locked": False}}))
    run = godot_script([step("player.teleport", pos=[4, 0, 1.3], yaw=90), step("player.move", seconds=3)],
                       "--scene=suite_level", f"--generated={suite_dir}", f"--load={save}")
    assert "3 rooms" in run.output
    end = run["player.move"]
    assert end["position"][0] == pytest.approx(-3.25 + RADIUS, abs=0.05)
    assert end["room"] == "bedroom"
    assert end["on_floor"]


def test_player_starts_at_manifest_spawn(godot_script, suite_dir):
    # The spawn faces west, so walking forward leads into the bedroom.
    run = godot_script([step("player.move", seconds=0.2)], "--scene=suite_level", f"--generated={suite_dir}")
    end = run["player.move"]
    assert end["position"][0] < 2.2 and end["position"][2] == pytest.approx(1.3, abs=0.05)
    assert end["room"] == "corridor"


def test_all_exports_load_side_by_side(godot_script, built_scene, tmp_path):
    # Without --scene every export loads; they must not overlap (the hotel
    # suite's walls used to sit inside the cottage) so the cottage stays
    # enterable. The player starts in front of the first model (cottage).
    export_scene(built_scene("cottage"), tmp_path / "cottage.glb")
    export_scene(built_scene("hotel_suite"), tmp_path / "hotel_suite.glb")
    run = godot_script([step("interactions.use", asset="door", actor="player"), step("wait", seconds=1.2),
                        step("player.move", seconds=1.5)], f"--generated={tmp_path}")
    assert "loaded cottage.glb" in run.output and "loaded hotel_suite.glb" in run.output
    assert run["player.move"]["position"][2] < 2.0  # walked through the door into the cottage


def test_m1_furnished_hotel_room_walkthrough(godot_script, built_scene, auto_room_dir, tmp_path):
    # Milestone M1: floor plan + archetype furnishing, exported with metadata.
    # From the entry spawn (facing west) the player walks through the
    # corridor and the bedroom door and is stopped by a nightstand.
    nightstand = built_scene("hotel_room_auto").find("nightstand_2").world_transform()[:3, 3]
    save = tmp_path / "entry_open.json"   # the suite door starts locked
    save.write_text(json.dumps({"entry/swing": {"state": "open", "locked": False}}))
    run = godot_script([step("player.move", seconds=4)], "--scene=hotel_room_auto", f"--generated={auto_room_dir}",
                       f"--load={save}")
    assert "3 rooms" in run.output
    end = run["player.move"]
    assert end["room"] == "bedroom"
    assert end["position"][0] == pytest.approx(nightstand[0] + 0.2 + RADIUS, abs=0.05)  # against its front
    assert end["blocked_by"]["asset"] == "nightstand_2"


def _nav(godot_script, generated, scene, a, b, *args) -> dict:
    """The navigation path between two floor points (x, z), once the navmesh has settled."""
    run = godot_script([step("wait_ready", timeout=120), step("world.nav_path", **{"from": [a[0], 0, a[1]]},
                                                               to=[b[0], 0, b[1]])],
                       f"--scene={scene}", f"--generated={generated}", *args)
    return run["world.nav_path"]


@pytest.fixture(scope="module")
def auto_room_dir(scene_export):
    return scene_export("hotel_room_auto")


def test_scene_builder_summary(godot_script, auto_room_dir):
    out = godot_script([step("wait", frames=2)], "--scene=hotel_room_auto", f"--generated={auto_room_dir}").output
    line = next(l for l in out.splitlines() if "loaded hotel_room_auto.glb" in l)
    assert "3 rooms" in line and "1 spawns" in line
    assert "geogen: unknown" not in out and "extras.geogen version" not in out


def test_navigation_reaches_rooms_through_doorways(godot_script, auto_room_dir):
    # From the entry to open floor in the bedroom (foot of the bed) and into the bathroom.
    to_bedroom = _nav(godot_script, auto_room_dir, "hotel_room_auto", (4.2, 1.3), (-0.6, 0.6))
    assert to_bedroom["points"] and to_bedroom["end_gap"] < 0.5
    to_bathroom = _nav(godot_script, auto_room_dir, "hotel_room_auto", (4.2, 1.3), (2.6, -0.6))
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


def test_player_climbs_stairs(godot_script, tmp_path):
    from geogen.layout import LayoutLoader
    export_scene(LayoutLoader().load("assets/staircase.yaml"), tmp_path / "staircase.glb")
    run = godot_script([step("player.teleport", pos=[0, 0, 3.2], yaw=0), step("player.move", seconds=1.4)],
                       "--scene=staircase", f"--generated={tmp_path}")
    end = run["player.move"]
    assert end["position"][1] == pytest.approx(3.0, abs=0.02) and end["on_floor"]


def test_player_climbs_building_stairs_to_next_storey(godot_script, tmp_path):
    from geogen.layout import LayoutLoader
    from test_building import SMALL
    building = LayoutLoader().load_string(SMALL)
    stairs = building.find("storey_0").find("stair_west").find("stairs").world_transform()
    forward = -stairs[:3, 2]                    # stairs climb along their local -Z
    start = stairs[:3, 3] - forward * (3.2 + 0.6)
    yaw = float(np.degrees(np.arctan2(-forward[0], -forward[2])))
    export_scene(building, tmp_path / "small_building.glb")
    run = godot_script([step("player.teleport", pos=[float(start[0]), 0, float(start[2])], yaw=yaw),
                        step("player.move", seconds=2.2)], "--scene=small_building", f"--generated={tmp_path}")
    end = run["player.move"]
    assert end["position"][1] == pytest.approx(3.65, abs=0.03) and end["on_floor"]


def _playtest(run_godot, generated, scene, walks=4):
    import subprocess
    from conftest import GODOT_PROJECT, _godot_binary
    # Fixed-fps makes the scripted bot run faster than real time.
    result = subprocess.run([_godot_binary(), "--headless", "--fixed-fps", "60", "--path", str(GODOT_PROJECT), "--",
                             "--scene", scene, f"--generated={generated}", f"--playtest={walks}"],
                            capture_output=True, text=True, timeout=600)
    line = next(l for l in result.stdout.splitlines() if l.startswith("playtest: "))
    return result.returncode, json.loads(line.removeprefix("playtest: "))


def test_playtest_furnished_suite(run_godot, auto_room_dir):
    code, report = _playtest(run_godot, auto_room_dir, "hotel_room_auto", walks=3)
    assert report["ok"] and code == 0, report
    assert report["rooms"] == 3 and len(report["walked"]) == 3
    assert report["targets"] > 0 and report["unreachable_targets"] == []


def test_playtest_building_from_entrance(run_godot, tmp_path):
    from geogen.layout import LayoutLoader
    from test_building import SMALL
    export_scene(LayoutLoader().load_string(SMALL), tmp_path / "small_building.glb")
    code, report = _playtest(run_godot, tmp_path, "small_building", walks=6)
    assert report["ok"] and code == 0, report
    # Lobby rooms + two guest floors, all reachable from the street entrance.
    assert report["unreachable"] == [] and report["rooms"] > 30


def test_playtest_reports_blocked_rooms(run_godot, tmp_path):
    # A room whose only door is too narrow for the player is reported by name.
    from geogen.generators.floorplan import FloorPlan
    plan = FloorPlan.from_spec({
        "rooms": {"hall": {"rect": [0, 0, 4, 3]}, "closet": {"rect": [4, 0, 2, 3], "type": "closet"}},
        "doors": [{"name": "front", "room": "hall", "side": "south", "width": 1.0},
                  {"between": ["hall", "closet"], "width": 0.5, "style": "opening"}],
    }).build("narrow")
    from geogen.core.node import SceneNode
    from geogen.core.transform import Transform
    spawn = SceneNode("spawn", transform=Transform(translation=np.array([-1.0, 0.0, -3.0])))
    spawn.meta["type"] = "spawn"
    plan.add_child(spawn)
    export_scene(plan, tmp_path / "narrow.glb")
    code, report = _playtest(run_godot, tmp_path, "narrow", walks=1)
    assert code == 1 and not report["ok"]
    assert report["unreachable"] == ["closet (closet)"]


def test_m2_hotel_playtest_from_street(run_godot, built_scene, tmp_path):
    # Milestone M2: generated hotel (lobby + 3 furnished guest floors), entered
    # from its street entrance spawn; every room and interaction reachable.
    export_scene(built_scene("hotel"), tmp_path / "hotel.glb")
    code, report = _playtest(run_godot, tmp_path, "hotel", walks=8)
    assert report["ok"] and code == 0, {k: v for k, v in report.items() if k != "reachable"}
    assert report["rooms"] == 104 and report["targets"] > 200  # lift shafts are excluded (nav: false)


def test_ride_lift_and_gates(godot_script, tmp_path):
    from geogen.layout import LayoutLoader
    from test_building import SMALL
    building = LayoutLoader().load_string(SMALL)
    car = [float(v) for v in building.find("lift_lift_shaft").world_transform()[:3, 3]]
    export_scene(building, tmp_path / "small_building.glb")
    args = ("--scene=small_building", f"--generated={tmp_path}")

    # Stand in the car, press the button, ride up one floor and walk out (-Z).
    run = godot_script([step("player.teleport", pos=[car[0], 0, car[2] + 0.2], yaw=0),
                        step("interactions.use", asset="lift_lift_shaft", actor="player"), step("wait", seconds=3.6),
                        step("interactions.list", asset="lift_lift_shaft"), step("player.move", seconds=1.2)], *args)
    assert run["interactions.list"][0]["state"] == "floor_1"
    end = run["player.move"]
    assert end["position"][1] == pytest.approx(3.65, abs=0.03) and end["room"] == "lift_lobby"
    # With the car down at the lobby, floor 1's gate blocks the empty shaft.
    run = godot_script([step("player.teleport", pos=[car[0], 3.65, car[2] - 2.4], yaw=180),
                        step("player.move", seconds=1.5)], *args)
    end = run["player.move"]
    assert end["position"][1] == pytest.approx(3.65, abs=0.03) and end["position"][2] < car[2] - 1.0


def test_godot_fixture_lights_and_switch(godot_script, auto_room_dir):
    def lights(*before):
        run = godot_script([*before, step("world.lights")], "--scene=hotel_room_auto", f"--generated={auto_room_dir}")
        return run["world.lights"]
    report = lights(step("wait", frames=5))
    assert report["total"] == report["light3d"] == 6          # KHR lights configured, not duplicated
    assert all(f["visible"] for f in report["fixtures"].values())
    report = lights(step("interactions.use", asset="bedroom_switch_1", actor="player"), step("wait", frames=40))
    assert report["fixtures"]["bedroom_light"]["visible"] is False
    assert report["fixtures"]["bathroom_light"]["visible"] is True


@pytest.fixture(scope="module")
def two_rooms_dir(tmp_path_factory):
    """Two rooms, front (a switch by each of its doors) and back, with a bench in the back
    against the wall between them (inner faces at z = 0.44 and 0.56)."""
    from geogen.core.transform import Transform
    from geogen.generators.floorplan import FloorPlan
    from geogen.layout import LayoutLoader
    from geogen.roots import find
    plan = FloorPlan.from_spec({
        "rooms": {"front": {"rect": [0, 0, 4, 4]}, "back": {"rect": [0, 4, 4, 3]}},
        "doors": [{"name": "entrance", "room": "front", "side": "south", "width": 0.9},
                  {"between": ["front", "back"], "at": 0.85, "width": 0.9}],
    }).build("two_rooms")
    bench = LayoutLoader().load(find("bench.yaml"))
    bench.transform = Transform(translation=np.array([-1.0, 0.0, 0.91]))
    plan.add_child(bench)
    out = tmp_path_factory.mktemp("generated_two_rooms")
    export_scene(plan, out / "two_rooms.glb")
    return out


def test_every_switch_of_a_light_toggles_it(godot_script, two_rooms_dir):
    # A room with a switch by each door: either one turns its light off, and the other back on.
    run = godot_script([step("interactions.use", asset="front_switch_1", actor="player"), step("wait", frames=30),
                        step("world.lights"),
                        step("interactions.use", asset="front_switch_2", actor="player"), step("wait", frames=30),
                        step("world.lights")], "--scene=two_rooms", f"--generated={two_rooms_dir}")
    assert [r["fixtures"]["front_light"]["visible"] for r in run.all("world.lights")] == [False, True]


def test_a_seat_behind_a_wall_is_not_offered(godot_script, two_rooms_dir):
    # Aiming at the wall in front, near the bench on its other side: no Sit (it used to walk
    # the player round to it). From the back room, the same seat is.
    wall = [[x, y, 0.44] for x in (-1.4, -1.0, -0.6) for y in (0.3, 0.6)]
    steps = [step("player.teleport", pos=[-1.0, 0, -0.2], yaw=0)]
    for point in wall:
        steps += [step("player.look", at=point), step("wait", frames=2), step("hud.aim")]
    steps += [step("player.teleport", pos=[-1.3, 0, 1.7], yaw=180), step("player.look", at=[-1.3, 0.3, 0.95]),
              step("wait", frames=2), step("hud.aim")]
    run = godot_script(steps, "--scene=two_rooms", f"--generated={two_rooms_dir}")
    aims = [a["focus"] for a in run.all("hud.aim")]
    assert aims[:-1] == [{}] * len(wall)
    assert aims[-1] == {"asset": "bench", "kind": "affordance", "type": "sit"}


def test_floor_plan_switch_can_be_aimed_at_and_used(godot_runtime, built_scene, auto_room_dir):
    """Floor-plan switches are light_switch.yaml, plate collider included: the focus ray
    finds one, and E throws it."""
    switch = built_scene("hotel_room_auto").find("bedroom_switch_1")
    world = switch.world_transform()
    centre = world[:3, 3] + [0.0, 0.0425, 0.0]
    out = world[:3, 2]                                  # the plate faces +Z of the asset
    rt = godot_runtime("hotel_room_auto", auto_room_dir)
    rt.call("wait_ready")
    stand = centre + out * 0.8
    rt.call("player.teleport", pos=[float(stand[0]), 0.0, float(stand[2])])
    rt.call("player.look", at=[float(c) for c in centre])
    focus = rt.call("player.status")["focus"]
    assert focus["kind"] == "interaction" and focus["asset"] == "bedroom_switch_1"
    hit = rt.call("world.raycast", **{"from": [float(stand[0]), 1.5, float(stand[2])],
                                      "to": [float(c) for c in centre - out * 0.05]})
    assert hit["interaction"]["asset"] == "bedroom_switch_1"
    assert rt.call("hud.aim")["active"] is True
    assert rt.call("player.use")["asset"] == "bedroom_switch_1"
    assert rt.call("interactions.list", asset="bedroom_switch_1")[0]["target"] == "off"


def stand(pos, yaw, pitch=0.0) -> list[dict]:
    """Steps that put the player at ``pos`` facing ``yaw``, looking ``pitch`` degrees up (-: down)."""
    return [step("player.teleport", pos=[float(v) for v in pos], yaw=yaw), step("player.look", pitch=pitch)]


INTERACTIONS = step("events.subscribe", streams=["interaction"])


def interaction_events(run) -> list[dict]:
    return [e["data"] for e in run.events if e["event"] == "interaction"]


@pytest.fixture(scope="module")
def small_building_dir(tmp_path_factory):
    from geogen.layout import LayoutLoader
    from test_building import SMALL
    out = tmp_path_factory.mktemp("generated_small")
    export_scene(LayoutLoader().load_string(SMALL), out / "small_building.glb")
    return out


# Standing in the first-floor corridor facing room 101's door.
GUEST_DOOR = stand([-9.2, 3.65, -0.3], 180)


def test_locked_guest_door_needs_its_key(godot_script, small_building_dir):
    args = ("--scene=small_building", f"--generated={small_building_dir}")
    run = godot_script([*GUEST_DOOR, INTERACTIONS, step("player.use", retry=2), step("wait", seconds=1.2),
                        step("player.move", seconds=1.2)], *args)
    assert interaction_events(run)[0]["event"] == "locked"
    assert run["player.move"]["room"] == "corridor"
    run = godot_script([*GUEST_DOOR, INTERACTIONS, step("player.lock", retry=2), step("player.use"),
                        step("wait", seconds=1.2), step("player.move", seconds=1.2)], *args, "--keys=key_room_101")
    assert [e["event"] or e["state"] for e in interaction_events(run)][:2] == ["unlocked", "open"]
    assert run["player.move"]["room"] == "room_101_entry"


def test_guest_door_closes_itself(godot_script, small_building_dir):
    run = godot_script([*GUEST_DOOR, INTERACTIONS, step("player.lock", retry=2), step("player.use"),
                        step("wait", frames=560)],
                       "--scene=small_building", f"--generated={small_building_dir}", "--keys=key_room_101",
                       engine_args=("--fixed-fps", "60"))
    states = [e["state"] for e in interaction_events(run)]
    assert states[-2:] == ["open", "closed"]           # opened, then closed after 6 s


def test_sit_lie_and_stand(godot_script, auto_room_dir):
    base = ("--scene=hotel_room_auto", f"--generated={auto_room_dir}")
    # Beside the bed, looking down at its middle: lie down.
    run = godot_script([*stand([-1.9, 0, -1.55], 180, -50), step("player.use", retry=2), step("wait", frames=20),
                        step("player.status")], *base)
    status = run["player.status"]
    assert status["pose"] == "lie" and status["pose_asset"] == "bed"
    assert status["eye"][1] == pytest.approx(0.62 + 0.25, abs=0.02)
    # At the foot of the bed: sit on its edge; walking stands you back up.
    run = godot_script([*stand([-0.6, 0, -0.4], 90, -45), step("player.use", retry=2), step("wait", seconds=0.4),
                        step("player.move", seconds=0.3)], *base)
    end = run["player.move"]
    assert end["position"][1] == pytest.approx(0.0, abs=0.02) and end["room"] == "bedroom"


def test_save_and_restore_interaction_state(godot_script, auto_room_dir, tmp_path):
    base = ("--scene=hotel_room_auto", f"--generated={auto_room_dir}")
    save = tmp_path / "save.json"
    godot_script([step("interactions.use", asset="bedroom_switch_1", actor="player"),
                  step("interactions.use", asset="nightstand", actor="player"), step("wait", frames=60),
                  step("state.save", path=str(save))], *base)
    saved = json.loads(save.read_text())
    # Keyed by stable ID (geogen.refs), not by exported node name.
    room = "hotel_room_auto:suite/bedroom"
    assert saved["format"] == "geogen-save"
    assert saved["world"][f"{room}/bedroom_switch_1#switch"] == "off"
    assert saved["world"][f"{room}/nightstand#drawer"] == "open"
    run = godot_script([step("wait", frames=5), step("world.lights")], *base, f"--load={save}")
    assert run["world.lights"]["fixtures"]["bedroom_light"]["visible"] is False


def test_m4_interactive_hotel_room(godot_script, built_scene, auto_room_dir):
    """Milestone M4: in the furnished suite, lock/unlock and open the room door,
    open the wardrobe and a nightstand drawer, switch the lights, sit on the
    chair and lie on the bed."""
    chair = built_scene("hotel_room_auto").find("chair").world_transform()[:3, 3]
    base = ("--scene=hotel_room_auto", f"--generated={auto_room_dir}")

    # The door: locked from the corridor spawn without the key, opens with it.
    run = godot_script([INTERACTIONS, step("player.use", retry=2), step("wait", seconds=1.0),
                        step("player.move", seconds=1.0)], *base)
    assert interaction_events(run)[0]["event"] == "locked"
    run = godot_script([step("player.lock", retry=2), step("player.use"), step("wait", seconds=1.0),
                        step("player.move", seconds=1.4)], *base, "--keys=key_suite")
    assert run["player.move"]["room"] in ("corridor", "bedroom")   # through the door
    # L with the key toggles the lock (here: unlocks it), without opening the door.
    run = godot_script([step("player.lock", retry=2), step("wait", frames=10), step("interactions.states")],
                       *base, "--keys=key_suite")
    assert run["interactions.states"]["hotel_room_auto:suite/entry#swing"] == {"locked": False, "state": "closed"}
    # Wardrobe doors, a drawer and the bedroom light.
    run = godot_script([step("interactions.use", asset=a, actor="player")
                        for a in ("wardrobe", "nightstand", "bedroom_switch_1")]
                       + [step("wait", frames=90), step("interactions.states"), step("world.lights")], *base)
    states = run["interactions.states"]
    room = "hotel_room_auto:suite/bedroom"
    assert states[f"{room}/wardrobe#doors"]["state"] == "open" and states[f"{room}/nightstand#drawer"]["state"] == "open"
    assert run["world.lights"]["fixtures"]["bedroom_light"]["visible"] is False
    # Sit on the chair (looking down at it from in front), lie on the bed.
    run = godot_script([*stand([chair[0], 0, chair[2] + 1.0], 0, -50), step("player.use", retry=2),
                        step("wait", frames=20), step("player.status")], *base)
    assert (run["player.status"]["pose"], run["player.status"]["pose_asset"]) == ("sit", "chair")
    run = godot_script([*stand([-1.9, 0, -1.55], 180, -50), step("player.use", retry=2), step("wait", frames=20),
                        step("player.status")], *base)
    assert (run["player.status"]["pose"], run["player.status"]["pose_asset"]) == ("lie", "bed")


STRIP = """
name: strip
city:
  seed: 3
  blocks: [5, 1]
  block_size: [30, 32]
  avenues: { ew: [0] }
  lot_width: [11, 14]
  buildings:
    residential: [{ recipe: detached_house, storeys: 1, interior: full }]
    commercial: [{ recipe: shop_row, storeys: 2, interior: lobby }]
spawns:
  start: { position: [-75, 0, -20], facing: east }
"""


@pytest.fixture(scope="module")
def strip_dir(tmp_path_factory):
    """A 5-block street exported as chunks (generated/strip_chunks/)."""
    from geogen.chunks import export_chunks
    from geogen.layout.composer import SceneComposer

    out = tmp_path_factory.mktemp("generated")
    export_chunks(SceneComposer().compose_string(STRIP), out / "strip_chunks", name="strip")
    return out


def _stream_walk(godot_script, generated, seconds, *extra):
    run = godot_script([step("player.move", seconds=seconds)], "--scene=strip", f"--generated={generated}",
                       "--stream-radius=20,1000,6", *extra)
    assert "geogen: streaming strip.chunks.json (6 chunks)" in run.output
    return run["player.move"]


def test_stream_loads_near_chunks_full_and_far_ones_as_lod(godot_script, strip_dir):
    end = _stream_walk(godot_script, strip_dir, 0.3)
    stream = end["stream"]
    assert "base" in stream["full"] and "block_0_0" in stream["full"]
    assert {"block_2_0", "block_3_0", "block_4_0"} <= set(stream["lod"])
    assert not set(stream["full"]) & set(stream["lod"])
    assert end["on_floor"]                      # the street loaded under the spawn


def test_stream_swaps_detail_as_the_player_walks(godot_script, strip_dir):
    end = _stream_walk(godot_script, strip_dir, 12)
    assert end["position"][0] > -35             # walked east along the avenue
    stream = end["stream"]
    assert "block_1_0" in stream["full"] and "block_0_0" in stream["lod"]
    assert stream["unloads"] > 0


def test_stream_loads_interiors_next_to_buildings(godot_script, strip_dir):
    run = godot_script([step("wait", frames=90), step("player.status")], "--scene=strip", f"--generated={strip_dir}",
                       "--stream-radius=20,1000,6", "--spawn=-80,0.2,-12.5")
    status = run["player.status"]
    assert status["stream"]["interiors"] and all(i.startswith("block_0_0/") for i in status["stream"]["interiors"])


def test_rules_hold_an_interaction_while_its_chunk_is_away(godot_runtime, strip_dir):
    """A door's state lives in the rules program while its chunk is unloaded: rules read it as the
    player left it, and what they do to it meanwhile is how it comes back (geogen.3.13)."""
    door = "strip:city/block_0_0/lot_0_0_0/lot_0_0_0_shop_row/storey_0/shop_door#swing"
    rt = godot_runtime("strip", strip_dir, "--stream-radius=20,1000,6")
    rt.call("wait_ready")
    rt.call("interactions.set", asset="shop_door", state="open", instant=True)
    rt.call("player.teleport", pos=[70, 0.2, -20])                  # block_0_0 unloads
    rt.call("time.advance", seconds=3)
    assert rt.call("state.get", key=door)["value"] == "open"
    assert rt.call("state.save")["world"][door] == "open"
    rt.call("state.set", key=door, value="closed")                  # back to how it starts
    rt.call("player.teleport", pos=[-75, 0.2, -20])
    rt.call("time.advance", seconds=3)
    assert [i["state"] for i in rt.call("interactions.list", asset="shop_door")] == ["closed"]


@pytest.fixture(scope="module")
def district(tmp_path_factory, built_scene):
    """Milestone M3: the town district as a streamed chunk export, plus its building entrances."""
    from geogen.chunks import export_chunks

    root = built_scene("town")
    out = tmp_path_factory.mktemp("generated")
    index = json.loads(export_chunks(root, out / "town_chunks", name="town").read_text())
    entrances = {}
    for node in root.iter_nodes():
        if node.name != "entrance_spawn":
            continue
        building = node.parent
        m = node.world_transform()
        forward = m[:3, 2]
        entrances.setdefault(building.meta["building"]["entry"], []).append({
            "name": building.name, "shell": "building.shell" in building.tags,
            "spawn": ",".join(f"{v:.3f}" for v in m[:3, 3]),
            "yaw": float(np.degrees(np.arctan2(-forward[0], -forward[2]))),
        })
    return out, index, entrances


def _district_walk(godot_script, generated, steps, *args):
    run = godot_script(steps, "--scene=town", f"--generated={generated}", *args)
    assert "geogen: streaming town.chunks.json" in run.output
    return run["player.move"]


def test_m3_district_exports_streamable_chunks(district):
    _, index, entrances = district
    assert index["format"] == "geogen-chunks" and index["spawns"][0]["name"] == "start"
    kinds = set(entrances)
    assert {"hotel", "shop_row", "office", "detached_house"} <= kinds
    interiors = {i["building"] for c in index["chunks"] for i in c["interiors"]}
    # Enterable buildings stream their interiors; background shells have none.
    assert any("hotel" in b for b in interiors) and any("shop_row" in b for b in interiors)
    assert all(e["shell"] for e in entrances["office"])
    assert not any("office" in b for b in interiors)


def test_m3_walk_the_street(godot_script, district):
    generated, _, _ = district
    end = _district_walk(godot_script, generated, [step("player.move", seconds=1.5)])
    assert end["on_floor"] and end["position"][2] > 2          # walked north across the avenue onto the sidewalk
    assert "base" in end["stream"]["full"]


def test_m3_enter_a_shop(godot_script, district):
    generated, _, entrances = district
    shop = next(e for e in entrances["shop_row"] if not e["shell"])
    end = _district_walk(godot_script, generated,
                         [step("interactions.use", asset="shop_door", actor="player"), step("wait", seconds=1.5),
                          step("player.move", seconds=1.6)], f"--spawn={shop['spawn']}", f"--yaw={shop['yaw']}")
    assert end["room"] == "shop"
    assert any(shop["name"] in i for i in end["stream"]["interiors"])


def test_m3_enter_the_hotel(godot_script, district):
    generated, _, entrances = district
    hotel = entrances["hotel"][0]
    end = _district_walk(godot_script, generated,
                         [step("interactions.use", asset="entrance", actor="player"), step("wait", seconds=2),
                          step("player.move", seconds=2)], f"--spawn={hotel['spawn']}", f"--yaw={hotel['yaw']}")
    assert end["room"] == "lobby"


def _stream_nav(godot_script, generated, spawn, a, b):
    return _nav(godot_script, generated, "strip", a, b, "--stream-radius=20,1000,12", f"--spawn={spawn}")


def test_stream_navigation_tiles_join_along_the_street(godot_script, strip_dir):
    # -72 is a 24 m tile boundary: the path runs straight through it.
    path = _stream_nav(godot_script, strip_dir, "-75,0,-20", (-75, -20), (-60, -20))
    assert path["reached"] and path["length"] == pytest.approx(15.0, abs=0.3)


def test_stream_navigation_reaches_into_a_shop(godot_script, strip_dir):
    # From the avenue, up the curb and through the (closed, ignored) shop door onto its floor.
    path = _stream_nav(godot_script, strip_dir, "-80,0.2,-18", (-80, -20), (-80.97, -7.5))
    assert path["reached"] and path["end_gap"] < 0.4
    assert max(p[1] for p in path["points"]) > 0.2          # on the shop floor, above the sidewalk
