"""Traffic in the Godot runtime (runtime/godot/scripts/traffic.gd), headless and sped up.

Vehicles drive the exported lane graph; these tests assert on the ``traffic
summary`` printed after a simulated span (see GeogenTraffic.report()).
"""

import json
import re

import pytest

from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from geogen.main import _build_registry

FAST = ("--fixed-fps", "60")


@pytest.fixture(scope="module")
def crossroads_dir(scene_export):
    return scene_export("crossroads")


def simulate(run_godot, generated, scene, seconds, *extra, timescale=8) -> dict:
    out = run_godot("--scene", scene, f"--generated={generated}", f"--timescale={timescale}",
                    f"--simulate={seconds}", *extra, engine_args=FAST)
    line = next(l for l in out.splitlines() if l.startswith("traffic summary: "))
    (report,) = json.loads(line.removeprefix("traffic summary: "))
    return report


def test_traffic_flows_without_overlaps_or_gridlock(run_godot, crossroads_dir):
    # The player stands on a sidewalk (the scene's spawn), out of the way.
    report = simulate(run_godot, crossroads_dir, "crossroads", 240)
    assert report["vehicles"] >= 8
    assert report["overlaps"] == 0
    assert report["min_moved"] > 150          # every vehicle kept going (no gridlock)
    # Red lights and pedestrians hold vehicles up, but a blocked vehicle picks another exit
    # after `give_up` (25 s), so nobody waits for long.
    assert report["max_idle"] < 75 and report["max_wait"] < 75
    # The avenue's intersections have lights: vehicles stop on red and go on green.
    assert report["red_stops"] >= 5 and report["green_claims"] >= 10
    assert report["claims"] > 100 and set(report["turns"]) == {"left", "right", "straight"}
    assert report["engines"] == report["vehicles"]      # every vehicle on the road has its engine sound


def test_a_player_in_the_lane_stops_traffic(run_godot, crossroads_dir):
    # Standing in the avenue's eastbound lane (z = 1.5): the next vehicle along stops short.
    report = simulate(run_godot, crossroads_dir, "crossroads", 180, "--spawn=-19,0.2,1.5")
    assert report["yields"] >= 1
    assert 0.5 < report["min_player_gap"] < 8.0
    assert report["overlaps"] == 0


def test_traffic_is_deterministic(run_godot, crossroads_dir):
    # The pedestrians too: traffic yields to them, so an NPC a frame out drifts the traffic later
    # (the navmesh used to sync on a worker thread, on whichever frame it finished; geogen.50).
    def run():
        out = run_godot("--scene", "crossroads", f"--generated={crossroads_dir}", "--timescale=8",
                        "--simulate=30", engine_args=FAST)
        lines = [l for l in out.splitlines() if l.startswith(("traffic summary: ", "npc summary: "))]
        return [re.sub(r'"cost_us":[0-9.]+,?', "", l) for l in lines]

    a, b = run(), run()
    assert len(a) == 2 and a == b


def test_open_route_vehicles_leave_and_come_back(run_godot, tmp_path):
    from pathlib import Path

    assets = Path(__file__).parent.parent / "assets"
    root = SceneComposer(assets_dir=assets).compose_string(
        "name: road\n"
        "routes:\n"
        "  road: { path: { spline: [[-60, 0, 10], [0, 0, 16], [60, 0, 10]] }, lanes: 2, speed: 13 }\n"
        "place:\n"
        "  traffic: { traffic: traffic/town.yaml, seed: 2 }\n"
        "spawns:\n"
        "  start: { position: [0, 0, 0], facing: north }\n")
    export_scene(root, tmp_path / "road.glb")
    report = simulate(run_godot, tmp_path, "road", 90)
    assert report["exits"] > 3 and report["overlaps"] == 0
    assert report["min_moved"] > 100


def test_pedestrians_stroll_and_rest_while_traffic_yields(run_godot, crossroads_dir):
    out = run_godot("--scene", "crossroads", f"--generated={crossroads_dir}", "--timescale=8",
                    "--simulate=180", engine_args=FAST)
    npcs = json.loads(next(l for l in out.splitlines() if l.startswith("npc summary: ")).split(": ", 1)[1])
    (traffic,) = json.loads(next(l for l in out.splitlines() if l.startswith("traffic summary: ")).split(": ", 1)[1])
    assert len(npcs) == 6
    for npc in npcs:
        used = npc["used"]
        assert used.get("self/stroll", 0) >= 1, npc["npc"]            # walked the sidewalks
        assert all(k.startswith(("self/", "bench")) for k in used)     # only outside affordances
    assert sum(1 for n in npcs if any(k.startswith("bench") for k in n["used"])) >= 3
    assert sum(n["crosswalks"] for n in npcs) >= 3                   # they cross at the crossings
    assert traffic["yields"] >= 1 and traffic["overlaps"] == 0
    assert traffic["min_moved"] > 50                                   # traffic still flows


def test_pedestrians_wait_for_the_walk_signal(run_godot, crossroads_dir):
    # The avenue's three intersections are signalled: six crossings with pavement at both ends,
    # a head at each end. 240 s is nearly six 41 s cycles.
    out = run_godot("--scene", "crossroads", f"--generated={crossroads_dir}", "--timescale=8",
                    "--simulate=240", "--npc-trace", engine_args=FAST)
    npcs = json.loads(next(l for l in out.splitlines() if l.startswith("npc summary: ")).split(": ", 1)[1])
    (traffic,) = json.loads(next(l for l in out.splitlines() if l.startswith("traffic summary: ")).split(": ", 1)[1])
    # The heads cycle walk -> flashing -> don't walk with the vehicle phases: about six times each,
    # and never showing walk or flashing while the traffic across the crossing isn't at red.
    assert traffic["pedestrian_heads"] == 12
    assert 30 <= traffic["walks"] <= 42 and 30 <= traffic["flashes"] <= 42 and 30 <= traffic["dont_walks"] <= 42
    assert traffic["walk_conflicts"] == 0
    # Nobody steps off a kerb onto a signalled crossing while it says don't walk: they wait
    # there (for less than a cycle), and they do get across.
    trace = [json.loads(l.split(": ", 1)[1]) for l in out.splitlines() if l.startswith("npc: ")]
    steps = [e for e in trace if e.get("event") == "signal_step"]
    waits = [e for e in trace if e.get("event") == "signal_wait"]
    # (A step is the frame the kerb line is crossed, half a second after setting off on walk,
    # so the flashing interval may just have begun; a steady don't-walk is never right.)
    assert len(steps) >= 4 and all(e["state"] in ("walk", "flash") for e in steps)
    assert sum(1 for e in steps if e["state"] == "walk") >= len(steps) - 1
    assert len(waits) >= 3 and all(e["state"] in ("dont", "flash") for e in waits)
    assert sum(n["signal_violations"] for n in npcs) == 0
    assert sum(n["signal_crossings"] for n in npcs) == len(steps)
    assert sum(n["signal_waits"] for n in npcs) == len(waits)
    assert 1.0 < max(n["signal_wait"] for n in npcs) < 41.0
    # Traffic keeps flowing, and still yields to people in the road.
    assert traffic["overlaps"] == 0 and traffic["min_moved"] > 150 and traffic["max_idle"] < 75
    assert traffic["yields"] >= 1 and traffic["green_claims"] >= 10


def test_the_hud_tells_the_player_at_a_crossing_when_to_walk(godot_runtime, crossroads_dir):
    """hud_crossing.gd: on the kerb of a signalled crossing, or on it, the HUD shows what its
    heads show, and changes with them; anywhere else it shows nothing."""
    manifest = json.loads((crossroads_dir / "crossroads.manifest.json").read_text())
    crossing = manifest["traffic"]["pedestrian_signals"][0]
    x0, z0, x1, z1 = crossing["crosswalk"]
    along_x = x1 - x0 > z1 - z0
    mid = ((x0 + x1) / 2, (z0 + z1) / 2)
    kerb = (x0 - 0.8, mid[1]) if along_x else (mid[0], z0 - 0.8)
    rt = godot_runtime("crossroads", crossroads_dir)
    rt.call("player.teleport", pos=[kerb[0], 0.2, kerb[1]])
    seen = {}
    for _ in range(14):                     # 14 x 3 s: a whole 41 s cycle
        shown = rt.call("hud.crossing")
        assert shown["visible"] and shown["crossing"] == crossing["id"] and shown["where"] == "kerb", shown
        seen[shown["state"]] = shown
        rt.call("time.advance", seconds=3)
    assert set(seen) == {"walk", "flash", "dont"}
    assert seen["walk"]["text"] == "WALK" and seen["dont"]["line"] == "Wait for the signal"
    rt.call("player.teleport", pos=[mid[0], 0.05, mid[1]])
    assert rt.call("hud.crossing")["where"] == "crossing"
    rt.call("player.teleport", pos=[manifest["spawns"][0]["position"][0], 0.2, manifest["spawns"][0]["position"][2] - 4.0])
    assert rt.call("hud.crossing") == {"visible": False}


def _summaries(out: str) -> tuple[list, dict, dict]:
    lines = out.splitlines()
    npcs = json.loads(next(l for l in lines if l.startswith("npc summary: ")).split(": ", 1)[1])
    (traffic,) = json.loads(next(l for l in lines if l.startswith("traffic summary: ")).split(": ", 1)[1])
    status = json.loads(next(l for l in lines if l.startswith("status: ")).split(": ", 1)[1])
    return npcs, traffic, status


def test_night_empties_the_streets_and_lights_the_lamps(run_godot, crossroads_dir):
    out = run_godot("--scene", "crossroads", f"--generated={crossroads_dir}", "--time=23:30", "--timescale=8",
                    "--simulate=90", "--status", engine_args=FAST)
    npcs, traffic, status = _summaries(out)
    assert status["night"] and status["lamps_on"] >= 10
    # Pedestrians go home (vanish out of the player's sight); the fleet thins to the schedule.
    assert sum(1 for n in npcs if n["away"]) >= 4
    assert traffic["share"] < 0.3 and traffic["vehicles"] <= 4 and traffic["parked"] >= 8
    assert traffic["overlaps"] == 0


def test_morning_brings_people_and_traffic_back(run_godot, crossroads_dir):
    # 06:40 plus two hours of world time (a day lasts 1440 s, run at 8x).
    out = run_godot("--scene", "crossroads", f"--generated={crossroads_dir}", "--time=06:40", "--timescale=8",
                    "--simulate=120", "--status", engine_args=FAST)
    npcs, traffic, status = _summaries(out)
    assert not status["night"] and status["clock"] >= "08:30"
    assert not any(n["away"] for n in npcs)
    assert sum(n["returns"] for n in npcs) >= 4
    assert traffic["share"] == 1.0 and traffic["parked"] == 0


@pytest.fixture(scope="module")
def crossing_dir(scene_export):
    return scene_export("level_crossing")


def _train_summary(out: str) -> dict:
    (report,) = json.loads(next(l for l in out.splitlines() if l.startswith("train summary: ")).split(": ", 1)[1])
    return report


def test_trains_close_the_crossing_and_traffic_waits(run_godot, crossing_dir):
    # 13:00 is 780 s of world time: departure 3 (offset 10, headway 240) is 50 s out,
    # inside its crossing window. Ten minutes of world time cover two more trains.
    out = run_godot("--scene", "level_crossing", f"--generated={crossing_dir}", "--time=13:00",
                    "--day-length=1440", "--timescale=8", "--simulate=600", engine_args=FAST)
    train = _train_summary(out)
    (traffic,) = json.loads(next(l for l in out.splitlines() if l.startswith("traffic summary: ")).split(": ", 1)[1])
    assert train["departures"] >= 3 and train["closed_time"] > 60
    assert train["horns"] >= 2                        # the driver sounds the horn for each closure
    events = [json.loads(l.split(": ", 1)[1]) for l in out.splitlines() if l.startswith("interaction event: ")]
    downs = [e for e in events if e["asset"].startswith("crossing_main_x1") and e["state"] == "down"]
    ups = [e for e in events if e["asset"].startswith("crossing_main_x1") and e["state"] == "up"]
    assert len(downs) >= 4 and len(ups) >= 4          # both barriers, several trains
    assert traffic["crossing_violations"] == 0 and traffic["overlaps"] == 0
    assert traffic["exits"] > 10                      # traffic kept flowing between trains


def test_trains_keep_the_timetable(run_godot, crossing_dir):
    # 12:40 = 760 s: departure 3 left 30 s ago and is dwelling at 'south' (arrive 21.5, depart 41.5).
    head = []
    for seconds in (1, 5):
        out = run_godot("--scene", "level_crossing", f"--generated={crossing_dir}", "--time=12:40",
                        "--day-length=1440", f"--simulate={seconds}", engine_args=FAST)
        (active,) = _train_summary(out)["active"]
        assert active["departure"] == 3
        head.append(active["head"])
    assert head[0] == head[1] == pytest.approx(165.6, abs=0.5)     # stopped: middle at the station
