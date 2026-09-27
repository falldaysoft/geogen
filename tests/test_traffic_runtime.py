"""Traffic in the Godot runtime (runtime/godot/scripts/traffic.gd), headless and sped up.

Vehicles drive the exported lane graph; these tests assert on the ``traffic
summary`` printed after a simulated span (see GeogenTraffic.report()).
"""

import json

import pytest

from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from geogen.main import _build_registry

FAST = ("--fixed-fps", "60")


@pytest.fixture(scope="module")
def crossroads_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("generated_traffic")
    export_scene(_build_registry()["crossroads"](), out / "crossroads.glb")
    return out


def simulate(run_godot, generated, scene, seconds, *extra, timescale=8) -> dict:
    out = run_godot("--scene", scene, f"--generated={generated}", f"--timescale={timescale}",
                    f"--simulate={seconds}", *extra, engine_args=FAST)
    line = next(l for l in out.splitlines() if l.startswith("traffic summary: "))
    (report,) = json.loads(line.removeprefix("traffic summary: "))
    return report


def test_traffic_flows_without_overlaps_or_gridlock(run_godot, crossroads_dir):
    # The player stands on a sidewalk (the scene's spawn), out of the way.
    report = simulate(run_godot, crossroads_dir, "crossroads", 240)
    assert report["vehicles"] >= 12
    assert report["overlaps"] == 0
    assert report["min_moved"] > 150          # every vehicle kept going (no gridlock)
    # A blocked vehicle picks another exit after `give_up` (25 s), so nobody waits much longer.
    assert report["max_idle"] < 45 and report["max_wait"] < 45
    assert report["claims"] > 100 and set(report["turns"]) == {"left", "right", "straight"}


def test_a_player_in_the_lane_stops_traffic(run_godot, crossroads_dir):
    # Standing in the avenue's eastbound lane (z = 1.5): the next vehicle along stops short.
    report = simulate(run_godot, crossroads_dir, "crossroads", 180, "--spawn=-19,0.2,1.5")
    assert report["yields"] >= 1
    assert 0.5 < report["min_player_gap"] < 8.0
    assert report["overlaps"] == 0


def test_traffic_is_deterministic(run_godot, crossroads_dir):
    a = simulate(run_godot, crossroads_dir, "crossroads", 30)
    b = simulate(run_godot, crossroads_dir, "crossroads", 30)
    assert a == b


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
    assert traffic["yields"] >= 1 and traffic["overlaps"] == 0
    assert traffic["min_moved"] > 50                                   # traffic still flows


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
def crossing_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("generated_rail")
    export_scene(_build_registry()["level_crossing"](), out / "level_crossing.glb")
    return out


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
