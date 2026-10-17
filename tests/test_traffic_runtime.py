"""Traffic in the Godot runtime (runtime/godot/scripts/traffic.gd), headless and sped up.

Vehicles drive the exported lane graph; these tests assert on what ``time.simulate`` reports
after a simulated span (traffic: GeogenTraffic.report(), npcs, trains).
"""

import json

import pytest

from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from geogen.main import _build_registry
from geogen.runtime_client import step

FAST = ("--fixed-fps", "60")


@pytest.fixture(scope="module")
def crossroads_dir(scene_export):
    return scene_export("crossroads")


def run_sim(godot_script, generated, scene, seconds, *extra, timescale=8, before=(), after=(), **start):
    """Run ``seconds`` of world time (sped up); the ScriptRun, with time.simulate's report."""
    return godot_script([*before, step("time.simulate", seconds=seconds), *after], f"--scene={scene}",
                        f"--generated={generated}", *extra, start={"timescale": timescale, **start},
                        engine_args=FAST)


def simulate(godot_script, generated, scene, seconds, **start) -> dict:
    (report,) = run_sim(godot_script, generated, scene, seconds, **start)["time.simulate"]["traffic"]
    return report


def test_traffic_flows_without_overlaps_or_gridlock(godot_script, crossroads_dir):
    # The player stands on a sidewalk (the scene's spawn), out of the way.
    report = simulate(godot_script, crossroads_dir, "crossroads", 240)
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


def test_a_player_in_the_lane_stops_traffic(godot_script, crossroads_dir):
    # Standing in the avenue's eastbound lane (z = 1.5): the next vehicle along stops short.
    report = simulate(godot_script, crossroads_dir, "crossroads", 180, spawn=[-19, 0.2, 1.5])
    assert report["yields"] >= 1
    assert 0.5 < report["min_player_gap"] < 8.0
    assert report["overlaps"] == 0


def test_traffic_is_deterministic(godot_script, crossroads_dir):
    # The pedestrians too: traffic yields to them, so an NPC a frame out drifts the traffic later
    # (the navmesh used to sync on a worker thread, on whichever frame it finished; geogen.50).
    def run():
        report = run_sim(godot_script, crossroads_dir, "crossroads", 30)["time.simulate"]
        return [report["traffic"], [{k: v for k, v in n.items() if k != "cost_us"} for n in report["npcs"]]]

    a, b = run(), run()
    assert a[0] and a[1] and a == b


def test_open_route_vehicles_leave_and_come_back(godot_script, tmp_path):
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
    report = simulate(godot_script, tmp_path, "road", 90)
    assert report["exits"] > 3 and report["overlaps"] == 0
    assert report["min_moved"] > 100


def test_pedestrians_stroll_and_rest_while_traffic_yields(godot_script, crossroads_dir):
    report = run_sim(godot_script, crossroads_dir, "crossroads", 180)["time.simulate"]
    npcs, (traffic,) = report["npcs"], report["traffic"]
    assert len(npcs) == 6
    for npc in npcs:
        used = npc["used"]
        assert used.get("self/stroll", 0) >= 1, npc["npc"]            # walked the sidewalks
        assert all(k.startswith(("self/", "bench")) for k in used)     # only outside affordances
    assert sum(1 for n in npcs if any(k.startswith("bench") for k in n["used"])) >= 3
    assert sum(n["crosswalks"] for n in npcs) >= 3                   # they cross at the crossings
    assert sum(n["unstuck"] for n in npcs) == 0                       # nobody wedged
    assert traffic["yields"] >= 1 and traffic["overlaps"] == 0
    assert traffic["min_moved"] > 50                                   # traffic still flows


def test_pedestrians_wait_for_the_walk_signal(godot_script, crossroads_dir):
    # The avenue's three intersections are signalled: six crossings with pavement at both ends,
    # a head at each end. 240 s is nearly six 41 s cycles.
    run = run_sim(godot_script, crossroads_dir, "crossroads", 240, "--npc-trace")
    out = run.output
    npcs, (traffic,) = run["time.simulate"]["npcs"], run["time.simulate"]["traffic"]
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


def _summaries(godot_script, generated, seconds, time) -> tuple[list, dict, dict, dict]:
    """NPC and traffic reports after ``seconds`` from ``time``, the clock, and the lights."""
    run = run_sim(godot_script, generated, "crossroads", seconds, time=time,
                  after=[step("time.clock"), step("world.lights")])
    (traffic,) = run["time.simulate"]["traffic"]
    return run["time.simulate"]["npcs"], traffic, run["time.clock"], run["world.lights"]


def test_night_empties_the_streets_and_lights_the_lamps(godot_script, crossroads_dir):
    npcs, traffic, clock, lights = _summaries(godot_script, crossroads_dir, 90, "23:30")
    assert clock["night"] and sum(1 for f in lights["fixtures"].values() if f["visible"]) >= 10
    # Pedestrians go home (vanish out of the player's sight); the fleet thins to the schedule.
    assert sum(1 for n in npcs if n["away"]) >= 4
    assert traffic["share"] < 0.3 and traffic["vehicles"] <= 4 and traffic["parked"] >= 8
    assert traffic["overlaps"] == 0


def test_morning_brings_people_and_traffic_back(godot_script, crossroads_dir):
    # 06:40 plus two hours of world time (a day lasts 1440 s, run at 8x).
    npcs, traffic, clock, _ = _summaries(godot_script, crossroads_dir, 120, "06:40")
    assert not clock["night"] and clock["clock"] >= "08:30"
    assert not any(n["away"] for n in npcs)
    assert sum(n["returns"] for n in npcs) >= 4
    assert traffic["share"] == 1.0 and traffic["parked"] == 0


@pytest.fixture(scope="module")
def crossing_dir(scene_export):
    return scene_export("level_crossing")


def test_trains_close_the_crossing_and_traffic_waits(godot_script, crossing_dir):
    # 13:00 is 780 s of world time: departure 3 (offset 10, headway 240) is 50 s out,
    # inside its crossing window. Ten minutes of world time cover two more trains.
    run = run_sim(godot_script, crossing_dir, "level_crossing", 600, time="13:00", day_length=1440,
                  before=[step("events.subscribe", streams=["interaction"])])
    (train,), (traffic,) = run["time.simulate"]["trains"], run["time.simulate"]["traffic"]
    assert train["departures"] >= 3 and train["closed_time"] > 60
    assert train["horns"] >= 2                        # the driver sounds the horn for each closure
    events = [e["data"] for e in run.events if e["event"] == "interaction"]
    downs = [e for e in events if e["asset"].startswith("crossing_main_x1") and e["state"] == "down"]
    ups = [e for e in events if e["asset"].startswith("crossing_main_x1") and e["state"] == "up"]
    assert len(downs) >= 4 and len(ups) >= 4          # both barriers, several trains
    assert traffic["crossing_violations"] == 0 and traffic["overlaps"] == 0
    assert traffic["exits"] > 10                      # traffic kept flowing between trains


def test_trains_keep_the_timetable(godot_script, crossing_dir):
    # 12:40 = 760 s: departure 3 left 30 s ago and is dwelling at 'south' (arrive 21.5, depart 41.5).
    head = []
    for seconds in (1, 5):
        run = run_sim(godot_script, crossing_dir, "level_crossing", seconds, time="12:40", day_length=1440,
                      timescale=1)
        (train,) = run["time.simulate"]["trains"]
        (active,) = train["active"]
        assert active["departure"] == 3
        head.append(active["head"])
    assert head[0] == head[1] == pytest.approx(165.6, abs=0.5)     # stopped: middle at the station
