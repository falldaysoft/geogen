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
    report = simulate(run_godot, crossroads_dir, "crossroads", 90, "--spawn=-19,0.2,1.5")
    assert report["yields"] >= 1
    assert 0.5 < report["min_person_gap"] < 8.0
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
