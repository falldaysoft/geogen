"""Railways, level crossings and trains (geogen/railway.py, geogen/trains.py)."""

from pathlib import Path

import numpy as np
import pytest

from geogen.core import meshops
from geogen.trains import DT, closures, simulate_run

ASSETS = Path(__file__).parent.parent / "assets"


@pytest.fixture(scope="module")
def scene(built_scene):
    return built_scene("level_crossing")


def _railway(scene):
    from geogen.traffic import build_traffic

    return build_traffic(scene)["railways"][0]


def test_track_meshes_are_valid(scene):
    rail = next(n for n in scene.iter_nodes() if n.name == "railway_main")
    for node, mesh in rail.iter_meshes():
        assert not meshops.validate(mesh).issues, node.name
    names = {n.name for n in rail.iter_nodes()}
    assert {"railway_main_ballast", "railway_main_sleepers", "railway_main_rail_l", "railway_main_rail_r"} <= names
    assert {"platform_south_r", "platform_north_r"} <= names


def test_line_stations_and_crossing(scene):
    rail = _railway(scene)
    assert rail["length"] == pytest.approx(444, abs=5)
    assert [s["name"] for s in rail["stations"]] == ["south", "north"]
    (crossing,) = rail["crossings"]
    # The road (two lanes) crosses mid-line; each lane gets a barrier and a stop point.
    assert crossing["s"][0] < rail["length"] / 2 < crossing["s"][1]
    assert sorted(ln["lane"] for ln in crossing["lanes"]) == ["road_n0", "road_p0"]
    assert len(crossing["barriers"]) == 2
    barriers = [n for n in scene.iter_nodes() if n.meta.get("crossing") == crossing["id"]]
    assert len(barriers) == 2
    for b in barriers:       # beside the road, not on the track
        p = b.world_transform()[:3, 3]
        assert 1.5 < abs(p[2]) < 4.0


def test_simulated_run_stops_at_stations():
    stations = [{"name": "a", "s": 100.0}, {"name": "b", "s": 300.0}]
    run, stops = simulate_run(400.0, False, 60.0, stations, 18.0, 0.6, 0.9, 20.0)
    assert np.all(np.diff(run) >= -1e-9) and run[0] == 60.0 and run[-1] == pytest.approx(400.0)
    assert [s["station"] for s in stops] == ["a", "b"]
    for stop in stops:        # standing still for the dwell, with the train's middle at the station
        k0, k1 = int(stop["arrive"] / DT) + 1, int(stop["depart"] / DT) - 1
        assert np.ptp(run[k0:k1]) < 1e-6
    assert run[int(stops[0]["arrive"] / DT) + 2] == pytest.approx(130.0)


def test_loop_run_does_a_lap():
    run, stops = simulate_run(500.0, True, 50.0, [{"name": "a", "s": 200.0}], 15.0, 0.6, 0.9, 10.0)
    assert run[0] == 0.0 and run[-1] == pytest.approx(500.0) and len(stops) == 1


def test_crossing_windows_cover_the_train_on_it():
    run, _ = simulate_run(400.0, False, 60.0, [], 18.0, 0.6, 0.9, 20.0)
    windows = closures(run, 60.0, [{"id": "x", "s": [190.0, 210.0]}], 20.0, 18.0, 400.0, False)["x"]
    (t0, t1), = windows
    on = [k * DT for k, s in enumerate(run) if s >= 190.0 and s - 60.0 <= 210.0]
    assert t0 <= on[0] - 19.9 and t1 >= on[-1]


def test_train_placement(scene):
    train = next(n for n in scene.iter_nodes() if n.meta.get("type") == "train")
    data = train.meta["train"]
    assert data["railway"] == "main" and len(data["cars"]) == 4
    assert [s["station"] for s in data["stops"]] == ["south", "north"]
    assert data["closures"]["main_x1"]
    offsets = [c["offset"] for c in data["cars"]]
    assert offsets == sorted(offsets) and offsets[0] > 0
    # The cars stand on the rails (their origins at the rail head).
    for car in train.children:
        assert car.world_transform()[1, 3] == pytest.approx(0.16, abs=0.02)
