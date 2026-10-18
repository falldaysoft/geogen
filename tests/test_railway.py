"""Railways, level crossings and trains (geogen/railway.py, geogen/trains.py)."""

from pathlib import Path

import numpy as np
import pytest

from geogen.core import meshops
from geogen.trains import DT, check_spacing, closures, simulate_run

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


def test_loop_run_does_a_lap_from_its_first_station_back_to_it():
    stations = [{"name": "b", "s": 400.0}, {"name": "a", "s": 200.0}]
    run, stops = simulate_run(500.0, True, 50.0, stations, 15.0, 0.6, 0.9, 10.0)
    assert run[0] == 225.0 and run[-1] == pytest.approx(725.0)        # middle at 'a', round to it again
    assert [s["station"] for s in stops] == ["a", "b"] and stops[0]["arrive"] == 0.0
    # Given a headway, the lap takes exactly that: the time to spare is longer dwells.
    run, stops = simulate_run(500.0, True, 50.0, stations, 15.0, 0.6, 0.9, 10.0, period=120.0)
    assert (len(run) - 1) * DT == 120.0 and run[-1] == pytest.approx(725.0)
    dwells = [s["depart"] - s["arrive"] for s in stops]
    assert min(dwells) > 10.0 and max(dwells) - min(dwells) <= DT
    with pytest.raises(ValueError, match="longer than the headway"):
        simulate_run(500.0, True, 50.0, stations, 15.0, 0.6, 0.9, 10.0, period=40.0)


def _line(*trains, length=500.0):
    from geogen.core.node import SceneNode

    root = SceneNode("scene")
    root.add_child(SceneNode("rail")).meta["railway"] = {"id": "loop", "length": length, "loop": True}
    run, _ = simulate_run(length, True, 50.0, [{"name": "a", "s": 100.0}, {"name": "b", "s": 350.0}],
                          15.0, 0.6, 0.9, 10.0, period=120.0)
    for name, offset in trains:
        node = root.add_child(SceneNode(name))
        node.meta.update(type="train", train={"railway": "loop", "length": 50.0, "dt": DT, "run": list(run),
                                               "duration": 120.0, "timetable": {"headway": 120.0, "offset": offset}})
    return root


def test_trains_on_a_loop_keep_their_distance():
    assert check_spacing(_line(("a", 0.0), ("b", 60.0))) == []          # half a lap apart
    (problem,) = check_spacing(_line(("a", 0.0), ("b", 8.0)))
    assert "railway 'loop'" in problem and "behind" in problem


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
