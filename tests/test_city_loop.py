"""City exits and a railway looping round the district (scenes/city.yaml, scenes/rail_loop_test.yaml)."""

import json

import numpy as np
import pytest

from geogen.export import export_scene
from geogen.railway import loop_around
from geogen.traffic import build_traffic, check_clearance, check_graph


def test_loop_around_is_a_closed_rounded_rectangle():
    pts = np.array(loop_around([100, 60], margin=10, corner=15))
    assert np.allclose(pts[:, 1], 0)
    assert pts[:, 0].max() == pytest.approx(60, abs=0.01) and pts[:, 0].min() == pytest.approx(-60, abs=0.01)
    assert pts[:, 2].max() == pytest.approx(40, abs=0.01) and pts[:, 2].min() == pytest.approx(-40, abs=0.01)
    assert np.linalg.norm(pts[0] - pts[-1]) < 3.0          # consecutive points close the loop
    assert pts[0, 0] == pytest.approx(45, abs=0.01) and pts[0, 2] == pytest.approx(-40, abs=0.01)   # SE corner


@pytest.fixture(scope="module")
def loop_scene(built_scene):
    return built_scene("rail_loop_test")


def test_avenues_run_out_of_town_as_exits(loop_scene):
    graph = build_traffic(loop_scene)
    assert check_graph(graph) == []
    exits = [ln for ln in graph["lanes"] if ln.get("end") == "exit"]
    entries = [ln for ln in graph["lanes"] if ln.get("start") == "entry"]
    assert len(exits) == 4 and len(entries) == 4          # one each way on each of four exit roads
    assert check_clearance(loop_scene, graph) == []


def test_loop_railway_crossings_stations_and_trains(loop_scene):
    rail = build_traffic(loop_scene)["railways"][0]
    assert rail["loop"] and len(rail["crossings"]) == 4
    assert all(len(x["barriers"]) == 2 for x in rail["crossings"])   # one kerb barrier per direction
    trains = [n for n in loop_scene.iter_nodes() if n.meta.get("type") == "train"]
    assert len(trains) == 2
    for t in trains:
        assert [s["station"] for s in t.meta["train"]["stops"]] == ["east", "west"]
    # The two trains stand apart in the export (they'd z-fight on top of each other).
    heads = [t.children[0].world_transform()[:3, 3] for t in trains]
    assert np.linalg.norm(heads[0] - heads[1]) > 50


def test_city_stations_stand_clear_of_the_crossings(built_scene):
    city = built_scene("city")
    graph = build_traffic(city)
    rail = graph["railways"][0]
    for station in rail["stations"]:
        lo, hi = station["s"] - station["length"] / 2, station["s"] + station["length"] / 2
        for x in rail["crossings"]:
            assert hi < x["s"][0] or lo > x["s"][1], (station, x)
    assert check_clearance(city, graph) == []


@pytest.fixture(scope="module")
def loop_dir(tmp_path_factory, loop_scene):
    out = tmp_path_factory.mktemp("generated")
    export_scene(loop_scene, out / "rail_loop_test.glb")
    return out


def _json_line(out: str, prefix: str):
    return json.loads(next(l for l in out.splitlines() if l.startswith(prefix)).removeprefix(prefix))


def test_trains_and_traffic_share_the_crossings(run_godot, loop_dir):
    out = run_godot("--scene=rail_loop_test", f"--generated={loop_dir}", "--spawn=0,0,90",
                    "--timescale=8", "--simulate=300")
    trains = _json_line(out, "train summary: ")
    assert len(trains) == 2 and all(t["departures"] >= 1 and t["closed_time"] > 0 for t in trains), trains
    traffic = _json_line(out, "traffic summary: ")[0]
    assert traffic["overlaps"] == 0 and traffic["exits"] > 0 and traffic["distance"] > 800, traffic
    stuck = traffic.get("stuck") or {}
    assert not stuck or stuck.get("blocked_by") in ("signal", "crossing", "vehicle"), stuck
