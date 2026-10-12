"""Traffic lane graphs (geogen/traffic.py): city streets, routes, conflicts, clearance, export."""

import json
from pathlib import Path

import numpy as np
import pytest

from geogen.layout.composer import SceneComposer
from geogen.traffic import (PEDESTRIAN_SPEED, build_traffic, check_clearance, check_graph, pedestrian_state,
                            route_lanes, signal_state)

ASSETS = Path(__file__).parent.parent / "assets"
SCHEMA = json.loads((Path(__file__).parent.parent / "docs/schema/geogen-traffic.v1.schema.json").read_text())

GRID = """
name: grid
city:
  seed: 1
  blocks: [2, 2]
  block_size: [30, 30]
  street_width: 8
  avenues: { ew: [1] }
  avenue_width: 12
  traffic: { drive: DRIVE, lanes: 1, avenue_lanes: 2 }
"""


def _grid(drive="right"):
    return SceneComposer(assets_dir=ASSETS).compose_string(GRID.replace("DRIVE", drive))


@pytest.fixture(scope="module")
def graph():
    return build_traffic(_grid())


def _heading(lane):
    p = np.asarray(lane["points"])
    d = p[-1] - p[0]
    return d / np.linalg.norm(d)


def test_graph_is_connected_and_valid(graph):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(graph, SCHEMA)
    assert check_graph(graph) == []
    kinds = {ln["kind"] for ln in graph["lanes"]}
    assert kinds == {"lane", "connector"}
    # 3 x 3 intersections; the centre one has four arms and all 12 movements (plus extra avenue lanes).
    centre = next(x for x in graph["intersections"] if x["id"] == "x_1_1")
    turns = {ln["turn"] for ln in graph["lanes"] if ln.get("intersection") == "x_1_1"}
    assert turns == {"straight", "left", "right"} and len(centre["connectors"]) >= 12


@pytest.mark.parametrize("drive", ["right", "left"])
def test_lanes_keep_to_the_drive_side(drive):
    g = build_traffic(_grid(drive))
    for ln in g["lanes"]:
        if ln["kind"] != "lane":
            continue
        h = _heading(ln)
        p = np.mean(ln["points"], axis=0)
        # Left of travel: heading +Z has left = +X (heading (hx, hz) -> left (hz, -hx)).
        left = np.array([h[2], 0.0, -h[0]])
        # The street's centre line passes through the mean of this lane and its opposite twin.
        twin = next(o for o in g["lanes"] if o["kind"] == "lane" and o["road"] == ln["road"]
                    and np.dot(_heading(o), h) < -0.9)
        centre = (p + np.mean(twin["points"], axis=0)) / 2
        side = np.dot(p - centre, left)
        assert (side < 0) if drive == "right" else (side > 0), ln["id"]


def test_turns_bend_the_right_way(graph):
    for ln in graph["lanes"]:
        if ln["kind"] != "connector":
            continue
        p = np.asarray(ln["points"])
        a, b = p[1] - p[0], p[-1] - p[-2]
        # Heading +Z (left = +X), a right turn ends heading -X: cross = ax*bz - az*bx = +1.
        cross = (a[0] * b[2] - a[2] * b[0]) / (np.linalg.norm(a) * np.linalg.norm(b))
        if ln["turn"] == "straight":
            assert abs(cross) < 0.2
        else:
            assert (cross > 0.2) if ln["turn"] == "right" else (cross < -0.2), ln["id"]


def test_turn_speeds_are_capped_by_curvature(graph):
    straight = [ln["speed"] for ln in graph["lanes"] if ln.get("turn") == "straight"]
    right = [ln["speed"] for ln in graph["lanes"] if ln.get("turn") == "right"]
    assert max(right) < min(straight) and min(right) > 2.0


def test_crossing_movements_conflict(graph):
    pairs = {(c["a"], c["b"]) for c in graph["conflicts"]}
    lanes = {ln["id"]: ln for ln in graph["lanes"]}
    lefts = [ln for ln in lanes.values() if ln.get("turn") == "left" and ln["intersection"] == "x_1_1"]
    assert lefts
    for ln in lefts:      # a left turn crosses oncoming traffic
        assert any(ln["id"] in pair for pair in pairs)
    for c in graph["conflicts"]:
        assert lanes[c["a"]]["intersection"] == lanes[c["b"]]["intersection"]
        assert 0 <= c["a_range"][0] <= c["a_range"][1] <= lanes[c["a"]]["length"] + 1e-6
    # Connectors cross the crosswalks on the way in and out.
    assert all(len(ln.get("crosswalks", [])) >= 1 for ln in lanes.values() if ln["kind"] == "connector")


def test_routes_loop_and_open():
    loop = route_lanes("ring", {"path": {"spline": [[0, 0, 0], [20, 0, 0], [20, 0, 20], [0, 0, 20]]},
                                "loop": True, "lanes": 2})
    assert len(loop) == 2 and all(ln.successors == [ln.id] for ln in loop)
    open_road = route_lanes("road", {"path": [[-30, 0, 5], [30, 0, 5]], "lanes": 2, "speed": 13})
    assert all(ln.end == "exit" for ln in open_road)
    p0, n0 = (np.mean(ln.points, axis=0) for ln in open_road)
    assert p0[2] > 5 > n0[2]         # heading +X, right-hand traffic keeps to +Z


def test_scene_routes_export_to_manifest(tmp_path):
    from geogen.export import export_scene, manifest_path

    root = SceneComposer(assets_dir=ASSETS).compose_string(
        "name: road_scene\n"
        "routes:\n"
        "  road: { path: [[-30, 0, 8], [30, 0, 8]], lanes: 2, speed: 13 }\n"
        "place:\n"
        "  car: { asset: vehicles/car.yaml }\n")
    path = export_scene(root, tmp_path / "road.glb")
    manifest = json.loads(manifest_path(path).read_text())
    traffic = manifest["traffic"]
    pytest.importorskip("jsonschema").validate(traffic, SCHEMA)
    assert {ln["id"] for ln in traffic["lanes"]} == {"road_p0", "road_n0"}
    assert check_graph(traffic) == []


SIGNALS = """
name: grid
city:
  seed: 1
  blocks: [2, 2]
  block_size: [30, 30]
  street_width: 8
  avenues: { ew: [1] }
  avenue_width: 12
  sidewalk: 3
  traffic: { lanes: 1, signals: all, cycle: CYCLE }
"""


def _signalled(cycle="{ green: 16 }"):
    root = SceneComposer(assets_dir=ASSETS).compose_string(SIGNALS.replace("CYCLE", cycle))
    return root, build_traffic(root)


def test_pedestrian_signals_walk_while_the_traffic_across_is_stopped():
    jsonschema = pytest.importorskip("jsonschema")
    root, graph = _signalled()
    jsonschema.validate(graph, SCHEMA)
    assert check_graph(graph) == []
    crossings = {p["id"]: p for p in graph["pedestrian_signals"]}
    xs = {x["id"]: x for x in graph["intersections"]}
    # Crossings with pavement at both ends: none over the streets round the district's edge.
    assert set(crossings) == {"x_0_1_east", "x_2_1_west", "x_1_0_north", "x_1_2_south",
                              "x_1_1_west", "x_1_1_east", "x_1_1_south", "x_1_1_north"}
    for p in crossings.values():
        x = xs[p["intersection"]]
        assert x["control"] == "signals" and p["crosswalk"] in graph["crosswalks"]
        x0, z0, x1, z1 = p["crosswalk"]
        along_x = (x1 - x0) > (z1 - z0)                       # the way people walk it
        assert p["length"] == pytest.approx(max(x1 - x0, z1 - z0))
        # The traffic that drives across it has the other phase; the traffic beside it, its own.
        for ln in graph["lanes"]:
            if ln.get("signal", {}).get("intersection") != p["intersection"]:
                continue
            h = _heading(ln)
            across = abs(h[2] if along_x else h[0]) > 0.9
            assert (ln["signal"]["phase"] != p["phase"]) == across, (p["id"], ln["id"])
        # Walk, then a clearance long enough to walk it all, ending with the phase's amber.
        cycle = x["cycle"]
        assert p["clearance"] * PEDESTRIAN_SPEED >= p["length"]
        assert p["walk"] >= 4.0 and p["walk"] + p["clearance"] == pytest.approx(cycle["green"] + cycle["amber"])
    assert (crossings["x_1_1_west"]["walk"], crossings["x_1_1_west"]["clearance"]) == (9.0, 10.0)    # 12 m avenue
    assert (crossings["x_1_1_north"]["walk"], crossings["x_1_1_north"]["clearance"]) == (12.0, 7.0)  # 8 m street
    # Through a whole cycle: walk comes with the green beside it and is over, clearance and all,
    # before the traffic across leaves red.
    for p in crossings.values():
        x = xs[p["intersection"]]
        seen = []
        for t in np.arange(0.0, 41.0, 0.25):
            state = pedestrian_state(p, x, t)
            if state != "dont":
                assert signal_state(x, 1 - p["phase"], t) == "red"
            if state == "walk":
                assert signal_state(x, p["phase"], t) == "green"
            if not seen or seen[-1] != state:
                seen.append(state)
        assert sorted(set(seen)) == ["dont", "flash", "walk"] and len(seen) <= 4
        assert "walk flash dont" in " ".join(seen + seen)


def test_pedestrian_heads_stand_at_both_ends_facing_across():
    root, graph = _signalled()
    heads = {}
    for n in root.iter_nodes():
        if n.meta.get("signal", {}).get("crossing"):
            heads.setdefault(n.meta["signal"]["crossing"], []).append(n)
    posts = [n.world_transform()[:3, 3] for n in root.iter_nodes() if "lane" in n.meta.get("signal", {})]
    assert posts and set(heads) == {p["id"] for p in graph["pedestrian_signals"]}
    for p in graph["pedestrian_signals"]:
        a, b = heads[p["id"]]
        assert a.meta["signal"] == {"intersection": p["intersection"], "phase": p["phase"], "crossing": p["id"]}
        x0, z0, x1, z1 = p["crosswalk"]
        pa, pb = a.world_transform()[:3, 3], b.world_transform()[:3, 3]
        for node, here, there in ((a, pa, pb), (b, pb, pa)):
            assert here[1] == pytest.approx(0.15)                                   # on the pavement
            assert not (x0 <= here[0] <= x1 and z0 <= here[2] <= z1)                # not in the road
            facing = node.world_transform()[:3, 2]                                  # lamps face +Z
            assert facing @ (there - here) / np.linalg.norm(there - here) > 0.99    # across, at the other end
            assert min(np.linalg.norm((here - q)[[0, 2]]) for q in posts) > 2.0     # clear of the vehicle heads
        assert np.linalg.norm(pa - pb) == pytest.approx(p["length"] + 0.6)


def test_a_cycle_too_short_to_cross_in_is_reported():
    _, graph = _signalled("{ green: 5 }")
    issues = check_graph(graph)
    # 12 m takes 10 s of clearance after at least 4 s of walk; a phase is 5 + 3 + 1.5 s.
    assert any(i.startswith("x_1_1_west: walk 4.0 s + clearance 10.0 s is longer than the phase") for i in issues)
    _, graph = _signalled("{ green: 12 }")
    assert check_graph(graph) == []


def test_town_lanes_are_clear(built_scene):
    root = built_scene("town")
    graph = build_traffic(root)
    assert graph is not None and check_graph(graph) == []
    assert check_clearance(root, graph) == []
    # Lanes stay clear of parked cars (they sit in the parking strip).
    from shapely.geometry import LineString, Point

    parked = [n.world_transform()[:3, 3] for n in root.iter_nodes() if "parked" in n.meta]
    assert parked
    for ln in graph["lanes"]:
        if ln["kind"] == "lane":
            line = LineString(np.asarray(ln["points"])[:, [0, 2]])
            assert min(line.distance(Point(p[0], p[2])) for p in parked) > 1.9, ln["id"]


def test_fleet_schedule_parses():
    from geogen.traffic import load_fleet

    fleet = load_fleet(ASSETS / "traffic" / "town.yaml")
    hours = [h for h, _ in fleet["schedule"]]
    assert hours == sorted(hours) and hours[0] == 0.0
    assert all(0.0 <= f <= 1.0 for _, f in fleet["schedule"])
