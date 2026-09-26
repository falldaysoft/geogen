"""Traffic lane graphs (geogen/traffic.py): city streets, routes, conflicts, clearance, export."""

import json
from pathlib import Path

import numpy as np
import pytest

from geogen.layout.composer import SceneComposer
from geogen.traffic import build_traffic, check_clearance, check_graph, route_lanes

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
