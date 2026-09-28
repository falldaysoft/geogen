"""Showcase scenes (assets/runtime_scenes.yaml, group showcase) in the Godot runtime."""

import json

import pytest

from geogen.export import export_scene


def _json_line(out: str, prefix: str):
    return json.loads(next(l for l in out.splitlines() if l.startswith(prefix)).removeprefix(prefix))


@pytest.fixture(scope="module")
def hotel_dir(tmp_path_factory, built_scene):
    out = tmp_path_factory.mktemp("generated")
    export_scene(built_scene("hotel_showcase"), out / "hotel_showcase.glb")
    return out


def test_hotel_showcase_playtest_from_the_street(run_godot, hotel_dir):
    out = run_godot("--scene=hotel_showcase", f"--generated={hotel_dir}", "--playtest=2",
                    engine_args=("--fixed-fps", "60"))
    result = _json_line(out, "playtest: ")
    assert result["ok"] and not result["unreachable"] and not result["unreachable_targets"], result


def test_hotel_showcase_staff_and_guests(run_godot, hotel_dir):
    out = run_godot("--scene=hotel_showcase", f"--generated={hotel_dir}", "--timescale=8", "--simulate=240")
    npcs = {n["npc"]: n for n in _json_line(out, "npc summary: ")}
    assert any(k.startswith("desk") for k in npcs["receptionist_npc"]["used"]), npcs["receptionist_npc"]
    # Guests keep to their own storey (floors at 3.65 and 6.4) and never leave their rooms' affordances.
    for name, floor in (("guest_101_npc", 3.65), ("guest_207_npc", 6.4)):
        guest = npcs[name]
        assert abs(guest["position"][1] - floor) < 1.2, guest
        assert guest["used"], guest
    pedestrians = [n for k, n in npcs.items() if k.startswith("pedestrians")]
    assert len(pedestrians) == 4 and all(p["decisions"] > 2 for p in pedestrians)


@pytest.fixture(scope="module")
def park_dir(tmp_path_factory, built_scene):
    out = tmp_path_factory.mktemp("generated")
    export_scene(built_scene("park"), out / "park.glb")
    return out


def test_park_visitors_keep_to_the_paths_and_sit_down(run_godot, park_dir):
    out = run_godot("--scene=park", f"--generated={park_dir}", "--timescale=8", "--simulate=300", "--time=16:00")
    visitors = _json_line(out, "npc summary: ")
    assert len(visitors) == 8
    used = {k.split("#")[0].rstrip("_0123456789") for v in visitors for k in v["used"]}
    assert {"bench", "picnic"} <= used, used
    for v in visitors:
        assert v["outside"] == 0.0, v                 # never left the railed park
        x, _, z = v["position"]
        assert abs(x) < 35 and abs(z) < 28, v
        assert len(v["failures"]) <= 2, v["failures"]


@pytest.fixture(scope="module")
def restaurant_dir(tmp_path_factory, built_scene):
    out = tmp_path_factory.mktemp("generated")
    export_scene(built_scene("restaurant"), out / "restaurant.glb")
    return out


def test_restaurant_playtest_and_service(run_godot, restaurant_dir):
    out = run_godot("--scene=restaurant", f"--generated={restaurant_dir}", "--playtest=2",
                    engine_args=("--fixed-fps", "60"))
    result = _json_line(out, "playtest: ")
    assert result["ok"] and not result["unreachable"] and not result["unreachable_targets"], result
    out = run_godot("--scene=restaurant", f"--generated={restaurant_dir}", "--timescale=8", "--simulate=240",
                    "--time=19:00")
    npcs = {n["npc"]: n for n in _json_line(out, "npc summary: ")}
    works = {"host_npc": "host_stand", "bartender_npc": "bar_counter", "chef_npc": ("range", "prep")}
    for npc, station in works.items():
        assert any(k.startswith(station) for k in npcs[npc]["used"]), npcs[npc]
    diners = [n for k, n in npcs.items() if k.startswith("diner")]
    seats = {k.split("#")[0].rstrip("_0123456789") for d in diners for k in d["used"]}
    assert {"booth", "stool"} <= seats and ("chair" in seats), seats
    assert sum(len(n["failures"]) for n in npcs.values()) <= 3


@pytest.fixture(scope="module")
def main_street_dir(tmp_path_factory, built_scene):
    out = tmp_path_factory.mktemp("generated")
    export_scene(built_scene("main_street"), out / "main_street.glb")
    return out


def test_main_street_shops_are_furnished_by_kind(built_scene):
    from geogen.export import gameplay_summary

    kinds = {r["type"] for r in gameplay_summary(built_scene("main_street"))["rooms"]}
    assert {"cafe", "grocer", "clothing"} <= kinds, kinds


def test_main_street_traffic_and_pedestrians_keep_moving(run_godot, main_street_dir):
    out = run_godot("--scene=main_street", f"--generated={main_street_dir}", "--timescale=4", "--simulate=120")
    traffic = _json_line(out, "traffic summary: ")[0]
    assert traffic["vehicles"] >= 4 and traffic["distance"] > 500, traffic
    assert traffic["overlaps"] == 0 and not traffic["stuck"], traffic
    walkers = _json_line(out, "npc summary: ")
    assert len(walkers) == 10 and sum(w["decisions"] for w in walkers) >= 15


@pytest.fixture(scope="module")
def island_dir(tmp_path_factory, built_scene):
    out = tmp_path_factory.mktemp("generated")
    export_scene(built_scene("island"), out / "island.glb")
    return out


def test_island_walk_from_the_jetty_to_the_lighthouse_gallery(run_godot, island_dir):
    out = run_godot("--scene=island", f"--generated={island_dir}", "--playtest=1", engine_args=("--fixed-fps", "60"))
    result = _json_line(out, "playtest: ")
    assert result["ok"] and not result["unreachable_targets"], result
    # In legs: one query across the whole island hits Godot's path search limit on a 5 cm navmesh.
    legs = [((0, 2.8, -64), (35.8, 4.3, 7)),            # the jetty's far end -> the lighthouse door
            ((35.8, 4.3, 7), (42.3, 14.35, 7)),          # -> up the spiral stair, round the gallery
            ((0, 2.8, -64), (-28, 7.9, -12.5))]          # the jetty -> the west cottage's door
    for a, b in legs:
        out = run_godot("--scene=island", f"--generated={island_dir}", "--nav=%s:%s" % (
            ",".join(map(str, a)), ",".join(map(str, b))))
        path = _json_line(out, "nav path: ")
        assert path["reached"] and path["end_gap"] < 0.3, (a, b, path)


def test_island_residents_go_about_their_day(run_godot, island_dir):
    out = run_godot("--scene=island", f"--generated={island_dir}", "--timescale=8", "--simulate=180")
    npcs = _json_line(out, "npc summary: ")
    assert len(npcs) == 2 and all(n["used"] and not n["failures"] for n in npcs), npcs
