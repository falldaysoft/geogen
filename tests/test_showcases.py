"""Showcase scenes (assets/runtime_scenes.yaml, group showcase) in the Godot runtime."""

import json

import pytest

pytestmark = pytest.mark.exhaustive      # each showcase lived in for minutes: pytest --all


def _json_line(out: str, prefix: str):
    return json.loads(next(l for l in out.splitlines() if l.startswith(prefix)).removeprefix(prefix))


@pytest.fixture(scope="module")
def hotel_dir(scene_export):
    return scene_export("hotel_showcase")


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


def test_the_parcel_for_room_207_is_fetched_and_delivered(godot_runtime, hotel_dir):
    """geogen.15.7: greet the receptionist for the parcel and room 207's key, walk up to room
    207 on the navmesh, unlock its door with the carried key (L), go in and greet the guest."""
    # The stair halls' doors stand open (as a player would open them on the way up); 207 stays locked.
    rt = godot_runtime("hotel_showcase", hotel_dir, "--open-doors=door_*stair*")
    status = lambda: next(q for q in rt.call("quests.list") if q["id"] == "deliver_parcel")
    carried = lambda: {i["id"] for i in rt.call("inventory.list")["items"]}
    assert status()["status"] == "available"
    rt.call("player.teleport", pos=[0.0, 0.12, 2.8])                     # at the reception desk
    rt.call("npcs.greet", npc="receptionist")
    rt.call("time.advance", seconds=1)
    assert status()["status"] == "active" and carried() == {"parcel", "key_room_207"}
    rt.call("npcs.greet", npc="receptionist")                              # once: no second parcel
    rt.call("time.advance", seconds=1)
    assert rt.call("inventory.list")["items"][0]["count"] == 1

    def walk(pos, timeout):
        # As a player would: open a closed door the way is blocked by (inside a suite), go on.
        for _ in range(4):
            walked = rt.call("player.walk_to", pos=pos, timeout=timeout)
            door = (walked.get("blocked_by") or {}).get("interaction") or {}
            if walked["reached"] or door.get("state") != "closed":
                return walked
            rt.call("interactions.set", asset=door["asset"], interaction=door["name"], state="open", wait=True)
        return walked

    walked = walk([7.3, 6.45, 0.4], 240)                                   # the corridor outside 207
    assert walked["reached"], walked
    rt.call("player.look", at=[7.3, 7.4, -0.9])
    assert rt.call("hud.aim")["focus"]["prompt"] == "Unlock (L)"
    rt.call("input.key", key="L")
    rt.call("input.key", key="E")
    rt.call("time.advance", seconds=1.5)
    walked = walk([6.3, 6.45, -4.6], 30)                                  # into the room
    assert walked["reached"], walked
    rt.call("time.advance", seconds=1)
    assert [o["state"] for o in status()["objectives"]] == ["done", "open"]
    rt.call("npcs.greet", npc="guest_207")
    rt.call("time.advance", seconds=1)
    assert status()["status"] == "complete" and carried() == set()


def test_the_parcel_for_room_207_runs_on_the_reference_machine(built_scene):
    """geogen.15.7 on the evaluator: the same steps as in Godot, as events."""
    from geogen.quests import collect
    from geogen.rules import Machine

    found = collect(built_scene("hotel_showcase"), "hotel_showcase")
    machine = Machine(found.program(), {"player": {"kind": "player"}})
    status = lambda: machine.get("actor:hotel_showcase.quest_deliver_parcel", "player")
    held = lambda item: machine.get(f"actor:hotel_showcase.item_{item}", "player")
    greet = lambda npc: machine.post({"type": "signal", "subject": f"hotel_showcase:{npc}", "name": "greeted",
                                      "actor": "player"})
    greet("guest_207")                                                     # nothing to give yet
    assert status() == "available"
    greet("receptionist")
    greet("receptionist")
    assert status() == "active" and held("parcel") == 1 and held("key_room_207") == 1
    machine.post({"type": "enter", "subject": "hotel_showcase:hotel/storey_2/room_207", "actor": "player",
                  "data": {"room": True}})
    greet("guest_207")
    assert status() == "complete" and held("parcel") == 0 and held("key_room_207") == 0
    assert machine.errors == []


@pytest.fixture(scope="module")
def park_dir(scene_export):
    return scene_export("park")


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
def restaurant_dir(scene_export):
    return scene_export("restaurant")


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
def main_street_dir(scene_export):
    return scene_export("main_street")


def test_main_street_shops_are_furnished_by_kind(built_scene):
    from geogen.export import gameplay_summary

    kinds = {r["type"] for r in gameplay_summary(built_scene("main_street"))["rooms"]}
    assert {"cafe", "grocer", "clothing"} <= kinds, kinds


def test_main_street_lanes_are_clear_of_what_stands_by_them(built_scene):
    # The bus shelter once stood in the mouth of a side street.
    from geogen.traffic import build_traffic, check_clearance, check_graph

    scene = built_scene("main_street")
    graph = build_traffic(scene)
    assert check_graph(graph) == []
    assert check_clearance(scene, graph) == []


def test_main_street_traffic_and_pedestrians_keep_moving(run_godot, main_street_dir):
    out = run_godot("--scene=main_street", f"--generated={main_street_dir}", "--timescale=4", "--simulate=120")
    traffic = _json_line(out, "traffic summary: ")[0]
    assert traffic["vehicles"] >= 4 and traffic["distance"] > 500, traffic
    # "stuck" names whoever has stood longest when the run ends: a car at a stop line isn't stuck.
    assert traffic["overlaps"] == 0 and traffic["max_idle"] < 30 and traffic["max_wait"] < 75, traffic
    walkers = _json_line(out, "npc summary: ")
    assert len(walkers) == 10 and sum(w["decisions"] for w in walkers) >= 15


@pytest.fixture(scope="module")
def island_dir(scene_export):
    return scene_export("island")


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


def test_island_places_report_the_player_arriving(run_godot, island_dir):
    # Named places (geogen.places): walking out along the jetty, and standing on the lighthouse gallery.
    def events(*args):
        out = run_godot("--scene=island", f"--generated={island_dir}", *args, engine_args=("--fixed-fps", "60"))
        return [json.loads(l.removeprefix("place event: ")) for l in out.splitlines() if l.startswith("place event: ")]

    assert {"type": "enter", "kind": "place", "place": "on_deck", "subject": "island:jetty/on_deck",
            "actor": "player"} in events("--spawn=0,2.9,-44.5", "--yaw=0", "--walk=3")
    assert {"type": "enter", "kind": "place", "place": "gallery", "subject": "island:lighthouse/gallery",
            "actor": "player"} in events("--spawn=39.5,14.4,8.2", "--walk=0.5")


def test_island_residents_go_about_their_day(run_godot, island_dir):
    out = run_godot("--scene=island", f"--generated={island_dir}", "--timescale=8", "--simulate=180")
    npcs = _json_line(out, "npc summary: ")
    assert len(npcs) == 2 and all(n["used"] and not n["failures"] for n in npcs), npcs
