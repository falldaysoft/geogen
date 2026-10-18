"""The hub (scenes/hub.yaml): a portal to every showcase scene, and a way back from each."""

import json
import math

import pytest

from geogen.catalogue import export_catalogue, load_catalogue
from geogen.export import gameplay_summary
from geogen.runtime_client import step


def _spawns(summary) -> set[str]:
    return {s["name"] for s in summary["spawns"]}


def test_hub_is_the_default_showcase():
    catalogue = load_catalogue()
    assert catalogue.default == "hub" and catalogue.entries["hub"].group == "showcase"


def test_hub_reaches_every_showcase_scene_and_back(built_scene):
    catalogue = load_catalogue()
    hub = gameplay_summary(built_scene("hub"))
    showcases = [e.name for e in catalogue.select("showcase") if e.name != "hub"]
    targets = {t["scene"]: t for t in hub["travel"]}
    assert set(showcases) <= set(targets), f"no hub portal to {set(showcases) - set(targets)}"
    for name in showcases:
        scene = gameplay_summary(built_scene(name))
        # The hub's portal lands on a spawn the scene has...
        assert targets[name]["spawn"] in _spawns(scene), (name, targets[name])
        # ...and the scene has a way back to a hub spawn.
        back = [t for t in scene.get("travel", []) if t.get("scene") == "hub"]
        assert back, f"{name} has no travel point back to the hub"
        assert back[0]["spawn"] in _spawns(hub), (name, back[0])


def test_cottage_inside_other_scenes_has_no_hub_portal(built_scene):
    town = built_scene("town")
    hub_points = [t for t in gameplay_summary(town)["travel"] if t.get("scene") == "hub"]
    assert len(hub_points) == 1            # the town's own park portal, not one per cottage lot


@pytest.fixture(scope="module")
def hub_dir(tmp_path_factory, built_scene):
    """The hub and the cottage exported; the hub's other targets are in the catalogue but not exported."""
    out = tmp_path_factory.mktemp("generated")
    catalogue = load_catalogue()
    export_catalogue(catalogue, built_scene, out, catalogue.select("hub,cottage"), log=lambda *_: None)
    return out


def _spawn(generated, scene, name) -> dict:
    manifest = json.loads((generated / f"{scene}.manifest.json").read_text())
    return next(s for s in manifest["spawns"] if s["name"] == name)


def _walk_outward(spawn: dict) -> dict:
    """A start (spawn, yaw) to walk from a hub arrival spawn back out through its portal (away from the centre)."""
    x, y, z = spawn["position"]
    yaw = math.degrees(math.atan2(-x, -z))   # the player looks along (-sin yaw, -cos yaw)
    return {"spawn": [x, y, z], "yaw": yaw}


TRAVEL = step("events.subscribe", streams=["travel"])


def _travelled(run) -> list[dict]:
    return [e["data"] for e in run.events if e["event"] == "travel"]


def test_walk_from_the_hub_to_the_cottage_and_back(godot_script, hub_dir):
    # From in front of the cottage portal, turn round (the arrival spawn faces the plaza) and walk in.
    run = godot_script([TRAVEL, step("player.move", seconds=3)], f"--generated={hub_dir}", "--scene=hub",
                       start=_walk_outward(_spawn(hub_dir, "hub", "from_cottage")))
    travelled = _travelled(run)
    assert len(travelled) == 1 and travelled[0]["scene"] == "cottage" and travelled[0]["spawn"] == "from_hub"
    arrival = _spawn(hub_dir, "cottage", "from_hub")["position"]
    result = run["player.move"]
    assert result["scene"] == "cottage" and (result["position"][0], result["position"][2]) == pytest.approx(
        (arrival[0], arrival[2]), abs=0.05)

    # And back: the cottage's hub portal is on the west lawn, behind its arrival spawn.
    run = godot_script([TRAVEL, step("player.move", seconds=3)], f"--generated={hub_dir}", "--scene=cottage",
                       start={"spawn": [-5.4, 0, 3.5], "yaw": 90})
    travelled = _travelled(run)
    assert len(travelled) == 1 and travelled[0]["scene"] == "hub" and travelled[0]["spawn"] == "from_cottage"
    arrival = _spawn(hub_dir, "hub", "from_cottage")["position"]
    result = run["player.move"]
    assert result["scene"] == "hub" and (result["position"][0], result["position"][2]) == pytest.approx(
        (arrival[0], arrival[2]), abs=0.05)


def test_a_scene_comes_back_as_it_was_left(godot_script, hub_dir):
    # The cottage's door, opened, is still open after a trip to the hub and back (geogen.70.1).
    run = godot_script([
        step("interactions.set", asset="door", state="open", instant=True),
        step("world.switch", scene="hub"),
        step("world.switch", scene="cottage"),
        step("interactions.list", asset="door"),
    ], f"--generated={hub_dir}", "--scene=cottage")
    (door,) = run["interactions.list"]
    assert door["state"] == "open" and door["target"] == "open"


@pytest.mark.exhaustive
def test_an_island_quest_is_marked_at_the_hub_portal_to_the_island(godot_runtime, built_scene, tmp_path_factory):
    """A quest's subject in another scene is marked at the portal that leads there: with the island's
    lighthouse quest going, the hub's island portal carries its marker (and no other portal does)."""
    out = tmp_path_factory.mktemp("generated_hub_island")
    catalogue = load_catalogue()
    export_catalogue(catalogue, built_scene, out, catalogue.select("hub,island"), log=lambda *_: None)
    rt = godot_runtime("island", out)
    rt.call("wait_ready")
    rt.call("window.size", size=[1280, 720])
    (here,) = rt.call("hud.markers")["markers"]
    assert here["kind"] == "point" and here["subject"] == "island:lighthouse/gallery"

    rt.call("world.switch", scene="hub")
    rt.call("wait_ready")
    (portal,) = rt.call("hud.markers")["markers"]
    assert portal["kind"] == "portal" and portal["scene"] == "island" and portal["label"] == "Island"
    assert (portal["quest"], portal["objective"], portal["package"]) == ("lighthouse_gallery", "gallery", "island")
    assert portal["subject"] == "hub:portal_island"
    manifest = json.loads((out / "hub.manifest.json").read_text())
    (point,) = [t for t in manifest["travel"] if t.get("scene") == "island"]
    assert point["uid"] == portal["subject"]
    # facing it, the marker is in the portal's opening; turned away, it is on the screen's edge
    rt.call("player.look", at=portal["position"])
    (portal,) = rt.call("hud.markers")["markers"]
    assert portal["on_screen"] and not portal["clamped"] and portal["screen"] == pytest.approx([640, 360], abs=2)
    yaw = rt.call("player.status")["yaw"]
    rt.call("player.look", yaw=yaw + 180, pitch=0)
    (portal,) = rt.call("hud.markers")["markers"]
    assert portal["behind"] and portal["clamped"] and not portal["on_screen"]

    # through the portal's scene and back on the island, the gallery itself is marked again
    rt.call("world.switch", scene="island")
    rt.call("wait_ready")
    (here,) = rt.call("hud.markers")["markers"]
    assert here["kind"] == "point" and here["subject"] == "island:lighthouse/gallery"


def _middle_of(rt, uid):
    box = rt.call("world.node", uid=uid)["bounds"]
    return [(box["min"][0] + box["max"][0]) / 2, box["min"][1] + 0.05, (box["min"][2] + box["max"][2]) / 2]


def _until(call, want, timeout=20.0):
    import time

    end = time.time() + timeout
    while not want(got := call()) and time.time() < end:
        time.sleep(0.1)
    return got


@pytest.mark.exhaustive
def test_the_jetty_trip_goes_from_the_hub_to_the_jetty_and_back(godot_runtime, built_scene, tmp_path_factory):
    """The content package jetty_trip (geogen.15.5): travelling to the island gives its quest, the
    jetty is the first step, coming back through the hub portal the last; its state carries over."""
    out = tmp_path_factory.mktemp("generated_jetty_trip")
    catalogue = load_catalogue()
    export_catalogue(catalogue, built_scene, out, catalogue.select("hub,island"), log=lambda *_: None)
    rt = godot_runtime("hub", out)
    rt.call("wait_ready")

    def trip():
        (quest,) = [q for q in rt.call("quests.list") if q["id"] == "jetty_trip"]
        return quest["status"], [(o["id"], o["state"]) for o in quest["objectives"]]

    def scene():
        return rt.call("world.info")["scene"]

    assert trip()[0] == "available"
    rt.call("player.teleport", pos=_middle_of(rt, "hub:portal_island"))          # into the portal: travel
    _until(scene, lambda s: s == "island")
    rt.call("wait_ready")
    assert _until(trip, lambda t: t[0] == "active") == ("active", [("jetty", "open"), ("home", "waiting")])
    assert rt.call("hud.quests", track="jetty_trip")["package"] == "jetty_trip"
    assert [m["subject"] for m in rt.call("hud.markers")["markers"]] == ["island:jetty/on_deck"]

    rt.call("world.switch", scene="hub")                                          # back early, not by travel
    rt.call("wait_ready")
    portals = _until(lambda: rt.call("hud.markers")["markers"], lambda m: m)
    assert [(m["kind"], m["subject"], m["quest"]) for m in portals] == [("portal", "hub:portal_island", "jetty_trip")]
    rt.call("world.switch", scene="island")
    rt.call("wait_ready")

    x, y, z = rt.call("world.node", uid="island:jetty/on_deck")["position"]
    rt.call("player.teleport", pos=[x, y + 0.1, z])
    assert _until(trip, lambda t: t[1][0][1] == "done")[1] == [("jetty", "done"), ("home", "open")]
    marker = _until(lambda: rt.call("hud.markers")["markers"], lambda m: m and m[0]["subject"] == "island:hub_portal")
    assert [m["subject"] for m in marker] == ["island:hub_portal"]
    rt.call("player.teleport", pos=_middle_of(rt, "island:hub_portal"))
    _until(scene, lambda s: s == "hub")
    rt.call("wait_ready")
    assert _until(trip, lambda t: t[0] == "complete") == ("complete", [("jetty", "done"), ("home", "done")])
