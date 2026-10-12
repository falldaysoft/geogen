"""The Godot runtime's event bus (runtime/godot/scripts/events.gd): one stream for interaction
use, state changes, signals, arrivals, places, travel, the clock, NPCs and a new game."""

import pytest

from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from test_places import PLATE

pytestmark = pytest.mark.xdist_group("test_events_runtime")

# A lever whose arrival raises `pulled`, and which declares a signal with a source.
LEVER = """
name: lever
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
parts:
  base:   {primitive: cube, size: [0.3, 0.1, 0.3], anchor: bottom_center}
  handle: {primitive: cube, size: [0.05, 0.6, 0.05], anchor: bottom_center, offset: [0, 0.1, 0]}
interactions:
  pull:
    targets: [handle]
    duration: 0.3
    states:
      up:   {next: down}
      down: {next: up, emit: pulled}
    motions:
      - {parts: [handle], rotate: x, pivot: [0, 0.1, 0], values: {down: 60}}
signals:
  pulled: {emit: pulled}
  moved:  {changed: pull}
"""

YARD = """
name: yard
size: [16, 3, 16]
spawns:
  start: {position: [0, 0, 7], facing: south}
slots:
  lever: {position: [-2, 0, 5]}
  plate: {position: [3, 0, 5]}
  door:  {position: [6, 0, 6], facing: west}
  stand: {position: [-3, 0, -2], facing: north}
place:
  lever: {asset: LEVER_FILE, slot: lever}
  plate: {asset: PLATE_FILE, slot: plate}
  way:   {asset: travel_door.yaml, slot: door, travel: {spawn: start}}
  walker:
    npc: npcs/resident.yaml
    slot: stand
    home: {rect: [-6, -6, 6, 2], y: 0}
    seed: 3
places:
  north_half: {box: [16, 2.2, 9], position: [0, 0, -3.5]}
"""



@pytest.fixture(scope="module")
def rt(tmp_path_factory, godot_runtime):
    out = tmp_path_factory.mktemp("generated_events")
    (out / "lever.yaml").write_text(LEVER)
    (out / "plate.yaml").write_text(PLATE)
    yard = YARD.replace("LEVER_FILE", str(out / "lever.yaml")).replace("PLATE_FILE", str(out / "plate.yaml"))
    export_scene(SceneComposer().compose_string(yard), out / "yard.glb")
    client = godot_runtime("yard", out)
    client.call("wait_ready")
    return client


def collect(rt, until, timeout=20):
    """The bus events up to and including the first that ``until`` accepts."""
    seen = []

    def take(message):
        if message["event"] == "events":
            seen.append(message["data"])
            return until(message["data"])
        return False

    rt.wait_event(take, timeout=timeout)
    return seen


def brief(events):
    return [(e["type"], e["subject"], e["name"], e["actor"]) for e in events]


def test_a_new_game_starts_for_the_player(rt):
    history = rt.call("events.history", types=["start"])
    assert brief(history["events"]) == [("start", "", "", "player")]
    assert history["events"][0]["package"] == "yard" and history["count"] >= 1


def test_using_an_interaction_is_use_change_arrival_and_signals_in_order(rt):
    rt.subscribe("events", subjects=["yard:lever"])
    assert rt.call("player.use", asset="lever")["used"][0]["ok"]
    events = collect(rt, lambda e: e["type"] == "signal" and e["name"] == "pulled")
    rt.unsubscribe()
    assert brief(events) == [
        ("use", "yard:lever", "pull", "player"),
        ("changed", "yard:lever", "pull", "player"),
        ("signal", "yard:lever", "moved", "player"),        # declared: {changed: pull}
        ("arrived", "yard:lever", "pull", "player"),
        ("signal", "yard:lever", "pulled", "player"),       # the state's emit: whoever pulled it is the actor
    ]
    changed, arrived = events[1], events[3]
    assert changed["key"] == "yard:lever#pull" and changed["data"] == {"from": "up", "to": "down"}
    assert arrived["data"] == {"state": "down", "emit": "pulled", "asset": "lever"}
    assert all(e["package"] == "yard" for e in events)

    # with nobody behind it, the same move has no actor
    rt.subscribe("events", types=["use", "changed", "signal"], subjects=["yard:lever"])
    rt.call("interactions.use", asset="lever")
    events = collect(rt, lambda e: e["name"] == "moved")
    rt.unsubscribe()
    assert brief(events) == [("use", "yard:lever", "pull", ""), ("changed", "yard:lever", "pull", ""),
                             ("signal", "yard:lever", "moved", "")]


def test_listeners_filter_by_type_and_subject(rt):
    rt.call("interactions.set", asset="lever", state="up", instant=True)      # silent: nothing happened
    before = rt.call("events.history")["count"]
    rt.subscribe("events", types=["changed"])
    rt.call("interactions.use", asset="lever", actor="yard:walker")
    rt.call("interactions.use", asset="way")              # the door: another subject, filtered below
    rt.call("time.step", frames=40)
    rt.call("time.resume")
    got = [m["data"] for m in rt.clear_events() if m["event"] == "events"]
    rt.unsubscribe()
    history = rt.call("events.history", since=before)["events"]
    assert got == [e for e in history if e["type"] == "changed"] and len(history) > len(got)
    assert ("changed", "yard:lever", "pull", "yard:walker") in brief(got)
    only = rt.call("events.history", since=before, types=["changed"], subjects=["yard:lever"])["events"]
    assert brief(only) == [("changed", "yard:lever", "pull", "yard:walker")]
    assert rt.call("events.history", since=rt.call("events.history")["count"])["events"] == []


def test_places_report_who_came_and_asset_signals_follow(rt):
    rt.call("player.teleport", pos=[0, 0, 7], yaw=0)
    rt.subscribe("events", types=["enter", "leave", "signal"], subjects=["yard:plate/top", "yard:plate"])
    rt.call("player.teleport", pos=[3, 0.1, 5], yaw=0)
    entered = collect(rt, lambda e: e["name"] == "occupied")
    rt.call("player.teleport", pos=[0, 0, 7], yaw=0)
    left = collect(rt, lambda e: e["name"] == "vacated")
    rt.unsubscribe()
    assert brief(entered) == [("enter", "yard:plate/top", "", "player"), ("signal", "yard:plate", "occupied", "player")]
    assert brief(left) == [("leave", "yard:plate/top", "", "player"), ("signal", "yard:plate", "vacated", "player")]
    assert entered[0]["data"] == {"place": "top", "kind": "place"}


def test_npcs_are_actors_and_the_clock_has_none(rt):
    rt.subscribe("events", types=["npc", "enter", "clock"])
    rt.call("time.clock", time="23:00")
    night = collect(rt, lambda e: e["type"] == "clock" and e["name"] == "night")[-1]     # days pass quickly headless
    assert (night["name"], night["actor"], night["subject"]) == ("night", "", "")
    rt.call("time.scale", scale=8)
    events = collect(rt, lambda e: e["type"] == "enter" and e["actor"] == "yard:walker", timeout=90)
    rt.call("time.scale", scale=1)
    rt.unsubscribe()
    assert events[-1]["subject"] == "yard:north_half"
    traces = [e for e in events if e["type"] == "npc"]
    assert traces and all(e["actor"] == e["subject"] == "yard:walker" for e in traces)
    assert {"decide", "step"} & {e["name"] for e in traces}
    # the older streams are views of the bus
    rt.subscribe("npc", "place")
    rt.call("player.teleport", pos=[3, 0.1, 5], yaw=0)
    place = rt.wait_event(lambda m: m["event"] == "place")["data"]
    rt.unsubscribe()
    assert place == {"type": "enter", "subject": "yard:plate/top", "place": "top", "kind": "place", "actor": "player"}


def test_travel_is_an_event(rt):
    rt.call("player.teleport", pos=[0, 0, 0], yaw=0)
    rt.subscribe("events", types=["travel"])
    rt.call("player.use", asset="way")
    (travel,) = collect(rt, lambda e: True)
    rt.unsubscribe()
    assert brief([travel]) == [("travel", "yard:way", "yard", "player")]
    assert travel["data"] == {"spawn": "start", "on": "use"}
