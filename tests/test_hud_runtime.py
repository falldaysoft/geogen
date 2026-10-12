"""The runtime HUD (runtime/godot/scripts/hud.gd) and its quest tracker (hud_quest_tracker.gd):
`hud.quests` returns what is on screen, which follows the quests' state through events."""

import time

import pytest

from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from geogen.runtime_client import ApiError
from test_state import GATE, LEVER
from test_wiring import PLATE

pytestmark = pytest.mark.xdist_group("test_hud_runtime")

# Three levers, a gate and a plate. `levers` starts with the game and counts the levers pulled;
# `errand` is given by using the first lever and fails when the alarm goes.
CAMP = """
name: camp
size: [12, 3, 12]
state:
  alarm: {scope: world, type: bool, default: false}
quests: [quests/levers.yaml, quests/errand.yaml]
slots:
  a: {position: [-3, 0, 0]}
  b: {position: [0, 0, 0]}
  c: {position: [3, 0, 0]}
  d: {position: [0, 0, -3]}
  e: {position: [3, 0, 3]}
place:
  lever:   {asset: lever.yaml, slot: a, tags: [lever]}
  lever_2: {asset: lever.yaml, slot: b, tags: [lever]}
  lever_3: {asset: lever.yaml, slot: c, tags: [lever]}
  gate:    {asset: gate.yaml, slot: d}
  plate:   {asset: plate.yaml, slot: e}
wire:
  - {from: lever_3.released, to: gate.open}
rules:
  - id: give_errand
    on: {use: lever, actor: player}
    do: [{start_quest: errand}]
spawns:
  start: {position: [0, 0, 5], facing: south}
"""

LEVERS = """
kind: quest
version: 1
title: Throw the switches
start: auto
objectives:
  - {id: all, text: Pull every lever, done: 'all("[tag=lever]", it.pull == "down")'}
  - {id: gate, text: Open the gate, after: all, done: 'gate.gate == "open"'}
  - {id: plate, text: Stand on the plate, optional: true, done: {enter: plate/top}}
"""

ERRAND = """
kind: quest
version: 1
title: An errand with a title long enough that it has to wrap in the tracker
fail: "world.alarm"
objectives:
  - {id: stand, text: Stand on the pressure plate in the far corner of the camp, done: {enter: plate/top}}
"""

LONG_TITLE = "An errand with a title long enough that it has to wrap in the tracker"


@pytest.fixture(scope="module")
def camp_dir(tmp_path_factory):
    assets = tmp_path_factory.mktemp("camp_assets")
    for name, text in (("lever", LEVER), ("gate", GATE), ("plate", PLATE), ("camp", CAMP),
                       ("quests/levers", LEVERS), ("quests/errand", ERRAND)):
        path = assets / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    out = tmp_path_factory.mktemp("generated_hud")
    export_scene(SceneComposer(assets_dir=assets).compose(assets / "camp.yaml"), out / "camp.glb")
    return out


@pytest.fixture(scope="module")
def rt(camp_dir, godot_runtime):
    client = godot_runtime("camp", camp_dir)
    client.call("wait_ready")
    return client


def shown(rt, want=lambda hud: True, timeout=10.0):
    """What the tracker displays, once it matches `want`: it redraws after the frame's events,
    so a change made by the request before may be one frame behind."""
    end = time.time() + timeout
    while True:
        hud = rt.call("hud.quests")
        if want(hud) or time.time() > end:
            return hud
        time.sleep(0.05)


def lines(hud):
    return [(line["id"], line["state"], line["progress_text"]) for line in hud["lines"]]


def test_the_tracker_follows_a_quest_from_start_to_complete(rt):
    hud = shown(rt, lambda h: h["visible"])
    assert hud["visible"] and hud["tracked"] == "levers" and hud["package"] == "camp"
    assert hud["title"] == "Throw the switches" and hud["status"] == "active" and hud["note"] == ""
    assert hud["active"] == ["levers"]
    # the gate waits on the levers, so it isn't listed yet; the optional one says so
    assert lines(hud) == [("all", "open", "0 of 3"), ("plate", "open", "")] and hud["waiting"] == 1
    assert hud["lines"][0]["text"] == "Pull every lever" and hud["lines"][0]["progress"] == [0, 3]
    assert hud["lines"][1]["text"] == "Stand on the plate (optional)" and hud["lines"][1]["optional"]
    # it shows what the console reports
    (listed, _) = rt.call("quests.list")
    assert listed["objectives"][0]["progress"] == [0, 3] and listed["objectives"][1]["state"] == "waiting"

    # the count moves with the world, with nothing asking the HUD to look
    rt.call("state.set", key="camp:lever_2#pull", value="down")
    assert lines(shown(rt, lambda h: h["lines"][0]["progress_text"] != "0 of 3"))[0] == ("all", "open", "1 of 3")
    rt.call("hud.quests", linger=1000)
    rt.call("state.set", key="camp:lever#pull", value="down")
    rt.call("state.set", key="camp:lever_3#pull", value="down")
    hud = shown(rt, lambda h: h["waiting"] == 0)
    assert lines(hud) == [("all", "done", "3 of 3"), ("gate", "open", ""), ("plate", "open", "")]

    # finished: it stays up marked complete, then goes
    rt.call("state.set", key="camp:gate#gate", value="open")
    hud = shown(rt, lambda h: h["status"] == "complete")
    assert hud["visible"] and hud["tracked"] == "levers" and hud["note"] == "Quest complete"
    assert lines(hud) == [("all", "done", "3 of 3"), ("gate", "done", "")] and hud["active"] == []
    hud = rt.call("hud.quests", linger=0)
    assert not hud["visible"] and hud["tracked"] == "" and hud["lines"] == [] and hud["rect"] == []


def test_a_key_cycles_the_tracked_quest(rt):
    for key in ("camp:lever#pull", "camp:lever_2#pull", "camp:lever_3#pull", "camp:gate#gate"):
        rt.call("state.set", key=key, value="closed" if key.endswith("gate") else "up")
    rt.call("state.set", key="actor:camp.quest_levers_all", value=False)
    rt.call("state.set", key="actor:camp.quest_levers_gate", value=False)
    rt.call("state.set", key="actor:camp.quest_levers", value="active")
    assert shown(rt, lambda h: h["visible"])["tracked"] == "levers"
    assert rt.call("input.key", key="Q") and shown(rt)["tracked"] == "levers"      # the only one: stays

    rt.call("interactions.set", asset="lever", state="up", wait=True)
    rt.call("player.use", asset="lever")                      # the scene's rule gives the errand
    hud = shown(rt, lambda h: len(h["active"]) == 2)
    assert hud["tracked"] == "levers" and hud["active"] == ["levers", "errand"]         # a new quest doesn't take over
    assert hud["note"] == "1 of 2    Q: next quest"

    rt.call("input.key", key="Q")
    hud = shown(rt, lambda h: h["tracked"] == "errand")
    assert hud["title"] == LONG_TITLE and hud["note"] == "2 of 2    Q: next quest"
    assert lines(hud) == [("stand", "open", "")]
    rt.call("input.key", key="Q")
    assert shown(rt, lambda h: h["tracked"] == "levers")["tracked"] == "levers"         # wraps
    assert rt.call("hud.quests", track="previous")["tracked"] == "errand"
    assert rt.call("hud.quests", track="next")["tracked"] == "levers"
    assert rt.call("hud.quests", track="errand")["tracked"] == "errand"
    with pytest.raises(ApiError, match="isn't an active quest"):
        rt.call("hud.quests", track="nope")


@pytest.mark.parametrize("size", [(1280, 720), (640, 360), (320, 200)])
def test_the_panel_stays_on_screen_and_wraps(rt, size):
    assert rt.call("window.size", size=list(size))["size"] == list(size)
    rt.call("hud.quests", track="errand")
    width, height = size
    hud = shown(rt, lambda h: h["viewport"] == [width, height] and h["rect"][0] + h["rect"][2] == width - 12)
    x, y, w, h = hud["rect"]
    assert x + w == width - 12 and y == 84                      # the top-right slot, under the debug overlay's lines
    assert 170 <= w <= 320
    assert x >= width / 2 + 60 or width < 640                   # clear of the prompt under the crosshair
    assert h > 80 and y + h <= height - 12                      # title and objective wrapped; all of it on screen
    assert hud["clipped"] == (height < 250)


def test_a_tall_tracker_is_clipped_at_the_bottom_of_a_small_window(rt):
    rt.call("window.size", size=[320, 160])
    hud = shown(rt, lambda h: h["viewport"] == [320, 160] and h["clipped"])
    assert hud["clipped"] and hud["rect"][1] + hud["rect"][3] <= 160 - 12 + 0.5
    rt.call("window.size", size=[1280, 720])


def test_the_tracker_survives_a_reload_and_shows_a_failure(rt):
    assert rt.call("hud.quests", track="errand", linger=1000)["tracked"] == "errand"
    rt.call("world.reload")
    hud = shown(rt, lambda h: h["visible"])
    assert hud["tracked"] == "errand" and hud["active"] == ["levers", "errand"]         # the pick is kept too
    saved = rt.call("state.save")

    rt.call("state.set", key="world:camp.alarm", value=True)
    hud = shown(rt, lambda h: h["status"] == "failed")
    assert hud["visible"] and hud["tracked"] == "errand" and hud["note"] == "Quest failed"
    assert rt.call("input.key", key="Q")                      # the key moves on from a finished quest at once
    hud = shown(rt, lambda h: h["tracked"] == "levers")
    assert hud["tracked"] == "levers" and hud["status"] == "active" and hud["active"] == ["levers"]

    # a restored save fires no events: the tracker re-reads
    rt.call("state.load", state=saved)
    assert shown(rt, lambda h: len(h["active"]) == 2)["active"] == ["levers", "errand"]
    rt.call("state.set", key="actor:camp.quest_levers", value="complete")
    rt.call("hud.quests", linger=0)
    assert shown(rt, lambda h: h["tracked"] == "errand")["tracked"] == "errand"         # the next active one
    rt.call("state.set", key="world:camp.alarm", value=True)
    hud = shown(rt, lambda h: not h["visible"])
    assert not hud["visible"] and hud["tracked"] == "" and hud["active"] == []          # nothing active: hidden


def test_another_actors_quests_are_not_shown(rt):
    rt.call("state.set", key="world:camp.alarm", value=False)
    rt.call("state.set", key="actor:camp.quest_errand", value="active", actor="npc:someone")
    assert rt.call("quests.list", actor="npc:someone")[1]["status"] == "active"
    time.sleep(0.2)
    assert not rt.call("hud.quests")["visible"]
