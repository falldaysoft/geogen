"""The runtime HUD (runtime/godot/scripts/hud.gd), its quest tracker (hud_quest_tracker.gd:
`hud.quests` returns what is on screen, which follows the quests' state through events) and
its quest toasts (hud_notifications.gd: `hud.notifications` returns the ones up and waiting)."""

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


def test_the_top_slots_make_room_for_the_overlay_only_while_it_is_up(rt):
    rt.call("window.size", size=[1280, 720])
    rt.call("hud.quests", track="errand")
    assert shown(rt, lambda h: h["visible"] and h["rect"][1] == 84)["rect"][1] == 84
    try:
        rt.call("input.key", key="F1")                          # the debug overlay off
        assert shown(rt, lambda h: h["rect"][1] == 12)["rect"][1] == 12
    finally:
        rt.call("input.key", key="F1")
    assert shown(rt, lambda h: h["rect"][1] == 84)["rect"][1] == 84


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
    assert lines(hud) == [("stand", "missed", "")]           # what it failed with open, crossed out
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


# -- notifications (hud_notifications.gd) -----------------------------------------------------------

@pytest.fixture(scope="module")
def fresh(camp_dir, godot_runtime):
    """A camp nothing has happened in yet, for the toasts."""
    client = godot_runtime("camp", camp_dir)
    client.call("wait_ready")
    return client


def toasts(rt, want=lambda hud: True, timeout=10.0, **args):
    """What `hud.notifications` shows and holds back, once it matches `want` (toasts appear a frame
    or more after what raised them)."""
    end = time.time() + timeout
    hud = rt.call("hud.notifications", **args)
    while not want(hud) and time.time() < end:
        time.sleep(0.05)
        hud = rt.call("hud.notifications")
    return hud


def kinds(notes):
    return [(note["kind"], note.get("objective", note["quest"])) for note in notes]


def test_toasts_that_arrive_together_queue_in_order(fresh):
    rt = fresh
    start = rt.call("hud.notifications")
    assert start["raised"] == 1                               # `levers` started with the game, and said so
    assert (start["hold"], start["limit"]) == (4.0, 3)
    hud = toasts(rt, hold=1000, gap=1000, limit=2, clear=True)
    assert not hud["visible"] and hud["showing"] == [] and hud["queued"] == [] and hud["rect"] == []

    for lever in ("lever", "lever_2", "lever_3"):
        rt.call("state.set", key=f"camp:{lever}#pull", value="down")
    rt.call("state.set", key="camp:gate#gate", value="open")
    hud = toasts(rt, lambda h: len(h["showing"]) == 1 and len(h["queued"]) == 1)
    # the first is up at once; the others wait their turn, in the order they happened. The gate
    # finished the quest: its toast folds into the quest's (one toast, one chime)
    assert kinds(hud["showing"]) == [("objective_done", "all")]
    assert kinds(hud["queued"]) == [("complete", "levers")] and hud["queued"][0]["done"] == "Open the gate"
    assert [n["seq"] for n in hud["showing"] + hud["queued"]] == [2, 4] and hud["raised"] == 4
    first = hud["showing"][0]
    assert first["heading"] == "Objective complete" and first["text"] == "Pull every lever"
    assert first["package"] == "camp" and first["quest"] == "levers"
    assert hud["chimes"] == start["chimes"] + 1               # a chime as each appears, not as it is raised

    hud = toasts(rt, lambda h: len(h["showing"]) == 2, gap=0)
    assert kinds(hud["showing"]) == [("objective_done", "all"), ("complete", "levers")] and hud["queued"] == []
    assert (hud["showing"][-1]["heading"], hud["showing"][-1]["text"]) == ("Quest complete", "Throw the switches")
    assert hud["chimes"] == start["chimes"] + 2
    ages = [n["age"] for n in hud["showing"]]
    assert ages == sorted(ages, reverse=True)                 # oldest first: new ones join below

    hud = toasts(rt, lambda h: not h["visible"], hold=0)       # their time is up: they fade and go
    assert hud["showing"] == [] and hud["queued"] == [] and hud["rect"] == [] and hud["raised"] == 4


@pytest.mark.parametrize("size", [(1280, 720), (640, 360)])
def test_a_quest_given_and_failed_is_announced_bottom_left(fresh, size):
    rt = fresh
    width, height = size
    rt.call("window.size", size=list(size))
    for key, value in (("world:camp.alarm", False), ("actor:camp.quest_errand", "available")):
        rt.call("state.set", key=key, value=value)
    rt.call("interactions.set", asset="lever", state="up", wait=True)
    saved = rt.call("state.save")
    before = toasts(rt, hold=1000, gap=0, limit=3, clear=True)["raised"]

    rt.call("player.use", asset="lever")                      # the scene's rule gives the errand
    hud = toasts(rt, lambda h: len(h["showing"]) == 1)
    assert kinds(hud["showing"]) == [("started", "errand")]
    assert (hud["showing"][0]["heading"], hud["showing"][0]["text"]) == ("Quest started", LONG_TITLE)
    rt.call("state.set", key="world:camp.alarm", value=True)
    hud = toasts(rt, lambda h: len(h["showing"]) == 2 and h["viewport"] == [width, height]
                 and h["rect"][1] + h["rect"][3] == height - 12 and h["rect"][3] > 2 * 52)
    assert kinds(hud["showing"]) == [("started", "errand"), ("failed", "errand")]
    assert hud["showing"][1]["heading"] == "Quest failed" and hud["raised"] == before + 2

    # the bottom-left corner: clear of the tracker (top right) and the prompt under the crosshair
    x, y, w, h = hud["rect"]
    assert x == 12 and y + h == height - 12 and 170 <= w <= 320
    assert x + w <= width / 2 - 60
    assert h > 2 * 52 and y > height / 2                      # two toasts, the long title wrapped
    tracker = rt.call("hud.quests", track="next", linger=1000)["rect"]
    assert not tracker or tracker[0] >= x + w or tracker[1] + tracker[3] <= y

    # a reload resumes the state: no events, so nothing new, and what is up stays
    rt.call("world.reload")
    hud = toasts(rt)
    assert kinds(hud["showing"]) == [("started", "errand"), ("failed", "errand")] and hud["raised"] == before + 2
    # a save from before the errand: nothing fires, and the toasts that are no longer true go
    rt.call("state.load", state=saved)
    hud = toasts(rt, lambda h: h["showing"] == [])
    assert hud["showing"] == [] and hud["queued"] == [] and hud["raised"] == before + 2
    assert rt.call("quests.list")[1]["status"] == "available"


def test_another_actors_quests_raise_no_toasts(fresh):
    rt = fresh
    rt.call("state.set", key="world:camp.alarm", value=False)
    before = toasts(rt, hold=1000, gap=0, clear=True)
    rt.call("state.set", key="actor:camp.quest_errand", value="active", actor="npc:someone")
    rt.call("state.set", key="actor:camp.quest_errand_stand", value=True, actor="npc:someone")
    rt.call("state.set", key="actor:camp.quest_errand", value="complete", actor="npc:someone")
    assert rt.call("quests.list", actor="npc:someone")[1]["status"] == "complete"
    # the quest events did happen (the console has them): they just aren't the player's
    end = time.time() + 10
    while rt.log_path.read_text().count('"actor":"npc:someone"') < 3 and time.time() < end:
        time.sleep(0.05)
    events = [line for line in rt.log_path.read_text().splitlines()
              if line.startswith("quest event:") and '"actor":"npc:someone"' in line]
    assert [e.split('"event":"')[1].split('"')[0] for e in events] == ["started", "objective_done", "complete"]
    time.sleep(0.2)
    hud = rt.call("hud.notifications")
    assert hud["raised"] == before["raised"] and hud["showing"] == [] and hud["queued"] == []
    assert hud["chimes"] == before["chimes"] and not hud["visible"]


def test_the_keys_panel_lists_every_key_and_takes_the_keyboard_while_open(fresh):
    """? (Shift+/ on most keyboards: the panel takes the physical / too) opens the keys panel
    (hud_keys.gd, geogen.85): the game's groups from main.gd, then each panel's hud_keys()."""
    rt = fresh
    rt.call("window.size", size=[1280, 720])
    hud = rt.call("hud.keys")
    assert not hud["open"] and hud["rect"] == [] and hud["hint"] == "" and hud["columns"] == 0
    groups = {g["title"]: dict(g["keys"]) for g in hud["groups"]}      # listed while closed too
    assert list(groups) == ["Moving", "Playing", "Scenes and debugging", "Quests", "Inventory", "Quest log",
                            "Talking"]
    assert groups["Playing"]["E"] == "use what you aim at" and groups["Playing"]["?"] == "this list"
    assert "Q" in groups["Quests"] and "J" in groups["Quest log"] and set(groups["Inventory"]) == {"I", "F"}

    assert rt.call("input.key", key="Slash")
    hud = rt.call("hud.keys")
    assert hud["open"] and hud["columns"] == 2 and hud["hint"] == "? / Esc: close"
    x, y, w, h = hud["rect"]
    assert x >= 0 and y >= 0 and x + w <= 1280 and y + h <= 720 and w == 760 and h == 560
    tracked = rt.call("hud.quests")["tracked"]
    rt.call("input.key", key="Q")                      # the tracker's key: the panel has the keyboard
    assert rt.call("hud.quests")["tracked"] == tracked and rt.call("hud.keys")["open"]
    rt.call("input.key", key="Escape")
    assert not rt.call("hud.keys")["open"]
    rt.call("input.key", key="Question")               # what a typed ? reads as
    assert rt.call("hud.keys")["open"]
    rt.call("input.key", key="Slash")                  # the same key closes it
    assert not rt.call("hud.keys")["open"]

    hud = rt.call("hud.keys", open=True)
    assert hud["open"] and hud["rect"][2] == 760
    rt.call("window.size", size=[320, 200])
    hud = rt.call("hud.keys", open=False)
    hud = rt.call("hud.keys", open=True)
    assert hud["columns"] == 1 and hud["viewport"] == [320, 200]
    x, y, w, h = hud["rect"]
    assert x >= 0 and y >= 0 and x + w <= 320 and y + h <= 200
    assert not rt.call("hud.keys", open=False)["open"]
    rt.call("window.size", size=[1280, 720])
