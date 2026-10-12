"""The HUD's quest log (runtime/godot/scripts/hud_quest_log.gd): J opens a screen listing the
player's active, completed and failed quests with their objectives, and picks what the tracker
follows. `hud.log` returns what it holds; keys go in through `input.key`, as the keyboard's would."""

import time

import pytest

from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from geogen.runtime_client import ApiError
from test_hud_runtime import CAMP, ERRAND, LEVERS, LONG_TITLE
from test_state import GATE, LEVER
from test_wiring import PLATE

pytestmark = pytest.mark.xdist_group("test_hud_log_runtime")

# A second scene to go to and come back from, with a quest of its own.
ANNEX = """
name: annex
size: [8, 3, 8]
quests: [quests/stroll.yaml]
slots:
  a: {position: [2, 0, -2]}
place:
  plate: {asset: plate.yaml, slot: a}
spawns:
  start: {position: [0, 0, 3], facing: south}
"""

STROLL = """
kind: quest
version: 1
title: A stroll
description: Cross the annex to the plate.
start: auto
objectives:
  - {id: plate, text: Stand on the plate, done: {enter: plate/top}}
"""


@pytest.fixture(scope="module")
def rt(tmp_path_factory, godot_runtime):
    assets = tmp_path_factory.mktemp("log_assets")
    for name, text in (("lever", LEVER), ("gate", GATE), ("plate", PLATE), ("camp", CAMP), ("annex", ANNEX),
                       ("quests/levers", LEVERS), ("quests/errand", ERRAND), ("quests/stroll", STROLL)):
        path = assets / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    out = tmp_path_factory.mktemp("generated_log")
    for scene in ("camp", "annex"):
        export_scene(SceneComposer(assets_dir=assets).compose(assets / f"{scene}.yaml"), out / f"{scene}.glb")
    client = godot_runtime("camp", out)
    client.call("wait_ready")
    client.call("window.size", size=[1280, 720])
    return client


def log(rt, want=lambda hud: True, timeout=10.0, **args):
    """What `hud.log` holds, once it matches `want` (it redraws after the frame's events)."""
    end = time.time() + timeout
    hud = rt.call("hud.log", **args)
    while not want(hud) and time.time() < end:
        time.sleep(0.05)
        hud = rt.call("hud.log")
    return hud


def objectives(hud):
    return [(o["id"], o["state"], o["progress_text"]) for o in hud["selected"]["objectives"]]


def listed(hud):
    return [(q["key"], q["status"], q["loaded"]) for q in hud["quests"]]


def test_the_log_opens_mid_quest_and_picks_the_tracked_quest(rt):
    rt.call("player.use", asset="lever")                       # one lever pulled; the scene's rule gives the errand
    rt.call("state.set", key="camp:lever_2#pull", value="down")
    hud = log(rt, lambda h: len(h["sections"]["active"]) == 2)
    assert not hud["open"] and hud["rect"] == [] and hud["hint"] == ""      # closed, and still answers
    assert hud["sections"] == {"active": ["camp/levers", "camp/errand"], "complete": [], "failed": []}

    assert rt.call("input.key", key="J")
    hud = log(rt, lambda h: h["open"] and h["rect"] == [200, 100, 880, 520])
    assert hud["open"] and hud["rect"] == [200, 100, 880, 520] and hud["viewport"] == [1280, 720]
    assert hud["tracked"] == "camp/levers" and hud["selected"]["key"] == "camp/levers"      # opens on the tracked one
    assert hud["selected"]["title"] == "Throw the switches" and hud["selected"]["status"] == "active"
    assert hud["selected"]["tracked"] and not hud["selected"]["can_track"]
    # every objective, the one that waits on another too (the tracker leaves it out)
    assert objectives(hud) == [("all", "open", "2 of 3"), ("gate", "waiting", ""), ("plate", "open", "")]
    assert hud["selected"]["objectives"][2]["optional"]
    assert [(q["id"], q["done"], q["of"], q["tracked"]) for q in hud["quests"]] == [
        ("levers", 0, 2, True), ("errand", 0, 1, False)]
    assert hud["hint"] == "Up / Down: select    J / Esc: close"
    # it says what the console says
    console = {q["id"]: q for q in rt.call("quests.list")}
    assert [(o["id"], o["state"]) for o in console["levers"]["objectives"]] == [(o[0], o[1]) for o in objectives(hud)]

    # the world moves under it
    rt.call("state.set", key="camp:lever_3#pull", value="down")
    hud = log(rt, lambda h: h["selected"]["objectives"][0]["state"] == "done")
    assert objectives(hud) == [("all", "done", "3 of 3"), ("gate", "open", ""), ("plate", "open", "")]

    # down to the errand; Enter makes the tracker follow it
    rt.call("input.key", key="Down")
    hud = log(rt, lambda h: h["selected"]["id"] == "errand")
    assert hud["selected"]["title"] == LONG_TITLE and hud["selected"]["can_track"]
    assert hud["hint"] == "Up / Down: select    Enter: track    J / Esc: close"
    assert objectives(hud) == [("stand", "open", "")]
    rt.call("input.key", key="Down")                           # the end of the list: stays
    assert log(rt)["selected"]["id"] == "errand"
    rt.call("input.key", key="Q")                              # the tracker's key: the log has the keyboard
    assert rt.call("hud.quests")["tracked"] == "levers"
    rt.call("input.key", key="Enter")
    assert rt.call("hud.quests")["tracked"] == "errand"
    hud = log(rt, lambda h: h["tracked"] == "camp/errand")
    assert hud["tracked"] == "camp/errand" and hud["selected"]["tracked"] and not hud["selected"]["can_track"]
    assert rt.call("hud.log", select="previous")["selected"]["id"] == "levers"
    assert rt.call("hud.log", select="next", track=True)["tracked"] == "camp/errand"
    with pytest.raises(ApiError, match="isn't a quest in the log"):
        rt.call("hud.log", select="nope")

    rt.call("input.key", key="Escape")
    hud = log(rt, lambda h: not h["open"])
    assert not hud["open"] and hud["rect"] == []
    rt.call("input.key", key="Q")                              # closed: keys are the game's again
    assert rt.call("hud.quests")["tracked"] == "levers"


def test_e_does_nothing_while_the_log_is_open(rt):
    def pull():
        return rt.call("state.get", key="camp:lever_2#pull")["value"]

    rt.call("interactions.set", asset="lever_2", state="up", wait=True)
    rt.call("player.teleport", pos=[0, 0, 0.45], yaw=0)
    rt.call("player.look", at=[0, 0.6, 0])                    # the handle, within reach
    assert rt.call("hud.aim")["focus"].get("asset") == "lever_2"
    rt.call("input.key", key="J")
    assert log(rt, lambda h: h["open"])["open"]
    rt.call("input.key", key="E")
    time.sleep(0.3)
    assert pull() == "up"
    rt.call("input.key", key="J")                              # the same key closes it
    assert not log(rt, lambda h: not h["open"])["open"]
    rt.call("input.key", key="E")
    end = time.time() + 10
    while pull() != "down" and time.time() < end:
        time.sleep(0.05)
    assert pull() == "down"


def test_finished_quests_move_to_their_sections_and_cannot_be_tracked(rt):
    rt.call("hud.quests", linger=0)
    rt.call("state.set", key="camp:gate#gate", value="open")
    rt.call("state.set", key="world:camp.alarm", value=True)
    hud = log(rt, lambda h: h["open"] and h["sections"]["failed"] and h["sections"]["complete"], open=True)
    assert hud["sections"] == {"active": [], "complete": ["camp/levers"], "failed": ["camp/errand"]}
    assert listed(hud) == [("camp/levers", "complete", True), ("camp/errand", "failed", True)]
    assert hud["tracked"] == ""
    hud = rt.call("hud.log", select="levers")
    assert hud["selected"]["status"] == "complete" and not hud["selected"]["can_track"]
    assert objectives(hud) == [("all", "done", "3 of 3"), ("gate", "done", ""), ("plate", "waiting", "")]
    with pytest.raises(ApiError, match="isn't active"):
        rt.call("hud.log", track=True)
    hud = rt.call("hud.log", select="camp/errand")
    assert hud["selected"]["status"] == "failed" and objectives(hud) == [("stand", "waiting", "")]
    assert hud["hint"] == "Up / Down: select    J / Esc: close"


def test_the_log_survives_a_scene_switch_and_lists_the_scene_left_behind(rt):
    assert rt.call("hud.log")["open"]
    rt.call("world.switch", scene="annex")
    rt.call("wait_ready")
    hud = log(rt, lambda h: len(h["quests"]) == 3)
    assert hud["open"]                                         # still up, over the new scene
    # this scene's quest, then the camp's as they stood
    assert listed(hud) == [("annex/stroll", "active", True), ("camp/levers", "complete", False),
                           ("camp/errand", "failed", False)]
    assert hud["tracked"] == "annex/stroll"
    hud = rt.call("hud.log", select="camp/levers")
    assert objectives(hud) == [("all", "done", "3 of 3"), ("gate", "done", ""), ("plate", "waiting", "")]
    hud = rt.call("hud.log", select="stroll")
    assert hud["selected"]["description"] == "Cross the annex to the plate." and hud["selected"]["tracked"]
    rt.call("hud.log", open=False)

    rt.call("world.switch", scene="camp")
    rt.call("wait_ready")
    hud = log(rt, lambda h: len(h["quests"]) == 3 and h["quests"][-1]["loaded"])
    assert listed(hud) == [("annex/stroll", "active", False), ("camp/levers", "complete", True),
                           ("camp/errand", "failed", True)]
    assert hud["tracked"] == "" and not rt.call("hud.quests")["visible"]


def test_a_quest_of_another_scene_is_tracked_when_you_are_back_there(rt):
    rt.call("state.set", key="world:camp.alarm", value=False)
    rt.call("state.set", key="actor:camp.quest_errand", value="active")
    rt.call("state.set", key="actor:camp.quest_levers", value="active")
    rt.call("world.switch", scene="annex")
    rt.call("wait_ready")
    hud = log(rt, lambda h: len(h["sections"]["active"]) == 3, open=True)
    assert hud["sections"]["active"] == ["annex/stroll", "camp/levers", "camp/errand"]
    assert hud["tracked"] == "annex/stroll"
    hud = rt.call("hud.log", select="camp/errand", track=True)
    assert hud["selected"]["tracked"] and not hud["selected"]["loaded"]
    rt.call("hud.log", open=False)
    rt.call("world.switch", scene="camp")
    rt.call("wait_ready")
    assert log(rt, lambda h: h["tracked"] == "camp/errand")["tracked"] == "camp/errand"
    assert rt.call("hud.quests")["tracked"] == "errand"


def test_the_log_follows_a_save_and_a_reload(rt):
    hud = log(rt, lambda h: h["open"], open=True)
    assert hud["sections"]["active"] == ["camp/levers", "camp/errand", "annex/stroll"]
    saved = rt.call("state.save")
    rt.call("state.set", key="actor:camp.quest_levers", value="complete")
    rt.call("state.set", key="actor:camp.quest_errand", value="failed")
    hud = log(rt, lambda h: len(h["sections"]["active"]) == 1)
    assert hud["sections"]["active"] == ["annex/stroll"]        # the scene left behind isn't part of this save
    assert (hud["sections"]["complete"], hud["sections"]["failed"]) == (["camp/levers"], ["camp/errand"])

    rt.call("state.load", state=saved)                         # nothing fires: the log re-reads
    hud = log(rt, lambda h: len(h["sections"]["active"]) == 3)
    assert hud["open"] and hud["sections"] == {"active": ["camp/levers", "camp/errand", "annex/stroll"],
                                               "complete": [], "failed": []}
    rt.call("world.reload")
    rt.call("wait_ready")
    hud = log(rt, lambda h: len(h["sections"]["active"]) == 3 and h["quests"][0]["loaded"])
    assert hud["open"] and hud["sections"]["active"] == ["camp/levers", "camp/errand", "annex/stroll"]
    assert rt.call("hud.log", open=False)["open"] is False


@pytest.mark.parametrize("size", [(640, 360), (320, 200)])
def test_the_log_fits_a_small_window(rt, size):
    width, height = size
    rt.call("window.size", size=list(size))
    hud = log(rt, lambda h: h["viewport"] == [width, height] and h["rect"] and h["rect"][2] <= width, open=True)
    x, y, w, h = hud["rect"]
    assert x >= 0 and y >= 0 and x + w <= width and y + h <= height
    rt.call("hud.log", open=False)
    rt.call("window.size", size=[1280, 720])
