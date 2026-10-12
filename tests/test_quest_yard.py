"""The quest yard (scenes/quest_yard.yaml): the showcase for trying quests and the quest HUD. Its three
quests in the reference evaluator, and played in Godot with the tracker reporting what it shows."""

import pytest

from geogen.quests import collect
from geogen.rules import Machine

pytestmark = pytest.mark.xdist_group("test_quest_yard")

PKG = "quest_yard"
SWITCHES = [f"{PKG}:switch_{n}#switch" for n in (1, 2, 3)]


def enter(machine, place, actor="player"):
    machine.post({"type": "enter", "subject": f"{PKG}:{place}", "actor": actor})


def greet(machine, actor="player"):
    machine.post({"type": "signal", "subject": f"{PKG}:steward", "name": "greeted", "actor": actor})


@pytest.fixture(scope="module")
def yard(built_scene):
    return collect(built_scene("quest_yard"), PKG)


def play(yard):
    machine = Machine(yard.program(), {"player": {"kind": "player"}, f"{PKG}:steward": {"kind": "npc"}})

    def states(quest):
        report = yard.report(machine, "player")[quest]
        return report["status"], {o["id"]: o["state"] for o in report["objectives"]}

    return machine, states


def test_the_round_is_walked_in_order(yard):
    machine, states = play(yard)
    assert states("yard_round") == ("active", {"first": "open", "second": "waiting", "third": "waiting",
                                               "detour": "open"})                    # start: auto
    enter(machine, "marker_2")                              # out of turn: nothing
    enter(machine, "marker_1", actor=f"{PKG}:steward")      # somebody else: nothing
    assert states("yard_round")[1]["first"] == "open" and states("yard_round")[1]["second"] == "waiting"
    enter(machine, "marker_1")
    enter(machine, "behind_portal")
    assert states("yard_round")[1] == {"first": "done", "second": "open", "third": "waiting", "detour": "done"}
    enter(machine, "marker_2")
    enter(machine, "marker_3")
    assert states("yard_round")[0] == "complete"


def test_lamps_out_is_given_counted_and_reported_back(yard):
    machine, states = play(yard)
    lamps = yard.quests["yard_lamps"]["objectives"][0]
    assert lamps["progress"]["of"] == 3 and len(lamps["subjects"]) == 3
    assert states("yard_lamps")[0] == "available"
    greet(machine)
    assert states("yard_lamps") == ("active", {"lamps": "open", "report": "waiting"})     # the greeting that gave it isn't the report
    for n, switch in enumerate(SWITCHES, 1):
        machine.use(switch, "player")
        machine.advance(1)
        assert yard.report(machine, "player")["yard_lamps"]["objectives"][0]["progress"] == [n, 3]
    assert states("yard_lamps") == ("active", {"lamps": "done", "report": "open"})
    greet(machine)
    assert states("yard_lamps")[0] == "complete"


def test_the_dash_is_made_or_missed(yard):
    machine, states = play(yard)
    assert states("yard_dash")[0] == "available"
    enter(machine, "finish_line")                           # not running yet
    enter(machine, "start_line")
    assert states("yard_dash") == ("active", {"finish": "open"})
    machine.advance(5.9)
    assert states("yard_dash")[0] == "active"
    machine.advance(0.2)
    assert states("yard_dash")[0] == "failed"
    enter(machine, "finish_line")                           # too late, and failed is final
    enter(machine, "start_line")
    assert states("yard_dash") == ("failed", {"finish": "waiting"})

    machine, states = play(yard)                            # another session: made in time
    enter(machine, "start_line")
    machine.advance(4)
    enter(machine, "finish_line")
    machine.advance(10)                                     # the time-up rule finds nothing active to fail
    assert states("yard_dash") == ("complete", {"finish": "done"})


@pytest.fixture(scope="module")
def rt(scene_export, godot_runtime):
    client = godot_runtime("quest_yard", scene_export("quest_yard"))
    client.call("wait_ready")
    return client


def statuses(rt):
    return {q["id"]: q["status"] for q in rt.call("quests.list")}


def shown(rt):
    hud = rt.call("hud.quests")
    return hud["tracked"], [(line["id"], line["state"], line["progress_text"]) for line in hud["lines"]]


def test_the_yard_is_played_and_the_tracker_follows(rt):
    assert statuses(rt) == {"yard_round": "active", "yard_lamps": "available", "yard_dash": "available"}
    assert shown(rt) == ("yard_round", [("first", "open", ""), ("detour", "open", "")])
    assert rt.call("hud.quests")["waiting"] == 2

    # The round, on foot between the plinths' squares.
    for place, done in (([9, 0.1, 2], "first"), ([6.5, 0.1, -6.5], "second"), ([-1, 0.1, -9.5], "third")):
        walked = rt.send("player.walk_to", pos=place, timeout=30).get(timeout=120)["result"]
        assert walked["reached"], (place, walked["status"], walked["blocked_by"])
        report = {o["id"]: o["state"] for o in rt.call("quests.list")[0]["objectives"]}
        assert report[done] == "done", (done, report)
    assert statuses(rt)["yard_round"] == "complete"
    assert rt.call("hud.quests", linger=0)["visible"] is False          # nothing else is active yet

    # Lamps out: the steward gives it, the board's switches count, the steward hears it's done.
    def greet():
        for _ in range(30):
            if rt.call("npcs.greet", npc="steward")["ok"]:
                return
            rt.call("time.advance", seconds=2)
        pytest.fail("the steward never stood still to be greeted")

    greet()
    assert statuses(rt)["yard_lamps"] == "active"
    assert shown(rt) == ("yard_lamps", [("lamps", "open", "0 of 3")])
    for n in (1, 2, 3):
        rt.call("player.use", asset=f"switch_{n}")
        assert rt.call("quests.list")[1]["objectives"][0]["progress"] == [n, 3]
    assert shown(rt) == ("yard_lamps", [("lamps", "done", "3 of 3"), ("report", "open", "")])
    rt.call("time.advance", seconds=10)                      # past the greeting's cooldown
    greet()
    assert statuses(rt)["yard_lamps"] == "complete"
    rt.call("hud.quests", linger=0)

    # The dash, with the real body: onto the start square, then straight to the finish.
    walked = rt.send("player.walk_to", pos=[-10, 0.1, 3], timeout=30).get(timeout=120)["result"]
    assert walked["reached"] and statuses(rt)["yard_dash"] == "active"
    assert shown(rt) == ("yard_dash", [("finish", "open", "")])
    walked = rt.send("player.walk_to", pos=[-5.5, 0.1, -9.3], timeout=30).get(timeout=120)["result"]
    assert walked["reached"] and walked["seconds"] < 6, walked["seconds"]
    assert statuses(rt)["yard_dash"] == "complete"
    rt.call("time.advance", seconds=8)                       # time runs out on a finished quest: still complete
    assert statuses(rt) == {"yard_round": "complete", "yard_lamps": "complete", "yard_dash": "complete"}
