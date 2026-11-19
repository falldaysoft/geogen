"""Perception in the Godot runtime (geogen.99.1), on the wants_house scene: Hana has senses and
reacts to stimuli (npcs/stimuli.yaml). A thing dropped in her room has her look up, leave her
chair and see to it; what she can't hear or see, and what she did herself, she doesn't notice.
With memory (geogen.99.2) her wants are about what she has seen or heard: a thing behind a shut
door is news to her when she sees it, one she only heard fall she goes to look for first, and a
spot she can see where a remembered thing no longer is she forgets."""


import json
from pathlib import Path

import pytest

from helpers import schema_validate

pytestmark = [pytest.mark.godot, pytest.mark.xdist_group("test_perception_runtime")]

LIVING = [-2.4, 0.05, 1.0]
BEDROOM = [2.2, 0.05, 0.0]
DOOR = "wants_house:flat/door_living_bedroom#swing"


@pytest.fixture(scope="module")
def rt(scene_export, godot_runtime):
    client = godot_runtime("wants_house", scene_export("wants_house"))
    client.call("wait_ready")
    client.call("time.pause")           # the world runs only inside time.advance
    return client


def hana(rt):
    return next(n for n in rt.call("npcs.list") if n["npc"].startswith("hana"))


def can(rt):
    return next(b for b in rt.call("world.bodies") if b.get("node") == "can")


def run_until(rt, done, seconds=120.0, step=0.25):
    spent = 0.0
    while spent < seconds:
        rt.call("time.advance", seconds=step, scale=4)
        spent += step
        got = done()
        if got:
            return got
    raise AssertionError(f"not within {seconds} s: {hana(rt)['doing']}, pose {hana(rt)['pose']}, notice {hana(rt)['notice']}, "
                         f"failures {hana(rt)['failures']}, want {hana(rt)['want']}")


def seated(rt):
    """Hana in her armchair with a long sit ahead of her (so only an interruption gets her up soon)."""
    rt.call("npcs.resume", npc="hana")
    rt.call("player.teleport", pos=LIVING, yaw=90)

    def sat():
        r = hana(rt)
        return r["pose"] == "sit" and r["doing"].startswith("armchair#0: wait") and r
    run_until(rt, lambda: rt.call("interactions.states")[DOOR]["state"] == "closed" and hana(rt)["want"]["doing"] == {})
    if not hana(rt)["doing"].startswith("armchair#0"):                     # (a forced option waits for her next decision)
        rt.call("npcs.force", npc="hana", option="armchair#0")
    return run_until(rt, sat)


def drop_the_can(rt, at):
    rt.call("player.teleport", pos=at, yaw=90)
    assert rt.call("player.carry", prop="can")["ok"]
    rt.call("time.advance", seconds=0.5, scale=1)
    rt.call("player.drop")
    rt.call("time.advance", seconds=0.1, scale=1)
    rt.call("player.teleport", pos=[at[0], at[1], at[2] - 2.2], yaw=90)    # out of the way of whoever comes for it


def can_id(rt):
    return can(rt)["uid"]


def test_the_definition_has_her_senses_and_what_she_reacts_to(built_scene):
    npc = next(n for n in built_scene("wants_house").iter_nodes() if n.name == "hana").meta["npc"]
    assert npc["senses"] == {"hearing": {"range": 10.0, "muffled": 0.3}, "sight": {"range": 8.0, "cone": 150.0}}
    assert npc["reacts"]["thud"] == {"event": "impact", "loudness": 1.0, "look": [0.8, 1.5], "interrupt": True}
    assert npc["reacts"]["door"]["of"] == "door" and not npc["reacts"]["door"]["interrupt"]
    assert npc["actions"]["sit"]["interruptible"] and "interruptible" not in npc["actions"]["put_away"]
    assert npc["memory"] == {"forget": 300.0, "every": 0.5, "search": "look_for"}
    assert "look_for" in npc["actions"]


def test_a_thing_dropped_in_her_room_gets_her_out_of_her_chair(rt):
    was = seated(rt)
    drop_the_can(rt, LIVING)
    # Without noticing she would sit out her 15 to 40 s; she is on it within a few.
    now = run_until(rt, lambda: hana(rt)["want"]["doing"].get("subject", "").endswith(":can") and hana(rt), seconds=8)
    assert now["noticed"] > was["noticed"] and now["cut_short"] == was["cut_short"] + 1
    assert now["notice"]["stimulus"] == "thud" and now["notice"]["subject"].endswith(":can")
    assert now["notice"]["by"] in ("heard", "seen")
    assert now["failures"] == was["failures"]                              # cut short is not a failure
    # Her own put-down is nothing to notice.
    held = run_until(rt, lambda: hana(rt)["prop"] != "" and hana(rt))
    run_until(rt, lambda: hana(rt)["prop"] == "" and hana(rt)["want"]["doing"] == {})
    rt.call("time.advance", seconds=2, scale=4)
    assert can(rt)["center"][1] > 0.3 and hana(rt)["noticed"] == held["noticed"]


def test_a_thing_dropped_behind_a_closed_door_is_not_noticed(rt):
    seated(rt)
    # The player carries the can through the doorway into the bedroom and shuts the door.
    rt.call("interactions.set", asset="door_living_bedroom", state="open", instant=True)
    rt.call("player.teleport", pos=[-0.6, 0.05, 0.0], yaw=-90)
    assert rt.call("player.carry", prop="can")["ok"]
    rt.call("time.advance", seconds=1, scale=1)
    rt.call("player.teleport", pos=BEDROOM, yaw=-90)
    rt.call("time.advance", seconds=1.5, scale=1)
    assert can(rt)["center"][0] > 1.0 and can(rt)["held"]
    rt.call("interactions.set", asset="door_living_bedroom", state="closed", instant=True)
    rt.call("time.advance", seconds=0.3, scale=1)
    was = hana(rt)
    assert was["pose"] == "sit"
    rt.call("player.drop")
    rt.call("time.advance", seconds=3, scale=1)
    now = hana(rt)
    assert can(rt)["center"][1] < 0.3 and can(rt)["center"][0] > 1.0       # it did fall, in there
    assert now["noticed"] == was["noticed"] and now["cut_short"] == was["cut_short"], (now["notice"], now["position"])
    assert now["pose"] == "sit"
    # Nor does she know it is on the floor in there: it is no want of hers.
    assert now["memory"].get(can_id(rt), {}).get("resting") != "floor", now["memory"]
    assert not [o for o in rt.call("npcs.options", npc="hana") if o["id"].endswith(":can")]


def test_a_door_opening_turns_her_head_but_leaves_her_be(rt):
    was = seated(rt)
    rt.call("interactions.use", asset="door_living_bedroom")               # someone opens it
    rt.call("time.advance", seconds=0.5, scale=1)
    now = hana(rt)
    assert now["noticed"] == was["noticed"] + 1 and now["notice"]["stimulus"] == "door", now["notice"]
    assert now["cut_short"] == was["cut_short"] and now["pose"] == "sit"
    assert now["head_turn"] > 5                                            # she looks its way


def fetch_the_can(rt):
    """The player picks the can up wherever it is and holds it."""
    if can(rt)["held"]:
        return
    rt.call("interactions.set", asset="door_living_bedroom", state="open", instant=True)
    at = can(rt)["center"]
    rt.call("player.teleport", pos=[at[0] + 0.6, 0.05, at[2]], yaw=90)
    assert rt.call("player.carry", prop="can")["ok"]
    rt.call("time.advance", seconds=0.5, scale=1)


def test_a_thing_she_only_heard_fall_she_goes_to_look_for(rt):
    fetch_the_can(rt)
    was = seated(rt)                                   # (the player brings the can along: she sees it held)
    # Dropped beside and behind her: heard, out of sight.
    rt.call("player.teleport", pos=[-2.45, 0.05, 1.7], yaw=90)
    rt.call("time.advance", seconds=0.3, scale=1)
    rt.call("player.drop")
    rt.call("time.advance", seconds=0.8, scale=1)                         # (a fall from the player's hands)
    rt.call("player.teleport", pos=[-1.5, 0.05, -1.5], yaw=90)
    heard = hana(rt)
    assert heard["notice"]["by"] == "heard" and heard["memory"][can_id(rt)]["by"] == "heard", heard["memory"]
    assert heard["memory"][can_id(rt)]["resting"] == ""                    # where it ended up isn't known
    looking = run_until(rt, lambda: hana(rt)["want"]["doing"].get("search") and hana(rt), seconds=10)
    assert looking["searches"] == was["searches"] + 1 and looking["want"]["doing"]["by"] == "heard"
    # Having looked, she knows it is on the floor, and puts it away.
    run_until(rt, lambda: hana(rt)["prop"] != "", seconds=20)
    now = run_until(rt, lambda: hana(rt)["prop"] == "" and hana(rt)["want"]["doing"] == {} and hana(rt))
    assert can(rt)["center"][1] > 0.3 and now["failures"] == was["failures"]
    assert now["searches"] == looking["searches"]


def test_a_spot_she_can_see_with_the_thing_gone_she_forgets(rt):
    fetch_the_can(rt)
    seated(rt)
    # Frozen in her chair, she can't look round; the can lands in plain view in front of her.
    rt.call("npcs.pause", npc="hana")
    rt.call("player.teleport", pos=[0.1, 0.05, 1.0], yaw=90)
    rt.call("time.advance", seconds=0.3, scale=1)
    rt.call("player.drop")
    rt.call("time.advance", seconds=1.5, scale=1)
    assert can(rt)["center"][1] < 0.3
    noticed = hana(rt)["memory"][can_id(rt)]
    assert noticed["by"] == "seen", noticed                                # she saw it land
    # The player takes it away into the bedroom and shuts the door before she looks again.
    fetch_the_can(rt)
    rt.call("player.teleport", pos=[-0.6, 0.05, 0.0], yaw=-90)            # (through the doorway, not the wall)
    rt.call("time.advance", seconds=1.0, scale=1)
    rt.call("player.teleport", pos=BEDROOM, yaw=-90)
    rt.call("time.advance", seconds=1.0, scale=1)
    assert can(rt)["held"]
    rt.call("interactions.set", asset="door_living_bedroom", state="closed", instant=True)
    rt.call("player.drop")
    rt.call("time.advance", seconds=1.0, scale=1)
    was = hana(rt)
    rt.call("npcs.resume", npc="hana")
    rt.call("time.advance", seconds=1.0, scale=1)
    now = hana(rt)
    assert can_id(rt) not in now["memory"], now["memory"]                  # the spot is empty: forgotten
    run_until(rt, lambda: hana(rt)["want"]["doing"] == {} and hana(rt)["decisions"] > was["decisions"] + 1, seconds=30)
    assert hana(rt)["searches"] == was["searches"]
    assert not [o for o in rt.call("npcs.options", npc="hana") if o["id"].endswith(":can")]


def test_a_save_keeps_what_she_remembers(rt):
    """(geogen.99.4) Memory is in the save, by the props' stable IDs, and loading takes it up."""
    fetch_the_can(rt)
    seated(rt)
    rt.call("npcs.pause", npc="hana")                  # (she doesn't look round, so only the save changes her memory)
    rt.call("player.teleport", pos=[0.1, 0.05, 1.0], yaw=90)
    rt.call("time.advance", seconds=0.3, scale=1)
    rt.call("player.drop")
    rt.call("time.advance", seconds=1.5, scale=1)
    knew = hana(rt)["memory"][can_id(rt)]
    saved = rt.call("state.save")
    schema_validate(saved, json.loads((Path(__file__).parent.parent / "docs/schema/geogen-save.v1.schema.json").read_text()))
    entry = next(v for k, v in saved["npcs"].items() if k.endswith("hana"))["memory"][can_id(rt)]
    assert entry["by"] == knew["by"] == "seen" and entry["resting"] == knew["resting"]
    # Forgotten in the meantime (an empty memory), then the save loaded: she knows it again.
    blank = {**saved, "npcs": {k: {**v, "memory": {}} for k, v in saved["npcs"].items()}}
    rt.call("state.load", state=blank)
    assert can_id(rt) not in hana(rt)["memory"]
    rt.call("state.load", state=saved)
    back = hana(rt)["memory"][can_id(rt)]
    assert back["by"] == "seen" and back["resting"] == knew["resting"] and back["at"] == knew["at"], (back, knew)
    assert back["age"] >= entry["age"]
    rt.call("npcs.resume", npc="hana")
