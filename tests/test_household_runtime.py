"""Plans in the Godot runtime (the dollhouse, geogen.101): a hungry person and a kitchen.
Nothing tells Ana to cook: the chain is found from what the things give and take."""

import pytest

pytestmark = [pytest.mark.godot, pytest.mark.xdist_group("test_household_runtime")]


def ana(rt):
    return next(n for n in rt.call("npcs.list") if n["npc"].startswith("ana"))


def run_until(rt, done, seconds=400.0, step=5.0):
    spent = 0.0
    while spent < seconds:
        rt.call("time.advance", seconds=step, scale=4)
        spent += step
        got = done(ana(rt))
        if got:
            return got
    raise AssertionError(f"not within {seconds} s: {ana(rt)}")


def launch(godot_runtime, scene_export, scene):
    rt = godot_runtime(scene, scene_export(scene))
    rt.call("wait_ready")
    rt.call("time.pause")           # the world runs only inside time.advance
    return rt


@pytest.fixture(scope="module")
def kitchen(scene_export, godot_runtime):
    return launch(godot_runtime, scene_export, "household_test")


def test_a_hungry_person_cooks_and_eats_by_a_plan_nobody_wrote(kitchen):
    rt = kitchen
    plan = run_until(rt, lambda a: a["plan"])
    assert plan["steps"] == ["fridge#0", "range#0.meal", "chair#0.eat"] and plan["goal"] == "chair#0.eat"
    hungry = ana(rt)["needs"]["hunger"]
    run_until(rt, lambda a: "raw_food" in a["carrying"])                # out of the fridge
    run_until(rt, lambda a: a["carrying"] == ["meal"])                  # cooked: the raw food is used up
    # Eating fills hunger as it goes, not at the end.
    part = run_until(rt, lambda a: a["pose"] == "sit" and a["needs"]["hunger"] > hungry + 0.1 and a)
    assert part["carrying"] == ["meal"] and part["plan"]["index"] == 2
    done = run_until(rt, lambda a: a["plans_done"] >= 1 and a)
    assert done["carrying"] == ["dirty_plate"] and done["needs"]["hunger"] > hungry + 0.5
    assert done["replans"] == 0 and done["failures"] == []
    # The plate it leaves is washed up when that is what matters most.
    washed = run_until(rt, lambda a: a["carrying"] == [] and a, seconds=600.0)
    assert washed["used"].get("sink#0") == 1


def test_a_microwave_in_the_room_changes_the_plan(scene_export, godot_runtime):
    rt = launch(godot_runtime, scene_export, "household_microwave")
    plan = run_until(rt, lambda a: a["plan"])
    assert plan["steps"] == ["fridge#0", "microwave#0", "chair#0.eat"]
    assert run_until(rt, lambda a: a["plans_done"] >= 1 and a)["carrying"] == ["dirty_plate"]


def test_a_plan_that_breaks_is_dropped_and_made_again(scene_export, godot_runtime):
    """The meal is taken from her on the way to the table: the chair's step isn't possible any
    more, the plan is dropped, and a later one starts at the fridge again."""
    rt = launch(godot_runtime, scene_export, "household_test")
    run_until(rt, lambda a: a["carrying"] == ["meal"])
    rt.call("inventory.give", item="meal", count=-1, actor=ana(rt)["actor"])
    dropped = run_until(rt, lambda a: a["replans"] >= 1 and a)
    assert any("nothing left of what it takes" in f for f in dropped["failures"]) and not dropped["used"].get("chair#0.eat")
    # The chair isn't tried again for a while (scoring.retry); then the chain starts at the fridge again.
    again = run_until(rt, lambda a: a["plan"].get("steps", [""])[0] == "fridge#0" and a)
    assert again["plan"]["steps"] == ["fridge#0", "range#0.meal", "chair#0.eat"]
    assert run_until(rt, lambda a: a["used"].get("chair#0.eat") and a)["carrying"] == ["dirty_plate"]     # eaten after all


def test_a_need_that_runs_out_has_its_consequence(scene_export, godot_runtime):
    """`rest` has `at_zero: collapse`: with none left she sleeps on the floor where she stands,
    before anything else, and that gives some of it back."""
    rt = launch(godot_runtime, scene_export, "household_test")
    run_until(rt, lambda a: a["plan"])
    assert rt.call("npcs.needs", npc="ana", needs={"rest": 0.0})["needs"]["rest"] == 0.0
    down = run_until(rt, lambda a: a["ran_out"] >= 1 and a["pose"] == "lie" and a, step=2.0)
    assert down["doing"].startswith("need/rest") and down["plan"] == {}
    up = run_until(rt, lambda a: a["pose"] == "stand" and a["used"].get("need/rest") and a)
    assert up["needs"]["rest"] > 0.1 and up["ran_out"] == 1 and up["failures"] == []


# --- company (geogen.101.6): scenes/household_pair.yaml, two people and a sofa ---

def both(rt):
    people = {n["npc"].removesuffix("_npc"): n for n in rt.call("npcs.list")}
    return people["ana"], people["ben"]


def turn_between(a, b):
    """Degrees between where ``a`` faces and where ``b`` stands."""
    import math
    dx, dz = b["position"][0] - a["position"][0], b["position"][2] - a["position"][2]
    return abs((math.degrees(math.atan2(dx, dz)) - a["yaw"] + 180.0) % 360.0 - 180.0)


def test_someone_short_of_company_walks_over_and_both_get_it(godot_runtime, scene_export):
    rt = launch(godot_runtime, scene_export, "household_pair")
    for who in ("ana", "ben"):
        rt.call("npcs.needs", npc=who, needs={"social": 1.0, "rest": 1.0})
    chats = lambda: sum(c for n in both(rt) for id, c in n["used"].items() if id.startswith("with/"))   # noqa: E731
    rt.call("time.advance", seconds=30, scale=4)            # (what was decided before the needs were set)
    for who in ("ana", "ben"):
        rt.call("npcs.needs", npc=who, needs={"social": 1.0})
    had = chats()
    rt.call("time.advance", seconds=60, scale=4)
    assert chats() == had                                   # content: nobody is bothered for a top-up
    ana, ben = both(rt)
    known, joined = ana["relations"].get(ben["uid"], 0.5), ben["joined"]

    rt.call("npcs.needs", npc="ana", needs={"social": 0.2})
    before = ben["needs"]["social"]
    together = None
    for _ in range(120):
        rt.call("time.advance", seconds=1, scale=4)
        ana, ben = both(rt)
        if "chat: wait" in ana["doing"]:
            together = (ana, ben)
        if chats() > had:
            break
    else:
        raise AssertionError((ana["doing"], ana["needs"], ana["failures"], ben["doing"]))
    assert together, "never seen talking"
    a, b = together                                         # held for each other, close, turned to each other
    assert a["with"] == b["uid"] and b["with"] == a["uid"]
    gap = sum((p - q) ** 2 for p, q in zip(a["position"], b["position"])) ** 0.5
    assert 0.5 < gap < 1.6, gap
    assert turn_between(a, b) < 25, (a["yaw"], a["position"], b["position"])
    if b["pose"] == "stand":
        assert "chat_reply" in b["doing"] and turn_between(b, a) < 25, (b["doing"], b["yaw"])
    assert ana["needs"]["social"] > 0.45 and ben["joined"] == joined + 1
    assert ben["needs"]["social"] > before - 0.1            # (it was full: what it got covers what it lost meanwhile)
    # They live together (`household:`), so they knew each other 0.5; now a little better, both.
    assert known >= 0.5                                     # (under 0.04: the next chat may have begun)
    assert known + 0.0199 <= ana["relations"][ben["uid"]] < known + 0.04
    assert ben["relations"][ana["uid"]] == ana["relations"][ben["uid"]]


def test_someone_seated_stays_seated_for_a_chat(godot_runtime, scene_export):
    rt = launch(godot_runtime, scene_export, "household_pair")
    for who in ("ana", "ben"):
        rt.call("npcs.needs", npc=who, needs={"social": 1.0, "rest": 1.0})
    rt.call("npcs.needs", npc="ben", needs={"rest": 0.1})      # tired: the sofa
    for _ in range(60):
        rt.call("time.advance", seconds=1, scale=4)
        if both(rt)[1]["pose"] == "sit":
            break
    rt.call("npcs.needs", npc="ana", needs={"social": 0.2})
    rt.call("npcs.needs", npc="ben", needs={"social": 0.6})
    seen = []
    for _ in range(40):
        rt.call("time.advance", seconds=0.5, scale=4)
        ana, ben = both(rt)
        if "chat: wait" in ana["doing"] and ben["pose"] == "sit":
            seen.append(ben)
    assert len(seen) >= 4, (ana["doing"], ana["failures"], ben["pose"], ben["doing"])
    assert all(n["doing"].startswith("sofa") for n in seen)         # still at what it was doing
    assert seen[-1]["needs"]["social"] > 0.62 and max(n["head_turn"] for n in seen) > 5
