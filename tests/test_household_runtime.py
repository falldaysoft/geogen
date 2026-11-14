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
