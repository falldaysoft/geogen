"""Plans in the Godot runtime (the dollhouse, geogen.101): a hungry person and a kitchen.
Nothing tells Ana to cook: the chain is found from what the things give and take."""

import time

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


# --- told what to do (geogen.101.8): orders, and the HUD's direct panel ---

def test_someone_told_to_eat_cooks_first_and_then_sees_to_herself_again(godot_runtime, scene_export):
    rt = launch(godot_runtime, scene_export, "household_test")
    full = {"hunger": 1.0, "rest": 1.0, "hygiene": 1.0, "order": 1.0, "social": 1.0}
    rt.call("npcs.needs", npc="ana", needs=full)
    rt.call("time.advance", seconds=40, scale=4)            # (what she set off on before she was full)
    rt.call("npcs.needs", npc="ana", needs=full)
    meals = sum(c for id, c in ana(rt)["used"].items() if id.endswith(".eat"))

    # What she could do with the chair: sit, or eat, which means the fridge and the range first.
    found = rt.call("npcs.choices", npc="ana", thing="chair")
    assert found["thing"] == "household_test:chair"
    by_label = {c["label"]: c for c in found["choices"]}
    assert set(by_label) == {"Sit at table", "Eat"} and all(c["available"] for c in by_label.values())
    assert by_label["Eat"]["plan"] == ["fridge#0", "range#0.meal", "chair#0.eat"]
    assert [c["label"] for c in rt.call("npcs.choices", npc="ana", thing="range")["choices"]] == ["Cook", "Make meal"]
    with pytest.raises(Exception, match="named 'piano'"):
        rt.call("npcs.choices", npc="ana", thing="piano")
    with pytest.raises(Exception, match="not something it can do"):
        rt.call("npcs.direct", npc="ana", option="piano#0")

    # Told to eat, then to wash up, then to sit: three orders, done in turn. The last is taken back.
    assert rt.call("npcs.direct", npc="ana", option="chair#0.eat")["orders"] == ["chair#0.eat"]
    rt.call("npcs.direct", npc="ana", option="sink#0")
    assert rt.call("npcs.direct", npc="ana", option="chair#0")["orders"] == ["chair#0.eat", "sink#0", "chair#0"]
    assert rt.call("npcs.cancel", npc="ana", index=2) == {"npc": "ana_npc", "cancelled": 1, "orders": ["chair#0.eat", "sink#0"]}
    seen = []
    for _ in range(60):
        rt.call("time.advance", seconds=3, scale=4)
        n = ana(rt)
        seen.append(n["doing"].split(":")[0])
        if not n["orders"]:
            break
    else:
        raise AssertionError((n["doing"], n["orders"], n["order"], n["failures"]))
    order = [id for i, id in enumerate(seen) if i == 0 or seen[i - 1] != id]
    assert [id for id in order if id in ("fridge#0", "range#0.meal", "chair#0.eat", "sink#0")] == \
        ["fridge#0", "range#0.meal", "chair#0.eat", "sink#0"], order
    assert not any(id.startswith("self/") for id in seen[:-1])      # nothing of her own in between
    assert sum(c for id, c in n["used"].items() if id.endswith(".eat")) == meals + 1
    assert n["orders_done"] == 2 and n["orders_failed"] == 0 and n["order"] == {"option": "sink#0", "ok": True, "reason": ""}
    decided = n["decisions"]
    rt.call("time.advance", seconds=30, scale=4)            # and with nothing left to do, she decides for herself
    assert ana(rt)["decisions"] > decided and ana(rt)["orders"] == [] and ana(rt)["orders_done"] == 2


def test_the_direct_panel_lists_what_a_thing_offers_and_queues_the_choice(godot_runtime, scene_export):
    rt = launch(godot_runtime, scene_export, "household_pair")
    for who in ("ana", "ben"):
        rt.call("npcs.needs", npc=who, needs={"social": 1.0, "rest": 1.0})
    rt.call("time.advance", seconds=30, scale=4)
    for who in ("ana", "ben"):
        rt.call("npcs.needs", npc=who, needs={"social": 1.0, "rest": 1.0})
    idle = rt.call("hud.direct")
    assert idle["active"] is False and idle["camera"] == "player"
    shown = rt.call("hud.direct", open=True, select="ana")
    assert shown["active"] and shown["selected"] == "ana_npc" and shown["name"] == "Ana" and shown["menu"] == []
    assert set(shown["needs"]) >= {"hunger", "social"}
    assert shown["camera"] == "overview"                    # looking down into the room, while directing

    sofa = rt.call("hud.direct", thing="sofa")              # three seats, one line
    assert sofa["thing"] == "household_pair:sofa" and [m["label"] for m in sofa["menu"]] == ["Sit"]
    # The other person: what they offer. Content as they both are, nobody would go over unasked.
    for _ in range(60):                                     # (on his way to a seat he is busy: "they are busy")
        chat = rt.call("hud.direct", thing="ben")
        assert [m["label"] for m in chat["menu"]] == ["Chat with Ben"]
        if chat["menu"][0]["available"]:
            break
        assert chat["menu"][0]["why"] == "they are busy"
        rt.call("time.advance", seconds=1, scale=4)
    assert chat["menu"][0]["available"] and chat["menu"][0]["plan"] == ["with/household_pair:ben/chat"]
    told = rt.call("hud.direct", choose=0)
    assert told["menu"] == [] and told["orders"] == [{"option": "with/household_pair:ben/chat", "label": "Chat with Ben"}]
    for _ in range(60):
        rt.call("time.advance", seconds=1, scale=4)
        people = both(rt)
        if not people[0]["orders"]:
            break
    assert people[0]["order"] == {"option": "with/household_pair:ben/chat", "ok": True, "reason": ""}
    assert people[1]["joined"] >= 1

    rt.call("hud.direct", thing="sofa", choose=0)
    assert len(rt.call("hud.direct")["orders"]) == 1
    assert rt.call("hud.direct", cancel=-1)["orders"] == []
    assert rt.call("hud.direct", select="ben")["name"] == "Ben"

    # The keys, as a person would use them (the world running: a paused tree hears no input). A click on a
    # line of the list does the same as its digit; the headless viewport is too small to hold the panel, so
    # that was checked in a window.
    rt.call("time.resume")
    rt.call("hud.direct", select="ana", thing="sofa")
    time.sleep(0.6)                                         # (past a timed redraw: the list is still there)
    assert len(rt.call("hud.direct")["lines"]) == 1
    rt.call("input.key", key="1")
    chosen = rt.call("hud.direct")
    assert [o["label"] for o in chosen["orders"]] == ["Sit"] and chosen["menu"] == []
    rt.call("input.key", key="Backspace")
    assert rt.call("hud.direct")["orders"] == []
    rt.call("input.key", key="Tab")
    assert rt.call("hud.direct")["selected"] == "ben_npc"
    rt.call("input.key", key="G")
    assert rt.call("hud.direct")["active"] is False
    rt.call("input.key", key="G")
    assert rt.call("hud.direct")["active"] is True
    rt.call("time.pause")
    closed = rt.call("hud.direct", open=False)
    assert closed["active"] is False and closed["camera"] == "player"
