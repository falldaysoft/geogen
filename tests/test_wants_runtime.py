"""Wants in the Godot runtime (geogen.97): NPCs carrying physics props, on the wants_test scene."""


import math

import pytest

pytestmark = [pytest.mark.godot, pytest.mark.xdist_group("test_wants_runtime")]


@pytest.fixture(scope="module")
def rt(scene_export, godot_runtime):
    client = godot_runtime("wants_test", scene_export("wants_test"))
    client.call("wait_ready")
    client.call("time.pause")           # the world runs only inside time.advance: the same steps every run
    return client


def body(rt, node):
    return next(b for b in rt.call("world.bodies") if b.get("node") == node)


def resident(rt):
    return next(n for n in rt.call("npcs.list") if n["npc"].startswith("resident"))


def run_until(rt, done, seconds=120.0, step=1.0):
    """Advance the world until ``done()`` (a dict or True), at most ``seconds`` of world time."""
    spent = 0.0
    while spent < seconds:
        rt.call("time.advance", seconds=step, scale=4)
        spent += step
        got = done()
        if got:
            return got
    raise AssertionError(f"not within {seconds} s: {resident(rt)['doing']}, failures {resident(rt)['failures']}, "
                         f"options {rt.call('npcs.options', npc='resident')[:4]}")


def to_the_floor(rt, node, resume=True):
    """The player takes ``node`` from wherever it is and drops it on the open floor at (0, 1.4)."""
    rt.call("npcs.pause", npc="resident")
    rt.call("player.teleport", pos=[1.4, 0.05, 1.4], yaw=90)
    assert rt.call("player.carry", prop=node)["ok"]
    rt.call("time.advance", seconds=1, scale=4)
    rt.call("player.drop")
    rt.call("time.advance", seconds=2, scale=4)                            # it lands
    assert body(rt, node)["center"][1] < 0.3
    if resume:
        rt.call("npcs.resume", npc="resident")


def test_the_resident_tidies_what_is_on_the_floor(rt):
    # The scene starts with a pumpkin on the floor and a can on the table.
    run_until(rt, lambda: resident(rt)["wants"] >= 1)
    rt.call("time.advance", seconds=2, scale=4)
    rest = body(rt, "pumpkin")
    assert rest["holder"] == "" and rest["speed"] < 0.05
    assert rest["center"][1] == pytest.approx(0.885, abs=0.03) and rest["center"][0] < -0.4   # on the table it prefers
    assert rt.call("npcs.places", npc="resident", want="tidy", prop="pumpkin")["resting_on"] == "thing"
    assert body(rt, "can")["center"][1] == pytest.approx(0.81, abs=0.03)   # what is on a table stays there
    assert not resident(rt)["failures"]


def test_the_world_offers_places_that_hold_a_prop(rt):
    to_the_floor(rt, "pumpkin", resume=False)
    found = rt.call("npcs.places", npc="resident", want="tidy", prop="pumpkin")
    assert found["resting_on"] == "floor"
    places = found["places"]
    owners = {p["uid"].rsplit(":", 1)[-1].split("/")[0] for p in places}
    # Nobody authored the chair's seat as a place.
    assert {"table", "chair", "bookshelf"} <= owners, owners
    assert all(p["height"] > 0.2 for p in places)                          # nowhere actors walk
    assert places[0]["score"] > places[-1]["score"]
    can = body(rt, "can")["center"]                                        # what is on the table is in the way
    assert all(abs(p["position"][0] - can[0]) > 0.15 or abs(p["position"][2] - can[2]) > 0.15
               for p in places if p["uid"].endswith("table"))
    rt.call("npcs.resume", npc="resident")


def tidied(rt, more=1):
    """Let the resident tidy ``more`` things from now, and what it has put down settle."""
    target = resident(rt)["wants"] + more
    run_until(rt, lambda: resident(rt)["wants"] >= target)
    rt.call("time.advance", seconds=2, scale=4)


def test_knocked_down_again_it_goes_somewhere_else(rt):
    if body(rt, "pumpkin")["center"][1] < 0.3:
        tidied(rt)                                                         # (run on its own: the scene starts with it down)
    first = body(rt, "pumpkin")["center"]
    to_the_floor(rt, "pumpkin")
    tidied(rt)
    second = body(rt, "pumpkin")["center"]
    assert second[1] > 0.3
    assert sum((a - b) ** 2 for a, b in zip(first, second)) ** 0.5 > 0.25  # not the spot it fell from


def test_an_npc_carries_a_prop_from_the_floor_to_the_table(rt):
    to_the_floor(rt, "pumpkin")
    before = resident(rt)["put_down"]
    rt.call("npcs.do", npc="resident", action="put_away", subject="pumpkin", place=[-0.6, 0.75, 1.3])
    npc = run_until(rt, lambda: resident(rt)["prop"] and resident(rt), step=0.25)
    rt.call("time.advance", seconds=0.1, scale=1)                          # (taken on that frame, it is at the hands by the next)
    held = body(rt, "pumpkin")
    assert held["holder"] == npc["uid"] and held["center"][1] > 0.6        # in its hand, off the floor
    run_until(rt, lambda: resident(rt)["put_down"] == before + 1)
    rt.call("time.advance", seconds=2, scale=4)
    rest = body(rt, "pumpkin")
    assert rest["holder"] == "" and resident(rt)["prop"] == ""
    assert rest["center"][1] == pytest.approx(0.75 + 0.135, abs=0.03)      # resting on the table top
    assert abs(rest["center"][0] + 0.6) < 0.15 and abs(rest["center"][2] - 1.3) < 0.15
    # (Run on its own, the player took the pumpkin out of her hands first: that tidy failed, not this.)
    assert rest["speed"] < 0.05 and not [f for f in resident(rt)["failures"] if f.startswith("do/")]


def test_the_player_takes_a_prop_out_of_an_npcs_hand(rt):
    rt.call("npcs.do", npc="resident", action="put_away", subject="pumpkin", place=[0.0, 0.0, 0.0])
    run_until(rt, lambda: resident(rt)["prop"] != "")
    assert rt.call("player.carry", prop="pumpkin")["ok"]
    assert body(rt, "pumpkin")["holder"] == "player" and resident(rt)["prop"] == ""
    run_until(rt, lambda: any("holding nothing" in f for f in resident(rt)["failures"]))
    rt.call("player.drop")
    rt.call("time.advance", seconds=2, scale=4)                            # it lands


def test_what_is_out_of_reach_or_too_heavy_is_refused(rt):
    before = len(resident(rt)["failures"])
    rt.call("npcs.do", npc="resident", action="put_away", subject="pumpkin", place=[0.0, 2.4, 0.0])   # by the ceiling
    run_until(rt, lambda: len(resident(rt)["failures"]) > before)
    assert "the place is out of reach" in resident(rt)["failures"][-1]
    assert body(rt, "pumpkin")["holder"] == ""                             # dropped where it stood
    # The can stands mid-table against the wall, with the chair in front: nowhere to stand in reach.
    rt.call("npcs.do", npc="resident", action="put_away", subject="can", place=[0.0, 0.0, 0.0])
    run_until(rt, lambda: len(resident(rt)["failures"]) > before + 1)
    assert "no subject point" in resident(rt)["failures"][-1]


def prop_uid(rt, node):
    return body(rt, node)["uid"]


def test_it_looks_at_what_it_goes_for_then_at_where_it_will_put_it(rt):
    # put_away's steps say where the head goes (look: subject, then a glance with look: place).
    uid = prop_uid(rt, "can")
    to_the_floor(rt, "can")
    looked, turned, doing = [], {}, {}

    def watch():
        r = resident(rt)
        on_it = r["want"]["doing"].get("subject") == uid
        if on_it:
            doing.update(r["want"]["doing"])
            if r["looking_at"] and r["looking_at"] not in looked:
                looked.append(r["looking_at"])
            if r["looking_at"]:
                turned[r["looking_at"]] = max(turned.get(r["looking_at"], 0.0), r["head_turn"])
        return bool(doing) and not on_it and r["prop"] == ""

    run_until(rt, watch, step=0.25)
    assert looked == ["subject", "place"], looked
    assert turned["subject"] > 10 and turned["place"] > 3, turned          # the head really turns (deg)
    assert doing["want"] == "tidy" and doing["place"] and doing["of"] >= 1 and doing["reluctance"] == 0
    r = resident(rt)
    # (Run on its own the pumpkin is still down, and she may be looking at that by now.)
    assert r["looking_at"] == "" or r["want"]["doing"].get("subject", uid) != uid
    assert r["want"]["handled"][uid] == pytest.approx(1.0, abs=0.1)
    rt.call("time.advance", seconds=2, scale=4)
    assert body(rt, "can")["center"][1] > 0.3


def test_reluctance_ends_a_want_that_keeps_being_undone(rt):
    # The resident's tidy has weight 0.5 and reluctance 0.15 a handling: knocked down again and
    # again, the can is worth less each time, and after the fourth it is left where it fell.
    uid = prop_uid(rt, "can")
    option = f"want/tidy/{uid}"
    scores = []

    def handled():
        return resident(rt)["want"]["handled"].get(uid, 0.0)

    for _ in range(6):
        to_the_floor(rt, "can", resume=False)
        offered = {o["id"]: o["score"] for o in rt.call("npcs.options", npc="resident")}
        if option in resident(rt)["want"]["left"]:
            break
        if option in offered:                                              # (not while its hands are full)
            scores.append(offered[option])
        before = handled()
        rt.call("npcs.resume", npc="resident")
        run_until(rt, lambda: handled() > before + 0.5 and resident(rt)["prop"] == "")
        rt.call("time.advance", seconds=2, scale=4)
        assert body(rt, "can")["center"][1] > 0.3                          # put somewhere, each time
    else:
        raise AssertionError(f"still wanted after six more handlings: {scores}")
    assert 3.3 < handled() < 4.1                                           # 4 x 0.15 uses up the 0.5, less a little fading
    assert all(b < a - 0.05 for a, b in zip(scores, scores[1:])), scores
    assert rt.call("npcs.places", npc="resident", want="tidy", prop="can")["resting_on"] == "floor"
    assert not [f for f in resident(rt)["failures"] if f.startswith("want/")]
    rt.call("npcs.resume", npc="resident")
    was = resident(rt)
    rt.call("time.advance", seconds=30, scale=4)
    now = resident(rt)
    assert option in now["want"]["left"] and option not in [o["id"] for o in now["want"]["seen"]]
    assert now["failures"] == was["failures"]                              # it has had enough: nothing failed
    assert body(rt, "can")["center"][1] < 0.3 and body(rt, "can")["holder"] == ""


def carried(rt, node, settled=lambda npc: True):
    """Have the resident carry ``node`` from the floor across the room to the bookshelf; returns
    (its report, the prop) on the way there, once ``settled(report)``."""
    to_the_floor(rt, node, resume=False)
    shelf = next(p for p in rt.call("npcs.places", npc="resident", want="tidy", prop=node)["places"]
                 if p["uid"].endswith("bookshelf"))
    rt.call("npcs.resume", npc="resident")
    rt.call("npcs.do", npc="resident", action="put_away", subject=node, place=shelf["position"])
    npc = run_until(rt, lambda: (lambda r: r["prop"] != "" and "go_to" in r["doing"] and settled(r) and r)(resident(rt)),
                    step=0.1)
    return npc, body(rt, node)


def ahead_and_aside(npc, prop):
    """How far ``prop``'s centre is in front of the NPC and off to its side (m)."""
    yaw = math.radians(npc["yaw"])
    dx, dz = prop["center"][0] - npc["position"][0], prop["center"][2] - npc["position"][2]
    return dx * math.sin(yaw) + dz * math.cos(yaw), dx * math.cos(yaw) - dz * math.sin(yaw)


def test_a_bulky_prop_is_carried_in_front_in_both_hands(rt):
    both = {"LeftHand", "RightHand"}
    # (Walking, the hands are a frame behind what they hold: 2 cm, and 8 cm at this 4x time scale.)
    npc, prop = carried(rt, "pumpkin", lambda r: set(r["hands"]["holding"]) == both and max(r["hands"]["holding"].values()) < 0.1)
    ahead, aside = ahead_and_aside(npc, prop)
    assert ahead > 0.2 and abs(aside) < 0.1, (ahead, aside)                # in front of the body, not at the thigh
    assert prop["center"][1] == pytest.approx(1.0, abs=0.15)
    run_until(rt, lambda: resident(rt)["prop"] == "")
    rt.call("time.advance", seconds=1, scale=1)
    assert resident(rt)["hands"]["holding"] == {}                          # hands off it once it is down
    # The can (12 cm) is a one-hand thing: taken and carried by the motions, in front of the right hip.
    carried(rt, "can")
    rt.call("time.advance", seconds=0.8, scale=1)                          # lifted to where it is carried
    npc, prop = resident(rt), body(rt, "can")
    assert npc["prop"] != "" and npc["hands"]["holding"] == {} and abs(ahead_and_aside(npc, prop)[1]) > 0.05
    run_until(rt, lambda: resident(rt)["prop"] == "")


def test_a_seat_with_something_on_it_is_not_sat_on(rt):
    def offered():
        return [o["id"] for o in rt.call("npcs.options", npc="resident")]

    rt.call("npcs.pause", npc="resident")
    assert "chair#0" in offered()
    to_the_floor(rt, "pumpkin", resume=False)
    seat = next(p for p in rt.call("npcs.places", npc="resident", want="tidy", prop="pumpkin")["places"]
                if p["uid"].endswith("chair"))
    before = resident(rt)["put_down"]
    rt.call("npcs.resume", npc="resident")
    rt.call("npcs.do", npc="resident", action="put_away", subject="pumpkin", place=seat["position"])
    run_until(rt, lambda: resident(rt)["put_down"] == before + 1, step=0.1)
    rt.call("npcs.pause", npc="resident")
    rt.call("time.advance", seconds=2, scale=4)
    assert abs(body(rt, "pumpkin")["center"][1] - seat["height"] - 0.135) < 0.03   # it is on the seat
    assert "chair#0" not in offered()
    to_the_floor(rt, "pumpkin")                                            # cleared, the chair is a seat again
    assert "chair#0" in offered()
