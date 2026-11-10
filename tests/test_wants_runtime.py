"""Wants in the Godot runtime (geogen.97): NPCs carrying physics props, on the wants_test scene."""


import pytest

pytestmark = [pytest.mark.godot, pytest.mark.xdist_group("test_wants_runtime")]


@pytest.fixture(scope="module")
def rt(scene_export, godot_runtime):
    client = godot_runtime("wants_test", scene_export("wants_test"))
    client.call("wait_ready")
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
