"""The qualities of places in the Godot runtime (geogen.97.6), on the wants_places scene: what a
want weighs when it picks where a thing goes (high, enclosed, hidden, warm, soft)."""


import pytest

pytestmark = [pytest.mark.godot, pytest.mark.xdist_group("test_wants_places_runtime")]


@pytest.fixture(scope="module")
def rt(scene_export, godot_runtime):
    client = godot_runtime("wants_places", scene_export("wants_places"), "--time=13:00")
    client.call("wait_ready")
    client.call("time.pause")           # the world runs only inside time.advance
    return client


def tomas(rt):
    return next(n for n in rt.call("npcs.list") if n["npc"].startswith("tomas"))


def can(rt):
    return next(b for b in rt.call("world.bodies") if b.get("node") == "can")


def places(rt):
    return rt.call("npcs.places", npc="tomas", want="tidy", prop="can")["places"]


def on(found, thing):
    return [p for p in found if p["uid"].endswith(thing)]


@pytest.fixture(scope="module")
def found(rt):
    """The places for the can, with Tomas held where he starts and the can on the open floor."""
    rt.call("npcs.pause", npc="tomas")
    if can(rt)["holder"] or can(rt)["center"][1] > 0.3:                    # (he was quick: put it back down)
        rt.call("player.teleport", pos=[1.4, 0.05, 1.4], yaw=90)
        rt.call("player.carry", prop="can")
        rt.call("time.advance", seconds=1, scale=4)
        rt.call("player.drop")
        rt.call("time.advance", seconds=2, scale=4)
    return places(rt)


def test_places_have_qualities_worked_out_from_the_world(found):
    assert all(set(p["qualities"]) >= {"high", "enclosed", "hidden", "warm", "soft"} for p in found)
    assert all(0.0 <= v <= 1.0 for p in found for v in p["qualities"].values())
    shelves, table, sofa = on(found, "bookshelf"), on(found, "table"), on(found, "sofa")
    assert shelves and table and sofa
    # High is height above the floor: a spot's own, whatever it is on.
    for p in found:
        assert p["qualities"]["high"] == pytest.approx(min(p["height"] / 2.0, 1.0), abs=0.02)
    # Between the shelves is shut in (sides, back and the shelf above); the table top is open.
    assert max(p["qualities"]["enclosed"] for p in shelves) >= 0.5
    assert max(p["qualities"]["enclosed"] for p in table) <= 0.25
    # Soft is the sofa's tag; warm is the lit lamp's, fading with distance over the table.
    assert all(p["qualities"]["soft"] == 1.0 for p in sofa)
    assert all(p["qualities"]["soft"] == 0.0 for p in table + shelves)
    assert max(p["qualities"]["warm"] for p in table) > 0.5
    assert all(p["qualities"]["warm"] < 0.71 for p in shelves)             # at most the sun's, across the room


def test_a_want_weighs_them(found):
    # Tomas's tidy: quality {high: 1.0, enclosed: 0.5}, and no liking for any height or thing.
    best = found[0]
    assert best["qualities"]["high"] >= max(p["qualities"]["high"] for p in found) - 0.15
    assert best["height"] > 1.0
    assert found[0]["score"] > found[-1]["score"]
    low = min(found, key=lambda p: p["height"])
    assert best["score"] - low["score"] > 0.3


def test_he_puts_the_can_up_out_of_the_way(rt, found):
    before = tomas(rt)["wants"]
    rt.call("npcs.resume", npc="tomas")
    rt.call("time.advance", seconds=1, scale=4)
    failed = len(tomas(rt)["failures"])                                    # (the fixture may have taken it out of his hands)
    spent = 0.0
    while tomas(rt)["wants"] == before and spent < 120:
        rt.call("time.advance", seconds=0.5, scale=4)
        spent += 0.5
    rt.call("npcs.pause", npc="tomas")
    rt.call("time.advance", seconds=2, scale=4)
    assert tomas(rt)["wants"] == before + 1, (tomas(rt)["doing"], tomas(rt)["failures"])
    assert can(rt)["holder"] == "" and can(rt)["center"][1] > 1.0          # nobody said "the bookshelf"
    assert not [f for f in tomas(rt)["failures"][failed:] if f.startswith("want/")]


def test_the_sun_warms_only_by_day(rt, found):
    # (The bookshelf is across the room from the lamp: only the sun could warm it.)
    day = max(p["qualities"]["warm"] for p in on(found, "bookshelf"))
    rt.call("time.clock", time="23:00")
    rt.call("time.advance", seconds=0.2, scale=1)
    night = places(rt)
    assert on(night, "bookshelf") and all(p["qualities"]["warm"] == 0.0 for p in on(night, "bookshelf"))
    assert max(p["qualities"]["warm"] for p in on(night, "table")) > 0.5   # the lamp is warm at any hour
    assert day in (0.0, 0.7)                                               # in the sun or not: nothing between
