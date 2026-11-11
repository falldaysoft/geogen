"""Wants about the state of things in the Godot runtime (geogen.97.5), on the wants_house scene:
Hana puts out the light of a room nobody is in and shuts doors left standing open."""


import pytest

pytestmark = [pytest.mark.godot, pytest.mark.xdist_group("test_wants_state_runtime")]

BEDROOM_SWITCH = "wants_house:flat/bedroom/bedroom_switch_1#switch"
DOOR = "wants_house:flat/door_living_bedroom#swing"
LIVING = [-2.4, 0.05, 1.0]
BEDROOM = [2.2, 0.05, 0.0]


@pytest.fixture(scope="module")
def rt(scene_export, godot_runtime):
    client = godot_runtime("wants_house", scene_export("wants_house"))
    client.call("wait_ready")
    client.call("time.pause")           # the world runs only inside time.advance
    return client


def hana(rt):
    return next(n for n in rt.call("npcs.list") if n["npc"].startswith("hana"))


def want_failures(rt):
    return [f for f in hana(rt)["failures"] if f.startswith("want/")]


def state(rt, key):
    return rt.call("interactions.states")[key]["state"]


def offered(rt):
    return [o["id"] for o in rt.call("npcs.options", npc="hana")]


def in_the_living_room(rt):
    """Let Hana come back to the living room (she counts as someone in a room too), and hold her there."""
    def there():
        at = hana(rt)["position"]
        return rt.call("world.room_at", pos=[at[0], 1.0, at[2]])["room"] == "living" and hana(rt)["pose"] != "stand"
    rt.call("npcs.resume", npc="hana")
    run_until(rt, there)                                                   # sat down in her armchair
    rt.call("npcs.pause", npc="hana")


def run_until(rt, done, seconds=120.0, step=0.5):
    spent = 0.0
    while spent < seconds:
        rt.call("time.advance", seconds=step, scale=4)
        spent += step
        got = done()
        if got:
            return got
    raise AssertionError(f"not within {seconds} s: {hana(rt)['doing']}, failures {hana(rt)['failures']}, "
                         f"want {hana(rt)['want']}, options {rt.call('npcs.options', npc='hana')[:4]}")


def test_the_scene_names_the_things_her_wants_are_about(built_scene):
    node = next(n for n in built_scene("wants_house").iter_nodes() if n.name == "hana")
    found = {name: sorted((thing.name, which) for thing, which in subjects)
             for name, subjects in node.meta["npc_want_subjects"].items()}
    # Every light switch, and the doors themselves (not their leaves, which carry no interaction).
    assert found["lights_out"] == [("bedroom_switch_1", "switch"), ("living_switch_1", "switch"), ("living_switch_2", "switch")]
    assert found["doors_shut"] == [("door_living_bedroom", "swing"), ("front_door", "swing")]


def test_she_puts_out_the_light_of_the_empty_room(rt):
    # She and the player start in the living room; nobody is in the bedroom, and its light is on.
    # (She may be on her way already: the world runs while the runtime gets ready.)
    run_until(rt, lambda: state(rt, BEDROOM_SWITCH) == "off" and hana(rt)["wants"] >= 1)
    states = rt.call("interactions.states")
    assert [k for k, v in states.items() if "living_switch" in k and v["state"] != "on"] == []   # not the room they are in
    run_until(rt, lambda: state(rt, DOOR) == "closed")                     # shut behind her, as she does
    assert not want_failures(rt)


def test_a_room_with_someone_in_it_keeps_its_light(rt):
    in_the_living_room(rt)
    rt.call("player.teleport", pos=BEDROOM, yaw=0)
    rt.call("interactions.set", asset="bedroom_switch_1", state="on", instant=True)
    rt.call("time.advance", seconds=0.2, scale=1)
    option = f"want/lights_out/{BEDROOM_SWITCH}"
    assert option not in offered(rt)                                       # the player is in there
    rt.call("player.teleport", pos=LIVING, yaw=90)
    rt.call("time.advance", seconds=0.2, scale=1)
    assert option in offered(rt)
    rt.call("npcs.resume", npc="hana")


def test_she_shuts_a_door_left_open(rt):
    in_the_living_room(rt)
    rt.call("interactions.set", asset="door_living_bedroom", state="open", instant=True)
    rt.call("time.advance", seconds=0.2, scale=1)
    assert f"want/doors_shut/{DOOR}" in offered(rt)
    before = hana(rt)["wants"]
    rt.call("npcs.resume", npc="hana")
    run_until(rt, lambda: state(rt, DOOR) == "closed" and hana(rt)["wants"] > before and hana(rt)["want"]["doing"] == {})
    assert not want_failures(rt)


def test_a_light_put_on_again_and_again_is_left_on(rt):
    # lights_out has weight 0.5 and reluctance 0.2: 2.5 handlings use it up (they fade over ten
    # minutes, and she sits a while between them here, so it takes three or four).
    option = f"want/lights_out/{BEDROOM_SWITCH}"
    for _ in range(5):
        in_the_living_room(rt)
        rt.call("interactions.set", asset="bedroom_switch_1", state="on", instant=True)
        rt.call("time.advance", seconds=0.2, scale=1)
        ids = offered(rt)
        if option in hana(rt)["want"]["left"]:
            break
        rt.call("npcs.resume", npc="hana")
        run_until(rt, lambda: state(rt, BEDROOM_SWITCH) == "off" and hana(rt)["want"]["doing"] == {})
    else:
        raise AssertionError(f"still wanted after five more: {hana(rt)['want']}")
    assert option not in ids
    assert 2.5 <= hana(rt)["want"]["handled"][BEDROOM_SWITCH] < 3.5
    rt.call("npcs.resume", npc="hana")
    rt.call("time.advance", seconds=20, scale=4)
    assert state(rt, BEDROOM_SWITCH) == "on" and not want_failures(rt)  # she has had enough: nothing failed
