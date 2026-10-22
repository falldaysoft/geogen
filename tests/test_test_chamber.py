"""Physics props (geogen.physics), places that detect props, and the test chamber example
(examples/test_chamber, geogen.43): a dropper drops a weighted cube, the player carries it
onto a floor button, and the button holds the exit door open while something is on it.
The later chambers (geogen.83) add goo pits that send the player back to the start, a
moving platform that carries them, retractable stairs to a raised exit, a timed pedestal
button and a fizzler that dissolves cubes."""

from pathlib import Path

import numpy as np
import pytest

from geogen import roots
from geogen.layout.loader import LayoutLoader
from geogen.quests import collect
from geogen.registry import SceneRegistry
from geogen.rules import Machine

EXAMPLE = Path(__file__).parent.parent / "examples" / "test_chamber"
SCENE = "test_chamber_01"
PROP = """
name: crate
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
physics: {mass: 12, carry: true, spawn: "{from}"}
params:
  from: {default: "", type: text}
parts:
  box: {primitive: cube, size: [0.5, 0.4, 0.6], anchor: bottom_center, material: wood}
  lid: {primitive: cube, size: [0.5, 0.05, 0.6], anchor: bottom_center, offset: [0, 0.4, 0], material: wood}
places:
  scale: {box: [1, 1, 1], detect: [actors, props]}
"""


def test_a_physics_block_makes_a_prop_with_a_box_from_its_geometry():
    crate = LayoutLoader().load_string(PROP.replace('"{from}"', '""'))
    physics = crate.meta["physics"]
    assert physics["mass"] == 12 and physics["carry"] and "spawn" not in physics
    assert physics["size"] == pytest.approx([0.5, 0.45, 0.6], abs=1e-3)
    assert physics["center"] == pytest.approx([0, 0.225, 0], abs=1e-3)
    assert not crate.interactions                            # no spawn: nothing hands it out
    place = next(c for c in crate.children if c.name == "scale")
    assert place.meta["place"]["detect"] == ["actors", "props"]


def test_a_spawned_prop_starts_stowed_and_hidden():
    crate = LayoutLoader().load_string(PROP, params={"from": "chute.mouth"})
    spawn = next(i for i in crate.interactions if i.name == "spawn")
    assert spawn.initial == "stowed" and set(spawn.states) == {"stowed", "live"}
    assert all(part.hidden for part in spawn.motions[0].parts)


def test_physics_keys_are_checked():
    with pytest.raises(ValueError, match="unknown keys"):
        LayoutLoader().load_string(PROP.replace("mass: 12", "weight: 12"))
    with pytest.raises(ValueError, match="detect"):
        LayoutLoader().load_string(PROP.replace("[actors, props]", "[cars]"))


@pytest.fixture(scope="module")
def chamber():
    try:
        roots.use(EXAMPLE)
        registry = SceneRegistry()
        registry.discover()
        scene = registry[SCENE]()
        yield scene, collect(scene, SCENE)
    finally:
        roots.use()


def test_the_cube_is_handed_out_inside_the_dropper(chamber):
    """The dropper is a source of weighted cubes (geogen.flows), one at a time."""
    scene, _ = chamber
    dropper = next(n for n in scene.iter_nodes() if n.name == "dropper")
    source = dropper.meta["flows"]["source"]
    assert source["max_alive"] == 1 and source["items"][0]["name"] == "weighted_cube"
    template = next(n for n in dropper.iter_nodes() if "item" in n.meta)
    assert template.hidden and template.meta["item"]["stuck"] == 0 and template.meta["physics"]["carry"]
    spawn_at = dropper.world_transform() @ np.r_[source["at"], 1.0]
    mouth = dropper.world_transform()[:3, 3]
    assert np.allclose(spawn_at[[0, 2]], mouth[[0, 2]], atol=1e-6)      # on the tube's axis
    assert 0.5 < spawn_at[1] - mouth[1] < 2.3                          # inside it, above the hatch


def test_the_button_holds_the_door_open_while_something_is_on_it(chamber):
    """The rules alone (the runtime counts what is on the pad as `occupants`)."""
    _, quests = chamber
    machine = Machine(quests.program(), {"player": {"kind": "player"}})
    key = lambda k: f"{SCENE}:{k}"
    machine.post({"type": "start", "actor": "player"})
    machine.post({"type": "signal", "subject": key("dropper"), "name": "emitted", "data": {"item": "weighted_cube"}})
    assert machine.get(key("dropper#hatch")) == "open"                  # a cube came: the hatch lets it out
    assert machine.get(key("exit_door#door")) == "closed"
    machine.set(key("button/pad#occupants"), 1)
    machine.advance(1.0)
    assert machine.get(key("button#press")) == "down" and machine.get(key("exit_door#door")) == "open"
    assert machine.get(key("line_n#circuit")) == "on" and machine.get(key("line_w#circuit")) == "on"
    machine.set(key("button/pad#occupants"), 0)
    machine.advance(1.0)
    assert machine.get(key("exit_door#door")) == "closed" and machine.get(key("line_n#circuit")) == "off"


# --- in Godot -----------------------------------------------------------------------------------


def _runtime(godot_runtime, tmp_path_factory, scene: str):
    from geogen.export import export_scene

    out = tmp_path_factory.mktemp("chamber")
    try:
        roots.use(EXAMPLE)
        registry = SceneRegistry()
        registry.discover()
        export_scene(registry[scene](), out / f"{scene}.glb")
    finally:
        roots.use()
    client = godot_runtime(scene, out)
    client.call("wait_ready", timeout=60)
    return client


@pytest.fixture(scope="module")
def runtime(godot_runtime, tmp_path_factory):
    return _runtime(godot_runtime, tmp_path_factory, SCENE)


@pytest.fixture(scope="module")
def chamber_02(godot_runtime, tmp_path_factory):
    return _runtime(godot_runtime, tmp_path_factory, "test_chamber_02")


@pytest.fixture(scope="module")
def chamber_03(godot_runtime, tmp_path_factory):
    return _runtime(godot_runtime, tmp_path_factory, "test_chamber_03")


def _cube(runtime) -> dict:
    """The dropper's cube (it may still be on its way: whichever test runs first waits for it)."""
    for _ in range(10):
        found = [b for b in runtime.call("world.bodies") if b["uid"].startswith("flow#")]
        if found and (found[0]["held"] or found[0]["speed"] < 0.05):
            return found[0]
        runtime.call("time.advance", seconds=0.5, scale=1)
    raise AssertionError(f"the dropper's cube never landed: {runtime.call('world.bodies')}")


def _get(runtime, k: str):
    return runtime.call("state.get", key=f"{SCENE}:{k}")["value"]


def _settle(runtime, seconds: float):
    runtime.call("time.advance", seconds=seconds, scale=1)


def _carry_to_button(runtime):
    cube = _cube(runtime)["position"]
    runtime.call("player.walk_to", pos=[cube[0], 0, cube[2] + 1.0])
    runtime.call("player.look", at=[cube[0], cube[1] + 0.35, cube[2]])
    assert runtime.call("player.use")["kind"] == "carry"
    assert _cube(runtime)["held"]
    runtime.call("player.walk_to", pos=[2.4, 0, 0.3])
    runtime.call("player.look", at=[2.4, 0.8, 1.8])
    _settle(runtime, 1.0)
    assert runtime.call("player.use")["kind"] == "drop"
    _settle(runtime, 1.5)


def test_the_dropper_drops_the_cube(runtime):
    _settle(runtime, 3.0)
    cube = _cube(runtime)
    assert cube["live"] and abs(cube["position"][1]) < 0.05           # landed on the floor
    assert np.hypot(cube["position"][0] + 3.6, cube["position"][2] + 1.8) < 0.3   # under the dropper


def test_standing_on_the_button_opens_the_door_only_while_there(runtime):
    """Walked onto, not put there: its edge ramps up within the player's step and slope."""
    runtime.call("player.teleport", pos=[2.4, 0, -0.5])
    runtime.call("player.look", at=[2.4, 1.65, 1.8])
    for _ in range(40):                                  # walk at it until over the cap
        if runtime.call("player.move", forward=1, seconds=0.1)["position"][2] > 1.6:
            break
    _settle(runtime, 0.5)
    assert runtime.call("player.status")["position"][1] > 0.12             # up on the cap
    assert _get(runtime, "button/pad#occupants") == 1 and _get(runtime, "exit_door#door") == "open"
    runtime.call("player.teleport", pos=[2.4, 0, -0.5])
    _settle(runtime, 0.5)
    assert _get(runtime, "button/pad#occupants") == 0 and _get(runtime, "exit_door#door") == "closed"


def test_the_cube_holds_the_button_down_and_the_player_leaves(runtime):
    _carry_to_button(runtime)
    assert not _cube(runtime)["held"]
    assert _get(runtime, "button/pad#occupants") == 1 and _get(runtime, "exit_door#door") == "open"
    saved = runtime.call("state.save")
    assert [it["item"] for it in saved["flows"]["items"]] == ["weighted_cube"]
    # Take the cube off again: the door shuts. Loading puts it back on the button.
    cube = _cube(runtime)["position"]
    runtime.call("player.look", at=[cube[0], cube[1] + 0.35, cube[2]])
    assert runtime.call("player.carry")["held"] == _cube(runtime)["uid"]
    runtime.call("player.walk_to", pos=[0, 0, -2])
    runtime.call("player.drop")
    _settle(runtime, 1.5)
    assert _get(runtime, "exit_door#door") == "closed"
    runtime.call("state.load", state=saved)
    _settle(runtime, 1.5)
    assert np.hypot(_cube(runtime)["position"][0] - 2.4, _cube(runtime)["position"][2] - 1.8) < 0.6
    assert _get(runtime, "button/pad#occupants") == 1 and _get(runtime, "exit_door#door") == "open"
    walked = runtime.call("player.walk_to", pos=[0, 0, 10])
    assert walked["blocked_by"] is None, walked
    _settle(runtime, 0.5)
    shown = runtime.call("hud.notifications")
    assert any(t["text"] == "Test chamber 01 complete" for t in shown["showing"]), shown


# --- the later chambers -------------------------------------------------------------------------


def _state(runtime, scene: str, k: str):
    return runtime.call("state.get", key=f"{scene}:{k}")["value"]


def _ride_back_at_near(runtime, scene: str, span: float):
    """Wait until the platform's deck has just come back to its near end (it dwells there).
    `ride` holds the state it is heading to: it flips to `near` as the deck leaves the far end."""
    for _ in range(60):
        if _state(runtime, scene, "platform#ride") == "far":
            break
        _settle(runtime, 0.25)
    for _ in range(60):
        if _state(runtime, scene, "platform#ride") == "near":
            break
        _settle(runtime, 0.25)
    _settle(runtime, span + 0.1)      # heading near at 1 m/s: there, with most of the pause left


def test_the_platform_carries_the_player_over_the_goo(chamber_02):
    """Stand on the deck and ride: the AnimatableBody3D under a moving part takes the player
    along (platform velocity), from the pit's south edge to its north."""
    rt, scene = chamber_02, "test_chamber_02"
    _ride_back_at_near(rt, scene, span=3.2)
    rt.call("player.teleport", pos=[0, 0.05, -1.6])
    _settle(rt, 0.5)
    assert abs(rt.call("player.status")["position"][2] + 1.6) < 0.3        # standing on the deck
    _settle(rt, 4.5)                                                        # the rest of the pause, then across
    z = rt.call("player.status")["position"][2]
    assert z > 0.8, z                                                       # carried to the far side


def test_the_goo_sends_the_player_back_to_the_start(chamber_02):
    rt, scene = chamber_02, "test_chamber_02"
    rt.call("player.teleport", pos=[3.0, 0.05, 0.0])                        # over the pit, off the deck
    _settle(rt, 3.0)
    p = rt.call("player.status")["position"]
    assert abs(p[0]) < 0.1 and abs(p[2] + 9.6) < 0.1, p                     # at the `start` spawn
    shown = rt.call("hud.notifications")["showing"]
    assert any(t["text"] == "Dissolved. Back to the start." for t in shown), shown


def test_the_pedestal_button_opens_the_exit_door(chamber_02):
    rt, scene = chamber_02, "test_chamber_02"
    rt.call("player.teleport", pos=[2.4, 0.05, 3.3])
    rt.call("player.look", at=[2.4, 1.05, 4.2])
    assert rt.call("player.use")["kind"] == "interaction"
    _settle(rt, 1.5)
    assert _state(rt, scene, "button#press") == "up" and _state(rt, scene, "exit_door#door") == "open"


def test_a_cube_on_the_button_raises_the_stairs_to_the_exit(chamber_03):
    """The cube ferried over on the platform holds the button; the stair blocks rise out of
    the floor, each an AnimatableBody3D, and the player climbs them to the raised exit."""
    rt, scene = chamber_03, "test_chamber_03"
    assert _state(rt, scene, "stairs#deploy") == "retracted"
    cube = _cube(rt)["position"]
    rt.call("player.walk_to", pos=[cube[0], 0, cube[2] + 1.0])
    rt.call("player.look", at=[cube[0], cube[1] + 0.35, cube[2]])
    assert rt.call("player.use")["kind"] == "carry" and _cube(rt)["held"]
    rt.call("player.walk_to", pos=[0, 0, -4.4])                             # the pit's south edge
    _ride_back_at_near(rt, scene, span=2.0)
    rt.call("player.look", at=[0, 1.65, 5.0])
    rt.call("player.move", forward=1, seconds=0.6)                          # onto the deck
    assert _cube(rt)["held"] and abs(rt.call("player.status")["position"][2] + 2.2) < 0.4
    _settle(rt, 4.5)                                                        # the rest of the pause, then across
    rt.call("player.move", forward=1, seconds=0.8)                          # and off, on the far side
    assert rt.call("player.status")["position"][2] > 0.5 and _cube(rt)["held"]
    rt.call("player.walk_to", pos=[3.0, 0, -0.5])
    rt.call("player.look", at=[3.0, 0.8, 1.0])
    _settle(rt, 0.5)
    assert rt.call("player.use")["kind"] == "drop"
    _settle(rt, 2.5)
    assert _state(rt, scene, "button/pad#occupants") >= 1
    assert _state(rt, scene, "stairs#deploy") == "extended" and _state(rt, scene, "exit_door#door") == "open"
    rt.call("player.teleport", pos=[0, 0.05, 0.0])
    rt.call("player.look", at=[0, 1.65, 6.0])
    for _ in range(5):
        rt.call("player.move", forward=1, seconds=0.7)
    p = rt.call("player.status")["position"]
    assert p[1] > 2.9 and p[2] > 4.0, p                                     # up on the landing
    walked = rt.call("player.walk_to", pos=[0, 3.0, 9.0])
    assert walked["blocked_by"] is None, walked
