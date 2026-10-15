"""Physics props (geogen.physics), places that detect props, and the test chamber example
(examples/test_chamber, geogen.43): a dropper drops a weighted cube, the player carries it
onto a floor button, and the button holds the exit door open while something is on it."""

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
    scene, _ = chamber
    cube = next(n for n in scene.iter_nodes() if n.name == "cube")
    dropper = next(n for n in scene.iter_nodes() if n.name == "dropper")
    spawn_at = cube.world_transform() @ np.r_[cube.meta["physics"]["spawn_at"], 1.0]
    mouth = dropper.world_transform()[:3, 3]
    assert np.allclose(spawn_at[[0, 2]], mouth[[0, 2]], atol=1e-6)      # on the tube's axis
    assert 0.5 < spawn_at[1] - mouth[1] < 2.0                          # inside it, above the hatch


def test_the_button_holds_the_door_open_while_something_is_on_it(chamber):
    """The rules alone (the runtime counts what is on the pad as `occupants`)."""
    _, quests = chamber
    machine = Machine(quests.program(), {"player": {"kind": "player"}})
    key = lambda k: f"{SCENE}:{k}"
    machine.post({"type": "start", "actor": "player"})
    assert machine.get(key("cube#spawn")) == "live" and machine.get(key("dropper#hatch")) == "open"
    assert machine.get(key("exit_door#door")) == "closed"
    machine.set(key("button/pad#occupants"), 1)
    machine.advance(1.0)
    assert machine.get(key("button#press")) == "down" and machine.get(key("exit_door#door")) == "open"
    assert machine.get(key("line_n#circuit")) == "on" and machine.get(key("line_w#circuit")) == "on"
    machine.set(key("button/pad#occupants"), 0)
    machine.advance(1.0)
    assert machine.get(key("exit_door#door")) == "closed" and machine.get(key("line_n#circuit")) == "off"
    machine.set(key("cube#spawn"), "stowed")              # lost: the dropper hands out another
    assert machine.get(key("cube#spawn")) == "live"


# --- in Godot -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime(godot_runtime, tmp_path_factory):
    from geogen.export import export_scene

    out = tmp_path_factory.mktemp("chamber")
    try:
        roots.use(EXAMPLE)
        registry = SceneRegistry()
        registry.discover()
        export_scene(registry[SCENE](), out / f"{SCENE}.glb")
    finally:
        roots.use()
    client = godot_runtime(SCENE, out)
    client.call("wait_ready", timeout=60)
    return client


def _cube(runtime) -> dict:
    return next(b for b in runtime.call("world.bodies") if b["node"] == "cube")


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
    assert f"{SCENE}:cube" in saved["bodies"]
    # Take the cube off again: the door shuts. Loading puts it back on the button.
    cube = _cube(runtime)["position"]
    runtime.call("player.look", at=[cube[0], cube[1] + 0.35, cube[2]])
    assert runtime.call("player.carry")["held"] == f"{SCENE}:cube"
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
