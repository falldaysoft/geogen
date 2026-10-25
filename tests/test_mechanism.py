"""Mechanisms (geogen.mechanism, geogen.86): an asset's parts as jointed rigid bodies with
motors, `drive` interaction motions, polar part arrays and affordances that ride a body."""

import numpy as np
import pytest

from geogen.export import exported_extras
from geogen.extras import schema
from geogen.layout.loader import LayoutLoader

# A turntable: a deck on a motorised hinge round a post, four arms, and a cab hung from
# each arm's end on a pin (one body per copy).
TURNTABLE = """
name: turntable
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
params:
  speed: {default: 20}
mechanism:
  bodies:
    table:
      parts: [deck, arm]
      mass: 500
      damping: {angular: 0.1}
      joint: {type: hinge, axis: y, pivot: [0, 0, 0], limits: [-90, 90], motor: {speed: "{speed}", force: 5000}}
    cab:
      parts: [cab_box]
      array: true
      mass: 40
      joint: {type: pin, to: table, pivot: cab_box.top}
parts:
  post: {primitive: cylinder, size: [0.2, 2.0, 0.2], anchor: bottom_center, material: metal}
  deck: {primitive: cylinder, size: [2.0, 0.2, 2.0], anchor: bottom_center, offset: [0, 0.3, 0], material: wood, collider: mesh}
  arm:
    primitive: cube
    size: [1.0, 0.1, 0.1]
    anchor: bottom_center
    offset: [1.4, 1.8, 0]
    array: {count: 4, turn: 90}
    material: metal
  cab_box:
    primitive: cube
    size: [0.3, 0.3, 0.3]
    anchor: bottom_center
    offset: [1.8, 1.4, 0]
    array: {count: 4, turn: 90}
    material: wood
  lever: {primitive: cylinder, size: [0.04, 0.5, 0.04], anchor: bottom_center, offset: [1.6, 0, 0], material: metal}
interactions:
  run:
    targets: [lever]
    initial: stopped
    duration: 4
    states:
      stopped: {prompt: Start, next: running}
      running: {prompt: Stop, next: stopped}
      coasting: {then: stopped, after: 5}
    motions:
      - {body: table, drive: {running: "{speed}"}, free: [coasting]}
affordances:
  - {type: sit, at: [1.0, 0.5, 0], facing: 90, body: table}
"""


def _load(text: str, **params):
    return LayoutLoader().load_string(text, params=params or None)


def test_a_mechanism_makes_bodies_of_parts_with_joints_and_motors():
    root = _load(TURNTABLE)
    bodies = root.meta["mechanism"]["bodies"]
    assert set(bodies) == {"table", "cab_0", "cab_1", "cab_2", "cab_3"}
    table = bodies["table"]
    assert table["nodes"] == ["deck", "arm_0", "arm_1", "arm_2", "arm_3"]
    assert table["mass"] == 500 and table["damping"] == {"linear": 0.0, "angular": 0.1}
    joint = table["joint"]
    assert joint["type"] == "hinge" and joint["axis"] == [0, 1, 0] and joint["limits"] == [-90, 90]
    assert joint["motor"] == {"speed": 20, "force": 5000}
    # A dynamic body can't be concave: the deck's mesh collider becomes a hull.
    deck = next(n for n in root.iter_nodes() if n.name == "deck")
    assert deck.meta["collider"] == "hull"
    # One cab body per copy, each pinned at its own box's top (the copies sit 90 degrees apart).
    cabs = [bodies[f"cab_{i}"] for i in range(4)]
    assert [c["nodes"] for c in cabs] == [[f"cab_box_{i}"] for i in range(4)]
    assert all(c["joint"]["type"] == "pin" and c["joint"]["to"] == "table" for c in cabs)
    pivots = np.array([c["joint"]["pivot"] for c in cabs])
    assert pivots[:, 1] == pytest.approx([1.7] * 4)
    assert pivots[0, [0, 2]] == pytest.approx([1.8, 0])
    assert pivots[1, [0, 2]] == pytest.approx([0, -1.8])


def test_a_drive_motion_sets_a_motor_speed_per_state():
    root = _load(TURNTABLE, speed=30)
    run = next(i for i in root.interactions if i.name == "run")
    drive = run.motions[0]
    assert drive.kind == "drive" and drive.body == "table" and drive.free == ["coasting"] and not drive.parts
    assert drive.values == {"running": 30.0}
    assert run.value(drive, "stopped") == 0.0            # a missing state brakes
    assert root.meta["mechanism"]["bodies"]["table"]["joint"]["motor"]["speed"] == 30


def test_extras_carry_the_mechanism_and_drive_by_exported_name():
    pytest.importorskip("jsonschema")
    import jsonschema

    root = _load(TURNTABLE)
    extras = exported_extras(root, name_of=lambda n: f"x_{n.name}")["geogen"]
    assert extras["mechanism"]["bodies"]["table"]["nodes"][:2] == ["x_deck", "x_arm_0"]
    motion = extras["interactions"]["run"]["motions"][0]
    assert motion == {"nodes": [], "type": "drive", "body": "table", "free": ["coasting"],
                      "values": {"stopped": 0.0, "running": 20.0, "coasting": 0.0}}
    assert extras["affordances"][0]["body"] == "table"
    jsonschema.validate({"geogen": extras}, schema())


def test_mechanism_mistakes_are_reported():
    with pytest.raises(ValueError, match="unknown keys"):
        _load(TURNTABLE.replace("mass: 500", "weight: 500"))
    with pytest.raises(ValueError, match="pin joint has no axis"):
        _load(TURNTABLE.replace("{type: pin, to: table,", "{type: pin, axis: y, to: table,"))
    with pytest.raises(ValueError, match="no hinge or slide joint"):
        _load(TURNTABLE.replace("{body: table, drive:", "{body: cab_0, drive:"))
    with pytest.raises(ValueError, match="unknown body"):
        _load(TURNTABLE.replace("{body: table, drive:", "{body: wheel, drive:"))
    with pytest.raises(ValueError, match="unknown part"):
        _load(TURNTABLE.replace("parts: [deck, arm]", "parts: [deck, rim]"))
    with pytest.raises(ValueError, match="every part arrayed"):
        _load(TURNTABLE.replace("parts: [cab_box]", "parts: [cab_box, deck]"))
    with pytest.raises(ValueError, match="joined to unknown body"):
        _load(TURNTABLE.replace("to: table, pivot", "to: base, pivot"))
    with pytest.raises(ValueError, match="needs a mechanism"):
        _load(TURNTABLE.replace("mechanism:\n", "mechanism_off:\n"))
    with pytest.raises(ValueError, match="not a mechanism body"):
        _load(TURNTABLE.replace("body: table}", "body: post}"))


ARRAYS = """
name: ring
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
parts:
  peg:
    primitive: cube
    size: [0.4, 0.2, 0.1]
    anchor: bottom_center
    offset: [2.0, 0.5, 0]
    rotation: [0, 10, 0]
    array: {count: 4, turn: 90}
    material: wood
  spoke:
    primitive: cube
    size: [0.1, 1.0, 0.1]
    anchor: center
    offset: [1.0, 2.5, 0]
    array: {count: 4, turn: 90, axis: z, about: [0, 3, 0]}
    material: metal
  floor:
    primitive: cube
    size: [0.6, 0.1, 0.6]
    anchor: bottom_center
    offset: [0, 1.0, 0]
    array: {count: 4, turn: 90, axis: z, about: [0, 3, 0], upright: [0, 2.0, 0]}
    material: wood
  roof:
    primitive: cube
    size: [0.6, 0.05, 0.6]
    anchor: bottom_center
    offset: [0, 1.8, 0]
    array: {count: 4, turn: 90, axis: z, about: [0, 3, 0], upright: [0, 2.0, 0]}
    material: wood
"""


def _part(root, name):
    return next(n for n in root.iter_nodes() if n.name == name)


def test_a_polar_array_swings_copies_round_an_axis():
    root = _load(ARRAYS)
    # About y (the default): each peg a quarter turn on, its own yaw growing with it.
    p0, p1, p2 = (_part(root, f"peg_{i}").transform for i in (0, 1, 2))
    assert p0.translation == pytest.approx([2.0, 0.6, 0])
    assert p1.translation == pytest.approx([0, 0.6, -2.0], abs=1e-9)
    assert p2.translation == pytest.approx([-2.0, 0.6, 0], abs=1e-9)
    assert np.degrees([p0.rotation[1], p1.rotation[1], p2.rotation[1]]) == pytest.approx([10, 100, 190])
    # About z through a point: spokes turn in the XY plane round (0, 3).
    s0, s1 = _part(root, "spoke_0").transform, _part(root, "spoke_1").transform
    assert s0.translation == pytest.approx([1.0, 3.0, 0])
    assert s1.translation == pytest.approx([0, 4.0, 0], abs=1e-9)
    assert np.degrees(s1.rotation[2]) == pytest.approx(90)


def test_an_upright_point_array_keeps_each_copy_assembled():
    root = _load(ARRAYS)
    for i in range(4):
        floor, roof = _part(root, f"floor_{i}").transform, _part(root, f"roof_{i}").transform
        assert roof.translation - floor.translation == pytest.approx([0, 0.775, 0], abs=1e-9)   # as written
        assert floor.rotation == pytest.approx([0, 0, 0])
    # The carried point (0, 2) goes round (0, 3): copy 1 hangs from (1, 3), copy 2 from (0, 4).
    assert _part(root, "floor_1").transform.translation == pytest.approx([1.0, 2.05, 0], abs=1e-9)
    assert _part(root, "floor_2").transform.translation == pytest.approx([0, 3.05, 0], abs=1e-9)


# ---- Godot: the core review scene ----------------------------------------------------------------

@pytest.fixture(scope="module")
def mechanisms(godot_runtime, scene_export):
    client = godot_runtime("mechanism_test", scene_export("mechanism_test"))
    client.call("wait_ready", timeout=90)
    return client


def _bodies(rt, asset: str) -> dict:
    return {b["body"]: b for b in rt.call("world.bodies") if b["uid"].startswith(f"mechanism_test:{asset}/")}


def test_the_review_scene_s_mechanisms_move_as_physics_says_in_godot(mechanisms):
    """A gate on a limited hinge swings open when walked into; a seesaw tips under a prop; a
    chain of pinned links hangs straight; a motorised slide lifts a rider; sails turn."""
    rt = mechanisms
    rt.call("time.advance", seconds=3, scale=4)
    assert _bodies(rt, "seesaw")["plank"]["tilt_deg"] > 12                      # the pumpkin's end is down
    chain = _bodies(rt, "chain")
    assert max(b["tilt_deg"] for b in chain.values()) < 3                         # hanging straight
    assert chain["lantern"]["position"][1] < chain["link_0"]["position"][1] - 1.2
    sails = _bodies(rt, "windmill")["sails"]
    assert sails["angular_speed"] == pytest.approx(20.0, abs=1.0) and sails["motor"]["on"]
    # Walk north into the gate: it swings away (a right-handed turn about +Y, toward +Z) and
    # stops at its limit, not through it.
    rt.call("player.teleport", pos=[0, 0.1, -6])
    rt.call("player.look", at=[0, 1.2, -4])
    for _ in range(30):
        if rt.call("player.move", forward=1, seconds=0.2)["position"][2] > -2.5:
            break
    assert rt.call("player.status")["position"][2] > -2.5                        # through
    rt.call("time.advance", seconds=6, scale=4)
    yaw = _bodies(rt, "gate")["leaf"]["yaw_deg"]
    assert -115 < yaw < -60, yaw
    # Ride the lift: stand on the deck, use the lever, go up 4 m, and back.
    rt.call("player.teleport", pos=[7.85, 0.3, 2])
    rt.call("time.advance", seconds=0.5, scale=4)
    rt.call("interactions.use", asset="lift", interaction="ride")
    rt.call("time.advance", seconds=6, scale=4)
    assert rt.call("player.status")["position"][1] == pytest.approx(4.2, abs=0.15)
    assert rt.call("interactions.list", asset="lift")[0]["state"] == "up"
    rt.call("interactions.use", asset="lift", interaction="ride")
    rt.call("time.advance", seconds=6, scale=4)
    assert rt.call("player.status")["position"][1] == pytest.approx(0.2, abs=0.15)


def test_array_mistakes_are_reported():
    with pytest.raises(ValueError, match="unknown keys"):
        _load(ARRAYS.replace("turn: 90}", "turn: 90, spin: 1}"))
    with pytest.raises(ValueError, match="axis is x, y or z"):
        _load(ARRAYS.replace("axis: z, about: [0, 3, 0]}", "axis: w, about: [0, 3, 0]}"))
    with pytest.raises(ValueError, match="later axis"):
        _load(ARRAYS.replace("rotation: [0, 10, 0]", "rotation: [0, 10, 5]"))
