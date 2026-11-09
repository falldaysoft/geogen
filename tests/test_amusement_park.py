"""The amusement park example (examples/amusement_park, geogen.86): four rides that are
mechanisms (jointed rigid bodies with motors) the Godot runtime simulates. The carousel is
checked in the default tier; the wheel, drop tower and swings in the exhaustive tier. The
fairgoers buy from the stalls, eat and bin the wrapper (geogen.87.1): the rules chain in the
reference machine here, the behaviour in Godot in the exhaustive tier."""

import math
import subprocess
import sys
from pathlib import Path

import pytest

from geogen import roots
from geogen.registry import SceneRegistry
from geogen.runtime_client import step

EXAMPLE = Path(__file__).parent.parent / "examples" / "amusement_park"
SCENE = "amusement_park"

pytestmark = pytest.mark.xdist_group("test_amusement_park")      # one export for the module


@pytest.fixture(scope="module")
def built():
    try:
        roots.use(EXAMPLE)
        registry = SceneRegistry()
        registry.discover()
        yield registry[SCENE]()
    finally:
        roots.use()


def test_the_project_passes_its_checks():
    done = subprocess.run([sys.executable, "-m", "geogen.main", "check", "--project", str(EXAMPLE)],
                          capture_output=True, text=True, cwd=EXAMPLE.parent.parent)
    assert done.returncode == 0, done.stdout + done.stderr


def test_every_ride_is_a_mechanism_with_a_drive(built):
    rides = {n.name: n for n in built.iter_nodes() if "mechanism" in n.meta}
    assert set(rides) == {"carousel", "wheel", "tower", "swings", "striker"}
    striker = rides.pop("striker")                                      # the strength tester's puck
    assert set(striker.meta["mechanism"]["bodies"]) == {"puck"}
    bodies = {name: set(node.meta["mechanism"]["bodies"]) for name, node in rides.items()}
    assert bodies["carousel"] == {"platform"}
    assert bodies["wheel"] == {"wheel", *(f"gondola_{i}" for i in range(8))}
    assert bodies["tower"] == {"car"}
    assert bodies["swings"] == {"top", *(f"chair_{i}" for i in range(8))}
    for name, node in rides.items():
        drives = [m for i in node.interactions for m in i.motions if m.kind == "drive"]
        assert len(drives) == 1, name
    # Every ride runs cycles by itself: stopped (or waiting) heads into running on a timer.
    for name, inter in (("carousel", "run"), ("wheel", "run"), ("swings", "run"), ("tower", "ride")):
        it = next(i for i in rides[name].interactions if i.name == inter)
        rest = it.states["stopped" if inter == "run" else "waiting"]
        assert rest.then in ("running", "lifting") and rest.after > 0, name
    # The drop tower's cycle lets the car fall free, then brakes with the motor.
    ride = next(i for i in rides["tower"].interactions if i.name == "ride")
    assert ride.motions[0].free == ["dropping"] and ride.states["dropping"].then == "braking"
    # Seats ride their bodies: the carousel's horses, the tower's bench, every swing chair.
    seats = {name: {a.get("body") for a in node.meta.get("affordances", []) if a["type"] == "sit"}
             for name, node in rides.items()}
    assert seats["carousel"] == {"platform"} and seats["tower"] == {"car"}
    assert seats["swings"] == {f"chair_{i}" for i in range(8)}
    assert seats["wheel"] == {f"gondola_{i}" for i in range(8)}
    # An operator's spot at every ride, and the seats advertise fun to the fairgoers.
    for name, node in rides.items():
        kinds = {a["type"]: a for a in node.meta["affordances"]}
        assert "operator" in kinds["stand"]["tags"] and kinds["sit"]["advertises"] == {"fun": pytest.approx(kinds["sit"]["advertises"]["fun"])}
    # The swing chairs' hinges turn with their copies: tangential all the way round.
    chairs = rides["swings"].meta["mechanism"]["bodies"]
    assert chairs["chair_0"]["joint"]["axis"] == pytest.approx([0, 0, 1])
    assert chairs["chair_2"]["joint"]["axis"] == pytest.approx([1, 0, 0], abs=1e-6)


# ---- Godot --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def park_export(tmp_path_factory):
    """A fresh export of the park, shared by the module's Godot runs (the navmesh bake too)."""
    from geogen.export import export_scene

    out = tmp_path_factory.mktemp("park")
    try:
        roots.use(EXAMPLE)
        registry = SceneRegistry()
        registry.discover()
        export_scene(registry[SCENE](), out / f"{SCENE}.glb")
    finally:
        roots.use()
    return out


@pytest.fixture(scope="module")
def park(godot_runtime, park_export):
    client = godot_runtime(SCENE, park_export)
    client.call("wait_ready", timeout=90)
    return client


def _bodies(rt, asset: str) -> dict:
    return {b["body"]: b for b in rt.call("world.bodies") if b["uid"].startswith(f"{SCENE}:{asset}/")}


def _run(rt, seconds: float):
    rt.call("time.advance", seconds=seconds, scale=4)


def _angle_about(p, centre) -> float:
    return math.degrees(math.atan2(-(p[2] - centre[1]), p[0] - centre[0]))


def test_the_carousel_carries_a_standing_rider_and_a_seated_one_in_godot(park):
    """Stand on the deck: the platform body (a motorised hinge) turns under the player and the
    platform velocity carries them round. Sit on a horse: the pose follows the body."""
    rt, centre = park, (-25.0, 0.0)
    rt.call("interactions.set", asset="carousel", interaction="run", state="stopped", instant=True)
    rt.call("player.teleport", pos=[-23.0, 0.4, 1.0])
    _run(rt, 1.0)
    start = rt.call("player.status")["position"]
    assert abs(start[1] - 0.36) < 0.05, start                         # standing on the deck
    rt.call("interactions.set", asset="carousel", interaction="run", state="running")
    _run(rt, 10.0)                                                    # 6 s ramp, then 18 deg/s
    platform = _bodies(rt, "carousel")["platform"]
    assert platform["motor"] == {"on": True, "speed": pytest.approx(18.0)}
    assert platform["angular_speed"] == pytest.approx(18.0, abs=0.5)
    now = rt.call("player.status")["position"]
    turned = (_angle_about(now, centre) - _angle_about(start, centre)) % 360
    assert 80 < turned < 140, (start, now, turned)                    # about 108 degrees round
    assert abs(math.hypot(now[0] - centre[0], now[2] - centre[1]) - math.hypot(start[0] - centre[0], start[2] - centre[1])) < 0.4
    # Sit on the nearest horse and keep going round with it.
    assert rt.call("player.sit", asset="carousel")["pose"] == "sit"
    seated = rt.call("player.status")["position"]
    _run(rt, 5.0)
    later = rt.call("player.status")["position"]
    assert abs(seated[1] - 1.5 - 0.09) < 0.1 and abs(later[1] - seated[1]) < 0.02
    assert 70 < (_angle_about(later, centre) - _angle_about(seated, centre)) % 360 < 110
    rt.call("player.stand")
    rt.call("interactions.set", asset="carousel", interaction="run", state="stopped")
    _run(rt, 8.0)
    assert _bodies(rt, "carousel")["platform"]["angular_speed"] < 0.5


@pytest.mark.exhaustive
def test_the_ferris_wheel_turns_with_its_gondolas_hanging_level_in_godot(park):
    rt = park
    rt.call("player.teleport", pos=[0, 0.1, -31])
    rt.call("interactions.set", asset="wheel", interaction="run", state="stopped", instant=True)   # rides cycle on their own
    rt.call("interactions.set", asset="wheel", interaction="run", state="running")
    _run(rt, 20.0)
    bodies = _bodies(rt, "wheel")
    assert bodies["wheel"]["angular_speed"] == pytest.approx(12.0, abs=0.5)
    gondolas = [bodies[f"gondola_{i}"] for i in range(8)]
    assert max(g["tilt_deg"] for g in gondolas) < 10, [g["tilt_deg"] for g in gondolas]
    # They are spread round the rim (6 m from the hub, 8.7 m up), not bunched or fallen.
    heights = sorted(g["position"][1] for g in gondolas)
    assert heights[0] < 4 and heights[-1] > 10
    rt.call("interactions.set", asset="wheel", interaction="run", state="stopped")


@pytest.mark.exhaustive
def test_fairgoers_ride_the_rides_and_operators_man_them_in_godot(park):
    """NPCs take ride seats (affordances on mechanism bodies) while a ride is stopped at the
    ground, go round with them, and the operators stand at the levers."""
    rt = park
    rt.call("player.teleport", pos=[0, 0.1, -31])
    rides = ("carousel#", "wheel#", "tower#", "swings#")
    seen, high = set(), set()
    for _ in range(10):
        _run(rt, 30.0)
        rows = rt.call("npcs.list")
        rows = rows if isinstance(rows, list) else rows.get("npcs", [])
        for r in rows:
            doing = str(r.get("doing", ""))
            if "operators" in r["actor"]:
                if any(doing.startswith(k) for k in rides):
                    seen.add("operator")
            elif any(doing.startswith(k) for k in rides):
                seen.add(doing.split("#")[0])
                assert r["position"][1] > -0.2, (r["actor"], doing, r["position"])
                if r["position"][1] > 2.5:
                    high.add(doing.split("#")[0])
    assert "operator" in seen and len(seen - {"operator"}) >= 2, seen
    assert high, "no rider was carried up by a ride"


@pytest.mark.exhaustive
def test_the_drop_tower_free_falls_and_brakes_with_a_seated_rider_in_godot(park):
    rt = park
    rt.call("interactions.set", asset="tower", interaction="ride", state="waiting", instant=True)   # rides cycle on their own
    rt.call("player.teleport", pos=[27.2, 0.3, 0])
    _run(rt, 2.0)
    assert rt.call("player.sit", asset="tower")["pose"] == "sit"
    rt.call("interactions.set", asset="tower", interaction="ride", state="lifting")
    _run(rt, 11.0)
    car = _bodies(rt, "tower")["car"]
    assert car["position"][1] > 15.5 and car["tilt_deg"] < 1.0                 # lifted to the limit
    assert rt.call("player.status")["position"][1] > 15.5                       # and the rider with it
    # Wait for the fall, then sample it: free fall gains about 9.8 m/s a second.
    for _ in range(40):
        _run(rt, 0.25)
        if rt.call("interactions.list", asset="tower")[0]["state"] == "dropping":
            break
    _run(rt, 0.25)
    v0, y0 = _bodies(rt, "tower")["car"]["speed"], rt.call("player.status")["position"][1]
    _run(rt, 0.5)
    v1, y1 = _bodies(rt, "tower")["car"]["speed"], rt.call("player.status")["position"][1]
    assert 4.0 < v1 - v0 < 5.6, (v0, v1)
    assert y0 - y1 > 1.5                                                         # the rider falls with it
    _run(rt, 4.0)
    car = _bodies(rt, "tower")["car"]
    assert car["speed"] < 1.5 and 0 <= car["position"][1] < 3.5                 # braked near the bottom
    rt.call("player.stand")


@pytest.mark.exhaustive
def test_the_swing_chairs_fly_out_as_the_crown_spins_in_godot(park):
    rt, centre = park, (19.0, -19.0)
    rt.call("player.teleport", pos=[0, 0.1, -31])
    # The rides cycle on their own: bring the swings to a stop the way the ride does (the motor
    # ramps down over 10 s; an instant stop at full speed leaves the chairs swinging for a
    # minute), restart the stopped dwell so it doesn't set off again, and let the chairs settle.
    rt.call("interactions.set", asset="swings", interaction="run", state="stopped")
    _run(rt, 15.0)
    rt.call("interactions.set", asset="swings", interaction="run", state="stopped", instant=True)
    _run(rt, 20.0)
    rest = _bodies(rt, "swings")["chair_0"]["position"]
    assert math.hypot(rest[0] - centre[0], rest[2] - centre[1]) < 4.6             # hanging under the arm's end
    rt.call("interactions.set", asset="swings", interaction="run", state="running")
    _run(rt, 16.0)
    bodies = _bodies(rt, "swings")
    assert bodies["top"]["angular_speed"] == pytest.approx(45.0, abs=1.0)
    radii = [math.hypot(bodies[f"chair_{i}"]["position"][0] - centre[0], bodies[f"chair_{i}"]["position"][2] - centre[1])
             for i in range(8)]
    assert min(radii) > 5.0, radii                                               # swung out past the arms
    assert max(b["tilt_deg"] for b in bodies.values()) > 15                       # leaning out on their chains
    rt.call("interactions.set", asset="swings", interaction="run", state="stopped")


def test_a_stall_sells_what_its_kind_says_and_eating_leaves_a_wrapper_for_a_bin(built):
    """The stall's `serve` interaction gives its item to whoever used it (an NPC actor here) and
    refuses a second; using the item (the scene's rule) swaps it for a wrapper; a trashcan's
    `bin` takes the wrapper, and only opens to someone carrying one."""
    from geogen.quests import collect
    from geogen.rules import Machine

    try:
        roots.use(EXAMPLE)                      # (another module's fixture may have switched the roots)
        program = collect(built, "amusement_park").program()
    finally:
        roots.use()
    stalls = {k: program["interactions"][k] for k in program["interactions"] if k.endswith("#serve")}
    bins = [k for k in program["interactions"] if k.endswith("#bin")]
    assert len(stalls) == 11 and len(bins) == 6
    npc = "amusement_park:visitors_1"
    m = Machine(program, {"player": {"kind": "player"}, npc: {"kind": "npc"}})
    for key in bins:
        assert not m.use(key, npc)                           # nothing to throw away yet
    key = next(k for k in stalls if k.endswith("stalls_west_1#serve"))
    assert m.use(key, npc)
    m.advance(0.5)
    carried = {k.split("item_")[1]: m.get(k, npc) for k in program["variables"] if "item_" in k and m.get(k, npc)}
    (item,) = carried
    assert carried[item] == 1 and item in {"hot_dog", "popcorn", "lemonade_cup", "candy_floss", "ice_cream",
                                           "pretzel", "toffee_apple", "balloon"}
    assert not m.use(key, npc)                               # "You already have one"
    if item == "balloon":
        return                                               # kept, not eaten
    m.post({"type": "use_item", "name": item, "actor": npc})
    m.advance(0.1)
    assert m.get(f"actor:amusement_park.item_{item}", npc) == 0
    assert m.get("actor:amusement_park.item_wrapper", npc) == 1
    assert m.use(bins[0], npc)
    m.advance(2.5)                                           # the bin and the stall are back at ready
    assert m.get("actor:amusement_park.item_wrapper", npc) == 0
    assert m.use(key, npc)                                   # hungry again: the stall serves


def test_in_godot_the_player_buys_eats_and_bins_with_the_same_blocks(godot_runtime, park_export):
    """The player's half of the chain: E on a stall (its `serve` interaction) gives the item, a
    second press is refused, F (`inventory.use`) eats it and leaves the wrapper, and a bin takes
    the wrapper."""
    rt = godot_runtime("amusement_park", park_export)
    rt.call("wait_ready", timeout=90)
    stall = next(a for a in rt.call("interactions.list") if a["interaction"] == "serve")["asset"]
    rt.call("interactions.use", asset=stall, interaction="serve", actor="player", wait=True)
    rt.call("time.advance", seconds=2.5, scale=1)
    (item,) = rt.call("inventory.list")["items"]
    assert item["count"] == 1 and (item["tags"] == ["food"] or item["id"] == "balloon")
    rt.call("interactions.use", asset=stall, interaction="serve", actor="player", wait=True)    # refused: still one
    rt.call("time.advance", seconds=2.5, scale=1)
    assert [i["count"] for i in rt.call("inventory.list")["items"]] == [1]
    if item["id"] == "balloon":
        return
    rt.call("inventory.use", item=item["id"])
    rt.call("time.advance", seconds=0.5, scale=1)
    assert [i["id"] for i in rt.call("inventory.list")["items"]] == ["wrapper"]
    rt.call("interactions.use", asset="bin_sw", interaction="bin", actor="player", wait=True)
    rt.call("time.advance", seconds=1.5, scale=1)
    assert rt.call("inventory.list")["items"] == []


@pytest.mark.exhaustive
def test_in_godot_fairgoers_buy_eat_and_bin(godot_script, park_export):
    """A crowd in the runtime: within a few minutes some visitors have bought (the stall's `buy`
    affordance), eaten (`use_item`, the `eat` clip) and thrown the wrapper in a bin, with none of
    the new steps failing."""
    run = godot_script([step("time.simulate", seconds=240)],
                       "--scene=amusement_park", f"--generated={park_export}",
                       start={"timescale": 8}, engine_args=("--fixed-fps", "60"))
    visitors = [n for n in run["time.simulate"]["npcs"] if n["npc"].startswith("visitors")]
    bought = sum(v for n in visitors for k, v in n["used"].items() if k.startswith("stalls_"))
    ate = sum(n["items_used"] for n in visitors)
    binned = sum(v for n in visitors for k, v in n["used"].items() if k.startswith("bin_"))
    assert bought >= 3 and ate >= 2 and binned >= 1, (bought, ate, binned)
    # (A `nearby` spot someone else walked onto first is an ordinary walk failure; the NPC decides again.)
    assert not [f for n in visitors for f in n["failures"]
                if ("use_item" in f or "nearby" in f or "binning" in f) and "occupied" not in f]
