"""The tower defense example (examples/tower_defense, geogen.47): robots walk a path from a gate
to a core in waves; towers built on pads for gold shoot them; leaks cost lives. Checks the
project, the balance against the reference simulation (geogen.flows.simulate), the rules on
the reference machine, and a game in Godot over the control API."""

import subprocess
import sys
from pathlib import Path

import pytest

from geogen import flows, roots
from geogen.quests import collect
from geogen.registry import SceneRegistry

EXAMPLE = Path(__file__).parent.parent / "examples" / "tower_defense"
SCENE = "tower_defense"
PADS = "abcdef"


@pytest.fixture(scope="module")
def built():
    try:
        roots.use(EXAMPLE)
        registry = SceneRegistry()
        registry.discover()
        scene = registry[SCENE]()
        yield scene, collect(scene, SCENE)
    finally:
        roots.use()


def _item(name):
    return flows.load_item(EXAMPLE / "assets" / "items" / f"{name}.yaml")


def test_the_project_passes_its_checks():
    done = subprocess.run([sys.executable, "-m", "geogen.main", "check", "--project", str(EXAMPLE)],
                          capture_output=True, text=True, cwd=EXAMPLE.parent.parent)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_path_runs_from_the_gate_to_the_core_past_every_pad(built):
    scene, _ = built
    net = flows.network(scene)
    (path,) = net["tracks"]
    (gate,) = net["sources"]
    assert gate["node"].name == "gate" and gate["track"] == 0      # the gate feeds the path
    assert path["end"] == "sink"
    core = next(n for n in scene.iter_nodes() if n.name == "core").world_transform()[:3, 3]
    end = path["points"][-1]
    assert ((end[0] - core[0]) ** 2 + (end[2] - core[2]) ** 2) ** 0.5 < 1.5
    for effector in net["effectors"]:                             # every tower reaches a stretch of path
        at = effector["at"]
        near = min(((p[0] - at[0]) ** 2 + (p[2] - at[2]) ** 2) ** 0.5 for p in path["points"])
        assert near < effector["range"] - 2.0, (effector["node"].name, near)


def _harm_by_wave(scene, pads):
    """Harm that reaches the core in each wave, waves sent one after another, with ``pads`` built."""
    out = flows.simulate(scene, 60 * 60 * 10, running={"gate": True, **{f"pad_{p}": True for p in pads}}, auto=True)
    starts = [e[0] for e in out["events"] if e[1] == "wave_started"]
    arrived = [e[0] for e in out["events"] if e[1] == "received"]
    harm = [0] * len(starts)
    for tick, name in zip(arrived, out["through"][0]):
        harm[sum(1 for s in starts if s <= tick) - 1] += _item(name)["data"]["harm"]
    return out, harm


def test_the_waves_ramp_against_the_towers(built):
    """No towers: the first wave walks straight through. Two (what the starting gold buys)
    hold the early waves; all six hold them all, losing a few lives at most."""
    scene, _ = built
    out, harm = _harm_by_wave(scene, "")
    assert out["wave"] == [8] and harm[0] == 6 and out["killed"] == [0]
    out, harm = _harm_by_wave(scene, "ab")
    assert harm[:3] == [0, 0, 0] and sum(harm) >= 20                  # two towers lose the game
    out, harm = _harm_by_wave(scene, PADS)
    assert out["wave"] == [8] and sum(harm) < 10 and not out["items"]


def test_the_rules_keep_lives_gold_and_the_game(built):
    _, quests = built
    from geogen.rules import Machine

    machine = Machine(quests.program(), {"player": {"kind": "player"}})
    key = lambda k: f"{SCENE}:{k}"
    world = lambda name: f"world:{SCENE}.{name}"
    signal = lambda subject, name, **data: machine.post(
        {"type": "signal", "subject": key(subject), "name": name, "data": data})
    assert machine.get(world("gold")) == 25 and machine.get(world("phase")) == "ready"
    assert machine.use(key("pad_a#build"), "player") and machine.use(key("pad_b#build"), "player")
    machine.advance(1.0)
    assert machine.get(world("gold")) == 5 and machine.get(key("pad_a#enabled")) is True
    assert not machine.use(key("pad_c#build"), "player")                 # 5 gold: refused
    assert ["notify", "A tower costs 10 gold", "player"] in machine.feedback
    machine.use(key("next_wave#press"), "player")
    machine.advance(1.0)
    assert machine.get(key("gate#running")) is True and machine.get(world("phase")) == "wave"
    signal("gate", "killed", item="grunt", hp=0, value=2, harm=1)
    signal("path", "received", item="brute", hp=36, value=8, harm=3)
    assert machine.get(world("gold")) == 7 and machine.get(world("lives")) == 17
    signal("gate", "wave_cleared", wave=1)
    assert machine.get(world("phase")) == "ready" and machine.get(world("gold")) == 12
    machine.use(key("next_wave#press"), "player")
    machine.advance(1.0)
    for _ in range(6):                                              # lives run out: the core falls
        signal("path", "received", item="brute", hp=36, value=8, harm=3)
    assert machine.get(world("lives")) == 0 and machine.get(world("phase")) == "lost"
    assert machine.get(key("core#damage")) == "destroyed" and machine.get(key("gate#running")) is False
    machine.use(key("reset#press"), "player")
    machine.advance(1.0)
    assert machine.get(world("lives")) == 20 and machine.get(world("gold")) == 25
    assert machine.get(world("phase")) == "ready" and machine.get(key("core#damage")) == "intact"
    assert machine.get(key("pad_a#build")) == "empty" and machine.get(key("gate#restarts")) == 1


# --- in Godot -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime(godot_runtime, tmp_path_factory):
    from geogen.export import export_scene

    out = tmp_path_factory.mktemp("tower_defense")
    try:
        roots.use(EXAMPLE)
        registry = SceneRegistry()
        registry.discover()
        export_scene(registry[SCENE](), out / f"{SCENE}.glb")
    finally:
        roots.use()
    client = godot_runtime(SCENE, out)
    client.call("wait_ready", timeout=90)
    return client


def _world(runtime, name):
    return runtime.call("state.get", key=f"world:{SCENE}.{name}")["value"]


def _get(runtime, key):
    return runtime.call("state.get", key=f"{SCENE}:{key}")["value"]


def _play_wave(runtime, limit=90):
    """Press NEXT WAVE and run until the wave is over (or the game is)."""
    runtime.call("interactions.use", asset="next_wave")
    runtime.call("time.step", frames=120)
    for _ in range(limit):
        if _world(runtime, "phase") != "wave":
            return
        runtime.call("time.step", frames=120)                    # 2 s
    raise AssertionError(f"the wave didn't end: {_world(runtime, 'phase')}")


def test_godot_plays_a_game(runtime):
    runtime.call("time.pause")
    runtime.call("time.step", frames=5)
    # No towers: the first wave's six grunts all reach the core.
    _play_wave(runtime)
    assert _world(runtime, "lives") == 14 and _world(runtime, "gold") == 30 and _get(runtime, "gate#wave") == 1
    assert _get(runtime, "core#damage") == "cracked" and _get(runtime, "gate#glow") == "off"
    # Towers on four pads: the second wave is stopped, and pays.
    runtime.call("state.set", key=f"world:{SCENE}.gold", value=40)
    for pad in "abcd":
        runtime.call("interactions.use", asset=f"pad_{pad}")
    runtime.call("time.step", frames=30)
    assert _world(runtime, "gold") == 0
    _play_wave(runtime)
    assert _world(runtime, "lives") == 14 and _get(runtime, "gate#killed") == 10
    assert _world(runtime, "gold") == 25                           # ten grunts at 2, and 5 for the wave
    # Start over, then lose: one life left and no towers.
    runtime.call("interactions.use", asset="reset")
    runtime.call("time.step", frames=60)
    assert _world(runtime, "lives") == 20 and _get(runtime, "gate#wave") == 0
    assert _get(runtime, "pad_a#build") == "empty" and not runtime.call("flows.list")["items"]
    runtime.call("state.set", key=f"world:{SCENE}.lives", value=1)
    _play_wave(runtime)
    assert _world(runtime, "phase") == "lost" and _get(runtime, "core#damage") == "destroyed"
    banner = runtime.call("hud.stats")["banner"]
    assert banner["visible"] and "fallen" in banner["text"]
    runtime.call("time.resume")
