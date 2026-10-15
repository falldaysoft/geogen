"""The marble run example (examples/marble_run, geogen.46): a hopper drops balls (physics
items from its source) down a brass trough into a catch bowl (a sink place)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from geogen import flows, roots
from geogen.quests import collect
from geogen.registry import SceneRegistry

EXAMPLE = Path(__file__).parent.parent / "examples" / "marble_run"
SCENE = "marble_run"


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


def test_the_project_passes_its_checks():
    done = subprocess.run([sys.executable, "-m", "geogen.main", "check", "--project", str(EXAMPLE)],
                          capture_output=True, text=True, cwd=EXAMPLE.parent.parent)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_hopper_drops_balls_over_the_start_of_the_trough(built):
    scene, _ = built
    hopper = next(n for n in scene.iter_nodes() if n.name == "hopper")
    track = next(n for n in scene.iter_nodes() if n.name == "track")
    source = hopper.meta["flows"]["source"]
    assert [i["name"] for i in source["items"]] == ["ball_red", "ball_blue", "ball_gold", "ball_green"]
    assert source["initial"] == "stopped"
    drop = (hopper.world_transform() @ np.r_[source["at"], 1.0])[:3]
    start = track.get_attachment("start").to_matrix()[:3, 3]
    assert np.allclose(drop, start, atol=0.02)                 # the balls' track starts at the neck
    net = flows.network(scene)
    assert [t["mode"] for t in net["tracks"]] == ["roll"] and net["sources"][0]["track"] == 0
    templates = [n.meta["item"] for n in hopper.iter_nodes() if flows.is_template(n)]
    assert all(t["mode"] == "physics" for t in templates) and len(templates) == 4


def test_the_lever_runs_the_hopper_and_the_bowl_counts(built):
    _, quests = built
    from geogen.rules import Machine

    machine = Machine(quests.program(), {"player": {"kind": "player"}})
    key = lambda k: f"{SCENE}:{k}"
    assert machine.get(key("hopper#running")) is False
    machine.use(key("lever#pull"), "player")
    machine.advance(1.0)
    assert machine.get(key("hopper#running")) is True
    for n in range(1, 11):
        machine.set(key("catch#received"), n)
        machine.post({"type": "signal", "subject": key("catch"), "name": "received", "data": {"item": "ball_red", "value": 1}})
    assert ["notify", "Another ten in the bowl", None] in machine.feedback
    assert machine.get(f"world:{SCENE}.milestone") == 20
    assert [s["label"] for s in quests.hud["stats"]] == ["Caught", "In play", "Dropped"]


def test_the_reference_rolls_balls_to_the_end(built):
    scene, _ = built
    out = flows.simulate(scene, 60 * 30, running={"hopper": True})
    drops = [e for e in out["events"] if e[1] == "dropped"]
    assert len(drops) >= 6 and 8 < drops[0][0] / 60 < 20, drops[:2]      # 20 m of trough at a few m/s


# --- in Godot -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime(godot_runtime, tmp_path_factory):
    from geogen.export import export_scene

    out = tmp_path_factory.mktemp("marble_run")
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


def _get(runtime, key):
    return runtime.call("state.get", key=f"{SCENE}:{key}")["value"]


def test_every_ball_that_runs_out_lands_in_the_bowl(runtime, built):
    runtime.call("time.pause")
    started = int(runtime.call("flows.list")["ticks"])           # nothing moves until the hopper runs
    runtime.call("state.set", key=f"{SCENE}:hopper#running", value=True)
    runtime.call("time.step", frames=60 * 15)
    report = runtime.call("flows.list")
    # Rolling is the reference's: the same balls at the same distances along the trough.
    expected = flows.simulate(built[0], int(report["ticks"]) - started, running={"hopper": True})
    on_track = {it["id"]: it["s"] for it in report["items"] if "s" in it}
    want = {it["id"]: it["s"] for it in expected["items"]}
    assert set(on_track) == set(want)
    for id, s in want.items():
        assert on_track[id] == pytest.approx(s, abs=1e-3)
    runtime.call("time.step", frames=60 * 25)
    emitted, alive, caught = _get(runtime, "hopper#emitted"), _get(runtime, "hopper#alive"), _get(runtime, "catch#received")
    runtime.call("time.resume")
    assert caught >= 8 and alive <= 10 and caught == emitted - alive, (emitted, alive, caught)
    shown = runtime.call("hud.stats")["stats"]["stats"]
    assert shown[0] == {"label": "Caught", "value": caught, "text": str(int(caught))}
