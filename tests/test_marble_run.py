"""The marble run example (examples/marble_run, geogen.46): a hopper drops balls (physics
items from its source) down a brass trough, round an exit funnel and into a catch bowl (a
sink place). The balls are physics bodies the whole way."""

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


def _at(scene, tag, attachment):
    node = next(n for n in scene.iter_nodes() if tag in n.tags)
    return node.get_attachment(attachment).to_matrix()[:3, 3]           # in the scene's frame


def test_the_balls_fall_from_the_hopper_through_the_run_into_the_bowl(built):
    scene, _ = built
    hopper = next(n for n in scene.iter_nodes() if n.name == "hopper")
    source = hopper.meta["flows"]["source"]
    assert [i["name"] for i in source["items"]] == ["ball_red", "ball_blue", "ball_gold", "ball_green"]
    assert source["initial"] == "stopped"
    templates = [n.meta["item"] for n in hopper.iter_nodes() if flows.is_template(n)]
    assert all(t["mode"] == "physics" for t in templates) and len(templates) == 4
    assert not flows.network(scene)["tracks"]                      # no track: physics all the way
    drop = (hopper.world_transform() @ np.r_[source["at"], 1.0])[:3]
    start = _at(scene, "machine.marble_run", "start")
    # Balls appear in the feed tube above the hopper's bowl and fall through it: the neck is
    # over the trough's start, the drop is off centre inside the rim and above the funnel.
    neck = hopper.world_transform()[:3, 3]
    assert np.allclose(neck[[0, 2]], start[[0, 2]], atol=0.01)
    assert 0.1 < np.hypot(*(drop - neck)[[0, 2]]) < 0.35 and drop[1] - start[1] > 0.8
    # The trough runs out over the funnel, inside its rim and above it; the neck is over the bowl.
    end, neck = _at(scene, "machine.marble_run", "end"), _at(scene, "machine.funnel", "neck")
    assert 0.2 < np.hypot(*(end - neck)[[0, 2]]) < 0.4 and end[1] - neck[1] > 0.35
    bowl = next(n for n in scene.iter_nodes() if n.name == "catch")
    assert np.allclose(bowl.world_transform()[[0, 2], 3], neck[[0, 2]], atol=0.01)


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


def test_every_ball_runs_the_course_into_the_bowl(runtime):
    runtime.call("time.pause")
    runtime.call("state.set", key=f"{SCENE}:hopper#running", value=True)
    escaped = set()
    for _ in range(30):                                          # 60 s of play
        runtime.call("time.step", frames=120)
        for it in runtime.call("flows.list")["items"]:
            if it["position"][1] < 0.6:                          # under the bowl's floor: off the run
                escaped.add(it["id"])
    emitted, alive, caught = _get(runtime, "hopper#emitted"), _get(runtime, "hopper#alive"), _get(runtime, "catch#received")
    runtime.call("time.resume")
    assert not escaped
    assert caught >= 6 and alive <= 10 and caught == emitted - alive, (emitted, alive, caught)
    shown = runtime.call("hud.stats")["stats"]["stats"]
    assert shown[0] == {"label": "Caught", "value": caught, "text": str(int(caught))}
