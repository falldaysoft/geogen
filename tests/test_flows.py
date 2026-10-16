"""Spawned items (geogen.flows, runtime flows.gd): items, sources and what the rules see."""

import json

import pytest

from geogen import flows
from geogen.export import export_scene, manifest_path
from geogen.layout.composer import SceneComposer
from geogen.quests import collect
from geogen.rules import Machine

BALL = """
name: ball
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
physics: {mass: 0.3, shape: sphere, bounce: 0.2, damping: {linear: 0.8, angular: 2}}
parts:
  body: {primitive: sphere, size: [0.16, 0.16, 0.16], anchor: bottom_center, material: red_metal}
"""

ITEM = """
kind: item
body: ball.yaml
mode: physics
data: {value: 2, colour: red}
stuck: 2
"""

HOPPER = """
name: hopper
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
source: {items: items/ball.yaml, rate: 4, at: mouth, max_alive: 3, initial: stopped, velocity: [0, 0, 1]}
parts:
  post: {primitive: cylinder, size: [0.1, 1.4, 0.1], anchor: bottom_center, material: metal}
attachments:
  mouth: {anchor: bottom_center, offset: [0, 1.5, 0], offset_mode: absolute}
"""

FLOOR = """
name: floor
origin: bottom_center
size: [10, 0.1, 10]
parts:
  slab: {primitive: cube, size: [1, 1, 1], anchor: bottom_center, bevel: 0, material: concrete, walkable: true}
"""

SCENE = """
name: yard
size: [10, 3, 10]
spawns:
  start: {position: [0, 0, -3], facing: north}
state:
  given: {scope: world, type: number, default: 0}
slots:
  hopper: {position: [0, 0, 0]}
place:
  floor: {asset: floor.yaml}
  hopper: {asset: hopper.yaml, slot: hopper}
rules:
  - id: tally
    on: {signal: hopper.emitted}
    when: 'event.item == "ball"'
    do: [{set: world.given, to: "world.given + 1"}]
  - id: one_more
    on: {start: true}
    do: [{input: hopper.emit}, {input: hopper.emit}]
"""


@pytest.fixture
def assets(tmp_path):
    for name, text in (("ball", BALL), ("items/ball", ITEM), ("hopper", HOPPER), ("floor", FLOOR), ("yard", SCENE)):
        path = tmp_path / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    return tmp_path


def compose(assets):
    return SceneComposer(assets_dir=assets).compose(assets / "yard.yaml")


def test_an_item_file_is_read_and_checked(assets):
    item = flows.load_item(assets / "items/ball.yaml")
    assert item == {"name": "ball", "mode": "physics", "body": {"asset": "ball.yaml", "params": {}},
                    "data": {"value": 2, "colour": "red"}, "lifetime": 0.0, "stuck": 2.0}
    bad = assets / "items/bad.yaml"
    for text, message in (("kind: item\nbody: ball.yaml\nmode: roll\n", "mode is one of"),
                          ("kind: item\nbody: ball.yaml\nflavour: 1\n", "unknown keys"),
                          ("kind: item\n", "body is an asset file"),
                          ("kind: item\nbody: ball.yaml\ndata: {ok: true}\n", "data is"),
                          ("kind: item\nbody: ball.yaml\ngait: {hop: 1}\n", "gait is"),
                          ("kind: item\nbody: ball.yaml\ngait: {stride: 0}\n", "stride is more than")):
        bad.write_text(text)
        with pytest.raises(flows.FlowError, match=message):
            flows.load_item(bad)
    bad.write_text("kind: item\nbody: ball.yaml\nmode: walk\ngait: {bob: 0.03, sway: 4}\n")
    assert flows.load_item(bad)["gait"] == {"stride": 0.5, "bob": 0.03, "sway": 4.0, "waddle": 0.0}


def test_a_source_gets_hidden_templates_of_its_items(assets):
    scene = compose(assets)
    hopper = next(n for n in scene.iter_nodes() if n.name == "hopper")
    source = hopper.meta["flows"]["source"]
    assert source["at"] == pytest.approx([0, 1.5, 0]) and source["initial"] == "stopped"
    assert source["items"] == [{"item": "items/ball.yaml", "weight": 1.0, "name": "ball"}]
    template = next(n for n in hopper.iter_nodes() if flows.is_template(n))
    assert template.hidden and template.meta["item"]["radius"] == pytest.approx(0.08, abs=0.005)
    assert template.meta["physics"]["shape"] == "sphere"
    # hidden: no renderer, mesh QA or layout check sees it
    meshes = {id(n) for n, _ in scene.iter_meshes()}
    assert not any(id(n) in meshes for n in template.iter_nodes())


def test_bad_sources_fail(assets):
    path = assets / "hopper.yaml"
    for old, new, message in (("at: mouth", "at: spout", "no attachment 'spout'"),
                              ("rate: 4", "rate: 4, colour: red", "unknown keys"),
                              ("initial: stopped", "initial: paused", "running or stopped")):
        path.write_text(HOPPER.replace(old, new))
        with pytest.raises(ValueError, match=message):
            compose(assets)
    path.write_text(HOPPER)
    (assets / "ball.yaml").write_text(BALL.replace("physics: {mass: 0.3, shape: sphere, bounce: 0.2, damping: {linear: 0.8, angular: 2}}\n", ""))
    with pytest.raises(flows.FlowError, match="needs a physics: block"):
        compose(assets)


def test_the_rules_see_a_sources_state_signals_and_inputs(assets):
    found = collect(compose(assets), "yard")
    decls = found.declarations
    assert decls.variables["yard:hopper#running"] == {"type": "bool", "default": False}
    assert decls.variables["yard:hopper#emitted"]["driven"] == "flows"
    assert decls.signals["yard:hopper#emitted"]["data"] == {"item": "text"}
    machine = Machine(found.program(), {"player": {"kind": "player"}})
    machine.post({"type": "start", "actor": "player"})
    assert machine.get("yard:hopper#queued") == 2                   # two emits asked for
    machine.post({"type": "signal", "subject": "yard:hopper", "name": "emitted", "data": {"item": "ball"}})
    assert machine.get("world:yard.given") == 1 and machine.errors == []


def test_the_export_carries_sources_and_templates(assets, tmp_path):
    out = tmp_path / "gen" / "yard.glb"
    out.parent.mkdir()
    export_scene(compose(assets), out)
    gltf = _gltf(out)
    extras = [n.get("extras", {}).get("geogen", {}) for n in gltf["nodes"]]
    assert any("flows" in e for e in extras) and any(e.get("item", {}).get("name") == "ball" for e in extras)
    state = json.loads(manifest_path(out).read_text())["state"]
    assert state["inputs"]["yard:hopper#emit"]["set"] == "yard:hopper#queued"


def _gltf(path):
    import struct

    data = path.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


# --- in Godot -----------------------------------------------------------------------------------


@pytest.fixture
def runtime(assets, tmp_path, godot_runtime):
    out = tmp_path / "gen"
    out.mkdir()
    export_scene(compose(assets), out / "yard.glb")
    client = godot_runtime("yard", out)
    client.call("wait_ready", timeout=60)
    return client


def _get(runtime, key):
    return runtime.call("state.get", key=f"yard:{key}")["value"]


def test_godot_makes_queued_items_then_runs_up_to_its_limit(runtime):
    runtime.call("time.advance", seconds=0.5, scale=1)
    report = runtime.call("flows.list")
    assert report["sources"][0]["emitted"] == 2 and _get(runtime, "hopper#queued") == 0, report
    assert runtime.call("state.get", key="world:yard.given")["value"] == 2                                    # the rule heard emitted {item}
    balls = report["items"]
    assert all(b["item"] == "ball" and b["data"]["value"] == 2 for b in balls)
    runtime.call("state.set", key="yard:hopper#running", value=True)
    runtime.call("time.advance", seconds=1.5, scale=1)
    report = runtime.call("flows.list")
    assert len(report["items"]) == 3 and _get(runtime, "hopper#alive") == 3, report      # max_alive
    assert all(item["position"][1] < 0.3 for item in report["items"][:2]), report          # fell to the floor
    # Resting balls are taken away after `stuck` s, and the running source makes more.
    runtime.call("time.advance", seconds=4.0, scale=1)
    report = runtime.call("flows.list")
    assert report["sources"][0]["emitted"] > 4 and _get(runtime, "hopper#alive") <= 3, report


# --- sinks --------------------------------------------------------------------------------------

SINK_SCENE = SCENE.replace("""place:
  floor:""", """places:
  hole: {box: [1.2, 0.8, 1.2], position: [0, 0, 0.6], sink: {accepts: [ball]}}
  bin:  {box: [1, 1, 1], position: [4, 0, 4], sink: true}
place:
  floor:""").replace("""  - id: one_more""", """  - id: banked
    on: {signal: hole.received}
    do: [{set: world.given, to: "world.given + event.value * 100"}]
  - id: one_more""")


@pytest.fixture
def sink_assets(assets):
    (assets / "yard.yaml").write_text(SINK_SCENE)
    return assets


def test_a_sink_place_counts_and_signals_with_the_items_data(sink_assets):
    scene = compose(sink_assets)
    hole = next(n for n in scene.iter_nodes() if n.name == "hole")
    assert hole.meta["place"]["sink"] == {"accepts": ["ball"]}
    decls = collect(scene, "yard").declarations
    assert decls.variables["yard:hole#received"]["driven"] == "flows"
    assert decls.signals["yard:hole#received"]["data"] == {"value": "number", "colour": "text", "item": "text"}
    assert next(n for n in scene.iter_nodes() if n.name == "bin").meta["place"]["sink"] == {"accepts": []}


def test_reading_a_field_items_dont_carry_fails(sink_assets):
    path = sink_assets / "yard.yaml"
    path.write_text(path.read_text().replace("event.value * 100", "event.weight"))
    from geogen.quests import QuestError
    with pytest.raises(QuestError, match=r"event.weight: a signal event here carries colour, item, value"):
        collect(compose(sink_assets), "yard")


def test_bad_sinks_fail(sink_assets):
    path = sink_assets / "yard.yaml"
    path.write_text(path.read_text().replace("sink: true", "sink: {take: [ball]}"))
    with pytest.raises(ValueError, match=r"sink is true or \{accepts"):
        compose(sink_assets)


def test_godot_sinks_take_items_in(sink_assets, tmp_path, godot_runtime):
    out = tmp_path / "gen"
    out.mkdir()
    export_scene(compose(sink_assets), out / "yard.glb")
    runtime = godot_runtime("yard", out)
    runtime.call("wait_ready", timeout=60)
    runtime.call("state.set", key="yard:hopper#running", value=True)
    runtime.call("time.advance", seconds=3.0, scale=1)
    runtime.call("time.pause")                          # read everything at one moment
    emitted = _get(runtime, "hopper#emitted")
    received = _get(runtime, "hole#received")
    # Every ball drops (moving 1 m/s toward +z) into the hole: none are left, and each paid 2 * 100.
    assert emitted >= 8 and received >= emitted - 3, (emitted, received)
    assert runtime.call("state.get", key="world:yard.given")["value"] == emitted + received * 200
    assert _get(runtime, "hopper#alive") <= emitted - received        # a ball may also come to rest and be cleared


def test_godot_saves_keep_physics_items(runtime):
    runtime.call("state.set", key="yard:hopper#running", value=True)
    runtime.call("time.advance", seconds=1.0, scale=1)
    runtime.call("time.pause")
    saved = runtime.call("state.save")
    before = {it["id"]: it["position"] for it in runtime.call("flows.list")["items"]}
    assert before and all("position" in it for it in saved["flows"]["items"])
    runtime.call("time.step", frames=120)
    runtime.call("state.load", state=saved)
    after = {it["id"]: it["position"] for it in runtime.call("flows.list")["items"]}
    runtime.call("time.resume")
    assert set(after) == set(before)
    for id, at in before.items():
        assert after[id] == pytest.approx(at, abs=0.01)
