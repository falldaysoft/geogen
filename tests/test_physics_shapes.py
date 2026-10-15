"""Physics prop shapes (geogen.physics `shape`, `damping`): a ball rolls where a box stays."""

import pytest

from geogen.layout.composer import SceneComposer
from geogen.layout.loader import LayoutLoader

BALL = """
name: ball
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
physics: {mass: 0.5, shape: sphere, damping: {linear: 0.1, angular: 0.2}}
parts:
  body: {primitive: sphere, size: [0.2, 0.2, 0.2], anchor: bottom_center, material: metal}
"""

SLAB = BALL.replace("shape: sphere, damping: {linear: 0.1, angular: 0.2}", "shape: box").replace(
    "primitive: sphere", "primitive: cube")

WEDGE = """
name: wedge
origin: bottom_center
size: [2, 1, 4]
parts:
  ramp: {primitive: prism, size: [1, 1, 1], anchor: bottom_center, apex: back, material: concrete}
"""

SCENE = """
name: slope
size: [10, 3, 10]
spawns:
  start: {position: [4, 0, 4], facing: north}
slots:
  wedge: {position: [0, 0, 0]}
  ball:  {position: [-0.5, 0.75, 0]}
  slab:  {position: [0.5, 0.75, 0]}
place:
  wedge: {asset: wedge.yaml, slot: wedge}
  ball:  {asset: ball.yaml, slot: ball}
  slab:  {asset: slab.yaml, slot: slab}
"""


def test_shapes_are_sized_from_the_geometry():
    ball = LayoutLoader().load_string(BALL).meta["physics"]
    assert ball["shape"] == "sphere" and ball["radius"] == pytest.approx(0.1, abs=1e-3)
    assert ball["damping"] == {"linear": 0.1, "angular": 0.2}
    capsule = LayoutLoader().load_string(BALL.replace("shape: sphere", "shape: capsule").replace(
        "primitive: sphere, size: [0.2, 0.2, 0.2]", "primitive: cube, size: [0.4, 1.6, 0.3]")).meta["physics"]
    assert capsule["radius"] == pytest.approx(0.2, abs=1e-3) and capsule["height"] == pytest.approx(1.6, abs=1e-3)
    assert "shape" not in LayoutLoader().load_string(BALL.replace(", shape: sphere", "")).meta["physics"]
    with pytest.raises(ValueError, match="shape is box, sphere, capsule"):
        LayoutLoader().load_string(BALL.replace("shape: sphere", "shape: cone"))
    with pytest.raises(ValueError, match="damping"):
        LayoutLoader().load_string(BALL.replace("angular: 0.2", "spin: 0.2"))


@pytest.fixture(scope="module")
def runtime(godot_runtime, tmp_path_factory):
    from geogen.export import export_scene

    assets = tmp_path_factory.mktemp("slope_assets")
    for name, text in (("ball", BALL), ("slab", SLAB), ("wedge", WEDGE), ("slope", SCENE)):
        (assets / f"{name}.yaml").write_text(text)
    out = tmp_path_factory.mktemp("slope")
    export_scene(SceneComposer(assets_dir=assets).compose(assets / "slope.yaml"), out / "slope.glb")
    client = godot_runtime("slope", out)
    client.call("wait_ready", timeout=60)
    return client


def test_a_ball_rolls_down_a_slope_a_box_stays_on(runtime):
    start = {b["node"]: b["position"] for b in runtime.call("world.bodies")}
    runtime.call("time.advance", seconds=3.0, scale=1)
    end = {b["node"]: b["position"] for b in runtime.call("world.bodies")}
    rolled = abs(end["ball"][2] - start["ball"][2])
    slid = abs(end["slab"][2] - start["slab"][2])
    assert rolled > 1.0 and end["ball"][1] < 0.3, (start, end)        # down the slope and off it
    assert slid < 0.2, (start, end)                                   # friction holds a box
