"""Move motions (geogen.40.2): a part goes to another placed thing's attachment, resolved when the
scene is composed into ordinary translate motions in the asset frame."""

import numpy as np
import pytest

from geogen.layout.composer import SceneComposer
from geogen.layout.interactions import apply_state

TRAY = """
name: tray
origin: bottom_center
size: [0.6, 0.05, 0.4]
parts:
  board: {primitive: cube, size: [1, 1, 1], anchor: bottom_center, bevel: 0}
attachments:
  center: {anchor: top_center}
"""
BLOCK = """
name: block
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
params:
  to: {default: "", type: text}
parts:
  stand: {primitive: cube, size: [0.3, 0.6, 0.3], anchor: bottom_center, bevel: 0}
  cube:  {primitive: cube, size: [0.1, 0.1, 0.1], anchor: bottom_center, offset: [0, 0.6, 0], bevel: 0}
interactions:
  carry:
    targets: [cube]
    states:
      here:  {next: there}
      there: {next: here}
    motions:
      - {parts: [cube], move: {there: "{to}"}}
"""
CORNER = """
name: corner
size: [2, 1, 2]
slots:
  here: {position: [0, 0, 0]}
place:
  block: {asset: block.yaml, slot: here, params: {to: tray.center}}
"""
ROOM = """
name: room
size: [8, 2, 8]
slots:
  a: {position: [-2, 0, 0], facing: east}
  b: {position: [2, 0, 1]}
  c: {position: [0, 0, -2]}
place:
  block: {asset: block.yaml, slot: a, params: {to: tray.center}}
  tray:  {asset: tray.yaml, slot: b}
  moved: {asset: block.yaml, slot: c, params: {to: tray.center}, state: there}
  corner: {scene: corner.yaml, slot: c}
"""


@pytest.fixture
def assets(tmp_path):
    for name, text in (("tray", TRAY), ("block", BLOCK), ("corner", CORNER), ("room", ROOM)):
        (tmp_path / f"{name}.yaml").write_text(text)
    return tmp_path


def named(root, name):
    return next(n for n in root.iter_nodes() if n.name == name)


def bottom(node):
    return (node.world_transform() @ np.array([0.0, -node.size[1] / 2, 0.0, 1.0]))[:3]


def test_a_part_moves_onto_another_things_attachment(assets):
    root = SceneComposer(assets_dir=assets).compose(assets / "room.yaml")
    tray_top = (named(root, "tray").world_transform() @ np.array([0, 0.05, 0, 1.0]))[:3]
    block = named(root, "block")
    (carry,) = block.interactions
    assert [m.kind for m in carry.motions] == ["translate"]
    cube = next(p for p in carry.targets)
    start = bottom(cube)
    apply_state(block, carry, "there")
    assert np.allclose(bottom(cube), tray_top, atol=1e-6)            # the cube's bottom on the tray's centre
    apply_state(block, carry, "here")
    assert np.allclose(bottom(cube), start, atol=1e-6)
    moved = named(root, "moved")                                     # placed in that state: posed there
    assert np.allclose(bottom(next(iter(moved.interactions[0].targets))), tray_top, atol=1e-6)
    # in a nested scene that has no tray, the target is found in the scene it is placed in
    nested = next(n for n in named(root, "corner").iter_nodes() if n.interactions)
    apply_state(nested, nested.interactions[0], "there")
    assert np.allclose(bottom(next(iter(nested.interactions[0].targets))), tray_top, atol=1e-6)
    extras = carry.to_extras(lambda n: n.name)
    assert extras["motions"][0]["type"] == "translate" and extras["motions"][0]["values"]["here"] == 0.0


def test_a_target_that_isnt_there_fails_the_scene(assets):
    (assets / "room.yaml").write_text(ROOM.replace("params: {to: tray.center}}", "params: {to: tary.center}}", 1))
    with pytest.raises(ValueError, match="moves to 'tary.center'"):
        SceneComposer(assets_dir=assets).compose(assets / "room.yaml")
