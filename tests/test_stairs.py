"""Stairs generator."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.generators.stairs import StairsGenerator
from geogen.layout import LayoutLoader
from geogen.player import load_player_spec


@pytest.mark.parametrize("style", ["straight", "l", "spiral"])
@pytest.mark.parametrize("turn", ["left", "right"])
def test_stairs_are_watertight_and_climb_the_rise(style, turn):
    gen = StairsGenerator(style=style, rise=2.9, turn=turn)
    node = gen.to_node("stairs")
    assert meshops.validate(node.mesh).watertight
    assert meshops.validate(node.find("stairs_railing").mesh).watertight
    v = node.mesh.vertices
    top = 2.9 if style != "spiral" else 2.9 + 1.0 - 0.0  # spiral newel rises above the top step
    assert np.ptp(v[:, 1]) == pytest.approx(top, abs=1e-6)
    assert node.meta["stairs"]["steps"] == gen.steps


def test_riser_count_respects_player_step_height():
    player = load_player_spec()
    for rise in (0.5, 2.7, 3.0, 4.2):
        gen = StairsGenerator(rise=rise)
        assert gen.riser <= gen.max_riser + 1e-9 < player.step_height
        assert gen.riser * gen.steps == pytest.approx(rise)


def test_straight_flight_dimensions():
    gen = StairsGenerator(style="straight", rise=3.0, tread=0.28, width=1.1)
    extent = np.ptp(gen.generate().vertices, axis=0)
    assert extent[0] == pytest.approx(1.1)
    assert extent[2] == pytest.approx(gen.steps * 0.28)


def test_l_turns_to_the_requested_side():
    left = StairsGenerator(style="l", turn="left").generate().vertices
    right = StairsGenerator(style="l", turn="right").generate().vertices
    # The second flight leaves the landing sideways: most of the mass moves to -X / +X.
    top_left = left[left[:, 1] > left[:, 1].max() - 0.05]
    top_right = right[right[:, 1] > right[:, 1].max() - 0.05]
    assert top_left[:, 0].mean() < 0 < top_right[:, 0].mean()


def test_yaml_staircase_and_bad_style():
    root = LayoutLoader().load("assets/staircase.yaml", params={"rise": 2.4})
    stairs = root.find("stairs")
    assert stairs.tags == ["architecture.stairs"] and stairs.meta["walkable"]
    pts = (stairs.world_transform() @ np.c_[stairs.mesh.vertices, np.ones(len(stairs.mesh.vertices))].T)[1]
    assert pts.min() == pytest.approx(0, abs=1e-6) and pts.max() == pytest.approx(2.4, abs=1e-6)
    with pytest.raises(ValueError, match="style"):
        StairsGenerator(style="escalator").generate()
