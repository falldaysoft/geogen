"""Procedural trees (space colonisation) and rocks (generators/nature.py)."""

import numpy as np
import pytest

from geogen.core import meshops, uvmap
from geogen.generators.nature import RockGenerator, TreeGenerator
from geogen.layout.loader import LayoutLoader


@pytest.mark.parametrize("style", ["deciduous", "conifer"])
def test_tree_skeleton_and_radii(style):
    gen = TreeGenerator(width=4, height=5, depth=4, style=style, seed=3)
    nodes, parents = gen.skeleton()
    assert parents[0] == -1 and (parents[1:] >= 0).all() and (parents[1:] < np.arange(1, len(parents))).all()
    assert len(nodes) > 40
    assert nodes[:, 1].max() <= 5.3 and np.abs(nodes[:, [0, 2]]).max() <= 2.6
    radii = gen.radii(parents)
    assert radii[0] == pytest.approx(gen.trunk_radius)
    assert (radii[1:] <= radii[parents[1:]] + 1e-9).all()      # never thicker than the parent
    again, _ = TreeGenerator(width=4, height=5, depth=4, style=style, seed=3).skeleton()
    other, _ = TreeGenerator(width=4, height=5, depth=4, style=style, seed=4).skeleton()
    np.testing.assert_array_equal(nodes, again)
    assert len(other) != len(nodes) or not np.allclose(other, nodes)


@pytest.mark.parametrize("style", ["deciduous", "conifer"])
def test_tree_node_meshes(style):
    node = TreeGenerator(width=4, height=5, depth=4, style=style, seed=1).to_node("tree")
    branches, foliage = node.find("branches"), node.find("foliage")
    assert branches is not None and foliage is not None
    for mesh in (node.mesh, branches.mesh, foliage.mesh):
        assert not meshops.validate(mesh).issues
    assert node.meta["collider"] == "hull"
    assert branches.meta["collider"] == "none" and foliage.meta["collider"] == "none"
    assert foliage.mesh.material is not None
    assert uvmap.texel_density(node.mesh) == pytest.approx(1.0, rel=0.15)   # bark UVs are metric
    # The part frame is centred: the trunk base sits at -height / 2.
    assert node.mesh.vertices[:, 1].min() == pytest.approx(-2.5, abs=0.05)


def test_conifer_crown_tapers_and_deciduous_is_round():
    def crown_widths(style):
        leaves = TreeGenerator(width=4, height=5, depth=4, style=style, seed=2).to_node().find("foliage").mesh.vertices
        low = leaves[leaves[:, 1] < np.percentile(leaves[:, 1], 30)]
        high = leaves[leaves[:, 1] > np.percentile(leaves[:, 1], 85)]
        return np.ptp(low[:, 0]), np.ptp(high[:, 0])

    low, high = crown_widths("conifer")
    assert high < 0.6 * low
    low, high = crown_widths("deciduous")
    assert high > 0.5 * low


def test_tree_asset_uses_bark_everywhere():
    tree = LayoutLoader().load("assets/maple_tree.yaml")
    part = next(n for n in tree.iter_nodes() if n.name == "tree")
    assert part.mesh.material.name == "bark"
    assert part.find("branches").mesh.material.name == "bark"
    assert part.find("foliage").mesh.material.name == "foliage_green"
    pine = LayoutLoader().load("assets/pine_tree.yaml")
    assert next(n for n in pine.iter_nodes() if n.name == "foliage").mesh.material.name == "foliage_pine"


def test_rocks_are_closed_varied_and_fill_their_box():
    shapes = []
    for seed in range(6):
        mesh = RockGenerator(width=1.2, height=0.8, depth=1.0, seed=seed).generate()
        assert not meshops.validate(mesh).issues
        extent = np.ptp(mesh.vertices, axis=0)
        np.testing.assert_allclose(extent, [1.2, 0.8, 1.0], atol=0.2)
        # A flat base to sit on.
        low = mesh.vertices[:, 1] < mesh.vertices[:, 1].min() + 1e-6
        assert low.sum() >= 5
        shapes.append(mesh.vertices)
    assert not np.allclose(shapes[0][:50], shapes[1][:50])
