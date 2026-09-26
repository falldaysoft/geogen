"""Affordance QA (layout/affordance_qa.py): posed actors, approach reachability, pose clashes."""

from pathlib import Path

import numpy as np
import pytest

from geogen.core.node import SceneNode
from geogen.core.transform import Transform
from geogen.generators.primitives import CubeGenerator
from geogen.layout.affordance_qa import affordance_spots, check_affordances, stage_actors
from geogen.layout.loader import LayoutLoader

ASSETS = Path(__file__).parent.parent / "assets"


@pytest.fixture(scope="module")
def loader():
    return LayoutLoader()


@pytest.fixture(scope="module")
def body(loader):
    return loader.load(ASSETS / "characters" / "humanoid.yaml", params={"preset": "feminine"})


def _box(name, size, at):
    node = SceneNode(name, mesh=CubeGenerator(size_x=size[0], size_y=size[1], size_z=size[2], bevel=0).generate())
    node.transform = Transform(translation=np.array([at[0], size[1] / 2 + at[1], at[2]]))
    return node


def _scene(loader, *, enclosed=False, table_at=None):
    root = SceneNode("test")
    chair = loader.load(ASSETS / "chair.yaml")
    chair.name = "chair"
    root.add_child(chair)
    root.add_child(SceneNode("spawn", transform=Transform(translation=np.array([0.0, 0.0, 4.0])),
                             meta={"type": "spawn"}))
    if enclosed:     # a closed 2.4 m pen around the chair; the spawn is outside
        for name, size, at in (("n", (2.4, 2, 0.1), (0, 0, -1.2)), ("s", (2.4, 2, 0.1), (0, 0, 1.2)),
                               ("w", (0.1, 2, 2.4), (-1.2, 0, 0)), ("e", (0.1, 2, 2.4), (1.2, 0, 0))):
            root.add_child(_box(name, size, at))
    if table_at is not None:
        root.add_child(_box("table", (1.0, 0.04, 1.0), table_at))
    return root


def test_open_chair_passes(loader, body):
    assert [str(i) for i in check_affordances(_scene(loader), body=body)] == []


def test_enclosed_chair_is_unreachable(loader, body):
    issues = check_affordances(_scene(loader, enclosed=True), body=body, clash=False)
    assert [i.kind for i in issues] == ["approach_unreachable"]


def test_blocked_approach(loader, body):
    root = _scene(loader)
    spot = affordance_spots(root)[0]
    # A crate right where the sitter would stand.
    root.add_child(_box("crate", (0.9, 1.2, 0.9), (spot.approach[0], 0, spot.approach[2])))
    assert "approach_blocked" in [i.kind for i in check_affordances(root, body=body, clash=False)]


def test_table_through_the_lap_clashes(loader, body):
    # A table top at thigh height, right over the seat.
    issues = check_affordances(_scene(loader, table_at=(0, 0.62, 0.25)), body=body)
    clash = [i for i in issues if i.kind == "pose_clash"]
    assert clash and "table" in clash[0].items


def test_stage_actors_poses_one_body_per_affordance(loader, body):
    root = _scene(loader)
    staged = stage_actors(root, body=body)
    actors = staged.find("affordance_actors").children
    assert len(actors) == len(affordance_spots(root)) == 1
    # Seated: the hips sit near the seat height, not standing height.
    hips = actors[0].find("Hips")
    assert hips is not None and hips.world_transform()[1, 3] < 0.75
