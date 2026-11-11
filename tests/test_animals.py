"""Four-legged bodies (geogen.100): a skeleton with its own bones, a ring-lofted body from
chains, motions declared bone by bone, and a cat walking about as an NPC in the Godot runtime."""


import math

import numpy as np
import pytest

from geogen.core import meshops
from geogen.core.skeleton import load_skeleton
from geogen.npc import ASSETS_DIR


def test_a_skeleton_declares_its_own_bones():
    skeleton = load_skeleton(ASSETS_DIR / "skeletons" / "quadruped.yaml", {"length": 0.4})
    assert {"Hips", "Head", "Tail3", "LeftFrontFoot", "RightHindUpperLeg", "RightEar"} <= set(skeleton.bones)
    assert skeleton["RightHindFoot"].parent == "RightHindLowerLeg" and skeleton["Tail1"].parent == "Hips"
    # Left bones mirror to the right; the shoulders are `length` ahead of the hips.
    assert np.allclose(skeleton["RightFrontFoot"].head, skeleton["LeftFrontFoot"].head * [-1, 1, 1])
    assert skeleton["LeftFrontUpperLeg"].head[2] - skeleton["LeftHindUpperLeg"].head[2] == pytest.approx(0.38, abs=0.01)
    for bone in skeleton.bones.values():                                   # a rest frame: orthonormal, right-handed
        assert np.allclose(bone.rotation @ bone.rotation.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(bone.rotation) == pytest.approx(1.0)
    spine = skeleton["Spine"]                                              # +Y along the bone, toward its child
    along = skeleton["Chest"].head - spine.head
    assert np.allclose(spine.rotation[:, 1], along / np.linalg.norm(along))
    # The humanoid still takes its bones from the profile.
    assert "LeftHand" in load_skeleton().bones


def test_a_hierarchy_must_be_whole(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("kind: skeleton\nbones: {Root: [0, 0, 0]}\nhierarchy: {Root: null, Tail: Root}\n")
    with pytest.raises(ValueError, match="needs a position"):
        load_skeleton(bad)
    bad.write_text("kind: skeleton\nbones: {Root: [0, 0, 0], Tail: [0, 0, 1]}\nhierarchy: {Tail: Root, Root: null}\n")
    with pytest.raises(ValueError, match="before its parent"):
        load_skeleton(bad)


def test_the_cat_is_one_closed_skinned_body_with_its_motions(built_scene):
    root = built_scene("cat")
    body = next(n for n in root.iter_nodes() if n.mesh is not None)
    assert meshops.validate(body.mesh).ok
    lo, hi = body.mesh.vertices.min(axis=0), body.mesh.vertices.max(axis=0)
    assert lo[1] == pytest.approx(0.0, abs=0.01)                           # paws on the floor
    assert 0.25 < hi[1] < 0.42 and 0.5 < hi[2] - lo[2] < 0.8               # a cat's size
    assert len(body.mesh.faces) < 3000
    assert len({tuple(np.round(c, 2)) for c in body.mesh.colors[:, :3]}) >= 3      # coat, eyes, nose
    clips = {c.name: c for n in root.iter_nodes() for c in getattr(n, "clips", None) or []}
    assert {"walk", "idle"} <= set(clips)
    assert clips["walk"].meta["speed"] == 0.45 and clips["walk"].loop
    # The walk moves all four legs.
    assert {"LeftHindUpperLeg", "LeftFrontUpperLeg", "RightHindUpperLeg", "RightFrontUpperLeg"} <= set(clips["walk"].tracks)


@pytest.mark.godot
@pytest.mark.xdist_group("test_animals_runtime")
def test_cats_walk_about_as_npcs(scene_export, godot_runtime):
    rt = godot_runtime("cat_test", scene_export("cat_test"))
    rt.call("wait_ready")
    rt.call("time.pause")
    start = {n["npc"]: n for n in rt.call("npcs.list")}
    assert len(start) == 2 and all(n["pose"] == "stand" for n in start.values())
    walked = dict.fromkeys(start, 0.0)
    clips = {name: set() for name in start}
    last = {name: n["position"] for name, n in start.items()}
    for _ in range(80):
        rt.call("time.advance", seconds=0.5, scale=4)
        for n in rt.call("npcs.list"):
            walked[n["npc"]] += math.dist(n["position"], last[n["npc"]])
            last[n["npc"]] = n["position"]
            clips[n["npc"]].add(n["clip"])
    now = {n["npc"]: n for n in rt.call("npcs.list")}
    for name, n in now.items():
        assert walked[name] > 1.0, (name, walked)                          # it got about
        assert {"walk", "idle"} <= clips[name], clips                      # on its own walk and idle
        # (Where it meant to wander may be where the other cat or the player stands: that is all that fails.)
        assert all("destination occupied" in f for f in n["failures"]) and n["distinct"] >= 2
        assert abs(n["position"][0]) < 2.0 and abs(n["position"][2]) < 2.0     # in the room
