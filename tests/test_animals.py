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


def test_one_shot_clips_carry_the_way_a_jump_goes():
    from geogen.core.skeleton import Pose
    from geogen.generators.clips import cycle_clip, leap_at

    skeleton = load_skeleton(ASSETS_DIR / "skeletons" / "quadruped.yaml")
    clip = cycle_clip(skeleton, Pose(), "jump_up", once=True, duration=0.5,
                      bones={"Hips": {"path": [[0, 0, 0], [-30, 0, 0], [0, 0, 0]]}},
                      leap={"run": [0, 0.5, 1], "rise": [0, 1, 1], "lift": [0, 0.1, 0]})
    assert not clip.loop and clip.duration == pytest.approx(0.5)
    leap = clip.meta["leap"]
    assert len(leap["times"]) == len(leap["run"]) == len(leap["lift"])
    start, end = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.8])
    assert np.allclose(leap_at(leap, start, end, 0.0), start) and np.allclose(leap_at(leap, start, end, 0.5), end)
    mid = leap_at(leap, start, end, 0.25)
    assert mid[2] == pytest.approx(0.4, abs=0.02) and mid[1] == pytest.approx(0.6, abs=0.02)   # risen, and lifted
    # The pitch keys spread over the clip: the start and the end are the pose, the middle not.
    q = clip.tracks["Hips"].rotations
    assert np.allclose(np.abs(q[0]), np.abs(q[-1])) and not np.allclose(np.abs(q[0]), np.abs(q[len(q) // 2]))
    with pytest.raises(ValueError, match="one-shot"):
        cycle_clip(skeleton, Pose(), "bad", leap={"run": [0, 1]})
    with pytest.raises(ValueError, match="no cycles"):
        cycle_clip(skeleton, Pose(), "bad", once=True, bones={"Hips": {"path": [[0, 0, 0]], "cycles": 2}})


def body_root(root):
    return next(n for n in root.iter_nodes() if "humanoid" in n.meta)


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
    assert {"walk", "idle", "jump_up", "jump_down", "pose_lie"} <= set(clips)
    assert clips["walk"].meta["speed"] == 0.45 and clips["walk"].loop
    assert not clips["jump_up"].loop and set(clips["jump_up"].meta["leap"]) == {"times", "run", "rise", "lift"}
    # It carries in its mouth, under the front of its head.
    assert body_root(root).meta["humanoid"]["carry"]["bone"] == "Head"
    # The walk moves all four legs.
    assert {"LeftHindUpperLeg", "LeftFrontUpperLeg", "RightHindUpperLeg", "RightFrontUpperLeg"} <= set(clips["walk"].tracks)


def test_a_bird_is_a_winged_body_that_hops_and_flies(built_scene):
    root = built_scene("sparrow")
    body = next(n for n in root.iter_nodes() if n.mesh is not None)
    assert meshops.validate(body.mesh).ok
    lo, hi = body.mesh.vertices.min(axis=0), body.mesh.vertices.max(axis=0)
    assert lo[1] == pytest.approx(0.0, abs=0.006)                          # feet on the ground
    assert 0.08 < hi[1] < 0.12 and 0.12 < hi[2] - lo[2] < 0.2              # a sparrow's size
    assert hi[0] - lo[0] > 0.2                                             # built with its wings out ...
    assert body_root(root).size[0] < 0.07                                 # ... standing with them folded
    clips = {c.name: c for n in root.iter_nodes() for c in getattr(n, "clips", None) or []}
    assert {"idle", "walk", "fly", "pose_fly"} <= set(clips)
    assert clips["fly"].meta["speed"] > clips["walk"].meta["speed"] > 0
    assert {"LeftWing1", "RightWing1"} <= set(clips["fly"].tracks)
    skeleton = load_skeleton(ASSETS_DIR / "skeletons" / "bird.yaml", {"size": 2.0})   # a pigeon's
    span = skeleton["LeftWingTip"].head[0] - skeleton["RightWingTip"].head[0]
    assert 0.4 < span < 0.6


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
    perched = dict.fromkeys(start, False)
    for _ in range(150):
        rt.call("time.advance", seconds=2.0, scale=8)
        for n in rt.call("npcs.list"):
            walked[n["npc"]] += math.dist(n["position"], last[n["npc"]])
            last[n["npc"]] = n["position"]
            clips[n["npc"]].add(n["clip"])
            if n["perched"]:
                # Up on the sofa's seat (0.5 m), lying there or about to.
                assert n["position"][1] == pytest.approx(0.5, abs=0.05) and n["position"][0] < -1.0, n
                perched[n["npc"]] = True
    now = {n["npc"]: n for n in rt.call("npcs.list")}
    # The toy mouse went about in a mouth, and was dropped again (geogen.100.8).
    assert sum(n.get("picked_up", 0) for n in now.values()) >= 1, now
    for name, n in now.items():
        assert walked[name] > 1.0, (name, walked)                          # it got about
        assert {"walk", "idle", "pose_lie"} <= clips[name], clips          # on its own walk and idle; napped
        # It jumped up onto the sofa and down again (geogen.100.6).
        assert perched[name] and n["leaps"] >= 2, (name, n["leaps"])
        # (Where it meant to wander may be where the other cat or the player stands: that is all that fails.)
        assert all("destination occupied" in f for f in n["failures"]) and n["distinct"] >= 2
        assert abs(n["position"][0]) < 2.0 and abs(n["position"][2]) < 2.0     # in the room
