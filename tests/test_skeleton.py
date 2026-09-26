"""Humanoid skeleton (geogen-z2b.16.2): profile bones, proportions, poses, FK, weights through CSG/decimate."""

import numpy as np
import pytest

from geogen.core import csg, meshops
from geogen.core.mesh import Mesh
from geogen.core.skeleton import Pose, load_profile, load_skeleton, profile_rest_world
from geogen.core.skin import Skin
from geogen.generators.primitives import CubeGenerator
from geogen.scenes.skin_test import build_skinned_tube


@pytest.fixture(scope="module")
def skeleton():
    return load_skeleton()


def test_skeleton_has_every_profile_bone_in_profile_order(skeleton):
    profile = load_profile()
    assert len(profile) == 56
    assert skeleton.names == [b.name for b in profile]
    assert {b.name: b.parent for b in skeleton.bones.values()} == {b.name: b.parent for b in profile}


def test_rest_orientations_are_the_profile_reference(skeleton):
    reference = profile_rest_world(load_profile())
    for name, bone in skeleton.bones.items():
        np.testing.assert_allclose(bone.rotation, reference[name][:3, :3], atol=1e-6, err_msg=name)


def test_default_proportions(skeleton):
    top, chin = skeleton.landmarks["head_top"], skeleton.landmarks["chin"]
    assert top[1] == pytest.approx(1.70)
    assert top[1] / (top[1] - chin[1]) == pytest.approx(6.5)
    # T-pose facing +Z, left at +X; fingertip span about the height.
    assert skeleton["LeftHand"].head[0] > 0.5 and skeleton["RightHand"].head[0] < -0.5
    assert skeleton["LeftToes"].head[2] > skeleton["LeftFoot"].head[2]
    span = 2 * (skeleton["LeftMiddleDistal"].head[0] + 0.03)
    assert span == pytest.approx(1.70, abs=0.08)
    assert skeleton["Root"].head[1] == 0.0


def test_right_side_mirrors_left(skeleton):
    for name in skeleton.names:
        if name.startswith("Left"):
            left, right = skeleton[name].head, skeleton["Right" + name[4:]].head
            np.testing.assert_allclose(right, left * [-1, 1, 1], atol=1e-9, err_msg=name)


def test_params_move_heads_not_topology():
    tall = load_skeleton(params={"height": 1.90, "shoulder_width": 0.40})
    short = load_skeleton(params={"height": 1.50})
    assert tall.names == short.names
    assert tall.landmarks["head_top"][1] == pytest.approx(1.90)
    assert tall["LeftUpperArm"].head[0] == pytest.approx(0.20)
    assert short["Head"].head[1] < tall["Head"].head[1]
    with pytest.raises(Exception, match="Unknown param"):
        load_skeleton(params={"wingspan": 2})


def test_fk_rest_matches_bone_heads(skeleton):
    world = skeleton.fk()
    for name, bone in skeleton.bones.items():
        np.testing.assert_allclose(world[name][:3, 3], bone.head, atol=1e-9)


def test_fk_elbow_bend_moves_only_the_forearm_chain(skeleton):
    # Elbow flexion: the forearm swings forward about the body's up axis.
    pose = Pose.from_spec({"bones": {"LeftLowerArm": [0, -90, 0]}}, skeleton)
    world = skeleton.fk(pose)
    elbow, hand = world["LeftLowerArm"][:3, 3], world["LeftHand"][:3, 3]
    np.testing.assert_allclose(elbow, skeleton["LeftLowerArm"].head)
    assert np.linalg.norm(hand - elbow) == pytest.approx(skeleton.length("LeftLowerArm"))
    np.testing.assert_allclose((hand - elbow) / np.linalg.norm(hand - elbow), [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(world["RightHand"][:3, 3], skeleton["RightHand"].head, atol=1e-12)


def test_pose_root_offset_and_node_apply_match_fk(skeleton):
    pose = Pose.from_spec({"root": [0.5, 0, -1], "bones": {"Spine": [10, 20, 0], "LeftUpperLeg": [-45, 0, 0]}},
                          skeleton)
    root, nodes = skeleton.build()
    pose.apply(skeleton, nodes)
    world = skeleton.fk(pose)
    for name in ("Hips", "Head", "LeftFoot", "RightHand"):
        np.testing.assert_allclose(nodes[name].world_transform(), world[name], atol=1e-9, err_msg=name)
    np.testing.assert_allclose(world["Hips"][:3, 3] - skeleton["Hips"].head, [0.5, 0, -1], atol=1e-9)


def test_body_frame_poses_are_anatomical(skeleton):
    pose = Pose.from_spec({"bones": {"LeftUpperLeg": [-90, 0, 0], "LeftUpperArm": [0, 0, -90],
                                     "RightUpperArm": [0, 0, 90]}}, skeleton)
    world = skeleton.fk(pose)

    def direction(bone, child):
        d = world[child][:3, 3] - world[bone][:3, 3]
        return d / np.linalg.norm(d)

    np.testing.assert_allclose(direction("LeftUpperLeg", "LeftLowerLeg"), [0, 0, 1], atol=1e-9)   # thigh forward
    np.testing.assert_allclose(direction("LeftUpperArm", "LeftLowerArm"), [0, -1, 0], atol=1e-9)  # arms down
    np.testing.assert_allclose(direction("RightUpperArm", "RightLowerArm"), [0, -1, 0], atol=1e-9)
    # The same rotation in bone frame means something else for a rolled bone.
    bone_frame = skeleton.fk(Pose.from_spec({"frame": "bone", "bones": {"LeftUpperArm": [0, 0, -90]}}))
    d = bone_frame["LeftLowerArm"][:3, 3] - bone_frame["LeftUpperArm"][:3, 3]
    assert not np.allclose(d / np.linalg.norm(d), [0, -1, 0], atol=1e-3)


def test_built_nodes_start_at_rest(skeleton):
    root, nodes = skeleton.build()
    assert root.name == "Root" and len(nodes) == 56
    for name, bone in skeleton.bones.items():
        np.testing.assert_allclose(nodes[name].world_transform(), bone.rest_world, atol=1e-9, err_msg=name)


def _skinned_cube() -> Mesh:
    cube = CubeGenerator(size_x=1.0, size_y=1.0, size_z=1.0, bevel=0).generate()
    t = np.clip(cube.vertices[:, 1] + 0.5, 0, 1)
    cube.joints = np.tile([3, 7, 0, 0], (len(cube.vertices), 1))
    cube.weights = np.column_stack([1 - t, t, np.zeros(len(t)), np.zeros(len(t))])
    return cube


def test_csg_carries_weights():
    cube = _skinned_cube()
    cutter = CubeGenerator(size_x=0.4, size_y=2.0, size_z=0.4, bevel=0).generate()   # unskinned
    result = csg.difference(cube, cutter)
    assert result.joints is not None and len(result.joints) == len(result.vertices)
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0)
    assert set(np.unique(result.joints[result.weights > 0])) <= {3, 7}
    # Weight on joint 7 still follows height (interpolated, not scrambled).
    w7 = np.where(result.joints == 7, result.weights, 0).sum(axis=1)
    np.testing.assert_allclose(w7, np.clip(result.vertices[:, 1] + 0.5, 0, 1), atol=1e-3)

    union = csg.union(cube, CubeGenerator(size_x=0.3, size_y=0.3, size_z=0.3, bevel=0).generate().transform(
        np.array([[1, 0, 0, 0.6], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])))
    np.testing.assert_allclose(union.weights.sum(axis=1), 1.0)   # unskinned operand filled from neighbours


def test_decimate_carries_weights():
    mesh = build_skinned_tube().find("body").mesh
    reduced = meshops.decimate(mesh, 0.4)
    assert len(reduced.faces) < len(mesh.faces)
    assert len(reduced.joints) == len(reduced.vertices)
    np.testing.assert_allclose(reduced.weights.sum(axis=1), 1.0)
    assert set(np.unique(reduced.joints[reduced.weights > 0])) <= {0, 1, 2}


def test_merge_keeps_weights_aligned():
    a, b = _skinned_cube(), CubeGenerator(size_x=1.0, size_y=1.0, size_z=1.0).generate()
    merged = Mesh.merge([a, b])
    assert merged.joints.shape == (len(merged.vertices), 4)
    np.testing.assert_allclose(merged.weights.sum(axis=1), 1.0)


def test_skeleton_nodes_skin_a_mesh(skeleton):
    """The built joint nodes plug straight into core.skin."""
    from geogen.core.node import SceneNode

    owner = SceneNode("person")
    root, nodes = skeleton.build()
    owner.add_child(root)
    cube = _skinned_cube()
    cube.joints = np.tile([skeleton.names.index("LeftLowerArm"), 0, 0, 0], (len(cube.vertices), 1))
    cube.weights = np.tile([1.0, 0, 0, 0], (len(cube.vertices), 1))
    body = owner.add_child(SceneNode("body", mesh=cube))
    body.skin = Skin.bind(body, [nodes[n] for n in skeleton.names])
    Pose.from_spec({"root": [0, 0, 2]}).apply(skeleton, nodes)
    np.testing.assert_allclose(body.world_mesh().vertices, cube.vertices + [0, 0, 2], atol=1e-9)
