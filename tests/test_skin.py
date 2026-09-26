"""Skeletal skinning: CPU skinning, clips, instancing and glTF skin export (geogen-z2b.16.1)."""

import json
from pathlib import Path

import numpy as np
import pytest

from geogen.core import meshops
from geogen.core.skin import Clip, Track, pose_clips, skin_mesh
from geogen.core.transform import quat_axis_angle, slerp
from geogen.export import _split_glb, export_scene
from geogen.scenes.skin_test import JOINTS, build_skinned_tube, create_skin_test_scene


def test_bind_pose_leaves_mesh_unchanged():
    tube = build_skinned_tube()
    body = tube.find("body")
    for joint in body.skin.joints:
        joint.transform.rotation[:] = 0          # back to the straight rest chain
    np.testing.assert_allclose(body.world_mesh().vertices, body.mesh.vertices, atol=1e-12)


def test_weights_are_normalised_and_blend_across_joints():
    mesh = build_skinned_tube().find("body").mesh
    np.testing.assert_allclose(mesh.weights.sum(axis=1), 1.0)
    at_joint = np.isclose(mesh.vertices[:, 1], 0.5)
    assert np.all(mesh.weights[at_joint][:, :2] == pytest.approx(0.5))


def test_single_joint_moves_its_vertices_rigidly():
    tube = build_skinned_tube()
    body = tube.find("body")
    mats = np.stack([np.eye(4)] * len(JOINTS))
    mats[2][:3, 3] = [0.0, 0.0, 1.0]                 # translate the top joint only
    posed = skin_mesh(body.mesh, mats)
    top = body.mesh.weights[:, 0] == 1.0
    top_only = (body.mesh.joints[:, 0] == 2) & top
    np.testing.assert_allclose(posed.vertices[top_only] - body.mesh.vertices[top_only], [[0, 0, 1]] * top_only.sum())
    bottom_only = (body.mesh.joints[:, 0] == 0) & top
    np.testing.assert_allclose(posed.vertices[bottom_only], body.mesh.vertices[bottom_only])


def test_posed_mesh_stays_closed():
    root = create_skin_test_scene()
    for _, mesh in root.iter_meshes():
        assert meshops.validate(mesh).ok


def test_track_sampling_matches_keys_and_slerps_between():
    a, b = quat_axis_angle([0, 0, 1], 0.0), quat_axis_angle([0, 0, 1], np.pi / 2)
    track = Track([0.0, 1.0], rotations=[a, b], translations=[[0, 0, 0], [0, 2, 0]])
    rot, trans = track.sample(0.5)
    np.testing.assert_allclose(rot, quat_axis_angle([0, 0, 1], np.pi / 4), atol=1e-12)
    np.testing.assert_allclose(trans, [0, 1, 0])
    np.testing.assert_allclose(track.sample(5.0)[0], b)          # clamped
    np.testing.assert_allclose(slerp(a, -b, 0.5), rot, atol=1e-12)  # shortest arc


def test_looping_clip_wraps():
    tube = build_skinned_tube()
    clip = tube.clips[0]
    clip.apply(tube, 0.25)
    first = tube.find("UpperChest").world_transform()
    clip.apply(tube, 0.25 + clip.duration)
    np.testing.assert_allclose(tube.find("UpperChest").world_transform(), first, atol=1e-9)


def test_unknown_clip_joint_raises():
    tube = build_skinned_tube()
    with pytest.raises(KeyError, match="Hips"):
        Clip("bad", {"Hips": Track([0.0], rotations=[[0, 0, 0, 1]])}).apply(tube, 0.0)


@pytest.mark.parametrize("copy", ["instance", "copy"])
def test_copies_remap_skin_joints(copy):
    tube = build_skinned_tube()
    other = tube.instance() if copy == "instance" else tube.copy()
    joints = other.find("body").skin.joints
    assert [j.name for j in joints] == list(JOINTS)
    assert all(j.root is other for j in joints)
    # Posing the copy leaves the original alone.
    before = tube.find("body").world_mesh().vertices.copy()
    pose_clips(other, "sway", 1.3)
    np.testing.assert_allclose(tube.find("body").world_mesh().vertices, before)
    assert not np.allclose(other.find("body").world_mesh().vertices, before)


def test_normals_and_weld_carry_weights():
    mesh = build_skinned_tube().find("body").mesh
    for derived in (meshops.compute_normals(mesh), meshops.weld_vertices(mesh)):
        assert len(derived.joints) == len(derived.weights) == len(derived.vertices)


@pytest.fixture(scope="module")
def skin_glb(tmp_path_factory):
    out = tmp_path_factory.mktemp("skin") / "skin_test.glb"
    export_scene(create_skin_test_scene(), out)
    gltf, _, _ = _split_glb(out.read_bytes())
    return out, gltf


def test_glb_has_skins_with_joint_attributes(skin_glb):
    _, gltf = skin_glb
    nodes = gltf["nodes"]
    assert len(gltf["skins"]) == 2
    for skin in gltf["skins"]:
        # Every character keeps the plain joint names (retargeting matches bones by name).
        assert [nodes[j]["name"] for j in skin["joints"]] == list(JOINTS)
        assert nodes[skin["skeleton"]]["name"] == "Spine"
        assert gltf["accessors"][skin["inverseBindMatrices"]]["type"] == "MAT4"
    skinned = [n for n in nodes if "skin" in n]
    assert len({n["mesh"] for n in skinned}) == 1           # instances share the mesh
    for primitive in gltf["meshes"][skinned[0]["mesh"]]["primitives"]:
        attrs = primitive["attributes"]
        counts = {k: gltf["accessors"][attrs[k]]["count"] for k in ("POSITION", "JOINTS_0", "WEIGHTS_0")}
        assert len(set(counts.values())) == 1, counts
        assert gltf["accessors"][attrs["JOINTS_0"]]["componentType"] == 5123
    # No rest-pose colliders for skinned meshes.
    assert not any(n["name"].endswith("colonly") for n in nodes)


def test_glb_clips_are_animations_on_joints(skin_glb):
    _, gltf = skin_glb
    names = {a["name"] for a in gltf["animations"]}
    assert names == {"tube_sway", "tube_b_sway"}
    extras = next(n for n in gltf["nodes"] if n["name"] == "tube")["extras"]["geogen"]
    assert extras["clips"] == [{"name": "sway", "animation": "tube_sway", "duration": 2.0, "loop": True}]
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).parent.parent / "docs/schema/geogen-extras.v1.schema.json").read_text())
    jsonschema.validate({"geogen": extras}, schema)


def test_godot_skins_match_python_pose(run_godot, skin_glb):
    """Godot imports Skeleton3D + skin + AnimationPlayer; at t = 1.0 (a clip key on a 30 fps frame,
    Godot's import bake rate) its bones and skinned bounds equal the Python CPU skinning."""
    path, _ = skin_glb
    out = run_godot("--scene", "skin_test", f"--generated={path.parent}", "--play=sway@1.0", "--skeletons",
                    "--quit-after=3")
    report = json.loads(next(l for l in out.splitlines() if l.startswith("skeletons: "))
                        .removeprefix("skeletons: "))
    assert set(sum(report["animations"].values(), [])) == {"tube_b_sway", "tube_sway"}

    root = create_skin_test_scene()
    pose_clips(root, "sway", 1.0)
    expected = sorted(((n.parent.name, m) for n, m in root.iter_meshes()), key=lambda item: item[0])
    skeletons = sorted(report["skeletons"], key=lambda s: s["path"])
    assert len(skeletons) == 2
    for (owner, mesh), skeleton in zip(expected, skeletons):
        assert f"/{owner}/" in skeleton["path"]
        assert sorted(skeleton["bones"]) == sorted(JOINTS)
        joint = root.find(owner).find("UpperChest").world_transform()[:3, 3]
        np.testing.assert_allclose(skeleton["bones"]["UpperChest"], joint, atol=1e-3)
        (bounds,) = skeleton["meshes"]
        np.testing.assert_allclose(bounds["min"], mesh.vertices.min(axis=0), atol=1e-3)
        np.testing.assert_allclose(bounds["max"], mesh.vertices.max(axis=0), atol=1e-3)
