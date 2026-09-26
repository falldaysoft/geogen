"""Ring-loft skins (geogen-z2b.16.3): closed skinned tubes along bone chains, unioned into one body."""

import numpy as np
import pytest
import trimesh

from geogen.core import meshops
from geogen.core.node import SceneNode
from geogen.core.skeleton import Pose, load_skeleton
from geogen.core.skin import Skin
from geogen.generators.ringloft import Chain, Ring, loft_body, loft_chain


@pytest.fixture(scope="module")
def skeleton():
    return load_skeleton()


def arm(skeleton, sides=10, **kw) -> Chain:
    return Chain("arm", ["LeftUpperArm", "LeftLowerArm", "LeftHand"], [
        Ring("LeftUpperArm", 0.0, 0.050, 0.052), Ring("LeftUpperArm", 0.5, 0.042, 0.045),
        Ring("LeftLowerArm", 0.0, 0.034, 0.036), Ring("LeftLowerArm", 0.35, 0.036, 0.038),
        Ring("LeftLowerArm", 1.0, 0.022, 0.030)], end=skeleton["LeftHand"].head + [0.02, 0, 0],
        sides=sides, spacing=0.04, blend=0.06, **kw)


def leg(skeleton) -> Chain:
    return Chain("leg", ["LeftUpperLeg", "LeftLowerLeg"], [
        Ring("LeftUpperLeg", -0.1, 0.080, 0.085), Ring("LeftUpperLeg", 0.6, 0.060, 0.062),
        Ring("LeftLowerLeg", 0.0, 0.048, 0.050), Ring("LeftLowerLeg", 0.3, 0.050, 0.055),
        Ring("LeftLowerLeg", 1.0, 0.032, 0.035)], end=skeleton["LeftFoot"].head, sides=10, spacing=0.045,
        blend=0.07)


def torso(skeleton) -> Chain:
    return Chain("torso", ["Hips", "Spine", "Chest", "UpperChest"], [
        Ring("Hips", -0.3, 0.14, 0.10, power=2.6), Ring("Chest", 0.5, 0.13, 0.09, power=2.6),
        Ring("UpperChest", 1.0, 0.07, 0.06)], end=skeleton["Neck"].head, sides=14, spacing=0.05)


def test_chain_is_closed_with_metric_uvs_and_weights(skeleton):
    mesh = loft_chain(skeleton, arm(skeleton))
    report = meshops.validate(mesh)
    assert report.ok, report.issues
    np.testing.assert_allclose(mesh.weights.sum(axis=1), 1.0)
    names = skeleton.names
    used = {names[j] for j in np.unique(mesh.joints[mesh.weights > 0])}
    assert used == {"LeftUpperArm", "LeftLowerArm", "LeftHand"}
    # v runs along the chain in metres: shoulder -> wrist (the last key ring) plus both round caps.
    assert np.ptp(mesh.uvs[:, 1]) == pytest.approx(
        np.linalg.norm(skeleton["LeftHand"].head - skeleton["LeftUpperArm"].head) + 0.050 + 0.022, abs=1e-6)


def test_rings_hit_their_radii(skeleton):
    mesh = loft_chain(skeleton, arm(skeleton, sides=12))
    # At the upper-arm head ring the tube is 2 * 0.052 deep (front axis = +Z) and 0.1 tall.
    x0 = skeleton["LeftUpperArm"].head[0]
    at = np.abs(mesh.vertices[:, 0] - x0) < 1e-6
    assert np.ptp(mesh.vertices[at, 2]) == pytest.approx(0.104, abs=1e-3)
    assert np.ptp(mesh.vertices[at, 1]) == pytest.approx(0.100, abs=1e-3)


def test_blend_zone_spans_the_joint(skeleton):
    chain = arm(skeleton)
    mesh = loft_chain(skeleton, chain)
    elbow_x = skeleton["LeftLowerArm"].head[0]
    upper = skeleton.names.index("LeftUpperArm")
    w_upper = np.where(mesh.joints == upper, mesh.weights, 0).sum(axis=1)
    x = mesh.vertices[:, 0]
    assert np.all(w_upper[x < elbow_x - chain.blend - 1e-6] == pytest.approx(1.0))
    assert np.all(w_upper[(x > elbow_x + chain.blend + 1e-6) & (x < skeleton["LeftHand"].head[0] - 0.07)] == 0)
    assert np.all((w_upper[np.abs(x - elbow_x) < 1e-6] - 0.5) ** 2 < 1e-9)


def test_bind_spreads_a_bone_over_others(skeleton):
    chain = arm(skeleton, bind={"LeftHand": ["LeftIndexProximal", "LeftMiddleProximal"]})
    mesh = loft_chain(skeleton, chain)
    used = {skeleton.names[j] for j in np.unique(mesh.joints[mesh.weights > 0])}
    assert "LeftHand" not in used and {"LeftIndexProximal", "LeftMiddleProximal"} <= used


def _posed_volume(skeleton, mesh, spec) -> float:
    owner = SceneNode("o")
    root, nodes = skeleton.build()
    owner.add_child(root)
    body = owner.add_child(SceneNode("b", mesh=mesh))
    body.skin = Skin.bind(body, [nodes[n] for n in skeleton.names])
    if spec:
        Pose.from_spec(spec, skeleton).apply(skeleton, nodes)
    posed = body.world_mesh()
    assert meshops.validate(posed).watertight
    return abs(trimesh.Trimesh(posed.vertices, posed.faces, process=False).volume)


@pytest.mark.parametrize("chain, spec", [
    ("arm", {"bones": {"LeftLowerArm": [0, -90, 0]}}),
    ("leg", {"bones": {"LeftLowerLeg": [90, 0, 0]}}),
])
def test_ninety_degree_bend_keeps_volume(skeleton, chain, spec):
    mesh = loft_chain(skeleton, {"arm": arm, "leg": leg}[chain](skeleton))
    rest = _posed_volume(skeleton, mesh, None)
    bent = _posed_volume(skeleton, mesh, spec)
    assert bent == pytest.approx(rest, rel=0.06)   # no candy-wrapper collapse


def test_body_union_is_one_watertight_skinned_mesh(skeleton):
    shoulder = arm(skeleton)
    shoulder.rings.insert(0, Ring("LeftUpperArm", -0.4, 0.05, 0.05))    # reach into the torso
    chains = [torso(skeleton), shoulder, leg(skeleton)]
    body = loft_body(skeleton, chains)
    report = meshops.validate(body)
    assert report.ok, report.issues
    assert len(body.joints) == len(body.vertices)
    np.testing.assert_allclose(body.weights.sum(axis=1), 1.0)
    separate = sum(len(loft_chain(skeleton, c).faces) for c in chains)
    assert len(body.faces) < separate * 1.2            # union doesn't explode the tri count
    parts = trimesh.Trimesh(body.vertices, body.faces, process=True).split(only_watertight=False)
    assert len(parts) == 1


def test_bad_chains_raise(skeleton):
    with pytest.raises(ValueError, match="distinct"):
        loft_chain(skeleton, Chain("x", ["LeftUpperArm"], [Ring("LeftUpperArm", 0.2, 0.05, 0.05),
                                                          Ring("LeftUpperArm", 0.2, 0.04, 0.04)]))
    with pytest.raises(ValueError, match="parallel"):
        loft_chain(skeleton, Chain("x", ["LeftUpperLeg"], [Ring("LeftUpperLeg", 0, 0.05, 0.05),
                                                          Ring("LeftUpperLeg", 1, 0.04, 0.04)], front=(0, 1, 0)))
