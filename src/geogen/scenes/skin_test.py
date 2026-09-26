"""Skinning test scene: a capped tube on a three-joint chain, bent and animated.

The skinned-export spike (geogen-z2b.16.1): proves glTF skins, joint
animations and CPU skinning agree between Python renders and Godot. The
tube is exported in its bent ``pose`` (the joint nodes' current transforms);
its ``sway`` clip bends it back and forth.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.skin import Clip, Skin, Track, normalize_weights
from ..core.transform import quat_axis_angle
from ..materials.loader import MaterialLoader

JOINTS = ("Spine", "Chest", "UpperChest")
BONE_LENGTH = 0.5
RADIUS = 0.12
BLEND = 0.12            # half-width (m) of the weight blend zone around each joint
RINGS_PER_BONE = 10
SEGMENTS = 16


def tube_mesh(length: float, radius: float, rings: int, segments: int) -> Mesh:
    """A closed tube along +Y from 0 to ``length`` with pole-capped ends and metric UVs.

    Each ring repeats its first vertex at 360 degrees so the UV seam is split.
    """
    angles = np.linspace(0, 2 * np.pi, segments + 1)
    per_ring = segments + 1
    ys = np.linspace(0, length, rings + 1)
    ring = np.stack([np.cos(angles), np.zeros(per_ring), -np.sin(angles)], axis=1) * radius
    vertices = np.vstack([*(ring + [0, y, 0] for y in ys), [[0, 0, 0]], [[0, length, 0]]])
    bottom, top = len(vertices) - 2, len(vertices) - 1
    faces = []
    for r in range(rings):
        for s in range(segments):
            a, b = r * per_ring + s, r * per_ring + s + 1
            c, d = a + per_ring, b + per_ring
            faces += [[a, b, d], [a, d, c]]
    last = rings * per_ring
    for s in range(segments):
        faces.append([bottom, s + 1, s])
        faces.append([top, last + s, last + s + 1])
    uvs = np.vstack([np.stack([np.tile(angles, rings + 1) * radius, vertices[:-2, 1]], axis=1),
                     [[0, 0], [0, length]]])
    return Mesh(vertices, np.array(faces), uvs=uvs)


def chain_weights(heights: np.ndarray, bone_length: float, joints: int, blend: float):
    """Weights along a straight chain: each joint owns its bone, blending linearly across ``blend`` at joints."""
    starts = np.arange(joints) * bone_length
    raw = np.zeros((len(heights), joints))
    for j, start in enumerate(starts):
        lo = 1.0 if j == 0 else np.clip((heights - (start - blend)) / (2 * blend), 0, 1)
        hi = 1.0 if j == joints - 1 else 1 - np.clip((heights - (start + bone_length - blend)) / (2 * blend), 0, 1)
        raw[:, j] = lo * hi
    index = np.tile(np.arange(joints), (len(heights), 1))
    return normalize_weights(index, raw)


def sway_clip(duration: float = 2.0, keys: int = 17, amplitude_deg: float = 30.0) -> Clip:
    """Each joint bends about Z (phase-shifted), the base also turns about Y."""
    times = np.linspace(0, duration, keys)
    tracks = {}
    for j, name in enumerate(JOINTS):
        phase = 2 * np.pi * times / duration - j * 0.6
        bend = np.radians(amplitude_deg) * np.sin(phase)
        rotations = np.array([quat_axis_angle([0, 0, 1], a) for a in bend])
        if j == 0:
            turn = np.radians(20) * np.sin(2 * np.pi * times / duration)
            rotations = np.array([_mul(quat_axis_angle([0, 1, 0], t), q) for t, q in zip(turn, rotations)])
        tracks[name] = Track(times, rotations=rotations)
    return Clip("sway", tracks, loop=True)


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion product a * b (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


# Exported pose: time in the sway clip the joints are posed at.
POSE_TIME = 0.5


def build_skinned_tube(name: str = "tube") -> SceneNode:
    """A character-like subtree: joint chain + skinned tube (sibling of the root joint) + ``sway`` clip."""
    owner = SceneNode(name)
    parent = owner
    joints = []
    for j, joint_name in enumerate(JOINTS):
        joint = SceneNode(joint_name)
        joint.transform.translation = np.array([0.0, 0.0 if j == 0 else BONE_LENGTH, 0.0])
        parent.add_child(joint)
        joints.append(joint)
        parent = joint

    mesh = tube_mesh(BONE_LENGTH * len(JOINTS), RADIUS, RINGS_PER_BONE * len(JOINTS), SEGMENTS)
    mesh.joints, mesh.weights = chain_weights(mesh.vertices[:, 1], BONE_LENGTH, len(JOINTS), BLEND)
    mesh.material = MaterialLoader().load("tile_subway")
    body = owner.add_child(SceneNode("body", mesh=mesh))
    body.meta["collider"] = "none"
    body.skin = Skin.bind(body, joints, name=f"{name}_skin")

    owner.clips.append(sway_clip())
    owner.clips[0].apply(owner, POSE_TIME)
    return owner


def create_skin_test_scene() -> SceneNode:
    root = SceneNode("skin_test")
    tube = root.add_child(build_skinned_tube())
    # A second instance, turned and moved, checks skins survive SceneNode.instance().
    other = root.add_child(tube.instance())
    other.name = "tube_b"
    other.transform.translation = np.array([0.8, 0.0, 0.0])
    other.transform.rotation = np.array([0.0, np.radians(90), 0.0])
    return root
