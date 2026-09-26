"""Skeletal skinning: skins bound to joint nodes, CPU linear-blend skinning, clips.

A skeleton is an ordinary ``SceneNode`` hierarchy of joints. A skinned mesh
node carries a ``Skin``: the joints its mesh's ``joints``/``weights`` index,
and one inverse bind matrix per joint taking the mesh's frame to that joint's
frame at bind (rest) time. Posing moves the joint nodes; ``SceneNode.world_mesh``
then deforms the mesh on the CPU (linear blend skinning), so the renderer,
viewer and QA see the posed body. The exporter writes the same data as a glTF
skin (JOINTS_0 / WEIGHTS_0, inverseBindMatrices), so engines skin on the GPU.

Keep the skinned mesh node outside the joint hierarchy (a sibling of the root
joint): glTF ignores a skinned node's own transform, and Godot turns joints
into Skeleton3D bones.

A ``Clip`` is a named set of joint tracks (times, rotations as x/y/z/w
quaternions, optional translations) sampled like glTF LINEAR channels; it is
exported as a glTF animation targeting the joint nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .mesh import Mesh
from .transform import Transform, matrix_from_quat, slerp

if TYPE_CHECKING:
    from .node import SceneNode


@dataclass
class Skin:
    """Joints driving a skinned mesh node, with inverse bind matrices (J x 4 x 4)."""

    joints: list[SceneNode]
    inverse_bind: NDArray[np.float64]
    name: str = "skin"

    @classmethod
    def bind(cls, mesh_node: SceneNode, joints: list[SceneNode], name: str = "skin") -> Skin:
        """Bind ``mesh_node``'s mesh to ``joints`` in their current (rest) pose."""
        mesh_world = mesh_node.world_transform()
        inverse_bind = np.array([np.linalg.inv(j.world_transform()) @ mesh_world for j in joints])
        return cls(list(joints), inverse_bind, name)

    def matrices(self, mesh_node: SceneNode) -> NDArray[np.float64]:
        """Per-joint skinning matrices taking bind-pose mesh vertices to posed ones, in the mesh node's frame."""
        to_mesh = np.linalg.inv(mesh_node.world_transform())
        return np.array([to_mesh @ j.world_transform() @ ibm for j, ibm in zip(self.joints, self.inverse_bind)])


def skin_mesh(mesh: Mesh, matrices: NDArray[np.float64]) -> Mesh:
    """Linear blend skinning: ``mesh`` deformed by per-joint ``matrices`` (see ``Skin.matrices``)."""
    if mesh.joints is None or mesh.weights is None:
        return mesh
    # Blend the matrices per vertex, then apply: sum_k w_k M_{j_k}.
    blended = np.einsum("vk,vkij->vij", mesh.weights, matrices[mesh.joints])
    vertices = np.einsum("vij,vj->vi", blended[:, :3, :3], mesh.vertices) + blended[:, :3, 3]
    normals = None
    if mesh.normals is not None:
        normals = np.einsum("vij,vj->vi", blended[:, :3, :3], mesh.normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    posed = mesh.copy()
    posed.vertices, posed.normals = vertices, normals
    return posed


def normalize_weights(joints: NDArray[np.int64], weights: NDArray[np.float64],
                      max_influences: int = 4) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Keep each vertex's strongest ``max_influences`` joints, padded to 4, weights summing to 1."""
    order = np.argsort(-weights, axis=1)[:, :max_influences]
    j = np.take_along_axis(joints, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    if j.shape[1] < 4:
        pad = 4 - j.shape[1]
        j = np.hstack([j, np.zeros((len(j), pad), np.int64)])
        w = np.hstack([w, np.zeros((len(w), pad))])
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return j.astype(np.int64), w


@dataclass
class Track:
    """Keyframes for one joint: ``times`` (K), ``rotations`` (K x 4, x/y/z/w), ``translations`` (K x 3)."""

    times: NDArray[np.float64]
    rotations: NDArray[np.float64] | None = None
    translations: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        if self.rotations is not None:
            self.rotations = np.asarray(self.rotations, dtype=np.float64)
        if self.translations is not None:
            self.translations = np.asarray(self.translations, dtype=np.float64)

    def _span(self, t: float) -> tuple[int, int, float]:
        times = self.times
        if t <= times[0] or len(times) == 1:
            return 0, 0, 0.0
        if t >= times[-1]:
            return len(times) - 1, len(times) - 1, 0.0
        i = int(np.searchsorted(times, t, side="right")) - 1
        return i, i + 1, (t - times[i]) / (times[i + 1] - times[i])

    def sample(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        """(rotation quaternion, translation) at time ``t`` (clamped), either None if not animated."""
        i, j, f = self._span(t)
        rotation = slerp(self.rotations[i], self.rotations[j], f) if self.rotations is not None else None
        translation = None
        if self.translations is not None:
            translation = self.translations[i] + (self.translations[j] - self.translations[i]) * f
        return rotation, translation


@dataclass
class Clip:
    """A named animation: joint name -> ``Track``. Joints are found by name under the clip's owner node."""

    name: str
    tracks: dict[str, Track] = field(default_factory=dict)
    loop: bool = True

    @property
    def duration(self) -> float:
        return max((float(tr.times[-1]) for tr in self.tracks.values()), default=0.0)

    def apply(self, owner: SceneNode, t: float) -> None:
        """Pose ``owner``'s joints at time ``t`` (looped clips wrap)."""
        if self.loop and self.duration > 0:
            t = t % self.duration
        by_name = {n.name: n for n in owner.iter_nodes()}
        for joint_name, track in self.tracks.items():
            node = by_name.get(joint_name)
            if node is None:
                raise KeyError(f"Clip '{self.name}': no joint '{joint_name}' under '{owner.name}'")
            rotation, translation = track.sample(t)
            set_local(node, rotation, translation)


def set_local(node: SceneNode, rotation: NDArray[np.float64] | None = None,
              translation: NDArray[np.float64] | None = None) -> None:
    """Set a node's local rotation (quaternion) and/or translation, keeping its scale."""
    m = node.transform.to_matrix()
    s = node.transform.scale
    if rotation is not None:
        m[:3, :3] = matrix_from_quat(rotation) * s[None, :]
    if translation is not None:
        m[:3, 3] = translation
    node.transform = Transform.from_matrix(m)
    node.transform.scale = s.copy()


def pose_clips(root: SceneNode, clip_name: str, t: float) -> int:
    """Pose every clip called ``clip_name`` under ``root`` at time ``t``; returns how many were posed."""
    count = 0
    for owner in root.iter_nodes():
        for clip in owner.clips:
            if clip.name == clip_name:
                clip.apply(owner, t)
                count += 1
    return count
