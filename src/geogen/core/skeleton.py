"""Humanoid skeletons: named bones placed by body proportions, poses and FK.

Bone names, hierarchy and rest orientations come from a skeleton *profile*
(``assets/skeletons/humanoid_profile.json``, Godot's SkeletonProfileHumanoid:
the same names as Unity Humanoid / VRM), so clips from animation libraries
bind by name and need no rest fixing. A skeleton definition
(``assets/skeletons/humanoid.yaml``, ``kind: skeleton``) only moves bone
heads: ``params`` + ``derived`` expressions give each Body/Face bone's world
rest head; Left* bones mirror to Right*; unlisted bones (fingers) sit at the
profile's offset in their parent's frame, scaled per profile group.

Rest is a T-pose facing +Z with the character's left at +X and ``Root`` on
the floor. ``Skeleton.build()`` makes the joint ``SceneNode`` hierarchy (the
input to ``core.skin``); a ``Pose`` holds per-bone local rotations relative
to rest (plus a root offset) and ``Skeleton.fk`` / ``Pose.apply`` pose them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .node import SceneNode
from .transform import Transform, matrix_from_quat, quat_from_matrix

SKELETONS_DIR = Path(__file__).resolve().parents[3] / "assets" / "skeletons"
DEFAULT_SKELETON = SKELETONS_DIR / "humanoid.yaml"
SKELETON_KIND = "skeleton"


@dataclass(frozen=True)
class ProfileBone:
    name: str
    parent: str | None
    group: str
    tail: str | None
    translation: tuple[float, float, float]       # reference rest, parent frame
    rotation: tuple[float, float, float, float]   # reference rest, parent frame (x, y, z, w)


@lru_cache(maxsize=None)
def load_profile(path: str | Path = SKELETONS_DIR / "humanoid_profile.json") -> tuple[ProfileBone, ...]:
    """Profile bones in hierarchy order (parents first)."""
    data = json.loads(Path(path).read_text())
    return tuple(ProfileBone(b["name"], b["parent"] or None, b["group"], b["tail"] or None,
                             tuple(b["reference_pose"]["translation"]), tuple(b["reference_pose"]["rotation"]))
                 for b in data["bones"])


def _matrix(rotation: NDArray, translation: NDArray) -> NDArray[np.float64]:
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = translation
    return m


def profile_rest_world(profile: tuple[ProfileBone, ...]) -> dict[str, NDArray[np.float64]]:
    """World rest matrices of the profile's reference pose."""
    world: dict[str, NDArray[np.float64]] = {}
    for bone in profile:
        local = _matrix(matrix_from_quat(bone.rotation), bone.translation)
        world[bone.name] = world[bone.parent] @ local if bone.parent else local
    return world


def mirror_name(name: str) -> str | None:
    """LeftHand -> RightHand (and back); None for centre bones."""
    for a, b in (("Left", "Right"), ("Right", "Left")):
        if name.startswith(a):
            return b + name[len(a):]
    return None


@dataclass
class Bone:
    name: str
    parent: str | None
    head: NDArray[np.float64]        # world rest position
    rotation: NDArray[np.float64]    # world rest orientation (3x3), the profile's
    group: str = "Body"

    @property
    def rest_world(self) -> NDArray[np.float64]:
        return _matrix(self.rotation, self.head)


@dataclass
class Skeleton:
    """Bones in hierarchy order, plus named rest ``landmarks`` and the params that placed them."""

    name: str
    bones: dict[str, Bone]
    landmarks: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Bone:
        return self.bones[name]

    @property
    def names(self) -> list[str]:
        return list(self.bones)

    def rest_local(self, name: str) -> NDArray[np.float64]:
        """A bone's rest transform in its parent's frame."""
        bone = self.bones[name]
        if bone.parent is None:
            return bone.rest_world
        return np.linalg.inv(self.bones[bone.parent].rest_world) @ bone.rest_world

    def children(self, name: str) -> list[str]:
        return [b.name for b in self.bones.values() if b.parent == name]

    def length(self, name: str) -> float:
        """Distance from a bone's head to its (first) child's head; 0 for end bones."""
        kids = self.children(name)
        return float(np.linalg.norm(self.bones[kids[0]].head - self.bones[name].head)) if kids else 0.0

    def posed_local(self, name: str, pose: Pose | None) -> NDArray[np.float64]:
        """A bone's local transform in ``pose``: rest, then the pose's rotation and offset."""
        local = self.rest_local(name)
        if pose is None:
            return local
        local = local @ pose.local(name)
        offset = pose.offsets.get(name)
        if offset is not None:
            local[:3, 3] += offset
        if self.bones[name].parent is None:
            local = _matrix(np.eye(3), pose.root) @ local
        return local

    def fk(self, pose: Pose | None = None) -> dict[str, NDArray[np.float64]]:
        """World matrices of every bone in ``pose`` (rest when None)."""
        world: dict[str, NDArray[np.float64]] = {}
        for name, bone in self.bones.items():
            local = self.posed_local(name, pose)
            world[name] = world[bone.parent] @ local if bone.parent else local
        return world

    def pose_quat(self, name: str, parent_world: NDArray[np.float64], world_rotation: NDArray[np.float64]):
        """The pose rotation (relative to rest) that gives bone ``name`` ``world_rotation`` under a
        parent at ``parent_world``."""
        local = parent_world[:3, :3].T @ world_rotation
        return quat_from_matrix(self.rest_local(name)[:3, :3].T @ local)

    def body_quat(self, name: str, rotation: NDArray[np.float64]):
        """A body-frame rotation (3x3, about the character's own axes) as a pose rotation for ``name``."""
        rest = self.bones[name].rotation
        return quat_from_matrix(rest.T @ rotation @ rest)

    def build(self) -> tuple[SceneNode, dict[str, SceneNode]]:
        """Joint node hierarchy at rest: (root joint node, name -> node)."""
        nodes: dict[str, SceneNode] = {}
        root = None
        for name, bone in self.bones.items():
            node = SceneNode(name, transform=Transform.from_matrix(self.rest_local(name)))
            node.transform.scale = np.ones(3)
            nodes[name] = node
            if bone.parent is None:
                root = node
            else:
                nodes[bone.parent].add_child(node)
        return root, nodes


@dataclass
class Pose:
    """Per-bone local rotations relative to rest (quaternions x, y, z, w) and a root offset (m)."""

    rotations: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    root: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    # Per-bone translation added to the rest position, in the parent's frame (e.g. Hips bob).
    offsets: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def copy(self) -> Pose:
        return Pose(dict(self.rotations), self.root.copy(), {k: v.copy() for k, v in self.offsets.items()})

    def local(self, name: str) -> NDArray[np.float64]:
        q = self.rotations.get(name)
        return np.eye(4) if q is None else _matrix(matrix_from_quat(q), np.zeros(3))

    @classmethod
    def from_spec(cls, spec: dict | None, skeleton: Skeleton | None = None) -> Pose:
        """``{bones: {Name: [x, y, z] degrees}, root: [x, y, z], frame: body | bone}``.

        Angles are Euler XYZ (like ``Transform``). In the default ``body`` frame they
        rotate about the character's own axes (X = its left, Y = up, Z = forward),
        the same for every bone, so ``LeftUpperLeg: [-60, 0, 0]`` lifts the thigh
        forward and ``LeftUpperArm: [0, 0, -70]`` lowers the arm. ``bone`` uses each
        bone's local axes (profile roll: differs per bone, as library clips see it).
        """
        spec = spec or {}
        frame = spec.get("frame", "body")
        if frame not in ("body", "bone"):
            raise ValueError(f"pose frame must be 'body' or 'bone', got {frame!r}")
        if frame == "body" and spec.get("bones") and skeleton is None:
            raise ValueError("a body-frame pose needs the skeleton")
        rotations = {}
        for name, degrees in (spec.get("bones") or {}).items():
            r = Transform(rotation=np.radians(np.asarray(degrees, dtype=np.float64))).to_matrix()[:3, :3]
            if frame == "body":
                if name not in skeleton.bones:
                    raise KeyError(f"pose: unknown bone '{name}'")
                rest = skeleton[name].rotation
                r = rest.T @ r @ rest
            rotations[name] = quat_from_matrix(r)
        return cls(rotations, np.asarray(spec.get("root", [0, 0, 0]), dtype=np.float64))

    def apply(self, skeleton: Skeleton, nodes: dict[str, SceneNode]) -> None:
        """Set the joint nodes' transforms (from ``Skeleton.build``) to this pose."""
        for name in skeleton.bones:
            nodes[name].transform = Transform.from_matrix(skeleton.posed_local(name, self))
            nodes[name].transform.scale = np.ones(3)


def _point(value, params: dict[str, float]) -> NDArray[np.float64]:
    from ..layout.expressions import resolve_value

    point = resolve_value(value, params)
    if not isinstance(point, list) or len(point) != 3:
        raise ValueError(f"expected an [x, y, z] point, got {value!r}")
    return np.array([float(v) for v in point])


def load_skeleton(path: str | Path = DEFAULT_SKELETON, params: dict[str, float] | None = None) -> Skeleton:
    """Build a skeleton from a ``kind: skeleton`` definition with param overrides."""
    from ..layout.expressions import resolve_params, resolve_value
    from ..layout.yaml_utils import safe_load_path

    path = Path(path)
    spec = safe_load_path(path)
    if spec.get("kind") != SKELETON_KIND:
        raise ValueError(f"{path}: expected kind '{SKELETON_KIND}', got {spec.get('kind')!r}")
    values = resolve_params(spec.get("params"), params)
    declared = dict(values)
    for name, expr in (spec.get("derived") or {}).items():
        values[name] = float(resolve_value(expr, values))

    profile = load_profile(path.parent / spec.get("profile", "humanoid_profile.json"))
    reference = profile_rest_world(profile)
    heads: dict[str, NDArray[np.float64]] = {}
    for name, value in (spec.get("bones") or {}).items():
        heads[name] = _point(value, values)
        mirror = mirror_name(name)
        if mirror is not None and mirror not in (spec.get("bones") or {}):
            heads[mirror] = heads[name] * [-1, 1, 1]
    known = {b.name for b in profile}
    unknown = set(heads) - known
    if unknown:
        raise ValueError(f"{path}: unknown bones {sorted(unknown)}")
    group_scale = {g: float(resolve_value(v, values)) for g, v in (spec.get("group_scale") or {}).items()}

    bones: dict[str, Bone] = {}
    for pb in profile:
        rotation = reference[pb.name][:3, :3]
        if pb.name in heads:
            head = heads[pb.name]
        elif pb.parent is None:
            raise ValueError(f"{path}: root bone '{pb.name}' needs a position")
        elif pb.group in group_scale:
            parent = bones[pb.parent]
            head = parent.head + parent.rotation @ (np.asarray(pb.translation) * group_scale[pb.group])
        else:
            raise ValueError(f"{path}: bone '{pb.name}' ({pb.group}) needs a position or a group_scale")
        bones[pb.name] = Bone(pb.name, pb.parent, head, rotation, pb.group)

    landmarks = {name: _point(v, values) for name, v in (spec.get("landmarks") or {}).items()}
    return Skeleton(spec.get("name", path.stem), bones, landmarks, declared)
