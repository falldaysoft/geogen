"""Humanoid bodies: a skeleton plus ring-lofted chains, skinned and posed.

An asset with a ``body:`` block (see ``assets/characters/humanoid.yaml``) is
built here instead of from ``parts:``::

    body:
      skeleton: skeletons/humanoid.yaml       # relative to assets/
      skeleton_params: {height: "{height}", ...}
      material: skin
      crease_angle: 75
      max_triangles: 1500                     # decimated (weights kept) when the loft is bigger
      chains:
        arm:
          bones: [LeftShoulder, LeftUpperArm, LeftLowerArm, LeftHand]
          end: {bone: LeftHand, along: 0.1}  # landmark name | bone name | [x, y, z] | {bone|landmark, along, offset}
          mirror: true                        # also build the Right* twin (exact mirror image)
          sides: 8
          spacing: 0.08
          blend: 0.05
          front: [0, 0, 1]
          caps: [round, round]
          bind: {LeftMiddleProximal: [LeftIndexProximal, ...]}
          weight_shift: {Head: -0.06}
          joint_blend: {LeftMiddleProximal: 0.02}  # per-joint blend width (the joint at that bone's head)
          rings:
            - [LeftUpperArm, 0.0, 0.05, 0.05]                     # bone, t, rx, rz
            - [LeftLowerArm, 1.0, 0.02, 0.03, {offset: [0, 0.01], power: 2.5, twist: 0}]
      poses: {stand: {bones: {LeftUpperArm: [0, 0, -72]}}}    # body-frame degrees (core.skeleton.Pose)
      pose: stand                             # the pose the asset is built (and exported) in

The node tree is ``<name>`` > ``Root`` (joint hierarchy, every profile bone)
and ``body`` (the skinned mesh, no collider). ``meta.humanoid`` records the
skeleton, pose and triangle count. Each skeletal pose is also a static clip
``pose_<name>`` on the root (exported as an animation), which the Godot NPC
runtime plays when it changes pose; the asset's top-level ``poses:`` (root
transforms relative to the affordance anchor) place the body.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.node import SceneNode
from ..core.skeleton import Pose, Skeleton, load_skeleton
from ..core.skin import Clip, Skin, Track
from ..core.transform import quat_from_matrix
from .ringloft import Chain, Ring, loft_body

CHAIN_KEYS = {"bones", "end", "mirror", "sides", "spacing", "blend", "front", "caps", "bind", "weight_shift",
              "rings", "cap_rings", "joint_rings", "joint_blend"}
RING_KEYS = {"offset", "power", "twist"}


def _point(skeleton: Skeleton, spec, where: str) -> np.ndarray:
    if isinstance(spec, str):
        if spec in skeleton.landmarks:
            return skeleton.landmarks[spec].copy()
        if spec in skeleton.bones:
            return skeleton[spec].head.copy()
        raise ValueError(f"{where}: '{spec}' is neither a landmark nor a bone")
    if isinstance(spec, dict):
        if "bone" in spec:
            bone = skeleton[spec["bone"]]
            point = bone.head + bone.rotation[:, 1] * float(spec.get("along", 0.0))
        elif "landmark" in spec:
            point = skeleton.landmarks[spec["landmark"]].copy()
        else:
            raise ValueError(f"{where}: a point mapping needs 'bone' or 'landmark'")
        return point + np.asarray(spec.get("offset", [0, 0, 0]), dtype=np.float64)
    return np.asarray(spec, dtype=np.float64)


def _ring(spec, where: str) -> Ring:
    if isinstance(spec, dict):
        spec = [spec["bone"], spec["t"], spec["rx"], spec["rz"],
                {k: v for k, v in spec.items() if k not in ("bone", "t", "rx", "rz")}]
    if len(spec) not in (4, 5):
        raise ValueError(f"{where}: a ring is [bone, t, rx, rz] or [bone, t, rx, rz, {{offset, power, twist}}]")
    extra = spec[4] if len(spec) == 5 else {}
    unknown = set(extra) - RING_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown ring keys {sorted(unknown)}")
    return Ring(str(spec[0]), float(spec[1]), float(spec[2]), float(spec[3]),
                offset=tuple(float(v) for v in extra.get("offset", (0.0, 0.0))),
                power=float(extra.get("power", 2.0)), twist=float(extra.get("twist", 0.0)))


def parse_chain(name: str, spec: dict, skeleton: Skeleton) -> Chain:
    unknown = set(spec) - CHAIN_KEYS
    if unknown:
        raise ValueError(f"chain '{name}': unknown keys {sorted(unknown)}")
    bones = [str(b) for b in spec["bones"]]
    missing = [b for b in bones if b not in skeleton.bones]
    if missing:
        raise ValueError(f"chain '{name}': unknown bones {missing}")
    caps = spec.get("caps", ["round", "round"])
    chain = Chain(
        name, bones, [_ring(r, f"chain '{name}'") for r in spec["rings"]],
        end=_point(skeleton, spec["end"], f"chain '{name}' end") if "end" in spec else None,
        front=tuple(float(v) for v in spec.get("front", (0, 0, 1))),
        sides=int(spec.get("sides", 10)), spacing=float(spec.get("spacing", 0.08)),
        cap_start=caps[0], cap_end=caps[1], cap_rings=int(spec.get("cap_rings", 3)),
        blend=float(spec.get("blend", 0.05)),
        bind={k: list(v) for k, v in (spec.get("bind") or {}).items()},
        weight_shift={k: float(v) for k, v in (spec.get("weight_shift") or {}).items()},
        joint_blend={k: float(v) for k, v in (spec.get("joint_blend") or {}).items()},
        joint_rings=(list(spec["joint_rings"]) if isinstance(spec.get("joint_rings"), list)
                     else bool(spec.get("joint_rings", True))),
    )
    for ring in chain.rings:
        if ring.bone not in bones:
            raise ValueError(f"chain '{name}': ring bone '{ring.bone}' is not in the chain")
    return chain


POSE_CLIP_SECONDS = 0.25


def pose_clips(skeleton: Skeleton, poses: dict) -> list[Clip]:
    """A static clip ``pose_<name>`` per skeletal pose, keying every joint's local rotation, so a
    runtime can switch poses by playing clips (until real transition clips exist)."""
    clips = []
    for pose_name, pose_spec in poses.items():
        pose = Pose.from_spec(pose_spec, skeleton)
        tracks = {}
        for bone in skeleton.names:
            local = skeleton.rest_local(bone) @ pose.local(bone)
            q = quat_from_matrix(local[:3, :3])
            tracks[bone] = Track([0.0, POSE_CLIP_SECONDS], rotations=[q, q])
        clips.append(Clip(f"pose_{pose_name}", tracks, loop=False))
    return clips


def build_humanoid(spec: dict, name: str, material_loader, assets_dir: Path) -> SceneNode:
    skeleton = load_skeleton(Path(assets_dir) / spec.get("skeleton", "skeletons/humanoid.yaml"),
                             spec.get("skeleton_params"))
    chains, mirrored = [], set()
    for chain_name, chain_spec in (spec.get("chains") or {}).items():
        chains.append(parse_chain(chain_name, chain_spec, skeleton))
        if chain_spec.get("mirror"):
            mirrored.add(chain_name)
    if not chains:
        raise ValueError(f"'{name}': body needs at least one chain")
    crease = float(spec.get("crease_angle", 75.0))
    mesh = loft_body(skeleton, chains, crease, mirrored)
    budget = spec.get("max_triangles")
    if budget and len(mesh.faces) > int(budget):
        from ..core import meshops

        mesh = meshops.decimate(mesh, int(budget) / len(mesh.faces), crease_angle=crease)
    if spec.get("material"):
        mesh.material = material_loader.load(spec["material"])

    root = SceneNode(name)
    joints, nodes = skeleton.build()
    root.add_child(joints)
    body = root.add_child(SceneNode("body", mesh=mesh))
    body.meta["collider"] = "none"
    body.skin = Skin.bind(body, [nodes[n] for n in skeleton.names], name=f"{name}_skin")

    poses = spec.get("poses") or {}
    root.clips.extend(pose_clips(skeleton, poses))
    pose_name = spec.get("pose")
    if pose_name is not None:
        if pose_name not in poses:
            raise ValueError(f"'{name}': unknown pose '{pose_name}'. Poses: {sorted(poses)}")
        Pose.from_spec(poses[pose_name], skeleton).apply(skeleton, nodes)
    posed = body.world_mesh()
    lo, hi = posed.vertices.min(axis=0), posed.vertices.max(axis=0)
    root.size = np.array([2 * max(abs(lo[0]), abs(hi[0])), hi[1], 2 * max(abs(lo[2]), abs(hi[2]))])
    root.meta["humanoid"] = {"skeleton": skeleton.name, "height": round(float(skeleton.landmarks["head_top"][1]), 4)
                             if "head_top" in skeleton.landmarks else round(float(hi[1]), 4),
                             "pose": pose_name, "triangles": int(len(mesh.faces))}
    root.tags = ["character"]
    return root
