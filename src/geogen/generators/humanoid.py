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
      clips: {walk: {generator: walk, speed: 1.2}, idle: {}}   # generators/clips.py, over `pose`

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

from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.skeleton import Pose, Skeleton, load_skeleton
from ..core.skin import Clip, Skin
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

# Vertex-colour multipliers over the light neutral skin / hair materials.
SKIN_TONES = [(1.0, 0.97, 0.95), (0.98, 0.9, 0.84), (0.9, 0.76, 0.66), (0.74, 0.56, 0.44),
              (0.55, 0.39, 0.3), (0.38, 0.26, 0.2)]          # skin_tone 0 (light) .. 1 (dark)
HAIR_COLORS = {"black": (0.13, 0.11, 0.1), "dark_brown": (0.28, 0.19, 0.13), "brown": (0.45, 0.3, 0.19),
               "auburn": (0.55, 0.26, 0.14), "blonde": (0.92, 0.78, 0.5), "light_blonde": (1.0, 0.92, 0.72),
               "red": (0.72, 0.3, 0.14), "grey": (0.62, 0.61, 0.6), "white": (0.92, 0.91, 0.9)}
EYE_COLORS = {"brown": (84, 56, 36), "dark_brown": (52, 34, 24), "hazel": (120, 96, 50), "green": (74, 110, 70),
              "blue": (64, 104, 150), "grey": (110, 120, 128)}
FACE_KEYS = {"box", "min_normal", "eye_color", "eye_v", "eye_spacing", "eye_size", "brow_color", "brow_thickness",
             "brow_arch", "lashes", "lip_color", "lip_strength", "mouth_v", "mouth_width", "blush", "freckles"}
_face_materials: dict = {}


def skin_tone(t: float) -> np.ndarray:
    """Interpolate the SKIN_TONES ramp at ``t`` in [0, 1]."""
    x = np.clip(float(t), 0.0, 1.0) * (len(SKIN_TONES) - 1)
    i = min(int(x), len(SKIN_TONES) - 2)
    return np.array(SKIN_TONES[i]) * (1 - (x - i)) + np.array(SKIN_TONES[i + 1]) * (x - i)


def _color(value, table: dict, what: str):
    if isinstance(value, str):
        if value not in table:
            raise ValueError(f"unknown {what} '{value}'. Known: {sorted(table)}")
        return table[value]
    return tuple(value)


def face_material(spec: dict, skin):
    """A face decal material (one per distinct face spec) over the skin material's base."""
    from ..materials.material import Material
    from ..textures.character import FaceTextureGenerator

    params = {k: v for k, v in spec.items() if k not in ("box", "min_normal")}
    unknown = set(spec) - FACE_KEYS
    if unknown:
        raise ValueError(f"face: unknown keys {sorted(unknown)}")
    if "eye_color" in params:
        params["eye_color"] = _color(params["eye_color"], EYE_COLORS, "eye colour")
    for key in ("brow_color", "lip_color"):
        if key in params:
            params[key] = tuple(params[key]) if not isinstance(params[key], str) else \
                tuple(int(c * 255) for c in _color(params[key], HAIR_COLORS, "colour"))
    key = (skin.name, tuple(sorted((k, str(v)) for k, v in params.items())))
    if key not in _face_materials:
        gen = FaceTextureGenerator(width=256, height=256, seed=71,
                                   base_color=getattr(skin.texture_generator, "base_color", (236, 204, 184)),
                                   **params)
        _face_materials[key] = Material(name="face", texture_generator=gen, texture_size=(256, 256),
                                        roughness=skin.roughness, metallic=0.0, normal_strength=0.0,
                                        tile_size=(1.0, 1.0))
    return _face_materials[key]


def apply_face(mesh: Mesh, box, material, min_normal: float = 0.25) -> Mesh:
    """Give the front of the head its own material slot with planar 0-1 UVs over ``box``
    ([x0, y0, x1, y1], rest pose). The region's vertices are duplicated so the rest of the
    body keeps its metric UVs; the copies keep positions, normals and weights."""
    x0, y0, x1, y1 = (float(v) for v in box)
    tri = mesh.vertices[mesh.faces]
    centre = tri.mean(axis=1)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    region = ((centre[:, 0] > x0) & (centre[:, 0] < x1) & (centre[:, 1] > y0) & (centre[:, 1] < y1)
              & (normal[:, 2] > min_normal) & (centre[:, 2] > 0))
    if not region.any():
        return mesh
    used = np.unique(mesh.faces[region])
    remap = np.full(len(mesh.vertices), -1)
    remap[used] = len(mesh.vertices) + np.arange(len(used))
    out = mesh.copy()
    out.vertices = np.vstack([mesh.vertices, mesh.vertices[used]])
    if mesh.normals is not None:
        out.normals = np.vstack([mesh.normals, mesh.normals[used]])
    planar = np.column_stack([(mesh.vertices[used, 0] - x0) / (x1 - x0), (mesh.vertices[used, 1] - y0) / (y1 - y0)])
    out.uvs = np.vstack([mesh.uvs, planar])
    for attr in ("joints", "weights", "colors"):
        values = getattr(mesh, attr)
        if values is not None:
            setattr(out, attr, np.vstack([values, values[used]]))
    out.faces = mesh.faces.copy()
    out.faces[region] = remap[mesh.faces[region]]
    out.materials = [mesh.material, material]
    out.face_materials = region.astype(np.int64)
    return out


def build_hair(spec: dict, skeleton: Skeleton, material_loader, crease: float) -> Mesh | None:
    """A hair shell: ring-lofted chains (like the body's) unioned, minus ``cut`` boxes
    ([x0, y0, z0, x1, y1, z1], rest pose), skinned by the chains' weights."""
    from ..core import csg
    from ..generators.primitives import CubeGenerator

    chains = [parse_chain(name, c, skeleton) for name, c in (spec.get("chains") or {}).items()]
    if not chains:
        return None
    mesh = loft_body(skeleton, chains, crease)
    cutters = []
    for box in spec.get("cut") or []:
        lo, hi = np.asarray(box[:3], dtype=np.float64), np.asarray(box[3:], dtype=np.float64)
        cube = CubeGenerator(size_x=hi[0] - lo[0], size_y=hi[1] - lo[1], size_z=hi[2] - lo[2], bevel=0).generate()
        cube.vertices = cube.vertices + (lo + hi) / 2
        cutters.append(cube)
    if cutters:
        mesh = csg.difference(mesh, *cutters, crease_angle=crease)
    mesh.material = material_loader.load(spec.get("material", "hair"))
    return mesh




def pose_clips(skeleton: Skeleton, poses: dict) -> list[Clip]:
    """A static clip ``pose_<name>`` per skeletal pose, keying every joint's local rotation, so a
    runtime can switch poses by playing clips."""
    from .clips import tracks_from_poses

    times = np.array([0.0, POSE_CLIP_SECONDS])
    return [Clip(f"pose_{name}", tracks_from_poses(skeleton, [Pose.from_spec(spec, skeleton)] * 2, times),
                 loop=False) for name, spec in poses.items()]


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
    if spec.get("face"):
        face = dict(spec["face"])
        mesh = apply_face(mesh, face["box"], face_material(face, mesh.material), float(face.get("min_normal", 0.25)))
    tone = skin_tone(spec.get("skin_tone", 0.2))
    mesh.colors = np.tile([*tone, 1.0], (len(mesh.vertices), 1))

    root = SceneNode(name)
    joints, nodes = skeleton.build()
    root.add_child(joints)
    body = root.add_child(SceneNode("body", mesh=mesh))
    body.meta["collider"] = "none"
    body.skin = Skin.bind(body, [nodes[n] for n in skeleton.names], name=f"{name}_skin")
    hair_style = spec.get("hair", "none")
    if hair_style not in (None, "none"):
        styles = spec.get("hair_styles") or {}
        if hair_style not in styles:
            raise ValueError(f"'{name}': unknown hair style '{hair_style}'. Styles: {sorted(styles)}")
        hair_mesh = build_hair(styles[hair_style], skeleton, material_loader, crease)
        if hair_mesh is not None:
            color = _color(spec.get("hair_color", "brown"), HAIR_COLORS, "hair colour")
            hair_mesh.colors = np.tile([*color, 1.0], (len(hair_mesh.vertices), 1))
            hair = root.add_child(SceneNode("hair", mesh=hair_mesh))
            hair.meta["collider"] = "none"
            hair.skin = Skin.bind(hair, [nodes[n] for n in skeleton.names], name=f"{name}_hair_skin")

    poses = spec.get("poses") or {}
    root.clips.extend(pose_clips(skeleton, poses))
    pose_name = spec.get("pose")
    if spec.get("clips"):
        from .clips import generate_clip

        base = Pose.from_spec(poses.get(pose_name) or {}, skeleton)
        root.clips.extend(generate_clip(skeleton, base, clip_name, clip_spec or {})
                          for clip_name, clip_spec in spec["clips"].items())
    if pose_name is not None:
        if pose_name not in poses:
            raise ValueError(f"'{name}': unknown pose '{pose_name}'. Poses: {sorted(poses)}")
        Pose.from_spec(poses[pose_name], skeleton).apply(skeleton, nodes)
    posed = Mesh.merge([n.world_mesh() for n in root.iter_nodes() if n.mesh is not None])
    lo, hi = posed.vertices.min(axis=0), posed.vertices.max(axis=0)
    root.size = np.array([2 * max(abs(lo[0]), abs(hi[0])), hi[1], 2 * max(abs(lo[2]), abs(hi[2]))])
    root.meta["humanoid"] = {"skeleton": skeleton.name, "height": round(float(skeleton.landmarks["head_top"][1]), 4)
                             if "head_top" in skeleton.landmarks else round(float(hi[1]), 4),
                             "pose": pose_name, "hair": hair_style,
                             "triangles": int(sum(len(n.mesh.faces) for n in root.iter_nodes() if n.mesh is not None))}
    root.tags = ["character"]
    return root
