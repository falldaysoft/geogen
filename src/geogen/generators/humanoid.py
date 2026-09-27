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

import copy
from pathlib import Path

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.skeleton import Pose, Skeleton, load_skeleton
from ..core.skin import Clip, Skin
from .ringloft import Chain, Ring, Span, loft_body

CHAIN_KEYS = {"bones", "end", "mirror", "sides", "spacing", "blend", "front", "caps", "bind", "weight_shift",
              "rings", "cap_rings", "joint_rings", "joint_blend", "hem"}
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
        hem=float(spec.get("hem", 0.006)),
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
    slots, _ = mesh.effective_face_materials()
    region = ((centre[:, 0] > x0) & (centre[:, 0] < x1) & (centre[:, 1] > y0) & (centre[:, 1] < y1)
              & (normal[:, 2] > min_normal) & (centre[:, 2] > 0) & (slots == 0))   # bare skin only
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
    slots, materials = mesh.effective_face_materials()
    out.materials = [*materials, material]
    out.face_materials = np.where(region, len(materials), slots)
    return out


CLOTH_COLORS = {"white": (0.95, 0.95, 0.94), "cream": (0.94, 0.9, 0.8), "black": (0.12, 0.12, 0.13),
                "charcoal": (0.25, 0.26, 0.28), "grey": (0.55, 0.56, 0.58), "navy": (0.16, 0.22, 0.4),
                "denim": (0.3, 0.42, 0.62), "light_denim": (0.56, 0.67, 0.8), "red": (0.72, 0.15, 0.15),
                "burgundy": (0.45, 0.1, 0.16), "coral": (0.92, 0.47, 0.42), "pink": (0.92, 0.64, 0.72),
                "mustard": (0.86, 0.66, 0.22), "olive": (0.42, 0.45, 0.24), "green": (0.22, 0.5, 0.32),
                "teal": (0.12, 0.47, 0.5), "lavender": (0.67, 0.6, 0.82), "brown": (0.42, 0.28, 0.18),
                "tan": (0.73, 0.59, 0.41), "sky": (0.55, 0.72, 0.9)}
GARMENT_KEYS = {"kind", "name", "type", "material", "layer", "thickness", "cover", "chains", "cut", "weights"}


def load_garment(assets_dir: Path, name: str, params: dict) -> dict:
    """A ``kind: garment`` definition (assets/clothing/<name>.yaml) resolved against the body's params."""
    from ..layout.expressions import resolve_value
    from ..layout.yaml_utils import safe_load_path

    path = Path(assets_dir) / "clothing" / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown garment '{name}' (no {path})")
    spec = safe_load_path(path)
    if spec.get("kind") != "garment":
        raise ValueError(f"{path}: expected kind 'garment'")
    unknown = set(spec) - GARMENT_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    return resolve_value(spec, params)


def split_by_slot(mesh: Mesh, slot_colors: list) -> Mesh:
    """Give every material slot its own vertices (colour-coded per slot), then merge slots sharing a
    base material so garments of one fabric draw together. ``slot_colors[i]`` is (rgb, base material)."""
    slots, _ = mesh.effective_face_materials()
    corner_slot = np.repeat(slots, 3)
    key = np.column_stack([mesh.faces.reshape(-1), corner_slot])
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    src = uniq[:, 0]
    out = Mesh(mesh.vertices[src], inverse.reshape(-1, 3),
               normals=mesh.normals[src] if mesh.normals is not None else None,
               uvs=mesh.uvs[src] if mesh.uvs is not None else None)
    if mesh.joints is not None:
        out.joints, out.weights = mesh.joints[src], mesh.weights[src]
    out.colors = np.array([[*slot_colors[k][0], 1.0] for k in uniq[:, 1]])
    bases: list = []
    remap = []
    for _, base in slot_colors:
        index = next((i for i, b in enumerate(bases) if b is base), None)
        if index is None:
            bases.append(base)
            index = len(bases) - 1
        remap.append(index)
    out.material = bases[0]
    if len(bases) > 1:
        out.materials = bases
        out.face_materials = np.asarray(remap)[slots]
    return out


def transfer_weights(shell: Mesh, body: Mesh, neighbours: int = 12, smooth: int = 2) -> None:
    """Give a loose garment the weights of the body under it: each shell vertex blends its nearest
    body vertices (rest pose, inverse distance), then weights are averaged over shell neighbours so
    the garment moves like the body it covers (a skirt hem between the legs follows both)."""
    from scipy.spatial import cKDTree

    from ..core.skin import normalize_weights

    joints = int(max(body.joints.max(), 0)) + 1
    dense_body = np.zeros((len(body.vertices), joints))
    np.put_along_axis(dense_body, body.joints, body.weights, axis=1)
    dist, idx = cKDTree(body.vertices).query(shell.vertices, k=neighbours)
    w = 1.0 / np.maximum(dist, 1e-4) ** 2
    dense = np.einsum("vk,vkj->vj", w, dense_body[idx]) / w.sum(axis=1, keepdims=True)
    # Smooth over the shell's own edges so the garment doesn't crease where its neighbours differ.
    edges = np.vstack([shell.faces[:, [0, 1]], shell.faces[:, [1, 2]], shell.faces[:, [2, 0]]])
    for _ in range(smooth):
        acc = dense.copy()
        count = np.ones(len(dense))
        np.add.at(acc, edges[:, 0], dense[edges[:, 1]])
        np.add.at(count, edges[:, 0], 1)
        dense = acc / count[:, None]
    index = np.tile(np.arange(joints), (len(dense), 1))
    shell.joints, shell.weights = normalize_weights(index, dense)


def build_hair(spec: dict, skeleton: Skeleton, material_loader, crease: float) -> Mesh | None:
    """A hair shell: ring-lofted chains (like the body's) unioned, minus ``cut`` shapes -- boxes
    ([x0, y0, z0, x1, y1, z1]) or ellipsoids ({center, radii}), rest pose -- skinned by the chains' weights."""
    from ..core import csg
    from ..generators.primitives import CubeGenerator

    chains = [parse_chain(name, c, skeleton) for name, c in (spec.get("chains") or {}).items()]
    if not chains:
        return None
    base = material_loader.load(spec.get("material", "hair"))
    materials = None
    if any(c.cap_end == "hem" for c in chains):
        # Hemmed (open) garments: the inner wall is its own slot, a copy named <material>_lining.
        lining = copy.copy(base)
        lining.name = f"{base.name}{LINING_SUFFIX}"
        materials = [base, lining]
    mesh = loft_body(skeleton, chains, crease, materials=materials)
    from ..generators.primitives import EllipsoidGenerator

    cutters = []
    for cut in spec.get("cut") or []:
        if isinstance(cut, dict):            # {center: [x, y, z], radii: [rx, ry, rz]}: rounded hairlines
            radii = np.asarray(cut["radii"], dtype=np.float64)
            shape = EllipsoidGenerator(size_x=2 * radii[0], size_y=2 * radii[1], size_z=2 * radii[2],
                                       segments=28, rings=14).generate()
            shape.vertices = shape.vertices + np.asarray(cut["center"], dtype=np.float64)
            cutters.append(shape)
            continue
        lo, hi = np.asarray(cut[:3], dtype=np.float64), np.asarray(cut[3:], dtype=np.float64)
        cube = CubeGenerator(size_x=hi[0] - lo[0], size_y=hi[1] - lo[1], size_z=hi[2] - lo[2], bevel=0).generate()
        cube.vertices = cube.vertices + (lo + hi) / 2
        cutters.append(cube)
    if cutters:
        from ..core.meshops import collapse_short_edges

        mesh = collapse_short_edges(csg.difference(mesh, *cutters, crease_angle=crease), 1e-4)
    mesh.material = base
    return mesh


LINING_SUFFIX = "_lining"
LINING_SHADE = 0.8            # the inside of a garment reads a little darker


def lining_faces(mesh: Mesh) -> np.ndarray:
    """Faces of a hemmed garment's inner wall (its ``*_lining`` material slot)."""
    if mesh.face_materials is None or not mesh.materials:
        return np.zeros(len(mesh.faces), bool)
    slots = [i for i, m in enumerate(mesh.materials) if m is not None and m.name.endswith(LINING_SUFFIX)]
    return np.isin(mesh.face_materials, slots)




def pose_clips(skeleton: Skeleton, poses: dict) -> list[Clip]:
    """A static clip ``pose_<name>`` per skeletal pose, keying every joint's local rotation, so a
    runtime can switch poses by playing clips."""
    from .clips import tracks_from_poses

    times = np.array([0.0, POSE_CLIP_SECONDS])
    return [Clip(f"pose_{name}", tracks_from_poses(skeleton, [Pose.from_spec(spec, skeleton)] * 2, times),
                 loop=False) for name, spec in poses.items()]


def build_humanoid(spec: dict, name: str, material_loader, assets_dir: Path, params: dict | None = None) -> SceneNode:
    import copy

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
    skin = material_loader.load(spec.get("material", "skin"))
    tone = tuple(skin_tone(spec.get("skin_tone", 0.2)))

    # Outfit: tight garments become spans on the body's chains (one slot each, with its own
    # copy of the garment material so slots stay apart through CSG); loose ones are shells.
    outfit_name = spec.get("outfit", "none")
    outfits = spec.get("outfits") or {"none": []}
    if outfit_name not in outfits:
        raise ValueError(f"'{name}': unknown outfit '{outfit_name}'. Outfits: {sorted(outfits)}")
    by_name = {c.name: c for c in chains}
    slot_colors = [(tone, skin)]
    materials = [skin]
    loose = []
    for item in outfits[outfit_name] or []:
        garment = load_garment(assets_dir, item["garment"], params or {})
        color = _color(item.get("color", "grey"), CLOTH_COLORS, "cloth colour")
        base = material_loader.load(garment.get("material", "cloth"))
        if garment.get("cover"):
            slot = len(materials)
            materials.append(copy.copy(base))
            slot_colors.append((color, base))
            for chain_name, (start, end) in garment["cover"].items():
                if chain_name not in by_name:
                    raise ValueError(f"garment '{item['garment']}': body has no chain '{chain_name}'")
                by_name[chain_name].spans.append(
                    Span((start[0], float(start[1])), (end[0], float(end[1])), float(garment.get("thickness", 0.005)),
                         slot, int(garment.get("layer", 1))))
        if garment.get("chains"):
            loose.append((garment, color, base))

    mesh = loft_body(skeleton, chains, crease, mirrored, materials if len(materials) > 1 else None)
    budget = spec.get("max_triangles")
    if budget and len(mesh.faces) > int(budget):
        from ..core import meshops

        mesh = meshops.decimate(mesh, int(budget) / len(mesh.faces), crease_angle=crease)
    if len(materials) > 1:
        # Map the union's slots (ordered by first appearance) back to ours by material identity.
        slots, result_materials = mesh.effective_face_materials()
        ours = [next(i for i, m in enumerate(materials) if m is r) for r in result_materials]
        mesh.face_materials = np.asarray(ours)[slots]
        mesh.materials = list(materials)
        mesh = split_by_slot(mesh, slot_colors)
    else:
        mesh.material = skin
        mesh.colors = np.tile([*tone, 1.0], (len(mesh.vertices), 1))
    if spec.get("face"):
        face = dict(spec["face"])
        mesh = apply_face(mesh, face["box"], face_material(face, skin), float(face.get("min_normal", 0.25)))

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
    if loose:
        shells = []
        for garment, color, base in loose:
            shell = build_hair({"chains": garment["chains"], "cut": garment.get("cut"),
                                "material": garment.get("material", "cloth")}, skeleton, material_loader, crease)
            if garment.get("weights", "body") == "body":
                transfer_weights(shell, mesh)
            shell.colors = np.tile([*color, 1.0], (len(shell.vertices), 1))
            inside = np.unique(shell.faces[lining_faces(shell)])
            shell.colors[inside, :3] *= LINING_SHADE
            shells.append(shell)
        clothes = root.add_child(SceneNode("clothes", mesh=Mesh.merge(shells)))
        clothes.meta["collider"] = "none"
        clothes.skin = Skin.bind(clothes, [nodes[n] for n in skeleton.names], name=f"{name}_clothes_skin")

    poses = dict(spec.get("poses") or {})
    root.clips.extend(pose_clips(skeleton, poses))
    seats = fit_seats(root, skeleton, poses)
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
                             "pose": pose_name, "hair": hair_style, "outfit": outfit_name,
                             "triangles": int(sum(len(n.mesh.faces) for n in root.iter_nodes() if n.mesh is not None))}
    root.meta["humanoid"]["seats"] = seats
    root.tags = ["character"]
    return root


SEAT_SINK = 0.005        # m the seat flesh/cloth presses into a seat
SEAT_CLEAR = 0.03        # m the calves keep in front of a seat's front edge
KNEE_OVERHANG = 0.3      # share of the thigh, behind the knee joint, that clears the seat front


def seat_contact(root, clip: str) -> tuple[float, SceneNode, np.ndarray, np.ndarray] | None:
    """The underside of what rests on a seat in ``clip`` (body frame y): the buttocks and the
    thighs, or cloth just under them. Also returns the posed actor, body mesh
    and each vertex's dominant bone."""
    from ..core.skin import pose_clips as apply_clip

    actor = root.instance()
    apply_clip(actor, clip, 0.0)
    body = next((n for n in actor.iter_nodes() if n.name == "body" and n.skin is not None), None)
    if body is None:
        return None
    mesh = body.world_mesh()
    names = np.array([j.name for j in body.skin.joints])
    dominant = names[mesh.joints[np.arange(len(mesh.vertices)), np.argmax(mesh.weights, axis=1)]]
    # The knee overhangs the seat's front edge (reach slides the sitter to keep it so).
    hip = actor.find("LeftUpperLeg").world_transform()[:3, 3]
    knee = actor.find("LeftLowerLeg").world_transform()[:3, 3]
    front = knee[2] - KNEE_OVERHANG * np.linalg.norm(knee - hip)
    seat = np.isin(dominant, ["Hips", "LeftUpperLeg", "RightUpperLeg"]) & (mesh.vertices[:, 2] < front)
    contact = float(mesh.vertices[seat][:, 1].min())
    # A skirt under the seat adds its thickness, but not its hanging hem (a seated skirt still
    # drapes below the thighs: geogen-z2b.16.16), so only cloth just under the flesh counts.
    clothes = next((n for n in actor.iter_nodes() if n.name == "clothes" and n.mesh is not None), None)
    if clothes is not None:
        cv = clothes.world_mesh().vertices
        under = cv[(cv[:, 1] < contact) & (cv[:, 1] > contact - 0.015) & (cv[:, 2] < front)]
        if len(under):
            contact = float(under[:, 1].min())
    return contact, actor, mesh, dominant


def fit_seats(root, skeleton: Skeleton, poses: dict) -> dict[str, float]:
    """Seated poses declare ``seat``: the seat surface's height in the body frame they were
    solved for. Measure where this body's seat flesh (and cloth) actually is, move the seat
    there (SEAT_SINK into it) and re-solve the ``floor: true`` IK targets (feet, hands on a
    table) by the same amount, so the feet stay on the floor. Returns {pose: seat height}."""
    seats = {}
    for name, spec in poses.items():
        if spec.get("seat") is None:
            continue
        spec = copy.deepcopy(spec)
        for attempt in range(8):
            clip = pose_clips(skeleton, {name: spec})[0]
            _replace_clip(root, clip)
            found = seat_contact(root, clip.name)
            if found is None:
                break
            delta = found[0] + SEAT_SINK - float(spec["seat"])
            spec["seat"] = float(spec["seat"]) + delta         # the seat as this clip has it
            if abs(delta) < 0.001 or attempt == 7:
                break
            # Lowering the feet swings the thighs down, which moves the contact again: iterate.
            for entry in (spec.get("ik") or {}).values():
                if isinstance(entry, dict) and entry.get("floor"):
                    entry["target"] = [entry["target"][0], float(entry["target"][1]) + delta, entry["target"][2]]
        poses[name] = spec
        seats[name] = round(float(spec["seat"]), 4)
    return seats


def _replace_clip(root, clip: Clip) -> None:
    root.clips[:] = [c for c in root.clips if c.name != clip.name] + [clip]


def fit_seated_pose(root, poses: dict) -> None:
    """Fit the ``sit`` root pose to this body: the seat surface (the anchor's height) goes where
    the body's fitted ``sit`` seat is (``fit_seats``), and ``reach`` records how far in front of
    the anchor the backs of the calves are, below seat level. Runtimes (and
    affordance_qa.pose_actor) slide the sitter forward on deeper seats (the affordance's
    ``depth`` to the front edge) so the shins clear the seat front."""
    sit = poses.get("sit")
    seat = (root.meta.get("humanoid") or {}).get("seats", {}).get("sit")
    if sit is None or seat is None:
        return
    found = seat_contact(root, "pose_sit")
    if found is None:
        return
    contact, _, mesh, dominant = found
    offset = list(sit["offset"])
    offset[1] = round(-seat, 4)
    calves = mesh.vertices[np.isin(dominant, ["LeftLowerLeg", "RightLowerLeg"])]
    below = calves[calves[:, 1] < contact]
    poses["sit"] = {**sit, "offset": offset}
    if len(below):
        poses["sit"]["reach"] = round(float(below[:, 2].min()) + offset[2], 4)
