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
          root: Hips                          # legs: the tube above its first joint follows this bone
          samples: [[Hips, 0.2]]              # extra resampled rings (where a skirt bends over the seat)
          rings:
            - [LeftUpperArm, 0.0, 0.05, 0.05]                     # bone, t, rx, rz
            - [LeftLowerArm, 1.0, 0.02, 0.03, {offset: [0, 0.01], power: 2.5, twist: 0}]
      poses: {stand: {bones: {LeftUpperArm: [0, 0, -72]}}}    # body-frame degrees (core.skeleton.Pose)
      pose: stand                             # the pose the asset is built (and exported) in
      clips: {walk: {generator: walk, speed: 1.2}, idle: {}}   # generators/clips.py, over `pose`
      clear_arms: true                        # swing the arms clear of this body in every pose and clip (arms.py)

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

from .. import roots as content_roots
from ..core import csg, meshops, uvmap
from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.skeleton import Pose, Skeleton, load_skeleton
from ..core.skin import Clip, Skin, normalize_weights
from .hair import build_sculpted_hair
from .hand import build_hand
from .head import SHAPE_DEFAULTS, HeadShape, build_head, sdf_normals
from .ringloft import Chain, Ring, Span, _signed_volume, loft_body, mirror_mesh

CHAIN_KEYS = {"bones", "end", "mirror", "sides", "spacing", "blend", "front", "caps", "bind", "weight_shift",
              "rings", "cap_rings", "joint_rings", "joint_blend", "hem", "root", "samples"}
RING_KEYS = {"offset", "power", "twist"}
# body.head (generators/head.py): size, the eye line (when there's no painted face to place it),
# neck radius, the Head->Neck weight blend and where it starts; plus the shape params.
HEAD_KEYS = {"size", "eyes", "neck_radius", "blend", "weight_shift"}
HAND_KEYS = {"size", "wrist"}


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
        root=str(spec["root"]) if spec.get("root") else None,
        samples=[(str(b), float(t)) for b, t in spec.get("samples") or []],
    )
    if chain.root is not None and chain.root not in skeleton.bones:
        raise ValueError(f"chain '{name}': unknown root bone '{chain.root}'")
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
             "brow_arch", "lashes", "lip_color", "lip_strength", "mouth_v", "mouth_width", "blush", "freckles", "smile",
             "age", "stubble"}
_face_materials: dict = {}
BROW_LUMINANCE = 0.35            # brows are painted no lighter than this
HEAD_AO = 0.55                   # how much of the head sculpt's ambient occlusion darkens its skin
FACE_TEXTURE = 512                # texels across the face decal (sharp eyes at a conversation's distance)


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
    x0, y0, x1, y1 = (float(v) for v in spec["box"])
    # Paint in true proportions; to 0.05, so faces still share a texture across heights.
    params["aspect"] = round((y1 - y0) / (x1 - x0) * 20) / 20
    unknown = set(spec) - FACE_KEYS
    if unknown:
        raise ValueError(f"face: unknown keys {sorted(unknown)}")
    if "eye_color" in params:
        params["eye_color"] = _color(params["eye_color"], EYE_COLORS, "eye colour")
    for key in ("brow_color", "lip_color"):
        if key in params:
            params[key] = tuple(params[key]) if not isinstance(params[key], str) else \
                tuple(int(c * 255) for c in _color(params[key], HAIR_COLORS, "colour"))
    if "brow_color" in params:
        # Brows are darker than light hair (blonde brows in the hair's colour vanish into the skin).
        rgb = np.asarray(params["brow_color"], dtype=np.float64)
        lum = float(rgb @ [0.2126, 0.7152, 0.0722]) / 255
        params["brow_color"] = tuple(int(c) for c in rgb * min(1.0, BROW_LUMINANCE / max(lum, 1e-6)))
    key = (skin.name, tuple(sorted((k, str(v)) for k, v in params.items())))
    if key not in _face_materials:
        gen = FaceTextureGenerator(width=FACE_TEXTURE, height=FACE_TEXTURE, seed=71,
                                   base_color=getattr(skin.texture_generator, "base_color", (236, 204, 184)),
                                   **params)
        _face_materials[key] = Material(name="face", texture_generator=gen, texture_size=(FACE_TEXTURE, FACE_TEXTURE),
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
GARMENT_KEYS = {"kind", "name", "type", "part", "material", "layer", "thickness", "cover", "chains", "cut", "weights",
                "ease", "sole"}
SOLE_KEYS = {"height", "color", "material"}
GARMENT_PARTS = ("top", "bottom", "dress", "shoes")      # what a garment is worn as (clipping QA's waist check)


def load_garment(assets_dir: Path, name: str, params: dict) -> dict:
    """A ``kind: garment`` definition (assets/clothing/<name>.yaml) resolved against the body's params."""
    from ..layout.expressions import resolve_value
    from ..layout.yaml_utils import safe_load_path

    path = content_roots.find(f"clothing/{name}.yaml", assets_dir)
    if not path.exists():
        raise ValueError(f"unknown garment '{name}' (no {path})")
    spec = safe_load_path(path)
    if spec.get("kind") != "garment":
        raise ValueError(f"{path}: expected kind 'garment'")
    unknown = set(spec) - GARMENT_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    if spec.get("part") not in (None, *GARMENT_PARTS):
        raise ValueError(f"{path}: part '{spec['part']}' is not one of {list(GARMENT_PARTS)}")
    return resolve_value(spec, params)


def build_sole(mesh: Mesh, skeleton: Skeleton, height: float) -> Mesh | None:
    """A shoe's sole under the left foot: the outline of the shoe's bottom (``mesh`` is the lofted
    body, rest pose, feet flat) a little proud of it, from just under the shoe up to ``height``
    above the floor, with a chamfered tread edge. Skinned like the shoe it wraps."""
    from scipy.spatial import cKDTree
    from shapely.geometry import MultiPoint

    names = skeleton.names
    foot = [names.index("LeftFoot"), names.index("LeftToes")]
    dominant = mesh.joints[np.arange(len(mesh.vertices)), np.argmax(mesh.weights, axis=1)]
    v = mesh.vertices
    pick = np.isin(dominant, foot) & (v[:, 0] > 0) & (v[:, 1] < height + 0.01)
    if pick.sum() < 8:
        return None
    y0 = float(v[pick, 1].min()) + 0.004               # the shoe's rounded bottom (under the floor) stays the lowest point
    outline = MultiPoint(v[pick][:, [0, 2]]).convex_hull.buffer(0.0025, quad_segs=3).simplify(0.0015)
    ring = np.asarray(outline.exterior.coords)[:-1]
    if outline.exterior.is_ccw:                       # looking down +y onto (x, z): clockwise is ccw seen from above
        ring = ring[::-1]
    centre = ring.mean(axis=0)
    inward = centre - ring
    inward /= np.linalg.norm(inward, axis=1, keepdims=True)
    levels = [(y0, 0.007), (y0 + 0.006, 0.0), (height, 0.0), (height, 0.004)]     # the tread inside the shoe's outline (a heel strike rolls onto its edge)
    n = len(ring)
    verts = [np.column_stack([xz[:, 0], np.full(n, y), xz[:, 1]])
             for y, inset in levels for xz in [ring + inward * inset]]
    verts = np.vstack(verts + [[centre[0], y0, centre[1]], [centre[0], height, centre[1]]])
    faces = []
    for k in range(len(levels) - 1):
        a, b = k * n, (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces += [[a + i, a + j, b + j], [a + i, b + j, b + i]]
    bottom, top = len(verts) - 2, len(verts) - 1
    last = (len(levels) - 1) * n
    for i in range(n):
        j = (i + 1) % n
        faces += [[bottom, j, i], [top, last + i, last + j]]
    sole = Mesh(verts, np.asarray(faces, dtype=np.int64))
    if _signed_volume(sole) < 0:
        sole.faces = sole.faces[:, ::-1].copy()
    near = cKDTree(v[pick]).query(verts)[1]
    sole.joints, sole.weights = mesh.joints[pick][near], mesh.weights[pick][near]
    return uvmap.box_project(sole)


def split_by_slot(mesh: Mesh, slot_colors: list, slots_out: list | None = None) -> Mesh:
    """Give every material slot its own vertices (colour-coded per slot), then merge slots sharing a
    base material so garments of one fabric draw together. ``slot_colors[i]`` is (rgb, base material).
    ``slots_out`` (a list) receives the new vertices' slots."""
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
    if slots_out is not None:
        slots_out.append(uniq[:, 1].copy())
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


WAIST_BONES = ("Hips", "Spine", "LeftUpperLeg", "RightUpperLeg")
HEM_SMOOTH = 0.003            # m a waist hem vertex may stand above or below the line between its neighbours


def dominant_bones(mesh: Mesh, names) -> np.ndarray:
    """Each vertex's heaviest bone, by name (``names``: the skin's joint names)."""
    return np.asarray(names)[mesh.joints[np.arange(len(mesh.vertices)), np.argmax(mesh.weights, axis=1)]]


def _welded(mesh: Mesh) -> np.ndarray:
    """An index per distinct vertex position (slots and creases have their own vertices)."""
    return np.unique(np.round(mesh.vertices / 1e-5).astype(np.int64), axis=0, return_inverse=True)[1].ravel()


def waist_hem(mesh: Mesh, mine: np.ndarray, dominant: np.ndarray) -> np.ndarray:
    """A top's hem round the hips: the boundary edges of its faces (the mask ``mine`` over a
    rest-pose ``mesh``'s faces) between vertices on the pelvis, spine or thighs (not its neck or
    cuffs), as (n, 2) vertex indices. Positions are welded first, so creases and other slots'
    copies of a vertex don't count as edges."""
    faces = mesh.faces[mine]
    if not len(faces):
        return np.empty((0, 2), dtype=np.int64)
    canon = _welded(mesh)
    rep = np.full(int(canon.max()) + 1, -1)
    rep[canon[faces.ravel()]] = faces.ravel()          # one of the top's vertices stands for each position
    f = canon[faces]
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(edges, axis=0, return_counts=True)
    edges = rep[uniq[count == 1]]
    waist = np.isin(dominant, WAIST_BONES)
    return edges[waist[edges].all(axis=1)]


def hem_shape(vertices: np.ndarray, edges: np.ndarray) -> tuple[dict[int, tuple[float, np.ndarray, int, int]], int]:
    """How smooth a hem loop is: {vertex: (how far it stands off the segment between its two
    neighbours, the nearest point of that segment's middle 40%, the neighbours)} for vertices
    with two neighbours, and the number without (ends and junctions)."""
    near: dict[int, list[int]] = {}
    for a, b in edges.tolist():
        near.setdefault(a, []).append(b)
        near.setdefault(b, []).append(a)
    out, ends = {}, 0
    for v, others in near.items():
        if len(others) != 2:
            ends += 1
            continue
        a, b = vertices[others[0]], vertices[others[1]]
        ab = b - a
        t = float(np.dot(vertices[v] - a, ab) / max(float(np.dot(ab, ab)), 1e-12))
        off = float(np.linalg.norm(vertices[v] - (a + np.clip(t, 0.0, 1.0) * ab)))
        out[v] = (off, a + np.clip(t, 0.3, 0.7) * ab, others[0], others[1])
    return out, ends


def _level_hem(mesh: Mesh, slot: int, names, passes: int = 24) -> None:
    """Straighten the waist hem of the top in material slot ``slot``, in place (before the slots
    get their own vertices). Decimation leaves teeth in it (a triangle of trouser up in the
    shirt, or of shirt down in the trousers) and drags the odd hem vertex along the body. A
    face with two edges on the hem is a tooth and goes to the other side; a hem vertex further
    than ``HEM_SMOOTH`` above or below the line between its neighbours moves level with it
    (with every vertex at its position, so the mesh stays closed; no further than leaves its
    faces whole) and takes their weights."""
    from ..core.skin import normalize_weights

    canon = _welded(mesh)
    count = int(mesh.joints.max()) + 1

    def normals(faces):
        tri = mesh.vertices[faces]
        return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])

    for _ in range(passes):
        mine = mesh.face_materials == slot
        edges = waist_hem(mesh, mine, dominant_bones(mesh, names))
        if not len(edges):
            return
        hem = {tuple(sorted(e)) for e in canon[edges].tolist()}
        f = canon[mesh.faces]
        sides = [[tuple(sorted((int(row[k]), int(row[(k + 1) % 3])))) for k in range(3)] for row in f]
        across: dict[tuple, list[int]] = {}
        for i, row in enumerate(sides):
            for edge in row:
                if edge in hem:
                    across.setdefault(edge, []).append(i)
        teeth = [i for i, row in enumerate(sides) if sum(edge in hem for edge in row) >= 2]
        if teeth:
            # One side's teeth a pass (each shortens the hem, so this ends): flipping a tooth and
            # the one it meshes with together would only swap them.
            ours = [i for i in teeth if mine[i]]
            slots = mesh.face_materials.copy()
            for i in ours:
                other = next((j for edge in sides[i] for j in across.get(edge, []) if j != i and not mine[j]), None)
                if other is not None:
                    slots[i] = mesh.face_materials[other]
            if not ours:
                slots[teeth] = slot
            mesh.face_materials = slots
            continue
        shape, _ = hem_shape(mesh.vertices, edges)
        moved = False
        for v, (_, target, a, b) in shape.items():
            # Along the body only (up, in the rest pose): the loop keeps its girth.
            target = np.array([mesh.vertices[v][0], target[1], mesh.vertices[v][2]])
            if abs(target[1] - mesh.vertices[v][1]) <= HEM_SMOOTH:
                continue
            group = np.flatnonzero(canon == canon[v])
            faces = mesh.faces[np.isin(mesh.faces, group).any(axis=1)]
            before, old = normals(faces), mesh.vertices[group].copy()
            for step in (1.0, 0.6, 0.3):                 # as far onto the line as leaves its faces whole
                mesh.vertices[group] = old + step * (target - old)
                after = normals(faces)
                flat = np.linalg.norm(after, axis=1) < 0.1 * np.linalg.norm(before, axis=1)
                if not (flat | (np.einsum("ij,ij->i", before, after) <= 0)).any():
                    break
            else:
                mesh.vertices[group] = old
                continue
            dense = np.zeros((2, count))
            np.put_along_axis(dense, mesh.joints[[a, b]], mesh.weights[[a, b]], axis=1)
            joints, weights = normalize_weights(np.arange(count)[None], dense.mean(axis=0)[None])
            mesh.joints[group], mesh.weights[group] = joints[0], weights[0]
            moved = True
        if not moved:
            return


HEM_BONES = ("Hips", "Spine", "Chest", "UpperChest")
HEM_LIP = 0.004               # m: the bare ring of a hem pair sits 2 mm from the garment's


def smooth_hem(mesh: Mesh, slot: int, names) -> None:
    """Make the waist hem of the top in material slot ``slot`` one smooth loop that moves as one
    (in place, before the slots get their own vertices): level it (``_level_hem``), then give
    it, and the ring under its lip, to the trunk alone. A hem that crosses the tops of the leg
    tubes would otherwise swing with the thighs there: it follows the pelvis, not the legs."""
    from scipy.spatial import cKDTree

    from ..core.skin import normalize_weights

    _level_hem(mesh, slot, names)
    edges = waist_hem(mesh, mesh.face_materials == slot, dominant_bones(mesh, names))
    if not len(edges):
        return
    lip = np.flatnonzero(cKDTree(mesh.vertices[np.unique(edges)]).query(mesh.vertices)[0] < HEM_LIP)
    trunk = np.isin(np.asarray(names), HEM_BONES)
    dense = np.zeros((len(lip), len(names)))
    np.put_along_axis(dense, mesh.joints[lip], mesh.weights[lip], axis=1)
    dense[:, ~trunk] = 0.0
    dense[dense.sum(axis=1) < 1e-6, list(names).index("Hips")] = 1.0
    mesh.joints[lip], mesh.weights[lip] = normalize_weights(
        np.tile(np.arange(len(names)), (len(lip), 1)), dense / dense.sum(axis=1, keepdims=True))


WAIST_OVERLAP = 0.03          # m a top and a bottom overlap at the waist, at least


def _position(path, s: float, bone: str) -> tuple[str, float]:
    """Arc length ``s`` of a chain path as a (bone, t) position on ``bone``'s line."""
    i = path.bones.index(bone)
    return bone, float((s - path.starts[i]) / path.lengths[i])


def lap_waist(worn: list, by_name: dict, skeleton: Skeleton, shells: list) -> float | None:
    """Make the outfit's top and bottom overlap by ``WAIST_OVERLAP`` at least, as geometry: a
    tight bottom that stops short of that under the top's hem runs on up under it (hidden: the
    top's layer wins), and a top whose hem is less than that below a loose bottom's waistband
    (its highest ring) comes down. Returns the overlap in metres (None without both)."""
    from .ringloft import ChainPath

    top = next((g for g in worn if g["part"] == "top" and g["spans"]), None)
    bottom = next((g for g in worn if g["part"] == "bottom"), None)
    if top is None or bottom is None:
        return None
    laps = []
    for name, low in bottom["spans"].items():
        high = top["spans"].get(name)
        if high is None:
            continue
        path = ChainPath(skeleton, by_name[name])
        a0, a1 = path.s_of(*high.start), path.s_of(*high.end)
        b0, b1 = path.s_of(*low.start), path.s_of(*low.end)
        if b1 < a1 and b1 - a0 < WAIST_OVERLAP and high.layer > low.layer:
            b1 = a0 + WAIST_OVERLAP
            low.end = _position(path, b1, low.end[0])
        laps.append(min(a1, b1) - max(a0, b0))
    if bottom["shell"] is not None:
        for chain_name, chain_spec in shells[bottom["shell"]][0]["chains"].items():
            chain = parse_chain(chain_name, chain_spec, skeleton)
            shell_path = ChainPath(skeleton, chain)
            band = max(float(shell_path.point(shell_path.s_of(r.bone, r.t))[1]) for r in chain.rings)
            for name, high in top["spans"].items():
                if not set(chain.bones) & set(by_name[name].bones):
                    continue                                  # not what the waistband goes round
                path = ChainPath(skeleton, by_name[name])
                a0 = path.s_of(*high.start)
                lap = band - float(path.point(a0)[1])
                if lap < WAIST_OVERLAP:
                    high.start = _position(path, a0 - (WAIST_OVERLAP - lap), high.start[0])
                    lap = WAIST_OVERLAP
                laps.append(lap)
    return min(laps) if laps else None


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


HUG_BONES = ("Hips", "Spine", "Chest", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg")
HUG_CLEARANCE = 0.01          # m a loose garment's outer wall stands off the body
HUG_REACH = 0.05              # m: how far inside the body's surface a buried start-cap vertex is still brought out


def hug_body(shell: Mesh, body: Mesh, names, clearance: float = HUG_CLEARANCE,
             s_range: tuple[float, float] | None = None) -> None:
    """Fit a loose garment round the trunk and legs it hangs on (rest pose, in place): a wall
    vertex nearer the body's centre line than the body's own surface plus ``clearance`` (wide
    hips under a waistband authored for narrower ones) moves straight out to it, the lining one
    cloth thickness inside the outer wall. So the waistband sits on the body under the top's hem
    for every build, with no skin showing through, and a skirt authored close to the body gets
    its room from the body under it, not from a fixed ease. The whole wall from the first key
    ring (arc length ``s_range[0]``) down is fitted, however far inside the body it was
    authored; the buried start cap before it only within ``HUG_REACH`` of the surface."""
    keep = np.isin(dominant_bones(body, names), HUG_BONES)
    tris = body.vertices[body.faces[keep[body.faces].all(axis=1)]]
    if not len(tris):
        return
    origin = shell.vertices * [0.0, 1.0, 0.0]
    radial = shell.vertices - origin
    radius = np.linalg.norm(radial, axis=1)
    ok = radius > 1e-6
    direction = np.where(ok[:, None], radial / np.maximum(radius, 1e-9)[:, None], 0.0)
    lining = np.zeros(len(radius), bool)
    outer = np.zeros(len(radius), bool)
    inner_faces = lining_faces(shell)
    lining[shell.faces[inner_faces].ravel()] = True
    outer[shell.faces[~inner_faces].ravel()] = True
    # A wall vertex answers for the patch of cloth round it, half-way to the neighbouring rings
    # and vertices: the body is measured over that patch (a hip that bulges between two rings,
    # or between two vertices of a 20-sided ring, would show through a wall fitted at the
    # vertices alone). The lining follows the outer wall: its own ray does.
    heights, turns = _patch(shell, outer)
    extent = np.zeros(len(radius))
    for dy, dt in [(0.0, 0.0)] + [(h, t) for h in (-1.0, 0.0, 1.0) for t in (-1.0, 0.0, 1.0) if h or t]:
        rows = np.flatnonzero(ok) if dy == 0 and dt == 0 else np.flatnonzero(ok & outer)
        angle = dt * turns[rows]
        c, s = np.cos(angle), np.sin(angle)
        d = direction[rows]
        d = np.column_stack([c * d[:, 0] + s * d[:, 2], d[:, 1], -s * d[:, 0] + c * d[:, 2]])
        o = origin[rows] + np.outer(dy * heights[rows] if dy else np.zeros(len(rows)), [0.0, 1.0, 0.0])
        extent[rows] = np.maximum(extent[rows], _furthest_hit(o, d, tris))
    want = extent + np.where(lining & ~outer, clearance * 0.4, clearance)
    wall = np.zeros(len(radius), bool) if s_range is None or shell.uvs is None else shell.uvs[:, 1] >= s_range[0] - 1e-6
    move = ok & (radius < want) & ((radius > want - HUG_REACH) | wall)
    shell.vertices[move] = origin[move] + direction[move] * want[move, None]


def _patch(shell: Mesh, outer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per vertex of a lofted shell: half the height to the next ring (the larger of the two
    sides) and half the angle round the centre line to the next vertex of its ring, from its
    UVs (v: arc length along the chain, one value per ring; u: the way round it)."""
    heights = np.zeros(len(shell.vertices))
    turns = np.zeros(len(shell.vertices))
    if shell.uvs is None:
        return heights, turns
    s = np.round(shell.uvs[:, 1], 5)
    levels, ring_of = np.unique(s, return_inverse=True)
    y = np.array([shell.vertices[(ring_of == k) & outer, 1].mean() if ((ring_of == k) & outer).any()
                  else shell.vertices[ring_of == k, 1].mean() for k in range(len(levels))])
    gap = np.abs(np.diff(y))
    half = np.zeros(len(levels))
    if len(levels) > 1:
        half[:-1] = gap / 2
        half[1:] = np.maximum(half[1:], gap / 2)
    heights = half[ring_of]
    for k in range(len(levels)):
        mine = ring_of == k
        sides = max(len(np.unique(np.round(shell.uvs[mine, 0], 6))) - 1, 1)
        turns[mine] = np.pi / sides
    return heights, turns


def _furthest_hit(origin: np.ndarray, direction: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Distance along each ray to its furthest crossing of ``tris`` (0: none): Moeller-Trumbore."""
    a, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    extent = np.zeros(len(origin))
    for i in range(0, len(origin), 256):
        d, o = direction[i:i + 256, None, :], origin[i:i + 256, None, :]
        p = np.cross(d, e2[None])
        det = np.einsum("fk,rfk->rf", e1, p)
        inv = 1.0 / np.where(np.abs(det) < 1e-12, 1e-12, det)
        t = o - a[None]
        u = np.einsum("rfk,rfk->rf", t, p) * inv
        q = np.cross(t, e1[None])
        v = np.einsum("rfk,rfk->rf", np.broadcast_to(d, q.shape), q) * inv
        dist = np.einsum("fk,rfk->rf", e2, q) * inv
        hit = (np.abs(det) > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (dist > 0)
        extent[i:i + 256] = np.where(hit, dist, 0.0).max(axis=1)
    return extent


LEG_SPLIT = 0.25               # how much more of its own side's thigh a skirt vertex follows (0: both alike .. 0.5)


CAP_EASE = 0.02          # m of arc length before the first key ring over which a skirt's start cap takes the body's weights


def leg_weights(shell: Mesh, skeleton: Skeleton, body: Mesh | None = None, rise: float = 0.07,
                fall: float = 0.11, split: float = LEG_SPLIT, seam: float = 0.08, cap: float | None = None) -> None:
    """Skirt weights: the waist rides the Hips; from ``rise`` above the hip joints to ``fall``
    below them the shell hands over to the two thighs: half each, ``split`` more of its own
    side's (eased across ``seam`` either side of the centre line), so the cloth mostly hangs
    between the legs in a stride but leans the way its leg goes. With the
    ``body``, ``rise`` is at least how far its thighs' fronts stand off the hip joints: how high
    their tops come when they lift level, so no ring that stays level is in their way. So it goes
    where the legs go together (up onto the lap when seated) and hangs from the pelvis when they
    part (a stride), and its rings fan round the hip like an elbow's. ``drape`` then gives each
    ring its girth back and room for the legs. No shin weights: a hem near the knee doesn't swing
    down with the lower leg when seated. ``cap``: the arc length (uv v) of the first key ring; the
    start cap before it, tucked under a top round the spine, rides what the ``body`` there does."""
    from ..core.skin import normalize_weights

    names = skeleton.names
    hip = skeleton["LeftUpperLeg"].head
    hip_y = float(hip[1])
    if body is not None:
        thighs = np.isin(dominant_bones(body, names), ("LeftUpperLeg", "RightUpperLeg"))
        top = body.vertices[thighs & (body.vertices[:, 1] > hip_y - 0.15)]
        if len(top):
            rise = max(rise, float((top[:, 2] - hip[2]).max()) + 0.02)
    y = shell.vertices[:, 1]
    t = np.clip((hip_y + rise - y) / (rise + fall), 0.0, 1.0)
    thigh = t * t * (3 - 2 * t)
    dense = np.zeros((len(y), len(names)))
    dense[:, names.index("Hips")] = 1 - thigh
    u = np.clip((shell.vertices[:, 0] + seam) / (2 * seam), 0.0, 1.0)
    left = 0.5 + split * (2 * u * u * (3 - 2 * u) - 1)
    dense[:, names.index("LeftUpperLeg")] = thigh * left
    dense[:, names.index("RightUpperLeg")] = thigh * (1 - left)
    if body is not None and cap is not None and shell.uvs is not None:
        # The start cap takes as much of the body's weights as is on bones other than the pelvis
        # and legs (bent forward, the spine would take the skin out through a cap on the Hips).
        under = Mesh(shell.vertices.copy(), shell.faces)
        transfer_weights(under, body, neighbours=4, smooth=0)      # (the skin right under it: smoothed, the pelvis leaks in)
        body_dense = np.zeros_like(dense)
        np.put_along_axis(body_dense, under.joints, under.weights, axis=1)
        share = 1.0 - body_dense[:, [names.index(n) for n in ("Hips", "LeftUpperLeg", "RightUpperLeg")]].sum(axis=1)
        share = np.clip(share, 0.0, 1.0) * np.clip((cap - shell.uvs[:, 1]) / CAP_EASE, 0.0, 1.0)
        dense = (1 - share[:, None]) * dense + share[:, None] * body_dense
    shell.joints, shell.weights = normalize_weights(np.tile(np.arange(len(names)), (len(y), 1)), dense)


def _key_range(garment: dict, skeleton: Skeleton) -> tuple[float, float] | None:
    """A single-tube loose garment's (drape.py can drape it) first and last key rings' arc lengths."""
    if len(garment["chains"]) != 1 or garment.get("cut"):
        return None
    from .ringloft import ChainPath

    chain = parse_chain(*next(iter(garment["chains"].items())), skeleton)
    keys = [ChainPath(skeleton, chain).s_of(r.bone, r.t) for r in chain.rings]
    return min(keys), max(keys)


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
    runtime can switch poses by playing clips. A pose with hands resting on a support says
    which (``rests``: its IK ends marked ``rest``), so a runtime can put them on the real one."""
    from .clips import tracks_from_poses

    times = np.array([0.0, POSE_CLIP_SECONDS])
    return [Clip(f"pose_{name}", tracks_from_poses(skeleton, [Pose.from_spec(spec, skeleton)] * 2, times),
                 loop=False, meta={"rests": Pose.rest_ends(spec)} if Pose.rest_ends(spec) else {})
            for name, spec in poses.items()]


def sculpted_head(spec: dict, face: dict | None, skeleton: Skeleton) -> tuple[HeadShape, np.ndarray]:
    """The body's ``head:`` block as a ``HeadShape`` and where its frame sits (the painted eye
    line, on the head bone's line), with eyes and mouth placed where the face texture paints them."""
    from ..textures.character import FaceTextureGenerator

    head_joint = skeleton["Head"].head
    size = float(spec.get("size", 1.0))
    shape = {k: float(spec[k]) for k in SHAPE_DEFAULTS if k in spec}
    unknown = set(spec) - set(SHAPE_DEFAULTS) - HEAD_KEYS
    if unknown:
        raise ValueError(f"head: unknown keys {sorted(unknown)}")
    eyes_y, eye_spacing, mouth_y = float(spec.get("eyes", head_joint[1] + 0.07 * size)), 0.072 * size, None
    if face:
        x0, y0, x1, y1 = (float(v) for v in face["box"])
        paint = FaceTextureGenerator
        eyes_y = y0 + paint.eye_v * (y1 - y0)
        eye_spacing = paint.eye_spacing * (x1 - x0)
        mouth_y = y0 + paint.mouth_v * (y1 - y0)
    origin = np.array([0.0, eyes_y, float(head_joint[2])])
    neck = skeleton["Neck"].head
    head = HeadShape(size, eye_spacing=eye_spacing,
                     mouth_drop=(eyes_y - mouth_y) if mouth_y is not None else 0.084 * size,
                     neck_radius=float(spec.get("neck_radius", 0.048 * size)),
                     neck_drop=eyes_y - float(neck[1]), **shape)
    return head, origin


def sdf_occlusion(sdf, points: np.ndarray, normals: np.ndarray, steps: int = 5, reach: float = 0.04) -> np.ndarray:
    """Ambient occlusion of surface ``points`` of a signed-distance ``sdf`` (1 open .. 0 shut): how
    far the field falls short of the distance stepped out along each normal, nearer steps
    counting more."""
    occ = np.zeros(len(points))
    weight = 1.0
    for i in range(1, steps + 1):
        h = reach * i / steps
        occ += weight * np.maximum(h - sdf(points + normals * h), 0.0) / h
        weight *= 0.7
    return np.clip(1.0 - occ / sum(0.7 ** k for k in range(steps)), 0.0, 1.0)


def head_weights(mesh: Mesh, skeleton: Skeleton, blend: float, shift: float) -> tuple[np.ndarray, np.ndarray]:
    """The head moves with Head; below the jaw (``shift`` under the head joint) it hands over to
    Neck over ``blend`` metres, as the torso chain's head did."""
    names = skeleton.names
    joint = float(skeleton["Head"].head[1]) + shift
    share = np.clip((mesh.vertices[:, 1] - (joint - blend)) / (2 * blend), 0.0, 1.0)
    share = share * share * (3 - 2 * share)
    dense = np.zeros((len(mesh.vertices), len(names)))
    dense[:, names.index("Head")] = share
    dense[:, names.index("Neck")] = 1 - share
    return normalize_weights(np.tile(np.arange(len(names)), (len(mesh.vertices), 1)), dense)


def build_humanoid(spec: dict, name: str, material_loader, assets_dir: Path, params: dict | None = None,
                   root_poses: dict | None = None) -> SceneNode:
    """``root_poses`` are the asset's top-level ``poses:`` (where the root goes per pose), which
    transition clips move the body between."""
    import copy

    skeleton = load_skeleton(content_roots.find(spec.get("skeleton", "skeletons/humanoid.yaml"), assets_dir),
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
    worn = []                     # meta.humanoid.garments: what is worn, and (below) its vertices
    levels = []                   # span ends given as {level: chain}
    for item in outfits[outfit_name] or []:
        garment = load_garment(assets_dir, item["garment"], params or {})
        color = _color(item.get("color", "grey"), CLOTH_COLORS, "cloth colour")
        base = material_loader.load(garment.get("material", "cloth"))
        worn.append({"name": item["garment"], "part": garment.get("part"), "slot": None, "shell": None, "spans": {}})
        if garment.get("cover"):
            slot = len(materials)
            worn[-1]["slot"] = slot
            materials.append(copy.copy(base))
            slot_colors.append((color, base))
            for chain_name, (start, end) in garment["cover"].items():
                if chain_name not in by_name:
                    raise ValueError(f"garment '{item['garment']}': body has no chain '{chain_name}'")
                # An end may be {level: <chain>}: level with where this garment's cover of that chain starts.
                ends = [(e.get("bone"), 0.0) if isinstance(e, dict) else (e[0], float(e[1])) for e in (start, end)]
                ends = [(bone or ends[1 - k][0], t) for k, (bone, t) in enumerate(ends)]
                # (stiffness is for the trunk: a sleeve or a trouser leg has no hollows to span)
                ease = float(garment.get("ease", 0.0)) if chain_name == "torso" else 0.0
                span = Span(ends[0], ends[1], float(garment.get("thickness", 0.005)), slot, int(garment.get("layer", 1)),
                            ease)
                by_name[chain_name].spans.append(span)
                worn[-1]["spans"][chain_name] = span
                levels += [(worn[-1], chain_name, k, e["level"]) for k, e in enumerate((start, end)) if isinstance(e, dict)]
        if garment.get("sole"):
            sole = dict(garment["sole"])
            if set(sole) - SOLE_KEYS:
                raise ValueError(f"garment '{item['garment']}': sole: unknown keys {sorted(set(sole) - SOLE_KEYS)}")
            worn[-1]["sole"] = len(materials)
            worn[-1]["sole_height"] = float(sole.get("height", 0.012))
            sole_base = material_loader.load(sole.get("material", "leather"))
            materials.append(copy.copy(sole_base))
            slot_colors.append((_color(sole.get("color", "black"), CLOTH_COLORS, "sole colour"), sole_base))
        if garment.get("chains"):
            worn[-1]["shell"] = len(loose)
            loose.append((garment, color, base))

    waist_overlap = lap_waist(worn, by_name, skeleton, loose)
    for entry, chain_name, k, ref in levels:
        from .ringloft import ChainPath

        if ref not in entry["spans"]:
            raise ValueError(f"garment '{entry['name']}': level '{ref}' is not a chain it covers")
        ref_path = ChainPath(skeleton, by_name[ref])
        height = ref_path.point(ref_path.s_of(*entry["spans"][ref].start))[1]
        span, path = entry["spans"][chain_name], ChainPath(skeleton, by_name[chain_name])
        bone = (span.start, span.end)[k][0]
        i = path.bones.index(bone)
        at = (bone, float((height - path.points[i][1]) / (path.points[i + 1][1] - path.points[i][1])))
        span.start, span.end = (at, span.end) if k == 0 else (span.start, at)
    mesh = loft_body(skeleton, chains, crease, mirrored, materials if len(materials) > 1 else None)
    budget = spec.get("max_triangles")
    if budget and len(mesh.faces) > int(budget):
        mesh = meshops.decimate(mesh, int(budget) / len(mesh.faces), crease_angle=crease)
    # Sculpted parts go on after the body's decimation, which would flatten them: soles (their flat
    # tread and sharp edge are the point), hands and the head, joined in one union.
    parts = []

    def sculpted(part: Mesh, slot: int = 0) -> None:
        if len(materials) > 1:
            part.face_materials = np.full(len(part.faces), slot, dtype=np.int64)
            part.materials = list(materials)
        part.material = skin
        parts.append(part)

    for entry in worn:
        if entry.get("sole") is not None:
            left = build_sole(mesh, skeleton, entry["sole_height"])
            if left is not None:
                for sole in (left, mirror_mesh(left, skeleton)):
                    sculpted(sole, entry["sole"])
    hands = None
    if spec.get("hands"):
        unknown = set(spec["hands"]) - HAND_KEYS
        if unknown:
            raise ValueError(f"'{name}': body.hands: unknown keys {sorted(unknown)} (known: {sorted(HAND_KEYS)})")
        wrist = tuple(float(r) for r in spec["hands"].get("wrist", (0.02, 0.028)))
        left, hands = build_hand(skeleton, float(spec["hands"].get("size", 1.0)), wrist)
        for hand_mesh in (left, mirror_mesh(left, skeleton)):
            sculpted(uvmap.box_project(hand_mesh))
    head = None
    if spec.get("head"):
        head, head_origin = sculpted_head(spec["head"], spec.get("face"), skeleton)
        head_mesh = build_head(head, head_origin)
        head_mesh.joints, head_mesh.weights = head_weights(head_mesh, skeleton, float(spec["head"].get("blend", 0.04)),
                                                           float(spec["head"].get("weight_shift", -0.06)))
        sculpted(uvmap.box_project(head_mesh))
    if parts:
        if len(materials) > 1 and mesh.materials is None:
            mesh.materials = list(materials)
        mesh = csg.union(mesh, *parts, crease_angle=crease)
    if len(materials) > 1:
        # Map the union's slots (ordered by first appearance) back to ours by material identity.
        slots, result_materials = mesh.effective_face_materials()
        ours = [next(i for i, m in enumerate(materials) if m is r) for r in result_materials]
        mesh.face_materials = np.asarray(ours)[slots]
        mesh.materials = list(materials)
        for entry in worn:
            if entry["part"] == "top" and entry["slot"] is not None:
                smooth_hem(mesh, entry["slot"], skeleton.names)
        slots_out: list = []
        mesh = split_by_slot(mesh, slot_colors, slots_out)
        vertex_slot = slots_out[0]
    else:
        mesh.material = skin
        mesh.colors = np.tile([*tone, 1.0], (len(mesh.vertices), 1))
        vertex_slot = np.zeros(len(mesh.vertices), dtype=np.int64)
    if spec.get("face"):
        face = dict(spec["face"])
        mesh = apply_face(mesh, face["box"], face_material(face, skin), float(face.get("min_normal", 0.25)))
    if head is not None or hands is not None:
        # Shade the head and hands from their sculpts, not from how the simplifier triangulated them.
        mesh.normals = meshops.ensure_normals(mesh, crease).normals.copy()
        if head is not None:
            on = np.abs(head(mesh.vertices - head_origin)) < 2e-4
            mesh.normals[on] = sdf_normals(head, mesh.vertices[on] - head_origin)
            # Shade it from its own sculpt: sockets, under the nose and lip, round the ears and
            # under the jaw (a flat-lit face otherwise reads as paint on an egg).
            if mesh.colors is not None:
                bare = on & np.all(np.abs(mesh.colors[:, :3] - tone) < 1e-6, axis=1)    # (not a collar on the neck)
                ao = sdf_occlusion(head, mesh.vertices[bare] - head_origin, mesh.normals[bare])
                mesh.colors[bare, :3] *= (1.0 - HEAD_AO * (1.0 - ao))[:, None]
        if hands is not None:
            for side in (1, -1):                         # the right hand is the left's mirror image
                local = mesh.vertices * [side, 1, 1]
                on = (local[:, 0] > 0) & (np.abs(hands(local)) < 2e-4)
                mesh.normals[on] = sdf_normals(hands, local[on]) * [side, 1, 1]

    root = SceneNode(name)
    joints, nodes = skeleton.build()
    root.add_child(joints)
    body = root.add_child(SceneNode("body", mesh=mesh))
    body.meta["collider"] = "none"
    body.skin = Skin.bind(body, [nodes[n] for n in skeleton.names], name=f"{name}_skin")
    # The closed body, kept beside the drawn one (bound now, in the rest pose) in case loose
    # clothes hide faces of it: fitting, draping and QA need the whole solid (closed_body).
    closed = root.add_child(SceneNode(CLOSED_BODY, mesh=mesh))
    closed.hidden = True
    closed.meta["collider"] = "none"
    closed.meta["build_only"] = True
    closed.skin = Skin.bind(closed, [nodes[n] for n in skeleton.names], name=f"{name}_closed_skin")
    hair_style = spec.get("hair", "none")
    if hair_style not in (None, "none"):
        styles = spec.get("hair_styles") or {}
        if hair_style not in styles:
            raise ValueError(f"'{name}': unknown hair style '{hair_style}'. Styles: {sorted(styles)}")
        color = _color(spec.get("hair_color", "brown"), HAIR_COLORS, "hair colour")
        style = styles[hair_style]
        if style.get("sculpt") is not None:
            if head is None:
                raise ValueError(f"'{name}': hair style '{hair_style}' is sculpted, which needs a body.head")
            hair_mesh = build_sculpted_hair(head, head_origin, style["sculpt"], skeleton, color)
            hair_mesh.material = material_loader.load(style.get("material", "hair"))
        else:
            hair_mesh = build_hair(style, skeleton, material_loader, crease)
            if hair_mesh is not None:
                hair_mesh.colors = np.tile([*color, 1.0], (len(hair_mesh.vertices), 1))
        if hair_mesh is not None:
            hair = root.add_child(SceneNode("hair", mesh=hair_mesh))
            hair.meta["collider"] = "none"
            hair.skin = Skin.bind(hair, [nodes[n] for n in skeleton.names], name=f"{name}_hair_skin")
    draped = []                   # loose shells drape.fit bakes pose correctives for
    if loose:
        shells = []
        starts = [0]
        for garment, color, base in loose:
            shell = build_hair({"chains": garment["chains"], "cut": garment.get("cut"),
                                "material": garment.get("material", "cloth")}, skeleton, material_loader, crease)
            keys = _key_range(garment, skeleton)
            hug_body(shell, mesh, skeleton.names, s_range=keys)
            mode = garment.get("weights", "body")
            if mode == "body":
                transfer_weights(shell, mesh)
            elif mode == "legs":
                leg_weights(shell, skeleton, mesh, cap=keys[0] if keys else None)
            else:
                raise ValueError(f"garment '{garment.get('name')}': unknown weights '{mode}' (body | legs)")
            shell.colors = np.tile([*color, 1.0], (len(shell.vertices), 1))
            inside = np.unique(shell.faces[lining_faces(shell)])
            shell.colors[inside, :3] *= LINING_SHADE
            shells.append(shell)
            starts.append(starts[-1] + len(shell.vertices))
        for entry in worn:
            if entry["shell"] is not None:
                entry["shell_vertices"] = np.arange(starts[entry["shell"]], starts[entry["shell"] + 1])
        for k, (garment, _, _) in enumerate(loose):
            keys = _key_range(garment, skeleton)
            if keys is not None:
                draped.append({"vertices": np.arange(starts[k], starts[k + 1]), "s_range": keys})
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
        roots = {n: dict(r) for n, r in (root_poses or {}).items()}
        if "sit" in roots and "sit" in seats:                  # as fit_seated_pose will place it
            roots["sit"]["offset"] = [roots["sit"]["offset"][0], -seats["sit"], roots["sit"]["offset"][2]]
        root.clips.extend(generate_clip(skeleton, base, clip_name, clip_spec or {}, poses=poses, roots=roots)
                          for clip_name, clip_spec in spec["clips"].items())
    from . import drape

    drape_report, hidden = drape.fit(root, draped, set(seats)) if draped else ({}, None)
    from .arms import clear_arms

    arms_report = clear_arms(root, draped) if spec.get("clear_arms", True) else {}
    # Skin faces the loose clothes hide in every pose and sampled frame (all three vertices well
    # inside the draped cloth) are left out of the drawn body; the closed one stays for the
    # generator. Vertices are kept in place, so garments' and morphs' indices still hold. Tight
    # garments keep every face: a top's hem under a skirt's waistband is what the waist QA reads.
    drop = hidden[mesh.faces].all(axis=1) if hidden is not None else None
    if drop is not None and mesh.face_materials is not None:
        drop &= mesh.face_materials == 0
    if drop is not None and drop.any():
        drawn = mesh.copy()
        drawn.faces = mesh.faces[~drop]
        if mesh.face_materials is not None:
            drawn.face_materials = mesh.face_materials[~drop]
        body.mesh = drawn
    else:
        root.remove_child(closed)
    if pose_name is not None:
        if pose_name not in poses:
            raise ValueError(f"'{name}': unknown pose '{pose_name}'. Poses: {sorted(poses)}")
        # As its clip has it: with the swing clear_arms gave the arms (and the drape's morphs).
        clip = next((c for c in root.clips if c.name == f"pose_{pose_name}"), None)
        if clip is not None:
            clip.apply(root, 0.0)
        else:
            Pose.from_spec(poses[pose_name], skeleton).apply(skeleton, nodes)
    posed = Mesh.merge([n.world_mesh() for n in root.iter_visible() if n.mesh is not None])
    lo, hi = posed.vertices.min(axis=0), posed.vertices.max(axis=0)
    root.size = np.array([2 * max(abs(lo[0]), abs(hi[0])), hi[1], 2 * max(abs(lo[2]), abs(hi[2]))])
    root.meta["humanoid"] = {"skeleton": skeleton.name, "height": round(float(skeleton.landmarks["head_top"][1]), 4)
                             if "head_top" in skeleton.landmarks else round(float(hi[1]), 4),
                             "pose": pose_name, "hair": hair_style, "outfit": outfit_name,
                             "triangles": int(sum(len(n.mesh.faces) for n in root.iter_visible() if n.mesh is not None))}
    if drop is not None and drop.any():
        root.meta["humanoid"]["hidden_faces"] = int(drop.sum())
    root.meta["humanoid"]["seats"] = seats
    root.meta["humanoid"]["shells"] = draped            # for drape.Draper (review, tests)
    if drape_report:
        root.meta["humanoid"]["drape"] = drape_report
    if arms_report:
        root.meta["humanoid"]["arms"] = arms_report     # {clip: degrees its arms swing out to clear this body}
    if waist_overlap is not None:
        root.meta["humanoid"]["waist_overlap"] = round(float(waist_overlap), 4)
    # What is worn, for clipping QA: each garment's vertices of `body` (tight) and of `clothes` (loose).
    # (The face's vertices are copies appended after the split: bare skin.)
    root.meta["humanoid"]["garments"] = [
        {"name": g["name"], "part": g["part"],
         "body": np.flatnonzero(np.isin(vertex_slot, [k for k in (g["slot"], g.get("sole")) if k is not None])),
         "clothes": g.get("shell_vertices", np.empty(0, dtype=np.int64))} for g in worn]
    root.tags = ["character"]
    return root


CLOSED_BODY = "body_closed"      # the whole body, hidden and build-only, beside a drawn `body` with faces left out


def closed_body(root: SceneNode) -> SceneNode | None:
    """The skinned body of a character to fit, drape and measure against: its closed body
    (``CLOSED_BODY``, kept when loose clothes hide faces of the drawn ``body``), else ``body``."""
    found = None
    for node in root.iter_nodes():
        if node.skin is not None and node.mesh is not None:
            if node.name == CLOSED_BODY:
                return node
            if node.name == "body" and found is None:
                found = node
    return found


SEAT_SINK = 0.0015       # m the seat flesh/cloth presses into a seat (within clipping QA's tolerance)
SEAT_CLEAR = 0.03        # m the calves keep in front of a seat's front edge
KNEE_OVERHANG = 0.3      # share of the thigh, behind the knee joint, that clears the seat front


def seat_contact(root, clip: str) -> tuple[float, SceneNode, np.ndarray, np.ndarray] | None:
    """The underside of what rests on a seat in ``clip`` (body frame y): the buttocks and the
    thighs, or cloth just under them. Also returns the posed actor, body mesh
    and each vertex's dominant bone."""
    from ..core.skin import pose_clips as apply_clip

    actor = root.instance(share_clips=True)
    apply_clip(actor, clip, 0.0)
    body = closed_body(actor)
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
    # A skirt under the seat flesh adds its thickness (drape.py lays it on the seat plane).
    clothes = next((n for n in actor.iter_nodes() if n.name == "clothes" and n.mesh is not None), None)
    if clothes is not None:
        from .drape import SEAT_CLOTH

        cv = clothes.world_mesh().vertices
        if ((cv[:, 1] < contact + 0.1) & (cv[:, 2] < front)).any():
            contact -= SEAT_CLOTH
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
