"""Ring-loft skins: closed tubes lofted along skeleton bone chains, skinned to the bones.

A ``Chain`` follows consecutive bones of a ``core.skeleton.Skeleton`` in its
rest pose (bone heads, then ``end``). Key ``Ring`` cross-sections sit at
``(bone, t)`` positions (t = 0 at the bone's head, 1 at the next head; values
outside 0-1 extend along the bone's line) with half-widths ``rx`` (the ring's
side axis) and ``rz`` (its front axis), an ``offset`` [side, front] in metres,
a superellipse ``power`` (2 = ellipse, higher = boxier) and a ``twist``.

The ring frame is the chain tangent plus the chain's ``front`` hint (body
frame, default +Z): front = the hint made perpendicular to the tangent,
side = front x tangent. So limbs keep one consistent cross-section frame
whatever the bones' rolls are.

Between key rings the loft resamples every ``spacing`` metres, interpolating
ring parameters with a monotone cubic (few authored rings give smooth limbs
without overshoot). Ends are ``round`` (hemispherical caps) or ``flat``.

Weights come from the arc length: each vertex belongs to its bone, blending
smoothly into the neighbouring bone within ``blend`` metres of a joint;
``bind`` spreads a bone's share over other bones (a mitten hand over the four
finger chains). ``loft_body`` unions all chains (manifold3d) into one
watertight mesh with metric UVs (cylindrical per chain) and JOINTS/WEIGHTS
indexed by ``skeleton.names``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core import csg, meshops
from ..core.mesh import Mesh
from ..core.skeleton import Skeleton
from ..core.skin import normalize_weights


@dataclass
class Ring:
    bone: str
    t: float
    rx: float
    rz: float
    offset: tuple[float, float] = (0.0, 0.0)
    power: float = 2.0
    twist: float = 0.0          # degrees about the tangent


@dataclass
class Span:
    """A tight garment over part of a chain: rings from ``start`` to ``end`` ((bone, t) positions)
    grow by ``thickness`` and their faces take material ``slot``; higher ``layer`` wins."""

    start: tuple[str, float]
    end: tuple[str, float]
    thickness: float
    slot: int
    layer: int = 0


@dataclass
class Chain:
    name: str
    bones: list[str]
    rings: list[Ring]
    end: NDArray[np.float64] | None = None      # where the last bone ends (default: its length along +Y)
    front: tuple[float, float, float] = (0.0, 0.0, 1.0)
    sides: int = 10
    spacing: float = 0.05
    cap_start: str = "round"
    cap_end: str = "round"
    cap_rings: int = 3
    blend: float = 0.05
    bind: dict[str, list[str]] = field(default_factory=dict)
    # Move a bone's weight boundary (its joint) along the chain, metres (e.g. Head: -0.06 so the jaw
    # below the head joint follows the head).
    weight_shift: dict[str, float] = field(default_factory=dict)
    joint_blend: dict[str, float] = field(default_factory=dict)   # per-joint blend width (by the bone after it)
    # Extra rings across joints' blend zones (smooth bending): True for every joint, or the
    # bones whose joint (head) gets them, e.g. [LeftLowerArm] for the elbow only.
    joint_rings: bool | list[str] = True
    spans: list[Span] = field(default_factory=list)   # tight garments (see Span)


def _smoothstep(x: NDArray) -> NDArray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class ChainPath:
    """A chain's rest-pose polyline: bone heads then the end point, with arc length."""

    def __init__(self, skeleton: Skeleton, chain: Chain) -> None:
        points = [skeleton[b].head for b in chain.bones]
        if chain.end is not None:
            end = np.asarray(chain.end, dtype=np.float64)
        else:
            last = skeleton[chain.bones[-1]]
            length = skeleton.length(chain.bones[-1]) or 0.1
            end = last.head + last.rotation[:, 1] * length
        self.points = np.array(points + [end])
        seg = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(seg, axis=1)
        if np.any(self.lengths < 1e-6):
            raise ValueError(f"chain '{chain.name}': zero-length bone in {chain.bones}")
        self.dirs = seg / self.lengths[:, None]
        self.starts = np.concatenate([[0.0], np.cumsum(self.lengths)])   # arc length at each bone head
        self.bones = chain.bones

    def s_of(self, bone: str, t: float) -> float:
        i = self.bones.index(bone)
        return float(self.starts[i] + t * self.lengths[i])

    def segment(self, s: float) -> int:
        return int(np.clip(np.searchsorted(self.starts, s, side="right") - 1, 0, len(self.lengths) - 1))

    def point(self, s: float) -> NDArray[np.float64]:
        i = self.segment(s)
        return self.points[i] + self.dirs[i] * (s - self.starts[i])

    def tangent(self, s: float, smooth: float) -> NDArray[np.float64]:
        """Segment direction, eased across joints within ``smooth`` metres (no ring pinch at kinks)."""
        i = self.segment(s)
        d = self.dirs[i]
        if i > 0 and s - self.starts[i] < smooth:
            f = 0.5 + 0.5 * (s - self.starts[i]) / smooth
            d = self.dirs[i - 1] * (1 - f) + d * f
        elif i < len(self.dirs) - 1 and self.starts[i + 1] - s < smooth:
            f = 0.5 + 0.5 * (self.starts[i + 1] - s) / smooth
            d = self.dirs[i + 1] * (1 - f) + d * f
        return d / np.linalg.norm(d)


def _interpolate(s_keys: NDArray, values: NDArray, s: NDArray) -> NDArray:
    """Monotone cubic interpolation of key ring parameters (constant outside the keys)."""
    if len(s_keys) == 1:
        return np.repeat(values[:1], len(s), axis=0)
    from scipy.interpolate import PchipInterpolator

    return PchipInterpolator(s_keys, values, axis=0, extrapolate=False)(np.clip(s, s_keys[0], s_keys[-1]))


def _sample_rings(path: ChainPath, chain: Chain) -> tuple[NDArray, NDArray, NDArray]:
    """(s per ring, params per ring: rx, rz, off_side, off_front, power, twist; incl. cap rings,
    material slot per interval between consecutive rings)."""
    keys = sorted(chain.rings, key=lambda r: path.s_of(r.bone, r.t))
    s_keys = np.array([path.s_of(r.bone, r.t) for r in keys])
    if np.any(np.diff(s_keys) <= 1e-6):
        raise ValueError(f"chain '{chain.name}': key rings must be at distinct positions")
    params = np.array([[r.rx, r.rz, r.offset[0], r.offset[1], r.power, r.twist] for r in keys], dtype=np.float64)
    count = max(2, int(np.ceil((s_keys[-1] - s_keys[0]) / chain.spacing)) + 1)
    extra = [np.linspace(s_keys[0], s_keys[-1], count)]
    joints = _joint_positions(path, chain)
    if chain.joint_rings is not True:
        wanted = set(chain.joint_rings or [])
        joints = np.array([j for j, bone in zip(joints, chain.bones[1:]) if bone in wanted])
    if len(joints):
        extra.append((joints[:, None] + np.array([-0.6, 0.0, 0.6]) * chain.blend).reshape(-1))
    s = np.concatenate([s_keys, *extra])
    s = np.unique(s[(s >= s_keys[0]) & (s <= s_keys[-1])])
    # Drop samples crowding a key ring or joint ring (keep those exact).
    fixed = np.concatenate([s_keys, *extra[1:]])
    min_gap = min(chain.spacing, chain.blend) * 0.35
    keep = np.array([np.any(np.isclose(fixed, x)) or np.min(np.abs(fixed - x)) > min_gap for x in s])
    s = s[keep]
    # A resample landing within float noise of a key ring (isclose keeps both) would make coincident rings.
    s = s[np.concatenate([[True], np.diff(s) > 1e-5])]
    spans =[(path.s_of(*sp.start), path.s_of(*sp.end), sp) for sp in chain.spans]
    bounds = sorted({b for a, e, _ in spans for b in (a, e) if s_keys[0] < b < s_keys[-1]})
    if bounds:
        s = np.unique(np.concatenate([s, bounds]))
        s = s[[np.any(np.isclose(bounds, x)) or np.min(np.abs(np.asarray(bounds) - x)) > min_gap * 0.5
               for x in s]]
    rings = _interpolate(s_keys, params, s)

    def cover(x: float) -> tuple[float, int]:
        """(thickness, slot) of the topmost garment at arc length x (0, 0 for bare skin)."""
        best = None
        for a, e, sp in spans:
            if a - 1e-9 <= x <= e + 1e-9 and (best is None or sp.layer >= best.layer):
                best = sp
        return (best.thickness, best.slot) if best is not None else (0.0, 0)

    # Garment edges become hem pairs: two rings at one position, the bare side and the covered one.
    out_s, out_rings, out_thick = [], [], []
    eps = 0.002                     # hem rings sit 2 mm apart (a crisp step, clear of tolerances)
    for x, ring in zip(s, rings):
        before, after = cover(x - eps), cover(x + eps)
        if np.any(np.isclose(bounds, x)) and before[0] != after[0]:
            out_s += [x, x + eps]
            out_rings += [ring, ring]
            out_thick += [before, after]
        else:
            out_s.append(x)
            out_rings.append(ring)
            out_thick.append(cover(x))
    s = np.array(out_s)
    rings = np.array(out_rings)
    rings[:, :2] += np.array([t for t, _ in out_thick])[:, None]
    slots = []
    for i in range(len(s) - 1):
        if s[i + 1] - s[i] <= 2 * eps:                     # a hem: the thicker (garment) side's slot
            slots.append(max(out_thick[i], out_thick[i + 1])[1])
        else:
            slots.append(cover((s[i] + s[i + 1]) / 2)[1])

    def cap(at_end: bool):
        style = chain.cap_end if at_end else chain.cap_start
        edge = rings[-1] if at_end else rings[0]
        if style != "round":
            return np.empty(0), np.empty((0, 6))
        length = min(edge[0], edge[1])
        phi = np.linspace(0, np.pi / 2, chain.cap_rings + 2)[1:-1]
        ds = length * np.sin(phi) * (1 if at_end else -1)
        caps = np.repeat(edge[None], len(phi), axis=0)
        caps[:, :2] *= np.cos(phi)[:, None]
        return (s[-1] if at_end else s[0]) + ds, caps

    s0, c0 = cap(False)
    s1, c1 = cap(True)
    end_slots = (slots[0] if slots else cover(s[0])[1], slots[-1] if slots else cover(s[-1])[1])
    slots = [end_slots[0]] * len(s0) + slots + [end_slots[1]] * len(s1)
    return np.concatenate([s0[::-1], s, s1]), np.vstack([c0[::-1], rings, c1]), np.array(slots, dtype=np.int64)


def _cap_pole(path: ChainPath, chain: Chain, s: NDArray, rings: NDArray, at_end: bool) -> float:
    """Arc length of the pole vertex closing a chain end."""
    edge = rings[-1] if at_end else rings[0]
    style = chain.cap_end if at_end else chain.cap_start
    if style != "round":
        return float(s[-1] if at_end else s[0])
    # Pole sits one cap radius past the last full ring (the keys' end).
    key_end = s[-1 - chain.cap_rings] if at_end else s[chain.cap_rings]
    length = min(rings[-1 - chain.cap_rings if at_end else chain.cap_rings][:2])
    return float(key_end + (length if at_end else -length))


def _joint_positions(path: ChainPath, chain: Chain) -> NDArray:
    """Arc length of each joint between consecutive chain bones (with ``weight_shift``)."""
    return np.array([path.starts[i] + chain.weight_shift.get(bone, 0.0)
                     for i, bone in enumerate(chain.bones) if i > 0])


def chain_weights(path: ChainPath, chain: Chain, s: NDArray, bone_index: dict[str, int]) -> NDArray:
    """Dense weights (len(s) x bones) from arc length, blending across each joint within ``blend``."""
    n = len(chain.bones)
    joints = _joint_positions(path, chain)
    widths = [chain.joint_blend.get(bone, chain.blend) for bone in chain.bones[1:]]
    raw = np.zeros((len(s), n))
    for i in range(n):
        lo = (np.ones(len(s)) if i == 0
              else _smoothstep((s - joints[i - 1] + widths[i - 1]) / (2 * widths[i - 1])))
        hi = (np.ones(len(s)) if i == n - 1
              else 1 - _smoothstep((s - joints[i] + widths[i]) / (2 * widths[i])))
        raw[:, i] = lo * hi
    dense = np.zeros((len(s), len(bone_index)))
    for i, bone in enumerate(chain.bones):
        targets = chain.bind.get(bone, [bone])
        for target in targets:
            dense[:, bone_index[target]] += raw[:, i] / len(targets)
    return dense / np.maximum(dense.sum(axis=1, keepdims=True), 1e-12)


def loft_chain(skeleton: Skeleton, chain: Chain) -> Mesh:
    """One closed, skinned, UV'd tube for ``chain``."""
    path = ChainPath(skeleton, chain)
    s, rings, interval_slots = _sample_rings(path, chain)
    sides = chain.sides
    front_hint = np.asarray(chain.front, dtype=np.float64)
    theta = np.linspace(0, 2 * np.pi, sides + 1)          # last column repeats the first (UV seam)
    c, sn = np.cos(theta), np.sin(theta)

    verts, ring_s = [], []
    for sk, (rx, rz, off_side, off_front, power, twist) in zip(s, rings):
        tangent = path.tangent(sk, chain.spacing)
        front = front_hint - np.dot(front_hint, tangent) * tangent
        if np.linalg.norm(front) < 1e-6:
            raise ValueError(f"chain '{chain.name}': front hint is parallel to the chain")
        front /= np.linalg.norm(front)
        side = np.cross(front, tangent)
        tw = np.radians(twist)
        side, front = side * np.cos(tw) + front * np.sin(tw), front * np.cos(tw) - side * np.sin(tw)
        e = 2.0 / power
        x = np.sign(c) * np.abs(c) ** e * rx + off_side
        z = np.sign(sn) * np.abs(sn) ** e * rz + off_front
        ring = path.point(sk) + np.outer(x, side) + np.outer(z, front)
        ring[-1] = ring[0]            # the seam column is the first one, exactly (welds must match)
        verts.append(ring)
        ring_s.append(np.full(sides + 1, sk))
    per = sides + 1
    count = len(s)
    vertices = np.vstack(verts)
    faces, face_slots = [], []
    for r in range(count - 1):
        a = r * per + np.arange(sides)
        faces += np.column_stack([a, a + 1, a + per + 1]).tolist() + np.column_stack([a, a + per + 1, a + per]).tolist()
        face_slots += [interval_slots[r]] * (2 * sides)
    poles = []
    for at_end in (False, True):
        ps = _cap_pole(path, chain, s, rings, at_end)
        rows = rings[-1] if at_end else rings[0]
        tangent = path.tangent(ps, chain.spacing)
        front = front_hint - np.dot(front_hint, tangent) * tangent
        front /= np.linalg.norm(front)
        side = np.cross(front, tangent)
        center = path.point(ps) + side * rows[2] + front * rows[3]
        poles.append((center, ps))
    start_pole, end_pole = len(vertices), len(vertices) + 1
    vertices = np.vstack([vertices, poles[0][0], poles[1][0]])
    last = (count - 1) * per
    for k in range(sides):
        faces.append([start_pole, k + 1, k])
        faces.append([end_pole, last + k, last + k + 1])
        face_slots += [interval_slots[0], interval_slots[-1]]
    faces = np.array(faces, dtype=np.int64)
    all_s = np.concatenate([np.concatenate(ring_s), [poles[0][1], poles[1][1]]])

    perimeter = float(np.mean(np.pi * (rings[:, 0] + rings[:, 1])))
    u = np.concatenate([np.tile(theta / (2 * np.pi) * perimeter, count), [0.0, 0.0]])
    mesh = Mesh(vertices, faces, uvs=np.column_stack([u, all_s]))
    if chain.spans:
        mesh.face_materials = np.array(face_slots, dtype=np.int64)
    if _signed_volume(mesh) < 0:
        mesh.faces = mesh.faces[:, ::-1].copy()
    bone_index = {name: i for i, name in enumerate(skeleton.names)}
    dense = chain_weights(path, chain, all_s, bone_index)
    mesh.joints, mesh.weights = normalize_weights(np.tile(np.arange(len(bone_index)), (len(all_s), 1)), dense)
    return mesh


def mirror_mesh(mesh: Mesh, skeleton: Skeleton) -> Mesh:
    """Reflect a skinned mesh across X (left <-> right), remapping joints to the mirrored bones."""
    from ..core.skeleton import mirror_name

    names = skeleton.names
    remap = np.array([names.index(mirror_name(n) or n) for n in names])
    out = mesh.copy()
    out.vertices = mesh.vertices * [-1, 1, 1]
    out.faces = mesh.faces[:, ::-1].copy()
    if mesh.normals is not None:
        out.normals = mesh.normals * [-1, 1, 1]
    if mesh.joints is not None:
        out.joints = remap[mesh.joints]
    return out


def _signed_volume(mesh: Mesh) -> float:
    v = mesh.vertices[mesh.faces]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)


def loft_body(skeleton: Skeleton, chains: list[Chain], crease_angle: float = 75.0,
              mirrored: set[str] | None = None, materials: list | None = None) -> Mesh:
    """Loft every chain and union them into one watertight skinned mesh (rest pose).

    Chains named in ``mirrored`` also get an exact mirror image (their Right* twin).
    With ``materials`` (one per slot, distinct objects), garment spans' slots become
    material groups that survive the union.
    """
    meshes = []
    for chain in chains:
        mesh = loft_chain(skeleton, chain)
        if materials is not None:
            if mesh.face_materials is None:
                mesh.face_materials = np.zeros(len(mesh.faces), dtype=np.int64)
            mesh.materials = list(materials)
            mesh.material = materials[0]
        meshes.append(mesh)
        if mirrored and chain.name in mirrored:
            meshes.append(mirror_mesh(mesh, skeleton))
    body = meshes[0] if len(meshes) == 1 else csg.union(*meshes, crease_angle=crease_angle)
    if len(meshes) == 1:
        body = meshops.compute_normals(body, crease_angle)
    return body
