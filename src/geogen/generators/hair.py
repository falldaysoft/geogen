"""Sculpted hair: a style declared as a signed-distance solid round the sculpted head.

A ``sculpt:`` hair style (``body.hair_styles.<name>.sculpt``) is, in the head's frame
(``generators/head.py``: x across, y up from the eye line, z forward, head-size units):

- ``cover``: hair grown out from the scalp above a hairline. The hairline's height is given at
  the ``front``, ``temple``, ``ear`` and ``back`` (``nape``) of the head and interpolated round it;
  ``thickness`` is [crown, sides, back]. Over ``taper`` the hair thins to nothing and dives
  ``dive`` under the skin, so the hairline is where hair and skin cross: no helmet rim.
- ``shapes``: ellipsoids (``{ellipsoid: [cx, cy, cz, rx, ry, rz]}``) and tapered capsules
  (``{capsule: [ax, ay, az, bx, by, bz, r0, r1]}``) smooth-unioned on over ``blend``: a fringe,
  the volume of a bob, a fall of long hair, a ponytail, a bun.
- ``cuts``: the same shapes carved out (``cut_blend``): keep the face clear.
- ``ridges``: ``[count, depth]``: clumps running down from the crown.

It is polygonised like the head (surface nets, simplify, relax, the SDF's normals), skinned to
Head (``weights: back`` hands what hangs below the neck to UpperChest), coloured darker toward
the scalp, and UV'd with strands running down (``u`` round the head, ``v`` down, metric).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.mesh import Mesh
from .head import HeadShape, _capsule, _ellipsoid, _smax, _smin, build_head, sdf_normals

SCULPT_KEYS = {"cover", "shapes", "cuts", "blend", "cut_blend", "ridges", "weights", "reach"}
COVER_KEYS = {"front", "temple", "ear", "back", "thickness", "taper", "dive"}
ROOT_SHADE = 0.72              # hair colour at the scalp (x the colour), full at the surface
TOLERANCE = 0.0035             # simplified to within this of the sculpt (the face gets 2 mm)


def _shape(spec: dict, p: NDArray) -> NDArray:
    if "ellipsoid" in spec:
        cx, cy, cz, rx, ry, rz = (float(v) for v in spec["ellipsoid"])
        return _ellipsoid(p, [cx, cy, cz], [rx, ry, rz])
    if "capsule" in spec:
        ax, ay, az, bx, by, bz, r0, r1 = (float(v) for v in spec["capsule"])
        return _capsule(p, [ax, ay, az], [bx, by, bz], r0, r1)
    raise ValueError(f"hair sculpt: a shape is {{ellipsoid: [...]}} or {{capsule: [...]}}, not {sorted(spec)}")


class HairShape:
    """The signed-distance hair for one style on one sculpted head (head frame, metres)."""

    def __init__(self, head: HeadShape, spec: dict) -> None:
        unknown = set(spec) - SCULPT_KEYS
        if unknown:
            raise ValueError(f"hair sculpt: unknown keys {sorted(unknown)} (known: {sorted(SCULPT_KEYS)})")
        self.head, self.spec, self.s = head, spec, head.s
        cover = spec.get("cover")
        if cover is not None and set(cover) - COVER_KEYS:
            raise ValueError(f"hair sculpt cover: unknown keys {sorted(set(cover) - COVER_KEYS)}")
        self.cover = cover

    def _hairline(self, phi: NDArray) -> NDArray:
        """Hairline height round the head: front (phi 0), temple (0.9 rad), above the ear (pi/2),
        then down behind it to the nape (2.3 rad), level across the back (pi)."""
        c = self.cover
        keys = np.array([0.0, 0.9, np.pi / 2, 2.3, np.pi])
        back = float(c.get("back", -0.09))
        heights = np.array([float(c.get("front", 0.075)), float(c.get("temple", 0.045)), float(c.get("ear", 0.02)),
                            back, back])
        a = np.abs(phi)
        i = np.clip(np.searchsorted(keys, a) - 1, 0, len(keys) - 2)
        f = (a - keys[i]) / (keys[i + 1] - keys[i])
        f = f * f * (3 - 2 * f)
        return heights[i] + (heights[i + 1] - heights[i]) * f

    def __call__(self, points: NDArray) -> NDArray:
        points = np.asarray(points, dtype=np.float64)
        p = points.reshape(-1, 3) / self.s
        head = self.head(points).reshape(-1) / self.s
        d = np.full(len(p), 1.0)
        phi = np.arctan2(p[:, 0], p[:, 2])
        if self.cover is not None:
            c = self.cover
            crown, sides, back = (float(v) for v in c.get("thickness", [0.015, 0.008, 0.01]))
            # Thickness: crown on top, sides and back below it, interpolated by height and azimuth.
            up = np.clip((p[:, 1] - 0.02) / 0.08, 0, 1)
            behind = np.clip(-np.cos(phi), 0, 1)
            low = sides + (back - sides) * behind
            thick = low + (crown - low) * up
            line = self._hairline(phi)
            taper, dive = float(c.get("taper", 0.025)), float(c.get("dive", 0.005))
            # Over the taper the hair thins to nothing and then dives under the skin, so the hairline
            # is where the two surfaces cross at an angle: no film edge, nothing coplanar.
            reach = np.clip((p[:, 1] - line) / taper, -0.5, 1)
            reach = (reach + 0.5) / 1.5
            reach = reach * reach * (3 - 2 * reach)
            grown = head - (-dive + (thick + dive) * reach)
            below = line - 0.5 * taper - 0.01 - p[:, 1]                # clipped deep under the skin
            d = np.maximum(grown, below)
        blend = float(self.spec.get("blend", 0.02))
        for shape in self.spec.get("shapes") or []:
            d = _smin(d, _shape(shape, p), blend)
        cut_blend = float(self.spec.get("cut_blend", 0.01))
        for shape in self.spec.get("cuts") or []:
            d = _smax(d, -_shape(shape, p), cut_blend)
        ridges = self.spec.get("ridges")
        if ridges:
            count, depth = int(ridges[0]), float(ridges[1])
            wobble = np.sin(count * phi + 2.0 * np.sin(3 * phi) + 6.0 * p[:, 1])
            d = d + depth * wobble * np.clip((head - 0.004) / 0.006, 0, 1)    # not where the hair is a film
        return (d * self.s).reshape(points.shape[:-1])

    def bounds(self) -> tuple[NDArray, NDArray]:
        """A box (head frame, metres) that holds the hair."""
        lo, hi = np.array([-0.16, -0.16, -0.17]), np.array([0.16, 0.2, 0.17])
        for shape in self.spec.get("shapes") or []:
            if "ellipsoid" in shape:
                c, r = np.array(shape["ellipsoid"][:3], float), np.array(shape["ellipsoid"][3:], float)
                lo, hi = np.minimum(lo, c - r - 0.02), np.maximum(hi, c + r + 0.02)
            else:
                v = np.array(shape["capsule"], float)
                r = max(v[6], v[7]) + 0.02
                lo = np.minimum(lo, np.minimum(v[:3], v[3:6]) - r)
                hi = np.maximum(hi, np.maximum(v[:3], v[3:6]) + r)
        return lo * self.s, hi * self.s


def build_sculpted_hair(head: HeadShape, origin, spec: dict, skeleton, color) -> Mesh:
    """A closed, skinned, coloured, UV'd hair mesh for the ``sculpt`` spec on ``head``."""
    from ..core.skin import normalize_weights

    hair = HairShape(head, spec)
    lo, hi = hair.bounds()
    mesh = build_head(hair, origin, bounds=(lo, hi), tolerance=TOLERANCE)
    local = mesh.vertices - np.asarray(origin, dtype=np.float64)
    mesh.normals = sdf_normals(hair, local)
    # Darker toward the scalp: how far out from the skin, against the style's thickness (a buzz cut
    # is only a few mm, so it is measured against its own).
    out = head(local)
    depth = np.clip(out / max(min(0.012 * head.s, 0.8 * float(out.max())), 1e-4), 0, 1)
    shade = ROOT_SHADE + (1 - ROOT_SHADE) * depth
    mesh.colors = np.column_stack([np.outer(shade, np.asarray(color[:3], dtype=np.float64)), np.ones(len(local))])
    # Strands run down: u round the head, v down it (metric).
    phi = np.arctan2(local[:, 0], local[:, 2])
    mesh.uvs = np.column_stack([phi * 0.1 * head.s, -local[:, 1]])
    # Skinned to Head; a fall of long hair below the neck goes with UpperChest.
    names = skeleton.names
    dense = np.zeros((len(local), len(names)))
    share = np.ones(len(local))
    if spec.get("weights", "head") == "back":
        neck_y = float(skeleton["Neck"].head[1]) - float(origin[1])
        share = np.clip((local[:, 1] - (neck_y - 0.06)) / 0.08, 0, 1)
        share = share * share * (3 - 2 * share)
        dense[:, names.index("UpperChest")] = 1 - share
    dense[:, names.index("Head")] = share
    mesh.joints, mesh.weights = normalize_weights(np.tile(np.arange(len(names)), (len(local), 1)), dense)
    return mesh
