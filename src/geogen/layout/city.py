"""City layout: a street grid of raised blocks, lots, zoning and street furniture.

A scene with a ``city:`` block builds a town district::

    city:
      seed: 4
      blocks: [3, 2]              # blocks along x (east) and z (north)
      block_size: [44, 36]        # curb to curb, metres
      street_width: 8             # curb to curb
      avenues: { ew: [1] }        # east-west streets (constant z, index 0..nz) that are avenues;
                                  # ns: [...] for north-south streets (constant x, index 0..nx)
      avenue_width: 12
      sidewalk: 3
      curb_height: 0.15
      lot_width: [10, 16]         # frontage range for subdivided lots
      landmarks: { "1,0": { asset: hotel.yaml } }   # a whole block for one building
      parks: ["0,1"]
      buildings:                  # catalogue per zone ({asset|scene, params, weight, setback})
        residential: [{ scene: scenes/cottage.yaml }, { asset: house_simple.yaml }]
        commercial:  [{ asset: shop.yaml, setback: 0 }]
      furniture:                  # along the curbs; spacing in metres (0 disables)
        lamp: { asset: street_lamp.yaml, spacing: 22 }
        tree: { asset: maple_tree.yaml, spacing: 11 }
        bench: { asset: bench.yaml, per_edge: 1, zones: [commercial, park] }
        hydrant: { asset: fire_hydrant.yaml, per_edge: 1 }

Geometry: one asphalt slab under the whole district (top at y = 0) with
crosswalks and centre lines; each block is a platform ``curb_height`` tall:
a concrete sidewalk ring (rounded corners, the curb is its outer face) and
lot slabs (grass for residential and parks, paving for commercial and
landmarks). Streets and sidewalks are walkable.

Zoning: landmark blocks hold a single building; lots fronting an avenue are
commercial; the rest residential. Blocks face their lots north/south unless
only a north-south avenue borders them. Each lot takes a random catalogue
entry that fits (front = the building's +Z, turned to face the street;
``setback`` metres behind the sidewalk, default 3 m residential, 0
commercial). Lots that nothing fits stay empty. Every building type is
loaded once and instanced (meshes shared). Nodes carry ``meta.lot``
(block, zone, frontage) for downstream tools.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.profile import Shape, rect
from ..core.transform import Transform

KNOWN = {"seed", "blocks", "block_size", "street_width", "avenues", "avenue_width", "sidewalk", "curb_height",
         "lot_width", "landmarks", "parks", "buildings", "furniture", "corner_radius"}

_DEFAULT_SETBACK = {"residential": 3.0, "commercial": 0.0, "landmark": 2.0}
_LOT_MATERIAL = {"residential": "grass", "park": "grass", "commercial": "concrete", "landmark": "concrete"}
# Yaw that turns an asset's +Z front to face each street side.
_YAW = {"south": np.pi, "north": 0.0, "east": np.pi / 2, "west": -np.pi / 2}
_OUT = {"south": np.array([0.0, -1.0]), "north": np.array([0.0, 1.0]), "east": np.array([1.0, 0.0]),
        "west": np.array([-1.0, 0.0])}


@dataclass
class Lot:
    block: tuple[int, int]
    zone: str
    side: str                 # street side the lot fronts
    lo: np.ndarray            # (x, z) min corner
    hi: np.ndarray            # (x, z) max corner

    @property
    def frontage(self) -> float:
        return float((self.hi - self.lo)[0 if self.side in ("north", "south") else 1])

    @property
    def depth(self) -> float:
        return float((self.hi - self.lo)[1 if self.side in ("north", "south") else 0])


def _key(value) -> tuple[int, int]:
    if isinstance(value, str):
        a, b = value.split(",")
        return int(a), int(b)
    return int(value[0]), int(value[1])


def _instance(node: SceneNode) -> SceneNode:
    """Deep copy that shares meshes (and their materials) with the original."""
    memo = {id(n.mesh): n.mesh for n in node.iter_nodes() if n.mesh is not None}
    return copy.deepcopy(node, memo)


def _slab(lo, hi, y0: float, y1: float) -> Mesh:
    from ..generators.primitives import CubeGenerator

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    mesh = CubeGenerator(size_x=hi[0] - lo[0], size_y=y1 - y0, size_z=hi[1] - lo[1], bevel=0).generate()
    m = np.eye(4)
    m[:3, 3] = [(lo[0] + hi[0]) / 2, (y0 + y1) / 2, (lo[1] + hi[1]) / 2]
    return mesh.transform(m)


def _ring(lo, hi, width: float, radius: float, height: float) -> Mesh:
    """Sidewalk ring: rounded outer rect minus the inner lot area, top at ``height``."""
    from ..generators.profiles import ExtrudeGenerator

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    size, centre = hi - lo, (lo + hi) / 2
    # Axis "y" maps profile (x, y) to world (x, -z).
    outer = rect(size[0], size[1], radius, segments=6)
    inner = rect(size[0] - 2 * width, size[1] - 2 * width)
    mesh = ExtrudeGenerator(shape=Shape(outer, [inner]), depth=height, axis="y", crease_angle=30.0).generate()
    m = np.eye(4)
    m[:3, 3] = [centre[0], height / 2, centre[1]]
    return mesh.transform(m)


class CityBuilder:
    def __init__(self, spec: dict[str, Any], load: Callable[[dict[str, Any]], SceneNode], material_loader):
        unknown = set(spec) - KNOWN
        if unknown:
            raise ValueError(f"city: unknown keys {sorted(unknown)}. Known: {sorted(KNOWN)}")
        self.spec = spec
        self.load = load
        self.materials = material_loader
        self.rng = np.random.default_rng(int(spec.get("seed", 0)))
        self.nx, self.nz = (int(v) for v in spec.get("blocks", [2, 2]))
        self.block = np.array(spec.get("block_size", [40.0, 32.0]), dtype=float)
        self.street = float(spec.get("street_width", 8.0))
        self.avenue = float(spec.get("avenue_width", 12.0))
        self.sidewalk = float(spec.get("sidewalk", 3.0))
        self.curb = float(spec.get("curb_height", 0.15))
        self.radius = float(spec.get("corner_radius", 1.5))
        avenues = spec.get("avenues", {}) or {}
        self.ns_avenues = {int(i) for i in avenues.get("ns", [])}
        self.ew_avenues = {int(i) for i in avenues.get("ew", [])}
        self.landmarks = {_key(k): v for k, v in (spec.get("landmarks") or {}).items()}
        self.parks = {_key(k) for k in spec.get("parks", [])}
        self.catalogue = spec.get("buildings", {}) or {}
        self._prototypes: dict[int, tuple[SceneNode, np.ndarray, np.ndarray]] = {}

        # Street centre lines and widths along each axis.
        self.ns_width = [self.avenue if i in self.ns_avenues else self.street for i in range(self.nx + 1)]
        self.ew_width = [self.avenue if j in self.ew_avenues else self.street for j in range(self.nz + 1)]
        self.x_edges = self._edges(self.ns_width, self.block[0])   # block i spans x_edges[i]
        self.z_edges = self._edges(self.ew_width, self.block[1])
        self.extent = np.array([self.x_edges[-1][1] + self.ns_width[-1], self.z_edges[-1][1] + self.ew_width[-1]])
        shift = self.extent / 2
        self.x_edges = [(a - shift[0], b - shift[0]) for a, b in self.x_edges]
        self.z_edges = [(a - shift[1], b - shift[1]) for a, b in self.z_edges]
        self.lo = -shift
        self.hi = shift

    @staticmethod
    def _edges(widths: list[float], block: float) -> list[tuple[float, float]]:
        edges, cursor = [], widths[0]
        for w in widths[1:]:
            edges.append((cursor, cursor + block))
            cursor += block + w
        return edges

    # --- layout -----------------------------------------------------------------------------------

    def block_rect(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([self.x_edges[i][0], self.z_edges[j][0]]), np.array([self.x_edges[i][1], self.z_edges[j][1]]))

    def lots(self) -> list[Lot]:
        out = []
        for i in range(self.nx):
            for j in range(self.nz):
                lo, hi = self.block_rect(i, j)
                inner_lo, inner_hi = lo + self.sidewalk, hi - self.sidewalk
                if (i, j) in self.parks:
                    out.append(Lot((i, j), "park", "south", inner_lo, inner_hi))
                    continue
                sides = {"south": j in self.ew_avenues, "north": j + 1 in self.ew_avenues,
                         "west": i in self.ns_avenues, "east": i + 1 in self.ns_avenues}
                if (i, j) in self.landmarks:
                    side = next((s for s, avenue in sides.items() if avenue), "south")
                    out.append(Lot((i, j), "landmark", side, inner_lo, inner_hi))
                    continue
                ew = not (sides["west"] or sides["east"]) or sides["south"] or sides["north"]
                rows = ("south", "north") if ew else ("west", "east")
                along = 0 if ew else 1
                mid = (inner_lo[1 - along] + inner_hi[1 - along]) / 2
                for side in rows:
                    zone = "commercial" if sides[side] else "residential"
                    row_lo, row_hi = inner_lo.copy(), inner_hi.copy()
                    if side in ("south", "west"):
                        row_hi[1 - along] = mid
                    else:
                        row_lo[1 - along] = mid
                    for a, b in self._split(row_lo[along], row_hi[along]):
                        lot_lo, lot_hi = row_lo.copy(), row_hi.copy()
                        lot_lo[along], lot_hi[along] = a, b
                        out.append(Lot((i, j), zone, side, lot_lo, lot_hi))
        return out

    def _split(self, a: float, b: float) -> list[tuple[float, float]]:
        lo_w, hi_w = (float(v) for v in self.spec.get("lot_width", [10.0, 16.0]))
        cuts, cursor = [], a
        while b - cursor > hi_w:
            w = float(self.rng.uniform(lo_w, hi_w))
            if b - cursor - w < lo_w:      # don't leave a sliver
                w = (b - cursor) / 2
            cuts.append((cursor, cursor + w))
            cursor += w
        cuts.append((cursor, b))
        return cuts

    # --- geometry ---------------------------------------------------------------------------------

    def build(self, root: SceneNode) -> list[Lot]:
        city = SceneNode(name="city", tags=["city"])
        city.meta["city"] = {"blocks": [self.nx, self.nz], "extent": [round(float(v), 3) for v in self.extent]}
        streets = SceneNode(name="streets", mesh=self._material(_slab(self.lo, self.hi, -0.2, 0.0), "asphalt"),
                            tags=["street.road"])
        streets.meta["walkable"] = True
        city.add_child(streets)
        markings = self._markings()
        for material, meshes in markings.items():
            if meshes:
                node = SceneNode(name=f"markings_{material}", mesh=self._material(Mesh.merge(meshes), material),
                                 tags=["street.marking"])
                node.meta["collider"] = "none"
                city.add_child(node)

        lots = self.lots()
        for i in range(self.nx):
            for j in range(self.nz):
                block = SceneNode(name=f"block_{i}_{j}", tags=["city.block"])
                block.meta["block"] = {"index": [i, j]}
                lo, hi = self.block_rect(i, j)
                walk = SceneNode(name=f"sidewalk_{i}_{j}",
                                 mesh=self._material(_ring(lo, hi, self.sidewalk, self.radius, self.curb), "concrete"),
                                 tags=["street.sidewalk"])
                walk.meta["walkable"] = True
                block.add_child(walk)
                for k, lot in enumerate(lt for lt in lots if lt.block == (i, j)):
                    block.add_child(self._lot_node(lot, k))
                city.add_child(block)
        self._furnish_streets(city, lots)
        root.add_child(city)
        return lots

    def _material(self, mesh: Mesh, name: str) -> Mesh:
        from ..core import meshops, uvmap

        mesh = meshops.compute_normals(uvmap.box_project(mesh), 30.0)
        mesh.material = self.materials.load(name)
        return mesh

    def _lot_node(self, lot: Lot, k: int) -> SceneNode:
        i, j = lot.block
        node = SceneNode(name=f"lot_{i}_{j}_{k}", tags=["city.lot", f"zone.{lot.zone}"])
        node.meta["lot"] = {"block": [i, j], "zone": lot.zone, "side": lot.side,
                            "rect": [round(float(v), 3) for v in (*lot.lo, *lot.hi)],
                            "frontage": round(lot.frontage, 3), "depth": round(lot.depth, 3)}
        ground = SceneNode(name=f"lot_{i}_{j}_{k}_ground",
                           mesh=self._material(_slab(lot.lo, lot.hi, 0.0, self.curb), _LOT_MATERIAL[lot.zone]))
        ground.meta["walkable"] = True
        node.add_child(ground)
        if lot.zone == "park":
            self._park(node, lot)
            return node
        entries = [self.landmarks[lot.block]] if lot.zone == "landmark" else list(self.catalogue.get(lot.zone, []))
        building = self._fit(entries, lot)
        if building is not None:
            node.add_child(building)
        return node

    def _prototype(self, entry: dict[str, Any]) -> tuple[SceneNode, np.ndarray, np.ndarray]:
        key = id(entry)
        if key not in self._prototypes:
            node = self.load({k: v for k, v in entry.items() if k in ("asset", "scene", "params", "furnish")})
            pts = []
            for n in node.iter_nodes():
                if n.mesh is not None and len(n.mesh.vertices) and n.meta.get("type") != "room_volume":
                    v = n.mesh.vertices
                    pts.append((n.world_transform() @ np.c_[v, np.ones(len(v))].T).T[:, :3])
            allp = np.vstack(pts) if pts else np.zeros((1, 3))
            self._prototypes[key] = (node, allp.min(axis=0), allp.max(axis=0))
        return self._prototypes[key]

    def _fit(self, entries: list[dict[str, Any]], lot: Lot) -> SceneNode | None:
        """Place a random catalogue entry that fits the lot (front to the street)."""
        order = list(self.rng.permutation(len(entries))) if entries else []
        weights = np.array([float(entries[k].get("weight", 1.0)) for k in order]) if order else None
        if order and weights is not None and weights.sum() > 0:
            order = [order[k] for k in self.rng.choice(len(order), size=len(order), replace=False,
                                                         p=weights / weights.sum())]
        for k in order:
            entry = entries[k]
            proto, bmin, bmax = self._prototype(entry)
            width, depth = bmax[0] - bmin[0], bmax[2] - bmin[2]
            setback = float(entry.get("setback", _DEFAULT_SETBACK.get(lot.zone, 2.0)))
            side_gap = 0.0 if lot.zone == "commercial" else 1.0
            if width > lot.frontage - 2 * side_gap + 1e-6 or depth > lot.depth - setback + 1e-6:
                continue
            node = _instance(proto)
            node.name = f"building_{entry.get('asset', entry.get('scene', 'x')).split('/')[-1].split('.')[0]}"
            yaw = _YAW[lot.side]
            out = _OUT[lot.side]
            centre = (lot.lo + lot.hi) / 2
            # Street edge of the lot, then back by the setback to the building's front face.
            edge = centre + out * (lot.depth / 2)
            front = edge - out * setback
            # In the building frame the front face is at z = bmax[2], centred on x.
            local = np.array([(bmin[0] + bmax[0]) / 2, 0.0, bmax[2]])
            c, s = np.cos(yaw), np.sin(yaw)
            rotated = np.array([c * local[0] + s * local[2], 0.0, -s * local[0] + c * local[2]])
            node.transform = Transform(translation=np.array([front[0], self.curb, front[1]]) - rotated,
                                       rotation=np.array([0.0, yaw, 0.0]))
            node.meta["building"] = {"entry": entry.get("asset", entry.get("scene")), "zone": lot.zone}
            return node
        return None

    def _park(self, node: SceneNode, lot: Lot) -> None:
        from .scatter import poisson_disk

        entry = (self.spec.get("furniture") or {}).get("tree")
        if not entry:
            return
        proto, _, _ = self._prototype(entry)
        for k, p in enumerate(poisson_disk(self.rng, lot.lo + 2.0, lot.hi - 2.0, 7.0)):
            tree = _instance(proto)
            tree.name = f"park_tree_{k + 1}"
            tree.transform = Transform(translation=np.array([p[0], self.curb, p[1]]),
                                       rotation=np.array([0.0, float(self.rng.uniform(0, 2 * np.pi)), 0.0]))
            node.add_child(tree)

    def _markings(self) -> dict[str, list[Mesh]]:
        """Crosswalk stripes at every block corner and dashed centre lines between them."""
        white, yellow = [], []
        y0, y1 = 0.0, 0.004
        stripe, gap = 0.5, 0.6
        # North-south streets (constant x): x span between blocks.
        ns_spans = self._street_spans(self.x_edges, self.ns_width, self.lo[0], self.hi[0])
        ew_spans = self._street_spans(self.z_edges, self.ew_width, self.lo[1], self.hi[1])
        for (a, b) in ns_spans:
            for (z0, z1) in self.z_edges:
                for zc0, zc1 in ((z0, z0 + self.sidewalk), (z1 - self.sidewalk, z1)):
                    for x in np.arange(a + 0.5, b - 0.5 - stripe + 1e-6, stripe + gap):
                        white.append(_slab([x, zc0 + 0.3], [x + stripe, zc1 - 0.3], y0, y1))
                # Centre line between this block's crosswalks.
                yellow += self._dashes(np.array([(a + b) / 2, z0 + self.sidewalk + 1.0]),
                                       np.array([(a + b) / 2, z1 - self.sidewalk - 1.0]), y0, y1)
        for (a, b) in ew_spans:
            for (x0, x1) in self.x_edges:
                for xc0, xc1 in ((x0, x0 + self.sidewalk), (x1 - self.sidewalk, x1)):
                    for z in np.arange(a + 0.5, b - 0.5 - stripe + 1e-6, stripe + gap):
                        white.append(_slab([xc0 + 0.3, z], [xc1 - 0.3, z + stripe], y0, y1))
                yellow += self._dashes(np.array([x0 + self.sidewalk + 1.0, (a + b) / 2]),
                                       np.array([x1 - self.sidewalk - 1.0, (a + b) / 2]), y0, y1)
        return {"road_paint_white": white, "road_paint_yellow": yellow}

    @staticmethod
    def _street_spans(edges, widths, lo: float, hi: float) -> list[tuple[float, float]]:
        spans = [(lo, edges[0][0])]
        spans += [(edges[k][1], edges[k + 1][0]) for k in range(len(edges) - 1)]
        spans.append((edges[-1][1], hi))
        return spans

    @staticmethod
    def _dashes(p0: np.ndarray, p1: np.ndarray, y0: float, y1: float, dash: float = 3.0, gap: float = 3.0,
                width: float = 0.12) -> list[Mesh]:
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length < dash:
            return []
        u = d / length
        out = []
        for t in np.arange(0.0, length - dash + 1e-6, dash + gap):
            a, b = p0 + u * t, p0 + u * (t + dash)
            lo = np.minimum(a, b) - (1 - np.abs(u)) * width / 2
            hi = np.maximum(a, b) + (1 - np.abs(u)) * width / 2
            out.append(_slab(lo, hi, y0, y1))
        return out

    # --- street furniture ------------------------------------------------------------------------

    def _furnish_streets(self, city: SceneNode, lots: list[Lot]) -> None:
        spec = self.spec.get("furniture") or {}
        if not spec:
            return
        group = SceneNode(name="street_furniture", tags=["street.furniture"])
        taken: list[np.ndarray] = []
        inset = 0.7          # from the curb face
        corner_clear = self.sidewalk + 2.5
        for i in range(self.nx):
            for j in range(self.nz):
                lo, hi = self.block_rect(i, j)
                block_lots = [lt for lt in lots if lt.block == (i, j)]
                edges = {"south": (np.array([lo[0], lo[1] + inset]), np.array([hi[0], lo[1] + inset])),
                         "north": (np.array([lo[0], hi[1] - inset]), np.array([hi[0], hi[1] - inset])),
                         "west": (np.array([lo[0] + inset, lo[1]]), np.array([lo[0] + inset, hi[1]])),
                         "east": (np.array([hi[0] - inset, lo[1]]), np.array([hi[0] - inset, hi[1]]))}
                for side, (a, b) in edges.items():
                    zones = {lt.zone for lt in block_lots if lt.side == side} or {lt.zone for lt in block_lots}
                    length = float(np.linalg.norm(b - a))
                    u = (b - a) / length
                    usable = (corner_clear, length - corner_clear)
                    if usable[1] <= usable[0]:
                        continue
                    for kind in ("lamp", "tree"):
                        entry = spec.get(kind)
                        if not entry or not float(entry.get("spacing", 0)):
                            continue
                        step = float(entry["spacing"])
                        start = usable[0] + (step / 2 if kind == "tree" else 0.0)
                        for t in np.arange(start, usable[1] + 1e-6, step):
                            self._put(group, entry, kind, a + u * t, side, taken)
                    for kind in ("bench", "hydrant", "trashcan"):
                        entry = spec.get(kind)
                        if not entry:
                            continue
                        if entry.get("zones") and not zones & set(entry["zones"]):
                            continue
                        for _ in range(int(entry.get("per_edge", 1))):
                            for _attempt in range(10):
                                p = a + u * float(self.rng.uniform(*usable))
                                if self._put(group, entry, kind, p, side, taken):
                                    break
        city.add_child(group)

    def _put(self, group: SceneNode, entry: dict[str, Any], kind: str, p: np.ndarray, side: str,
             taken: list[np.ndarray]) -> bool:
        clearance = 1.6 if kind in ("lamp", "tree") else 1.2
        if any(np.linalg.norm(p - q) < clearance for q in taken):
            return False
        proto, _, _ = self._prototype(entry)
        node = _instance(proto)
        node.name = f"{kind}_{len(group.children) + 1}"
        # Face the street: +Z out of the block.
        node.transform = Transform(translation=np.array([p[0], self.curb, p[1]]),
                                   rotation=np.array([0.0, _YAW[side], 0.0]))
        node.meta["street_furniture"] = kind
        group.add_child(node)
        taken.append(p)
        return True


def build_city(root: SceneNode, spec: dict[str, Any], load, material_loader) -> list[Lot]:
    """Add a ``city`` node to ``root`` (see module docstring); returns the lots."""
    builder = CityBuilder(spec, load, material_loader)
    root.size = np.array([builder.extent[0], 10.0, builder.extent[1]])
    return builder.build(root)
