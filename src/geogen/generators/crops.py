"""Procedural crops: a patch of plants filling the part box, at a growth stage.

``primitive: crop`` plants ``kind`` (wheat, carrot or pumpkin) in staggered rows across the
part's footprint, ``spacing`` apart, each a little different (``seed``). ``growth`` (0-1) is
how far they have got, so a plot's stages are one asset at several values:

- below ``SPROUT`` every kind is a pair of seedling leaves;
- wheat: tufts of stalks with leaf blades, ears from mid-growth that ripen from green to
  straw-gold and nod over;
- carrot: a rosette of narrow arching fronds; the orange shoulder of the root shows above
  the soil once it is grown;
- pumpkin: a vine crawling over the ground with broad cupped leaves; a ribbed fruit swells
  from mid-growth and turns from green to orange.

The part's height is the full-grown height; the plants stand on the box's bottom (a root
reaches a little below it, into the soil). Every piece is a closed solid (tubes with caps,
folded blades, a lathed fruit) with metric UVs. Crops are planted by the dozen and
instanced across a field, so the pieces are low-poly: a grown 1.1 m wheat patch is ~6.5k
triangles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from .base import MeshGenerator, box_attachment_points
from .nature import TreeGenerator

KINDS = ("wheat", "carrot", "pumpkin")
SPROUT = 0.25            # below this growth every kind is a seedling
RIPE = 0.8               # wheat and pumpkins change colour from here
SPACING = {"wheat": 0.2, "carrot": 0.2, "pumpkin": 0.5}
MATERIALS = {"leaf": "crop_leaf", "straw": "wheat_straw", "carrot": "carrot_root", "pumpkin": "pumpkin_skin"}
_TUBES = TreeGenerator(segments=4)


@dataclass
class CropGenerator(MeshGenerator):
    width: float = 1.0
    height: float = 0.9
    depth: float = 1.0
    kind: str = "wheat"
    growth: float = 1.0
    seed: int = 0
    spacing: float = 0.0          # between plants (0: the kind's own)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"crop kind is one of {', '.join(KINDS)}, got {self.kind!r}")
        if not 0.0 <= self.growth <= 1.0:
            raise ValueError(f"crop growth is 0-1, got {self.growth!r}")

    # --- layout -------------------------------------------------------------------------------

    def positions(self) -> np.ndarray:
        """Where each plant stands, (x, z): staggered rows filling the footprint, jittered."""
        rng = np.random.default_rng(self.seed)
        step = self.spacing or SPACING[self.kind]
        margin = step * 0.5
        xs = np.arange(-self.width / 2 + margin, self.width / 2 - margin + 1e-9, step)
        zs = np.arange(-self.depth / 2 + margin, self.depth / 2 - margin + 1e-9, step * 0.87)
        if not len(xs):
            xs = np.array([0.0])
        if not len(zs):
            zs = np.array([0.0])
        out = []
        for row, z in enumerate(zs):
            for x in xs + (step / 2 if row % 2 and len(xs) > 1 else 0.0):
                if abs(x) <= self.width / 2 - margin * 0.6:
                    out.append([x, z])
        pts = np.array(out) + rng.uniform(-0.15, 0.15, (len(out), 2)) * step
        return pts

    # --- geometry -----------------------------------------------------------------------------

    def pieces(self) -> dict[str, list[Mesh]]:
        """The plants' closed pieces by material role (leaf, straw, carrot, pumpkin), standing
        on y = 0."""
        rng = np.random.default_rng(self.seed + 1)
        out: dict[str, list[Mesh]] = {role: [] for role in MATERIALS}
        for x, z in self.positions():
            base = np.array([x, 0.0, z])
            turn = rng.uniform(0, 2 * np.pi)
            if self.growth < SPROUT:
                self._seedling(out, base, turn, rng)
            else:
                getattr(self, f"_{self.kind}")(out, base, turn, rng)
        return out

    def _seedling(self, out, base, turn, rng) -> None:
        size = 0.02 + 0.07 * self.growth / SPROUT
        for k in range(2):
            out["leaf"].append(_blade(_near(base, turn + np.pi * k), turn + np.pi * k, rise=np.radians(rng.uniform(35, 55)),
                                      droop=np.radians(40), length=size * rng.uniform(0.8, 1.1),
                                      width=size * 0.35, stations=4))

    def _wheat(self, out, base, turn, rng) -> None:
        g = (self.growth - SPROUT) / (1 - SPROUT)
        tall = self.height * (0.3 + 0.7 * g)
        ripe = self.growth >= RIPE
        role = "straw" if ripe else "leaf"
        for k in range(3):
            a = turn + 2 * np.pi * k / 3 + rng.uniform(-0.4, 0.4)
            lean = np.array([np.sin(a), 0.0, np.cos(a)]) * rng.uniform(0.04, 0.1) * tall
            h = tall * rng.uniform(0.78, 0.9)            # the ear on top reaches the full height
            nod = lean * (1.8 if ripe else 0.6)            # ripe ears weigh the tops over
            t = np.linspace(0, 1, 4)[:, None]
            pts = _near(base, a) + t * np.array([0.0, h, 0.0]) + t ** 2 * nod
            out[role].append(_TUBES._tube(pts, np.linspace(0.0045, 0.0025, 4), 3))
            if self.growth >= 0.5:                       # the ear, along the stalk's top
                tip = pts[-1]
                d = pts[-1] - pts[-2]
                d /= np.linalg.norm(d)
                length = 0.05 + 0.04 * g
                ear = tip - d * 0.01 + np.outer(np.linspace(0, 1, 4), d * length)    # starts inside the stalk
                out[role].append(_TUBES._tube(ear, np.array([0.006, 0.009, 0.008, 0.003]), 3))
        for k in range(2):
            out["leaf" if not ripe else "straw"].append(
                _blade(_near(base, turn + np.pi * k + 0.6) + np.array([0.0, tall * 0.15, 0.0]), turn + np.pi * k + 0.6,
                       rise=np.radians(55), droop=np.radians(70), length=tall * 0.4, width=0.009, stations=5))

    def _carrot(self, out, base, turn, rng) -> None:
        g = (self.growth - SPROUT) / (1 - SPROUT)
        tall = self.height * (0.35 + 0.65 * g)
        for k in range(6):
            a = turn + 2 * np.pi * k / 6 + rng.uniform(-0.3, 0.3)
            out["leaf"].append(_blade(_near(base, a), a, rise=np.radians(rng.uniform(50, 68)), droop=np.radians(75),
                                      length=tall * rng.uniform(0.85, 1.1), width=0.028, stations=4))
        if self.growth >= 0.6:                           # the shoulder above the soil, the root below
            r = 0.014 + 0.012 * g
            pts = base + np.outer([0.025, 0.005, -0.05, -0.1], [0.0, 1.0, 0.0])
            out["carrot"].append(_TUBES._tube(pts, np.array([r * 0.8, r, r * 0.7, r * 0.15]), 6))

    def _pumpkin(self, out, base, turn, rng) -> None:
        g = (self.growth - SPROUT) / (1 - SPROUT)
        reach = (self.spacing or SPACING["pumpkin"]) * (0.25 + 0.3 * g)
        # The vine: low over the ground, wandering away from the root.
        s = np.linspace(0, 1, 6)
        heading = turn + np.cumsum(rng.uniform(-0.5, 0.5, 6))
        steps = np.c_[np.sin(heading), np.zeros(6), np.cos(heading)] * (reach / 5)
        vine = base + np.vstack([np.zeros(3), np.cumsum(steps[:-1], axis=0)])
        vine[:, 1] = 0.012 + 0.01 * np.sin(s * np.pi)
        out["leaf"].append(_TUBES._tube(vine, np.full(6, 0.008), 5))
        leaves = 3 + int(4 * g)
        for k in range(leaves):
            p = vine[min(1 + k * 4 // max(leaves, 1), 5)]
            length = self.height * (0.4 + 0.4 * g) * rng.uniform(0.8, 1.0)
            out["leaf"].append(_blade(p, turn + 2.4 * k, rise=np.radians(45), droop=np.radians(60),
                                      length=length, width=length * 0.5, stations=4))
        if self.growth >= 0.45:
            radius = 0.04 + 0.12 * (self.growth - 0.45) / 0.55
            fruit = _pumpkin_fruit(radius, rng.integers(8, 11))
            spot = vine[-1]                              # at the vine's end, clear of the leaves
            m = np.eye(4)
            m[:3, 3] = [spot[0], 0.0, spot[2]]
            out["pumpkin" if self.growth >= RIPE else "leaf"].append(fruit.transform(m))

    # --- the node -----------------------------------------------------------------------------

    def generate(self) -> Mesh:
        from ..core import meshops, uvmap

        meshes = [m for role in self.pieces().values() for m in role]
        shift = np.eye(4)
        shift[1, 3] = -self.height / 2
        return meshops.compute_normals(uvmap.box_project(Mesh.merge(meshes)), 50.0).transform(shift)

    def to_node(self, name: str | None = None) -> SceneNode:
        from ..core import meshops, uvmap
        from ..materials.loader import MaterialLoader

        loader = MaterialLoader()
        shift = np.eye(4)
        shift[1, 3] = -self.height / 2          # the plants stand on the box's bottom
        groups = []
        for role, meshes in self.pieces().items():
            if meshes:
                mesh = meshops.compute_normals(uvmap.box_project(Mesh.merge(meshes)), 50.0)
                mesh.material = loader.load(MATERIALS[role])
                groups.append(mesh)
        node = SceneNode(name=name or "crop", mesh=Mesh.merge(groups).transform(shift), tags=["vegetation.crop"])
        node.meta["collider"] = "none"           # walk through it, aim past it
        return node

    def get_attachment_points(self, size: np.ndarray):
        return box_attachment_points(size)

    def get_surfaces(self, size: np.ndarray):
        return {}


def _near(base: np.ndarray, azimuth: float) -> np.ndarray:
    """A leaf's start, 3 mm out from the plant's middle, so leaves don't share their first point."""
    return base + np.array([np.sin(azimuth), 0.0, np.cos(azimuth)]) * 0.003


def _blade(start, azimuth, rise, droop, length, width, stations=5) -> Mesh:
    """A leaf: an arched, V-folded closed strip (the palm frond, scaled down)."""
    return TreeGenerator._frond(np.asarray(start, dtype=np.float64), azimuth, rise, droop, length, width, stations)


def _pumpkin_fruit(radius: float, ribs: int, rings: int = 7) -> Mesh:
    """A ribbed, squashed sphere sitting on y = 0 with a short stalk dimple on top: a closed
    lathe of ``ribs`` lobes."""
    sides = ribs * 3
    theta = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    lobes = 1.0 + 0.08 * np.cos(ribs * theta)
    phi = np.linspace(0.12, np.pi - 0.12, rings)
    verts = []
    for p in phi:
        r = np.sin(p) * radius * lobes
        y = (1 - np.cos(p)) * radius * 0.72
        verts += [[ri * np.cos(t), y, ri * np.sin(t)] for ri, t in zip(r, theta)]
    bottom = [0.0, (1 - np.cos(0.0)) * radius * 0.72 + radius * 0.01, 0.0]
    top = [0.0, (1 - np.cos(np.pi)) * radius * 0.72 - radius * 0.12, 0.0]     # the dimple
    verts += [bottom, top]
    faces = []
    for i in range(rings - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = a + sides, b + sides
            faces += [[a, b, c], [b, d, c]]
    lo, hi = len(verts) - 2, len(verts) - 1
    last = (rings - 1) * sides
    for j in range(sides):
        j2 = (j + 1) % sides
        faces += [[lo, j2, j], [hi, last + j, last + j2]]
    mesh = Mesh(np.array(verts), np.array(faces, dtype=np.int64))
    if mesh.to_trimesh().volume < 0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]])
    stalk = _TUBES._tube(np.array([[0.0, top[1], 0.0], [0.0, top[1] + radius * 0.25, 0.0],
                                   [radius * 0.08, top[1] + radius * 0.35, 0.0]]),
                         np.array([radius * 0.09, radius * 0.07, radius * 0.06]), 5)
    return Mesh.merge([mesh, stalk])
