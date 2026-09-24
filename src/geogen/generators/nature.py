"""Procedural trees (space colonisation) and rocks (displaced convex cages).

``primitive: tree`` grows a branch skeleton toward attraction points
scattered in a crown volume (Runions et al., "space colonisation"):

- ``style: deciduous`` fills an ellipsoid crown above a clear trunk;
  ``conifer`` a cone around a tall leader;
- branch radii follow the pipe model (``r^2.5`` of a parent = the sum of its
  children's), so limbs taper naturally from ``trunk_radius``;
- every branch chain is swept into a capped tube (bark mesh, metric UVs:
  u around the circumference, v along the branch); the trunk below the
  crown is the part's own mesh (hull collider), the rest a ``branches``
  child without a collider;
- leaf clusters (noise-displaced blobs, flattened for conifers) sit on the
  branch tips in a ``foliage`` child node with ``foliage_material``.

The part's ``size`` is the tree's bounding box (width, height, depth);
``seed`` makes each tree different. ``primitive: rock`` builds a random
convex cage (``seed``), subdivides and displaces it: boulders and scree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from .base import MeshGenerator

MAX_CLUSTERS = 130      # leaf clusters per tree


@dataclass
class TreeGenerator(MeshGenerator):
    width: float = 4.0
    height: float = 5.0
    depth: float = 4.0
    style: str = "deciduous"          # deciduous | conifer
    seed: int = 0
    trunk_height: float = 0.35        # clear trunk, fraction of height (deciduous)
    trunk_radius: float = 0.16
    attractors: int = 260
    step: float = 0.35                # branch segment length (m)
    leaf_size: float = 0.55           # leaf cluster radius (m)
    segments: int = 7                 # sides of branch tubes
    foliage_material: str = ""            # default: foliage_green (deciduous) / foliage_pine (conifer)

    # --- skeleton -----------------------------------------------------------------------------

    def _attraction_points(self, rng: np.random.Generator) -> np.ndarray:
        rx, rz = self.width / 2, self.depth / 2
        pts = []
        while len(pts) < self.attractors:
            p = rng.uniform(-1, 1, 3)
            if self.style == "conifer":
                # Cone: radius shrinks linearly to the top; crown starts low.
                y = rng.uniform(0.12, 1.0)
                r = (1.0 - y) * 1.05
                if p[0] ** 2 + p[2] ** 2 <= r * r:
                    pts.append([p[0] * rx, y * self.height, p[2] * rz])
            else:
                if (p ** 2).sum() <= 1.0:
                    base = self.trunk_height * self.height
                    ry = (self.height - base) / 2
                    pts.append([p[0] * rx, base + ry + p[1] * ry, p[2] * rz])
        return np.array(pts)

    def skeleton(self) -> tuple[np.ndarray, np.ndarray]:
        """Branch nodes (positions) and their parent indices (-1 for the root)."""
        rng = np.random.default_rng(self.seed)
        attract = self._attraction_points(rng)
        influence = max(self.width, self.depth) * 0.45
        kill = self.step * 1.6
        nodes = [np.zeros(3)]
        parents = [-1]
        # Trunk: straight up into the crown (conifers: a leader to near the top).
        top = (self.trunk_height if self.style != "conifer" else 0.1) * self.height
        while nodes[-1][1] < top:
            nodes.append(nodes[-1] + np.array([0.0, self.step, 0.0]))
            parents.append(len(nodes) - 2)
        if self.style == "conifer":
            leader = len(nodes) - 1
            while nodes[leader][1] < self.height * 0.95:
                nodes.append(nodes[leader] + np.array([0.0, self.step, 0.0]))
                parents.append(leader)
                leader = len(nodes) - 1
        alive = np.ones(len(attract), dtype=bool)
        for _ in range(200):
            pos = np.array(nodes)
            live = attract[alive]
            if not len(live):
                break
            d = np.linalg.norm(live[:, None, :] - pos[None, :, :], axis=2)
            nearest = d.argmin(axis=1)
            close = d[np.arange(len(live)), nearest] <= influence
            if not close.any():
                break
            grow: dict[int, np.ndarray] = {}
            for a, n in zip(np.flatnonzero(close), nearest[close]):
                v = live[a] - pos[n]
                grow[n] = grow.get(n, np.zeros(3)) + v / max(np.linalg.norm(v), 1e-9)
            added = 0
            for n, direction in grow.items():
                direction = direction / max(np.linalg.norm(direction), 1e-9)
                if self.style == "conifer":
                    direction = direction + np.array([0.0, -0.35, 0.0])   # drooping boughs
                    direction /= np.linalg.norm(direction)
                new = pos[n] + direction * self.step
                if np.min(np.linalg.norm(pos - new, axis=1)) < self.step * 0.5:
                    continue
                nodes.append(new)
                parents.append(int(n))
                added += 1
            pos = np.array(nodes)
            dist = np.linalg.norm(attract[:, None, :] - pos[None, :, :], axis=2).min(axis=1)
            alive &= dist > kill
            if not added:
                break
        return np.array(nodes), np.array(parents)

    def radii(self, parents: np.ndarray) -> np.ndarray:
        """Pipe-model radii: tips thin, each parent carries its children's cross sections."""
        n = len(parents)
        children: list[list[int]] = [[] for _ in range(n)]
        for i, p in enumerate(parents):
            if p >= 0:
                children[p].append(i)
        weight = np.zeros(n)
        tip = 1.0
        for i in range(n - 1, -1, -1):     # children always come after their parent
            weight[i] = tip if not children[i] else sum(weight[c] for c in children[i])
        exponent = 2.5
        r = weight ** (1 / exponent)
        return r * (self.trunk_radius / r[0])

    # --- meshes -------------------------------------------------------------------------------

    def _chains(self, parents: np.ndarray, radii: np.ndarray) -> list[list[int]]:
        """Split the skeleton into chains: each follows its thickest child; others start new chains."""
        n = len(parents)
        children: list[list[int]] = [[] for _ in range(n)]
        for i, p in enumerate(parents):
            if p >= 0:
                children[p].append(i)
        chains, stack = [], [(0, None)]
        while stack:
            start, parent = stack.pop()
            chain = ([parent] if parent is not None else []) + [start]
            node = start
            while children[node]:
                kids = sorted(children[node], key=lambda c: -radii[c])
                for other in kids[1:]:
                    stack.append((other, node))
                node = kids[0]
                chain.append(node)
            chains.append(chain)
        return chains

    def _tube(self, pts: np.ndarray, radii: np.ndarray) -> Mesh:
        """Capped tube through ``pts`` with per-point radii (parallel-transport frames)."""
        k = self.segments
        tangents = np.gradient(pts, axis=0)
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
        ref = np.array([1.0, 0.0, 0.0]) if abs(tangents[0][0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        normal = np.cross(tangents[0], ref)
        normal /= np.linalg.norm(normal)
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        rings, length = [], 0.0
        uvs = []
        for i, (p, t, r) in enumerate(zip(pts, tangents, radii)):
            if i > 0:
                # Transport the normal along the curve.
                normal = normal - t * float(normal @ t)
                normal /= max(np.linalg.norm(normal), 1e-9)
                length += float(np.linalg.norm(pts[i] - pts[i - 1]))
            binormal = np.cross(t, normal)
            ring = p + r * (np.cos(angles)[:, None] * normal + np.sin(angles)[:, None] * binormal)
            rings.append(ring)
            uvs.append(np.column_stack([angles * r, np.full(k, length)]))
        verts = np.vstack(rings)
        uv = np.vstack(uvs)
        faces = []
        for i in range(len(pts) - 1):
            a, b = i * k, (i + 1) * k
            for j in range(k):
                j2 = (j + 1) % k
                faces += [[a + j, a + j2, b + j2], [a + j, b + j2, b + j]]
        # Caps: fans around the end centres.
        start_c, end_c = len(verts), len(verts) + 1
        verts = np.vstack([verts, pts[0], pts[-1]])
        uv = np.vstack([uv, [0.0, 0.0], [0.0, length]])
        last = (len(pts) - 1) * k
        for j in range(k):
            j2 = (j + 1) % k
            faces.append([start_c, j2, j])
            faces.append([end_c, last + j, last + j2])
        return Mesh(vertices=verts, faces=np.array(faces, dtype=np.int64), uvs=uv)

    def generate(self) -> Mesh:
        trunk, branches, *_ = self._bark()
        return Mesh.merge([trunk, branches]) if branches is not None else trunk

    def _bark(self) -> tuple[Mesh, Mesh | None, np.ndarray, np.ndarray, np.ndarray]:
        """(trunk up to the crown, the other branches, skeleton nodes, parents, radii)."""
        from ..core import meshops

        nodes, parents = self.skeleton()
        radii = self.radii(parents)
        chains = self._chains(parents, radii)
        # The first chain runs from the root up the thickest limbs; the trunk is its
        # part below the crown (conifers: the whole leader), the rest joins the branches.
        first = chains[0]
        crown = (self.trunk_height if self.style != "conifer" else 1.0) * self.height + self.step
        cut = next((i for i, n in enumerate(first) if nodes[n][1] > crown), len(first))
        cut = max(cut, 2)
        trunk = self._tube(nodes[first[:cut]], radii[first[:cut]])
        pieces = [first[cut - 1:]] if cut < len(first) else []
        tubes = []
        for chain in pieces + chains[1:]:
            if len(chain) < 2:
                continue
            r = radii[chain].copy()
            r[0] = min(r[0], r[1] * 1.15)      # a branch starts inside its parent, not wider than itself
            tubes.append(self._tube(nodes[chain], r))
        branches = meshops.compute_normals(Mesh.merge(tubes), 60.0) if tubes else None
        return meshops.compute_normals(trunk, 60.0), branches, nodes, parents, radii

    def _foliage(self, nodes: np.ndarray, parents: np.ndarray, radii: np.ndarray) -> Mesh | None:
        from ..core import meshops, uvmap
        from ..core.subdiv import displace
        from .primitives import SphereGenerator

        # Leaves on every thin twig (not just the tips) fill the crown out.
        twig = radii.min() * (2.4 if self.style == "conifer" else 1.9)
        tips = np.flatnonzero((radii <= twig) & (nodes[:, 1] > self.height * 0.15))
        if not len(tips):
            return None
        rng = np.random.default_rng(self.seed + 1)
        if len(tips) > MAX_CLUSTERS:
            tips = np.sort(rng.choice(tips, MAX_CLUSTERS, replace=False))
        base = SphereGenerator(radius=1.0, segments=10, rings=6).generate()
        blobs = []
        for tip in tips:
            s = self.leaf_size * rng.uniform(0.75, 1.2)
            scale = np.array([s, s * (0.45 if self.style == "conifer" else 0.8), s])
            m = np.diag([*scale, 1.0])
            m[:3, 3] = nodes[tip]
            blob = base.transform(m)
            blobs.append(displace(blob, s * 0.18, s * 0.9, 2, int(rng.integers(1 << 30))))
        leaves = Mesh.merge(blobs)
        return meshops.compute_normals(uvmap.box_project(leaves), 75.0)

    def to_node(self, name: str | None = None) -> SceneNode:
        from ..materials.loader import MaterialLoader

        trunk, branches, nodes, parents, radii = self._bark()
        # Centre on the trunk base (the skeleton grows from the origin): the part frame
        # puts the bounding box's centre at the origin, so shift down by half the height.
        shift = np.eye(4)
        shift[1, 3] = -self.height / 2 - 0.02    # trunk base 2 cm into the ground (no coplanar base cap)
        node = SceneNode(name=name or "tree", mesh=trunk.transform(shift), tags=["vegetation.trunk"])
        node.meta["tree"] = {"style": self.style, "seed": self.seed, "branches": int(len(nodes))}
        node.meta["collider"] = "hull"        # the trunk; branches and leaves don't block
        if branches is not None:
            limbs = SceneNode(name="branches", mesh=branches.transform(shift), tags=["vegetation.branches"])
            limbs.meta["collider"] = "none"
            node.add_child(limbs)
        foliage = self._foliage(nodes, parents, radii)
        if foliage is not None:
            foliage = foliage.transform(shift)
            name_ = self.foliage_material or ("foliage_pine" if self.style == "conifer" else "foliage_green")
            foliage.material = MaterialLoader().load(name_)
            leaves = SceneNode(name="foliage", mesh=foliage, tags=["vegetation.foliage"])
            leaves.meta["collider"] = "none"
            node.add_child(leaves)
        return node

    def get_attachment_points(self, size: np.ndarray):
        from .primitives import CubeGenerator

        return CubeGenerator().get_attachment_points(size)

    def get_surfaces(self, size: np.ndarray):
        return {}


@dataclass
class RockGenerator(MeshGenerator):
    """Random convex cage (``points`` on a squashed sphere), subdivided and displaced."""

    width: float = 1.0
    height: float = 0.7
    depth: float = 0.9
    seed: int = 0
    points: int = 14
    levels: int = 3
    roughness: float = 0.12           # displacement amplitude, fraction of the smallest size
    crease: float | None = 50.0       # keep cage edges sharper than this (None: smooth pebble)

    def generate(self) -> Mesh:
        from scipy.spatial import ConvexHull

        from ..core import meshops, uvmap
        from ..core.subdiv import displace, subdivide

        rng = np.random.default_rng(self.seed)
        pts = rng.normal(size=(self.points, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        pts *= rng.uniform(0.75, 1.0, (self.points, 1))
        hull = ConvexHull(pts)
        faces = hull.simplices.copy()
        # Orient faces outward.
        centre = pts.mean(axis=0)
        for i, f in enumerate(faces):
            n = np.cross(pts[f[1]] - pts[f[0]], pts[f[2]] - pts[f[0]])
            if n @ (pts[f[0]] - centre) < 0:
                faces[i] = f[[0, 2, 1]]
        used = np.unique(faces)
        remap = -np.ones(len(pts), dtype=np.int64)
        remap[used] = np.arange(len(used))
        cage = Mesh(vertices=pts[used], faces=remap[faces])
        mesh = subdivide(cage, self.levels, self.crease)
        # Fit the box, flatten the base a little so it sits on the ground.
        lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
        size = np.array([self.width, self.height, self.depth])
        v = (mesh.vertices - (lo + hi) / 2) / np.maximum(hi - lo, 1e-9) * size
        v[:, 1] = np.maximum(v[:, 1], -self.height / 2 + 0.08 * self.height)
        mesh = Mesh(vertices=v, faces=mesh.faces)
        amp = self.roughness * float(size.min())
        mesh = displace(mesh, amp, float(size.max()) * 0.5, 4, self.seed + 7)
        # Re-flatten the base after displacement so the stone sits on the ground.
        v = mesh.vertices.copy()
        floor = float(np.percentile(v[:, 1], 8))
        v[:, 1] = np.maximum(v[:, 1], floor)
        v[:, 1] -= (v[:, 1].min() + v[:, 1].max()) / 2      # keep it centred in the part box
        mesh = Mesh(vertices=v, faces=mesh.faces)
        return meshops.compute_normals(uvmap.box_project(mesh), 70.0)

    def get_attachment_points(self, size: np.ndarray):
        from .primitives import CubeGenerator

        return CubeGenerator().get_attachment_points(size)

    def get_surfaces(self, size: np.ndarray):
        return {}
