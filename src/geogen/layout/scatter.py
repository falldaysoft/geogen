"""Seeded scatter placement for scenes.

A placement with ``scatter:`` places many copies of its asset::

    place:
      pines:
        asset: pine_tree.yaml
        params: { scale: { random: [0.8, 1.3] } }   # per-instance params
        scatter:
          seed: 7
          rect: [-20, -20, 20, 20]      # x0, z0, x1, z1 (scene frame), or:
          # path: [[x, z], ...]  +  spacing: 4  (+ jitter, offset: sideways metres)
          # on: house.floor               (a surface of a placed object)
          count: 30                     # at most this many
          spacing: 3                    # minimum distance between copies (Poisson disk)
          avoid: [house, road]          # keep off these placed objects' footprints
          margin: 0.5                   # extra clearance around avoided objects
          yaw: [0, 360]                 # random heading range (degrees); or a number
          scale: [0.9, 1.1]             # random uniform scale range; or a number
          radius: 1.0                   # each copy's keep-out radius against other scatters
                                        # (default: spacing / 2; 0.75 along paths)

Parameter values may be ``{random: [lo, hi]}`` (uniform float) or
``{choice: [a, b, ...]}``; they are drawn per copy. Everything is
deterministic for a seed. Copies are named ``<placement>_<n>``, carry
``meta.scatter`` and avoid each other and earlier scatters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.node import SceneNode
from ..core.transform import Transform

KNOWN = {"seed", "rect", "path", "on", "count", "spacing", "avoid", "margin", "yaw", "scale", "jitter", "offset",
         "radius"}


def resolve_random_params(params: dict[str, Any] | None, rng: np.random.Generator) -> dict[str, Any] | None:
    """Draw ``{random: [lo, hi]}`` / ``{choice: [...]}`` values."""
    if not params:
        return params
    out = {}
    for key, value in params.items():
        if isinstance(value, dict) and "random" in value:
            lo, hi = value["random"]
            out[key] = float(rng.uniform(float(lo), float(hi)))
        elif isinstance(value, dict) and "choice" in value:
            options = list(value["choice"])
            out[key] = options[int(rng.integers(len(options)))]
        else:
            out[key] = value
    return out


def poisson_disk(rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray, spacing: float,
                 accept=lambda p: True, limit: int = 10_000, tries: int = 30) -> list[np.ndarray]:
    """Bridson's Poisson-disk sampling in the rectangle [lo, hi] (2D)."""
    size = hi - lo
    if np.any(size <= 0):
        return []
    cell = spacing / np.sqrt(2)
    grid_shape = np.maximum(np.ceil(size / cell).astype(int), 1)
    grid = -np.ones(grid_shape, dtype=int)
    points: list[np.ndarray] = []
    active: list[int] = []

    def fits(p: np.ndarray) -> bool:
        if np.any(p < lo) or np.any(p > hi):
            return False
        g = ((p - lo) / cell).astype(int)
        g0, g1 = np.maximum(g - 2, 0), np.minimum(g + 3, grid_shape)
        for i in range(g0[0], g1[0]):
            for j in range(g0[1], g1[1]):
                k = grid[i, j]
                if k >= 0 and np.linalg.norm(points[k] - p) < spacing:
                    return False
        return True

    def add(p: np.ndarray) -> None:
        points.append(p)
        g = np.minimum(((p - lo) / cell).astype(int), grid_shape - 1)
        grid[g[0], g[1]] = len(points) - 1
        active.append(len(points) - 1)

    for _ in range(tries):     # seed point that the filter accepts
        p = lo + rng.random(2) * size
        if accept(p):
            add(p)
            break
    while active and len(points) < limit:
        idx = active[int(rng.integers(len(active)))]
        base = points[idx]
        for _ in range(tries):
            r = spacing * (1 + rng.random())
            a = rng.random() * 2 * np.pi
            p = base + r * np.array([np.cos(a), np.sin(a)])
            if fits(p) and accept(p):
                add(p)
                break
        else:
            active.remove(idx)
    return points


def _footprint(node: SceneNode, to_scene: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    pts = []
    for n in node.iter_nodes():
        if n.mesh is None or not len(n.mesh.vertices):
            continue
        v = n.mesh.vertices
        pts.append((to_scene @ n.world_transform() @ np.c_[v, np.ones(len(v))].T).T[:, [0, 2]])
    if not pts:
        return None
    allp = np.vstack(pts)
    return allp.min(axis=0), allp.max(axis=0)


def scatter(root: SceneNode, name: str, obj_def: dict[str, Any], loaded: dict[str, SceneNode], load,
            occupied: list[tuple[np.ndarray, float]]) -> list[SceneNode]:
    """Place copies for one scatter placement; returns the added nodes."""
    spec = obj_def["scatter"]
    unknown = set(spec) - KNOWN
    if unknown:
        raise ValueError(f"scatter '{name}': unknown keys {sorted(unknown)}. Known: {sorted(KNOWN)}")
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    spacing = float(spec.get("spacing", 2.0))
    count = int(spec.get("count", 1000))
    margin = float(spec.get("margin", 0.3))
    radius = float(spec.get("radius", 0.75 if "path" in spec else spacing / 2))
    to_scene = np.linalg.inv(root.world_transform())

    avoid_boxes = []
    for other in spec.get("avoid", []):
        if other not in loaded:
            raise ValueError(f"scatter '{name}': avoid refers to unknown object '{other}'")
        box = _footprint(loaded[other], to_scene)
        if box is not None:
            avoid_boxes.append((box[0] - margin, box[1] + margin))

    def clear(p: np.ndarray) -> bool:
        if any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in avoid_boxes):
            return False
        return all(np.linalg.norm(p - q) >= r + radius for q, r in occupied)

    # Candidate positions (scene frame x, z) plus surface frames where relevant.
    frames: list[tuple[np.ndarray, float, np.ndarray | None]] = []   # (xz, y, local normal or None)
    if "rect" in spec:
        x0, z0, x1, z1 = (float(v) for v in spec["rect"])
        for p in poisson_disk(rng, np.array([x0, z0]), np.array([x1, z1]), spacing, clear):
            frames.append((p, 0.0, None))
    elif "path" in spec:
        path = np.asarray(spec["path"], dtype=np.float64)
        step = float(spec.get("spacing", 3.0))
        jitter, offset = float(spec.get("jitter", 0.0)), float(spec.get("offset", 0.0))
        seg = np.diff(path, axis=0)
        lengths = np.linalg.norm(seg, axis=1)
        s_all = np.arange(0.0, lengths.sum() + 1e-9, step)
        cum = np.r_[0, np.cumsum(lengths)]
        for s in s_all:
            i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
            t = (s - cum[i]) / max(lengths[i], 1e-9)
            d = seg[i] / max(lengths[i], 1e-9)
            side = np.array([-d[1], d[0]])
            p = path[i] + seg[i] * t + side * offset + rng.uniform(-jitter, jitter, 2)
            if clear(p):
                frames.append((p, 0.0, None))
    elif "on" in spec:
        target_name, surface_name = str(spec["on"]).split(".", 1)
        target = loaded.get(target_name)
        surface = target.surfaces.get(surface_name) if target is not None else None
        if target is None or surface is None:
            raise ValueError(f"scatter '{name}': no surface '{spec['on']}'")
        m = to_scene @ target.world_transform()
        for uv in poisson_disk(rng, np.zeros(2), np.array([surface.u_extent, surface.v_extent]), spacing):
            local = surface.origin + surface.u_axis * uv[0] + surface.v_axis * uv[1]
            world = m @ np.r_[local, 1.0]
            p = world[[0, 2]]
            if clear(p):
                frames.append((p, float(world[1]), m[:3, :3] @ surface.normal))
    else:
        raise ValueError(f"scatter '{name}' needs rect:, path: or on:")

    yaw_spec = spec.get("yaw", [0.0, 360.0])
    scale_spec = spec.get("scale", 1.0)
    placed = []
    for k, (p, y, _normal) in enumerate(frames[:count]):
        params = resolve_random_params(obj_def.get("params"), rng)
        node = load({**obj_def, "params": params})
        yaw = float(rng.uniform(*yaw_spec)) if isinstance(yaw_spec, (list, tuple)) else float(yaw_spec)
        scale = float(rng.uniform(*scale_spec)) if isinstance(scale_spec, (list, tuple)) else float(scale_spec)
        node.name = f"{name}_{k + 1}"
        node.transform = Transform(translation=np.array([p[0], y, p[1]]),
                                   rotation=np.array([0.0, np.radians(yaw), 0.0]),
                                   scale=np.array([scale, scale, scale]))
        node.meta["scatter"] = {"group": name, "index": k + 1, "seed": int(spec.get("seed", 0))}
        root.add_child(node)
        loaded[node.name] = node
        occupied.append((p, radius))
        placed.append(node)
    return placed
