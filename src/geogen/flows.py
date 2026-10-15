"""Spawned items: things the runtime makes while the game runs, moves and takes away.

Everything placed at build time stays put in the export. Items are the exception: copies
of a template the runtime makes as it goes (balls from a hopper, enemies in waves), each
with its own data, gone again when a sink takes it in, it falls out of the world or it
outlives its time. Four parts, each a block on an asset:

- **Item** (``kind: item``, ``assets/items/*.yaml``): the template. ``body`` (an asset, with
  ``params``), ``mode`` (``physics``: a rigid body, the body asset's ``physics:`` block;
  ``belt`` / ``walk``: carried along a track), ``data`` (numbers or text each copy carries:
  ``value``, ``hp``, ``harm``), ``speed``, ``radius``, ``lifetime`` (s; 0: no limit),
  ``stuck`` (s at rest before a physics item is taken away; 0: never).
- **Source** (asset ``source:``): puts items out at an attachment, ``rate`` a second while
  it runs, up to ``max_alive`` at once and ``count`` in all (0: no limit).
- **Sink** (``sink:`` on a place, ``geogen.places``): takes in the items that enter it
  (``accepts``: only those items). Its state ``received`` counts them (driven) and it raises
  ``received {item, ...}`` with the item's data.
- Tracks and effectors (later): paths items follow, things that act on items in range.

Items have no stable IDs and no rules state: the contract (``docs/rules.md``) stays closed.
The rules see a source's counts as its state (``running`` and ``queued``, which rules set;
``emitted`` and ``alive``, which the runtime drives) and what happens as signals with data
(``emitted {item}``). A source's inputs: ``start``, ``stop`` and ``emit`` (one more, as soon
as there is room).

The build records ``meta.flows = {source: {...}}`` on the source asset and gives it a hidden
child per item it can make, the template, with ``meta.item`` (the item's definition). The
runtime (``runtime/godot/scripts/flows.gd``) takes the templates out of the world on load
and copies them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.node import SceneNode

META = "flows"
ITEM_META = "item"
KIND = "item"
MODES = ("physics", "belt", "walk")
ITEM_KEYS = {
    "kind": "item",
    "name": "the item's name (default: the file's)",
    "body": "the asset each copy is: a file, or {asset, params}",
    "mode": "physics (a rigid body: the body has physics:) | belt | walk (along a track)",
    "data": "{field: number or text}: what each copy carries (value, hp, harm): signals give it to rules",
    "speed": "m/s along a track (walk: the item's own pace)",
    "radius": "m: how much room it takes (default: from the body's bounds)",
    "lifetime": "s before it is taken away wherever it is (0: no limit)",
    "stuck": "s a physics item may rest anywhere but a sink before it is taken away (0: never; default 10)",
}
SOURCE_KEYS = {
    "items": "an item file, or [{item, weight}]: what it makes (weighted at random)",
    "rate": "items a second while it runs (default 1)",
    "at": "the attachment items appear at (default: the asset's origin)",
    "max_alive": "most of its items in the world at once (default 10)",
    "count": "items in all, then it stops (0: no limit)",
    "initial": "running (default) | stopped: whether it runs from the start",
    "velocity": "[x, y, z] m/s a physics item starts with, in the asset's frame",
    "spread": "m: items appear up to this far from `at`, across the asset's x and z",
    "seed": "the seed of its random choices (default 0)",
}
SOURCE_STATE = {
    "running": "bool: it makes items while true (rules set it; inputs start / stop)",
    "queued": "number: items asked for with `emit`, made as soon as there is room (rules set it)",
    "emitted": "number: items made so far (the runtime drives it)",
    "alive": "number: its items in the world now (the runtime drives it)",
}


class FlowError(ValueError):
    """An item or source that is malformed."""


# ---- items --------------------------------------------------------------------------------------

def load_item(path: str | Path) -> dict[str, Any]:
    """A ``kind: item`` file, normalised (``body`` as ``{asset, params}``)."""
    from .layout.yaml_utils import safe_load_path

    path = Path(path)
    if not path.exists():
        raise FlowError(f"no item file {path}")
    spec = safe_load_path(path) or {}
    where = str(path)
    if spec.get("kind") != KIND:
        raise FlowError(f"{where}: an item file has kind: item")
    unknown = set(spec) - set(ITEM_KEYS) - {"version", "description"}
    if unknown:
        raise FlowError(f"{where}: unknown keys {sorted(unknown)}; known: {sorted(ITEM_KEYS)}")
    body = spec.get("body")
    if isinstance(body, str):
        body = {"asset": body}
    if not isinstance(body, dict) or not isinstance(body.get("asset"), str):
        raise FlowError(f"{where}: body is an asset file or {{asset, params}}")
    mode = spec.get("mode", "physics")
    if mode not in MODES:
        raise FlowError(f"{where}: mode is one of {', '.join(MODES)}, got {mode!r}")
    data = spec.get("data") or {}
    if not isinstance(data, dict) or not all(isinstance(v, (int, float, str)) and not isinstance(v, bool)
                                             for v in data.values()):
        raise FlowError(f"{where}: data is {{field: number or text}}")
    out: dict[str, Any] = {
        "name": str(spec.get("name", path.stem)),
        "mode": mode,
        "body": {"asset": body["asset"], "params": dict(body.get("params") or {})},
        "data": {str(k): v for k, v in data.items()},
        "lifetime": float(spec.get("lifetime", 0.0)),
        "stuck": float(spec.get("stuck", 10.0)),
    }
    for key in ("speed", "radius"):
        if key in spec:
            out[key] = float(spec[key])
    return out


# ---- sources ------------------------------------------------------------------------------------

def parse_source(spec: Any, root: SceneNode, name: str) -> dict[str, Any]:
    """Normalise an asset's ``source:`` block (items are still file names: the composer
    attaches their templates, ``attach_templates``)."""
    if not isinstance(spec, dict):
        raise FlowError(f"{name}: source is {{{', '.join(SOURCE_KEYS)}}}, got {spec!r}")
    unknown = set(spec) - set(SOURCE_KEYS)
    if unknown:
        raise FlowError(f"{name}: source: unknown keys {sorted(unknown)}; known: {sorted(SOURCE_KEYS)}")
    items = spec.get("items")
    if isinstance(items, str):
        items = [{"item": items}]
    if not isinstance(items, list) or not items or not all(isinstance(i, dict) and isinstance(i.get("item"), str)
                                                           for i in items):
        raise FlowError(f"{name}: source items is an item file or [{{item, weight}}]")
    out: dict[str, Any] = {
        "items": [{"item": i["item"], "weight": float(i.get("weight", 1.0))} for i in items],
        "rate": float(spec.get("rate", 1.0)),
        "max_alive": int(spec.get("max_alive", 10)),
        "count": int(spec.get("count", 0)),
        "initial": str(spec.get("initial", "running")),
        "seed": int(spec.get("seed", 0)),
        "spread": float(spec.get("spread", 0.0)),
    }
    if out["initial"] not in ("running", "stopped"):
        raise FlowError(f"{name}: source initial is running or stopped, got {out['initial']!r}")
    if out["rate"] < 0 or out["max_alive"] < 1 or out["count"] < 0:
        raise FlowError(f"{name}: source rate and count can't be negative, and max_alive is at least 1")
    at = np.zeros(3)
    if spec.get("at"):
        attachment = root.attachments.get(str(spec["at"]))
        if attachment is None:
            raise FlowError(f"{name}: source at: no attachment '{spec['at']}' "
                            f"(it has: {', '.join(root.attachments) or 'none'})")
        at = attachment.resolve(root.size).to_matrix()[:3, 3]
    out["at"] = [round(float(v), 6) for v in at]
    if "velocity" in spec:
        velocity = spec["velocity"]
        if not isinstance(velocity, list) or len(velocity) != 3:
            raise FlowError(f"{name}: source velocity is [x, y, z]")
        out["velocity"] = [float(v) for v in velocity]
    return out


def source_of(node: SceneNode) -> dict[str, Any] | None:
    flows = node.meta.get(META)
    return flows.get("source") if isinstance(flows, dict) else None


def attach_templates(root: SceneNode, loader: Any, assets_dir: Path | str | None) -> None:
    """Give every source under ``root`` that has none yet a hidden template child per item
    (the item's body, ``meta.item`` its definition). The composer calls it at the end of
    every scene build (inner scenes first), so items are named relative to the assets
    directory (``roots.find``)."""
    from . import roots

    for node in list(root.iter_nodes()):
        source = source_of(node)
        if source is None or all("name" in entry for entry in source["items"]):
            continue                            # attached already (an inner scene did it)
        for entry in source["items"]:
            path = roots.find(entry["item"], assets_dir)
            item = load_item(path)
            body = loader.load(roots.find(item["body"]["asset"], assets_dir), params=item["body"]["params"] or None)
            if item["mode"] == "physics" and not isinstance(body.meta.get("physics"), dict):
                raise FlowError(f"{path}: a physics item's body needs a physics: block "
                                f"({item['body']['asset']} has none)")
            if body.meta.get("physics", {}).get("spawn"):
                raise FlowError(f"{path}: an item's body can't have a physics spawn (its source hands it out)")
            if "radius" not in item:
                lo, hi = _bounds(body)
                item["radius"] = round(float(max(hi[0] - lo[0], hi[2] - lo[2])) / 2, 4) if lo is not None else 0.25
            entry["name"] = item["name"]
            body.name = f"{item['name']}_template"
            body.meta[ITEM_META] = item
            body.hidden = True
            body.transform.translation = np.array(source["at"], dtype=float)
            node.add_child(body)


def _bounds(node: SceneNode) -> tuple[np.ndarray | None, np.ndarray | None]:
    inv = np.linalg.inv(node.world_transform())
    pts = [(inv @ n.world_transform() @ np.c_[n.mesh.vertices, np.ones(len(n.mesh.vertices))].T).T[:, :3]
           for n in node.iter_nodes() if n.mesh is not None and len(n.mesh.vertices)]
    if not pts:
        return None, None
    allp = np.vstack(pts)
    return allp.min(axis=0), allp.max(axis=0)


def is_template(node: SceneNode) -> bool:
    return isinstance(node.meta.get(ITEM_META), dict)


# ---- what the rules see -------------------------------------------------------------------------

def sink_of(node: SceneNode) -> dict[str, Any] | None:
    place = node.meta.get("place")
    return place.get("sink") if node.meta.get("type") == "place" and isinstance(place, dict) else None


def flow_state(node: SceneNode) -> dict[str, dict[str, Any]]:
    """The variables a source has: ``running`` and ``queued`` (rules set them), ``emitted``
    and ``alive`` (the runtime drives them); a sink's ``received``; a track's ``running`` and
    ``received`` (what its end took in)."""
    source = source_of(node)
    if sink_of(node) is not None:
        return {"received": {"type": "number", "default": 0, "driven": "flows"}}
    if track_of(node) is not None and source is None:
        return {"running": {"type": "bool", "default": True},
                "received": {"type": "number", "default": 0, "driven": "flows"}}
    if source is None:
        return {}
    return {"running": {"type": "bool", "default": source["initial"] == "running"},
            "queued": {"type": "number", "default": 0},
            "emitted": {"type": "number", "default": 0, "driven": "flows"},
            "alive": {"type": "number", "default": 0, "driven": "flows"}}


def flow_signals(node: SceneNode, fields: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """The signals a source or sink raises, with the data they carry: ``emitted {item}``;
    ``received {item, ...}`` with ``fields`` (``item_fields``: what the scene's items carry)."""
    if sink_of(node) is not None or (track_of(node) is not None and source_of(node) is None):
        return {"received": {**(fields or {}), "item": "text"}}
    if source_of(node) is None:
        return {}
    return {"emitted": {"item": "text"}}


def item_fields(root: SceneNode) -> dict[str, str]:
    """{field: number | text} of the data every item in ``root`` carries (by template)."""
    out: dict[str, str] = {}
    for node in root.iter_nodes():
        if is_template(node):
            for field, value in node.meta[ITEM_META]["data"].items():
                out[field] = "text" if isinstance(value, str) else "number"
    return out


def flow_inputs(node: SceneNode, thing: str) -> dict[str, dict[str, Any]]:
    """A source's inputs: ``start`` / ``stop`` (running), ``emit`` (one more queued); a track's
    ``start`` / ``stop``."""
    if track_of(node) is not None and source_of(node) is None:
        return {"start": {"set": f"{thing}#running", "to": True}, "stop": {"set": f"{thing}#running", "to": False}}
    if source_of(node) is None:
        return {}
    queued = f"{thing}#queued"
    return {"start": {"set": f"{thing}#running", "to": True},
            "stop": {"set": f"{thing}#running", "to": False},
            "emit": {"set": queued, "to": {"op": "+", "args": [{"get": "instance", "id": thing, "name": "queued"}, 1]}}}


# ---- tracks -------------------------------------------------------------------------------------

TRACK_KEYS = {
    "path": "[[x, y, z], ...] or {spline: [...], samples}: where items go, in the asset's (scene's) frame",
    "speed": "m/s: a belt's speed; walking items keep their own (default 1)",
    "spacing": "m: the least gap between items' centres (default: their radii)",
    "mode": "belt (items move together and queue) | walk (each at its own pace)",
    "loop": "the path closes on itself: items go round",
    "end": "auto (on to the track that starts here, else sink) | sink | stop | drop (fall off as a physics body)",
    "accepts": "[item names] an end sink takes (empty: any)",
}
SCENE_TRACK_KEYS = {**TRACK_KEYS, "surface": "{width, material, height}: a ribbon along the path (scene tracks)"}
TRACK_ENDS = ("auto", "sink", "stop", "drop")
TRACK_STEP = 0.25                 # m between a track's points
JOIN = 0.5                        # m: a source or track end this close to a track's start feeds it
TRACK_STATE = {
    "running": "bool: items on it move while true (inputs start / stop)",
    "received": "number: items its end took in (a sink end; the runtime drives it)",
}


def parse_track(spec: Any, name: str, scene: bool = False) -> dict[str, Any]:
    """Normalise a ``track:`` block (an asset's) or a ``tracks:`` entry (a scene's): the path
    resampled every ``TRACK_STEP`` m in its frame, its ``length`` and its settings."""
    from .core.profile import catmull_rom
    from .traffic import resample

    keys = SCENE_TRACK_KEYS if scene else TRACK_KEYS
    if not isinstance(spec, dict):
        raise FlowError(f"{name}: a track is {{{', '.join(keys)}}}, got {spec!r}")
    unknown = set(spec) - set(keys)
    if unknown:
        raise FlowError(f"{name}: track: unknown keys {sorted(unknown)}; known: {sorted(keys)}")
    loop = bool(spec.get("loop", False))
    path = spec.get("path")
    if isinstance(path, dict) and "spline" in path:
        points = catmull_rom(np.asarray(path["spline"], dtype=np.float64), int(path.get("samples", 8)), closed=loop)
    else:
        points = np.asarray(path if path is not None else [], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise FlowError(f"{name}: track path is [[x, y, z], ...] (two or more) or {{spline: [...]}}")
    if loop:
        points = np.vstack([points, points[:1]])
    points = resample(points, TRACK_STEP)
    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    if length < 0.1:
        raise FlowError(f"{name}: the track is shorter than 10 cm")
    mode, end = spec.get("mode", "belt"), spec.get("end", "auto")
    if mode not in ("belt", "walk"):
        raise FlowError(f"{name}: track mode is belt or walk, got {mode!r}")
    if end not in TRACK_ENDS:
        raise FlowError(f"{name}: track end is one of {', '.join(TRACK_ENDS)}, got {end!r}")
    out = {"points": [[round(float(v), 4) for v in p] for p in points], "length": round(length, 4),
           "speed": float(spec.get("speed", 1.0)), "spacing": float(spec.get("spacing", 0.0)),
           "mode": mode, "loop": loop, "end": "stop" if loop else end,
           "accepts": [str(a) for a in spec.get("accepts", [])]}
    if out["speed"] <= 0:
        raise FlowError(f"{name}: track speed must be positive")
    return out


def track_of(node: SceneNode) -> dict[str, Any] | None:
    flows = node.meta.get(META)
    return flows.get("track") if isinstance(flows, dict) else None


def scene_tracks(spec: Any, materials: Any) -> list[SceneNode]:
    """A scene's ``tracks:`` block: a node per track (``meta.flows.track``), with a ribbon
    along it when it asks for a ``surface``."""
    if not isinstance(spec, dict):
        raise FlowError(f"tracks is {{name: {{{', '.join(TRACK_KEYS)}}}}}, got {spec!r}")
    out = []
    for name, entry in spec.items():
        track = parse_track(entry, f"tracks.{name}", scene=True)
        node = SceneNode(str(name), meta={META: {"track": track}}, tags=["flows.track"])
        surface = entry.get("surface")
        if surface:
            from .railway import _sweep

            surface = {} if surface is True else surface
            half, height = float(surface.get("width", 1.0)) / 2, float(surface.get("height", 0.02))
            points = np.asarray(track["points"], dtype=np.float64)
            ribbon = SceneNode(f"{name}_surface", mesh=_sweep([[-half, -0.1], [half, -0.1], [half, height],
                                                                [-half, height]], points[:-1] if track["loop"] else points,
                                                               track["loop"], materials.load(surface.get("material", "dirt"))))
            ribbon.meta["walkable"] = True
            node.add_child(ribbon)
        out.append(node)
    return out


# ---- the network: tracks and sources in world space, as a runtime sees them ---------------------

def _ordered(nodes: list[SceneNode], at: Any) -> list[SceneNode]:
    """Nodes in a stable order both runtimes share: by the world point ``at(node)`` gives."""
    return sorted(nodes, key=lambda n: tuple(round(float(v), 3) for v in at(n)))


def network(root: SceneNode) -> dict[str, Any]:
    """The scene's tracks and sources in world space, joined by geometry: a track whose end is
    within ``JOIN`` of another's start feeds it (``end: auto``), a source whose spawn point is
    within ``JOIN`` of a track's start puts its items on it. Tracks and sources are ordered by
    position (the order the runtime steps them in)."""
    tracks: list[dict[str, Any]] = []
    for node in root.iter_nodes():
        track = track_of(node)
        if track is None:
            continue
        world = node.world_transform()
        points = np.c_[np.asarray(track["points"]), np.ones(len(track["points"]))] @ world.T
        points = points[:, :3]
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        tracks.append({**track, "node": node, "world": points, "cum": cum, "length": float(cum[-1])})
    tracks.sort(key=lambda t: tuple(round(float(v), 3) for v in t["world"][0]))
    for t in tracks:
        t["next"] = None
        if t["loop"] or t["end"] not in ("auto",):
            continue
        nearest = min((u for u in tracks if u is not t), key=lambda u: float(np.linalg.norm(u["world"][0] - t["world"][-1])),
                      default=None)
        if nearest is not None and np.linalg.norm(nearest["world"][0] - t["world"][-1]) <= JOIN:
            t["next"] = tracks.index(nearest)
    sources = []
    for node in root.iter_nodes():
        source = source_of(node)
        if source is None:
            continue
        at = (node.world_transform() @ np.r_[source["at"], 1.0])[:3]
        templates = {n.meta[ITEM_META]["name"]: n.meta[ITEM_META] for n in node.iter_nodes() if is_template(n)}
        track = min(range(len(tracks)), key=lambda k: float(np.linalg.norm(tracks[k]["world"][0] - at)), default=None)
        if track is not None and np.linalg.norm(tracks[track]["world"][0] - at) > JOIN:
            track = None
        sources.append({**source, "node": node, "world_at": at, "templates": templates, "track": track})
    sources.sort(key=lambda s: tuple(round(float(v), 3) for v in s["world_at"]))
    return {"tracks": tracks, "sources": sources}


def check_network(root: SceneNode) -> list[str]:
    """What is wrong with the scene's items, sources and tracks (an ``--inspect`` check)."""
    net = network(root)
    out = []
    for source in net["sources"]:
        name = source["node"].name
        modes = {item["mode"] for item in source["templates"].values()}
        if modes - {"physics"} and source["track"] is None:
            out.append(f"{name}: its {'/'.join(sorted(modes - {'physics'}))} items need a track that starts within "
                       f"{JOIN} m of its spawn point")
        if "physics" in modes and len(modes) > 1:
            out.append(f"{name}: makes physics items and track items: a source makes one kind")
    return out


def _pick(source: dict[str, Any], weights: list[float]) -> int:
    """Smooth weighted round robin: deterministic, so runtimes agree on which item comes next."""
    current = source.setdefault("_current", [0.0] * len(weights))
    total = sum(weights)
    for k, w in enumerate(weights):
        current[k] += w
    best = max(range(len(weights)), key=lambda k: (current[k], -k))
    current[best] -= total
    return best


def at_distance(track: dict[str, Any], s: float) -> np.ndarray:
    """The world point ``s`` m along a track (from ``network``)."""
    cum, pts = track["cum"], track["world"]
    s = min(max(s, 0.0), float(cum[-1]))
    k = min(int(np.searchsorted(cum, s, side="right")) - 1, len(pts) - 2)
    seg = cum[k + 1] - cum[k]
    t = 0.0 if seg <= 0 else (s - cum[k]) / seg
    return pts[k] + (pts[k + 1] - pts[k]) * t


def simulate(root: SceneNode, ticks: int, dt: float = 1.0 / 60.0,
             running: dict[str, bool] | None = None) -> dict[str, Any]:
    """The reference for what the runtime does with track items (belt and walk modes; physics
    items and sinks on places are the physics engine's, so they aren't simulated): ``ticks``
    steps of ``dt`` from an empty world. ``running`` overrides sources' and tracks' running
    state by node name. Returns ``{items: [{id, item, track (index), s}], emitted: [per source],
    received: [per track], events: [[tick, kind, ...]]}``. flows.gd steps the same way."""
    net = network(root)
    running = running or {}
    tracks, sources = net["tracks"], net["sources"]
    for t in tracks:
        t["received"] = 0
        t["on"] = running.get(t["node"].name, True)
    for s in sources:
        s["due"], s["made"] = 0.0, 0
        s.pop("_current", None)
        s.pop("_pending", None)
        s["on"] = running.get(s["node"].name, s["initial"] == "running")
    items: list[dict[str, Any]] = []
    events: list[list[Any]] = []
    next_id = 1

    def gap(a: dict[str, Any], b: dict[str, Any], track: dict[str, Any]) -> float:
        return max(track["spacing"], a["radius"] + b["radius"])

    for tick in range(ticks):
        for k, s in enumerate(sources):                     # sources first
            if s["track"] is None or all(t["mode"] == "physics" for t in s["templates"].values()):
                continue
            s["due"] = min(s["due"] + dt * s["rate"], 1.0) if s["on"] and s["rate"] > 0 else 0.0
            track = tracks[s["track"]]
            alive = sum(1 for it in items if it["source"] == k)
            if s["due"] >= 1.0 and alive < s["max_alive"] and (s["count"] == 0 or s["made"] < s["count"]):
                pick = s.pop("_pending", None)
                if pick is None:
                    pick = _pick(s, [e["weight"] for e in s["items"]])
                item = s["templates"][s["items"][pick]["name"]]
                new = {"radius": item["radius"]}
                on_track = [it for it in items if it["track"] == s["track"]]
                if not all(it["s"] >= gap(new, it, track) for it in on_track):
                    s["_pending"] = pick                    # the same pick waits for room
                else:
                    s["due"] -= 1.0
                    s["made"] += 1
                    items.append({"id": next_id, "item": item["name"], "source": k, "track": s["track"], "s": 0.0,
                                  "radius": item["radius"], "speed": item.get("speed", track["speed"]),
                                  "data": dict(item["data"])})
                    events.append([tick, "emitted", k, next_id])
                    next_id += 1
        moved: set[int] = set()
        for index, track in enumerate(tracks):
            if not track["on"]:
                continue
            order = sorted((it for it in items if it["track"] == index and it["id"] not in moved),
                           key=lambda it: -it["s"])
            length = track["length"]
            for j, it in enumerate(order):
                v = track["speed"] if track["mode"] == "belt" else it["speed"]
                target = it["s"] + v * dt
                limit = float("inf")
                if j > 0:
                    lead = order[j - 1]
                    limit = lead["s"] - gap(it, lead, track)
                elif track["loop"] and len(order) > 1:
                    lead = order[-1]
                    limit = lead["s"] + length - gap(it, lead, track)
                elif track["next"] is not None:
                    ahead = [o for o in items if o["track"] == track["next"]]
                    if ahead:
                        lead = min(ahead, key=lambda o: o["s"])
                        limit = length + lead["s"] - gap(it, lead, tracks[track["next"]])
                target = max(it["s"], min(target, limit))
                if track["loop"]:
                    it["s"] = target % length
                elif target >= length:
                    if track["next"] is not None:
                        it["track"], it["s"] = track["next"], target - length
                        moved.add(it["id"])
                    elif track["end"] == "stop":
                        it["s"] = length
                    elif track["end"] in ("auto", "sink") and (not track["accepts"] or it["item"] in track["accepts"]):
                        items.remove(it)
                        track["received"] += 1
                        events.append([tick, "received", index, it["id"]])
                    elif track["end"] == "drop":
                        items.remove(it)
                        events.append([tick, "dropped", index, it["id"]])
                    else:
                        it["s"] = length
                else:
                    it["s"] = target
    return {"items": [{"id": it["id"], "item": it["item"], "track": it["track"], "s": round(it["s"], 6)} for it in items],
            "emitted": [s["made"] for s in sources], "received": [t["received"] for t in tracks], "events": events}
