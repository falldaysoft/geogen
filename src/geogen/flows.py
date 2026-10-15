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
    and ``alive`` (the runtime drives them); a sink's ``received``."""
    source = source_of(node)
    if sink_of(node) is not None:
        return {"received": {"type": "number", "default": 0, "driven": "flows"}}
    if source is None:
        return {}
    return {"running": {"type": "bool", "default": source["initial"] == "running"},
            "queued": {"type": "number", "default": 0},
            "emitted": {"type": "number", "default": 0, "driven": "flows"},
            "alive": {"type": "number", "default": 0, "driven": "flows"}}


def flow_signals(node: SceneNode, fields: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """The signals a source or sink raises, with the data they carry: ``emitted {item}``;
    ``received {item, ...}`` with ``fields`` (``item_fields``: what the scene's items carry)."""
    if sink_of(node) is not None:
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
    """A source's inputs: ``start`` / ``stop`` (running), ``emit`` (one more queued)."""
    if source_of(node) is None:
        return {}
    queued = f"{thing}#queued"
    return {"start": {"set": f"{thing}#running", "to": True},
            "stop": {"set": f"{thing}#running", "to": False},
            "emit": {"set": queued, "to": {"op": "+", "args": [{"get": "instance", "id": thing, "name": "queued"}, 1]}}}
