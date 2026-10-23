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
- **Track** (asset ``track:`` or scene ``tracks:``): a path belt and walk items follow, or that
  gravity rolls items down (``roll``: balls in a chute, which fall off a ``drop`` end as bodies).
- **Effector** (asset ``effector:``): acts on items in range: ``damage`` (items with
  ``data.hp``; at 0 the item is destroyed and its source raises ``killed``) or ``take``.
- **Waves** (``kind: waves``, ``assets/waves/*.yaml``): what a source sends in turn, wave by
  wave (``source: {waves: waves/x.yaml}``).

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
    "title": "its name when carried (default: the name, capitalised)",
    "tags": "[tag, ...] for the inventory (key, quest)",
    "opens": "[lock key, ...] (globs) it unlocks besides its own id: a master key's key_room_*",
    "use": "a verb (Eat, Drink): an inventory offers using the item; a rule on {use_item: <id>} says what happens",
    "body": "the asset each copy is: a file, or {asset, params}",
    "mode": "physics (a rigid body: the body has physics:) | belt | walk (along a track)",
    "data": "{field: number or text}: what each copy carries (value, hp, harm): signals give it to rules",
    "speed": "m/s along a track (walk: the item's own pace)",
    "radius": "m: how much room it takes (default: from the body's bounds)",
    "lifetime": "s before it is taken away wherever it is (0: no limit)",
    "stuck": "s a physics item may rest anywhere but a sink before it is taken away (0: never; default 10)",
    "gait": "{stride: m, bob: m, sway: deg, waddle: deg}: a walk item's body bobs (twice a stride), rocks "
            "side to side and swings its heading as it goes (the runtime draws it; positions are unchanged)",
}
GAIT_KEYS = {"stride": 0.5, "bob": 0.0, "sway": 0.0, "waddle": 0.0}
SOURCE_KEYS = {
    "items": "an item file, or [{item, weight}]: what it makes (weighted, in turn)",
    "waves": "a waves file: it sends those, wave by wave, instead of items at a rate",
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
    "restarts": "number: input `reset` adds one; the runtime then takes its items away, stops it and "
                "starts its waves and counts over",
}


class FlowError(ValueError):
    """An item or source that is malformed."""


# ---- items --------------------------------------------------------------------------------------

def carried(path: str | Path) -> dict[str, Any]:
    """What an inventory knows of the item file ``path``: ``{title, tags?, opens?}`` (geogen.18).
    The file is checked as ``load_item`` checks it."""
    from .layout.yaml_utils import safe_load_path

    item = load_item(path)
    spec = safe_load_path(path) or {}
    out: dict[str, Any] = {"title": str(spec.get("title") or item["name"].replace("_", " ").capitalize())}
    for key in ("tags", "opens"):
        listed = spec.get(key) or []
        if not isinstance(listed, list) or not all(isinstance(t, str) and t for t in listed):
            raise FlowError(f"{path}: {key} is a list of words")
        if listed:
            out[key] = list(listed)
    use = spec.get("use")
    if use is not None:
        if not isinstance(use, str) or not use.strip():
            raise FlowError(f"{path}: use is a verb (Eat, Drink), got {use!r}")
        out["use"] = use.strip()
    return out


def load_item(path: str | Path) -> dict[str, Any]:
    """A ``kind: item`` file, normalised (``body`` as ``{asset, params}``)."""
    from .layout.yaml_utils import load_kind

    path = Path(path)
    spec = load_kind(path, KIND, None, ITEM_KEYS, FlowError)
    where = str(path)
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
    if "gait" in spec:
        gait = spec["gait"]
        if not isinstance(gait, dict) or set(gait) - set(GAIT_KEYS) \
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in gait.values()):
            raise FlowError(f"{where}: gait is {{{', '.join(GAIT_KEYS)}}} (numbers), got {gait!r}")
        out["gait"] = {k: float(gait.get(k, default)) for k, default in GAIT_KEYS.items()}
        if out["gait"]["stride"] <= 0:
            raise FlowError(f"{where}: gait stride is more than 0 m")
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
    waves = spec.get("waves")
    if isinstance(items, str):
        items = [{"item": items}]
    if waves == "":
        waves = None                            # a param left empty: items at a rate
    if waves is not None:
        if not isinstance(waves, str):
            raise FlowError(f"{name}: source waves is a waves file (it wins over items)")
        items = []
    elif not isinstance(items, list) or not items or not all(isinstance(i, dict) and isinstance(i.get("item"), str)
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
    if waves is not None:
        out["waves"] = waves                    # the file: attach_templates reads it
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
        if source is None or (not isinstance(source.get("waves"), str) and all("name" in e for e in source["items"])):
            continue                            # attached already (an inner scene did it)
        if isinstance(source.get("waves"), str):
            waves = load_waves(roots.find(source["waves"], assets_dir))
            source["waves"] = waves
            files = list(dict.fromkeys(group["item"] for wave in waves["list"] for group in wave["items"]))
            source["items"] = [{"item": f, "weight": 1.0} for f in files]
        names: dict[str, str] = {}
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
                box = body.mesh_bounds(body)
                item["radius"] = round(float(max(box[1][0] - box[0][0], box[1][2] - box[0][2])) / 2, 4) \
                    if box is not None else 0.25
            entry["name"] = names[entry["item"]] = item["name"]
            body.name = f"{item['name']}_template"
            body.meta[ITEM_META] = item
            body.hidden = True
            body.transform.translation = np.array(source["at"], dtype=float)
            node.add_child(body)
        if isinstance(source.get("waves"), dict):
            for wave in source["waves"]["list"]:
                for group in wave["items"]:
                    group["name"] = names[group.pop("item")]


WAVES_KEYS = {
    "kind": "waves",
    "waves": "[{items: [{item, count, gap}], delay}]: each wave's groups, in order",
    "auto": "true: the next wave follows when one is cleared; false (default): the source stops until started again",
}


def load_waves(path: str | Path) -> dict[str, Any]:
    """A ``kind: waves`` file: ``{auto, list: [{delay, items: [{item, count, gap}]}]}``."""
    from .layout.yaml_utils import load_kind

    path = Path(path)
    spec = load_kind(path, "waves", None, WAVES_KEYS, FlowError)
    where = str(path)
    waves = spec.get("waves")
    if not isinstance(waves, list) or not waves:
        raise FlowError(f"{where}: waves is a list of {{items: [{{item, count, gap}}], delay}}")
    out = []
    for k, wave in enumerate(waves):
        groups = wave.get("items") if isinstance(wave, dict) else None
        if not isinstance(groups, list) or not groups or set(wave) - {"items", "delay"}:
            raise FlowError(f"{where}: waves.{k} is {{items: [{{item, count, gap}}], delay}}")
        entry = {"delay": float(wave.get("delay", 2.0)), "items": []}
        for g, group in enumerate(groups):
            if not isinstance(group, dict) or not isinstance(group.get("item"), str) or set(group) - {"item", "count", "gap"}:
                raise FlowError(f"{where}: waves.{k}.items.{g} is {{item, count, gap}}")
            entry["items"].append({"item": group["item"], "count": int(group.get("count", 1)),
                                   "gap": float(group.get("gap", 1.0))})
        out.append(entry)
    return {"auto": bool(spec.get("auto", False)), "list": out}


# ---- effectors ----------------------------------------------------------------------------------

EFFECTOR_KEYS = {
    "range": "m from its `at` point (default 5)",
    "rate": "goes a second (default 1)",
    "targets": "first (furthest along) | last | nearest | strongest (most hp): which item in range",
    "accepts": "[item names] it acts on (empty: any)",
    "effect": "{damage: n} (items with data.hp) | {take: true}",
    "at": "the attachment it acts from (default: the origin)",
    "aim": "{yaw: part, pitch: part, turn_rate: deg/s}: parts turned toward the target (the look of it)",
    "projectile": "{kind: instant | bolt | tracer, speed, color: [r, g, b]}: what a shot looks like",
    "enabled": "true (default) | false: its `enabled` state at the start (inputs enable / disable)",
}
TARGETS = ("first", "last", "nearest", "strongest")
PROJECTILES = ("instant", "bolt", "tracer")


def parse_effector(spec: Any, root: SceneNode, part_nodes: dict[str, SceneNode], name: str) -> dict[str, Any]:
    """Normalise an asset's ``effector:`` block."""
    if not isinstance(spec, dict):
        raise FlowError(f"{name}: effector is {{{', '.join(EFFECTOR_KEYS)}}}, got {spec!r}")
    unknown = set(spec) - set(EFFECTOR_KEYS)
    if unknown:
        raise FlowError(f"{name}: effector: unknown keys {sorted(unknown)}; known: {sorted(EFFECTOR_KEYS)}")
    effect = spec.get("effect", {"damage": 1})
    if not isinstance(effect, dict) or len(effect) != 1 or not ({"damage", "take"} & set(effect)):
        raise FlowError(f"{name}: effector effect is {{damage: n}} or {{take: true}}, got {effect!r}")
    out: dict[str, Any] = {
        "range": float(spec.get("range", 5.0)),
        "rate": float(spec.get("rate", 1.0)),
        "targets": str(spec.get("targets", "first")),
        "accepts": [str(a) for a in spec.get("accepts", [])],
        "effect": {"damage": float(effect["damage"])} if "damage" in effect else {"take": True},
        "enabled": bool(spec.get("enabled", True)),
    }
    if out["targets"] not in TARGETS:
        raise FlowError(f"{name}: effector targets is one of {', '.join(TARGETS)}, got {out['targets']!r}")
    if out["range"] <= 0 or out["rate"] <= 0:
        raise FlowError(f"{name}: effector range and rate must be positive")
    at = np.zeros(3)
    if spec.get("at"):
        attachment = root.attachments.get(str(spec["at"]))
        if attachment is None:
            raise FlowError(f"{name}: effector at: no attachment '{spec['at']}' (it has: {', '.join(root.attachments) or 'none'})")
        at = attachment.resolve(root.size).to_matrix()[:3, 3]
    out["at"] = [round(float(v), 6) for v in at]
    aim = spec.get("aim")
    if aim:
        if not isinstance(aim, dict) or set(aim) - {"yaw", "pitch", "turn_rate"}:
            raise FlowError(f"{name}: effector aim is {{yaw: part, pitch: part, turn_rate}}")
        for key in ("yaw", "pitch"):
            if key in aim and aim[key] not in part_nodes:
                raise FlowError(f"{name}: effector aim {key}: no part '{aim[key]}'")
        out["aim"] = {**{k: str(aim[k]) for k in ("yaw", "pitch") if k in aim},
                      "turn_rate": float(aim.get("turn_rate", 360.0))}
    projectile = spec.get("projectile", {"kind": "tracer"})
    if not isinstance(projectile, dict) or projectile.get("kind", "tracer") not in PROJECTILES \
            or set(projectile) - {"kind", "speed", "color"}:
        raise FlowError(f"{name}: effector projectile is {{kind: {' | '.join(PROJECTILES)}, speed, color}}")
    out["projectile"] = {"kind": projectile.get("kind", "tracer"), "speed": float(projectile.get("speed", 20.0)),
                         "color": [float(c) for c in projectile.get("color", [1.0, 0.75, 0.3])]}
    return out


def effector_of(node: SceneNode) -> dict[str, Any] | None:
    flows = node.meta.get(META)
    return flows.get("effector") if isinstance(flows, dict) else None


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
    effector = effector_of(node)
    if effector is not None and source is None:
        return {"enabled": {"type": "bool", "default": effector["enabled"]},
                "taken": {"type": "number", "default": 0, "driven": "flows"}}
    if source is None:
        return {}
    return {"running": {"type": "bool", "default": source["initial"] == "running"},
            "queued": {"type": "number", "default": 0},
            "emitted": {"type": "number", "default": 0, "driven": "flows"},
            "alive": {"type": "number", "default": 0, "driven": "flows"},
            "killed": {"type": "number", "default": 0, "driven": "flows"},
            "wave": {"type": "number", "default": 0, "driven": "flows"},
            "restarts": {"type": "number", "default": 0}}


def flow_signals(node: SceneNode, fields: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """The signals a source or sink raises, with the data they carry: ``emitted {item}``;
    ``received {item, ...}`` with ``fields`` (``item_fields``: what the scene's items carry)."""
    if sink_of(node) is not None or (track_of(node) is not None and source_of(node) is None):
        return {"received": {**(fields or {}), "item": "text"}}
    if effector_of(node) is not None and source_of(node) is None:
        return {"took": {**(fields or {}), "item": "text"}}
    source = source_of(node)
    if source is None:
        return {}
    out = {"emitted": {"item": "text"}, "killed": {**(fields or {}), "item": "text"}}
    if source.get("waves"):
        out.update({name: {"wave": "number"} for name in ("wave_started", "wave_cleared", "all_cleared")})
    return out


def item_fields(root: SceneNode) -> dict[str, str]:
    """{field: number | text} of the data every item in ``root`` carries (by template)."""
    out: dict[str, str] = {}
    for node in root.iter_nodes():
        if is_template(node):
            for field, value in node.meta[ITEM_META]["data"].items():
                out[field] = "text" if isinstance(value, str) else "number"
    return out


def flow_inputs(node: SceneNode, thing: str) -> dict[str, dict[str, Any]]:
    """A source's inputs: ``start`` / ``stop`` (running), ``emit`` (one more queued), ``reset``
    (one more restart: the runtime clears it and starts over); a track's ``start`` / ``stop``."""
    if track_of(node) is not None and source_of(node) is None:
        return {"start": {"set": f"{thing}#running", "to": True}, "stop": {"set": f"{thing}#running", "to": False}}
    if effector_of(node) is not None and source_of(node) is None:
        return {"enable": {"set": f"{thing}#enabled", "to": True}, "disable": {"set": f"{thing}#enabled", "to": False}}
    if source_of(node) is None:
        return {}
    queued = f"{thing}#queued"
    return {"start": {"set": f"{thing}#running", "to": True},
            "stop": {"set": f"{thing}#running", "to": False},
            "emit": {"set": queued, "to": {"op": "+", "args": [{"get": "instance", "id": thing, "name": "queued"}, 1]}},
            "reset": {"set": f"{thing}#restarts",
                      "to": {"op": "+", "args": [{"get": "instance", "id": thing, "name": "restarts"}, 1]}}}


# ---- tracks -------------------------------------------------------------------------------------

TRACK_KEYS = {
    "path": "[[x, y, z], ...] or {spline: [...], samples}: where items go, in the asset's (scene's) frame",
    "speed": "m/s: a belt's speed; walking items keep their own; a roll track's starting speed (default 1)",
    "spacing": "m: the least gap between items' centres (default: their radii)",
    "mode": "belt (items move together and queue) | walk (each at its own pace) | roll (gravity: balls down a chute)",
    "damping": "roll: how fast a rolling item slows, per second (default 0.1)",
    "loop": "the path closes on itself: items go round",
    "end": "auto (on to the track that starts here, else sink) | sink | stop | drop (fall off as a physics body)",
    "accepts": "[item names] an end sink takes (empty: any)",
}
SCENE_TRACK_KEYS = {**TRACK_KEYS, "surface": "{width, material, height}: a ribbon along the path (scene tracks)"}
TRACK_ENDS = ("auto", "sink", "stop", "drop")
TRACK_MODES = ("belt", "walk", "roll")
ROLL_GRAVITY = 9.81 * 5.0 / 7.0          # a solid ball rolling without slipping: (5/7) g sin(slope)
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
    if mode not in TRACK_MODES:
        raise FlowError(f"{name}: track mode is one of {', '.join(TRACK_MODES)}, got {mode!r}")
    if end not in TRACK_ENDS:
        raise FlowError(f"{name}: track end is one of {', '.join(TRACK_ENDS)}, got {end!r}")
    out = {"points": [[round(float(v), 4) for v in p] for p in points], "length": round(length, 4),
           "speed": float(spec.get("speed", 1.0)), "spacing": float(spec.get("spacing", 0.0)),
           "mode": mode, "loop": loop, "end": "stop" if loop else end,
           "accepts": [str(a) for a in spec.get("accepts", [])]}
    if mode == "roll":
        out["damping"] = float(spec.get("damping", 0.1))
    if out["speed"] < 0 or (out["speed"] == 0 and mode != "roll"):
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
    effectors = []
    for node in root.iter_nodes():
        effector = effector_of(node)
        if effector is not None:
            effectors.append({**effector, "node": node, "world_at": (node.world_transform() @ np.r_[effector["at"], 1.0])[:3]})
    effectors.sort(key=lambda e: tuple(round(float(v), 3) for v in e["world_at"]))
    return {"tracks": tracks, "sources": sources, "effectors": effectors}


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
        if "physics" in modes and source["track"] is None and len(modes) > 1:
            out.append(f"{name}: makes physics items and track items: a source makes one kind")
    for effector in net["effectors"]:
        reach = [t for t in net["tracks"] if np.min(np.linalg.norm(t["world"] - effector["world_at"], axis=1)) <= effector["range"]]
        if net["tracks"] and not reach:
            out.append(f"{effector['node'].name}: no track comes within its range ({effector['range']} m)")
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


def _fall(track: dict[str, Any], s: float) -> float:
    """How steeply a track falls at ``s``: -dy/ds of the segment there (sin of its slope)."""
    cum, pts = track["cum"], track["world"]
    k = min(max(int(np.searchsorted(cum, s, side="right")) - 1, 0), len(pts) - 2)
    seg = cum[k + 1] - cum[k]
    return 0.0 if seg <= 0 else float(pts[k][1] - pts[k + 1][1]) / seg


def at_distance(track: dict[str, Any], s: float) -> np.ndarray:
    """The world point ``s`` m along a track (from ``network``)."""
    cum, pts = track["cum"], track["world"]
    s = min(max(s, 0.0), float(cum[-1]))
    k = min(int(np.searchsorted(cum, s, side="right")) - 1, len(pts) - 2)
    seg = cum[k + 1] - cum[k]
    t = 0.0 if seg <= 0 else (s - cum[k]) / seg
    return pts[k] + (pts[k + 1] - pts[k]) * t


def simulate(root: SceneNode, ticks: int, dt: float = 1.0 / 60.0,
             running: dict[str, bool] | None = None, auto: bool | None = None) -> dict[str, Any]:
    """The reference for what the runtime does with track items (belt, walk and roll modes;
    physics items off tracks and sinks on places are the physics engine's): ``ticks``
    steps of ``dt`` from an empty world. ``running`` overrides sources' and tracks' running
    and effectors' enabled state by node name; ``auto`` overrides every waves file's ``auto``
    (True: each wave follows the last, as if a rule started the source again). Returns
    ``{items: [{id, item, track (index), s, hp?}], emitted, killed, wave: [per source],
    received: [per track], through: [per track: the names of the items it took in],
    taken: [per effector], events: [[tick, kind, ...]]}``. flows.gd steps the same way: sources (in position order),
    then tracks, then effectors."""
    net = network(root)
    running = running or {}
    tracks, sources, effectors = net["tracks"], net["sources"], net["effectors"]
    for t in tracks:
        t["received"] = 0
        t["through"] = []
        t["on"] = running.get(t["node"].name, True)
    for s in sources:
        s.update(due=0.0, made=0, killed=0, wave=0, phase="idle", timer=0.0, group=0, left=0)
        s.pop("_current", None)
        s.pop("_pending", None)
        s["on"] = running.get(s["node"].name, s["initial"] == "running")
    for e in effectors:
        e.update(cool=0.0, taken=0)
        e["on"] = running.get(e["node"].name, e["enabled"])
    items: list[dict[str, Any]] = []
    events: list[list[Any]] = []
    state = {"next_id": 1}

    def gap(a: dict[str, Any], b: dict[str, Any], track: dict[str, Any]) -> float:
        return max(track["spacing"], a["radius"] + b["radius"])

    def try_spawn(k: int, s: dict[str, Any], tick: int, name: str | None) -> bool:
        """Put the source's next item (``name``, else its pick) on its track if the start is clear."""
        track = tracks[s["track"]]
        if name is None:
            pick = s.pop("_pending", None)
            if pick is None:
                pick = _pick(s, [e["weight"] for e in s["items"]])
            name = s["items"][pick]["name"]
        else:
            pick = None
        item = s["templates"][name]
        if not all(it["s"] >= gap(item, it, track) for it in items if it["track"] == s["track"]):
            if pick is not None:
                s["_pending"] = pick            # the same pick waits for room
            return False
        s["made"] += 1
        new = {"id": state["next_id"], "item": name, "source": k, "track": s["track"], "s": 0.0, "travelled": 0.0,
               "radius": item["radius"], "speed": item.get("speed", track["speed"]), "v": track["speed"],
               "data": dict(item["data"])}
        items.append(new)
        events.append([tick, "emitted", k, state["next_id"]])
        state["next_id"] += 1
        return True

    def waves(k: int, s: dict[str, Any], tick: int) -> None:
        plan, alive = s["waves"], any(it["source"] == k for it in items)
        if s["phase"] == "clearing":
            if not alive:
                events.append([tick, "wave_cleared", k, s["wave"]])
                s["phase"] = "idle"
                if s["wave"] >= len(plan["list"]):
                    events.append([tick, "all_cleared", k, s["wave"]])
                    s["on"] = False
                elif not (plan["auto"] if auto is None else auto):
                    s["on"] = False
            return
        if not s["on"]:
            return
        if s["phase"] == "idle":
            if s["wave"] < len(plan["list"]):
                s["phase"], s["timer"] = "delay", plan["list"][s["wave"]]["delay"]
            return
        if s["phase"] == "delay":
            s["timer"] -= dt
            if s["timer"] <= 0:
                s["wave"] += 1
                events.append([tick, "wave_started", k, s["wave"]])
                s.update(phase="spawning", group=0, timer=0.0, left=plan["list"][s["wave"] - 1]["items"][0]["count"])
            return
        groups = plan["list"][s["wave"] - 1]["items"]
        s["timer"] -= dt
        if s["timer"] > 0 or sum(1 for it in items if it["source"] == k) >= s["max_alive"]:
            return
        group = groups[s["group"]]
        if not try_spawn(k, s, tick, group["name"]):
            return
        s["left"] -= 1
        s["timer"] = group["gap"]
        if s["left"] <= 0:
            s["group"] += 1
            if s["group"] >= len(groups):
                s["phase"] = "clearing"
            else:
                s["left"] = groups[s["group"]]["count"]

    def remove(it: dict[str, Any]) -> None:
        items.remove(it)

    for tick in range(ticks):
        for k, s in enumerate(sources):                     # sources first
            if s["track"] is None or (all(t["mode"] == "physics" for t in s["templates"].values())
                                      and tracks[s["track"]]["mode"] != "roll"):
                continue
            if s.get("waves"):
                waves(k, s, tick)
                continue
            s["due"] = min(s["due"] + dt * s["rate"], 1.0) if s["on"] and s["rate"] > 0 else 0.0
            alive = sum(1 for it in items if it["source"] == k)
            if s["due"] >= 1.0 and alive < s["max_alive"] and (s["count"] == 0 or s["made"] < s["count"]):
                if try_spawn(k, s, tick, None):
                    s["due"] -= 1.0
        moved: set[int] = set()
        for index, track in enumerate(tracks):
            if not track["on"]:
                continue
            order = sorted((it for it in items if it["track"] == index and it["id"] not in moved),
                           key=lambda it: -it["s"])
            length = track["length"]
            for j, it in enumerate(order):
                if track["mode"] == "roll":
                    it["v"] = max(0.0, it["v"] + (ROLL_GRAVITY * _fall(track, it["s"]) - track["damping"] * it["v"]) * dt)
                    v = it["v"]
                else:
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
                if track["mode"] == "roll" and target - it["s"] < v * dt:
                    it["v"] = (target - it["s"]) / dt         # held up behind the one ahead
                it["travelled"] += target - it["s"]
                if track["loop"]:
                    it["s"] = target % length
                elif target >= length:
                    if track["next"] is not None:
                        it["track"], it["s"] = track["next"], target - length
                        moved.add(it["id"])
                    elif track["end"] == "stop":
                        it["s"] = length
                    elif track["end"] in ("auto", "sink") and (not track["accepts"] or it["item"] in track["accepts"]):
                        remove(it)
                        track["received"] += 1
                        track["through"].append(it["item"])
                        events.append([tick, "received", index, it["id"]])
                    elif track["end"] == "drop":
                        remove(it)
                        events.append([tick, "dropped", index, it["id"]])
                    else:
                        it["s"] = length
                else:
                    it["s"] = target
        for index, e in enumerate(effectors):              # then effectors
            if not e["on"]:
                continue
            e["cool"] -= dt
            if e["cool"] > 0:
                continue
            damage = "damage" in e["effect"]
            found = []
            for it in items:
                if (e["accepts"] and it["item"] not in e["accepts"]) or (damage and "hp" not in it["data"]):
                    continue
                d = float(np.linalg.norm(at_distance(tracks[it["track"]], it["s"]) - e["world_at"]))
                if d <= e["range"]:
                    found.append((it, d))
            if not found:
                e["cool"] = 0.0
                continue
            if e["targets"] == "first":
                it, _ = min(found, key=lambda f: (-f[0]["travelled"], f[0]["id"]))
            elif e["targets"] == "last":
                it, _ = min(found, key=lambda f: (f[0]["travelled"], f[0]["id"]))
            elif e["targets"] == "nearest":
                it, _ = min(found, key=lambda f: (f[1], f[0]["id"]))
            else:
                it, _ = min(found, key=lambda f: (-f[0]["data"]["hp"], -f[0]["travelled"], f[0]["id"]))
            e["cool"] = 1.0 / e["rate"]
            events.append([tick, "fired", index, it["id"]])
            if damage:
                it["data"]["hp"] = it["data"]["hp"] - e["effect"]["damage"]
                if it["data"]["hp"] <= 0:
                    remove(it)
                    sources[it["source"]]["killed"] += 1
                    events.append([tick, "killed", it["source"], it["id"]])
            else:
                remove(it)
                e["taken"] += 1
                events.append([tick, "took", index, it["id"]])
    return {"items": [{"id": it["id"], "item": it["item"], "track": it["track"], "s": round(it["s"], 6),
                       **({"hp": it["data"]["hp"]} if "hp" in it["data"] else {})} for it in items],
            "emitted": [s["made"] for s in sources], "killed": [s["killed"] for s in sources],
            "wave": [s["wave"] for s in sources], "received": [t["received"] for t in tracks],
            "through": [t["through"] for t in tracks],
            "taken": [e["taken"] for e in effectors], "events": events}
