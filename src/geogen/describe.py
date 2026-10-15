"""``--describe``: the catalogue of what can be built with, as text, JSON or markdown.

Someone (or an LLM) asked for "a garage with a car and an open door" needs the
existing vocabulary: the primitives and their keys, the assets with their
params, attachment points, surfaces and interactions, the scenes, and the data
kinds (NPCs, traffic, trains, ...). This module collects all of that into
plain dicts and renders them three ways from the same data:

- ``describe_text`` / ``describe_json``: ``python -m geogen.main --describe [NAME] [--json]``.
  Without a name it lists everything from the YAML alone (fast); with one it
  builds the asset and adds what only the built asset knows (size, attachment
  positions, surfaces, interaction states, affordances).
- ``markdown_pages``: the browsable site in ``docs/catalogue`` (``--describe --markdown DIR``),
  with ``write_thumbnails`` rendering a small image per asset, redone only when
  the asset's YAML changes. ``tests/test_describe.py`` checks the committed text is current.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import roots
from .core.node import SceneNode
from .layout.schema import (
    ANCHORS_DOC,
    ASSET_KEYS,
    PART_KEYS,
    PLACEMENT_KEYS,
    PRIMITIVES,
    SCENE_KEYS,
    PLACE_KEYS,
    SLOT_KEYS,
    Key,
)
from .layout.yaml_utils import safe_load, safe_load_path

ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"
FORMAT = "geogen-describe"
VERSION = 1

# Asset libraries in subdirectories; their assets are placed as `<dir>/<name>.yaml`.
ASSET_SUBDIRS = ("vehicles", "characters")

# Catalogue sections, by the head of an asset's first recognised tag.
CATEGORIES = {
    "Furniture": ("furniture", "seat", "bed", "desk", "table", "shelf", "comfy", "appliance", "electronics"),
    "Bathroom": ("bathroom",),
    "Decor and lighting": ("decor", "light"),
    "Shops and restaurants": ("shop", "kitchen", "dining", "bar", "cook", "host", "staff"),
    "Buildings and openings": ("building", "architecture", "structure", "door", "window", "room"),
    "Vehicles": ("vehicle",),
    "Nature": ("vegetation", "terrain", "water"),
    "Street and park": ("street", "street_furniture", "park", "picnic", "bus", "travel", "outside"),
    "Characters": ("character",),
}
OTHER = "Other"
# Untagged assets, by file stem.
CATEGORY_OF = {
    "Buildings and openings": ("house_simple", "house_peaked", "room", "shop", "staircase", "handrail", "awning",
                               "fascia_sign", "hotel", "hotel_floor", "hotel_lobby", "hotel_suite",
                               "restaurant_building", "lighthouse"),
    "Street and park": ("road", "road_section", "sidewalk", "plaza", "ground", "signal_head", "crossing_barrier",
                        "street_lamp", "fire_hydrant", "mailbox", "trashcan", "bench", "bus_shelter", "signpost",
                        "info_plinth", "portal", "travel_door", "park_gate", "park_grounds", "playground", "bandstand",
                        "picnic_table", "jetty", "rowing_boat", "lettering"),
    "Nature": ("bush", "maple_tree", "pine_tree", "palm_tree", "rock_small", "rock_medium", "rock_large", "pond",
               "island_terrain", "island_grounds"),
    "Characters": ("humanoid", "capsule"),
    "Furniture": ("chair", "table", "table_setting"),
}

# Data files (`kind:`): what they are and how a scene uses them. Keys come from the loaders.
KINDS: dict[str, dict[str, str]] = {
    "npc": {"doc": "An NPC: body, needs, preferences, activities and daily routine; the runtime interprets it.",
            "use": "place: {resident: {npc: npcs/resident.yaml, on: house.floor, home: house.floor, seed: 1}}",
            "keys": "geogen.npc:NPC_KEYS"},
    "npc_actions": {"doc": "The actions NPCs run at affordances, as steps (go_to, face, pose, wait, use, play, vanish).",
                    "use": "referenced by affordances: {type: sit, action: sit}", "keys": "geogen.npc:STEP_KEYS"},
    "archetype": {"doc": "A seeded distribution over a body's params: each seed is a different person.",
                  "use": "place: {anna: {archetype: characters/archetypes/young_woman.yaml, seed: 3}}",
                  "keys": "geogen.characters:ARCHETYPE_KEYS"},
    "garment": {"doc": "A piece of clothing fitted to the humanoid; outfits are picked with its `outfit` param.",
                "use": "place: {x: {asset: characters/humanoid.yaml, params: {outfit: dress}}}",
                "keys": "geogen.generators.humanoid:GARMENT_KEYS"},
    "traffic": {"doc": "A moving fleet: weighted vehicles, count, driving behaviour, turn weights.",
                "use": "place: {cars: {traffic: traffic/town.yaml, seed: 1, count: 10}}",
                "keys": "geogen.traffic:FLEET_KEYS"},
    "train": {"doc": "A train: consist, speed and timetable, run on one of the scene's `railways:`.",
              "use": "place: {train: {train: trains/commuter.yaml, railway: main}}", "keys": "geogen.trains:TRAIN_KEYS"},
    "item": {"doc": "A spawned item: the body a source copies while the game runs, its mode and the data it carries.",
             "use": "source: {items: items/marble.yaml, rate: 1}   (an asset key; see geogen.flows)",
             "keys": "geogen.flows:ITEM_KEYS"},
    "waves": {"doc": "Waves a source sends in turn: each a list of item groups (count, gap) after a delay.",
              "use": "source: {waves: waves/raid.yaml}   (an asset key; see geogen.flows)",
              "keys": "geogen.flows:WAVES_KEYS"},
    "quest": {"doc": "A quest: objectives done by conditions or events, compiled to actor state and rules.",
              "use": "quests: [quests/lighthouse_gallery.yaml]   (a scene key; see docs/quests.md)",
              "keys": "geogen.quests:QUEST_KEYS"},
    "package": {"doc": "A content package: state, rules and quests that span the scenes it names.",
                "use": "packages: [name]   (in runtime_scenes.yaml; see docs/rules.md, Content packages)",
                "keys": "geogen.content:PACKAGE_KEYS"},
    "skeleton": {"doc": "The humanoid skeleton: params, bone heads and landmarks.",
                 "use": "body: {skeleton: skeletons/humanoid.yaml}", "keys": ""},
    "player_spec": {"doc": "Player-scale constraints (capsule, eye and step height, door and corridor widths).",
                    "use": "read by layout QA, the furnishing solver and the Godot runtime", "keys": ""},
    "scene_catalogue": {"doc": "The Godot runtime's scenes (showcase | test) and its default.",
                        "use": "python -m geogen.main --export-catalogue", "keys": ""},
}

THUMB = 256
THUMB_VERSION = 1


# ---- reading the YAML ------------------------------------------------------------------------

def header_comment(text: str) -> str:
    """The comment block at the top of a YAML file, without the ``#``s."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            lines.append(line[2:] if line.startswith("# ") else line[1:])
        elif line.strip() or lines:
            break
    return "\n".join(lines).strip("\n")


def summary_of(description: str) -> str:
    """First sentence of the header, on one line."""
    lines = [line.strip() for line in description.split("\n\n")[0].splitlines() if not line.startswith("  ")]
    text = ""
    for line in lines:
        # Old headers are a title line followed by notes: stop where a new sentence starts.
        if text and (text[-1] in ".:" or not (line[:1].islower() or line[:1] in "(0123456789")):
            break
        text = f"{text} {line}".strip()
    match = re.search(r"\.(?=\s+[A-Z(]|$)", text)
    text = (text[:match.start()] if match else text).rstrip(":.")
    return text + "." if text else ""


def param_comments(text: str) -> dict[str, str]:
    """Trailing comments of one-line ``params:`` entries (``width: {default: 2}   # metres``)."""
    out: dict[str, str] = {}
    inside = False
    for line in text.splitlines():
        if re.match(r"^params:\s*(#.*)?$", line):
            inside = True
        elif inside and line[:1].strip() and not line.startswith("#"):
            break
        elif inside:
            match = re.match(r"^\s{2}(\w+):[^#]*#\s*([^#]+)", line)
            if match:
                out[match.group(1)] = match.group(2).strip()
    return out


def _num(value: Any) -> Any:
    """Numbers rounded to mm (integers stay integers); everything else unchanged."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return value
    rounded = round(float(value), 3)
    return int(rounded) if rounded == int(rounded) else rounded


def _clean(value: Any) -> Any:
    """JSON-ready copy: numpy -> lists, floats rounded."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    return _num(value)


def _params(data: dict[str, Any], text: str) -> dict[str, dict[str, Any]]:
    comments = param_comments(text)
    out = {}
    for name, spec in (data.get("params") or {}).items():
        spec = spec if isinstance(spec, dict) else {"default": spec}
        entry: dict[str, Any] = {"default": _num(spec.get("default"))}
        if "choices" in spec:
            entry["choices"] = list(spec["choices"])
        if spec.get("type"):
            entry["type"] = spec["type"]
        if name in comments:
            entry["doc"] = comments[name]
        out[name] = entry
    return out


def asset_files(assets_dir: Path = ASSETS_DIR) -> dict[str, tuple[str, Path]]:
    """Every asset and scene file: ``{name: ("asset" | "scene", path)}`` (data files with a ``kind:`` are skipped)."""
    out: dict[str, tuple[str, Path]] = {}
    for path in roots.glob("*.yaml", assets_dir):
        data = safe_load_path(path)
        if isinstance(data, dict) and "kind" not in data:
            out[path.stem] = ("scene" if {"place", "compose", "city"} & set(data) else "asset", path)
    for sub in ASSET_SUBDIRS:
        for path in roots.glob(f"{sub}/*.yaml", assets_dir):
            data = safe_load_path(path)
            if isinstance(data, dict) and "kind" not in data:
                out.setdefault(path.stem, ("asset", path))
    for path in roots.glob("scenes/*.yaml", assets_dir):
        out.setdefault(path.stem, ("scene", path))
    for path in roots.glob(f"scenes/*/{roots.SCENE_FILE}", assets_dir):      # scene folders
        out.setdefault(path.parent.name, ("scene", path))
    return out


def category_of(name: str, tags: list[str], file: str) -> str:
    for category, stems in CATEGORY_OF.items():
        if name in stems:
            return category
    if file.startswith("characters/"):
        return "Characters"
    for tag in tags:
        head = str(tag).split(".")[0]
        for category, heads in CATEGORIES.items():
            if head in heads:
                return category
    return OTHER


def static_entry(name: str, kind: str, path: Path, assets_dir: Path = ASSETS_DIR) -> dict[str, Any]:
    """What the YAML alone says about an asset or scene."""
    text = path.read_text()
    data = safe_load(text) or {}
    file = roots.relative(path, assets_dir)
    description = header_comment(text)
    tags = [str(t) for t in data.get("tags") or []]
    entry: dict[str, Any] = {
        "name": name, "type": kind, "file": file, "summary": summary_of(description), "description": description,
        "tags": tags, "params": _params(data, text),
    }
    if kind == "scene":
        entry["category"] = "Scenes"
        placements = {**(data.get("compose") or {}), **(data.get("place") or {})}
        entry["places"] = {
            str(n): next((f"{{{k}: {d[k]}}}" for k in ("asset", "scene", "npc", "archetype", "traffic", "train")
                          if isinstance(d, dict) and k in d), "?")
            for n, d in placements.items()}
        entry["spawns"] = sorted((data.get("spawns") or {}))
        entry["slots"] = sorted((data.get("slots") or {}))
        entry["features"] = [k for k in ("city", "routes", "railways") if k in data]
        return entry
    entry["category"] = category_of(name, tags, file)
    entry["built_from"] = next((k for k in ("floorplan", "building", "body", "room") if k in data), "parts")
    if data.get("origin"):
        entry["origin"] = data["origin"]
    entry["presets"] = {str(k): _clean(v or {}) for k, v in (data.get("presets") or {}).items()}
    entry["parts"] = {str(n): str(p.get("primitive", "?")) for n, p in (data.get("parts") or {}).items()
                      if isinstance(p, dict)}
    return entry


# ---- what the built asset knows ---------------------------------------------------------------

def _bounds(root: SceneNode) -> tuple[np.ndarray, np.ndarray, int] | None:
    inv = np.linalg.inv(root.world_transform())
    lo, hi, triangles = None, None, 0
    for _, mesh in root.iter_meshes():
        if not len(mesh.vertices):
            continue
        pts = mesh.vertices @ inv[:3, :3].T + inv[:3, 3]      # iter_meshes yields world-space meshes
        lo = pts.min(axis=0) if lo is None else np.minimum(lo, pts.min(axis=0))
        hi = pts.max(axis=0) if hi is None else np.maximum(hi, pts.max(axis=0))
        triangles += len(mesh.faces)
    return None if lo is None else (lo, hi, triangles)


def _direction(normal: np.ndarray) -> str:
    """``+Z`` for axis-aligned vectors, else the vector."""
    for axis, label in enumerate("XYZ"):
        if abs(abs(float(normal[axis])) - 1.0) < 1e-3:
            return ("+" if normal[axis] > 0 else "-") + label
    return str(_clean(normal))


def built_entry(node: SceneNode) -> dict[str, Any]:
    """What only the built asset knows: extent, attachment positions, surfaces, interactions, meta."""
    out: dict[str, Any] = {}
    if getattr(node, "size", None) is not None:
        out["size"] = _clean(node.size)
    box = _bounds(node)
    if box is not None:
        out["bounds"] = {"min": _clean(box[0]), "max": _clean(box[1])}
        out["triangles"] = int(box[2])
    if node.host_cutters:
        out["opening"] = True
    inv = np.linalg.inv(node.world_transform())
    attachments = {}
    for name, point in node.attachments.items():
        transform = node.get_attachment(name)
        entry: dict[str, Any] = {}
        if transform is not None:
            entry["position"] = _clean((inv @ transform.to_matrix())[:3, 3])
        if getattr(point, "facing", None):
            entry["facing"] = point.facing
        attachments[name] = entry
    if attachments:
        out["attachments"] = attachments
    if node.surfaces:
        out["surfaces"] = {name: {"size": _clean([s.u_extent, s.v_extent]), "facing": _direction(s.normal)}
                           for name, s in node.surfaces.items()}
    interactions = {}
    for interaction in node.interactions:
        spec: dict[str, Any] = {
            "initial": interaction.initial,
            "states": {name: state.to_dict() for name, state in interaction.states.items()},
            "targets": [n.name for n in interaction.targets],
            "motions": [{"type": m.kind, "parts": [p.name for p in m.parts],
                         "values": _clean({s: interaction.value(m, s) for s in interaction.states})}
                        for m in interaction.motions],
        }
        if interaction.lock:
            spec["lock"] = dict(interaction.lock)
        interactions[interaction.name] = spec
    if interactions:
        out["interactions"] = interactions
    if node.meta.get("affordances"):
        out["affordances"] = [_clean({k: a[k] for k in ("type", "action", "position", "yaw", "height", "tags",
                                                         "advertises", "interaction", "slots") if k in a})
                              for a in node.meta["affordances"]]
    for key in ("light", "container", "clearance", "footprint", "portal", "travel"):
        if node.meta.get(key):
            out[key] = _clean(node.meta[key])
    named = {n.meta["place"]["id"]: _clean({**{k: v for k, v in n.meta["place"].items() if k != "id"},
                                            "position": [float(v) for v in n.transform.translation]})
             for n in node.children if n.meta.get("type") == "place"}
    if named:
        out["named_places"] = named
    vehicle = node.meta.get("vehicle")
    if vehicle:
        out["vehicle"] = _clean({k: vehicle[k] for k in ("class", "max_speed", "turn_radius", "accel", "decel",
                                                          "wheelbase", "track", "clearance") if k in vehicle})
    if node.meta.get("poses"):
        out["poses"] = sorted(node.meta["poses"])
    clips = sorted({c.name for n in node.iter_nodes() for c in n.clips})
    if clips:
        out["clips"] = clips
    rooms = [{"name": n.name, "type": next((t.split(".", 1)[1] for t in n.tags if t.startswith("room.")), "")}
             for n in node.iter_nodes() if "room" in n.tags]
    if rooms:
        out["rooms"] = rooms
    return out


def example_placement(entry: dict[str, Any]) -> str:
    """A placement line for a scene's ``place:`` showing the asset's main options."""
    bits = [f"asset: {entry['file']}"]
    if entry.get("opening"):
        bits += ["on: <object>.<wall surface>", "at: {u: 0.5, v: {abs: 0}}"]
    params = []
    presets = [p for p, values in entry.get("presets", {}).items() if values] or list(entry.get("presets", {}))
    if presets:
        params.append(f"preset: {presets[0]}")
    # Choice params first: they are what a caller most often picks.
    for name, spec in sorted(entry.get("params", {}).items(), key=lambda item: "choices" not in item[1]):
        if len(params) >= 2:
            break
        if not any(name in entry["presets"].get(p, {}) for p in presets[:1]):
            value = spec["choices"][-1] if spec.get("choices") else spec["default"]
            params.append(f"{name}: {value}")
    if params:
        bits.append("params: {" + ", ".join(params) + "}")
    for name, interaction in entry.get("interactions", {}).items():
        other = next((s for s in interaction["states"] if s != interaction["initial"]), None)
        if other:
            bits.append(f"state: {other}" if len(entry["interactions"]) == 1 else f"state: {{{name}: {other}}}")
            break
    return f"{entry['name']}: {{{', '.join(bits)}}}"


def describe_asset(static: dict[str, Any], node: SceneNode | None) -> dict[str, Any]:
    """The full entry for an asset: its YAML facts plus, given the built node, the resolved ones."""
    entry = dict(static)
    if node is not None:
        entry.update(built_entry(node))
    if entry["type"] == "asset":
        entry["example"] = example_placement(entry)
    return entry


def load_asset(entry: dict[str, Any], assets_dir: Path = ASSETS_DIR) -> SceneNode:
    """Build an asset with its default params."""
    from .layout import LayoutLoader

    return LayoutLoader().load(roots.find(entry["file"], assets_dir))


# ---- the rest of the vocabulary ---------------------------------------------------------------

def _key(key: Key, default: Any = None) -> dict[str, Any]:
    value = key.default if key.default is not None else default
    out: dict[str, Any] = {"type": key.type, "doc": key.doc}
    if value is not None and value != "":
        out["default"] = _clean(value)
    return out


def _generator_defaults(name: str) -> dict[str, Any]:
    """Defaults of the generator dataclass behind a primitive (what the YAML keys fall back to)."""
    import dataclasses

    from .layout.loader import PRIMITIVE_REGISTRY

    cls = PRIMITIVE_REGISTRY.get(name)
    if cls is None or not dataclasses.is_dataclass(cls):
        return {}
    return {f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}


def primitives() -> dict[str, dict[str, Any]]:
    out = {}
    for name, prim in PRIMITIVES.items():
        defaults = _generator_defaults(name)
        out[name] = {"doc": prim.doc, "size": prim.size, "example": prim.example,
                     "keys": {k: _key(v, defaults.get(k)) for k, v in prim.keys.items()}}
    return out


def _kind_keys(spec: str) -> list[str]:
    if not spec:
        return []
    import importlib

    module, _, attr = spec.partition(":")
    return sorted(set(getattr(importlib.import_module(module), attr)) - {"kind", "version"})


def kinds(assets_dir: Path = ASSETS_DIR) -> dict[str, dict[str, Any]]:
    """Data files by ``kind:``, with the keys their loaders accept."""
    out: dict[str, dict[str, Any]] = {}
    for path in roots.glob("**/*.yaml", assets_dir):
        text = path.read_text()
        match = re.search(r"^kind:\s*(\w+)", text, re.M)
        if not match:
            continue
        kind = match.group(1)
        known = KINDS.get(kind, {})
        entry = out.setdefault(kind, {"doc": known.get("doc", ""), "use": known.get("use", ""),
                                      "keys": _kind_keys(known.get("keys", "")), "files": {}})
        data = safe_load(text) or {}
        if not entry["keys"]:
            entry["keys"] = sorted(set(entry["keys"]) | (set(data) - {"kind", "version"}))
        entry["files"][roots.relative(path, assets_dir)] = (
            summary_of(header_comment(text)) or str(data.get("description", "")))
    return dict(sorted(out.items()))


def room_types(assets_dir: Path = ASSETS_DIR) -> dict[str, dict[str, Any]]:
    """Room archetypes the furnishing solver knows (a floor-plan room's ``type:``)."""
    out = {}
    for path in roots.glob("room_types/*.yaml", assets_dir):
        text = path.read_text()
        data = safe_load(text) or {}
        items = sorted({str(i["asset"]).removesuffix(".yaml") for i in (data.get("items") or {}).values()
                        if isinstance(i, dict) and "asset" in i})
        out[path.stem] = {"summary": summary_of(header_comment(text)), "items": items}
    return out


def materials(assets_dir: Path = ASSETS_DIR) -> dict[str, str]:
    """Material names with their texture type."""
    out = {}
    for path in roots.glob("materials/*.yaml", assets_dir):
        data = safe_load_path(path) or {}
        out[path.stem] = str((data.get("texture") or {}).get("type", ""))
    return out


def catalogue(assets_dir: Path = ASSETS_DIR) -> dict[str, Any]:
    """Everything, from the YAML alone (no asset is built)."""
    entries = {name: static_entry(name, kind, path, assets_dir) for name, (kind, path) in asset_files(assets_dir).items()}
    try:
        from .catalogue import load_catalogue

        for name, item in load_catalogue().entries.items():
            if name in entries:
                entries[name]["runtime"] = item.group
                if not entries[name]["summary"]:
                    entries[name]["summary"] = item.description
    except (OSError, ValueError):
        pass
    return {
        "format": FORMAT, "version": VERSION,
        "primitives": primitives(),
        "part_keys": {k: _key(v) for k, v in PART_KEYS.items()},
        "asset_keys": {k: v for k, v in ASSET_KEYS.items() if not k.startswith("_")},
        "scene_keys": dict(SCENE_KEYS), "slot_keys": dict(SLOT_KEYS), "placement_keys": dict(PLACEMENT_KEYS),
        "place_keys": dict(PLACE_KEYS),
        "anchors": ANCHORS_DOC,
        "assets": {n: e for n, e in entries.items() if e["type"] == "asset"},
        "scenes": {n: e for n, e in entries.items() if e["type"] == "scene"},
        "kinds": kinds(assets_dir), "room_types": room_types(assets_dir), "materials": materials(assets_dir),
    }


def by_category(assets: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    order = [*CATEGORIES, OTHER]
    groups: dict[str, list[dict[str, Any]]] = {c: [] for c in order}
    for entry in assets.values():
        groups[entry["category"]].append(entry)
    return {c: sorted(g, key=lambda e: e["name"]) for c, g in groups.items() if g}


# ---- text --------------------------------------------------------------------------------------

GUIDE = """\
A scene (assets/scenes/NAME.yaml) places objects; an asset (assets/NAME.yaml) is built from primitive parts.
Metres, +Y up, +Z front / north, +X east. `{expr}` anywhere interpolates params (+ - * /, min, max, sind, ...;
comparisons, and / or / not and `a if c else b`, e.g. when: "{storeys > 1 and style == 'brick'}").

  name: garage_scene
  size: [14, 5, 14]
  slots:
    drive: {position: [4, 0, 4], facing: south}
  place:
    house: {asset: house_peaked.yaml}
    door:  {asset: door.yaml, on: house.front_wall, at: {u: 0.5, v: {abs: 0}}, params: {width: 1.1}, state: open}
    car:   {asset: vehicles/car.yaml, slot: drive, params: {preset: estate, paint: car_red}}

Check it: python -m geogen.main -s garage_scene -r out.png --views   (and --inspect for positions and QA)."""


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":")).replace('"', "")
    return str(value)


def _param_text(name: str, spec: dict[str, Any]) -> str:
    text = f"{name}={_value(spec['default'])}"
    if spec.get("choices"):
        text += f" ({' | '.join(map(str, spec['choices']))})"
    elif spec.get("type"):
        text += f" ({spec['type']})"
    return text


def _wrap(prefix: str, items: list[str], width: int = 118, sep: str = ", ") -> list[str]:
    """``prefix`` + items joined by ``sep``, wrapped with a hanging indent."""
    lines, line, empty = [], prefix, True
    indent = " " * min(len(prefix), 6)
    for i, item in enumerate(items):
        piece = item + (sep if i < len(items) - 1 else "")
        if len(line) + len(piece) > width and not empty:
            lines.append(line.rstrip())
            line = indent
        line += piece
        empty = False
    lines.append(line.rstrip())
    return lines


def _keys_text(keys: dict[str, Any], indent: str = "  ") -> list[str]:
    lines = []
    width = max((len(k) for k in keys), default=0)
    for name, spec in keys.items():
        if isinstance(spec, str):
            lines.append(f"{indent}{name:<{width}}  {spec}")
            continue
        default = f" = {_value(spec['default'])}" if "default" in spec else ""
        lines.append(f"{indent}{name:<{width}}  {spec['type']}{default}: {spec['doc']}")
    return lines


def primitive_text(name: str, prim: dict[str, Any]) -> list[str]:
    lines = [f"{name} (primitive; size {prim['size']}): {prim['doc']}"]
    lines += _keys_text(prim["keys"])
    lines.append(f"  example: {prim['example']}")
    return lines


def list_text(cat: dict[str, Any]) -> str:
    """The whole vocabulary, compact enough to paste into a prompt."""
    lines = ["# geogen vocabulary (python -m geogen.main --describe NAME: one primitive, asset, scene or kind in full)",
             "", GUIDE, "", "## Primitives (asset parts: `parts: {name: {primitive: ..., size: [x, y, z], ...}}`)"]
    for name, prim in cat["primitives"].items():
        keys = [f"{k}={_value(s['default'])}" if "default" in s else k for k, s in prim["keys"].items()]
        head = f"  {name}: {prim['doc'].split('. ')[0].rstrip('.')}."
        lines += _wrap(head + " Keys: ", keys) if keys else [head]
    lines += ["", "## Part keys (every primitive)"] + _keys_text(cat["part_keys"])
    lines += ["  anchors: " + cat["anchors"]]
    lines += ["", "## Asset keys"] + _keys_text(cat["asset_keys"])
    lines += ["", "## Scene keys"] + _keys_text(cat["scene_keys"])
    lines += ["", "## Slot keys"] + _keys_text(cat["slot_keys"])
    lines += ["", "## Placement keys (`place: {name: {...}}`)"] + _keys_text(cat["placement_keys"])
    lines += ["", "## Place keys (`places:`)"] + _keys_text(cat["place_keys"])
    lines += ["", "## Assets (`asset: <file>`; params with defaults)"]
    for category, entries in by_category(cat["assets"]).items():
        lines.append(f"[{category}]")
        for e in entries:
            params = [_param_text(n, s) for n, s in e["params"].items()]
            presets = f" Presets: {', '.join(e['presets'])}." if e.get("presets") else ""
            head = f"  {e['file']}: {e['summary']}{presets}"
            lines += _wrap(head + (" Params: " if params else ""), params) if params else [head]
    lines += ["", "## Scenes (`-s NAME`; nest with `scene: <file>`)"]
    for e in cat["scenes"].values():
        params = [_param_text(n, s) for n, s in e["params"].items()]
        head = f"  {e['file']}: {e['summary']}"
        lines += _wrap(head + (" Params: " if params else ""), params) if params else [head]
    lines += ["", "## Kinds (data files)"]
    for kind, spec in cat["kinds"].items():
        lines.append(f"  {kind}: {spec['doc']} Use: {spec['use']}")
        lines += _wrap("    files: ", [f.removesuffix(".yaml") for f in spec["files"]])
    lines += ["", "## Room types (floor-plan room `type:`, furnished by `furnish: true`)"]
    lines += _wrap("  ", list(cat["room_types"]))
    lines += ["", "## Materials (`material: <name>`)"]
    lines += _wrap("  ", list(cat["materials"]))
    return "\n".join(lines)


def _vec(values: Any) -> str:
    return "(" + ", ".join(_value(v) for v in values) + ")"


def _interaction_text(name: str, spec: dict[str, Any]) -> str:
    states = []
    for state, s in spec["states"].items():
        text = state + ("*" if state == spec["initial"] else "")
        if s.get("next"):
            text += f" -> {s['next']}"
        if s.get("then"):
            text += f" (then {s['then']}" + (f" after {_value(s['after'])} s" if "after" in s else "") + ")"
        states.append(text)
    verb = {"rotate": "rotates", "translate": "translates", "visible": "shows/hides"}
    moves = "; ".join(f"{verb.get(m['type'], m['type'])} {', '.join(m['parts'])}" for m in spec["motions"])
    lock = f"; lock {spec['lock'].get('key')}" if spec.get("lock") else ""
    return f"{name}: {', '.join(states)} (* initial{'; ' + moves if moves else ''}{lock})"


def _affordance_text(a: dict[str, Any]) -> str:
    text = f"{a['type']} at {_vec(a['position'])}"
    if a.get("height") is not None:
        text += f" height {_value(a['height'])}"
    if a.get("tags"):
        text += f" [{', '.join(a['tags'])}]"
    return text


def facts(entry: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """An entry's sections as (label, lines): shared by the text and markdown renderings."""
    out: list[tuple[str, list[str]]] = []
    if "size" in entry or "bounds" in entry:
        bits = []
        if "size" in entry:
            bits.append(" x ".join(_value(v) for v in entry["size"]) + " m (x, y, z)")
        if "bounds" in entry:
            bits.append(f"geometry {_vec(entry['bounds']['min'])} to {_vec(entry['bounds']['max'])} from the origin")
            bits.append(f"{entry['triangles']} triangles")
        out.append(("size", ["; ".join(bits)]))
    if entry.get("opening"):
        out.append(("opening", ["cuts its own hole in the surface it is placed `on` (origin: bottom centre of the "
                                "opening on the wall face, +Z out of the wall)"]))
    if entry.get("presets"):
        out.append(("presets", [f"{name}: " + (", ".join(f"{k}={_value(v)}" for k, v in values.items()) or "defaults")
                                for name, values in entry["presets"].items()]))
    if entry.get("parts"):
        out.append(("parts", ["; ".join(f"{n} ({p})" for n, p in entry["parts"].items())]))
    if entry.get("rooms"):
        out.append(("rooms", [", ".join(f"{r['name']} ({r['type']})" if r["type"] else r["name"]
                                        for r in entry["rooms"])]))
    if entry.get("attachments"):
        out.append(("attachments", [f"{n} at {_vec(a['position'])}" + (f" facing {a['facing']}" if a.get("facing") else "")
                                    if "position" in a else n for n, a in entry["attachments"].items()]))
    if entry.get("surfaces"):
        out.append(("surfaces", [f"{n}: {_value(s['size'][0])} x {_value(s['size'][1])} m facing {s['facing']}"
                                 for n, s in entry["surfaces"].items()]))
    if entry.get("interactions"):
        out.append(("interactions", [_interaction_text(n, s) for n, s in entry["interactions"].items()]))
    if entry.get("affordances"):
        out.append(("affordances", [_affordance_text(a) for a in entry["affordances"]]))
    for key in ("light", "container", "clearance", "portal", "travel", "vehicle"):
        if entry.get(key):
            out.append((key, [", ".join(f"{k} {_value(v)}" for k, v in entry[key].items())]))
    if entry.get("named_places"):
        out.append(("named places", [
            f"{n}: " + (f"polygon of {len(p['polygon'])} points, {_value(p['height'])} m high" if "polygon" in p
                        else "box " + " x ".join(_value(v) for v in p["size"]) + " m") + f" at {_vec(p['position'])}"
            for n, p in entry["named_places"].items()]))
    if entry.get("poses"):
        out.append(("poses", [", ".join(entry["poses"])]))
    if entry.get("clips"):
        out.append(("clips", [", ".join(entry["clips"])]))
    if entry.get("places"):
        out.append(("places", [f"{n}: {ref}" for n, ref in entry["places"].items()]))
    for key in ("slots", "spawns", "features"):
        if entry.get(key):
            out.append((key, [", ".join(entry[key])]))
    return out


MAX_LINES = 40      # long sections (a hotel's surfaces) are cut in the text output


def entry_text(entry: dict[str, Any]) -> str:
    tags = f"  [{', '.join(entry['tags'])}]" if entry["tags"] else ""
    lines = [f"{entry['name']} ({entry['type']}: {entry['file']}){tags}"]
    lines += ["  " + line for line in entry["description"].splitlines()]
    if entry["params"]:
        lines.append("params:")
        width = min(28, max(len(_param_text(n, s)) for n, s in entry["params"].items()))
        lines += [f"  {_param_text(n, s):<{width}}  {s.get('doc', '')}".rstrip() for n, s in entry["params"].items()]
    for label, values in facts(entry):
        if len(values) == 1:
            lines += _wrap(f"{label}: ", values[0].split("; "), sep="; ")
            continue
        lines.append(f"{label}:")
        lines += [f"  {v}" for v in values[:MAX_LINES]]
        if len(values) > MAX_LINES:
            lines.append(f"  ... and {len(values) - MAX_LINES} more (--json lists them all)")
    if entry.get("example"):
        lines += ["place:", f"  {entry['example']}"]
    if entry["type"] == "scene":
        lines.append(f"resolved tree and QA: python -m geogen.main -s {entry['name']} --inspect")
    return "\n".join(lines)


def kind_text(kind: str, spec: dict[str, Any]) -> str:
    lines = [f"{kind} (kind): {spec['doc']}", f"  use: {spec['use']}"]
    lines += _wrap("  keys: ", spec["keys"])
    lines.append("  files:")
    lines += [f"    {f}: {doc}".rstrip(": ") for f, doc in spec["files"].items()]
    return "\n".join(lines)


def lookup(cat: dict[str, Any], name: str, assets_dir: Path = ASSETS_DIR,
           build: Callable[[dict[str, Any]], SceneNode | None] = load_asset) -> tuple[str, Any]:
    """Find ``name`` among primitives, assets, scenes (by name or file), kinds and kind files:
    ``(what, entry)``. Assets are built for the resolved facts."""
    stem = name.removesuffix(".yaml")
    if stem in cat["primitives"]:
        return "primitive", cat["primitives"][stem]
    for group in ("assets", "scenes"):
        entry = cat[group].get(stem.rsplit("/", 1)[-1])
        if entry is not None and stem in (entry["name"], entry["file"].removesuffix(".yaml")):
            return group[:-1], describe_asset(entry, build(entry) if group == "assets" else None)
    if stem in cat["kinds"]:
        return "kind", cat["kinds"][stem]
    path = roots.find(f"{stem}.yaml", assets_dir)
    if path.is_file():
        return "file", {"file": f"{stem}.yaml", "text": path.read_text(), "data": safe_load_path(path)}
    import difflib

    known = [*cat["primitives"], *cat["assets"], *cat["scenes"], *cat["kinds"]]
    hint = difflib.get_close_matches(stem, known, n=3, cutoff=0.5)
    raise KeyError(f"--describe: nothing named '{name}'." + (f" Did you mean {', '.join(hint)}?" if hint else ""))


def describe(names: list[str] | None = None, as_json: bool = False, assets_dir: Path = ASSETS_DIR) -> str:
    """The ``--describe`` output: everything (no names) or the named entries in full."""
    cat = catalogue(assets_dir)
    if not names:
        return json.dumps(cat, indent=2) if as_json else list_text(cat)
    found = [(name, *lookup(cat, name, assets_dir)) for name in names]
    if as_json:
        result = {name: (entry["data"] if what == "file" else entry) for name, what, entry in found}
        return json.dumps(result if len(found) > 1 else next(iter(result.values())), indent=2)
    blocks = []
    for name, what, entry in found:
        if what == "primitive":
            blocks.append("\n".join(primitive_text(name, entry)))
        elif what == "kind":
            blocks.append(kind_text(name, entry))
        elif what == "file":
            blocks.append(f"{entry['file']}\n{entry['text'].rstrip()}")
        else:
            blocks.append(entry_text(entry))
    return "\n\n".join(blocks)


# ---- markdown ----------------------------------------------------------------------------------

def _md(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|",
            *("| " + " | ".join(_md(c) for c in row) + " |" for row in rows), ""]


def _description_md(description: str) -> list[str]:
    """A YAML header comment as markdown: indented lines become code blocks."""
    lines: list[str] = []
    code = False
    for line in description.splitlines():
        indented = line.startswith("  ") and line.strip()
        if indented and not code:
            lines += ["", "```yaml"]
        elif code and not indented:
            lines += ["```", ""]
        code = bool(indented)
        lines.append(line[2:] if code else line)
    if code:
        lines.append("```")
    tidy = [line for i, line in enumerate(lines) if line or (i and lines[i - 1])]
    return [*tidy, ""] if tidy else []


def _keys_table(keys: dict[str, Any]) -> list[str]:
    if all(isinstance(v, str) for v in keys.values()):
        return _table(["Key", "Meaning"], [[f"`{k}`", v] for k, v in keys.items()])
    return _table(["Key", "Type", "Default", "Meaning"],
                  [[f"`{k}`", s["type"], f"`{_value(s['default'])}`" if "default" in s else "", s["doc"]]
                   for k, s in keys.items()])


def asset_page(entry: dict[str, Any]) -> str:
    name = entry["name"]
    lines = [f"# {name}", "", f"`asset: {entry['file']}` · [source](../../../assets/{entry['file']})"
             + (" · " + " ".join(f"`{t}`" for t in entry["tags"]) if entry["tags"] else ""), "",
             f"<img src=\"../img/{name}.png\" width=\"{THUMB}\" alt=\"{name}\"> "
             f"<img src=\"../img/{name}_views.png\" width=\"{2 * THUMB}\" alt=\"{name}: front and side\">", ""]
    lines += _description_md(entry["description"])
    if entry.get("example"):
        lines += ["```yaml", "place:", f"  {entry['example']}", "```", ""]
    if entry["params"]:
        lines += ["## Params", ""]
        lines += _table(["Param", "Default", "Choices", "Notes"],
                        [[f"`{n}`", f"`{_value(s['default'])}`", ", ".join(map(str, s.get("choices", [])))
                          or s.get("type", ""), s.get("doc", "")] for n, s in entry["params"].items()])
    for label, values in facts(entry):
        lines += [f"## {label.capitalize()}", ""]
        if label == "surfaces":
            lines += _table(["Surface", "Size (u x v, m)", "Facing"],
                            [[f"`{n}`", f"{_value(s['size'][0])} x {_value(s['size'][1])}", s["facing"]]
                             for n, s in entry["surfaces"].items()])
        elif label == "attachments":
            lines += _table(["Attachment", "Position", "Facing"],
                            [[f"`{n}`", _vec(a["position"]) if "position" in a else "", a.get("facing", "")]
                             for n, a in entry["attachments"].items()])
        elif label == "parts":
            lines += [", ".join(f"`{n}` ({p})" for n, p in entry["parts"].items()), ""]
        elif len(values) == 1:
            lines += [values[0], ""]
        else:
            lines += [f"- {v}" for v in values] + [""]
    return "\n".join(lines).rstrip() + "\n"


def _grid(entries: list[dict[str, Any]], columns: int = 5) -> list[str]:
    """A thumbnail grid (HTML table: GitHub markdown has no image grid)."""
    lines = ["<table>"]
    for i in range(0, len(entries), columns):
        lines.append("<tr>")
        for e in entries[i:i + columns]:
            lines.append(f"<td align=\"center\"><a href=\"assets/{e['name']}.md\">"
                         f"<img src=\"img/{e['name']}.png\" width=\"150\" alt=\"{e['name']}\"><br>{e['name']}</a></td>")
        lines.append("</tr>")
    return [*lines, "</table>", ""]


def index_page(cat: dict[str, Any]) -> str:
    lines = ["# Geogen catalogue", "",
             "Everything a scene can be built from. Generated by `docs/make_catalogue.sh` "
             "(`python -m geogen.main --describe --markdown docs/catalogue`); don't edit by hand. "
             "The same data prints as text or JSON with `python -m geogen.main --describe [NAME] [--json]`.", "",
             "- [Primitives](primitives.md): the shapes asset parts are made of, and the keys every part takes",
             "- [Vocabulary](vocabulary.md): asset, scene, slot and placement keys, data kinds, room types, materials",
             "- [Scenes](scenes.md): the composed scenes (`-s NAME`)", "",
             "```yaml", *(line[2:] for line in GUIDE.splitlines() if line.startswith("  ")), "```", ""]
    for category, entries in by_category(cat["assets"]).items():
        lines += [f"## {category}", ""] + _grid(entries)
        lines += _table(["Asset", "What it is", "Params"],
                        [[f"[{e['name']}](assets/{e['name']}.md)", e["summary"],
                          ", ".join(f"`{n}`" for n in e["params"])] for e in entries])
    return "\n".join(lines).rstrip() + "\n"


def primitives_page(cat: dict[str, Any]) -> str:
    lines = ["# Primitives", "",
             "An asset's `parts:` are primitives. Each takes the common part keys below plus its own.", "",
             "```yaml", "name: stool", "size: [0.4, 0.45, 0.4]        # metres; part sizes are fractions of this",
             "parts:", "  seat: {primitive: cylinder, size: [1, 0.1, 1], anchor: top_center, offset: [0, -0.05, 0], "
             "material: wood}", "  leg:  {primitive: cylinder, size: [0.15, 0.9, 0.15], attach_to: seat, at: bottom, "
             "from: top, material: metal}", "```", "", "## Part keys", ""]
    lines += _keys_table(cat["part_keys"])
    lines += [f"Anchors: {cat['anchors']}.", ""]
    for name, prim in cat["primitives"].items():
        lines += [f"## {name}", "", f"<img src=\"img/primitives/{name}.png\" width=\"{THUMB}\" alt=\"{name}\">", "",
                  prim["doc"] + (" `size` is optional (taken from the geometry)." if prim["size"] == "optional" else ""),
                  ""]
        if prim["keys"]:
            lines += _keys_table(prim["keys"])
        lines += ["```yaml", prim["example"], "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def vocabulary_page(cat: dict[str, Any]) -> str:
    lines = ["# Vocabulary", ""]
    for title, key in (("Asset keys", "asset_keys"), ("Scene keys", "scene_keys"), ("Slot keys", "slot_keys"),
                       ("Placement keys", "placement_keys"), ("Place keys", "place_keys")):
        lines += [f"## {title}", ""] + _keys_table(cat[key])
    lines += ["## Kinds", "", "Data files with a top-level `kind:`.", ""]
    for kind, spec in cat["kinds"].items():
        lines += [f"### {kind}", "", spec["doc"], "", f"Use: `{spec['use']}`", ""]
        if spec["keys"]:
            lines += ["Keys: " + ", ".join(f"`{k}`" for k in spec["keys"]), ""]
        lines += _table(["File", "What it is"], [[f"[{f}](../../assets/{f})", doc] for f, doc in spec["files"].items()])
    lines += ["## Room types", "", "A floor-plan room's `type:`; `furnish: true` fills the room from it.", ""]
    lines += _table(["Room type", "What it is", "Furnished with"],
                    [[f"`{n}`", r["summary"], ", ".join(r["items"])] for n, r in cat["room_types"].items()])
    lines += ["## Materials", "", "`material: <name>` on a part.", ""]
    lines += _table(["Material", "Texture"], [[f"`{n}`", t] for n, t in cat["materials"].items()])
    return "\n".join(lines).rstrip() + "\n"


def scenes_page(cat: dict[str, Any]) -> str:
    lines = ["# Scenes", "", "Composed scenes: view one with `python -m geogen.main -s NAME`, nest one with "
             "`scene: <file>`. `--inspect` prints the resolved tree.", ""]
    lines += _table(["Scene", "What it is", "Runtime"],
                    [[f"[{e['name']}](#{e['name']})", e["summary"], e.get("runtime", "")] for e in cat["scenes"].values()])
    for e in cat["scenes"].values():
        lines += [f"## {e['name']}", "", f"`scene: {e['file']}` · [source](../../assets/{e['file']})", ""]
        lines += _description_md(e["description"])
        if e["params"]:
            lines += _table(["Param", "Default", "Notes"],
                            [[f"`{n}`", f"`{_value(s['default'])}`", s.get("doc", "")] for n, s in e["params"].items()])
        for label, values in facts(e):
            lines += [f"**{label.capitalize()}**: " + ("; ".join(f"`{v}`" for v in values) if label == "places"
                                                       else values[0]), ""]
    return "\n".join(lines).rstrip() + "\n"


def static_pages(cat: dict[str, Any]) -> dict[str, str]:
    """The pages that need no asset built: ``{relative path: markdown}``."""
    return {"README.md": index_page(cat), "primitives.md": primitives_page(cat),
            "vocabulary.md": vocabulary_page(cat), "scenes.md": scenes_page(cat)}


def asset_page_path(name: str) -> str:
    return f"assets/{name}.md"


# ---- thumbnails --------------------------------------------------------------------------------

def _thumbnail(root: SceneNode, views: list[str]):
    """``views`` side by side, each THUMB px square (rendered at twice that, then reduced)."""
    from PIL import Image

    from .render import RenderOptions, SceneRenderer

    renderer = SceneRenderer(root, RenderOptions(width=2 * THUMB, height=2 * THUMB))
    sheet = Image.new("RGB", (THUMB * len(views), THUMB))
    for i, view in enumerate(views):
        sheet.paste(renderer.render(view).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS), (i * THUMB, 0))
    # A palette keeps ~200 committed images small.
    return sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def primitive_node(name: str) -> SceneNode:
    """The primitive's ``example`` part in a 1 m container."""
    from .layout import LayoutLoader

    return LayoutLoader().load_string(f"name: {name}\nsize: [1, 1, 1]\nparts:\n  {name}: {PRIMITIVES[name].example}\n")


def write_thumbnails(out_dir: Path, cat: dict[str, Any], nodes: dict[str, SceneNode], force: bool = False,
                     assets_dir: Path = ASSETS_DIR, log: Callable[[str], None] = print) -> int:
    """Render ``img/<asset>.png`` (iso), ``img/<asset>_views.png`` (front, side) and
    ``img/primitives/<name>.png``: only those whose source changed since ``img/hashes.json``."""
    img = out_dir / "img"
    (img / "primitives").mkdir(parents=True, exist_ok=True)
    hashes_path = img / "hashes.json"
    hashes = json.loads(hashes_path.read_text()) if hashes_path.is_file() else {}
    jobs: list[tuple[str, str, list[tuple[Path, list[str]]], Callable[[], SceneNode]]] = []
    for name, entry in cat["assets"].items():
        digest = hashlib.sha1(roots.find(entry["file"], assets_dir).read_bytes() + bytes([THUMB_VERSION])).hexdigest()
        jobs.append((name, digest, [(img / f"{name}.png", ["iso"]), (img / f"{name}_views.png", ["front", "side"])],
                     lambda name=name: nodes[name]))
    for name, prim in cat["primitives"].items():
        digest = hashlib.sha1((prim["example"] + str(THUMB_VERSION)).encode()).hexdigest()
        jobs.append((f"primitives/{name}", digest, [(img / "primitives" / f"{name}.png", ["iso"])],
                     lambda name=name: primitive_node(name)))
    done = 0
    for key, digest, files, make in jobs:
        if not force and hashes.get(key) == digest and all(path.is_file() for path, _ in files):
            continue
        root = SceneNode("root")
        root.add_child(make().instance())
        for path, views in files:
            _thumbnail(root, views).save(path, optimize=True)
        hashes[key] = digest
        done += 1
        log(f"rendered {key}")
    known = {job[0] for job in jobs}
    hashes_path.write_text(json.dumps({k: v for k, v in sorted(hashes.items()) if k in known}, indent=1) + "\n")
    return done


def write_markdown(out_dir: str | Path, thumbnails: bool = True, force: bool = False,
                   assets_dir: Path = ASSETS_DIR, log: Callable[[str], None] = print) -> Path:
    """Write the whole site into ``out_dir`` (every asset is built once, with default params)."""
    out_dir = Path(out_dir)
    cat = catalogue(assets_dir)
    pages = static_pages(cat)
    nodes: dict[str, SceneNode] = {}
    for name, entry in cat["assets"].items():
        nodes[name] = load_asset(entry, assets_dir)
        pages[asset_page_path(name)] = asset_page(describe_asset(entry, nodes[name]))
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    for stale in (out_dir / "assets").glob("*.md"):
        if f"assets/{stale.name}" not in pages:
            stale.unlink()
    for rel, text in pages.items():
        (out_dir / rel).write_text(text)
    log(f"wrote {len(pages)} pages to {out_dir}")
    if thumbnails:
        count = write_thumbnails(out_dir, cat, nodes, force=force, assets_dir=assets_dir, log=log)
        for stale in (out_dir / "img").glob("*.png"):
            if stale.stem.removesuffix("_views") not in cat["assets"]:
                stale.unlink()
        log(f"rendered {count} thumbnails")
    return out_dir / "README.md"
