"""``--inspect``: a text (or JSON) dump of a resolved scene for checking it from numbers.

The tree shows the nodes that matter for placement: asset roots (``meta.source``), rooms,
storeys, doors, lights, switches, spawns, NPCs, vehicles and anything else tagged, with
world position, yaw and world bounds. Plain mesh parts are folded into their object's
line (``parts 12, 1.4k tris``) unless ``parts=True``. Siblings with identical subtrees
collapse to one line (``chair ×4``) listing where each copy stands.

After the tree come the checks (mesh validation, layout QA, affordance QA, coplanar faces,
object overlaps, traffic lanes) as one-line findings naming the nodes involved.
Renders are for appearance; this is for correctness.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core.node import SceneNode

# Tags (whole tag or its head) that mark a part of something rather than a thing.
PART_TAG_HEADS = {"floor", "ceiling", "wall", "trim", "threshold", "joint", "lining", "collider"}
PART_TAGS = {"door.leaf"}
PART_TYPES = {"collider", "room_volume"}
# Meta that makes an untagged node worth a line of its own.
SHOWN_META = {"source", "npc", "vehicle", "travel", "light", "switch", "room", "storey", "door",
              "spawn", "signal", "gate", "lift", "building", "lot"}
SUMMARISED_META = {"source", "collider", "walkable", "placed_by", "footprint", "clearance",
                   "furnish_report", "openings", "door_swings", "wall_inset", "clear_height",
                   "size", "type"}
MAX_POSITIONS = 12      # collapsed groups list each copy's position up to this many


@dataclass
class Entry:
    """One line of the tree: a shown node (or ``count`` identical siblings)."""

    node: SceneNode
    path: str
    world: np.ndarray
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    parts: int = 0
    tris: int = 0
    children: list[Entry] = field(default_factory=list)
    copies: list[Entry] = field(default_factory=list)   # the others when collapsed

    @property
    def count(self) -> int:
        return 1 + len(self.copies)


def _num(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _vec(v) -> str:
    return "[" + ", ".join(_num(float(x)) for x in v) + "]"


def _yaw(m: np.ndarray) -> float:
    return math.degrees(math.atan2(m[0, 2], m[2, 2])) % 360


def _is_shown(node: SceneNode) -> bool:
    if node.meta.get("type") in PART_TYPES:
        return False
    if node.meta.get("type") or node.interactions or SHOWN_META & set(node.meta):
        return True
    tags = [t for t in node.tags if t not in PART_TAGS and t.split(".")[0] not in PART_TAG_HEADS]
    return bool(tags)


def _mesh_box(node: SceneNode, world: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    key = id(node.mesh)
    if key not in cache:
        v = node.mesh.vertices
        cache[key] = (v.min(axis=0), v.max(axis=0))
    lo, hi = cache[key]
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = corners @ world[:3, :3].T + world[:3, 3]
    return w.min(axis=0), w.max(axis=0)


def build_tree(root: SceneNode, parts: bool = False) -> Entry:
    """The shown-node tree under ``root`` with world bounds and folded part counts."""
    cache: dict = {}

    def union(a, b):
        if a is None:
            return b
        return a if b is None else (np.minimum(a[0], b[0]), np.maximum(a[1], b[1]))

    def visit(node: SceneNode, world: np.ndarray, owner: Entry):
        world = world @ node.transform.to_matrix()
        entry = owner
        if parts or _is_shown(node):
            entry = Entry(node, f"{owner.path}/{node.name}" if owner.path else node.name, world)
            owner.children.append(entry)
        box = None
        if node.mesh is not None and len(node.mesh.faces) and node.meta.get("type") != "collider":
            box = _mesh_box(node, world, cache)
            entry.parts += 1
            entry.tris += len(node.mesh.faces)
        for child in node.children:
            box = union(box, visit(child, world, entry))
        if entry is not owner and box is not None:
            entry.lo, entry.hi = box
        return box

    top = Entry(root, "", root.world_transform())
    box = None
    if root.mesh is not None and len(root.mesh.faces):
        box = _mesh_box(root, top.world, cache)
        top.parts, top.tris = 1, len(root.mesh.faces)
    for child in root.children:
        box = union(box, visit(child, top.world, top))
    if box is not None:
        top.lo, top.hi = box
    _collapse(top)
    return top


def _base(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def _signature(e: Entry, origin: np.ndarray) -> tuple:
    rel = np.linalg.inv(origin) @ e.world
    return (_base(e.node.name), json.dumps(e.node.meta.get("source"), sort_keys=True, default=str),
            tuple(e.node.tags), e.parts, e.tris,
            tuple(np.round(rel[:3, 3], 2)) if e.world is not origin else (),
            tuple(_signature(c, origin) for c in e.children))


def _collapse(e: Entry) -> None:
    for child in e.children:
        _collapse(child)
    groups: dict[tuple, Entry] = {}
    kept = []
    for child in e.children:
        sig = _signature(child, child.world)
        first = groups.get(sig)
        if first is None:
            groups[sig] = child
            kept.append(child)
        else:
            first.copies.append(child)
    e.children = kept


def _meta_summary(node: SceneNode) -> list[str]:
    m = node.meta
    out = []
    if isinstance(m.get("room"), dict) and m.get("type") != "room_volume":
        out.append(f"room {m['room'].get('type', '')}".rstrip())
    if isinstance(m.get("light"), dict):
        light = m["light"]
        auto = f" auto {light['auto']}" if light.get("auto") else ""
        out.append(f"light {_num(float(light.get('energy', 0)))}e {_num(float(light.get('range', 0)))}m{auto}")
    if isinstance(m.get("switch"), dict):
        out.append(f"switch → {m['switch'].get('light')}")
    if isinstance(m.get("npc"), dict):
        npc = m["npc"]
        out.append(f"npc {npc.get('definition', '?')} seed {npc.get('seed', 0)}")
    if isinstance(m.get("door"), dict):
        door = m["door"]
        width, height = _num(float(door.get("width", 0))), _num(float(door.get("height", 0)))
        out.append(f"{door.get('style', 'door')} {width}×{height} m → {'/'.join(door.get('rooms', []))}")
    if isinstance(m.get("vehicle"), dict):
        out.append(f"vehicle {m['vehicle'].get('class', '')}")
    if isinstance(m.get("travel"), dict):
        t = m["travel"]
        out.append(f"travel → {t.get('scene')}:{t.get('spawn')} on {t.get('on')}")
    if m.get("type") == "spawn":
        out.append("spawn")
    if m.get("affordances"):
        kinds = Counter(a.get("type", "?") for a in m["affordances"] if isinstance(a, dict))
        out.append("affords " + ", ".join(f"{k}×{n}" if n > 1 else k for k, n in kinds.items()))
    if isinstance(m.get("container"), dict):
        out.append("container")
    if m.get("furnish_report"):
        out.append(f"unmet furnishing rules: {len(m['furnish_report'])}")
    handled = SUMMARISED_META | {"room", "light", "switch", "door", "npc", "vehicle", "travel", "affordances",
                                 "container", "interactions"}
    rest = sorted(k for k in m if k not in handled)
    if rest:
        out.append("meta " + ",".join(rest))
    return out


def _interactions(node: SceneNode) -> list[str]:
    return [f"{i.name}: {i.initial} [{'/'.join(i.states)}]" for i in node.interactions]


def _source(node: SceneNode) -> str:
    src = node.meta.get("source")
    if not src:
        return ""
    params = ", ".join(f"{k}={v}" for k, v in src.get("params", {}).items())
    return f"{src['asset']}({params})" if params else src["asset"]


def _detail_lines(node: SceneNode, world: np.ndarray) -> list[str]:
    lines = []
    if node.size is not None:
        for name in node.list_attachments():
            try:
                t = node.get_attachment(name)
            except ValueError:
                continue
            m = t.to_matrix()
            lines.append(f"@{name} {_vec(m[:3, 3])} yaw {_num(_yaw(m))}°")
    for name, s in node.surfaces.items():
        centre = s.origin + s.u_axis * s.u_extent / 2 + s.v_axis * s.v_extent / 2
        c = world[:3, :3] @ centre + world[:3, 3]
        n = world[:3, :3] @ s.normal
        n = n / (np.linalg.norm(n) or 1)
        lines.append(f"▭{name} {_vec(c)} normal {_vec(n)} {_num(s.u_extent)}×{_num(s.v_extent)} m")
    return lines


def _line(e: Entry) -> str:
    node = e.node
    bits = [node.name + (f" ×{e.count}" if e.count > 1 else "")]
    src = _source(node)
    if src:
        bits.append(f"({src})")
    tags = [t for t in node.tags if t != "room"]
    if tags:
        bits.append("#" + " #".join(tags))
    if e.count == 1:
        bits.append(f"at {_vec(e.world[:3, 3])}")
        yaw = _yaw(e.world)
        if abs(yaw) > 0.05 and abs(yaw - 360) > 0.05:
            bits.append(f"yaw {_num(yaw)}°")
    if e.lo is not None:
        if e.count == 1:
            bits.append(f"box {_vec(e.lo)}..{_vec(e.hi)} size {_vec(e.hi - e.lo)}")
        else:
            bits.append(f"size {_vec(e.hi - e.lo)}")
    extra = _meta_summary(node) + _interactions(node)
    if e.parts or e.tris:
        extra.append(f"parts {e.parts}, {_tris(e.tris)} tris")
    return " ".join(bits) + ("  | " + " | ".join(extra) if extra else "")


def _tris(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _positions(e: Entry) -> str:
    all_ = [e, *e.copies]
    if len(all_) > MAX_POSITIONS:
        lo = np.min([c.lo for c in all_ if c.lo is not None], axis=0) if e.lo is not None else None
        hi = np.max([c.hi for c in all_ if c.hi is not None], axis=0) if e.hi is not None else None
        return f"spread over {_vec(lo)}..{_vec(hi)}" if lo is not None else ""
    def where(c: Entry) -> str:
        yaw = _yaw(c.world)
        turned = f"@{_num(yaw)}°" if 0.05 < yaw < 359.95 else ""
        return f"{c.node.name}{_vec(c.world[:3, 3])}{turned}"
    return "at " + " ".join(where(c) for c in all_)


def select(tree: Entry, pattern: str) -> list[Entry]:
    """Entries whose path or name matches ``pattern`` (glob; a path suffix also matches)."""
    found = []

    def walk(e: Entry) -> None:
        for c in e.children:
            names = [c, *c.copies]
            for x in names:
                if (fnmatch.fnmatch(x.node.name, pattern) or fnmatch.fnmatch(x.path, pattern)
                        or fnmatch.fnmatch(x.path, "*/" + pattern)):
                    found.append(x)
            walk(c)
    walk(tree)
    return found


def format_tree(tree: Entry, depth: int | None = None, detail: bool = False, top: bool = True) -> list[str]:
    lines: list[str] = []

    def emit(e: Entry, level: int) -> None:
        pad = "  " * level
        lines.append(pad + _line(e))
        if e.count > 1:
            lines.append(pad + "    " + _positions(e))
        if detail and e.count == 1:
            lines.extend(pad + "    " + d for d in _detail_lines(e.node, e.world))
        if depth is not None and level >= depth:
            hidden = sum(1 for _ in _iter(e)) - 1
            if hidden:
                lines.append(pad + f"    … {hidden} more below (raise --depth)")
            return
        for c in e.children:
            emit(c, level + 1)

    if top:
        root = tree.node
        head = f"{root.name}"
        if tree.lo is not None:
            head += f"  box {_vec(tree.lo)}..{_vec(tree.hi)} size {_vec(tree.hi - tree.lo)}"
        nodes = sum(1 for _ in root.iter_nodes())
        tris = sum(len(n.mesh.faces) for n in root.iter_nodes() if n.mesh is not None)
        head += f"  | {nodes} nodes, {_tris(tris)} tris"
        lines.append(head)
        if depth is not None and depth < 1:
            hidden = sum(1 for _ in _iter(tree)) - 1
            if hidden:
                lines.append(f"    … {hidden} more below (raise --depth)")
        else:
            for c in tree.children:
                emit(c, 1)
    else:
        emit(tree, 0)
    return lines


def _iter(e: Entry):
    yield e
    for c in e.children:
        yield from _iter(c)


def tree_json(e: Entry, detail: bool = False) -> dict[str, Any]:
    node = e.node
    out: dict[str, Any] = {"name": node.name, "path": e.path}
    if e.count > 1:
        out["count"] = e.count
        out["copies"] = [{"name": c.node.name, "at": _round(c.world[:3, 3])} for c in [e, *e.copies]]
    else:
        out["at"] = _round(e.world[:3, 3])
        out["yaw"] = round(_yaw(e.world), 2)
    src = node.meta.get("source")
    if src:
        out["asset"] = src["asset"]
        if src.get("params"):
            out["params"] = src["params"]
    if node.tags:
        out["tags"] = list(node.tags)
    if e.lo is not None:
        out["bounds"] = {"min": _round(e.lo), "max": _round(e.hi), "size": _round(e.hi - e.lo)}
    summary = _meta_summary(node)
    if summary:
        out["summary"] = summary
    if node.interactions:
        out["interactions"] = {i.name: {"initial": i.initial, "states": list(i.states)} for i in node.interactions}
    if e.parts or e.tris:
        out["parts"], out["tris"] = e.parts, e.tris
    if detail and e.count == 1:
        out["detail"] = _detail_lines(node, e.world)
    if e.children:
        out["children"] = [tree_json(c, detail) for c in e.children]
    return out


def _round(v) -> list[float]:
    return [round(float(x), 3) + 0.0 for x in v]


# Checks ---------------------------------------------------------------------------------

CHECKS = ("mesh", "layout", "affordances", "coplanar", "overlaps", "traffic")


@dataclass
class Finding:
    check: str
    kind: str
    message: str
    items: tuple[str, ...] = ()


def run_checks(root: SceneNode, checks: tuple[str, ...] = CHECKS) -> list[Finding]:
    """Every QA check over ``root``; a check that raises becomes an ``error`` finding."""
    findings: list[Finding] = []
    for check in checks:
        try:
            findings.extend(_CHECKS[check](root))
        except Exception as exc:  # noqa: BLE001 - report, don't abort the dump
            findings.append(Finding(check, "error", f"{type(exc).__name__}: {exc}"))
    return findings


def _mesh_check(root: SceneNode) -> list[Finding]:
    from .core import meshops

    seen: dict[int, list[str]] = {}
    reports = {}
    for node in root.iter_nodes():
        if node.mesh is None or not len(node.mesh.faces):
            continue
        key = id(node.mesh)
        if key not in reports:
            reports[key] = meshops.validate(node.mesh)
            seen[key] = []
        seen[key].append(node.name)
    out = []
    for key, report in reports.items():
        if report.issues:
            names = seen[key]
            more = f" (+{len(names) - 1} instances)" if len(names) > 1 else ""
            out.append(Finding("mesh", "invalid", f"{names[0]}{more}: {', '.join(report.issues)}", (names[0],)))
    return out


def _qa(kind: str):
    def run(root: SceneNode) -> list[Finding]:
        from .layout import qa

        fn = {"layout": qa.check_layout, "coplanar": qa.coplanar_overlaps, "overlaps": qa.object_overlaps}[kind]
        return [Finding(kind, i.kind, str(i).split("] ", 1)[-1], tuple(i.items)) for i in fn(root)]
    return run


def _affordance_check(root: SceneNode) -> list[Finding]:
    from .layout.affordance_qa import check_affordances

    if not any(n.meta.get("affordances") for n in root.iter_nodes()):
        return []
    return [Finding("affordances", i.kind, str(i).split("] ", 1)[-1], tuple(i.items))
            for i in check_affordances(root)]


def _traffic_check(root: SceneNode) -> list[Finding]:
    from .traffic import build_traffic, check_clearance, check_graph

    if not any("traffic" in n.meta for n in root.iter_nodes()):
        return []
    graph = build_traffic(root)
    if graph is None:
        return []
    return [Finding("traffic", "lane", str(i)) for i in check_graph(graph) + check_clearance(root, graph)]


_CHECKS = {"mesh": _mesh_check, "layout": _qa("layout"), "affordances": _affordance_check,
           "coplanar": _qa("coplanar"), "overlaps": _qa("overlaps"), "traffic": _traffic_check}

_COORDS = re.compile(r" around \[[^\]]*\]")
_LONG_PATH = re.compile(r"(?<![\w/])[\w.-]+(?:/[\w.-]+)*/([\w.-]+/[\w.-]+)(?![\w/])")


def format_findings(findings: list[Finding]) -> list[str]:
    """One line per finding; overlaps repeated at different places fold into ``(×N)``."""
    if not findings:
        return ["checks: no findings"]
    folded: dict[tuple[str, str, str], list[Finding]] = {}
    for f in findings:
        message = _LONG_PATH.sub(r"…/\1", _COORDS.sub("", f.message)) if f.kind == "overlap" else f.message
        folded.setdefault((f.check, f.kind, message), []).append(f)
    by_check = Counter(f.check for f in findings)
    lines = ["checks: " + ", ".join(f"{c} {n}" for c, n in by_check.items())]
    for (check, kind, message), group in folded.items():
        if len(group) == 1:
            lines.append(f"  [{check}/{kind}] {group[0].message}")
        else:
            lines.append(f"  [{check}/{kind}] {message} (×{len(group)})")
    return lines


def inspect_scene(root: SceneNode, *, as_json: bool = False, depth: int | None = None,
                  pattern: str | None = None, parts: bool = False, detail: bool = False,
                  checks: tuple[str, ...] = CHECKS) -> str:
    """The ``--inspect`` report for ``root`` as text (or JSON)."""
    tree = build_tree(root, parts=parts)
    chosen = select(tree, pattern) if pattern else None
    findings = run_checks(root, checks) if checks else []
    if chosen is not None:
        names = {x.node.name for c in chosen for x in _iter(c)}
        findings = [f for f in findings if names & set(f.items) or any(n in f.message for n in names)]
    if as_json:
        data: dict[str, Any] = {"scene": root.name}
        if chosen is not None:
            data["selected"] = [tree_json(c, detail=True) for c in chosen]
        else:
            data["tree"] = tree_json(tree, detail)
        data["findings"] = [f.__dict__ for f in findings]
        return json.dumps(data, indent=1, default=str)
    lines: list[str] = []
    if chosen is not None:
        if not chosen:
            lines.append(f"nothing matches {pattern!r}")
        for c in chosen:
            lines.extend(format_tree(c, depth, detail=True, top=False))
    else:
        lines.extend(format_tree(tree, depth, detail))
    if checks:
        lines.append("")
        lines.extend(format_findings(findings))
    return "\n".join(lines)
