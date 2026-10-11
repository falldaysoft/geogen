"""References: one way to name things in a scene, and stable IDs for them.

A reference names a node by the names it was authored with::

    house                       # a placement of this scene
    plot/house                  # a placement inside a nested placement
    town:park/bench             # the same, in another package
    town:[tag=furniture.bench]  # every node with that tag
    town:park/[tag=door]        # ... below town's `park`
    house.front_wall            # with a member (a surface or attachment) for the keys that take one

A *package* is a registry scene name for now. Each path step names nodes by
their authored name (placement key, part name, room id), never the uniquified
name an export gives them. A step searches the *scope* of the node before
it: every asset and placed scene is a scope with its own names, so ``door``
in a cottage is its door placement, not the ``door`` part of a wardrobe
somewhere inside. ``suite/bed`` works however many nodes lie between the
two, as long as they belong to ``suite``. A name may be a glob (``bench_*``)
and of several siblings called ``chair`` the first is ``chair``, the second
``chair#2`` (``chair*`` selects them all).

A filter step ``[key=value, ...]`` searches the whole subtree, across scopes.
Keys: ``tag`` (a tag or any dotted prefix of one), ``type`` (``meta.type``),
``asset`` (the asset's file stem) and ``name`` (a glob). That is how to
reach into generated content (city lots, recipes), whose names are only
stable per seed.

``resolve`` wants exactly one node and ``select`` any number. Both raise
``RefError`` with near names and what exists at the failing step.

A node's stable ID is ``<package>:<authored names from the root>``
(``cottage:house/door_frame``); ``assign_ids`` writes it to ``meta.id`` for
export, and an ID is itself a reference that resolves to that node. Saved
state is keyed by these IDs, so it survives a re-export.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Callable, Iterable

from .core.node import SceneNode

FILTER_KEYS = ("tag", "type", "asset", "name")
INDEX_FORMAT = "geogen-refs"
INDEX_VERSION = 1

_STEP = re.compile(r"^([^\[\]]*)(?:\[([^\[\]]*)\])?$")
_PACKAGE = re.compile(r"^([A-Za-z0-9_\-]+):")


class RefError(ValueError):
    """A reference that is malformed, matches nothing, or matches too much.

    ``where`` is the source of the reference (``file:line: key``) when known;
    ``with_source`` adds it once the caller knows it.
    """

    def __init__(self, message: str, ref: str = "", where: str = "") -> None:
        self.message = message
        self.ref = ref
        self.where = where
        super().__init__(f"{where}: {message}" if where else message)

    def with_source(self, where: str) -> RefError:
        return self if self.where or not where else RefError(self.message, self.ref, where)


@dataclass
class Step:
    name: str | None = None                       # authored name or glob; None for a pure filter
    filters: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        filters = ", ".join(f"{k}={v}" for k, v in self.filters.items())
        return (self.name or "") + (f"[{filters}]" if filters else "")


@dataclass
class Ref:
    text: str
    package: str | None = None
    steps: list[Step] = field(default_factory=list)
    member: str | None = None                     # surface / attachment name after the first '.'


def parse(text: Any, member: bool = False) -> Ref:
    """Parse a reference. With ``member``, what follows the first ``.`` of the
    last step is a member name (``suite.bedroom.floor`` -> ``suite``, ``bedroom.floor``)."""
    if isinstance(text, Ref):
        return text
    if not isinstance(text, str) or not text.strip():
        raise RefError(f"expected a reference, got {text!r}", str(text))
    ref = Ref(text=text)
    rest = text.strip()
    match = _PACKAGE.match(rest)
    if match:
        ref.package, rest = match.group(1), rest[match.end():]
    parts = _split(rest, "/") if rest else []
    if member and parts:
        head, dot, tail = _partition_outside_brackets(parts[-1], ".")
        if dot:
            if not tail:
                raise RefError(f"'{text}': nothing after the '.'", text)
            parts[-1], ref.member = head, tail
    for part in parts:
        match = _STEP.match(part.strip())
        if match is None or not part.strip():
            raise RefError(f"'{text}': can't read step '{part}' (expected name, name[key=value] or [key=value])", text)
        name, filters = match.group(1).strip() or None, {}
        for item in _split(match.group(2) or "", ","):
            key, eq, value = (s.strip() for s in item.partition("="))
            if not eq or not value:
                raise RefError(f"'{text}': filter '{item.strip()}' should be key=value", text)
            if key not in FILTER_KEYS:
                raise RefError(f"'{text}': unknown filter '{key}'{_near(key, FILTER_KEYS)} "
                               f"(filters: {', '.join(FILTER_KEYS)})", text)
            filters[key] = value
        ref.steps.append(Step(name, filters))
    return ref


def _split(text: str, sep: str) -> list[str]:
    """Split on ``sep`` outside brackets; empty input gives no parts."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        depth += (ch == "[") - (ch == "]")
        if ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    if text[start:].strip() or parts:
        parts.append(text[start:])
    return parts


def _partition_outside_brackets(text: str, sep: str) -> tuple[str, str, str]:
    depth = 0
    for i, ch in enumerate(text):
        depth += (ch == "[") - (ch == "]")
        if ch == sep and depth == 0:
            return text[:i], sep, text[i + 1:]
    return text, "", ""


# -- the authored tree -------------------------------------------------------------------------

def is_scope(node: SceneNode) -> bool:
    """Whether ``node`` is an asset or placed scene: a namespace of its own."""
    return "source" in node.meta or bool(node.meta.get("scope"))


def child_keys(node: SceneNode) -> list[str]:
    """One key per child: its name, with ``#n`` for the nth of several siblings of that name."""
    seen: dict[str, int] = {}
    keys = []
    for child in node.children:
        key = child.meta.get("ref_key")           # an index's stub tree knows its keys
        if key is None:
            seen[child.name] = seen.get(child.name, 0) + 1
            key = child.name if seen[child.name] == 1 else f"{child.name}#{seen[child.name]}"
        keys.append(str(key))
    return keys


def path_of(node: SceneNode, root: SceneNode | None = None) -> str:
    """``node``'s authored path below ``root`` (default: its tree's root); '' for the root itself."""
    segments = []
    while node.parent is not None and node is not root:
        parent = node.parent
        index = next(i for i, c in enumerate(parent.children) if c is node)
        segments.append(child_keys(parent)[index])
        node = parent
    return "/".join(reversed(segments))


def node_id(node: SceneNode, package: str, root: SceneNode | None = None) -> str:
    """The stable ID of ``node``: ``<package>:<authored path>``."""
    return f"{package}:{path_of(node, root)}"


def _scope_members(node: SceneNode) -> Iterable[tuple[str, SceneNode]]:
    """(key, node) for everything in ``node``'s scope: its descendants, down to
    and including nested scopes but not what is inside them."""
    for key, child in zip(child_keys(node), node.children):
        yield key, child
        if not is_scope(child):
            yield from _scope_members(child)


def _named(node: SceneNode, name: str) -> list[SceneNode]:
    """Nodes called ``name`` in ``node``'s scope (not looking inside a match)."""
    out = []
    for key, child in zip(child_keys(node), node.children):
        if key == name or fnmatchcase(key, name):
            out.append(child)
        elif not is_scope(child):
            out.extend(_named(child, name))
    return out


def _matches(node: SceneNode, filters: dict[str, str]) -> bool:
    for key, value in filters.items():
        if key == "tag":
            if not any(tag == value or tag.startswith(value + ".") for tag in node.tags):
                return False
        elif key == "type":
            if node.meta.get("type") != value:
                return False
        elif key == "asset":
            if asset_of(node) != value:
                return False
        elif key == "name":
            if not fnmatchcase(node.name, value):
                return False
    return True


def asset_of(node: SceneNode) -> str | None:
    """The file stem of the asset ``node`` is the root of (``door``), if it is one."""
    source = node.meta.get("source")
    asset = source.get("asset") if isinstance(source, dict) else node.meta.get("asset")
    if not asset:
        return None
    return str(asset).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _apply(nodes: list[SceneNode], step: Step) -> list[SceneNode]:
    out: list[SceneNode] = []
    seen: set[int] = set()
    for node in nodes:
        if step.name is not None:
            found = [n for n in _named(node, step.name) if _matches(n, step.filters)]
        else:
            found = [n for n in node.iter_nodes(include_self=False) if _matches(n, step.filters)]
        for n in found:
            if id(n) not in seen:
                seen.add(id(n))
                out.append(n)
    return out


Packages = Callable[[str], "SceneNode | None"]


def select(root: SceneNode, ref: str | Ref, package: str | None = None, optional: bool = True,
           packages: Packages | None = None, member: bool = False) -> list[SceneNode]:
    """Every node ``ref`` names below ``root`` (the tree of ``package``).

    A reference to another package looks its tree up with ``packages(name)``.
    Without ``optional``, matching nothing is an error.
    """
    ref = parse(ref, member=member)
    label = package or root.name
    if ref.package is not None and package is not None and ref.package != package:
        other = packages(ref.package) if packages is not None else None
        if other is None:
            raise RefError(f"'{ref.text}': package '{ref.package}' isn't available here "
                           f"(this is '{package}')", ref.text)
        root, label = other, ref.package
    nodes = [root]
    for i, step in enumerate(ref.steps):
        found = _apply(nodes, step)
        if not found:
            if optional:
                return []
            raise RefError(_missing(ref, i, nodes, root, label), ref.text)
        nodes = found
    return nodes


def resolve(root: SceneNode, ref: str | Ref, package: str | None = None,
            packages: Packages | None = None, member: bool = False) -> SceneNode:
    """The one node ``ref`` names; RefError if there is none or more than one."""
    ref = parse(ref, member=member)
    nodes = select(root, ref, package, optional=False, packages=packages)
    if len(nodes) > 1:
        base = nodes[0].root
        paths = sorted(path_of(n, base) for n in nodes)
        shown = ", ".join(f"'{p}'" for p in paths[:6]) + (f" and {len(paths) - 6} more" if len(paths) > 6 else "")
        raise RefError(f"'{ref.text}' matches {len(nodes)} things: {shown}. Use a longer path, "
                       f"or select: where several are wanted", ref.text)
    return nodes[0]


def _near(name: str, known: Iterable[str]) -> str:
    close = difflib.get_close_matches(name, sorted(set(known)), n=1, cutoff=0.5)
    return f" Did you mean '{close[0]}'?" if close else ""


def _listed(names: Iterable[str], limit: int = 12) -> str:
    names = sorted(set(names))
    return ", ".join(names[:limit]) + (f", ... ({len(names)} in all)" if len(names) > limit else "") or "nothing"


def _missing(ref: Ref, index: int, nodes: list[SceneNode], root: SceneNode, label: str) -> str:
    """Why step ``index`` matched nothing below ``nodes``: near names and what is there."""
    step = ref.steps[index]
    here = nodes[0]
    at = f"'{path_of(here, root)}' in '{label}'" if here is not root else f"'{label}'"
    if len(nodes) > 1:
        at = f"the {len(nodes)} matches of '{'/'.join(str(s) for s in ref.steps[:index])}'"
    if step.name is None:
        tags = {t for n in nodes for d in n.iter_nodes(include_self=False) for t in d.tags}
        hint = _near(step.filters["tag"], tags) if "tag" in step.filters else ""
        return f"'{ref.text}': nothing under {at} matches [{str(step)[1:-1]}].{hint}"
    members = [(key, n) for node in nodes for key, n in _scope_members(node)]
    names = {key for key, _ in members}
    if any(key == step.name for key, _ in members):
        return f"'{ref.text}': '{step.name}' under {at} doesn't match [{', '.join(f'{k}={v}' for k, v in step.filters.items())}]"
    deeper = [path_of(n, here) for n in here.iter_nodes(include_self=False) if n.name == step.name]
    hint = _near(step.name, names)
    if deeper and not hint:
        hint = f" There is one at '{deeper[0]}'." if len(deeper) == 1 else f" There are {len(deeper)} deeper, e.g. '{deeper[0]}'."
    return f"'{ref.text}': no '{step.name}' in {at}.{hint} It has: {_listed(names)}"


def surface_of(node: SceneNode, name: str, ref: str = ""):
    """``node``'s surface ``name``; RefError naming the near ones if it has none."""
    surface = node.surfaces.get(name)
    if surface is None:
        raise RefError(f"'{node.name}' has no surface '{name}'.{_near(name, node.surfaces)} "
                       f"Surfaces: {_listed(node.surfaces, 20)}", ref or name)
    return surface


# -- stable IDs and the export index -----------------------------------------------------------

def identified(node: SceneNode) -> bool:
    """Whether a node gets an exported ID: things placed, tagged, typed or interactive."""
    return (is_scope(node) or bool(node.tags) or bool(node.interactions)
            or any(k in node.meta for k in ("type", "travel", "npc", "room")))


def assign_ids(root: SceneNode, package: str) -> dict[str, SceneNode]:
    """Write ``meta.id`` on every identified node below ``root``; returns ID -> node."""
    out: dict[str, SceneNode] = {}

    def visit(node: SceneNode, path: str) -> None:
        node.meta.pop("id", None)
        if identified(node) or node is root:
            node.meta["id"] = f"{package}:{path}"
            out[node.meta["id"]] = node  # type: ignore[index]
        for key, child in zip(child_keys(node), node.children):
            visit(child, f"{path}/{key}" if path else key)

    visit(root, "")
    return out


def build_index(root: SceneNode, package: str, names: Callable[[SceneNode], str] | None = None) -> dict[str, Any]:
    """The package's reference index: what other packages' references are checked against.

    ``names`` gives each node's exported name. IDs must have been assigned.
    """
    nodes = []
    for node in root.iter_nodes():
        if "id" not in node.meta:
            continue
        entry: dict[str, Any] = {"id": node.meta["id"]}
        if names is not None:
            entry["node"] = names(node)
        if node.tags:
            entry["tags"] = list(node.tags)
        if node.meta.get("type"):
            entry["type"] = node.meta["type"]
        if asset_of(node):
            entry["asset"] = asset_of(node)
        if is_scope(node):
            entry["scope"] = True
        nodes.append(entry)
    return {"format": INDEX_FORMAT, "version": INDEX_VERSION, "package": package, "nodes": nodes}


def index_tree(index: dict[str, Any]) -> SceneNode:
    """A stand-in tree for an exported package, built from its index, that references resolve on."""
    package = index["package"]
    root = SceneNode(package)
    by_path: dict[str, SceneNode] = {"": root}

    def ensure(path: str) -> SceneNode:
        node = by_path.get(path)
        if node is None:
            parent_path, _, key = path.rpartition("/")
            node = SceneNode(key.split("#")[0], meta={"ref_key": key})
            ensure(parent_path).add_child(node)
            by_path[path] = node
        return node

    for entry in index.get("nodes", []):
        node = ensure(entry["id"].partition(":")[2])
        node.tags = list(entry.get("tags", []))
        node.meta["id"] = entry["id"]
        for key in ("type", "asset", "scope", "node"):
            if key in entry:
                node.meta[key] = entry[key]
    return root


# -- references while a scene is being composed ------------------------------------------------

class SceneRefs:
    """A scene's references, resolved against what has been placed so far.

    ``source`` is the scene's YAML file; every lookup takes the key path of
    the reference in it, so errors read ``file:line: place.bench.on: ...``.
    """

    def __init__(self, root: SceneNode, source: Any = None, pending: Iterable[str] = ()) -> None:
        self.root = root
        self.source = source
        self.pending = set(pending)               # placement keys of the scene (placed or still to come)
        self.names = {root.name}
        if source is not None:
            from pathlib import Path

            self.names.add(Path(source).stem)

    def where(self, keys: Iterable[str | int]) -> str:
        from .layout.yaml_utils import source_of

        return source_of(self.source, list(keys))

    def _run(self, keys: Iterable[str | int], call: Callable[[], Any]) -> Any:
        try:
            return call()
        except RefError as exc:
            raise exc.with_source(self.where(keys)) from None

    def _parse(self, text: Any, member: bool = False) -> Ref:
        ref = parse(text, member=member)
        if ref.package is not None and ref.package not in self.names:
            raise RefError(f"'{ref.text}' is in package '{ref.package}': a placement can only refer to "
                           f"things in its own scene ('{sorted(self.names)[0]}')", ref.text)
        return ref

    def _not_yet(self, exc: RefError, ref: Ref) -> RefError:
        first = ref.steps[0].name if ref.steps else None
        if first in self.pending and not _named(self.root, first):
            return RefError(f"{exc.message} ('{first}' is a placement of this scene that isn't placed yet)",
                            exc.ref)
        return exc

    def node(self, text: Any, keys: Iterable[str | int]) -> SceneNode:
        """The one node ``text`` names."""
        return self.member(text, keys, member=False)[0]

    def member(self, text: Any, keys: Iterable[str | int], member: bool = True) -> tuple[SceneNode, str | None]:
        """(node, member name) for ``<ref>.<member>``."""
        def call() -> tuple[SceneNode, str | None]:
            ref = self._parse(text, member)
            try:
                return resolve(self.root, ref), ref.member
            except RefError as exc:
                raise self._not_yet(exc, ref) from None

        return self._run(keys, call)

    def nodes(self, text: Any, keys: Iterable[str | int], optional: bool = False) -> list[SceneNode]:
        """Every node ``text`` names (at least one unless ``optional``)."""
        def call() -> list[SceneNode]:
            ref = self._parse(text)
            try:
                return select(self.root, ref, optional=optional)
            except RefError as exc:
                raise self._not_yet(exc, ref) from None

        return self._run(keys, call)

    def surface(self, text: Any, keys: Iterable[str | int]) -> tuple[SceneNode, Any]:
        """(node, Surface) for ``<ref>.<surface>``."""
        def call() -> tuple[SceneNode, Any]:
            ref = self._parse(text, True)
            if ref.member is None:
                raise RefError(f"'{ref.text}' should be <object>.<surface>", ref.text)
            try:
                node = resolve(self.root, ref)
            except RefError as exc:
                raise self._not_yet(exc, ref) from None
            return node, surface_of(node, ref.member, ref.text)

        return self._run(keys, call)

    def fail(self, message: str, keys: Iterable[str | int], ref: str = "") -> RefError:
        return RefError(message, ref, self.where(keys))
