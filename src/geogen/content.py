"""Content packages: state, rules and quests that span scenes they don't own.

A scene's own ``state:`` / ``rules:`` / ``quests:`` are its implicit package, named after
the scene. Content that belongs to no one scene (a quest from the hub to the island and
back) is a ``kind: package`` file, ``assets/packages/<name>.yaml``::

    kind: package
    version: 1
    id: errand                        # default: the file name; not a scene's name
    title: The errand                 # optional, with `description`
    scenes: [camp, annex]             # the catalogue scenes it runs in and may reach into
    state:                            # its own variables: scope world | actor, as in a scene
      lever_pulled: {scope: world, type: bool, default: false}
    rules:                            # geogen.wiring: rules, wire and bind
      - on: {signal: "camp:lever.pulled"}
        do: [{set: world.lever_pulled, to: true}]
      - when: "world.lever_pulled"
        do: [{input: "annex:gate.open"}]
    quests: [quests/errand.yaml]      # kind: quest files (geogen.quests), relative to assets/

Everything it names in a scene says which scene: ``camp:lever`` (``geogen.refs``), in a
condition ``ref("camp:lever").pull`` and ``all("camp:[tag=light]", ...)``. ``world.x`` and
``actor.x`` are the package's own variables; ``world("camp").x`` reads a scene's.

Only one scene is loaded at a time, so a rule, binding or objective may touch one scene
(plus the package's own state, which is what crosses: it is kept while the game runs,
whatever scene is loaded). ``compile_package`` reports one that touches two.

``export_packages`` compiles the packages a catalogue deploys (``packages:`` in
``assets/runtime_scenes.yaml``) against the *exported* scenes: each scene's reference
index (``<scene>.refs.json``) is what references resolve on, and its manifest's ``state``
/ ``rules`` what conditions and effects are checked against, as ``travel.check_travel``
checks spawns. It writes ``<name>.content.json`` (format ``geogen-content`` v1, schema
``docs/schema/geogen-content.v1.schema.json``) beside ``catalogue.json``, which lists it.

A runtime loads the deployed packages once. When a scene one of them names loads, it adds
that package's variables, and the rules and bindings with no ``scene`` or with that one,
after the scene's own (``apply`` is the reference); when the scene unloads it keeps the
package's values (``carried``) and gives them to the next scene's program (``resume``).
Nothing about the rules semantics changes (``docs/rules.md``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from . import conditions, refs, roots
from .core.node import SceneNode

KIND = "package"
FORMAT = "geogen-content"
VERSION = 1
SCHEMA = "docs/schema/geogen-content.v1.schema.json"
DIRECTORY = "packages"            # assets/packages/<name>.yaml
PACKAGE_KEYS = {"kind", "version", "id", "title", "description", "scenes", "state", "rules", "wire", "bind", "quests"}
# Names a key's or latch's owner is read from ('world:<pkg>.x', 'once:<rule id>'), so no package has them.
RESERVED = ("world", "actor", "once", "watch", "quest", "signal", "player")

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ContentError(ValueError):
    """A content package that is malformed, or that the scenes it names can't support."""


@dataclass
class Scene:
    """An exported scene, as a content package sees it."""
    name: str
    state: dict[str, Any] | None = None       # its manifest's `state` section (geogen-state)
    rules: dict[str, Any] | None = None       # ... `rules` (geogen-rules, without variables)
    quests: dict[str, Any] | None = None      # ... `quests` (geogen-quests)
    tree: SceneNode | None = None             # its reference index's stand-in tree (refs.load_index)

    @classmethod
    def exported(cls, name: str, manifest: str | Path) -> Scene:
        """The scene exported with ``manifest`` (a manifest or chunk index beside its refs index)."""
        data = json.loads(Path(manifest).read_text())
        return cls(name, data.get("state"), data.get("rules"), data.get("quests"), refs.load_index(manifest))

    @classmethod
    def built(cls, name: str, root: SceneNode) -> Scene:
        """A built scene that hasn't been exported: what its export would hold."""
        from .quests import summary

        refs.assign_ids(root, name)
        sections = summary(root, name)
        return cls(name, sections.get("state"), sections.get("rules"), sections.get("quests"),
                   refs.index_tree(refs.build_index(root, name)))


def file_name(package: str) -> str:
    """The exported file of a package, beside ``catalogue.json``."""
    return f"{package}.content.json"


def load_package(path: str | Path) -> dict[str, Any]:
    """A ``kind: package`` file as written, with ``id`` defaulting to the file name."""
    from .layout.yaml_utils import load_kind

    return {"id": Path(path).stem, **load_kind(path, KIND, VERSION, error=ContentError)}


def package_of(key: str) -> str:
    """The package a variable key, a hidden latch or a stable ID belongs to ("" for a scene
    rule's latch, whose id says nothing)."""
    head, sep, rest = key.partition(":")
    if "#" not in key and head in ("world", "actor"):
        return rest.partition(".")[0]
    if head in ("once", "watch"):
        return rest.partition(":")[0] if ":" in rest else ""
    return head if sep else ""


def touched(entry: dict[str, Any]) -> set[str]:
    """The packages a compiled rule or binding reads, writes or listens to."""
    from .rules import _reads

    on = entry.get("on") or {}
    subject = on.get("subject")
    names = [*([subject] if isinstance(subject, str) else subject or []), on.get("key"), entry.get("set")]
    trees = [entry.get("when"), entry.get("to")]
    for effect in entry.get("do", []):
        names += [effect.get("set"), effect.get("subject")]
        trees.append(effect.get("to"))
    for tree in trees:
        names += _reads(tree)
        for node in conditions.walk(tree):
            names += node.get("over", [])
    return {package_of(name) for name in names if name}


def compile_package(spec: dict[str, Any], scenes: dict[str, Scene], source: Any = None,
                    assets_dir: str | Path | None = None) -> dict[str, Any]:
    """The ``geogen-content`` file of a package, compiled against the scenes it names.

    ``scenes`` are the exported scenes available (at least the ones it lists). Quest files
    are read from ``assets_dir`` (default: the directory above the package file's).
    Raises ``ContentError`` listing every problem with the file and line it was written at.
    """
    from .layout.yaml_utils import source_of
    from .quests import QuestError, compile_quests, load_quest
    from .state import Declarations, StateError, parse_variables
    from .wiring import KEYS, Source

    name = str(spec.get("id", "")) if isinstance(spec, dict) else ""
    problems: dict[str, None] = {}

    def where(*keys: str | int) -> str:
        if source is None:
            return f"package '{name}': " + ".".join(str(k) for k in keys)
        return source_of(source, list(keys))

    def fail() -> ContentError:
        return ContentError("\n".join(problems))

    if not isinstance(spec, dict) or not _NAME.match(name) or name in RESERVED:
        raise ContentError(f"{source or 'package'}: a package needs an id of letters, digits and '_' that isn't "
                           f"one of {', '.join(RESERVED)} (got {name!r})")
    for key in spec:
        if key not in PACKAGE_KEYS:
            problems[f"{where(key)}: unknown key '{key}'.{refs._near(str(key), PACKAGE_KEYS)} "
                     f"Keys: {', '.join(sorted(PACKAGE_KEYS))}"] = None
    listed = spec.get("scenes")
    if not isinstance(listed, list) or not listed or not all(isinstance(s, str) for s in listed):
        problems[f"{where('scenes')}: a package needs scenes: [<scene>, ...], the scenes it runs in"] = None
        raise fail()
    listed = list(dict.fromkeys(listed))
    for index, scene in enumerate(listed):
        if scene == name:
            problems[f"{where('id')}: '{name}' is also a scene: a scene's own state and rules are its package, "
                     f"so a content package needs another name"] = None
        elif scene not in scenes:
            problems[f"{where('scenes', index)}: no exported scene '{scene}'.{refs._near(scene, scenes)} "
                     f"Exported: {refs._listed(scenes)}"] = None
    if problems:
        raise fail()

    # What its scenes declare, and the programs they run: references, conditions and effects are
    # checked against these exactly as a scene's own rules are against its own.
    decls = Declarations(name)
    trees: dict[str, SceneNode] = {}
    scene_rules: list[dict[str, Any]] = []
    scene_bindings: list[dict[str, Any]] = []
    for scene in listed:
        found = scenes[scene]
        trees[scene] = found.tree if found.tree is not None else SceneNode(scene)
        if found.state:
            theirs = Declarations.from_json(found.state)
            for table in ("variables", "interactions", "signals", "inputs"):
                getattr(decls, table).update(getattr(theirs, table))
        for rule in (found.rules or {}).get("rules", []):
            if not str(rule.get("id", "")).startswith("signal:"):      # declared signals: decls makes them again
                scene_rules.append({**rule, "id": f"{scene}::{rule['id']}"})
        scene_bindings.extend((found.rules or {}).get("bindings", []))

    try:
        for variable, declared in parse_variables(spec.get("state"), source, scoped=True).items():
            decls.declare(f"{declared['scope']}:{name}.{variable}", {k: v for k, v in declared.items() if k != "scope"})
    except StateError as exc:
        problems[str(exc)] = None

    assets_dir = Path(assets_dir) if assets_dir is not None else (
        Path(str(source)).parent.parent if source is not None else Path("."))
    root = SceneNode(name, meta={"uid": f"{name}:"})        # what the package's own names hang from
    quests: list[tuple[dict[str, Any], Any, SceneNode]] = []
    files = spec.get("quests") or []
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        problems[f"{where('quests')}: quests is a list of quest files (quests/name.yaml)"] = None
        files = []
    for index, file in enumerate(files):
        path = roots.find(file, assets_dir)
        if not path.is_file():
            known = [roots.relative(p, assets_dir) for p in roots.glob("quests/*.yaml", assets_dir)]
            problems[f"{where('quests', index)}: no quest file '{file}'.{refs._near(file, known)}"] = None
            continue
        try:
            quest = load_quest(path)
        except QuestError as exc:
            problems[str(exc)] = None
            continue
        for scene in listed:
            if quest["id"] in ((scenes[scene].quests or {}).get("quests") or {}):
                problems[f"{where('quests', index)}: scene '{scene}' already has a quest '{quest['id']}': "
                         f"give this one another id"] = None
        quests.append((quest, str(path), root))
    wiring = [Source({key: spec[key] for key in KEYS if spec.get(key)}, source, root, False)]
    if problems:
        raise fail()
    try:
        found = compile_quests(quests, decls, rules=scene_rules, wiring=wiring if wiring[0].spec else [],
                               by_name=False, trees=trees, bindings=scene_bindings)
    except (QuestError, StateError) as exc:
        raise ContentError(str(exc)) from None

    def placed(entry: dict[str, Any], shown: str, at: str) -> dict[str, Any]:
        """``entry`` with the one scene it runs in, if it touches one."""
        there = sorted(touched(entry) - {name, ""})
        if len(there) > 1:
            problems[f"{at}: {shown} touches {' and '.join(there)}, but only one scene is loaded at a time. "
                     f"Split it: one rule sets state of this package (world.<name> or actor.<name>) in the "
                     f"first scene, another reads it in the second"] = None
        return {**entry, "scene": there[0]} if there else dict(entry)

    rules = []
    for rule in (*found.wiring, *found.rules):
        at = found.written.get(rule["id"], where("rules"))
        rules.append({**placed(rule, f"rule '{rule['id']}'", at), "id": f"{name}:{rule['id']}"})
    bindings = []
    for binding in found.bindings:
        at = found.written.get(f"bind:{binding['set']}", where("bind"))
        entry = placed(binding, "the binding", at)
        if "scene" in entry and package_of(binding["set"]) != entry["scene"]:
            problems[f"{at}: the binding reads '{entry['scene']}', which isn't always loaded, so the package's "
                     f"own state can't be bound to it: set it with a rule"] = None
        bindings.append(entry)
    if problems:
        raise fail()

    out: dict[str, Any] = {"format": FORMAT, "version": VERSION, "schema": SCHEMA, "package": name}
    for key in ("title", "description"):
        if spec.get(key):
            out[key] = str(spec[key])
    out["scenes"] = listed
    out["variables"] = {key: variable for key, variable in decls.variables.items() if package_of(key) == name}
    out["bindings"] = bindings
    out["rules"] = rules
    if found:
        out["quests"] = found.to_json()
    return out


def export_packages(catalogue: Any, out_dir: str | Path, log: Callable[[str], None] = print) -> list[str]:
    """Compile the catalogue's packages against the scenes exported in ``out_dir`` and write
    each ``<name>.content.json`` there. Returns the problems, each starting with where it was
    written.

    A package with a problem is not written (and an older export of it is removed, so the
    runtime never runs content that no longer fits its scenes). One whose scenes aren't all
    exported yet is left out too, without a problem: it is written when they are.
    """
    from .catalogue import entry_manifest

    out_dir = Path(out_dir)
    exported: dict[str, Scene] = {}
    problems: list[str] = []
    for name, path in getattr(catalogue, "packages", {}).items():
        target = out_dir / file_name(name)
        try:
            spec = load_package(path)
            if spec["id"] != name:
                raise ContentError(f"{path}: its id is '{spec['id']}', but the catalogue lists it as '{name}'")
            listed = [s for s in spec.get("scenes") or [] if isinstance(s, str)]
            unknown = [s for s in listed if s not in catalogue.entries]
            if unknown:
                raise ContentError("\n".join(
                    f"{path}: scenes: '{s}' isn't in the catalogue.{refs._near(s, catalogue.entries)}" for s in unknown))
            if name in catalogue.entries:
                raise ContentError(f"{path}: '{name}' is also a scene of the catalogue: a content package needs "
                                   f"another name")
            waiting = [s for s in listed if not (out_dir / entry_manifest(catalogue.entries[s])).exists()]
            if waiting:
                target.unlink(missing_ok=True)
                log(f"package {name}: waiting for {', '.join(waiting)} to be exported")
                continue
            for scene in listed:
                if scene not in exported:
                    exported[scene] = Scene.exported(scene, out_dir / entry_manifest(catalogue.entries[scene]))
            content = compile_package(spec, {s: exported[s] for s in listed}, path)
        except (ContentError, OSError, ValueError) as exc:
            target.unlink(missing_ok=True)
            problems.extend(str(exc).splitlines())
            continue
        target.write_text(json.dumps(content, indent=2) + "\n")
        log(f"package {name} -> {target}")
    return problems


def index_entries(catalogue: Any, out_dir: str | Path) -> list[dict[str, Any]]:
    """``catalogue.json``'s ``packages``: every deployed package, with its file, whether it is
    exported, and then the scenes it runs in."""
    out = []
    for name in getattr(catalogue, "packages", {}):
        entry: dict[str, Any] = {"name": name, "file": file_name(name), "exported": False, "scenes": []}
        try:
            content = json.loads((Path(out_dir) / entry["file"]).read_text())
            if content.get("format") == FORMAT and content.get("package") == name:
                entry.update(exported=True, scenes=list(content.get("scenes", [])))
        except (OSError, ValueError):
            pass
        out.append(entry)
    return out


# ---- what a runtime does with one: the reference ------------------------------------------------

def apply(contents: Iterable[dict[str, Any]], scene: str, program: dict[str, Any]) -> dict[str, Any]:
    """The rules program a runtime runs while ``scene`` is loaded: the scene's own ``program``
    (``geogen.rules``) with, for every package of ``contents`` that names the scene, its
    variables and the rules and bindings that run there, after the scene's own."""
    out = {**program, "variables": dict(program.get("variables", {})), "rules": list(program.get("rules", [])),
           "bindings": list(program.get("bindings", []))}
    for content in contents:
        if scene not in content.get("scenes", []):
            continue
        out["variables"].update(content.get("variables", {}))
        for section in ("rules", "bindings"):
            out[section].extend({k: v for k, v in entry.items() if k != "scene"}
                                for entry in content.get(section, []) if entry.get("scene", scene) == scene)
    return out


def carried(save: dict[str, Any], contents: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """What of a save (``rules.Machine.save``) belongs to the packages ``contents``: their world
    and actor values and their rules' latches, ``{world: {...}, actors: {id: {...}}}``. A
    runtime keeps this when a scene unloads.

    The watch latch of a rule tied to a scene stays behind: the scene's things start over
    when it loads again, so its condition is judged afresh (``once`` latches are kept)."""
    contents = list(contents)
    names = {content["package"] for content in contents}
    afresh = {f"watch:{rule['id']}" for content in contents for rule in content.get("rules", []) if "scene" in rule}

    def kept(values: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in values.items() if package_of(k) in names and k not in afresh}

    return {"world": kept(save.get("world", {})),
            "actors": {actor: kept(values) for actor, values in save.get("actors", {}).items()}}


def resume(save: dict[str, Any] | None, kept: dict[str, Any], contents: Iterable[dict[str, Any]],
           actors: Iterable[str] | None = None) -> dict[str, Any]:
    """The save a scene's program starts from: its own ``save`` from when it was last loaded
    (None: never) with the packages' values replaced by the ones ``kept`` (``carried``).
    ``actors`` limits whose actor state comes along (the ones in this scene)."""
    from .rules import SAVE_FORMAT, SAVE_VERSION

    names = {content["package"] for content in contents}
    out = json.loads(json.dumps(save)) if save else {"format": SAVE_FORMAT, "version": SAVE_VERSION}
    world = {k: v for k, v in out.get("world", {}).items() if package_of(k) not in names}
    out["world"] = {**world, **kept.get("world", {})}
    people = {actor: {k: v for k, v in values.items() if package_of(k) not in names}
              for actor, values in out.get("actors", {}).items()}
    for actor, values in kept.get("actors", {}).items():
        if actors is None or actor in actors:
            people[actor] = {**people.get(actor, {}), **values}
    out["actors"] = people
    return out
