"""Quests: tasks with progress and an end, compiled to actor state and rules.

A quest is a ``kind: quest`` file (``assets/quests/*.yaml``) that a scene lists
under ``quests:``. ``docs/quests.md`` is the design; in short::

    kind: quest
    version: 1
    id: lights_out                  # default: the file name
    title: Lights out
    holder: player                  # player | npc | any
    start: given                    # given: a start_quest effect; auto: as soon as it is available
    available: "clock.night"        # can't start until this holds
    ordered: false                  # true: each objective waits for the one before
    objectives:
      - {id: lights, text: Switch off every light,
         done: 'all("house/[tag=light_switch]", it.switch == "off")'}      # a condition: latches when true
      - {id: bed, text: Go to bed, after: lights, done: {enter: house/bedroom}}   # or an event
      - {id: window, text: Close the window, optional: true, done: 'window.swing == "closed"'}
    fail: 'clock.time >= "06:00" and not clock.night'
    rewards: [{set: actor.slept_well, to: true}]

Nothing here runs a quest. ``collect`` compiles the quests of a built scene to
actor-scoped variables (``actor:<pkg>.quest_<id>``: unavailable | available |
active | complete | failed; a done latch per objective; a counter for ``count:``)
and rules of ``geogen.rules``, so the reference ``Machine``, and any runtime that
implements those semantics, runs them with no quest logic of its own. It raises
``QuestError`` listing every problem with the file and line it was written at:
references that don't resolve, undeclared state, ``after`` loops and objectives
nothing can cause.

Exports put ``Quests.to_json()`` in the manifest and chunk index as ``quests``
(schema ``docs/schema/geogen-quests.v1.schema.json``) and the quest variables in
``state``. Other content drives quests with the effects ``{start_quest: id}``,
``{fail_quest: id}`` and ``{complete_objective: quest.objective}``
(``Quests.effect`` compiles one to conditional ``set`` effects).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import conditions, refs
from .conditions import ConditionError
from .core.node import SceneNode
from .state import Declarations, StateError, fits

KIND = "quest"
FORMAT = "geogen-quests"
VERSION = 1
SCHEMA = "docs/schema/geogen-quests.v1.schema.json"
META = "quests"                   # node.meta key the composer records a scene's quests under
STATES = ("unavailable", "available", "active", "complete", "failed")
HOLDERS = {
    "player": "only the player can hold it",
    "npc": "only NPCs can hold it",
    "any": "the player or an NPC",
}
STARTS = {
    "given": "a `start_quest` effect starts it (a rule on any event: E on an NPC, entering a place)",
    "auto": "it starts as soon as it is available: at a new game unless `available` holds it back",
}
QUEST_KEYS = {"kind", "version", "id", "title", "description", "holder", "start", "available", "ordered",
              "objectives", "fail", "rewards"}
OBJECTIVE_KEYS = {"id", "text", "after", "done", "optional", "marker"}
EVENT_SOURCES = {
    "enter": "the holder enters this place or room (a reference)",
    "leave": "the holder leaves this place or room",
    "signal": "<thing>.<name>: the thing raises this signal, caused by the holder",
    "use": "<thing> or <thing>.<interaction>: the holder uses it",
    "changed": "<thing>.<state> | world.<name> | actor.<name> changes (`to`: only to that value), caused by the holder",
}
EVENT_KEYS = {*EVENT_SOURCES, "to", "when", "count"}
EFFECTS = {
    "set": "<thing>.<state> | world.<name> | actor.<name>, with `to`: a value or an expression",
    "emit": "raise a signal of this name (`of`: the thing it comes from; default: the scene)",
    "start_quest": "start a quest (by id) for the actor, if it is available to them",
    "fail_quest": "fail an active quest",
    "complete_objective": "<quest>.<objective>: mark an open objective done",
}
MARKER_KEYS = {"at", "area"}

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KIND_OF_ACTOR = {"get": "actor", "name": "kind"}

Tree = Any


class QuestError(ValueError):
    """Quests that are malformed, or that the scene they are in can't support."""


# ---- reading YAML -------------------------------------------------------------------------------

@dataclass
class _Objective:
    id: str
    index: int
    spec: dict[str, Any]
    after: list[str] = field(default_factory=list)
    optional: bool = False
    latch: str = ""
    counter: str | None = None
    count: int = 1
    tree: Tree = None               # the compiled done condition (condition objectives)
    on: dict[str, Any] | None = None    # ... or the event that does it
    when: Tree = None               # ... and what must hold when it happens
    out: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Quest:
    id: str
    spec: dict[str, Any]
    source: Any = None              # the file it was written in, for diagnostics
    node: SceneNode | None = None   # the scene that contributes it: references resolve from here
    status: str = ""
    objectives: list[_Objective] = field(default_factory=list)

    def where(self, *keys: str | int) -> str:
        from .layout.yaml_utils import source_of

        if self.source is None:
            return f"quest '{self.id}': " + ".".join(str(k) for k in keys)
        return source_of(self.source, list(keys))

    def objective(self, name: str) -> _Objective | None:
        return next((o for o in self.objectives if o.id == name), None)


def load_quest(path: str | Path) -> dict[str, Any]:
    """A ``kind: quest`` file as written, with ``id`` defaulting to the file name."""
    from .layout.yaml_utils import safe_load_path

    path = Path(path)
    data = safe_load_path(path)
    if not isinstance(data, dict) or data.get("kind") != KIND or int(data.get("version", 0)) != VERSION:
        raise QuestError(f"{path}: expected kind '{KIND}' version {VERSION}")
    return {"id": path.stem, **data}


def declare_scene(root: SceneNode, data: dict[str, Any], source: Any, assets_dir: Path) -> None:
    """Record a scene's ``quests: [file, ...]`` on its root (``meta.quests``, not exported)."""
    from .layout.yaml_utils import source_of

    listed = data.get("quests")
    if not listed:
        return
    if not isinstance(listed, list) or not all(isinstance(item, str) for item in listed):
        raise QuestError(f"{source_of(source, ['quests'])}: quests is a list of quest files (quests/name.yaml)")
    found = []
    for index, name in enumerate(listed):
        path = Path(assets_dir) / name
        if not path.is_file():
            known = [p.relative_to(assets_dir).as_posix() for p in sorted(Path(assets_dir).glob("quests/*.yaml"))]
            raise QuestError(f"{source_of(source, ['quests', index])}: no quest file '{name}'."
                             f"{refs._near(name, known)}")
        found.append({"source": str(path), "spec": load_quest(path)})
    root.meta[META] = found


def _parse(spec: Any, source: Any, node: SceneNode | None, problems: list[str]) -> _Quest | None:
    """The shape of one quest: keys, ids, ``after``. Conditions are compiled later."""
    if not isinstance(spec, dict) or not _NAME.match(str(spec.get("id", ""))):
        problems.append(f"{source or 'quest'}: a quest needs an id of letters, digits and '_' "
                        f"(got {spec.get('id') if isinstance(spec, dict) else spec!r})")
        return None
    quest = _Quest(str(spec["id"]), spec, source, node)

    def bad(message: str, *keys: str | int) -> None:
        problems.append(f"{quest.where(*keys)}: {message}")

    for key in spec:
        if key not in QUEST_KEYS:
            bad(f"unknown key '{key}'.{refs._near(str(key), QUEST_KEYS)} Keys: {', '.join(sorted(QUEST_KEYS))}", key)
    if not isinstance(spec.get("title"), str) or not spec["title"].strip():
        bad("a quest needs a title", "title")
    for key, known in (("holder", HOLDERS), ("start", STARTS)):
        if spec.get(key, next(iter(known))) not in known:
            bad(f"{key} is one of {', '.join(known)}, got {spec[key]!r}", key)
    objectives = spec.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        bad("a quest needs objectives: [{id, text, done}, ...]", "objectives")
        return None
    previous: str | None = None
    for index, entry in enumerate(objectives):
        at = ("objectives", index)
        if not isinstance(entry, dict) or not _NAME.match(str(entry.get("id", ""))):
            bad("an objective needs an id of letters, digits and '_'", *at)
            continue
        name = str(entry["id"])
        if quest.objective(name) is not None:
            bad(f"two objectives are called '{name}'", *at, "id")
            continue
        for key in entry:
            if key not in OBJECTIVE_KEYS:
                bad(f"unknown key '{key}'.{refs._near(str(key), OBJECTIVE_KEYS)} "
                    f"Keys: {', '.join(sorted(OBJECTIVE_KEYS))}", *at, key)
        if not isinstance(entry.get("text"), str) or not entry["text"].strip():
            bad("an objective needs text (what the player is told)", *at, "text")
        if "done" not in entry:
            bad("an objective needs done: a condition, or an event ({enter: <place>})", *at)
        after = entry.get("after", [previous] if spec.get("ordered") and previous else [])
        after = [after] if isinstance(after, str) else after
        if not isinstance(after, list) or not all(isinstance(a, str) for a in after):
            bad("after is an objective id or a list of them", *at, "after")
            after = []
        objective = _Objective(name, index, entry, list(after), bool(entry.get("optional", False)))
        quest.objectives.append(objective)
        if not objective.optional:
            previous = name
    known = [o.id for o in quest.objectives]
    for objective in quest.objectives:
        for other in objective.after:
            if other not in known:
                bad(f"after '{other}': this quest has no objective '{other}'.{refs._near(other, known)} "
                    f"It has: {', '.join(known)}", "objectives", objective.index, "after")
        objective.after = [a for a in objective.after if a in known]
    if not any(not o.optional for o in quest.objectives):
        bad("every objective is optional, so the quest could never complete", "objectives")
    state: dict[str, int] = {}

    def visit(name: str, path: list[str]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            loop = path[path.index(name):] + [name]
            bad(f"objective '{name}' waits on itself: {' -> '.join(loop)}",
                "objectives", quest.objective(name).index, "after")     # type: ignore[union-attr]
            return
        state[name] = 1
        for other in quest.objective(name).after:                       # type: ignore[union-attr]
            visit(other, path + [name])
        state[name] = 2

    for objective in quest.objectives:
        visit(objective.id, [])
    return quest


# ---- condition trees ----------------------------------------------------------------------------

def _term(key: str) -> dict[str, Any]:
    """The condition term that reads a variable key."""
    thing, sep, name = key.rpartition("#")
    if sep:
        return {"get": "instance", "id": thing, "name": name}
    scope, _, rest = key.partition(":")
    package, _, name = rest.partition(".")
    return {"get": scope, "package": package, "name": name}


def _is(key: str, value: Any) -> Tree:
    return {"op": "==", "args": [_term(key), value]}


def _all(*parts: Tree) -> Tree:
    parts = tuple(p for p in parts if p is not None and p is not True)
    return True if not parts else parts[0] if len(parts) == 1 else {"op": "and", "args": list(parts)}


def _not(tree: Tree) -> Tree:
    return {"op": "not", "args": [tree]}


def _either(test: Tree, then: Tree, otherwise: Tree) -> Tree:
    return {"op": "if", "args": [test, then, otherwise]}


def _terms(tree: Tree, over: list[str] | None = None) -> Iterable[tuple[dict[str, Any], list[str] | None]]:
    """Every node of a tree with the things its ``it`` terms range over."""
    if isinstance(tree, dict):
        yield tree, over
        for arg in tree.get("args", []):
            yield from _terms(arg, over)
        if "where" in tree:
            yield from _terms(tree["where"], tree.get("over", []))


def _shown(key: str) -> str:
    """A variable key the way a condition writes it: ``gate.gate``, ``actor.has_ticket``."""
    thing, sep, name = key.rpartition("#")
    return f"{thing.partition(':')[2] or thing}.{name}" if sep else f"{key.partition(':')[0]}.{key.partition('.')[2]}"


def _keys(term: Any, over: list[str] | None) -> list[str]:
    """The variable keys one term reads ([] for a literal, the clock or an actor built-in)."""
    from .rules import key_of

    if not isinstance(term, dict) or "get" not in term:
        return []
    if term["get"] == "it":
        return [f"{thing}#{term['name']}" for thing in over or []]
    if term["get"] in ("instance", "world") or (term["get"] == "actor" and "package" in term):
        return [key_of(term)]
    return []


# ---- compiling ----------------------------------------------------------------------------------

class Quests:
    """The compiled quests of a package: what the manifest lists and the rules that run them."""

    def __init__(self, declarations: Declarations) -> None:
        self.declarations = declarations
        self.package = declarations.package
        self.quests: dict[str, dict[str, Any]] = {}     # id -> the manifest entry
        self.rules: list[dict[str, Any]] = []

    def __bool__(self) -> bool:
        return bool(self.quests)

    def status_key(self, quest: str) -> str:
        return self._quest(quest)["status"]

    def _quest(self, quest: str) -> dict[str, Any]:
        if quest not in self.quests:
            raise QuestError(f"no quest '{quest}'.{refs._near(quest, self.quests)} "
                             + (f"Quests: {refs._listed(self.quests)}" if self.quests else "There are none"))
        return self.quests[quest]

    def _holds(self, quest: dict[str, Any]) -> Tree:
        """The condition that the actor may hold ``quest`` (None: anyone may)."""
        return None if quest["holder"] == "any" else {"op": "==", "args": [dict(_KIND_OF_ACTOR), quest["holder"]]}

    def _completes(self, quest: dict[str, Any]) -> dict[str, Any]:
        """The effect that completes an active quest whose required objectives are all done."""
        status = quest["status"]
        done = [_term(o["done"]) for o in quest["objectives"] if not o["optional"]]
        return {"set": status, "to": _either(_all(_is(status, "active"), *done), "complete", _term(status))}

    def effect(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """``{start_quest: id}``, ``{fail_quest: id}`` or ``{complete_objective: quest.objective}``
        as ``set`` effects of a rule. Each only acts from the state it expects, so it is safe
        whatever else fired."""
        if "start_quest" in spec:
            quest = self._quest(str(spec["start_quest"]))
            status = quest["status"]
            return [{"set": status, "to": _either(_all(_is(status, "available"), self._holds(quest)),
                                                 "active", _term(status))}]
        if "fail_quest" in spec:
            status = self._quest(str(spec["fail_quest"]))["status"]
            return [{"set": status, "to": _either(_is(status, "active"), "failed", _term(status))}]
        if "complete_objective" in spec:
            name, _, objective = str(spec["complete_objective"]).partition(".")
            quest = self._quest(name)
            entry = next((o for o in quest["objectives"] if o["id"] == objective), None)
            if entry is None:
                known = [o["id"] for o in quest["objectives"]]
                raise QuestError(f"quest '{name}' has no objective '{objective}'.{refs._near(objective, known)} "
                                 f"It has: {', '.join(known)} (write {name}.<objective>)")
            latches = {o["id"]: o["done"] for o in quest["objectives"]}
            opened = _all(_is(quest["status"], "active"), *(_term(latches[a]) for a in entry["after"]))
            out = [{"set": entry["done"], "to": {"op": "or", "args": [_term(entry["done"]), opened]}}]
            return out if entry["optional"] else [*out, self._completes(quest)]
        raise QuestError(f"not a quest effect: {sorted(spec)} (start_quest, fail_quest, complete_objective)")

    def program(self, rules: Iterable[dict[str, Any]] = (), bindings: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
        """A rules program (``geogen.rules``) that runs these quests, with other rules after them."""
        return self.declarations.program([*self.rules, *rules], bindings)

    def to_json(self) -> dict[str, Any]:
        """The manifest's ``quests`` section."""
        return {"format": FORMAT, "version": VERSION, "schema": SCHEMA, "package": self.package,
                "quests": self.quests, "rules": self.rules}

    # -- reading a running machine ---------------------------------------------------------------

    def status(self, machine: Any, quest: str, actor: str) -> str:
        """The lifecycle state of ``quest`` for ``actor`` in a ``rules.Machine``."""
        return machine.get(self.status_key(quest), actor)

    def report(self, machine: Any, actor: str) -> dict[str, dict[str, Any]]:
        """Every quest's state for ``actor``: what a quest log shows.

        ``{id: {title, status, objectives: [{id, text, optional, state: done | open | waiting,
        progress?: [n, of]}]}}``. An objective is open while its quest is active and
        everything it waits on is done.
        """
        out = {}
        for name, quest in self.quests.items():
            status = machine.get(quest["status"], actor)
            done = {o["id"]: machine.get(o["done"], actor) for o in quest["objectives"]}
            objectives = []
            for o in quest["objectives"]:
                state = ("done" if done[o["id"]] else
                         "open" if status == "active" and all(done[a] for a in o["after"]) else "waiting")
                entry = {"id": o["id"], "text": o["text"], "optional": o["optional"], "state": state}
                if "progress" in o:
                    entry["progress"] = [int(machine.value(o["progress"]["value"], actor)), o["progress"]["of"]]
                objectives.append(entry)
            out[name] = {"title": quest["title"], "status": status, "objectives": objectives}
        return out


class _Compiler:
    def __init__(self, declarations: Declarations, by_name: bool, selectors: dict[str, list[str]] | None,
                 rules: Iterable[dict[str, Any]]) -> None:
        self.decls = declarations
        self.package = declarations.package
        self.by_name = by_name                   # no scene: thing 'x' is '<package>:x', unchecked
        self.selectors = selectors or {}
        self.other_rules = list(rules)
        self.out = Quests(declarations)
        self.problems: dict[str, None] = {}
        self.quests: list[_Quest] = []
        self.quest_rules: dict[str, list[dict[str, Any]]] = {}

    def bad(self, where: str, message: str) -> None:
        self.problems[f"{where}: {message}"] = None

    # -- names ---------------------------------------------------------------------------------

    def uid(self, node: SceneNode) -> str:
        known = node.meta.get("uid")
        return (known if isinstance(known, str) and known.startswith(self.package + ":")
                else refs.node_id(node, self.package, node.root))

    def thing(self, quest: _Quest, text: Any, member: bool = False) -> tuple[str, SceneNode | None, str | None]:
        """(ID, node, member) of ``<reference>[.<member>]``; QuestError when it names nothing."""
        if not isinstance(text, str) or not text.strip():
            raise QuestError(f"expected a reference, got {text!r}")
        if self.by_name:
            head, dot, tail = text.rpartition(".") if member else (text, "", "")
            return (f"{self.package}:{head if dot else text}", None, tail if dot else None)
        try:
            ref = refs.parse(text, member=member)
            node = refs.resolve(quest.node, ref)
        except refs.RefError as exc:
            raise QuestError(exc.message) from None
        if not refs.identified(node):
            raise QuestError(f"'{text}' is a plain part, which has no ID: name an asset, a place, a room, "
                             f"or something tagged or with interactions")
        return self.uid(node), node, ref.member

    def condition(self, quest: _Quest, text: Any) -> Tree:
        """A condition compiled and resolved; QuestError for one that can't be, or that reads
        state nobody declares."""
        try:
            tree = conditions.compile_condition(text)
            if self.by_name:
                for node in conditions.walk(tree):
                    if node.get("get") == "ref":
                        node.update(get="instance", id=f"{self.package}:{node.pop('ref')}")
                    elif node.get("get") in ("world", "actor") and node["name"] not in conditions.ACTOR_BUILTINS:
                        node.setdefault("package", self.package)
                    elif "select" in node:
                        if node["select"] not in self.selectors:
                            raise ConditionError(f"no things given for selector '{node['select']}'")
                        node["over"] = list(self.selectors[node.pop("select")])
            else:
                tree = conditions.resolve(tree, quest.node, self.package)
            conditions.validate(tree)
        except (ConditionError, refs.RefError) as exc:
            raise QuestError(str(exc)) from None
        undeclared = self.decls.check_condition(tree)
        if undeclared:
            raise QuestError("; ".join(undeclared))
        for node, over in _terms(tree):         # a choice compared with a value it can't have
            if node.get("op") not in ("==", "!=", "in"):
                continue
            args = node["args"]
            for term in args:
                for key in _keys(term, over):
                    spec = self.decls.spec(key) or {}
                    for value in (a for a in args if isinstance(a, str)):
                        if spec.get("type") == "choice" and value not in spec["choices"]:
                            raise QuestError(f"'{_shown(key)}' is never '{value}'.{refs._near(value, spec['choices'])} "
                                             f"It is one of: {', '.join(spec['choices'])}")
        return tree

    def key(self, quest: _Quest, text: Any) -> str:
        """The variable key of ``<thing>.<state>``, ``world.<name>`` or ``actor.<name>``."""
        tree = self.condition(quest, text)
        keys = _keys(tree, None)
        if not keys:
            raise QuestError(f"'{text}' isn't state: write <thing>.<state>, world.<name> or actor.<name>")
        return keys[0]

    # -- one quest -----------------------------------------------------------------------------

    def declare(self, quest: _Quest) -> None:
        """The quest's variables, so every quest's conditions can read every other's."""
        spec = quest.spec
        quest.status = f"actor:{self.package}.quest_{quest.id}"
        variables = {quest.status: {"type": "choice", "choices": list(STATES),
                                    "default": "unavailable" if "available" in spec else "available"}}
        for objective in quest.objectives:
            objective.latch = f"{quest.status}_{objective.id}"
            variables[objective.latch] = {"type": "bool", "default": False}
            done = objective.spec.get("done")
            if isinstance(done, dict) and "count" in done:
                count = done["count"]
                if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                    self.bad(quest.where("objectives", objective.index, "done", "count"),
                             f"count is a whole number of times, got {count!r}")
                else:
                    objective.count = count
                    objective.counter = f"{objective.latch}_count"
                    variables[objective.counter] = {"type": "number", "default": 0}
        for key, variable in variables.items():
            try:
                self.decls.declare(key, variable)
            except StateError:
                self.bad(quest.where("id"), f"quest state '{key.partition('.')[2]}' is already declared: another "
                                            f"quest, objective or the scene's state has that name")

    def event(self, quest: _Quest, objective: _Objective) -> None:
        """An event objective's ``done: {enter: gallery}`` as the ``on`` of its rule."""
        done = objective.spec["done"]
        sources = [k for k in EVENT_SOURCES if k in done]
        for key in done:
            if key not in EVENT_KEYS:
                hint = (" The clock is state, not an event: write a condition (clock.night)" if key == "clock"
                        else refs._near(str(key), EVENT_KEYS))
                raise QuestError(f"unknown key '{key}'.{hint} Events: {', '.join(EVENT_SOURCES)}")
        if len(sources) != 1:
            raise QuestError(f"an event is one of {', '.join(EVENT_SOURCES)}")
        kind = sources[0]
        if "to" in done and kind != "changed":
            raise QuestError("only changed: takes to:")
        if kind in ("enter", "leave"):
            subject, node, _ = self.thing(quest, done[kind])
            if node is not None and node.meta.get("type") != "place" and not isinstance(node.meta.get("room"), dict):
                here = [n.name for n in quest.node.iter_nodes() if n.meta.get("type") == "place"]  # type: ignore[union-attr]
                raise QuestError(f"'{done[kind]}' is not a place or a room. "
                                 + (f"Places here: {refs._listed(here)}" if here else "Declare one under places:"))
            objective.on = {"type": kind, "subject": subject}
        elif kind == "signal":
            subject, _, name = self.thing(quest, done[kind], member=True)
            if not name:
                raise QuestError(f"write signal: <thing>.<name>, got '{done[kind]}'")
            objective.on = {"type": "signal", "subject": subject, "name": name}
        elif kind == "use":
            subject, _, name = self.thing(quest, done[kind], member=True)
            known = [k.rpartition("#")[2] for k in self.decls.interactions if k.rpartition("#")[0] == subject]
            if name is not None and name not in known or not known:
                raise QuestError(f"'{subject}' has no interaction{'' if name is None else f' {name!r}'} to use."
                                 f"{refs._near(name or '', known)} "
                                 + (f"It has: {', '.join(known)}" if known else "It has none"))
            objective.on = {"type": "use", "subject": subject, **({"name": name} if name else {})}
        else:
            key = self.key(quest, done[kind])
            objective.on = {"type": "changed", "key": key}
            if "to" in done:
                spec = self.decls.spec(key) or {}
                if not fits(spec, done["to"]):
                    raise QuestError(f"'{done[kind]}' is a {spec.get('type')}"
                                     + (f" ({', '.join(spec['choices'])})" if spec.get("type") == "choice" else "")
                                     + f": it can't change to {done['to']!r}")
                objective.on["to"] = done["to"]

    def effects(self, quest: _Quest, listed: Any, *at: str | int) -> list[dict[str, Any]]:
        """Authored effects (``EFFECTS``) as the ``do`` of a rule."""
        out: list[dict[str, Any]] = []
        if not isinstance(listed, list):
            self.bad(quest.where(*at), "effects are a list: [{set: actor.name, to: true}, ...]")
            return out
        for index, effect in enumerate(listed):
            where = quest.where(*at, index)
            try:
                kinds = [k for k in EFFECTS if isinstance(effect, dict) and k in effect]
                if len(kinds) != 1:
                    raise QuestError(f"an effect is one of {', '.join(EFFECTS)}")
                kind = kinds[0]
                unknown = set(effect) - {kind, *({"set": ["to"], "emit": ["of"]}.get(kind, []))}
                if unknown:
                    raise QuestError(f"unknown keys {sorted(unknown)} on a {kind} effect")
                if kind == "set":
                    key = self.key(quest, effect["set"])
                    if "to" not in effect:
                        raise QuestError("set needs to: a value or an expression")
                    value, spec = effect["to"], self.decls.spec(key) or {}
                    if isinstance(value, float) and value.is_integer():
                        value = int(value)
                    if isinstance(value, str) and not fits(spec, value):
                        value = self.condition(quest, value)        # an expression: "actor.visits + 1"
                    elif not fits(spec, value):
                        raise QuestError(f"'{effect['set']}' is a {spec.get('type')}: it can't be {value!r}")
                    out.append({"set": key, "to": value})
                elif kind == "emit":
                    if not isinstance(effect["emit"], str) or not effect["emit"]:
                        raise QuestError("emit takes a signal name")
                    subject = (self.thing(quest, effect["of"])[0] if "of" in effect
                               else self.uid(quest.node) if quest.node is not None else f"{self.package}:")
                    out.append({"emit": effect["emit"], "subject": subject})
                else:
                    out.extend(self.out.effect({kind: effect[kind]}))
            except QuestError as exc:
                self.bad(where, str(exc))
        return out

    def compile(self, quest: _Quest) -> None:
        spec, status = quest.spec, quest.status
        entry: dict[str, Any] = {
            "title": str(spec.get("title", "")), "holder": spec.get("holder", "player"),
            "start": spec.get("start", "given"), "status": status, "objectives": []}
        if spec.get("description"):
            entry["description"] = str(spec["description"])
        guards: dict[str, Tree] = {}
        for name in ("available", "fail"):
            if name in spec:
                try:
                    guards[name] = entry[name] = self.condition(quest, spec[name])
                except QuestError as exc:
                    self.bad(quest.where(name), str(exc))
        latches = {o.id: o.latch for o in quest.objectives}
        for objective in quest.objectives:
            at = ("objectives", objective.index, "done")
            done, subjects = objective.spec.get("done"), []
            try:
                if isinstance(done, dict):
                    self.event(quest, objective)
                    if "when" in done:
                        objective.when = self.condition(quest, done["when"])
                    subjects = [objective.on["subject"]] if objective.on.get("subject") else []   # type: ignore[union-attr]
                elif done is not None:
                    objective.tree = self.condition(quest, done)
                    if objective.latch in _reads(objective.tree):
                        raise QuestError(f"objective '{objective.id}' waits on itself: its condition reads its "
                                         f"own done state")
                    subjects = list(dict.fromkeys(
                        thing for node, _ in _terms(objective.tree)
                        for thing in ([node["id"]] if node.get("get") == "instance" else node.get("over", []))))
            except QuestError as exc:
                self.bad(quest.where(*at), str(exc))
            out: dict[str, Any] = {
                "id": objective.id, "text": str(objective.spec.get("text", "")), "optional": objective.optional,
                "after": list(objective.after), "done": objective.latch}
            if objective.counter:
                out["count"] = {"key": objective.counter, "of": objective.count}
                out["progress"] = {"value": _term(objective.counter), "of": objective.count}
            elif isinstance(objective.tree, dict) and objective.tree.get("op") == "all":
                out["progress"] = {"value": {**objective.tree, "op": "count"}, "of": len(objective.tree["over"])}
            marker = objective.spec.get("marker")
            try:
                if marker == "hidden":
                    out["marker"], subjects = {"hidden": True}, []
                elif isinstance(marker, dict):
                    if set(marker) - MARKER_KEYS:
                        raise QuestError(f"a marker is hidden, a reference or {{at, area}}, got {marker!r}")
                    if "at" in marker:
                        subjects = [self.thing(quest, marker["at"])[0]]
                    if "area" in marker:
                        out["marker"] = {"area": float(marker["area"])}
                elif marker is not None:
                    subjects = [self.thing(quest, marker)[0]]
            except (QuestError, TypeError, ValueError) as exc:
                self.bad(quest.where("objectives", objective.index, "marker"), str(exc))
            out["subjects"] = subjects
            objective.out = out
            entry["objectives"].append(out)
        self.out.quests[quest.id] = entry

        rule = f"quest:{quest.id}:"
        holds = self.out._holds(entry)

        def move(old: str, new: str) -> dict[str, Any]:
            return {"set": status, "to": _either(_is(status, old), new, _term(status))}

        rules: list[dict[str, Any]] = []
        if "available" in guards:
            rules.append({"id": rule + "available", "when": _all(_is(status, "unavailable"), holds, guards["available"]),
                          "do": [move("unavailable", "available")]})
            rules.append({"id": rule + "unavailable", "when": _all(_is(status, "available"), _not(guards["available"])),
                          "do": [move("available", "unavailable")]})
        if entry["start"] == "auto":
            rules.append({"id": rule + "start", "when": _all(_is(status, "available"), holds),
                          "do": self.out.effect({"start_quest": quest.id})})
        for objective in quest.objectives:
            if objective.on is None and objective.tree is None:
                continue                            # already reported
            opened = _all(_is(status, "active"), _not(_term(objective.latch)),
                          *(_term(latches[a]) for a in objective.after))
            if objective.counter:
                counter = _term(objective.counter)
                do = [{"set": objective.counter, "to": {"op": "+", "args": [counter, 1]}},
                      {"set": objective.latch, "to": {"op": ">=", "args": [counter, objective.count]}}]
            else:
                do = [{"set": objective.latch, "to": True}]
            if not objective.optional:
                do.append(self.out._completes(entry))
            compiled: dict[str, Any] = {"id": rule + objective.id}
            if objective.on is not None:
                compiled["on"] = objective.on
                compiled["when"] = _all(opened, objective.when)
            else:
                compiled["when"] = _all(opened, objective.tree)
            rules.append({**compiled, "do": do})
        if "fail" in guards:
            rules.append({"id": rule + "fail", "when": _all(_is(status, "active"), guards["fail"]),
                          "do": [move("active", "failed")]})
        self.quest_rules.setdefault(quest.id, []).extend(rules)

    # -- can it be done? -----------------------------------------------------------------------

    def causes(self) -> None:
        """Every objective must be something the world can bring about."""
        from .rules import _reads as reads

        rules = [*self.decls.signal_rules(), *self.out.rules, *self.other_rules]
        settable: dict[str, set[Any] | None] = {}       # key -> the values rules set it to (None: any)
        emitted: set[tuple[str | None, str]] = set()
        for rule in rules:
            for effect in rule.get("do", []):
                if "set" in effect:
                    value = effect["to"]
                    if isinstance(value, dict) or settable.get(effect["set"], set()) is None:
                        settable[effect["set"]] = None
                    else:
                        settable.setdefault(effect["set"], set()).add(value)     # type: ignore[union-attr]
                elif "emit" in effect:
                    emitted.add((effect.get("subject"), effect["emit"]))

        def reachable(key: str) -> set[Any] | None:
            """The values a variable can take (None: any of its type)."""
            if key.endswith("_locked"):                 # locks are the player's to turn
                return None
            sets = settable.get(key, set())
            if sets is None:
                return None
            if key not in self.decls.interactions:
                return {(self.decls.spec(key) or {}).get("default"), *sets}
            machine = self.decls.interactions[key]
            seen, queue = set(), [machine["initial"], *sets]
            while queue:
                state = queue.pop()
                if state in seen or state not in machine["states"]:
                    continue
                seen.add(state)
                queue.extend(machine["states"][state].get(link) for link in ("next", "then")
                             if machine["states"][state].get(link) is not None)
            return seen

        def changes(key: str) -> bool:
            values = reachable(key)
            return values is None or len(values) > 1

        def needs(tree: Tree, over: list[str] | None) -> Iterable[str]:
            """State a condition can only be true with, that can't be reached."""
            if not isinstance(tree, dict) or "op" not in tree:
                return
            if tree["op"] == "and":
                for arg in tree["args"]:
                    yield from needs(arg, over)
            elif tree["op"] in ("all", "any"):
                found = [list(needs(tree["where"], [thing])) for thing in tree["over"]]
                if tree["op"] == "all" or all(found):
                    for problems in found:
                        yield from problems
            elif tree["op"] in ("==", "in"):
                wanted = [a for a in tree["args"] if not isinstance(a, dict)]
                for term in tree["args"]:
                    for key in _keys(term, over):
                        values = reachable(key)
                        if wanted and values is not None and not any(w in values for w in wanted):
                            start = (self.decls.spec(key) or {}).get("default")
                            yield (f"nothing moves '{_shown(key)}' to {' or '.join(repr(w) for w in wanted)}: "
                                   f"it starts {start!r}, and no use or rule takes it there")

        for quest in self.quests:
            for objective in quest.objectives:
                where = quest.where("objectives", objective.index, "done")
                if objective.tree is not None:
                    keys = list(dict.fromkeys(reads(objective.tree)))
                    live = any(n.get("get") == "clock" or (n.get("get") == "actor" and "package" not in n)
                               for n, _ in _terms(objective.tree))     # the clock, or where the actor is
                    unreachable = list(dict.fromkeys(needs(objective.tree, None)))
                    if unreachable:
                        for problem in unreachable:
                            self.bad(where, f"objective '{objective.id}' can't be done: {problem}")
                    elif not keys and not live:
                        self.bad(where, f"objective '{objective.id}' reads no state, so nothing can make it true")
                    elif not live and not any(changes(k) for k in keys):
                        self.bad(where, f"objective '{objective.id}' can't be done: nothing changes what it reads "
                                        f"({', '.join(_shown(k) for k in keys[:6])}): no use or rule sets it")
                elif objective.on is not None:
                    on = objective.on
                    if on["type"] == "signal":
                        thing, name = on["subject"], on["name"]
                        emits = {state["emit"] for key, machine in self.decls.interactions.items()
                                 if key.rpartition("#")[0] == thing
                                 for state in machine["states"].values() if state.get("emit")}
                        declared = {s["name"] for s in self.decls.signals.values() if s["subject"] == thing}
                        raised = emits | declared | {n for s, n in emitted if s == thing}
                        if name not in raised:
                            self.bad(where, f"objective '{objective.id}' can't be done: '{thing}' never raises "
                                            f"'{name}'.{refs._near(name, raised)} "
                                            + (f"It raises: {refs._listed(raised)}" if raised else "It raises nothing"))
                    elif on["type"] == "use":
                        usable = any(state.get("next") is not None
                                     for key, machine in self.decls.interactions.items()
                                     if key.rpartition("#")[0] == on["subject"]
                                     and on.get("name") in (None, key.rpartition("#")[2])
                                     for state in machine["states"].values())
                        if not usable:
                            self.bad(where, f"objective '{objective.id}' can't be done: '{on['subject']}' can't be "
                                            f"used (no state of it has a next; only rules move it)")
                    elif on["type"] == "changed":
                        values = reachable(on["key"])
                        if values is not None and (len(values) < 2 or ("to" in on and on["to"] not in values)):
                            self.bad(where, f"objective '{objective.id}' can't be done: nothing changes "
                                            f"'{_shown(on['key'])}'" + (f" to {on['to']!r}" if "to" in on else "")
                                            + ": no use or rule sets it")


def _reads(tree: Tree) -> list[str]:
    from .rules import _reads as reads

    return reads(tree)


def compile_quests(specs: Iterable[dict[str, Any] | tuple[dict[str, Any], Any, SceneNode | None]],
                   declarations: Declarations, selectors: dict[str, list[str]] | None = None,
                   rules: Iterable[dict[str, Any]] = ()) -> Quests:
    """Compile quest specs against ``declarations`` (their variables are added to it).

    A spec is the dict of a quest file, or ``(spec, source file, scene node)``. With a
    scene node, references resolve from it (``collect`` does this). Without one, a thing
    ``x`` is simply ``<package>:x`` and ``selectors`` gives the things each selector
    matches: enough to compile a quest for a trace or a test with no scene.
    ``rules`` are the package's other compiled rules: what else can set state.
    Raises ``QuestError`` listing every problem.
    """
    entries = [(s, None, None) if isinstance(s, dict) else s for s in specs]
    compiler = _Compiler(declarations, all(node is None for _, _, node in entries), selectors, rules)
    problems: list[str] = []
    for spec, source, node in entries:
        quest = _parse(spec, source, node, problems)
        if quest is None:
            continue
        other = next((q for q in compiler.quests if q.id == quest.id), None)
        if other is not None:
            problems.append(f"{quest.where('id')}: quest '{quest.id}' is contributed twice (also by "
                            f"{other.source or 'another scene'}): a quest belongs to one placed scene")
            continue
        compiler.quests.append(quest)
    compiler.problems.update(dict.fromkeys(problems))
    for quest in compiler.quests:
        compiler.declare(quest)
    for quest in compiler.quests:               # every entry exists before effects name quests
        compiler.compile(quest)
    for quest in compiler.quests:
        rules_of = compiler.quest_rules.get(quest.id, [])
        if quest.spec.get("rewards"):
            do = compiler.effects(quest, quest.spec["rewards"], "rewards")
            if do:
                rules_of.append({"id": f"quest:{quest.id}:rewards",
                                 "on": {"type": "changed", "key": quest.status, "to": "complete"}, "do": do})
        compiler.out.rules.extend(rules_of)
    if not compiler.problems:
        compiler.causes()
    if not compiler.problems:
        from .rules import check_program

        compiler.problems.update(dict.fromkeys(check_program(compiler.out.program(compiler.other_rules))))
    if compiler.problems:
        raise QuestError("\n".join(compiler.problems))
    return compiler.out


def collect(root: SceneNode, package: str, rules: Iterable[dict[str, Any]] = ()) -> Quests:
    """The quests of the built scene ``root``, compiled, with its state declarations
    (``Quests.declarations``: ``state.collect`` plus the quests' variables).

    Raises ``StateError`` or ``QuestError`` listing every problem.
    """
    from .state import collect as collect_state

    declarations = collect_state(root, package)
    specs = [(entry["spec"], entry.get("source"), node)
             for node in root.iter_nodes() for entry in node.meta.get(META) or []]
    if not specs:
        return Quests(declarations)
    return compile_quests(specs, declarations, rules=rules)


def summary(root: SceneNode, package: str | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The manifest's ``state`` and ``quests`` sections for ``root`` (None for one with nothing in it)."""
    if package is None:
        package = str(root.meta.get("uid", "")).partition(":")[0] or root.name
    found = collect(root, package)
    return (found.declarations.to_json() if found.declarations else None, found.to_json() if found else None)
