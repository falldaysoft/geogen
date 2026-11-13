"""The compiler authored gameplay content shares: references, conditions, events and effects
written in YAML, compiled to the terms, ``on`` and ``do`` of ``geogen.rules``.

Quests (``geogen.quests``), a scene's rules, wires and bindings (``geogen.wiring``), HUDs
(``geogen.hud``), dialogues (``geogen.dialogue``) and interaction guards are all compiled by one
``Compiler``, so each can name what the others declare. A ``Context`` says what a piece of
content is and where it was written; problems are collected with ``Compiler.bad`` (file and
line) and a method that can't go on raises ``ContentError``, which the caller reports the same
way. ``quests.compile_quests`` owns the run: it makes the compiler, hands it to the others and
raises everything found at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from . import conditions, refs, roots
from .conditions import ConditionError
from .core.node import SceneNode
from .rules import SOUNDS, term_keys
from .rules import skip_effect as rules_skip
from .state import LOCKED, Declarations, StateError, fits

EVENT_SOURCES = {
    "enter": "the holder enters this place or room (a reference)",
    "leave": "the holder leaves this place or room",
    "signal": "<thing>.<name>: the thing raises this signal, caused by the holder",
    "use": "<thing> or <thing>.<interaction>: the holder uses it",
    "changed": "<thing>.<state> | world.<name> | actor.<name> changes (`to`: only to that value), caused by the holder",
    "use_item": "<item id>: the holder uses it from their inventory (the item declares use:)",
}
EVENT_KEYS = {*EVENT_SOURCES, "to", "when", "count"}
EFFECTS = {
    "set": "<thing>.<state> | world.<name> | actor.<name>, with `to`: a value or an expression",
    "emit": "raise a signal of this name (`of`: the thing it comes from; default: the scene; "
            "`data`: {field: value or expression} the signal carries)",
    "input": "<thing>.<input>: drive an input an asset declares (`value` when the input takes one)",
    "start_quest": "start a quest (by id) for the actor, if it is available to them",
    "fail_quest": "fail an active quest",
    "complete_objective": "<quest>.<objective>: mark an open objective done",
    "reset": "<thing> or a selector: it and everything in it back to how it started (interactions, locks, "
             "declared variables), so a puzzle can be tried again",
    "reset_quest": "a quest back to where it started for the actor: its status, objectives and counts",
    "give": "an item (its file's id: items/<id>.yaml) for the actor to carry; `count` (default 1)",
    "take": "an item the actor carries, taken away; `count` (default 1; never below none)",
    "notify": "text to show the actor (a HUD toast)",
    "play": "a sound to play for the actor: " + ", ".join(SOUNDS),
    "skip": "move the clock forward: {minutes | hours | days: n}, or {to: \"HH:MM\"} (the next time it shows that); "
            "what comes due on the way happens in order (sleeping until morning)",
}

Tree = Any


class ContentError(ValueError):
    """Authored content that is malformed, or that the scene it is in can't support."""


@dataclass
class Context:
    """What the compiler compiles a condition or effect for: an id, the file it was written in
    (for diagnostics), the scene node references resolve from, and what ``self`` means.

    Quests are one; wiring, HUDs, dialogues and interaction guards make one each.
    ``label`` names it in diagnostics when there is no file (default: ``quest '<id>'``).
    """

    id: str
    spec: dict[str, Any] = field(default_factory=dict)
    source: Any = None              # the file it was written in, for diagnostics
    node: SceneNode | None = None   # the scene that contributes it: references resolve from here
    self_node: SceneNode | None = None  # what `self` means (rules written in an asset: the instance)
    label: str = ""

    def where(self, *keys: str | int) -> str:
        from .layout.yaml_utils import source_of

        if self.source is None:
            label = self.label or f"quest '{self.id}'"
            return f"{label}: " + ".".join(str(k) for k in keys)
        return source_of(self.source, list(keys))


def _shown(key: str) -> str:
    """A variable key the way a condition writes it: ``gate.gate``, ``actor.has_ticket``."""
    thing, sep, name = key.rpartition("#")
    return f"{thing.partition(':')[2] or thing}.{name}" if sep else f"{key.partition(':')[0]}.{key.partition('.')[2]}"


class Compiler:
    """Compiles references, conditions, events and effects for one package. ``out`` is what the
    run produces (a ``quests.Quests``): its ``wiring`` and ``rules`` are read for what rules
    emit, its ``items`` gets the items named, and its ``effect`` compiles the quest effects."""

    def __init__(self, declarations: Declarations, by_name: bool, selectors: dict[str, list[str]] | None,
                 rules: Iterable[dict[str, Any]], trees: dict[str, SceneNode] | None = None,
                 bindings: Iterable[dict[str, Any]] = (), out: Any = None) -> None:
        self.decls = declarations
        self.package = declarations.package
        self.by_name = by_name                   # no scene: thing 'x' is '<package>:x', unchecked
        self.selectors = selectors or {}
        self.other_rules = list(rules)
        self.other_bindings = list(bindings)
        self.trees = trees                       # a content package: its scenes' trees, by name
        self.out = out
        self.problems: dict[str, None] = {}
        self.listening: list[tuple[str, dict[str, Any]]] = []   # (where, on) of rules on a signal
        self.event_reads: list[tuple[str, dict[str, Any]]] = []  # (where, rule) of wiring, checked once all is known

    def bad(self, where: str, message: str) -> None:
        self.problems[f"{where}: {message}"] = None

    # -- names ---------------------------------------------------------------------------------

    def uid(self, node: SceneNode) -> str:
        known = node.meta.get("uid")
        if self.trees is not None and isinstance(known, str):
            return known                         # another package's thing, by the ID its export gave it
        return (known if isinstance(known, str) and known.startswith(self.package + ":")
                else refs.node_id(node, self.package, node.root))

    def scene_of(self, text: str) -> SceneNode:
        """The tree a content package's reference is in: it must say which scene (``island:jetty``)."""
        trees = self.trees or {}
        ref = refs.parse(text)
        if ref.package is None:
            raise ContentError(f"'{text}': say which scene it is in, e.g. '{next(iter(trees), 'scene')}:{text}' "
                             f"(this package's scenes: {refs._listed(trees)})")
        if ref.package not in trees:
            raise ContentError(f"'{text}': '{ref.package}' isn't one of this package's scenes."
                             f"{refs._near(ref.package, trees)} Its scenes: {refs._listed(trees)}")
        return trees[ref.package]

    def area(self, node: SceneNode) -> bool:
        """Whether actors can enter and leave ``node``: a place or a room."""
        if self.trees is not None:              # an index's stand-in: what the export recorded
            return node.meta.get("type") == "place" or "room" in node.tags
        return node.meta.get("type") == "place" or isinstance(node.meta.get("room"), dict)

    def thing(self, quest: Context, text: Any, member: bool = False) -> tuple[str, SceneNode | None, str | None]:
        """(ID, node, member) of ``<reference>[.<member>]``; ContentError when it names nothing."""
        if not isinstance(text, str) or not text.strip():
            raise ContentError(f"expected a reference, got {text!r}")
        if self.by_name:
            head, dot, tail = text.rpartition(".") if member else (text, "", "")
            return (f"{self.package}:{head if dot else text}", None, tail if dot else None)
        if quest.self_node is not None and (text == "self" or member and text.startswith("self.")):
            return self.uid(quest.self_node), quest.self_node, text.partition(".")[2] or None
        try:
            ref = refs.parse(text, member=member)
            node = refs.resolve(self.scene_of(text) if self.trees is not None else quest.node, ref)
        except refs.RefError as exc:
            raise ContentError(exc.message) from None
        if self.trees is not None:
            if "uid" not in node.meta:
                raise ContentError(f"'{text}' has no ID in its scene's export: name an asset, a place, a room, "
                                 f"or something tagged or with interactions")
            return node.meta["uid"], node, ref.member
        if not refs.identified(node):
            raise ContentError(f"'{text}' is a plain part, which has no ID: name an asset, a place, a room, "
                             f"or something tagged or with interactions")
        return self.uid(node), node, ref.member

    def reset(self, quest: Context, text: Any) -> list[dict[str, Any]]:
        """``reset: <thing or selector>`` as ``set`` effects: every interaction, lock and declared
        variable of the things and of everything inside them back to its default. Bound and
        runtime-driven state is left out (``_unbound`` drops what bindings turn out to own)."""
        if isinstance(text, str) and "[" in text and text != "self" and not self.by_name:
            try:
                base = self.scene_of(text) if self.trees is not None else quest.node
                nodes = refs.select(base, text)
            except refs.RefError as exc:
                raise ContentError(exc.message) from None
            things = [n.meta["uid"] if self.trees is not None else self.uid(n) for n in nodes
                      if (self.trees is not None and "uid" in n.meta) or (self.trees is None and refs.identified(n))]
            if not things:
                raise ContentError(f"reset: '{text}' matches nothing with state")
        else:
            things = [self.thing(quest, text)[0]]
        out: dict[str, Any] = {}
        decls = self.decls
        owned = lambda key: any(key.rpartition("#")[0] == t or key.startswith(t + "/") for t in things)
        for key, spec in decls.variables.items():
            if "#" in key and owned(key) and not spec.get("driven"):
                out[key] = spec["default"]
        for key, machine in decls.interactions.items():
            if owned(key):
                out[key] = machine["initial"]
                out[f"{key}{LOCKED}"] = bool(machine.get("locked", False))
        if not out:
            raise ContentError(f"reset: '{text}' has no state to reset")
        return [{"set": key, "to": value, "_reset": True} for key, value in out.items()]

    def condition(self, quest: Context, text: Any) -> Tree:
        """A condition compiled and resolved; ContentError for one that can't be, or that reads
        state nobody declares."""
        try:
            tree = conditions.compile_condition(text)
            for node in conditions.walk(tree):      # has("parcel"): the item's count, declared on first use
                if node.get("get") == "actor" and "package" not in node \
                        and str(node.get("name", "")).startswith(conditions.ITEM_PREFIX):
                    self.item(node["name"][len(conditions.ITEM_PREFIX):])
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
            elif self.trees is not None:
                for node in conditions.walk(tree):
                    if node.get("get") == "ref" or "select" in node:
                        self.scene_of(node.get("ref") or node["select"])
                tree = conditions.resolve(tree, quest.node, self.package, packages=self.trees.get,
                                          states=lambda n: self.decls.names(str(n.meta.get("uid", ""))))
            else:
                tree = conditions.resolve(tree, quest.node, self.package, self_node=quest.self_node)
            conditions.validate(tree)
        except (ConditionError, refs.RefError) as exc:
            raise ContentError(str(exc)) from None
        undeclared = self.decls.check_condition(tree)
        if undeclared:
            raise ContentError("; ".join(undeclared))
        for node, over in conditions.walk_over(tree):         # a choice compared with a value it can't have
            if node.get("op") not in ("==", "!=", "in"):
                continue
            args = node["args"]
            for term in args:
                for key in term_keys(term, over):
                    spec = self.decls.spec(key) or {}
                    for value in (a for a in args if isinstance(a, str)):
                        if spec.get("type") == "choice" and value not in spec["choices"]:
                            raise ContentError(f"'{_shown(key)}' is never '{value}'.{refs._near(value, spec['choices'])} "
                                             f"It is one of: {', '.join(spec['choices'])}")
        return tree

    def item(self, item: str) -> str:
        """The actor variable counting ``item`` (``items/<item>.yaml``), declared the first time
        anything names it; ContentError for an item there is no file for."""
        key = f"actor:{self.package}.{conditions.ITEM_PREFIX}{item}"
        if self.decls.spec(key) is None:
            from .flows import FlowError, carried

            path = roots.find(f"items/{item}.yaml")
            if not path.is_file():
                known = [p.stem for p in roots.glob("items/*.yaml")]
                raise ContentError(f"no item '{item}'.{refs._near(item, known)} "
                                 + (f"Items: {', '.join(known)}" if known else "There are no items/*.yaml"))
            try:
                self.out.items[item] = carried(path)
            except FlowError as exc:
                raise ContentError(str(exc)) from None
            self.decls.declare(key, {"type": "number", "default": 0})
        return key

    def key(self, quest: Context, text: Any) -> str:
        """The variable key of ``<thing>.<state>``, ``world.<name>`` or ``actor.<name>``."""
        tree = self.condition(quest, text)
        keys = term_keys(tree, None)
        if not keys:
            raise ContentError(f"'{text}' isn't state: write <thing>.<state>, world.<name> or actor.<name>")
        return keys[0]

    def event(self, quest: Context, done: dict[str, Any], sources: dict[str, str] = EVENT_SOURCES,
              keys: Iterable[str] = EVENT_KEYS) -> dict[str, Any]:
        """An event written in YAML (``{enter: gallery}``) as the ``on`` of a rule.
        ``sources`` are the kinds allowed here and ``keys`` everything the mapping may hold."""
        keys = {*keys}
        for key in done:
            if key not in keys:
                hint = (" The clock is state, not an event: write a condition (clock.night)"
                        if key == "clock" else refs._near(str(key), keys))
                raise ContentError(f"unknown key '{key}'.{hint} Events: {', '.join(sources)}")
        found = [k for k in sources if k in done]
        if len(found) != 1:
            raise ContentError(f"an event is one of {', '.join(sources)}")
        kind = found[0]
        on = self._event(quest, done, kind)
        if "actor" in done:
            if done["actor"] not in ("player", "npc", "none", "any"):
                raise ContentError(f"actor is player, npc, none or any, got {done['actor']!r}")
            on["actor"] = done["actor"]
        return on

    def _event(self, quest: Context, done: dict[str, Any], kind: str) -> dict[str, Any]:
        if "to" in done and kind != "changed":
            raise ContentError("only changed: takes to:")
        if kind in ("enter", "leave"):
            subject, node, _ = self.thing(quest, done[kind])
            if node is not None and not self.area(node):
                here = [n.name for n in node.root.iter_nodes() if n.meta.get("type") == "place"]
                raise ContentError(f"'{done[kind]}' is not a place or a room. "
                                 + (f"Places here: {refs._listed(here)}" if here else "Declare one under places:"))
            return {"type": kind, "subject": subject}
        if kind == "start":
            return {"type": "start"}
        if kind == "clock":
            if done[kind] not in ("night", "day"):
                raise ContentError(f"clock: night or day, got {done[kind]!r}")
            return {"type": "clock", "name": done[kind]}
        if kind == "travel":
            return {"type": "travel", **({} if done[kind] is True else {"name": str(done[kind])})}
        if kind == "signal":
            subject, _, name = self.thing(quest, done[kind], member=True)
            if not name:
                raise ContentError(f"write signal: <thing>.<name>, got '{done[kind]}'")
            return {"type": "signal", "subject": subject, "name": name}
        if kind == "use":
            subject, _, name = self.thing(quest, done[kind], member=True)
            known = [k.rpartition("#")[2] for k in self.decls.interactions if k.rpartition("#")[0] == subject]
            if name is not None and name not in known or not known:
                raise ContentError(f"'{subject}' has no interaction{'' if name is None else f' {name!r}'} to use."
                                 f"{refs._near(name or '', known)} "
                                 + (f"It has: {', '.join(known)}" if known else "It has none"))
            return {"type": "use", "subject": subject, **({"name": name} if name else {})}
        if kind == "use_item":
            item = done[kind]
            if not isinstance(item, str) or not item.isidentifier():
                raise ContentError(f"use_item: an item's id (items/<id>.yaml), got {item!r}")
            self.item(item)
            if "use" not in self.out.items.get(item, {}):
                raise ContentError(f"item '{item}' declares no use: (a verb: Eat, Drink), so no inventory offers using it")
            return {"type": "use_item", "name": item}
        key = self.key(quest, done[kind])
        on = {"type": "changed", "key": key}
        if "to" in done:
            spec = self.decls.spec(key) or {}
            if not fits(spec, done["to"]):
                raise ContentError(f"'{done[kind]}' is a {spec.get('type')}"
                                 + (f" ({', '.join(spec['choices'])})" if spec.get("type") == "choice" else "")
                                 + f": it can't change to {done['to']!r}")
            on["to"] = done["to"]
        return on

    def effects(self, quest: Context, listed: Any, *at: str | int) -> list[dict[str, Any]]:
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
                    raise ContentError(f"an effect is one of {', '.join(EFFECTS)}")
                kind = kinds[0]
                unknown = set(effect) - {kind, *({"set": ["to"], "emit": ["of", "data"], "input": ["value"],
                                                  "give": ["count"], "take": ["count"]}.get(kind, []))}
                if unknown:
                    raise ContentError(f"unknown keys {sorted(unknown)} on a {kind} effect")
                if kind == "set":
                    key = self.key(quest, effect["set"])
                    if "to" not in effect:
                        raise ContentError("set needs to: a value or an expression")
                    value, spec = effect["to"], self.decls.spec(key) or {}
                    if isinstance(value, float) and value.is_integer():
                        value = int(value)
                    if isinstance(value, str) and not fits(spec, value):
                        value = self.condition(quest, value)        # an expression: "actor.visits + 1"
                    elif not fits(spec, value):
                        raise ContentError(f"'{effect['set']}' is a {spec.get('type')}: it can't be {value!r}")
                    out.append({"set": key, "to": value})
                elif kind == "input":
                    thing, _, name = self.thing(quest, effect["input"], member=True)
                    if not name:
                        raise ContentError(f"write input: <thing>.<input>, got '{effect['input']}'")
                    try:
                        wired = self.decls.input(thing, name)
                    except StateError as exc:
                        raise ContentError(str(exc)) from None
                    if ("to" in wired) == ("value" in effect):
                        raise ContentError(f"input '{name}' " + ("sets a fixed value: it takes no value" if "to" in wired
                                                               else "takes a value: add value:"))
                    value, spec = wired.get("to", effect.get("value")), self.decls.spec(wired["set"]) or {}
                    if isinstance(value, dict):
                        pass                                # a tree the build compiled (a source's emit)
                    elif isinstance(value, str) and not fits(spec, value):
                        value = self.condition(quest, value)
                    elif not fits(spec, value):
                        raise ContentError(f"input '{name}' is a {spec.get('type')}: it can't be {value!r}")
                    out.append({"set": wired["set"], "to": value})
                elif kind == "emit":
                    if not isinstance(effect["emit"], str) or not effect["emit"]:
                        raise ContentError("emit takes a signal name")
                    subject = (self.thing(quest, effect["of"])[0] if "of" in effect
                               else self.uid(quest.node) if quest.node is not None else f"{self.package}:")
                    data = effect.get("data") or {}
                    if not isinstance(data, dict):
                        raise ContentError("emit data is a mapping of field: value")
                    compiled = {str(field): (value if isinstance(value, (bool, int, float)) else self.condition(quest, value))
                                for field, value in data.items()}
                    out.append({"emit": effect["emit"], "subject": subject, **({"data": compiled} if compiled else {})})
                elif kind == "reset":
                    out.extend(self.reset(quest, effect["reset"]))
                elif kind in ("give", "take"):
                    count = effect.get("count", 1)
                    if not isinstance(effect[kind], str) or not effect[kind].isidentifier():
                        raise ContentError(f"{kind} takes an item's id (items/<id>.yaml), got {effect[kind]!r}")
                    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                        raise ContentError(f"count is a whole number of items, got {count!r}")
                    key = self.item(effect[kind])
                    held = {"get": "actor", "package": self.package, "name": conditions.ITEM_PREFIX + effect[kind]}
                    out.append({"set": key, "to": {"op": "+", "args": [held, count]} if kind == "give" else
                                {"op": "max", "args": [{"op": "-", "args": [held, count]}, 0]}})
                elif kind == "notify":
                    if not isinstance(effect["notify"], str) or not effect["notify"].strip():
                        raise ContentError("notify takes the text to show")
                    out.append({"notify": effect["notify"]})
                elif kind == "skip":
                    try:
                        out.append(rules_skip(effect["skip"]))
                    except ValueError as exc:
                        raise ContentError(str(exc)) from None
                elif kind == "play":
                    if effect["play"] not in SOUNDS:
                        raise ContentError(f"play: no sound {effect['play']!r}.{refs._near(str(effect['play']), SOUNDS)} "
                                         f"Sounds: {', '.join(SOUNDS)}")
                    out.append({"play": effect["play"]})
                else:
                    out.extend(self.out.effect({kind: effect[kind]}))
            except ContentError as exc:
                self.bad(where, str(exc))
        return out

    def event_data(self, on: dict[str, Any], seen: frozenset[str] = frozenset()) -> set[str] | None:
        """The data fields the events an ``on`` matches carry (``event.<field>``); None when
        that can't be known here (a signal the export doesn't declare)."""
        if on["type"] == "changed":
            return {"from", "to"}
        if on["type"] != "signal":
            return set()
        key = f"{on['subject']}#{on['name']}"
        declared = self.decls.signals.get(key)
        if declared is None:
            emits = any(state.get("emit") == on["name"] for k, m in self.decls.interactions.items()
                        if k.rpartition("#")[0] == on["subject"] for state in m["states"].values())
            if emits or on["name"] == "locked":
                return set()
            fields = {f for rule in (*self.out.wiring, *self.out.rules, *self.other_rules) for e in rule.get("do", [])
                      if e.get("emit") == on["name"] and e.get("subject") == on["subject"] for f in e.get("data", {})}
            return fields or None
        if "from" in declared and key not in seen:
            return self.event_data(declared["from"], seen | {key})
        return set(declared.get("data", {}))

    def check_event_reads(self, where: str, rule: dict[str, Any]) -> None:
        """A rule reading ``event.<field>`` its event never carries: a misspelt field."""
        trees = [rule.get("when"), *(e.get("to") for e in rule["do"]),
                 *(t for e in rule["do"] for t in (e.get("data") or {}).values())]
        read = list(dict.fromkeys(f for t in trees for f in conditions.event_fields(t)))
        if not read:
            return
        on = rule.get("on")
        if on is None:
            self.bad(where, f"event.{read[0]}: only a rule with on: (an event) has event data")
            return
        carried = self.event_data(on)
        for name in read:
            if carried is not None and name not in carried:
                self.bad(where, f"event.{name}: a {on['type']} event here carries "
                                + (f"{', '.join(sorted(carried))}" if carried else "no data")
                                + refs._near(name, carried))

    def raised(self, thing: str) -> set[str]:
        """The signals a thing can raise: its interactions' emits (and ``locked``), what it or
        an NPC definition declares, and what rules emit from it."""
        machines = [m for key, m in self.decls.interactions.items() if key.rpartition("#")[0] == thing]
        out = {state["emit"] for machine in machines for state in machine["states"].values() if state.get("emit")}
        out |= {"locked"} if machines else set()
        out |= {s["name"] for s in self.decls.signals.values() if s["subject"] == thing}
        for rule in (*self.out.wiring, *self.out.rules, *self.other_rules):
            out |= {e["emit"] for e in rule.get("do", []) if e.get("emit") and e.get("subject") == thing}
        return out

    def heard(self) -> None:
        """A rule that waits for a signal nothing raises never fires: a misspelt name."""
        for where, on in self.listening:
            raised = self.raised(on["subject"])
            if on["name"] not in raised:
                self.bad(where, f"'{on['subject']}' never raises '{on['name']}'.{refs._near(on['name'], raised)} "
                                + (f"It raises: {refs._listed(raised)}" if raised else "It raises nothing"))
