"""Wiring: the rules and bindings scenes and assets write, compiled to a rules program.

Three keys, in a scene or an asset (``docs/rules.md`` has the semantics)::

    rules:
      - id: open_gate                       # optional: names it in traces and errors
        on: {signal: lever.pulled}          # an event; without one, `when` is watched
        when: "world.power"                 # a condition (geogen.conditions)
        do:                                 # effects, in order
          - {set: gate.gate, to: open}      # <thing>.<state> | world.<name> | actor.<name>
          - {input: lamp.light, value: true}  # an input an asset declares
          - {emit: opened, of: gate}
          - {start_quest: lights_out}       # also fail_quest, complete_objective
          - {notify: "The gate opens"}      # tell the actor; {play: <sound>} (geogen.rules.SOUNDS)
        after: 2                            # seconds before the effects run
        once: true                          # once per actor (once in all with no actor)
        if: "{quests}"                      # build time: no such rule when the param is 0 (wires too)
    wire:
      - {from: lever.pulled, to: gate.open}     # a signal drives an input (`value`, `when`, `after` too)
    bind:
      lamp.lit: "plate.occupants > 0 and world.power"     # a standing relation: never set, never saved

Events (``on``): ``{enter | leave: <place or room>}``, ``{signal: <thing>.<name>}``,
``{use: <thing>[.<interaction>]}``, ``{changed: <state>, to}``, ``{start: true}`` (a new game),
``{clock: night | day}``, ``{travel: true | <scene>}``; ``actor: player | npc | none | any``
narrows who.

Names are relative to where they are written, so every placed instance wires itself:
``self`` is the instance (``self.lit``, ``{signal: self.pulled}``: an asset, or a placed
scene with ``scope: instance`` state) and other names are things inside it (in a scene,
references from that scene).

The loaders only record these (``meta.wiring``, not exported). ``geogen.quests.collect``
compiles them with the scene's quests, so a rule can give a quest and a quest can rely
on what a rule sets, and raises ``QuestError`` listing every problem with file and line.
Exports put the result in the manifest and chunk index as ``rules`` (a ``geogen-rules``
program without its variables and interactions, which ``state`` holds).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core.node import SceneNode

META = "wiring"                   # node.meta key the loaders record wiring under
KEYS = ("rules", "wire", "bind")
RULE_KEYS = {"id", "on", "when", "do", "after", "once", "if"}
WIRE_KEYS = {"from", "to", "value", "when", "after", "if"}
RULE_EVENTS = {
    "enter": "an actor enters this place or room (a reference)",
    "leave": "an actor leaves this place or room",
    "signal": "<thing>.<name>: the thing raises this signal",
    "use": "<thing> or <thing>.<interaction>: an actor uses it",
    "changed": "<thing>.<state> | world.<name> | actor.<name> changes (`to`: only to that value)",
    "start": "true: a new game begins for an actor",
    "clock": "night | day: dusk or dawn (never has an actor)",
    "travel": "true, or a scene name: an actor travels (there)",
}
EVENT_KEYS = {*RULE_EVENTS, "to", "actor"}


@dataclass
class Source:
    """One scene's or asset instance's wiring, as written."""
    spec: dict[str, Any]            # {rules?, wire?, bind?}
    source: Any                     # its YAML file
    node: SceneNode                 # references resolve from here
    asset: bool                     # an asset instance: `self` is the node


def declare(root: SceneNode, data: dict[str, Any], source: Any, asset: bool) -> None:
    """Record a scene's or asset's ``rules:`` / ``wire:`` / ``bind:`` on its root."""
    spec = {key: data[key] for key in KEYS if data.get(key)}
    if spec:
        root.meta[META] = {"source": str(source) if source is not None else None, "asset": asset, "spec": spec}


def sources(root: SceneNode) -> list[Source]:
    """The wiring of every node below ``root``, in scene order."""
    return [Source(entry["spec"], entry.get("source"), node, bool(entry.get("asset")))
            for node in root.iter_nodes() for entry in [node.meta.get(META)] if entry]


def compile_wiring(compiler: Any, entries: list[Source]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(rules, bindings) for ``entries``; problems go to ``compiler`` (a ``quests._Compiler``)."""
    from .quests import QuestError, _Quest

    rules: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for entry in entries:
        owner = compiler.uid(entry.node)
        prefix = "" if owner.endswith(":") else f"{owner}#"     # the package's own scene needs none
        src = _Quest(owner, {}, entry.source, entry.node, entry.node)      # `self`: this instance
        if entry.source is None:
            src.where = lambda *keys, _o=owner: f"{_o}: " + ".".join(str(k) for k in keys)   # type: ignore[method-assign]

        def attempt(where: str, build: Any) -> Any:
            try:
                return build()
            except QuestError as exc:
                compiler.bad(where, str(exc))
                return None

        def listed(key: str, form: str) -> list[tuple[int, dict[str, Any]]]:
            value = entry.spec.get(key) or []
            if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
                compiler.bad(src.where(key), f"{key} is a list of {form}")
                return []
            return list(enumerate(value))

        for index, rule in listed("rules", "{on, when, do}"):
            at = ("rules", index)
            unknown = set(rule) - RULE_KEYS
            if unknown:
                compiler.bad(src.where(*at), f"unknown keys {sorted(unknown)}. Keys: {', '.join(sorted(RULE_KEYS))}")
                continue
            if not built(rule):
                continue
            out: dict[str, Any] = {"id": f"{prefix}{rule.get('id', f'rule_{index + 1}')}"}
            if "on" in rule:
                on = attempt(src.where(*at, "on"), lambda: _event(compiler, src, rule["on"]))
                if on is None:
                    continue
                out["on"] = on
                if on["type"] == "signal":
                    compiler.listening.append((src.where(*at, "on"), on))
            elif "when" not in rule:
                compiler.bad(src.where(*at), "a rule needs on: (an event) or when: (a condition to watch)")
                continue
            if "when" in rule:
                when = attempt(src.where(*at, "when"), lambda: compiler.condition(src, rule["when"]))
                if when is None:
                    continue
                out["when"] = when
            out["do"] = compiler.effects(src, rule.get("do"), *at, "do")
            if not out["do"]:
                if isinstance(rule.get("do"), list) and not rule["do"]:
                    compiler.bad(src.where(*at), "a rule needs do: [effects]")
                continue
            for key in ("after", "once"):
                if rule.get(key):
                    out[key] = float(rule[key]) if key == "after" else True
            rules.append(out)
            compiler.out.written[out["id"]] = src.where(*at)

        for index, wire in listed("wire", "{from: <thing>.<signal>, to: <thing>.<input>}"):
            at = ("wire", index)
            unknown = set(wire) - WIRE_KEYS
            if unknown or "from" not in wire or "to" not in wire:
                compiler.bad(src.where(*at), f"a wire is {{from: <thing>.<signal>, to: <thing>.<input>}} "
                                             f"(also value, when, after)" + (f"; unknown {sorted(unknown)}" if unknown else ""))
                continue
            if not built(wire):
                continue
            on = attempt(src.where(*at, "from"), lambda: _signal(compiler, src, wire["from"]))
            effect = {"input": wire["to"], **({"value": wire["value"]} if "value" in wire else {})}
            do = compiler.effects(src, [effect], *at, "to")
            when = attempt(src.where(*at, "when"), lambda: compiler.condition(src, wire["when"])) if "when" in wire else True
            if on is None or not do or when is None:
                continue
            out = {"id": f"{prefix}wire_{index + 1}", "on": on, "do": do}
            if when is not True:
                out["when"] = when
            if wire.get("after"):
                out["after"] = float(wire["after"])
            rules.append(out)
            compiler.out.written[out["id"]] = src.where(*at)

        bind = entry.spec.get("bind") or {}
        if not isinstance(bind, dict):
            compiler.bad(src.where("bind"), "bind is a mapping of <state>: <expression>")
            bind = {}
        for target, expression in bind.items():
            at = ("bind", target)
            key = attempt(src.where(*at), lambda: compiler.key(src, target))
            tree = attempt(src.where(*at), lambda: compiler.condition(src, expression))
            if key is not None and tree is not None:
                bindings.append({"set": key, "to": tree})
                compiler.out.written[f"bind:{key}"] = src.where(*at)
    return rules, bindings


def built(entry: dict[str, Any]) -> bool:
    """Whether a rule, wire or ``quests:`` entry is part of this build: its ``if:`` (a param
    expression, resolved when the file was loaded) isn't 0, false or ""."""
    return entry.get("if", True) not in (False, 0, "", None)


def _event(compiler: Any, src: Any, spec: Any) -> dict[str, Any]:
    from .quests import QuestError

    if not isinstance(spec, dict):
        raise QuestError(f"an event is a mapping ({{signal: lever.pulled}}), got {spec!r}")
    return compiler.event(src, spec, RULE_EVENTS, EVENT_KEYS)


def _signal(compiler: Any, src: Any, text: Any) -> dict[str, Any]:
    """The ``on`` of a wire's ``from``: a signal the thing declares or one of its interactions emits."""
    from . import refs
    from .quests import QuestError

    on = compiler.event(src, {"signal": text}, RULE_EVENTS, EVENT_KEYS)
    thing, name, decls = on["subject"], on["name"], compiler.decls
    offered = {s["name"] for s in decls.signals.values() if s["subject"] == thing}
    offered |= {state["emit"] for key, machine in decls.interactions.items() if key.rpartition("#")[0] == thing
                for state in machine["states"].values() if state.get("emit")}
    if name not in offered:
        raise QuestError(f"'{thing}' has no signal '{name}'.{refs._near(name, offered)} "
                         + (f"Its signals: {refs._listed(offered)}" if offered else "It declares none"))
    return on
