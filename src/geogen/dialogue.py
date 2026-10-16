"""Dialogue (geogen.17; "Dialogue" in docs/rules.md): passages with choices and effects,
anchored to things in the world.

A ``kind: dialogue`` file (``dialogues/<id>.yaml``) that a scene lists under ``dialogues:``::

    anchor: receptionist            # a thing, NPC, place or room (a reference)
    start: hello                    # or [{when: <condition>, passage: x}, ..., <passage>]
    passages:
      hello:
        text: "Good evening. You have {held(\\"parcel\\")} parcels."   # {condition}s shown as values
        choices:
          - {id: parcel, text: "Any post?", when: 'not has("parcel")', do: [{give: parcel}], goto: thanks}
          - {text: Goodbye}         # no goto: closes

Like a quest it has no runtime logic: it compiles to actor state and rules.
- The open passage is ``actor:<pkg>.dialogue_<id>`` (text; "" when closed).
- Opening: an NPC anchor opens it when greeted, a place or room when entered, anything else
  when one of its interactions is used (``open:`` names another event, as a rule's ``on``).
  It opens at ``start`` (the first entry whose ``when`` holds) unless one is open already.
- Choosing: the runtime raises the anchor's signal ``dialogue`` with ``{passage, choice}``
  (choice "" closes it); a compiled rule per choice checks that passage is the one open and
  the choice's ``when`` still holds, then runs its effects and moves to ``goto`` (or closes).
- The manifest's ``dialogues`` section lists each one's anchor, state key and passages: text
  as parts (strings and condition trees to show the value of) and choices with their ``when``.

Checks: references, conditions and effects (as the quest compiler checks them), a ``goto`` or
``start`` naming no passage, and passages nothing leads to.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from . import refs, roots
from .core.node import SceneNode

KIND = "dialogue"
VERSION = 1
META = "dialogues"           # node.meta key the composer records a scene's dialogues under
SIGNAL = "dialogue"          # what the anchor raises when a choice is made
DIALOGUE_KEYS = {"kind", "version", "id", "title", "anchor", "open", "start", "passages"}
PASSAGE_KEYS = {"text", "choices"}
CHOICE_KEYS = {"id", "text", "when", "do", "goto"}
_PART = re.compile(r"\{([^{}]+)\}")


class DialogueError(ValueError):
    pass


def load_dialogue(path: str | Path) -> dict[str, Any]:
    """A ``kind: dialogue`` file as written, with ``id`` defaulting to the file name."""
    from .layout.yaml_utils import safe_load_path

    path = Path(path)
    data = safe_load_path(path)
    if not isinstance(data, dict) or data.get("kind") != KIND or int(data.get("version", 0)) != VERSION:
        raise DialogueError(f"{path}: expected kind '{KIND}' version {VERSION}")
    return {"id": path.stem, **data}


def declare_scene(root: SceneNode, data: dict[str, Any], source: Any, assets_dir: Path) -> None:
    """Record a scene's ``dialogues: [file | {file, if}]`` on its root (``meta.dialogues``)."""
    from .layout.yaml_utils import source_of
    from .wiring import built

    listed = data.get("dialogues")
    if not listed:
        return
    if not isinstance(listed, list) or not all(
            isinstance(item, str) or isinstance(item, dict) and isinstance(item.get("file"), str)
            and not set(item) - {"file", "if"} for item in listed):
        raise DialogueError(f"{source_of(source, ['dialogues'])}: dialogues is a list of dialogue files "
                            f"(dialogues/name.yaml, or {{file, if: <param expression>}})")
    found = []
    for index, name in enumerate(listed):
        if isinstance(name, dict):
            if not built(name):
                continue
            name = name["file"]
        path = roots.find(name, assets_dir)
        if not path.is_file():
            known = [roots.relative(p, assets_dir) for p in roots.glob("dialogues/*.yaml", assets_dir)]
            raise DialogueError(f"{source_of(source, ['dialogues', index])}: no dialogue file '{name}'."
                                f"{refs._near(name, known)}")
        found.append({"source": str(path), "spec": load_dialogue(path)})
    if found:
        root.meta[META] = found


def sources(root: SceneNode) -> list[tuple[dict[str, Any], Any, SceneNode]]:
    """Every scene's dialogues below ``root``: (spec, source file, the scene node)."""
    return [(entry["spec"], entry.get("source"), node) for node in root.iter_nodes()
            for entry in node.meta.get(META) or []]


def compile_dialogues(compiler: Any, entries: Iterable[tuple[dict[str, Any], Any, SceneNode | None]]) -> dict[str, Any]:
    """Compile ``entries`` with ``compiler`` (a ``quests._Compiler``): their state is declared,
    their rules join ``compiler.out.wiring``; returns the manifest's ``dialogues`` section."""
    out: dict[str, Any] = {}
    for spec, source, node in entries:
        entry = _compile(compiler, spec, source, node)
        if entry is not None:
            if entry["id"] in out:
                compiler.bad(f"{source}: id", f"dialogue '{entry['id']}' is contributed twice")
                continue
            out[entry.pop("id")] = entry
    return out


def _compile(compiler: Any, spec: dict[str, Any], source: Any, node: SceneNode | None) -> dict[str, Any] | None:
    from .quests import QuestError, _Quest

    did = str(spec.get("id", ""))
    src = _Quest(did, spec, source, node, None)
    src.where = (lambda *keys: f"dialogue '{did}': " + ".".join(str(k) for k in keys)) if source is None \
        else src.where                                                       # type: ignore[method-assign]
    bad = lambda keys, message: compiler.bad(src.where(*keys), message)    # noqa: E731
    unknown = set(spec) - DIALOGUE_KEYS
    if unknown:
        bad(["kind"], f"unknown keys {sorted(unknown)}. Keys: {', '.join(sorted(DIALOGUE_KEYS))}")
        return None
    passages = spec.get("passages")
    if not did.isidentifier() or not isinstance(passages, dict) or not passages:
        bad(["passages"], "a dialogue has an id and passages: {name: {text, choices}}")
        return None
    try:
        anchor, anchor_node, _ = compiler.thing(src, spec.get("anchor"))
    except (QuestError, TypeError) as exc:
        bad(["anchor"], f"the anchor (a thing, NPC, place or room) doesn't resolve: {exc}")
        return None
    state = f"actor:{compiler.package}.dialogue_{did}"
    try:
        compiler.decls.declare(state, {"type": "text", "default": ""})
    except Exception:
        bad(["id"], f"dialogue state 'dialogue_{did}' is already declared")
        return None
    compiler.decls.signals[f"{anchor}#{SIGNAL}"] = {"subject": anchor, "name": SIGNAL,
                                                    "data": {"passage": "text", "choice": "text"}}
    held = {"get": "actor", "package": compiler.package, "name": f"dialogue_{did}"}
    is_open = lambda passage: {"op": "==", "args": [dict(held), passage]}   # noqa: E731
    event = lambda field: {"get": "event", "name": field}                  # noqa: E731

    def condition(text: Any, *keys: Any) -> Any:
        try:
            return compiler.condition(src, text)
        except QuestError as exc:
            bad(keys, str(exc))
            return None

    # Where it starts: the first entry whose `when` holds.
    start = spec.get("start", next(iter(passages)))
    entries = start if isinstance(start, list) else [start]
    opening: Any = ""
    for index in reversed(range(len(entries))):
        item = entries[index]
        passage = item.get("passage") if isinstance(item, dict) else item
        if passage not in passages:
            bad(["start", index] if isinstance(start, list) else ["start"],
                f"no passage '{passage}'.{refs._near(str(passage), passages)}")
            continue
        when = condition(item["when"], "start", index, "when") if isinstance(item, dict) and "when" in item else None
        opening = passage if when is None else {"op": "if", "args": [when, passage, opening]}
    starts = {item.get("passage") if isinstance(item, dict) else item for item in entries}

    rules: list[dict[str, Any]] = []
    on = _opener(compiler, src, spec, anchor, anchor_node, bad)
    if on is not None:
        rules.append({"id": f"dialogue:{did}:open", "on": on, "when": is_open(""),
                      "do": [{"set": state, "to": opening}]})
    rules.append({"id": f"dialogue:{did}:close", "on": {"type": "signal", "subject": anchor, "name": SIGNAL},
                  "when": {"op": "==", "args": [event("choice"), ""]}, "do": [{"set": state, "to": ""}]})

    shown: dict[str, Any] = {}
    reached = set(starts)
    for name, passage in passages.items():
        at = ("passages", name)
        if not isinstance(passage, dict) or set(passage) - PASSAGE_KEYS or not isinstance(passage.get("text", ""), str):
            bad(at, f"a passage is {{text, choices}}, got {passage!r}")
            continue
        parts = _parts(str(passage.get("text", "")), lambda text: condition(text, *at, "text"))
        choices = []
        for index, choice in enumerate(passage.get("choices") or []):
            where = (*at, "choices", index)
            if not isinstance(choice, dict) or set(choice) - CHOICE_KEYS or not isinstance(choice.get("text"), str):
                bad(where, f"a choice is {{id, text, when, do, goto}} with text, got {choice!r}")
                continue
            cid = str(choice.get("id", index + 1))
            goto = choice.get("goto", "")
            if goto and goto not in passages:
                bad((*where, "goto"), f"no passage '{goto}'.{refs._near(str(goto), passages)}")
                continue
            if goto:
                reached.add(goto)
            when = condition(choice["when"], *where, "when") if "when" in choice else None
            do = compiler.effects(src, choice.get("do") or [], *where, "do") if choice.get("do") else []
            test = [is_open(name), {"op": "==", "args": [event("passage"), name]},
                    {"op": "==", "args": [event("choice"), cid]}, *([when] if when is not None else [])]
            rules.append({"id": f"dialogue:{did}:{name}:{cid}", "on": {"type": "signal", "subject": anchor, "name": SIGNAL},
                          "when": {"op": "and", "args": test}, "do": [*do, {"set": state, "to": goto}]})
            choices.append({"id": cid, "text": choice["text"], **({"when": when} if when is not None else {}),
                            **({"goto": goto} if goto else {})})
        shown[name] = {"text": parts, "choices": choices}
    for name in passages:
        if name not in reached:
            bad(("passages", name), f"nothing leads to passage '{name}': no start or choice goes there")
    for rule in rules:
        compiler.out.written[rule["id"]] = src.where("passages")
    compiler.out.wiring.extend(rules)
    return {"id": did, "title": str(spec.get("title", "")), "anchor": anchor, "state": state, "passages": shown}


def _opener(compiler: Any, src: Any, spec: dict[str, Any], anchor: str, node: SceneNode | None, bad) -> dict[str, Any] | None:
    """The event that opens it: ``open:`` as written, else by what the anchor is."""
    from .quests import QuestError
    from .state import npc_signal

    if "open" in spec:
        try:
            on = compiler.event(src, spec["open"])
        except QuestError as exc:
            bad(["open"], str(exc))
            return None
        return {**on, "actor": on.get("actor", "player")}
    if node is not None and npc_signal(node):
        return {"type": "signal", "subject": anchor, "name": npc_signal(node), "actor": "player"}
    if node is not None and (node.meta.get("type") in ("place", "room_volume") or "room" in node.tags):
        return {"type": "enter", "subject": anchor, "actor": "player"}
    if node is not None and node.interactions:
        return {"type": "use", "subject": anchor, "actor": "player"}
    bad(["anchor"], f"'{spec.get('anchor')}' isn't an NPC, a place or room, or anything with an interaction: "
                    f"write open: (an event) to say what opens it")
    return None


def _parts(text: str, compile_one) -> list[Any]:
    """``text`` as parts: strings, and condition trees for its ``{...}``."""
    parts: list[Any] = []
    at = 0
    for match in _PART.finditer(text):
        if match.start() > at:
            parts.append(text[at:match.start()])
        tree = compile_one(match.group(1))
        if tree is not None:
            parts.append({"value": tree})
        at = match.end()
    if at < len(text):
        parts.append(text[at:])
    return parts
