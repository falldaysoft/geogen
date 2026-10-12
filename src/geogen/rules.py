"""State, events, rules and bindings: the reference evaluator.

The layer under wiring, quests, dialogue and inventory is declarative data (a
*program*) plus the semantics in ``docs/rules.md``. ``Machine`` implements those
semantics in plain Python: a runtime (Godot's is one) must reach the same state
and raise the same signals from the same inputs. ``run_trace`` replays a trace
file (``tests/data/rules/*.json``), which is how both are checked.

A program is JSON, with things named by stable ID and conditions compiled
(``geogen.conditions``)::

    {"format": "geogen-rules", "version": 1,
     "variables": {"world:demo.power": {"type": "bool", "default": false},
                   "actor:demo.visits": {"type": "number", "default": 0},
                   "demo:lamp#lit": {"type": "bool", "default": false}},
     "interactions": {"demo:lever#pull": {"initial": "up", "duration": 0.5,
                                         "states": {"up": {"next": "down"},
                                                    "down": {"next": "up", "emit": "pulled"}}}},
     "bindings": [{"set": "demo:lamp#lit", "to": <condition tree>}],
     "rules": [{"id": "open_gate",
                "on": {"type": "signal", "subject": "demo:lever", "name": "pulled"},
                "when": <condition tree>,
                "do": [{"set": "demo:gate#gate", "to": "open"}, {"emit": "opened", "subject": "demo:gate"}],
                "after": 0, "once": false}]}

Variable keys say who owns the value: ``<thing id>#<name>`` (one placed instance;
an interaction's state is one of these), ``world:<package>.<name>`` (shared) and
``actor:<package>.<name>`` (one value per actor: the player or an NPC).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

from . import conditions
from .conditions import ConditionError
from .state import TYPES, fits as _fits

FORMAT = "geogen-rules"
VERSION = 1
SAVE_FORMAT = "geogen-save"
MAX_DEPTH = 16             # how many events deep one input may cascade
EVENT_TYPES = {
    "signal": "a thing raised a named signal: an interaction arriving at a state with `emit`, or a rule's emit",
    "changed": "a variable took a new value (key, data.from, data.to)",
    "use": "an actor used an interaction (before it moves)",
    "enter": "an actor entered a place or room (subject)",
    "leave": "an actor left a place or room (subject)",
    "travel": "an actor travelled to another scene or spawn",
    "clock": "the clock crossed dusk or dawn (name: night | day)",
    "start": "a new game began for an actor",
}
ACTORLESS = ("clock",)     # event types that never have an actor

State = dict[str, Any]
Event = dict[str, Any]


class RuleError(ValueError):
    """A program that is malformed, or that did something the semantics forbid."""


def key_of(term: dict[str, Any]) -> str:
    """The variable key a resolved condition term reads."""
    if term["get"] == "instance":
        return f"{term['id']}#{term['name']}"
    return f"{term['get']}:{term['package']}.{term['name']}"


def is_actor_key(key: str) -> bool:
    return key.startswith("actor:")


def _reads(tree: Any) -> list[str]:
    """Variable keys a condition tree reads ('it' terms as each thing's key)."""
    out: list[str] = []

    def visit(t: Any, over: list[str] | None) -> None:
        if not isinstance(t, dict):
            return
        if "get" in t:
            if t["get"] == "it":
                out.extend(f"{thing}#{t['name']}" for thing in over or [])
            elif t["get"] in ("instance", "world") or (t["get"] == "actor" and "package" in t):
                out.append(key_of(t))
            return
        for arg in t.get("args", []):
            visit(arg, over)
        if "where" in t:
            visit(t["where"], t.get("over", []))

    visit(tree, None)
    return out


def _reads_actor(tree: Any) -> bool:
    return any(n.get("get") == "actor" for n in conditions.walk(tree))


# ---- checking a program ------------------------------------------------------------------------

def binding_order(program: dict[str, Any]) -> list[dict[str, Any]]:
    """Bindings ordered so each comes after the ones it reads; RuleError on a cycle."""
    bindings = {b["set"]: b for b in program.get("bindings", [])}
    order: list[dict[str, Any]] = []
    done: dict[str, bool] = {}          # key -> finished (False while on the stack)

    def visit(key: str, path: list[str]) -> None:
        if key not in bindings or done.get(key):
            return
        if key in done:
            raise RuleError(f"bindings form a cycle: {' -> '.join(path + [key])}")
        done[key] = False
        for read in _reads(bindings[key]["to"]):
            visit(read, path + [key])
        done[key] = True
        order.append(bindings[key])

    for key in bindings:
        visit(key, [])
    return order


def check_program(program: dict[str, Any]) -> list[str]:
    """Everything wrong with a program, as messages ([] when it is sound)."""
    problems: list[str] = []
    if program.get("format") != FORMAT or program.get("version") != VERSION:
        problems.append(f"not a {FORMAT} v{VERSION} program")
    variables = program.get("variables", {})
    interactions = program.get("interactions", {})
    known = set(variables) | set(interactions) | {f"{k}_locked" for k in interactions}
    for key, spec in variables.items():
        if spec.get("type") not in TYPES:
            problems.append(f"{key}: type must be one of {', '.join(TYPES)}")
        elif "default" not in spec:
            problems.append(f"{key}: needs a default")
        elif not _fits(spec, spec["default"]):
            problems.append(f"{key}: default {spec['default']!r} isn't a {spec['type']}")
    for key, spec in interactions.items():
        states = spec.get("states", {})
        if spec.get("initial") not in states:
            problems.append(f"{key}: initial state {spec.get('initial')!r} isn't one of its states")
        for name, state in states.items():
            for link in ("next", "then"):
                if state.get(link) is not None and state[link] not in states:
                    problems.append(f"{key}: state '{name}' {link}: '{state[link]}' isn't one of its states")

    def tree(where: str, t: Any) -> None:
        try:
            conditions.validate(t)
        except ConditionError as e:
            problems.append(f"{where}: {e}")
            return
        for read in _reads(t):
            if read not in known:
                problems.append(f"{where}: reads '{read}', which isn't declared")

    bound = set()
    for binding in program.get("bindings", []):
        key = binding.get("set", "")
        where = f"binding of {key}"
        if key in bound:
            problems.append(f"{where}: bound twice")
        bound.add(key)
        if key not in variables:
            problems.append(f"{where}: '{key}' isn't a declared variable" if key not in interactions
                            else f"{where}: an interaction's state can't be bound (drive it with a rule)")
        if is_actor_key(key) or _reads_actor(binding.get("to")):
            problems.append(f"{where}: bindings can't involve actor state (a binding has no actor)")
        tree(where, binding.get("to"))
    try:
        binding_order(program)
    except RuleError as e:
        problems.append(str(e))

    seen = set()
    for rule in program.get("rules", []):
        rid = rule.get("id")
        where = f"rule {rid}"
        if not rid or rid in seen:
            problems.append(f"{where}: rules need distinct ids")
        seen.add(rid)
        on = rule.get("on")
        if on is None and "when" not in rule:
            problems.append(f"{where}: needs 'on' (an event) or 'when' (a condition to watch)")
        if on is not None and on.get("type") not in EVENT_TYPES:
            problems.append(f"{where}: unknown event type {on.get('type')!r} (types: {', '.join(EVENT_TYPES)})")
        if "when" in rule:
            tree(where, rule["when"])
        has_actor = (_reads_actor(rule.get("when")) if on is None
                     else on.get("type") not in ACTORLESS and on.get("actor") != "none")
        if not has_actor and "when" in rule and on is not None and _reads_actor(rule["when"]):
            problems.append(f"{where}: its condition reads actor state, but its event never has an actor")
        if not rule.get("do"):
            problems.append(f"{where}: does nothing")
        for effect in rule.get("do", []):
            if "set" in effect:
                key = effect["set"]
                if key not in known:
                    problems.append(f"{where}: sets '{key}', which isn't declared")
                elif key in bound:
                    problems.append(f"{where}: sets '{key}', which is bound (bindings own their value)")
                if is_actor_key(key) and not has_actor:
                    problems.append(f"{where}: sets actor state '{key}', but nothing here says which actor "
                                    f"(its trigger has none)")
                if "to" not in effect:
                    problems.append(f"{where}: set '{key}' needs 'to'")
                else:
                    tree(where, effect["to"])
                    if _reads_actor(effect["to"]) and not has_actor:
                        problems.append(f"{where}: the value for '{key}' reads actor state without an actor")
                    literal = effect["to"]
                    if key in interactions and isinstance(literal, str) and literal not in interactions[key]["states"]:
                        problems.append(f"{where}: '{key}' has no state '{literal}'")
            elif "emit" in effect:
                if not isinstance(effect["emit"], str) or not effect["emit"]:
                    problems.append(f"{where}: emit needs a signal name")
            else:
                problems.append(f"{where}: unknown effect {sorted(effect)} (effects: set, emit)")
    return problems


# ---- the machine -------------------------------------------------------------------------------

class Machine:
    """A program running: its state, its clock and its pending timers.

    Feed it inputs (``post``, ``use``, ``advance``, ``set_clock``); read ``get`` and
    the ``signals`` / ``fired`` / ``errors`` logs. ``actors`` maps actor IDs to
    ``{"kind": "player" | "npc"}``.
    """

    def __init__(self, program: dict[str, Any], actors: dict[str, dict[str, Any]] | None = None) -> None:
        problems = check_program(program)
        if problems:
            raise RuleError("; ".join(problems))
        self.program = program
        self.variables: dict[str, dict[str, Any]] = dict(program.get("variables", {}))
        self.interactions: dict[str, dict[str, Any]] = program.get("interactions", {})
        for key, spec in self.interactions.items():
            self.variables[key] = {"type": "choice", "choices": list(spec["states"]), "default": spec["initial"]}
            self.variables[f"{key}_locked"] = {"type": "bool", "default": bool(spec.get("locked", False))}
        self.rules: list[dict[str, Any]] = program.get("rules", [])
        self._bindings = binding_order(program)
        self._bound = {b["set"] for b in self._bindings}
        self.world: State = {}                       # values that differ from their defaults
        self.actors: dict[str, State] = {}
        self.actor_info: dict[str, dict[str, Any]] = {}
        self.derived: State = {}                     # binding values: never saved
        self.clock: dict[str, Any] = {"time": "12:00", "hour": 12.0, "night": False, "day": 0}
        self.now = 0.0
        self.timers: list[dict[str, Any]] = []       # {at, seq, kind, ...}, kept sorted
        self._seq = 0
        self.signals: list[list[Any]] = []           # [subject, name, actor]
        self.fired: list[list[Any]] = []             # [rule id, actor]
        self.errors: list[str] = []
        for actor, info in (actors or {}).items():
            self.add_actor(actor, **info)
        self._settle([])                             # bindings and watches see the starting state

    # -- reading -------------------------------------------------------------------------------

    def add_actor(self, actor: str, kind: str = "npc", **info: Any) -> None:
        self.actors.setdefault(actor, {})
        self.actor_info[actor] = {"kind": kind, "room": "", **info}

    def get(self, key: str, actor: str | None = None) -> Any:
        """The value of a variable (``actor`` for ``actor:`` keys)."""
        if key in self.derived:
            return self.derived[key]
        if key not in self.variables:
            raise RuleError(f"'{key}' isn't declared")
        if is_actor_key(key):
            if actor is None:
                raise RuleError(f"'{key}' is actor state: whose?")
            return self.actors.get(actor, {}).get(key, self.variables[key]["default"])
        return self.world.get(key, self.variables[key]["default"])

    def _lookup(self, actor: str | None):
        def lookup(term: dict[str, Any]) -> Any:
            if term["get"] == "clock":
                return self.clock[term["name"]]
            if term["get"] == "actor" and "package" not in term:
                if actor is None:
                    raise RuleError(f"'actor.{term['name']}' read without an actor")
                return self.actor_info.get(actor, {}).get(term["name"], "")
            return self.get(key_of(term), actor)
        return lookup

    def value(self, tree: Any, actor: str | None = None) -> Any:
        return conditions.evaluate(tree, self._lookup(actor))

    # -- inputs --------------------------------------------------------------------------------

    def post(self, event: Event) -> None:
        """An event from outside (a place entered, a signal an asset raised, a new game)."""
        self._settle([{**event, "depth": 0}])

    def use(self, key: str, actor: str | None = None) -> bool:
        """An actor uses an interaction; False when it can't be used now."""
        if key not in self.interactions:
            raise RuleError(f"'{key}' isn't an interaction")
        thing, _, name = key.rpartition("#")
        state = self.interactions[key]["states"][self.get(key)]
        if state.get("next") is None:
            return False
        if self.get(f"{key}_locked"):
            self._settle([{"type": "signal", "subject": thing, "name": "locked", "actor": actor, "depth": 0}])
            return False
        queue = [{"type": "use", "subject": thing, "name": name, "actor": actor, "depth": 0}]
        self._set(key, state["next"], actor, queue, 0)
        self._settle(queue)
        return True

    def set(self, key: str, value: Any, actor: str | None = None) -> None:
        """Set a variable from outside (a test, a console, a loaded tool)."""
        queue: list[Event] = []
        self._set(key, value, actor, queue, 0)
        self._settle(queue)

    def set_clock(self, **values: Any) -> None:
        """Move the clock (``time`` "HH:MM", ``night``, ``day``); dusk and dawn are events."""
        was_night = self.clock["night"]
        self.clock.update(values)
        if "time" in values:
            hours, minutes = values["time"].split(":")
            self.clock["hour"] = int(hours) + int(minutes) / 60.0
        queue = []
        if self.clock["night"] != was_night:
            queue.append({"type": "clock", "name": "night" if self.clock["night"] else "day", "depth": 0})
        self._settle(queue)

    def advance(self, seconds: float) -> None:
        """Let ``seconds`` pass: interactions arrive, dwells and delayed rules run, in time order."""
        end = self.now + seconds
        while self.timers and self.timers[0]["at"] <= end + 1e-9:
            timer = self.timers.pop(0)
            self.now = max(self.now, timer["at"])
            self._settle(self._timer(timer))
        self.now = end

    # -- the semantics -------------------------------------------------------------------------

    def _schedule(self, delay: float, **timer: Any) -> None:
        self._seq += 1
        self.timers.append({"at": round(self.now + delay, 6), "seq": self._seq, **timer})
        self.timers.sort(key=lambda t: (t["at"], t["seq"]))

    def _timer(self, timer: dict[str, Any]) -> list[Event]:
        queue: list[Event] = []
        kind, key, actor = timer["kind"], timer.get("key"), timer.get("actor")
        if kind == "rule":
            rule = next((r for r in self.rules if r["id"] == timer["rule"]), None)
            if rule is not None:
                self._effects(rule, actor, queue, 0)
        elif kind == "arrive" and self.get(key) == timer["state"]:      # not reversed on the way
            thing = key.rpartition("#")[0]
            state = self.interactions[key]["states"][timer["state"]]
            if state.get("emit"):
                queue.append({"type": "signal", "subject": thing, "name": state["emit"], "actor": actor, "depth": 0})
            if state.get("then") is not None:
                if state.get("after"):
                    self._schedule(float(state["after"]), kind="dwell", key=key, state=timer["state"], actor=actor)
                else:
                    self._set(key, state["then"], actor, queue, 0)
        elif kind == "dwell" and self.get(key) == timer["state"]:
            self._set(key, self.interactions[key]["states"][timer["state"]]["then"], actor, queue, 0)
        return queue

    def _set(self, key: str, value: Any, actor: str | None, queue: list[Event], depth: int) -> None:
        if key not in self.variables:
            raise RuleError(f"'{key}' isn't declared")
        if key in self._bound:
            raise RuleError(f"'{key}' is bound and can't be set")
        spec = self.variables[key]
        if spec["type"] == "bool" and not isinstance(value, bool):
            value = conditions.truthy(value)
        if not _fits(spec, value):
            raise RuleError(f"'{key}' is a {spec['type']}"
                            + (f" ({', '.join(spec['choices'])})" if spec["type"] == "choice" else "")
                            + f": can't be {value!r}")
        if is_actor_key(key):
            if actor is None:
                raise RuleError(f"'{key}' is actor state, and this has no actor")
            store = self.actors.setdefault(actor, {})
        else:
            store = self.world
        old = store.get(key, spec["default"])
        if old == value and isinstance(old, bool) == isinstance(value, bool):
            return
        if value == spec["default"]:
            store.pop(key, None)
        else:
            store[key] = value
        self._changed(key, old, value, actor, queue, depth)
        if key in self.interactions:
            self._schedule(float(self.interactions[key].get("duration", 0.9)), kind="arrive", key=key,
                           state=value, actor=actor)

    def _changed(self, key: str, old: Any, new: Any, actor: str | None, queue: list[Event], depth: int) -> None:
        thing, sep, name = key.rpartition("#")
        queue.append({"type": "changed", "key": key, "subject": thing if sep else None, "name": name if sep else key,
                      "actor": actor, "data": {"from": old, "to": new}, "depth": depth + 1})

    def _matches(self, on: dict[str, Any], event: Event) -> bool:
        if on["type"] != event["type"]:
            return False
        subject = on.get("subject")
        if subject is not None and event.get("subject") not in (subject if isinstance(subject, list) else [subject]):
            return False
        for field in ("name", "key"):
            if on.get(field) is not None and on[field] != event.get(field):
                return False
        if "to" in on and event.get("data", {}).get("to") != on["to"]:
            return False
        who = on.get("actor", "any")
        if who == "any":
            return True
        actor = event.get("actor")
        return actor is None if who == "none" else (actor is not None and self.actor_info.get(actor, {}).get("kind") == who)

    def _fire(self, rule: dict[str, Any], actor: str | None, queue: list[Event], depth: int) -> None:
        if rule.get("once"):
            latch = f"once:{rule['id']}"
            store = self.actors.setdefault(actor, {}) if actor is not None else self.world
            if store.get(latch):
                return
            store[latch] = True
        self.fired.append([rule["id"], actor])
        if rule.get("after"):
            self._schedule(float(rule["after"]), kind="rule", rule=rule["id"], actor=actor)
        else:
            self._effects(rule, actor, queue, depth)

    def _effects(self, rule: dict[str, Any], actor: str | None, queue: list[Event], depth: int) -> None:
        for effect in rule["do"]:
            try:
                if "set" in effect:
                    self._set(effect["set"], self.value(effect["to"], actor), actor, queue, depth)
                else:
                    queue.append({"type": "signal", "subject": effect.get("subject"), "name": effect["emit"],
                                  "actor": actor, "depth": depth + 1})
            except (RuleError, ConditionError) as e:
                self.errors.append(f"rule {rule['id']}: {e}")

    def _holds(self, rule: dict[str, Any], actor: str | None) -> bool:
        try:
            return "when" not in rule or conditions.truthy(self.value(rule["when"], actor))
        except (RuleError, ConditionError) as e:
            self.errors.append(f"rule {rule['id']}: {e}")
            return False

    def _settle(self, queue: list[Event]) -> None:
        """Process events until nothing is left: rules, then bindings, then watches."""
        for _ in range(MAX_DEPTH + 1):
            while queue:
                event = queue.pop(0)
                depth = event.get("depth", 0)
                if depth > MAX_DEPTH:
                    self.errors.append(f"cascade deeper than {MAX_DEPTH} events dropped at {event['type']} "
                                       f"{event.get('key') or event.get('name') or ''}".rstrip())
                    continue
                actor = event.get("actor")
                if event["type"] == "signal":
                    self.signals.append([event.get("subject"), event["name"], actor])
                elif event["type"] in ("enter", "leave") and actor in self.actor_info and event.get("data", {}).get("room"):
                    self.actor_info[actor]["room"] = event["subject"] if event["type"] == "enter" else ""
                for rule in self.rules:
                    if rule.get("on") is not None and self._matches(rule["on"], event) and self._holds(rule, actor):
                        self._fire(rule, actor, queue, depth)
                self._rebind(queue, depth)
            self._rebind(queue, 0)
            self._watch(queue)
            if not queue:
                return
        self.errors.append(f"watched conditions kept triggering each other: stopped after {MAX_DEPTH} rounds")

    def _rebind(self, queue: list[Event], depth: int) -> None:
        for binding in self._bindings:
            key = binding["set"]
            spec = self.variables[key]
            try:
                value = self.value(binding["to"])
                if spec["type"] == "bool":
                    value = conditions.truthy(value)
                if not _fits(spec, value):
                    raise RuleError(f"a {spec['type']} can't be {value!r}")
            except (RuleError, ConditionError) as e:
                self.errors.append(f"binding of {key}: {e}")
                continue
            old = self.derived.get(key, spec["default"])
            self.derived[key] = value
            if old != value:
                self._changed(key, old, value, None, queue, depth)

    def _watch(self, queue: list[Event]) -> None:
        """Rules without ``on`` fire when their condition becomes true (per actor when it reads
        actor state). Every condition is judged against the same settled state before any fires."""
        newly: list[tuple[dict[str, Any], str | None]] = []
        for rule in self.rules:
            if rule.get("on") is not None:
                continue
            latch = f"watch:{rule['id']}"
            subjects: Iterable[str | None] = list(self.actors) if _reads_actor(rule["when"]) else [None]
            for actor in subjects:
                store = self.actors[actor] if actor is not None else self.world
                truth = self._holds(rule, actor)
                if truth and not store.get(latch):
                    store[latch] = True
                    newly.append((rule, actor))
                elif not truth:
                    store.pop(latch, None)
        for rule, actor in newly:
            self._fire(rule, actor, queue, 0)

    # -- saving --------------------------------------------------------------------------------

    def save(self) -> dict[str, Any]:
        """Everything needed to carry on later. World and actor state stay apart; bindings aren't saved."""
        return copy.deepcopy({"format": SAVE_FORMAT, "version": 1, "now": self.now, "clock": self.clock,
                              "world": self.world, "actors": self.actors, "actor_info": self.actor_info,
                              "timers": self.timers})

    def load(self, data: dict[str, Any]) -> None:
        if data.get("format") != SAVE_FORMAT:
            raise RuleError(f"not a {SAVE_FORMAT} file")
        data = copy.deepcopy(data)
        self.now, self.clock = data["now"], data["clock"]
        self.world, self.actors, self.actor_info = data["world"], data["actors"], data["actor_info"]
        self.timers = data["timers"]
        self._seq = max((t["seq"] for t in self.timers), default=0)
        self.derived = {}
        self._rebind([], 0)             # recompute, silently: nothing changed while saved


# ---- traces ------------------------------------------------------------------------------------

def run_trace(trace: dict[str, Any] | str | Path) -> list[str]:
    """Replay a trace against the reference machine; the mismatches ([] when it conforms).

    A trace is ``{program, actors, steps}``. A step is one of ``{"use": key, "actor"}``,
    ``{"event": {...}}``, ``{"set": key, "to": value, "actor"}``, ``{"advance": seconds}``,
    ``{"clock": {...}}``, ``{"save_load": true}`` (a round trip through a save) or
    ``{"expect": {...}}`` with any of ``state`` ({key: value}), ``actors`` ({id: {key: value}}),
    ``signals`` / ``fired`` (exactly what happened since the last expect: [subject, name] pairs
    and rule ids) and ``errors`` (a count).
    """
    if not isinstance(trace, dict):
        trace = json.loads(Path(trace).read_text())
    machine = Machine(trace["program"], trace.get("actors"))
    problems: list[str] = []
    for index, step in enumerate(trace["steps"]):
        where = f"step {index + 1}"
        if "use" in step:
            used = machine.use(step["use"], step.get("actor"))
            if "used" in step and used != step["used"]:
                problems.append(f"{where}: use {step['use']} gave {used}, expected {step['used']}")
        elif "event" in step:
            machine.post(step["event"])
        elif "set" in step:
            machine.set(step["set"], step["to"], step.get("actor"))
        elif "advance" in step:
            machine.advance(step["advance"])
        elif "clock" in step:
            machine.set_clock(**step["clock"])
        elif "save_load" in step:
            saved = json.loads(json.dumps(machine.save()))
            logs = machine.signals, machine.fired, machine.errors
            machine = Machine(trace["program"], trace.get("actors"))
            machine.load(saved)
            machine.signals, machine.fired, machine.errors = logs    # loading is not an input: nothing fires
        elif "expect" in step:
            expect = step["expect"]
            for key, want in expect.get("state", {}).items():
                if machine.get(key) != want:
                    problems.append(f"{where}: {key} is {machine.get(key)!r}, expected {want!r}")
            for actor, values in expect.get("actors", {}).items():
                for key, want in values.items():
                    if machine.get(key, actor) != want:
                        problems.append(f"{where}: {actor} {key} is {machine.get(key, actor)!r}, expected {want!r}")
            if "signals" in expect:
                got = [s[:2] for s in machine.signals]
                if got != expect["signals"]:
                    problems.append(f"{where}: signals {got}, expected {expect['signals']}")
            if "fired" in expect:
                got = [f[0] for f in machine.fired]
                if got != expect["fired"]:
                    problems.append(f"{where}: fired {got}, expected {expect['fired']}")
            if "errors" in expect and len(machine.errors) != expect["errors"]:
                problems.append(f"{where}: {len(machine.errors)} errors, expected {expect['errors']}: {machine.errors}")
            machine.signals, machine.fired, machine.errors = [], [], []
        else:
            problems.append(f"{where}: unknown step {sorted(step)}")
    return problems
