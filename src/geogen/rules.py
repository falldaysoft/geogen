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

Events carry ``data`` (a ``changed`` event's ``from`` / ``to``, what a runtime's signal
says: ``killed {value: 5}``). A rule answering an event reads it as ``event.<field>`` in
its ``when`` and effects; an ``emit`` effect may give its signal ``data`` (trees), and
``forward: true`` passes on the data of the event that fired the rule.

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
SAVE_VERSION = 1
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
ACTOR_KINDS = ("player", "npc")
# Feedback effects: what a rule tells its actor. Neither is state or an event: the machine
# logs them (`feedback`) and a runtime shows or plays them; nothing reacts to them.
FEEDBACK = {
    "notify": "text shown to the rule's actor (a HUD toast; with no actor, to the player)",
    "play": "a named sound (SOUNDS) played for the rule's actor",
}
# Clock-time durations: an `after` (or a skip) of {minutes | hours | days: n} counts game-clock
# minutes, not physics seconds. Programs hold them as {"minutes": n}.
CLOCK_UNITS = {"minutes": 1, "hours": 60, "days": 1440}
# The sounds `play` names: runtime/godot/scripts/sound.gd CHIMES has each.
SOUNDS = ("notice", "refused", "success", "click", "started", "objective_done", "complete", "failed",
          "shot", "hit", "destroyed")

State = dict[str, Any]
Event = dict[str, Any]


class RuleError(ValueError):
    """A program that is malformed, or that did something the semantics forbid."""


def duration(value: Any, where: str = "") -> float | dict[str, float]:
    """An authored ``after``: seconds (a number), or game-clock time as ``{minutes | hours |
    days: n}`` (summed). Returns the program's form: the seconds, or ``{"minutes": n}``."""
    if isinstance(value, dict):
        unknown = set(value) - set(CLOCK_UNITS)
        if not value or unknown or not all(isinstance(n, (int, float)) and not isinstance(n, bool) and n >= 0
                                           for n in value.values()):
            raise ValueError(f"{where}{': ' if where else ''}seconds, or clock time as {{minutes | hours | days: n}}, "
                             f"got {value!r}")
        minutes = float(sum(n * CLOCK_UNITS[unit] for unit, n in value.items()))
        if minutes <= 0:
            raise ValueError(f"{where}{': ' if where else ''}a clock time needs a length, got {value!r}")
        return {"minutes": minutes}
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}{': ' if where else ''}seconds, or clock time as {{minutes | hours | days: n}}, "
                         f"got {value!r}")
    return float(value)


def clock_minutes(after: Any) -> float | None:
    """The game-clock minutes of a program ``after`` ({"minutes": n}), None for seconds."""
    return float(after["minutes"]) if isinstance(after, dict) else None


def skip_effect(value: Any, where: str = "skip") -> dict[str, Any]:
    """An authored ``skip`` (clock time ``{minutes | hours | days: n}``, or ``{to: "HH:MM"}``: the next
    time the clock shows it) as the program's effect."""
    if isinstance(value, dict) and set(value) == {"to"}:
        if not _is_time(value["to"]):
            raise ValueError(f"{where}: to is a clock time \"HH:MM\", got {value['to']!r}")
        return {"skip": {"to": value["to"]}}
    return {"skip": duration(value if isinstance(value, dict) else None, where)}


def _is_time(text: Any) -> bool:
    if not isinstance(text, str) or text.count(":") != 1:
        return False
    hours, minutes = text.split(":")
    return hours.isdigit() and minutes.isdigit() and int(hours) < 24 and int(minutes) < 60 and len(minutes) == 2


def key_of(term: dict[str, Any]) -> str:
    """The variable key a resolved condition term reads."""
    if term["get"] == "instance":
        return f"{term['id']}#{term['name']}"
    return f"{term['get']}:{term['package']}.{term['name']}"


def is_actor_key(key: str) -> bool:
    return key.startswith("actor:")


def term_keys(term: Any, over: list[str] | None) -> list[str]:
    """The variable keys one term reads ([] for a literal, the clock or an actor built-in);
    an ``it`` term reads its name on each thing in ``over``."""
    if not isinstance(term, dict) or "get" not in term:
        return []
    if term["get"] == "it":
        return [f"{thing}#{term['name']}" for thing in over or []]
    if term["get"] in ("instance", "world") or (term["get"] == "actor" and "package" in term):
        return [key_of(term)]
    return []


def reads(tree: Any) -> list[str]:
    """Variable keys a condition tree reads ('it' terms as each thing's key)."""
    return [key for term, over in conditions.walk_over(tree) for key in term_keys(term, over)]


def _reads_actor(tree: Any) -> bool:
    return any(n.get("get") == "actor" for n in conditions.walk(tree))


def _reads_event(tree: Any) -> bool:
    return bool(conditions.event_fields(tree))


def rule_reads_event(rule: dict[str, Any]) -> bool:
    """Whether a rule reads the data of the event that fired it (so a delayed one keeps it)."""
    return _reads_event(rule.get("when")) or any(
        _reads_event(e.get("to")) or e.get("forward") or any(_reads_event(t) for t in (e.get("data") or {}).values())
        for e in rule.get("do", []))


def rule_touches_actor(rule: dict[str, Any]) -> bool:
    """Whether a rule reads or writes actor state: a periodic rule (``every``) that does runs per actor."""
    return _reads_actor(rule.get("when")) or any(
        is_actor_key(e.get("set", "")) or _reads_actor(e.get("to"))
        or any(_reads_actor(t) for t in (e.get("data") or {}).values())
        for e in rule.get("do", []))


def _every_ok(every: Any) -> bool:
    """A period: seconds or ``{minutes: n}``, more than none."""
    return _after_ok(every) and (every["minutes"] if isinstance(every, dict) else every) > 0


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
        for read in reads(bindings[key]["to"]):
            visit(read, path + [key])
        done[key] = True
        order.append(bindings[key])

    for key in bindings:
        visit(key, [])
    return order


def _after_ok(after: Any) -> bool:
    if isinstance(after, dict):
        return (set(after) == {"minutes"} and isinstance(after["minutes"], (int, float))
                and not isinstance(after["minutes"], bool) and after["minutes"] >= 0)
    return isinstance(after, (int, float)) and not isinstance(after, bool) and after >= 0


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
            if "after" in state and not _after_ok(state["after"]):
                problems.append(f"{key}: state '{name}' after: seconds, or {{minutes: n}} of clock time")
    guarded = [(key, spec["when"]) for key, spec in interactions.items() if "when" in spec]
    for key, spec in interactions.items():
        if "refused" in spec and ("when" not in spec or not isinstance(spec["refused"], str) or not spec["refused"]):
            problems.append(f"{key}: refused is the text shown when its `when` stops a use (it needs a when)")

    def tree(where: str, t: Any) -> None:
        try:
            conditions.validate(t)
        except ConditionError as e:
            problems.append(f"{where}: {e}")
            return
        for read in reads(t):
            if read not in known:
                problems.append(f"{where}: reads '{read}', which isn't declared")

    for key, when in guarded:
        tree(f"{key} when", when)
        if _reads_event(when):
            problems.append(f"{key} when: reads event data, but a use has no event to read")

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
        if _reads_event(binding.get("to")):
            problems.append(f"{where}: bindings can't read event data (a binding has no event)")
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
        every = rule.get("every")
        if on is None and "when" not in rule and every is None:
            problems.append(f"{where}: needs 'on' (an event), 'when' (a condition to watch) or 'every' (a period)")
        if every is not None:
            if on is not None:
                problems.append(f"{where}: has both 'on' and 'every': a periodic rule runs on its timer, not an event")
            if not _every_ok(every):
                problems.append(f"{where}: every: seconds, or {{minutes: n}} of clock time, more than 0")
            if rule.get("once"):
                problems.append(f"{where}: 'once' with 'every': a periodic rule that runs once is a delayed rule (after)")
            if rule.get("after"):
                problems.append(f"{where}: 'after' with 'every': the first run is one period after the start")
            if rule_reads_event(rule):
                problems.append(f"{where}: reads event data, but it runs on a timer: only a rule with 'on' has an event")
        if "actor" in rule and (every is None or rule["actor"] not in ACTOR_KINDS):
            problems.append(f"{where}: actor: player | npc narrows who a periodic rule (every) runs for"
                            + ("" if every is None else f", got {rule['actor']!r}"))
        if on is not None and on.get("type") not in EVENT_TYPES:
            problems.append(f"{where}: unknown event type {on.get('type')!r} (types: {', '.join(EVENT_TYPES)})")
        if "when" in rule:
            tree(where, rule["when"])
        has_actor = (rule_touches_actor(rule) if every is not None else _reads_actor(rule.get("when")) if on is None
                     else on.get("type") not in ACTORLESS and on.get("actor") != "none")
        if not has_actor and "when" in rule and on is not None and _reads_actor(rule["when"]):
            problems.append(f"{where}: its condition reads actor state, but its event never has an actor")
        if on is None and every is None and rule_reads_event(rule):
            problems.append(f"{where}: reads event data, but it watches a condition: only a rule with 'on' has an event")
        if not rule.get("do"):
            problems.append(f"{where}: does nothing")
        if "after" in rule and not _after_ok(rule["after"]):
            problems.append(f"{where}: after: seconds, or {{minutes: n}} of clock time")
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
                data = effect.get("data", {})
                if not isinstance(data, dict):
                    problems.append(f"{where}: emit data is a mapping of field: value")
                else:
                    for field, value in data.items():
                        tree(f"{where} emit data {field}", value)
            elif "notify" in effect:
                if not isinstance(effect["notify"], str) or not effect["notify"]:
                    problems.append(f"{where}: notify needs text")
            elif "play" in effect:
                if effect["play"] not in SOUNDS:
                    problems.append(f"{where}: play: no sound {effect['play']!r} (sounds: {', '.join(SOUNDS)})")
            elif "skip" in effect:
                skip = effect["skip"]
                if not (isinstance(skip, dict) and (set(skip) == {"to"} and _is_time(skip["to"])
                                                    or set(skip) == {"minutes"} and _after_ok(skip) and skip["minutes"] > 0)):
                    problems.append(f"{where}: skip is {{minutes: n}} or {{to: \"HH:MM\"}}, got {skip!r}")
            else:
                problems.append(f"{where}: unknown effect {sorted(effect)} (effects: set, emit, notify, play, skip)")
    return problems


# ---- the machine -------------------------------------------------------------------------------

class Machine:
    """A program running: its state, its clock and its pending timers.

    Feed it inputs (``post``, ``use``, ``advance``, ``set_clock``); read ``get`` and
    the ``signals`` / ``fired`` / ``feedback`` / ``errors`` logs. ``actors`` maps actor IDs to
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
        self.timers: list[dict[str, Any]] = []       # {at | clock, seq, kind, ...}, kept sorted
        self._seq = 0
        self._base: float | None = None              # the clock minute the clock timer firing now was due
        self._catching = False
        self._every_pending: list[tuple[dict[str, Any], str | None]] = []   # periodic rules not yet on a timer
        self.signals: list[list[Any]] = []           # [subject, name, actor]
        self.signal_data: list[dict[str, Any]] = []  # the data of each of `signals`, in step
        self.fired: list[list[Any]] = []             # [rule id, actor]
        self.feedback: list[list[Any]] = []          # [notify | play, text or sound, actor]: never saved
        self.errors: list[str] = []
        self._every_pending += [(rule, None) for rule in self.rules if "every" in rule and not rule_touches_actor(rule)]
        for actor, info in (actors or {}).items():
            self.add_actor(actor, **info)
        self._settle([])                             # bindings and watches see the starting state

    # -- reading -------------------------------------------------------------------------------

    def add_actor(self, actor: str, kind: str = "npc", **info: Any) -> None:
        if actor not in self.actors:                 # new here: its periodic rules start with it
            self._every_pending += [(rule, actor) for rule in self.rules if "every" in rule
                                    and rule_touches_actor(rule) and rule.get("actor", kind) == kind]
        self.actors.setdefault(actor, {})
        self.actor_info[actor] = {"kind": kind, "room": "", **info}

    def _start_every(self) -> None:
        """Put the periodic rules that are waiting on timers: at the first input after the machine or
        an actor arrives (not ``set_clock``, so a host that sets the clock first counts from that)."""
        for rule, actor in self._every_pending:
            self._schedule(rule["every"], kind="every", rule=rule["id"], actor=actor)
        self._every_pending = []

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

    def _lookup(self, actor: str | None, data: dict[str, Any] | None = None):
        def lookup(term: dict[str, Any]) -> Any:
            if term["get"] == "clock":
                return self.clock[term["name"]]
            if term["get"] == "event":
                if data is None or term["name"] not in data:
                    raise RuleError(f"'event.{term['name']}': " + ("there is no event here" if data is None else
                                    f"the event has no '{term['name']}' (it has {', '.join(data) or 'no data'})"))
                return data[term["name"]]
            if term["get"] == "actor" and "package" not in term:
                if actor is None:
                    raise RuleError(f"'actor.{term['name']}' read without an actor")
                return actor if term["name"] == "id" else self.actor_info.get(actor, {}).get(term["name"], "")
            return self.get(key_of(term), actor)
        return lookup

    def value(self, tree: Any, actor: str | None = None, data: dict[str, Any] | None = None) -> Any:
        """The value of a condition tree for ``actor``; ``data`` is the triggering event's."""
        return conditions.evaluate(tree, self._lookup(actor, data))

    # -- inputs --------------------------------------------------------------------------------

    def post(self, event: Event) -> None:
        """An event from outside (a place entered, a signal an asset raised, a new game)."""
        self._start_every()
        self._settle([{**event, "depth": 0}])

    def use(self, key: str, actor: str | None = None) -> bool:
        """An actor uses an interaction; False when it can't be used now."""
        if key not in self.interactions:
            raise RuleError(f"'{key}' isn't an interaction")
        self._start_every()
        thing, _, name = key.rpartition("#")
        state = self.interactions[key]["states"][self.get(key)]
        if state.get("next") is None:
            return False
        if self.get(f"{key}_locked"):
            self._settle([{"type": "signal", "subject": thing, "name": "locked", "actor": actor, "depth": 0}])
            return False
        if not self.allows(key, actor):
            if self.interactions[key].get("refused"):
                self.feedback += [["notify", self.interactions[key]["refused"], actor], ["play", "refused", actor]]
            return False
        queue = [{"type": "use", "subject": thing, "name": name, "actor": actor, "depth": 0}]
        self._set(key, state["next"], actor, queue, 0)
        self._settle(queue)
        return True

    def allows(self, key: str, actor: str | None = None) -> bool:
        """Whether the interaction's ``when`` holds for ``actor`` now (true without one). It only
        gates use: rules and ``set`` drive the interaction whatever it says."""
        when = self.interactions[key].get("when")
        if when is None:
            return True
        try:
            return conditions.truthy(self.value(when, actor))
        except (RuleError, ConditionError) as e:
            self.errors.append(f"{key} when: {e}")
            return False

    def set(self, key: str, value: Any, actor: str | None = None) -> None:
        """Set a variable from outside (a test, a console, a loaded tool)."""
        self._start_every()
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
        self._start_every()
        end = self.now + seconds
        while self.timers and "at" in self.timers[0] and self.timers[0]["at"] <= end + 1e-9:
            timer = self.timers.pop(0)
            self.now = max(self.now, timer["at"])
            self._settle(self._timer(timer))
        self.now = end

    # -- the semantics -------------------------------------------------------------------------

    def clock_minute(self) -> float:
        """The clock as minutes since day 0 began: what clock timers are due at."""
        return self.clock.get("day", 0) * 1440 + round(self.clock["hour"] * 60, 6)

    def _schedule(self, delay: float | dict[str, float], **timer: Any) -> None:
        """A timer ``delay`` seconds from now, or, for clock time ({"minutes": n}), n clock minutes
        from now: from when the clock timer firing now was due, so a chain of them catches up."""
        self._seq += 1
        minutes = clock_minutes(delay)
        if minutes is None:
            self.timers.append({"at": round(self.now + float(delay), 6), "seq": self._seq, **timer})
        else:
            base = self._base if self._base is not None else self.clock_minute()
            self.timers.append({"clock": round(base + minutes, 6), "seq": self._seq, **timer})
        self.timers.sort(key=_timer_order)

    def _clock_dwell(self, key: str, state: Any, actor: str | None) -> None:
        """An interaction entered ``state``: a clock-time dwell counts from now (not from its
        arrival, which takes seconds), and one waiting in the state it left is dropped."""
        self.timers = [t for t in self.timers if not ("clock" in t and t["kind"] == "dwell" and t["key"] == key)]
        spec = self.interactions[key]["states"].get(state, {})
        if spec.get("then") is not None and clock_minutes(spec.get("after")) is not None:
            self._schedule(spec["after"], kind="dwell", key=key, state=state, actor=actor)

    def _catch_up(self) -> None:
        """Run the clock timers the clock has reached, in order, each as its own input."""
        if self._catching:
            return
        self._catching = True
        try:
            while True:
                due = [t for t in self.timers if "clock" in t and t["clock"] <= self.clock_minute() + 1e-6]
                if not due:
                    return
                timer = due[0]
                self.timers.remove(timer)
                self._base = timer["clock"]
                try:
                    self._settle(self._timer(timer))
                finally:
                    self._base = None
        finally:
            self._catching = False

    def _skip(self, skip: dict[str, Any]) -> None:
        """Move the clock forward (a skip effect); what comes due on the way runs after this input."""
        start = self.clock_minute()
        if "to" in skip:
            hours, minutes = skip["to"].split(":")
            day_start = start - start % 1440
            target = day_start + int(hours) * 60 + int(minutes)
            if target <= start + 1e-6:
                target += 1440
        else:
            target = start + float(skip["minutes"])
        target = round(target)
        day, minute = divmod(int(target), 1440)
        self.clock.update({"day": day, "time": f"{minute // 60:02d}:{minute % 60:02d}", "hour": minute / 60.0})

    def _timer(self, timer: dict[str, Any]) -> list[Event]:
        queue: list[Event] = []
        kind, key, actor = timer["kind"], timer.get("key"), timer.get("actor")
        if kind == "rule":
            rule = next((r for r in self.rules if r["id"] == timer["rule"]), None)
            if rule is not None:
                self._effects(rule, actor, queue, 0, timer.get("data"))
        elif kind == "every":
            rule = next((r for r in self.rules if r["id"] == timer["rule"]), None)
            if rule is not None:                     # runs if its condition holds now; the next is a period on
                if self._holds(rule, actor):
                    self.fired.append([rule["id"], actor])
                    self._effects(rule, actor, queue, 0)
                self._schedule(rule["every"], kind="every", rule=rule["id"], actor=actor)
        elif kind == "arrive" and self.get(key) == timer["state"]:      # not reversed on the way
            thing = key.rpartition("#")[0]
            state = self.interactions[key]["states"][timer["state"]]
            if state.get("emit"):
                queue.append({"type": "signal", "subject": thing, "name": state["emit"], "actor": actor, "depth": 0})
            if state.get("then") is not None:
                if clock_minutes(state.get("after")) is not None:
                    pass                        # counting since it was entered (_clock_dwell)
                elif state.get("after"):
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

    def _fire(self, rule: dict[str, Any], actor: str | None, queue: list[Event], depth: int,
              data: dict[str, Any] | None = None) -> None:
        if rule.get("once"):
            latch = f"once:{rule['id']}"
            store = self.actors.setdefault(actor, {}) if actor is not None else self.world
            if store.get(latch):
                return
            store[latch] = True
        self.fired.append([rule["id"], actor])
        if rule.get("after"):
            kept = {"data": data} if data is not None and rule_reads_event(rule) else {}
            self._schedule(rule["after"], kind="rule", rule=rule["id"], actor=actor, **kept)
        else:
            self._effects(rule, actor, queue, depth, data)

    def _effects(self, rule: dict[str, Any], actor: str | None, queue: list[Event], depth: int,
                 data: dict[str, Any] | None = None) -> None:
        for effect in rule["do"]:
            try:
                if "set" in effect:
                    self._set(effect["set"], self.value(effect["to"], actor, data), actor, queue, depth)
                elif "notify" in effect or "play" in effect:
                    kind = "notify" if "notify" in effect else "play"
                    self.feedback.append([kind, effect[kind], actor])
                elif "skip" in effect:
                    self._skip(effect["skip"])
                else:
                    sent = dict(data or {}) if effect.get("forward") else {}
                    sent.update({field: self.value(tree, actor, data) for field, tree in effect.get("data", {}).items()})
                    queue.append({"type": "signal", "subject": effect.get("subject"), "name": effect["emit"],
                                  "actor": actor, "depth": depth + 1, **({"data": sent} if sent else {})})
            except (RuleError, ConditionError) as e:
                self.errors.append(f"rule {rule['id']}: {e}")

    def _holds(self, rule: dict[str, Any], actor: str | None, data: dict[str, Any] | None = None) -> bool:
        try:
            return "when" not in rule or conditions.truthy(self.value(rule["when"], actor, data))
        except (RuleError, ConditionError) as e:
            self.errors.append(f"rule {rule['id']}: {e}")
            return False

    def _settle(self, queue: list[Event]) -> None:
        """Process events until nothing is left: rules, then bindings, then watches; then the
        clock timers the clock has reached (a skip, or a clock moved on while nobody ran them)."""
        self._process(queue)
        self._catch_up()

    def _process(self, queue: list[Event]) -> None:
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
                    self.signal_data.append(event.get("data") or {})
                elif event["type"] in ("enter", "leave") and actor in self.actor_info and event.get("data", {}).get("room"):
                    self.actor_info[actor]["room"] = event["subject"] if event["type"] == "enter" else ""
                elif event["type"] == "changed" and event.get("key") in self.interactions:
                    self._clock_dwell(event["key"], event.get("data", {}).get("to"), actor)
                data = event.get("data") or {}
                for rule in self.rules:
                    if rule.get("on") is not None and self._matches(rule["on"], event) and self._holds(rule, actor, data):
                        self._fire(rule, actor, queue, depth, data)
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
            if rule.get("on") is not None or "every" in rule:
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
        """Everything needed to carry on later (docs/schema/geogen-save.v1.schema.json).

        World and actor state stay apart, each a flat ``{key: value}`` of what differs from the
        defaults (the keys name their owners) plus its hidden latches; bindings aren't saved.
        """
        return copy.deepcopy({"format": SAVE_FORMAT, "version": SAVE_VERSION, "now": self.now, "clock": self.clock,
                              "world": self.world, "actors": self.actors, "actor_info": self.actor_info,
                              "timers": self.timers})

    def load(self, data: dict[str, Any]) -> None:
        """Take up a save. Sections it leaves out are empty; actors it doesn't describe stay as they were."""
        if data.get("format") != SAVE_FORMAT:
            raise RuleError(f"not a {SAVE_FORMAT} file")
        if data.get("version", SAVE_VERSION) != SAVE_VERSION:
            raise RuleError(f"a {SAVE_FORMAT} of version {data['version']}, not {SAVE_VERSION}")
        data = copy.deepcopy(data)
        self.now, self.clock = float(data.get("now", 0.0)), data.get("clock", self.clock)
        self.world, self.actors = data.get("world", {}), data.get("actors", {})
        known, saved = self.actor_info, data.get("actor_info", {})
        self.actor_info = {actor: saved.get(actor) or known.get(actor) or {"kind": "npc", "room": ""}
                           for actor in {**self.actors, **saved}}
        self.timers = sorted(data.get("timers", []), key=_timer_order)
        self._every_pending = []                     # the save holds the periodic timers
        self._seq = max((t["seq"] for t in self.timers), default=0)
        self.derived = {}
        self._rebind([], 0)             # recompute, silently: nothing changed while saved


def _timer_order(timer: dict[str, Any]) -> tuple[int, float, int]:
    """Timers in firing order: those in seconds (advance takes them from the front), then clock ones."""
    return (0, timer["at"], timer["seq"]) if "at" in timer else (1, timer["clock"], timer["seq"])


# ---- traces ------------------------------------------------------------------------------------

def run_trace(trace: dict[str, Any] | str | Path) -> list[str]:
    """Replay a trace against the reference machine; the mismatches ([] when it conforms).

    A trace is ``{program, actors, steps}``. A step is one of ``{"use": key, "actor"}``,
    ``{"event": {...}}``, ``{"set": key, "to": value, "actor"}``, ``{"advance": seconds}``,
    ``{"clock": {...}}``, ``{"save_load": true}`` (a round trip through a save) or
    ``{"expect": {...}}`` with any of ``state`` ({key: value}), ``actors`` ({id: {key: value}}),
    ``signals`` / ``fired`` / ``feedback`` (exactly what happened since the last expect:
    [subject, name] pairs, rule ids, [kind, value, actor]), ``signal_data`` ([subject, name, data]
    of the same signals), ``clock`` ({field: value}), ``timers`` (how many are pending) and
    ``errors`` (a count).
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
            for section in step["save_load"].get("without", []) if isinstance(step["save_load"], dict) else []:
                del saved[section]
            logs = machine.signals, machine.signal_data, machine.fired, machine.feedback, machine.errors
            machine = Machine(trace["program"], trace.get("actors"))
            machine.load(saved)
            machine.signals, machine.signal_data, machine.fired, machine.feedback, machine.errors = logs    # loading is not an input: nothing fires
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
            if "signal_data" in expect:
                got = [[s[0], s[1], d] for s, d in zip(machine.signals, machine.signal_data)]
                if got != expect["signal_data"]:
                    problems.append(f"{where}: signal data {got}, expected {expect['signal_data']}")
            if "fired" in expect:
                got = [f[0] for f in machine.fired]
                if got != expect["fired"]:
                    problems.append(f"{where}: fired {got}, expected {expect['fired']}")
            if "feedback" in expect and machine.feedback != expect["feedback"]:
                problems.append(f"{where}: feedback {machine.feedback}, expected {expect['feedback']}")
            for field, want in expect.get("clock", {}).items():
                if machine.clock.get(field) != want:
                    problems.append(f"{where}: clock {field} is {machine.clock.get(field)!r}, expected {want!r}")
            if "timers" in expect and len(machine.timers) != expect["timers"]:
                problems.append(f"{where}: {len(machine.timers)} timers pending, expected {expect['timers']}")
            if "errors" in expect and len(machine.errors) != expect["errors"]:
                problems.append(f"{where}: {len(machine.errors)} errors, expected {expect['errors']}: {machine.errors}")
            machine.signals, machine.fired, machine.feedback, machine.errors = [], [], [], []
            machine.signal_data = []
        else:
            problems.append(f"{where}: unknown step {sorted(step)}")
    return problems
