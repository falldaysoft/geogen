"""State declarations: what variables, signals and inputs a scene has, and who owns them.

Every variable has exactly one owner, and where it is declared says which
(``docs/rules.md``). An asset's ``state:`` belongs to each placed instance::

    state:                                   # in an asset
      lit:  {type: bool, default: false}     # bool | number | text | choice
      mode: {type: choice, choices: [dim, bright]}

A scene's ``state:`` belongs to the package, and must say how::

    state:                                   # in a scene
      bridge_down: {scope: world, type: bool, default: false}   # one shared value
      has_ticket:  {scope: actor, type: bool, default: false}   # one per player or NPC

Assets also name what they offer to wiring, relative to themselves so every
instance wires itself::

    signals:                                 # named outputs
      pulled:   {emit: pulled}               # an interaction arrives at a state with this emit
      switched: {changed: lit, to: true}     # a variable (or interaction) takes a value
      occupied: {enter: plate}               # an actor enters a place of the asset (leave: too)
    inputs:                                  # named things it accepts
      open:   {set: gate, to: open}          # drive an interaction to a state
      unlock: {set: gate_locked, to: false}
      light:  {set: lit}                     # no `to`: whoever drives it gives the value

``emit`` / ``changed`` / ``set`` take ``of: <reference>`` to mean a thing inside the
asset (a floor plan's door) instead of the asset itself. A placement's
``state: {lit: true}`` starts that instance with another value.

The loaders only record these (``meta.declares``, not exported). ``collect`` turns a
built scene's into a ``Declarations``: keys by stable ID (``<thing id>#<name>``,
``world:<package>.<name>``, ``actor:<package>.<name>``), every interaction as the
instance state it is, and signals and inputs resolved. It raises ``StateError``
listing everything wrong, each with the file and line it was written at. Exports put
``Declarations.to_json()`` in the manifest and chunk index as ``state`` (schema
``docs/schema/geogen-state.v1.schema.json``); ``variables`` and ``interactions`` are
those of a rules program (``geogen.rules``), so ``program()`` is one.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from . import refs
from .core.node import SceneNode

FORMAT = "geogen-state"
VERSION = 1
SCHEMA = "docs/schema/geogen-state.v1.schema.json"
META = "declares"                 # node.meta key the loaders record declarations under
TYPES = ("bool", "number", "text", "choice")
SCOPES = {
    "world": "one value shared by everyone in the package",
    "actor": "one value per actor: the player or an NPC",
}
SIGNAL_SOURCES = {
    "emit": "an interaction arrives at a state with this `emit`",
    "changed": "this variable or interaction takes a new value (`to`: only that value)",
    "enter": "an actor enters this place or room (a reference; `self` for the asset)",
    "leave": "an actor leaves this place or room",
}
VARIABLE_KEYS = {"type", "default", "choices", "scope"}
INPUT_KEYS = {"set", "to", "of"}
LOCKED = "_locked"                # <interaction>_locked is that interaction's lock, a bool

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALIASES = {"string": "text", "str": "text", "boolean": "bool", "int": "number", "float": "number",
            "integer": "number", "enum": "choice"}


class StateError(ValueError):
    """Declarations that are malformed, clash, or name something that isn't there."""


def fits(spec: dict[str, Any], value: Any) -> bool:
    """Whether ``value`` is one a variable declared as ``spec`` can hold."""
    kind = spec.get("type")
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "text":
        return isinstance(value, str)
    return value in spec.get("choices", [])


def _described(spec: dict[str, Any]) -> str:
    return f"a choice of {', '.join(spec['choices'])}" if spec["type"] == "choice" else f"a {spec['type']}"


def _where(source: Any, keys: Iterable[str | int]) -> str:
    from .layout.yaml_utils import source_of

    return source_of(source, list(keys))


# ---- reading YAML -------------------------------------------------------------------------------

def _mapping(spec: Any, source: Any, key: str, form: str) -> dict[str, Any]:
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise StateError(f"{_where(source, [key])}: {key} is a mapping of name -> {form}")
    for name, entry in spec.items():
        where = _where(source, [key, name])
        if not _NAME.match(str(name)):
            raise StateError(f"{where}: names are letters, digits and '_' (they are read as thing.{name})")
        if not isinstance(entry, dict):
            raise StateError(f"{where}: write {form}, got {entry!r}")
    return {str(k): v for k, v in spec.items()}


def _known(entry: dict[str, Any], known: Iterable[str], where: str) -> None:
    for key in entry:
        if key not in known:
            raise StateError(f"{where}: unknown key '{key}'.{refs._near(str(key), known)} Keys: {', '.join(sorted(known))}")


def parse_variables(spec: Any, source: Any = None, scoped: bool = False) -> dict[str, dict[str, Any]]:
    """A ``state:`` block as ``{name: {type, default, choices?, scope?}}``.

    ``scoped`` is a scene's block, where each variable must say ``scope: world | actor``;
    otherwise it is an asset's, whose state belongs to the instance.
    """
    out = {}
    for name, var in _mapping(spec, source, "state", "{type, default}").items():
        where = _where(source, ["state", name])
        _known(var, VARIABLE_KEYS, where)
        if name.endswith(LOCKED):
            raise StateError(f"{where}: names ending '{LOCKED}' are interactions' locks")
        scope = var.get("scope")
        if scoped and scope not in SCOPES:
            raise StateError(f"{where}: a scene's state needs scope: world (one shared value) or actor (one per "
                             f"player or NPC)" + ("; instance state is declared by the asset" if scope else ""))
        if not scoped and scope not in (None, "instance"):
            raise StateError(f"{where}: an asset's state belongs to each placed instance; {scope} state is "
                             f"declared by a scene")
        kind = var.get("type")
        if kind not in TYPES:
            hint = f" Did you mean '{_ALIASES[kind]}'?" if kind in _ALIASES else ""
            raise StateError(f"{where}: type must be one of {', '.join(TYPES)}, got {kind!r}.{hint}")
        entry: dict[str, Any] = {"type": kind}
        if kind == "choice":
            choices = var.get("choices")
            if not isinstance(choices, list) or not choices:
                raise StateError(f"{where}: a choice needs choices: [..]")
            entry["choices"] = [str(c) for c in choices]
        elif "choices" in var:
            raise StateError(f"{where}: only type: choice takes choices")
        if "default" not in var and kind != "choice":
            raise StateError(f"{where}: needs a default")
        default = var.get("default", entry.get("choices", [None])[0])
        if kind == "number" and isinstance(default, float) and default.is_integer():
            default = int(default)
        if not fits(entry, default):
            raise StateError(f"{where}: default {default!r} isn't {_described(entry)}")
        entry["default"] = default
        if scoped:
            entry["scope"] = scope
        out[name] = entry
    return out


def parse_signals(spec: Any, source: Any = None) -> dict[str, dict[str, Any]]:
    """A ``signals:`` block: each entry one source (``SIGNAL_SOURCES``), names still unresolved."""
    out = {}
    for name, signal in _mapping(spec, source, "signals", "{emit | changed | enter | leave: <name>}").items():
        where = _where(source, ["signals", name])
        kinds = [k for k in SIGNAL_SOURCES if k in signal]
        if len(kinds) != 1:
            raise StateError(f"{where}: a signal has one of {', '.join(SIGNAL_SOURCES)}")
        kind = kinds[0]
        _known(signal, {kind, *({"emit": ["of"], "changed": ["of", "to"]}.get(kind, []))}, where)
        if not isinstance(signal[kind], str) or not signal[kind]:
            raise StateError(f"{where}: {kind} takes a name, got {signal[kind]!r}")
        out[name] = dict(signal)
    return out


def parse_inputs(spec: Any, source: Any = None) -> dict[str, dict[str, Any]]:
    """An ``inputs:`` block: ``{set: <variable or interaction>, to?, of?}``, names still unresolved."""
    out = {}
    for name, entry in _mapping(spec, source, "inputs", "{set: <state>, to: <value>}").items():
        where = _where(source, ["inputs", name])
        _known(entry, INPUT_KEYS, where)
        if not isinstance(entry.get("set"), str) or not entry["set"]:
            raise StateError(f"{where}: an input needs set: <a variable or interaction of the asset>")
        out[name] = dict(entry)
    return out


def declare_asset(root: SceneNode, data: dict[str, Any], source: Any = None) -> None:
    """Record an asset's ``state:`` / ``signals:`` / ``inputs:`` on its root (``meta.declares``)."""
    if not any(data.get(k) for k in ("state", "signals", "inputs")):
        return
    root.meta[META] = {"source": str(source) if source is not None else None,
                       "state": parse_variables(data.get("state"), source),
                       "signals": parse_signals(data.get("signals"), source),
                       "inputs": parse_inputs(data.get("inputs"), source)}


def declare_scene(root: SceneNode, data: dict[str, Any], source: Any = None) -> None:
    """Record a scene's ``state:`` (world and actor variables) on its root."""
    if data.get("state"):
        root.meta[META] = {"source": str(source) if source is not None else None,
                           "state": parse_variables(data["state"], source, scoped=True)}


def instance_state(node: SceneNode) -> dict[str, dict[str, Any]]:
    """The instance variables ``node`` declares or has (a train's stations), not its interactions."""
    declared = (node.meta.get(META) or {}).get("state", {})
    return {**{name: var for name, var in declared.items() if "scope" not in var}, **train_state(node)}


def state_names(node: SceneNode) -> list[str]:
    """Every state name a thing has: its interactions, their locks and its declared variables."""
    return ([i.name for i in node.interactions] + [f"{i.name}{LOCKED}" for i in node.interactions]
            + list(instance_state(node)))


def set_values(node: SceneNode, values: dict[str, Any], where: str) -> None:
    """Start an instance with other values than its asset's defaults (a placement's ``state:``)."""
    declared = instance_state(node)
    for name, value in values.items():
        spec = declared[name]
        if spec["type"] == "number" and isinstance(value, float) and value.is_integer():
            value = int(value)
        if not fits(spec, value):
            raise StateError(f"{where}: '{name}' is {_described(spec)}: can't be {value!r}")
        node.meta[META].setdefault("values", {})[name] = value


# ---- the collected declarations -------------------------------------------------------------------

class Declarations:
    """What a package declares, keyed the way the runtime and saves name it."""

    def __init__(self, package: str) -> None:
        self.package = package
        self.variables: dict[str, dict[str, Any]] = {}      # key -> {type, default, choices?}
        self.interactions: dict[str, dict[str, Any]] = {}   # <thing>#<name> -> {initial, duration, states, locked?}
        self.signals: dict[str, dict[str, Any]] = {}        # <thing>#<name> -> {subject, name, from?}
        self.inputs: dict[str, dict[str, Any]] = {}         # <thing>#<name> -> {set, to?}
        self._names: dict[str, list[str]] | None = None

    def __bool__(self) -> bool:
        return bool(self.variables or self.interactions or self.signals or self.inputs)

    def declare(self, key: str, spec: dict[str, Any]) -> None:
        """Add a variable (what quests and content packages compile to)."""
        if self.spec(key) is not None:
            raise StateError(f"'{key}' is already declared")
        if spec.get("type") not in TYPES or "default" not in spec or not fits(spec, spec["default"]):
            raise StateError(f"'{key}': needs a type ({', '.join(TYPES)}) and a default of that type")
        self.variables[key] = dict(spec)
        self._names = None

    def spec(self, key: str) -> dict[str, Any] | None:
        """``{type, default, choices?}`` of any variable: declared, an interaction or its lock."""
        if key in self.variables:
            return self.variables[key]
        if key in self.interactions:
            machine = self.interactions[key]
            return {"type": "choice", "choices": list(machine["states"]), "default": machine["initial"]}
        if key.endswith(LOCKED) and key[:-len(LOCKED)] in self.interactions:
            return {"type": "bool", "default": bool(self.interactions[key[:-len(LOCKED)]].get("locked", False))}
        return None

    def names(self, thing: str) -> list[str]:
        """The state names of one thing (by ID)."""
        if self._names is None:
            self._names = {}
            for key in self.interactions:
                owner, _, name = key.rpartition("#")
                self._names.setdefault(owner, []).extend([name, name + LOCKED])
            for key in self.variables:
                owner, sep, name = key.rpartition("#")
                if sep:
                    self._names.setdefault(owner, []).append(name)
        return self._names.get(thing, [])

    def scoped(self, scope: str, package: str | None = None) -> list[str]:
        """Names of the ``world`` or ``actor`` variables of a package (default: this one)."""
        prefix = f"{scope}:{package or self.package}."
        return [key[len(prefix):] for key in self.variables if key.startswith(prefix)]

    def undeclared(self, key: str) -> str | None:
        """Why ``key`` can't be read or set (None when it is declared): near names, and the
        right owner when the name exists under another."""
        if self.spec(key) is not None:
            return None
        thing, sep, name = key.rpartition("#")
        if sep:
            known = self.names(thing)
            return (f"'{thing}' has no state '{name}'.{refs._near(name, known)} "
                    + (f"It has: {refs._listed(known)}" if known else "It has none"))
        scope, _, rest = key.partition(":")
        package, _, name = rest.partition(".")
        for other in SCOPES:
            if other != scope and name in self.scoped(other, package):
                return f"'{name}' is {other} state, not {scope} state: write {other}.{name}"
        known = self.scoped(scope, package)
        return (f"no {scope} state '{name}' is declared in '{package}'.{refs._near(name, known)} "
                + (f"Declared: {refs._listed(known)}" if known else f"It declares no {scope} state"))

    def check_condition(self, tree: Any) -> list[str]:
        """What a resolved condition (``geogen.conditions``) reads that isn't declared."""
        from .rules import _reads

        return [problem for problem in dict.fromkeys(self.undeclared(key) for key in _reads(tree)) if problem]

    def signal(self, thing: str, name: str) -> dict[str, Any]:
        """The declared signal ``name`` of a thing (by ID); StateError naming the near ones."""
        return self._offered(self.signals, "signal", thing, name)

    def input(self, thing: str, name: str) -> dict[str, Any]:
        """The declared input ``name`` of a thing (by ID); StateError naming the near ones."""
        return self._offered(self.inputs, "input", thing, name)

    def _offered(self, table: dict[str, dict[str, Any]], kind: str, thing: str, name: str) -> dict[str, Any]:
        entry = table.get(f"{thing}#{name}")
        if entry is None:
            known = [k.rpartition("#")[2] for k in table if k.rpartition("#")[0] == thing]
            raise StateError(f"'{thing}' has no {kind} '{name}'.{refs._near(name, known)} "
                             + (f"Its {kind}s: {refs._listed(known)}" if known else f"It declares no {kind}s"))
        return entry

    def signal_rules(self) -> list[dict[str, Any]]:
        """The rules that raise the signals with a source (``from``): what declaring one means."""
        return [{"id": f"signal:{key}", "on": signal["from"],
                 "do": [{"emit": signal["name"], "subject": signal["subject"]}]}
                for key, signal in self.signals.items() if "from" in signal]

    def program(self, rules: Iterable[dict[str, Any]] = (), bindings: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
        """A rules program (``geogen.rules``) over these declarations."""
        from .rules import FORMAT as RULES_FORMAT, VERSION as RULES_VERSION

        return {"format": RULES_FORMAT, "version": RULES_VERSION, "variables": self.variables,
                "interactions": self.interactions, "bindings": list(bindings),
                "rules": [*self.signal_rules(), *rules]}

    def to_json(self) -> dict[str, Any]:
        """The manifest's ``state`` section."""
        return {"format": FORMAT, "version": VERSION, "schema": SCHEMA, "package": self.package,
                "variables": self.variables, "interactions": self.interactions,
                "signals": self.signals, "inputs": self.inputs}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Declarations:
        """The declarations of an exported package, from its manifest's ``state`` section."""
        if data.get("format") != FORMAT or data.get("version") != VERSION:
            raise StateError(f"not a {FORMAT} v{VERSION} section")
        out = cls(data["package"])
        for name in ("variables", "interactions", "signals", "inputs"):
            setattr(out, name, dict(data.get(name, {})))
        return out


def _machine(interaction: Any) -> dict[str, Any]:
    """An interaction as the rules program has it: only what decides its state."""
    states = {}
    for name, state in interaction.states.items():
        entry = {k: getattr(state, k) for k in ("next", "then", "emit") if getattr(state, k) is not None}
        if state.after is not None:
            entry["after"] = float(state.after)
        states[name] = entry
    out = {"initial": interaction.initial, "duration": float(interaction.duration), "states": states}
    if interaction.lock and interaction.lock.get("locked"):
        out["locked"] = True
    return out


def npc_signal(node: SceneNode) -> str | None:
    """The signal an NPC raises when an actor greets it (``attention.greet.signal`` of its
    definition, ``greeted`` by default), with the greeter as the event's actor. None for
    anything else, and for an NPC that can't be greeted."""
    npc = node.meta.get("npc")
    if node.meta.get("type") != "npc" or not isinstance(npc, dict):
        return None
    greet = (npc.get("attention") or {}).get("greet") or {}
    return str(greet["signal"]) if greet.get("signal") else None


def train_state(node: SceneNode) -> dict[str, dict[str, Any]]:
    """The variables a train has: ``at_<station>`` for every station it calls at, true while
    a train of that service stands there. The runtime sets them from the timetable
    (``driven``), so rules can read them and react to them changing but not set them."""
    train = node.meta.get("train")
    if node.meta.get("type") != "train" or not isinstance(train, dict):
        return {}
    names = dict.fromkeys(str(stop["station"]) for stop in train.get("stops", []))
    return {f"at_{name}": {"type": "bool", "default": False, "driven": "train"}
            for name in names if _NAME.match(name)}


def collect(root: SceneNode, package: str) -> Declarations:
    """Everything the built scene ``root`` declares, as package ``package``.

    Raises ``StateError`` with every problem: a clash between declarations, or a
    signal or input naming state, an emit or a thing that isn't there.
    """
    out = Declarations(package)
    problems: dict[str, None] = {}          # in order, once each: instances repeat their asset's mistakes
    declared_at: dict[str, str] = {}
    offering: list[tuple[SceneNode, str, dict[str, Any]]] = []

    def uid(node: SceneNode) -> str:
        known = node.meta.get("uid")
        return known if isinstance(known, str) and known.startswith(package + ":") else refs.node_id(node, package, root)

    for node in root.iter_nodes():
        declares = node.meta.get(META)
        greeting = npc_signal(node)
        timetable = train_state(node)
        if not node.interactions and not declares and not greeting and not timetable:
            continue
        thing = uid(node)
        for name, spec in timetable.items():
            out.variables[f"{thing}#{name}"] = spec
        if greeting:
            out.signals[f"{thing}#{greeting}"] = {"subject": thing, "name": greeting}
        for interaction in node.interactions:
            out.interactions[f"{thing}#{interaction.name}"] = _machine(interaction)
        if not declares:
            continue
        source = declares.get("source")
        for name, var in declares.get("state", {}).items():
            where = _where(source, ["state", name])
            spec = {k: v for k, v in var.items() if k != "scope"}
            if "scope" not in var:
                if any(name == i.name for i in node.interactions):
                    problems[f"{where}: '{name}' is already an interaction of this asset (its state is '{name}')"] = None
                    continue
                spec["default"] = declares.get("values", {}).get(name, spec["default"])
                out.variables[f"{thing}#{name}"] = spec
                continue
            key = f"{var['scope']}:{package}.{name}"
            other = next((f"{s}:{package}.{name}" for s in SCOPES if s != var["scope"]
                          and f"{s}:{package}.{name}" in out.variables), None)
            if other is not None:
                problems[f"{where}: '{name}' is {var['scope']} state here and {other.partition(':')[0]} state at "
                         f"{declared_at[other]}"] = None
            elif key in out.variables and out.variables[key] != spec:
                problems[f"{where}: {var['scope']} state '{name}' is declared differently at {declared_at[key]}"] = None
            else:
                out.variables[key] = spec
                declared_at.setdefault(key, where)
        if declares.get("signals") or declares.get("inputs"):
            offering.append((node, thing, declares))

    for node, thing, declares in offering:      # once every thing's state is known
        source = declares.get("source")

        def target(ref: Any, where: str) -> tuple[SceneNode, str, str]:
            """(node, ID, how to call it) for ``of:`` / a place: the asset itself by default."""
            if ref in (None, "self"):
                return node, thing, "this asset"
            try:
                found = refs.resolve(node, str(ref))
            except refs.RefError as exc:
                raise StateError(f"{where}: {exc.message}") from None
            if not refs.identified(found):
                raise StateError(f"{where}: '{ref}' is a plain part, which has no ID: name an asset, a room, "
                                 f"or something tagged or with interactions")
            return found, uid(found), f"'{ref}'"

        def variable(name: str, owner: str, shown: str, where: str, value: Any, given: bool) -> str:
            """The key of state ``name`` of a thing, checking a value meant for it."""
            key = f"{owner}#{name}"
            spec = out.spec(key)
            if spec is None:
                known = out.names(owner)
                scoped = next((s for s in SCOPES if name in out.scoped(s)), None)
                raise StateError(
                    f"{where}: {shown} has no state '{name}'.{refs._near(name, known)} "
                    + (f"It has: {refs._listed(known)}" if known else "It has none")
                    + (f". '{name}' is {scoped} state, which belongs to the scene, not an instance" if scoped else ""))
            if given and not fits(spec, value):
                raise StateError(f"{where}: '{name}' is {_described(spec)}: can't be {value!r}")
            return key

        for name, signal in declares.get("signals", {}).items():
            where = _where(source, ["signals", name])
            try:
                entry: dict[str, Any] = {"subject": thing, "name": name}
                if "emit" in signal:
                    found, owner, shown = target(signal.get("of"), where)
                    emits = [s.emit for i in found.interactions for s in i.states.values() if s.emit]
                    if signal["emit"] not in emits:
                        raise StateError(f"{where}: no interaction of {shown} emits '{signal['emit']}'."
                                         f"{refs._near(signal['emit'], emits)} "
                                         + (f"It emits: {refs._listed(emits)}" if emits else "It emits nothing"))
                    if found is not node or signal["emit"] != name:     # else the interaction raises it already
                        entry["from"] = {"type": "signal", "subject": owner, "name": signal["emit"]}
                elif "changed" in signal:
                    found, owner, shown = target(signal.get("of"), where)
                    key = variable(signal["changed"], owner, shown, where, signal.get("to"), "to" in signal)
                    entry["from"] = {"type": "changed", "key": key, **({"to": signal["to"]} if "to" in signal else {})}
                else:
                    kind = "enter" if "enter" in signal else "leave"
                    found, owner, shown = target(signal[kind], where)
                    if found is not node and found.meta.get("type") != "place" \
                            and not isinstance(found.meta.get("room"), dict):
                        here = [n.name for n in node.iter_nodes() if n.meta.get("type") == "place"]
                        raise StateError(f"{where}: {shown} is not a place or a room. "
                                         + (f"Places here: {refs._listed(here)}" if here
                                            else "Declare one under places:"))
                    entry["from"] = {"type": kind, "subject": owner}
                out.signals[f"{thing}#{name}"] = entry
            except StateError as exc:
                problems[str(exc)] = None
        for name, spec in declares.get("inputs", {}).items():
            where = _where(source, ["inputs", name])
            try:
                found, owner, shown = target(spec.get("of"), where)
                key = variable(spec["set"], owner, shown, where, spec.get("to"), "to" in spec)
                out.inputs[f"{thing}#{name}"] = {"set": key, **({"to": spec["to"]} if "to" in spec else {})}
            except StateError as exc:
                problems[str(exc)] = None

    if not problems:
        from .rules import check_program

        problems.update(dict.fromkeys(p for p in check_program(out.program()) if p))
    if problems:
        raise StateError("\n".join(problems))
    return out


def summary(root: SceneNode, package: str | None = None) -> dict[str, Any] | None:
    """The manifest's ``state`` section for ``root`` (None when it declares nothing).

    The package defaults to the one ``root``'s stable IDs were assigned in.
    """
    if package is None:
        package = str(root.meta.get("uid", "")).partition(":")[0] or root.name
    declarations = collect(root, package)
    return declarations.to_json() if declarations else None
