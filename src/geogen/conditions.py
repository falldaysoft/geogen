"""Conditions over live state, compiled so a runtime never parses expression text.

A condition is an expression (the syntax of ``layout.expressions``: comparisons,
``and`` / ``or`` / ``not``, ``a if c else b``, arithmetic, ``min`` / ``max`` / ``abs`` /
``round`` / ``floor`` / ``ceil``) whose terms read state that only exists while the
world runs::

    door.swing == "open"                       # a thing's state, by reference
    ref("plot/house/door").swing == "open"     # ... when the reference isn't a plain name
    self.power == "on"                         # the instance the condition is written on
    world.bridge_down and not actor.has_ticket # the package's shared and per-actor variables
    world("town").bridge_down                  # ... another package's (a content package reading a scene's)
    clock.time >= "22:30" or clock.night       # the world clock
    actor.room == "lobby"                      # where the acting character is
    all("[tag=light]", it.power == "off")      # every thing a selector matches
    count("house/[tag=door]", it.swing == "open") >= 2
    any("[tag=door]", it.swing == "open")

``compile_condition`` turns the text into a JSON tree; ``resolve`` then replaces
references and selectors with stable IDs (``geogen.refs``) against a built scene, so
what is exported names things the way saves do. ``evaluate`` is the reference
evaluator: a runtime must give the same answers.

The tree: a literal is a JSON number, string or boolean. Everything else is an object::

    {"op": "==", "args": [a, b]}        # also != < <= > >= + - * / neg not and or
                                        # if (condition, then, else), in (value, candidates...)
                                        # min max abs round floor ceil
    {"op": "all", "over": [id, ...], "where": tree}     # also any, count
    {"get": "instance", "id": "cottage:house/door", "name": "swing"}
    {"get": "world", "package": "cottage", "name": "bridge_down"}
    {"get": "actor", "package": "cottage", "name": "has_ticket"}
    {"get": "actor", "name": "room"}    # built in: ACTOR_BUILTINS
    {"get": "clock", "name": "time"}    # CLOCK
    {"get": "it", "name": "power"}      # inside all / any / count: the thing being tested

Before ``resolve`` a tree holds ``{"get": "ref", "ref": "door", "name": "swing"}``,
``{"get": "self", ...}``, ``world`` / ``actor`` terms without their package and
quantifiers with ``"select": "<selector>"`` in place of ``over``.

Rules (shared with the runtime): false, 0 and "" are false; ``and`` / ``or`` / ``not``
give a boolean; ``==`` between text and a number is false; ordering needs two numbers
or two texts ("HH:MM" times order as text); arithmetic needs numbers.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Callable, Iterable

from . import refs
from .core.node import SceneNode
from .layout.expressions import COMPARISONS, ExpressionError, compare, truthy

SCHEMA = "docs/schema/geogen-condition.v1.schema.json"

CLOCK = {
    "time": 'the time of day as "HH:MM" text (compare with text: clock.time >= "22:30")',
    "hour": "hours since midnight, with a fraction (13.5 at 13:30)",
    "night": "true between dusk and dawn",
    "day": "whole days since the world started",
}
ACTOR_BUILTINS = {
    "room": "the id of the room the actor is in, or \"\"",
    "kind": "\"player\" or \"npc\"",
}
ROOTS = ("world", "actor", "clock", "self", "it")
QUANTIFIERS = ("all", "any", "count")

_BINARY = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "min": min,
    "max": max,
    "abs": abs,
    "round": lambda x: math.floor(x + 0.5),
    "floor": math.floor,
    "ceil": math.ceil,
}
_ARITY = {"not": (1, 1), "neg": (1, 1), "if": (3, 3), "in": (2, None), "and": (2, None), "or": (2, None),
          "min": (1, None), "max": (1, None), "abs": (1, 1), "round": (1, 1), "floor": (1, 1), "ceil": (1, 1)}
OPS = tuple(COMPARISONS.values()) + tuple(_BINARY.values()) + tuple(_ARITY)

Tree = Any          # a literal or a node dict
Lookup = Callable[[dict[str, Any]], Any]


class ConditionError(ExpressionError):
    """A condition that can't be compiled, resolved or evaluated."""


# ---- compiling ----------------------------------------------------------------------------------

def compile_condition(text: Any, params: dict[str, Any] | None = None) -> Tree:
    """The tree for a condition. ``params`` are build-time values (an asset's or
    scene's params): a bare name must be one of them and becomes its value."""
    if isinstance(text, (bool, int, float)):
        return text
    if not isinstance(text, str) or not text.strip():
        raise ConditionError(f"expected a condition, got {text!r}")
    source = text.strip()
    if source.startswith("{") and source.endswith("}") and "{" not in source[1:]:
        source = source[1:-1]                    # written "{...}" like other expressions
    try:
        parsed = ast.parse(source.strip(), mode="eval")
    except SyntaxError as e:
        raise ConditionError(f"can't read condition {text!r}: {e.msg}") from e
    return _compile(parsed.body, params or {}, False)


def _compile(node: ast.AST, params: dict[str, Any], inside: bool) -> Tree:
    def sub(n: ast.AST) -> Tree:
        return _compile(n, params, inside)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (bool, int, float, str)):
            return node.value
        raise ConditionError(f"unsupported value {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in params:
            return params[node.id]
        if node.id in ("true", "false"):
            return node.id == "true"
        hint = (f"'{node.id}.<name>' reads it" if node.id in ROOTS else
                f"a thing's state is '{node.id}.<state>', a variable 'world.{node.id}' or 'actor.{node.id}'")
        raise ConditionError(f"'{node.id}' isn't a param here; {hint}")
    if isinstance(node, ast.Attribute):
        return _term(node, inside)
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY:
            raise ConditionError(f"unsupported operator in '{ast.unparse(node)}'")
        return {"op": _BINARY[type(node.op)], "args": [sub(node.left), sub(node.right)]}
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return {"op": "not", "args": [sub(node.operand)]}
        if isinstance(node.op, ast.USub):
            operand = sub(node.operand)
            if isinstance(operand, (int, float)) and not isinstance(operand, bool):
                return -operand
            return {"op": "neg", "args": [operand]}
        if isinstance(node.op, ast.UAdd):
            return sub(node.operand)
        raise ConditionError(f"unsupported operator in '{ast.unparse(node)}'")
    if isinstance(node, ast.BoolOp):
        return {"op": "and" if isinstance(node.op, ast.And) else "or", "args": [sub(v) for v in node.values]}
    if isinstance(node, ast.IfExp):
        return {"op": "if", "args": [sub(node.test), sub(node.body), sub(node.orelse)]}
    if isinstance(node, ast.Compare):
        parts, left = [], node.left
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                if not isinstance(right, (ast.Tuple, ast.List)) or not right.elts:
                    raise ConditionError(f"'in' needs a list of values: '{ast.unparse(node)}'")
                part: Tree = {"op": "in", "args": [sub(left)] + [sub(e) for e in right.elts]}
                parts.append(part if isinstance(op, ast.In) else {"op": "not", "args": [part]})
            elif type(op) in COMPARISONS:
                parts.append({"op": COMPARISONS[type(op)], "args": [sub(left), sub(right)]})
            else:
                raise ConditionError(f"unsupported comparison in '{ast.unparse(node)}'")
            left = right
        return parts[0] if len(parts) == 1 else {"op": "and", "args": parts}
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name in QUANTIFIERS:
            if inside:
                raise ConditionError(f"'{ast.unparse(node)}': {name}() can't be nested in another all / any / count")
            if (node.keywords or len(node.args) != 2 or not isinstance(node.args[0], ast.Constant)
                    or not isinstance(node.args[0].value, str)):
                raise ConditionError(f"'{ast.unparse(node)}': write {name}(\"<selector>\", <test using it.<state>>)")
            return {"op": name, "select": node.args[0].value, "where": _compile(node.args[1], params, True)}
        if name in _FUNCTIONS and node.args and not node.keywords:
            return {"op": name, "args": [sub(a) for a in node.args]}
        raise ConditionError(f"unsupported call '{ast.unparse(node)}' "
                             f"(allowed: {', '.join(sorted(_FUNCTIONS) + list(QUANTIFIERS))}, ref)")
    raise ConditionError(f"unsupported syntax '{ast.unparse(node)}'")


def _term(node: ast.Attribute, inside: bool) -> dict[str, Any]:
    owner = node.value
    if isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id == "ref":
        if owner.keywords or len(owner.args) != 1 or not isinstance(owner.args[0], ast.Constant) \
                or not isinstance(owner.args[0].value, str):
            raise ConditionError(f"'{ast.unparse(node)}': write ref(\"<reference>\").<state>")
        return {"get": "ref", "ref": owner.args[0].value, "name": node.attr}
    if isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id in ("world", "actor"):
        if owner.keywords or len(owner.args) != 1 or not isinstance(owner.args[0], ast.Constant) \
                or not isinstance(owner.args[0].value, str):
            raise ConditionError(f"'{ast.unparse(node)}': write {owner.func.id}(\"<package>\").<name>")
        return {"get": owner.func.id, "package": owner.args[0].value, "name": node.attr}
    if not isinstance(owner, ast.Name):
        raise ConditionError(f"'{ast.unparse(node)}': a term is <thing>.<state>; "
                             f"write a longer reference as ref(\"path/to/thing\").<state>")
    if owner.id == "it" and not inside:
        raise ConditionError(f"'{ast.unparse(node)}': 'it' only means something inside all / any / count")
    if owner.id == "clock" and node.attr not in CLOCK:
        raise ConditionError(f"'{ast.unparse(node)}': the clock has {', '.join(CLOCK)}")
    if owner.id in ROOTS:
        return {"get": owner.id, "name": node.attr}
    return {"get": "ref", "ref": owner.id, "name": node.attr}


def walk(tree: Tree) -> Iterable[dict[str, Any]]:
    """Every node dict of a tree (not its literals), parents first."""
    if isinstance(tree, dict):
        yield tree
        for arg in tree.get("args", []):
            yield from walk(arg)
        if "where" in tree:
            yield from walk(tree["where"])


# ---- resolving ----------------------------------------------------------------------------------

def _state_names(node: SceneNode) -> list[str]:
    from .state import state_names

    return state_names(node)


def resolve(tree: Tree, root: SceneNode, package: str, self_node: SceneNode | None = None,
            states: Callable[[SceneNode], Iterable[str]] = _state_names,
            packages: refs.Packages | None = None) -> Tree:
    """``tree`` with references and selectors replaced by stable IDs.

    ``root`` is the built scene of ``package``; ``self_node`` what ``self`` means
    (the instance a condition is written on). ``states(node)`` lists the state
    names a thing has: its interactions by default. Raises ``ConditionError``
    naming what a bad term could have been.
    """
    def uid(node: SceneNode, ref: refs.Ref | None = None) -> str:
        return refs.node_id(node, (ref.package if ref is not None and ref.package else package), node.root)

    def check(node: SceneNode, name: str, shown: str) -> None:
        known = list(states(node))
        if name not in known:
            raise ConditionError(f"'{shown}.{name}': '{shown}' has no state '{name}'{refs._near(name, known)} "
                                 f"({'it has ' + ', '.join(known) if known else 'it has none'})")

    def visit(t: Tree, members: list[SceneNode] | None, select: str) -> Tree:
        if not isinstance(t, dict):
            return t
        if "get" in t:
            kind, name = t["get"], t["name"]
            if kind == "ref":
                try:
                    ref = refs.parse(t["ref"])
                    node = refs.resolve(root, ref, package, packages=packages)
                except refs.RefError as e:
                    raise ConditionError(str(e)) from e
                check(node, name, t["ref"])
                return {"get": "instance", "id": uid(node, ref), "name": name}
            if kind == "self":
                if self_node is None:
                    raise ConditionError(f"'self.{name}': there is no 'self' here (it is the instance of an asset)")
                check(self_node, name, "self")
                return {"get": "instance", "id": uid(self_node), "name": name}
            if kind == "it":
                for node in members or []:
                    check(node, name, f"{select} ({refs.path_of(node)})")
                return dict(t)
            if kind == "world" or (kind == "actor" and name not in ACTOR_BUILTINS):
                return {"get": kind, "package": t.get("package", package), "name": name}
            return dict(t)
        if t["op"] in QUANTIFIERS:
            if "over" in t:
                return {**t, "where": visit(t["where"], None, "")}
            try:
                ref = refs.parse(t["select"])
                nodes = refs.select(root, ref, package, optional=False, packages=packages)
            except refs.RefError as e:
                raise ConditionError(str(e)) from e
            return {"op": t["op"], "over": [uid(n, ref) for n in nodes],
                    "where": visit(t["where"], nodes, t["select"])}
        return {"op": t["op"], "args": [visit(a, members, select) for a in t["args"]]}

    return visit(tree, None, "")


# ---- evaluating ---------------------------------------------------------------------------------

def _number(value: Any, op: str) -> float:
    if isinstance(value, str):
        raise ConditionError(f"'{op}' needs numbers, got text {value!r}")
    return float(value)


def evaluate(tree: Tree, lookup: Lookup) -> Any:
    """The value of a resolved tree. ``lookup(term)`` gives the current value of
    a ``get`` node (``it`` terms arrive as the ``instance`` term of each thing)."""
    def run(t: Tree, it: str | None) -> Any:
        if not isinstance(t, dict):
            return t
        if "get" in t:
            if t["get"] == "it":
                if it is None:
                    raise ConditionError("'it' outside all / any / count")
                return lookup({"get": "instance", "id": it, "name": t["name"]})
            if t["get"] in ("ref", "self"):
                raise ConditionError(f"unresolved term {t}: resolve() the condition first")
            return lookup(t)
        op = t["op"]
        if op in QUANTIFIERS:
            if "over" not in t:
                raise ConditionError(f"unresolved selector '{t.get('select')}': resolve() the condition first")
            hits = (truthy(run(t["where"], thing)) for thing in t["over"])
            return all(hits) if op == "all" else any(hits) if op == "any" else float(sum(hits))
        args = t["args"]
        if op == "and":
            return all(truthy(run(a, it)) for a in args)
        if op == "or":
            return any(truthy(run(a, it)) for a in args)
        if op == "not":
            return not truthy(run(args[0], it))
        if op == "if":
            return run(args[1] if truthy(run(args[0], it)) else args[2], it)
        values = [run(a, it) for a in args]
        try:
            if op == "in":
                return any(compare("==", values[0], v) for v in values[1:])
            if op in COMPARISONS.values():
                return compare(op, values[0], values[1])
        except ExpressionError as e:
            raise ConditionError(str(e)) from e
        numbers = [_number(v, op) for v in values]
        if op == "neg":
            return -numbers[0]
        if op == "+":
            return numbers[0] + numbers[1]
        if op == "-":
            return numbers[0] - numbers[1]
        if op == "*":
            return numbers[0] * numbers[1]
        if op == "/":
            if numbers[1] == 0:
                raise ConditionError("division by zero")
            return numbers[0] / numbers[1]
        if op in _FUNCTIONS:
            return float(_FUNCTIONS[op](*numbers))
        raise ConditionError(f"unknown op '{op}'")

    return run(tree, None)


def holds(tree: Tree, lookup: Lookup) -> bool:
    """Whether a condition is true."""
    return truthy(evaluate(tree, lookup))


def validate(tree: Tree) -> None:
    """Raise ``ConditionError`` unless ``tree`` is a well-formed, resolved tree
    (what a runtime may be handed)."""
    def check(t: Tree, inside: bool) -> None:
        if isinstance(t, (bool, int, float, str)):
            return
        if not isinstance(t, dict):
            raise ConditionError(f"not a condition node: {t!r}")
        if "get" in t:
            kind, name = t["get"], t.get("name")
            ok = isinstance(name, str) and (
                (kind == "instance" and isinstance(t.get("id"), str))
                or (kind in ("world", "actor") and isinstance(t.get("package"), str))
                or (kind == "actor" and name in ACTOR_BUILTINS)
                or (kind == "clock" and name in CLOCK)
                or (kind == "it" and inside))
            if not ok:
                raise ConditionError(f"bad or unresolved term: {t}")
            return
        op = t.get("op")
        if op in QUANTIFIERS:
            if inside or not isinstance(t.get("over"), list) or "where" not in t:
                raise ConditionError(f"bad or unresolved quantifier: {t}")
            return check(t["where"], True)
        args = t.get("args")
        if op not in OPS or not isinstance(args, list):
            raise ConditionError(f"unknown op: {t}")
        lo, hi = _ARITY.get(op, (2, 2))
        if len(args) < lo or (hi is not None and len(args) > hi):
            raise ConditionError(f"'{op}' takes {lo if hi == lo else f'{lo} or more'} arguments, got {len(args)}")
        for a in args:
            check(a, inside)

    check(tree, False)
