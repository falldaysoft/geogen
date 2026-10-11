"""Safe expression evaluation for parametric asset YAML.

Supports `{expr}` interpolation in YAML values, where `expr` is a small
arithmetic expression referencing declared `params:` entries.

Supported syntax:
- Numeric literals: `3`, `0.15`, `-2.5`
- Parameter references: `width`, `wall_thickness`
- Binary operators: `+`, `-`, `*`, `/`
- Unary sign: `-x`, `+x`
- Parentheses: `(a + b) * c`
- Unit-suffixed literals: `50cm`, `2m`, `1.5mm` (meters-normalised)
- Percent literals: `20%` → 0.2
- A few functions: `sind(deg)`, `cosd(deg)`, `min(a, b, ...)`, `max(...)`, `abs(x)`, `round`, `floor`, `ceil`
  (e.g. placing a signpost's arms around its post)
- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=` (chained too: `2 <= storeys < 5`), `in` / `not in`
  a list of literals; numbers compare with numbers and text with text
- Booleans: `and`, `or`, `not`, `true`, `false`; 0, "" and false are false
- A conditional: `a if condition else b`
- Text literals, to compare a choice param (`paint == "taxi"`) or to pick one
  (`"brick" if storeys > 1 else "wood"`)

A comparison or boolean evaluates to 1.0 or 0.0, which is what `when:` reads.

Not supported (by design): other function calls, attribute access, names other
than declared params. The evaluator rejects any AST node outside the allow-list.

Conditions over live state (a door's state, a quest variable) can't be
evaluated here: `geogen.conditions` compiles those for the runtime.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any


# Matches a {…} interpolation, allowing nested braces are disallowed (simple regex).
_INTERP_RE = re.compile(r"\{([^{}]+)\}")

# Matches a bare unit-suffixed literal, e.g. "50cm", "1.5m", "20%".
# Used for values given as strings *without* wrapping braces.
_UNIT_LITERAL_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(m|cm|mm|%)\s*$")


class ExpressionError(ValueError):
    """Raised when an expression cannot be safely evaluated."""


_FUNCTIONS = {
    "sind": lambda deg: math.sin(math.radians(deg)),
    "cosd": lambda deg: math.cos(math.radians(deg)),
    "min": min,
    "max": max,
    "abs": abs,
    "round": lambda x: math.floor(x + 0.5),
    "floor": math.floor,
    "ceil": math.ceil,
}


def _literal_with_unit(token: str) -> float | None:
    """Parse a bare unit-suffixed literal ("50cm", "20%") into a float.

    Returns None if the token is not a unit-suffixed literal.
    """
    m = _UNIT_LITERAL_RE.match(token)
    if m is None:
        return None
    value = float(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return value
    if unit == "cm":
        return value * 0.01
    if unit == "mm":
        return value * 0.001
    if unit == "%":
        return value / 100.0
    raise ExpressionError(f"Unknown unit: {unit!r}")


def truthy(value: Any) -> bool:
    """Whether an expression value counts as true: not false, 0 or ""."""
    return bool(value)


def _number(value: Any, node: ast.AST) -> float:
    if isinstance(value, str):
        if isinstance(node, ast.Name):
            raise ExpressionError(f"Choice parameter '{node.id}' can't be used in arithmetic")
        raise ExpressionError(f"Text {value!r} can't be used in arithmetic")
    return float(value)


def compare(op: str, left: Any, right: Any) -> bool:
    """One comparison. ``==`` / ``!=`` take any two values (text never equals a
    number); ordering needs two numbers or two texts."""
    text = isinstance(left, str), isinstance(right, str)
    if op in ("==", "!="):
        return ((left == right) if text[0] == text[1] else False) == (op == "==")
    if text[0] != text[1]:
        raise ExpressionError(f"Can't order {left!r} and {right!r}: one is text, the other a number")
    return {"<": left < right, "<=": left <= right, ">": left > right, ">=": left >= right}[op]


COMPARISONS = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}


def _eval_ast(node: ast.AST, params: dict[str, Any]) -> float | str | bool:
    """Recursively evaluate a restricted AST."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, params)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (bool, str)):
            return node.value
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ExpressionError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id not in params:
            if node.id in ("true", "false"):
                return node.id == "true"
            raise ExpressionError(
                f"Unknown parameter '{node.id}'. Available: {sorted(params)}"
            )
        value = params[node.id]
        return value if isinstance(value, (str, bool)) else float(value)
    if isinstance(node, ast.BinOp):
        left = _number(_eval_ast(node.left, params), node.left)
        right = _number(_eval_ast(node.right, params), node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ExpressionError(f"Unsupported operator: {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left, params)
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                if not isinstance(comparator, (ast.Tuple, ast.List)):
                    raise ExpressionError(f"'in' needs a list of values: {ast.unparse(node)}")
                found = any(compare("==", left, _eval_ast(e, params)) for e in comparator.elts)
                if found != isinstance(op, ast.In):
                    return False
                continue
            if type(op) not in COMPARISONS:
                raise ExpressionError(f"Unsupported comparison: {type(op).__name__}")
            right = _eval_ast(comparator, params)
            if not compare(COMPARISONS[type(op)], left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(truthy(_eval_ast(v, params)) for v in node.values)
        return any(truthy(_eval_ast(v, params)) for v in node.values)
    if isinstance(node, ast.IfExp):
        return _eval_ast(node.body if truthy(_eval_ast(node.test, params)) else node.orelse, params)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords or not node.args:
            raise ExpressionError(f"Unsupported call: {ast.unparse(node)} (allowed: {sorted(_FUNCTIONS)})")
        args = [_number(_eval_ast(a, params), a) for a in node.args]
        try:
            return float(_FUNCTIONS[node.func.id](*args))
        except TypeError as e:
            raise ExpressionError(f"Bad arguments in {ast.unparse(node)}: {e}") from e
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not truthy(_eval_ast(node.operand, params))
        operand = _number(_eval_ast(node.operand, params), node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ExpressionError(f"Unsupported unary op: {type(node.op).__name__}")
    raise ExpressionError(f"Unsupported expression node: {type(node).__name__}")


def evaluate(expr: str, params: dict[str, float]) -> float | str:
    """Evaluate a single expression string (no surrounding braces).

    Handles unit suffixes like `50cm` or `20%` directly, falling back to
    Python's AST parser for general arithmetic. A number, or text when the
    expression picks one; true and false are 1.0 and 0.0.
    """
    literal = _literal_with_unit(expr)
    if literal is not None:
        return literal
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Could not parse expression {expr!r}: {e}") from e
    value = _eval_ast(tree, params)
    return float(value) if isinstance(value, bool) else value


def interpolate_string(value: str, params: dict[str, float]) -> str | float:
    """Resolve `{…}` interpolations and unit literals in a string value.

    - If `value` is exactly one `{expr}`, returns a float.
    - If `value` is a bare unit literal (e.g. `"50cm"`), returns a float.
    - Otherwise returns a string with each `{expr}` replaced by its value
      (when the expression resolves to a number, it's formatted with `str`).
    """
    # Bare unit literal, no braces needed.
    literal = _literal_with_unit(value)
    if literal is not None:
        return literal

    # Exactly one interpolation that spans the whole string → return scalar.
    m = _INTERP_RE.fullmatch(value)
    if m is not None:
        name = m.group(1).strip()
        if isinstance(params.get(name), str):      # a choice param: "{hair_style}" -> "bob"
            return params[name]
        return evaluate(m.group(1), params)

    # Multiple / embedded interpolations → keep as string.
    def _sub(match: re.Match[str]) -> str:
        if isinstance(params.get(match.group(1).strip()), str):
            return params[match.group(1).strip()]
        return str(evaluate(match.group(1), params))

    return _INTERP_RE.sub(_sub, value)


def resolve_value(value: Any, params: dict[str, float]) -> Any:
    """Recursively resolve interpolations inside a YAML-loaded value.

    - Strings are passed through `interpolate_string` (may yield a float).
    - Lists and dicts are walked.
    - Numbers / bools / None pass through unchanged.
    """
    if isinstance(value, str):
        return interpolate_string(value, params)
    if isinstance(value, list):
        return [resolve_value(item, params) for item in value]
    if isinstance(value, dict):
        return {k: resolve_value(v, params) for k, v in value.items()}
    return value


def resolve_params(
    declared: dict[str, Any] | None,
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build the final parameter dict by applying overrides to declared defaults.

    `declared` is the YAML `params:` block, e.g.
        {"width": {"default": 6}, "height": {"default": 3}}

    A param with ``choices`` is a string selector ({default: bob, choices: [bob, long]}),
    usable as a whole-value interpolation ("{hair_style}") but not in arithmetic; one with
    ``type: text`` takes any string ({default: Hotel, type: text}), also inside longer
    strings ("To the {dest}").
    Overrides take precedence.
    """
    declared = declared or {}
    overrides = overrides or {}

    resolved: dict[str, float] = {}
    for name, spec in declared.items():
        if not isinstance(spec, dict):
            raise ExpressionError(
                f"Param '{name}' must be a mapping (e.g. {{default: 1}}), got {type(spec).__name__}"
            )
        if "default" not in spec:
            raise ExpressionError(f"Param '{name}' requires a 'default' value")
        default = spec["default"]
        if spec.get("type") == "text":         # free text: signs' labels ("{label}", "To {dest}")
            resolved[name] = str(default)
            continue
        if "choices" in spec:
            if default not in spec["choices"]:
                raise ExpressionError(f"Param '{name}': default {default!r} not in choices {spec['choices']}")
            resolved[name] = default
            continue
        if isinstance(default, str):
            # Defaults can also use unit literals, e.g. default: "50cm"
            default = interpolate_string(default, {})
        resolved[name] = float(default)

    # Apply overrides, rejecting unknown names.
    for name, value in overrides.items():
        if name not in resolved:
            raise ExpressionError(
                f"Unknown param override '{name}'. Declared params: {sorted(resolved)}"
            )
        choices = declared[name].get("choices")
        if declared[name].get("type") == "text":
            resolved[name] = str(value)
        elif choices is not None:
            if value not in choices:
                raise ExpressionError(f"Param '{name}' must be one of {choices}, got {value!r}")
            resolved[name] = value
        else:
            resolved[name] = float(value)

    return resolved
