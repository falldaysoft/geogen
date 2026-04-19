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

Not supported (by design): function calls, attribute access, comparison,
boolean, names other than declared params. The evaluator rejects any AST node
outside the allow-list.
"""

from __future__ import annotations

import ast
import re
from typing import Any


# Matches a {…} interpolation, allowing nested braces are disallowed (simple regex).
_INTERP_RE = re.compile(r"\{([^{}]+)\}")

# Matches a bare unit-suffixed literal, e.g. "50cm", "1.5m", "20%".
# Used for values given as strings *without* wrapping braces.
_UNIT_LITERAL_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(m|cm|mm|%)\s*$")


class ExpressionError(ValueError):
    """Raised when an expression cannot be safely evaluated."""


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


def _eval_ast(node: ast.AST, params: dict[str, float]) -> float:
    """Recursively evaluate a restricted AST."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, params)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ExpressionError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id not in params:
            raise ExpressionError(
                f"Unknown parameter '{node.id}'. Available: {sorted(params)}"
            )
        return float(params[node.id])
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, params)
        right = _eval_ast(node.right, params)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ExpressionError(f"Unsupported operator: {type(node.op).__name__}")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand, params)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ExpressionError(f"Unsupported unary op: {type(node.op).__name__}")
    raise ExpressionError(f"Unsupported expression node: {type(node).__name__}")


def evaluate(expr: str, params: dict[str, float]) -> float:
    """Evaluate a single expression string (no surrounding braces).

    Handles unit suffixes like `50cm` or `20%` directly, falling back to
    Python's AST parser for general arithmetic.
    """
    literal = _literal_with_unit(expr)
    if literal is not None:
        return literal
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Could not parse expression {expr!r}: {e}") from e
    return _eval_ast(tree, params)


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
        return evaluate(m.group(1), params)

    # Multiple / embedded interpolations → keep as string.
    def _sub(match: re.Match[str]) -> str:
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
        resolved[name] = float(value)

    return resolved
