"""Reading authored YAML blocks: one way to reject a block's unknown keys and bad values.

Every message starts with ``where`` (a ``source_of`` path, or what the block is), so a mistake
is reported where it was written rather than as a bare ``float()`` failure.
"""

from __future__ import annotations

import difflib
from typing import Any, Iterable


def known_keys(spec: Any, known: Iterable[str], where: str, error: type[Exception] = ValueError) -> None:
    """Refuse keys of the mapping ``spec`` that aren't in ``known``."""
    known = sorted(str(k) for k in known)
    extra = sorted(str(k) for k in spec if str(k) not in known)
    if extra:
        close = difflib.get_close_matches(extra[0], known, n=1, cutoff=0.5)
        hint = f" (did you mean '{close[0]}'?)" if close else ""
        raise error(f"{where}: unknown keys {extra}{hint}; known: {known}")


def block(spec: Any, known: Iterable[str], where: str, error: type[Exception] = ValueError) -> dict[str, Any]:
    """``spec`` checked to be a mapping of only ``known`` keys."""
    known = list(known)
    if not isinstance(spec, dict):
        raise error(f"{where}: expected {{{', '.join(str(k) for k in known)}}}, got {spec!r}")
    known_keys(spec, known, where, error)
    return spec


def number(spec: dict[str, Any], key: str, default: Any, where: str, error: type[Exception] = ValueError) -> float:
    """``spec[key]`` (or ``default``) as a float."""
    value = spec.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise error(f"{where}: {key} is a number, got {value!r}") from None


def integer(spec: dict[str, Any], key: str, default: Any, where: str, error: type[Exception] = ValueError) -> int:
    """``spec[key]`` (or ``default``) as an int."""
    value = spec.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise error(f"{where}: {key} is a whole number, got {value!r}") from None


def choice(spec: dict[str, Any], key: str, default: Any, choices: Iterable[Any], where: str,
           error: type[Exception] = ValueError) -> Any:
    """``spec[key]`` (or ``default``), which must be one of ``choices``."""
    value = spec.get(key, default)
    choices = list(choices)
    if value not in choices:
        raise error(f"{where}: {key} is one of {', '.join(str(c) for c in choices)}, got {value!r}")
    return value
