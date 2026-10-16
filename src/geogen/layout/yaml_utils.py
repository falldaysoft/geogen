"""YAML loading utilities with 1.2-ish bool resolution.

PyYAML's default (YAML 1.1) interprets the unquoted scalars `on`, `off`,
`yes`, `no`, `y`, `n` as booleans. That conflicts with our `on:` keyword
for surface-based placement. We work around it by defining a SafeLoader
subclass whose implicit bool resolver only matches `true`/`false`
(YAML 1.2 behaviour).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


class GeogenSafeLoader(yaml.SafeLoader):
    """SafeLoader that only treats `true`/`false` as booleans."""


# Wipe PyYAML's bool resolvers from this subclass and re-add a restricted one.
# Resolvers live in a dict keyed by the first character of the scalar.
for _ch in list("yYnNtTfFoO"):
    if _ch in GeogenSafeLoader.yaml_implicit_resolvers:
        GeogenSafeLoader.yaml_implicit_resolvers[_ch] = [
            (tag, regexp)
            for tag, regexp in GeogenSafeLoader.yaml_implicit_resolvers[_ch]
            if tag != "tag:yaml.org,2002:bool"
        ]

GeogenSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def safe_load(stream: Any) -> Any:
    """Drop-in replacement for `yaml.safe_load` using GeogenSafeLoader."""
    return yaml.load(stream, Loader=GeogenSafeLoader)


def safe_load_path(path: str | Path) -> Any:
    """Load a YAML file using GeogenSafeLoader."""
    with open(path) as f:
        return safe_load(f)


def load_kind(path: str | Path, kind: str, version: int | None = 1, keys: Any = None,
              error: type[Exception] = ValueError) -> dict[str, Any]:
    """A data file with a top-level ``kind:``, checked: the kind, the ``version`` (unless None)
    and, given ``keys``, no keys beyond them (``kind``, ``version`` and ``description`` are
    always allowed). Raises ``error`` naming the file."""
    path = Path(path)
    if not path.exists():
        raise error(f"no {kind} file {path}")
    data = safe_load_path(path) or {}
    if not isinstance(data, dict) or data.get("kind") != kind \
            or (version is not None and int(data.get("version", 0)) != version):
        raise error(f"{path}: expected kind '{kind}'" + (f" version {version}" if version is not None else ""))
    if keys is not None:
        unknown = set(data) - set(keys) - {"kind", "version", "description"}
        if unknown:
            raise error(f"{path}: unknown keys {sorted(unknown)}; known: {sorted(keys)}")
    return data


_composed: dict[str, tuple[float, Any]] = {}


def locate(path: str | Path, keys: list[str | int] | tuple[str | int, ...]) -> int | None:
    """The 1-based line of the value at ``keys`` (mapping keys / list indices) in a YAML file.

    Follows the path as far as it exists, so a missing last key gives its
    parent's line. None if the file can't be read. Merge keys (``<<``) are
    looked through.
    """
    path = Path(path)
    try:
        stamp = path.stat().st_mtime
        cached = _composed.get(str(path))
        if cached is None or cached[0] != stamp:
            with open(path) as f:
                cached = (stamp, yaml.compose(f, Loader=GeogenSafeLoader))
            _composed[str(path)] = cached
    except (OSError, yaml.YAMLError):
        return None
    node = cached[1]
    if node is None:
        return None
    line = node.start_mark.line
    for key in keys:
        found = _child(node, key)
        if found is None:
            break
        key_line, node = found
        line = key_line
    return line + 1


def _child(node: Any, key: str | int) -> tuple[int, Any] | None:
    """(line of the key, value node) for ``key`` in a mapping or sequence node."""
    if isinstance(node, yaml.SequenceNode):
        if isinstance(key, int) and 0 <= key < len(node.value):
            return node.value[key].start_mark.line, node.value[key]
        return None
    if not isinstance(node, yaml.MappingNode):
        return None
    merged = []
    for key_node, value in node.value:
        if key_node.value == str(key):
            return key_node.start_mark.line, value
        if key_node.value == "<<":
            merged.extend(value.value if isinstance(value, yaml.SequenceNode) else [value])
    for other in merged:
        found = _child(other, key)
        if found is not None:
            return found
    return None


def source_of(path: str | Path | None, keys: list[str | int] | tuple[str | int, ...]) -> str:
    """Where a value was written, for diagnostics: ``assets/scenes/park.yaml:12: place.bench.on``."""
    key_path = ".".join(str(k) for k in keys)
    if path is None:
        return key_path
    path = Path(path)
    try:
        shown = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        shown = path
    line = locate(path, keys)
    return f"{shown}:{line}: {key_path}" if line is not None else f"{shown}: {key_path}"
