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
