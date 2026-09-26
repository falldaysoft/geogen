"""Character archetypes: reusable, seeded character definitions.

An archetype (``kind: archetype``, ``assets/characters/archetypes/*.yaml``)
names a body asset and a *distribution* over its params::

    kind: archetype
    version: 1
    body: characters/humanoid.yaml
    extends: base_adult.yaml               # optional: start from another archetype
    params:
      preset: { choice: [[feminine], [feminine, slim]], weights: [3, 1] }
      skin_tone: { random: [0.0, 1.0] }
      hair_style: { choice: [long, bob, ponytail] }
      outfit: casual_f                     # plain values are fixed
    variants:                              # optional: pick one first (by weight)
      - { name: woman, weight: 1, params: { preset: [feminine], hair_style: {choice: [bob, bun]} } }
      - { name: man, weight: 1, params: { preset: [masculine], hair_style: crop } }
      - { archetype: older_man.yaml, weight: 1 }   # or hand over to another archetype

Draw forms (``draw_params``, also used by ``scatter`` params):
``{random: [lo, hi]}`` (uniform), ``{normal: [mean, sd], clamp: [lo, hi]}``,
``{choice: [...], weights: [...]}``. ``extends`` merges the parent's params
first (the child's keys win). A ``variants:`` list correlates params that
must agree (a body preset with a hair style): one variant is drawn by
``weight``; its ``params`` override the archetype's, or, with
``archetype:``, the person is drawn from that archetype instead (a mix
such as ``townsfolk.yaml``), with the variant's ``params`` on top.

``resolve_character(path, seed)`` returns ``(asset, params)``; the same
archetype and seed always give the same person, and people with identical
resolved params share one body prototype through the loader cache.

Scenes use archetypes directly (``place: {ann: {archetype:
characters/archetypes/young_woman.yaml, seed: 3}}``, or with ``scatter:`` for a
crowd), and NPC definitions name one as their body
(``body: {archetype: characters/archetypes/young_woman.yaml}``; the NPC
placement's ``seed`` picks the person).
"""

from __future__ import annotations

import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .layout.yaml_utils import safe_load_path

ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"

ARCHETYPE_KIND = "archetype"
VERSION = 1
ARCHETYPE_KEYS = {"kind", "version", "name", "description", "body", "extends", "params", "tags", "variants"}
VARIANT_KEYS = {"name", "weight", "params", "archetype", "tags"}


def is_draw(value: Any) -> bool:
    return isinstance(value, dict) and bool({"random", "choice", "normal"} & set(value))


def draw(value: Any, rng: np.random.Generator, where: str = "param") -> Any:
    """One value from a draw spec (plain values pass through)."""
    if not is_draw(value):
        return value
    if "random" in value:
        lo, hi = value["random"]
        return round(float(rng.uniform(float(lo), float(hi))), 4)
    if "normal" in value:
        mean, sd = value["normal"]
        x = float(rng.normal(float(mean), float(sd)))
        if "clamp" in value:
            lo, hi = value["clamp"]
            x = min(max(x, float(lo)), float(hi))
        return round(x, 4)
    options = list(value["choice"])
    if not options:
        raise ValueError(f"{where}: empty choice")
    weights = value.get("weights")
    if weights is None:
        return options[int(rng.integers(len(options)))]
    w = np.asarray(weights, dtype=np.float64)
    if len(w) != len(options) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError(f"{where}: weights must be {len(options)} non-negative numbers, got {weights!r}")
    return options[int(rng.choice(len(options), p=w / w.sum()))]


def draw_params(params: dict[str, Any] | None, rng: np.random.Generator) -> dict[str, Any] | None:
    """Draw every draw-spec value in ``params`` (in key order, so a seed is stable)."""
    if not params:
        return params
    return {key: draw(value, rng, key) for key, value in params.items()}


@lru_cache(maxsize=64)
def _load(path: str) -> dict[str, Any]:
    data = safe_load_path(path) or {}
    if data.get("kind") != ARCHETYPE_KIND or int(data.get("version", 0)) != VERSION:
        raise ValueError(f"{path}: expected kind '{ARCHETYPE_KIND}' version {VERSION}")
    unknown = set(data) - ARCHETYPE_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; known: {sorted(ARCHETYPE_KEYS)}")
    params: dict[str, Any] = {}
    body = None
    tags: list[str] = []
    parent: dict[str, Any] = {}
    if data.get("extends"):
        parent = _load(str((Path(path).parent / data["extends"]).resolve()))
        params.update(parent["params"])
        body, tags = parent["body"], list(parent["tags"])
    params.update(data.get("params") or {})
    body = data.get("body", body)
    tags += [t for t in data.get("tags") or [] if t not in tags]
    variants = []
    for i, v in enumerate(data.get("variants") or []):
        if not isinstance(v, dict) or set(v) - VARIANT_KEYS:
            raise ValueError(f"{path}: variant {i}: keys are {sorted(VARIANT_KEYS)}, got {v!r}")
        variant = {"name": str(v.get("name", v.get("archetype", i))), "weight": float(v.get("weight", 1.0)),
                   "params": dict(v.get("params") or {}), "tags": list(v.get("tags") or [])}
        if "archetype" in v:
            variant["archetype"] = str((Path(path).parent / v["archetype"]).resolve())
            _load(variant["archetype"])       # validate eagerly (and catch cycles as recursion errors)
        variants.append(variant)
    if not variants and "extends" in data:
        variants = parent["variants"]
    if not body and variants and all("archetype" in v for v in variants):
        body = "(variants)"
    if not body:
        raise ValueError(f"{path}: needs a body asset")
    return {"name": data.get("name", Path(path).stem), "body": body, "params": params, "tags": tags,
            "variants": variants}


def load_archetype(path: str | Path) -> dict[str, Any]:
    """``{name, body, params (draw specs), tags}`` with ``extends`` applied."""
    return _load(str(Path(path).resolve()))


def _resolve(path: str, rng: np.random.Generator, overrides: dict[str, Any]) -> dict[str, Any]:
    archetype = _load(path)
    specs = dict(archetype["params"])
    tags = list(archetype["tags"])
    names = [archetype["name"]]
    body = archetype["body"]
    if archetype["variants"]:
        variants = archetype["variants"]
        w = np.array([v["weight"] for v in variants])
        variant = variants[int(rng.choice(len(variants), p=w / w.sum()))]
        tags += [t for t in variant["tags"] if t not in tags]
        if "archetype" in variant:
            inner = _resolve(variant["archetype"], rng, {**variant["params"], **overrides})
            inner["tags"] = [*tags, *[t for t in inner["tags"] if t not in tags]]
            inner["names"] = [*names, *inner["names"]]
            return inner
        specs.update(variant["params"])
        names.append(variant["name"])
    specs.update(overrides)
    return {"body": body, "params": draw_params(specs, rng) or {}, "tags": tags, "names": names}


def resolve_character(path: str | Path, seed: int = 0,
                      overrides: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """The person an archetype gives for ``seed``: ``(body asset path, params)``.

    ``overrides`` (plain values or draw specs) replace the archetype's
    params before drawing. The RNG is seeded by the archetype's name and
    ``seed``, so different archetypes with the same seed don't correlate.
    """
    person = _person(path, seed, overrides)
    return person["body"], person["params"]


def _person(path: str | Path, seed: int, overrides: dict[str, Any] | None) -> dict[str, Any]:
    full = str(Path(path).resolve())
    rng = np.random.default_rng([zlib.crc32(_load(full)["name"].encode()), int(seed) & 0xFFFFFFFF])
    return _resolve(full, rng, dict(overrides or {}))


def load_character(loader, assets_dir: Path, path: str | Path, seed: int = 0,
                   overrides: dict[str, Any] | None = None):
    """Build the archetype's person for ``seed`` (a body asset node, tagged and annotated)."""
    person = _person(Path(assets_dir) / path, seed, overrides)
    node = loader.load(Path(assets_dir) / person["body"], params=person["params"])
    node.tags = [*node.tags, *[t for t in person["tags"] if t not in node.tags]]
    node.meta["character"] = {"archetype": "/".join(person["names"]), "seed": int(seed)}
    return node
