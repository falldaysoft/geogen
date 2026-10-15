"""HUD stats: the numbers and banners a scene shows on screen while it runs.

A scene declares what its game shows, as conditions over its state (``geogen.conditions``)::

    hud:
      stats:                                    # a row each, top left, in order
        - {label: Lives, value: world.lives}
        - {label: Wave, value: world.wave, of: 8}          # shown "3 / 8"; `of` may be an expression
        - {label: Caught, value: hole.received}
      banners:                                  # the first whose `when` holds is shown, mid-screen
        - {when: 'world.phase == "lost"', text: "The core has fallen", tone: bad}
        - {when: 'world.phase == "won"', text: "Every wave beaten", tone: good}

The loaders only record it (``meta.hud``). ``quests.collect`` compiles it with the rules
(references resolved, reads checked against what is declared) and the export writes the
manifest's and chunk index's ``hud`` section: ``{stats: [{label, value, of?}], banners:
[{when, text, tone}]}`` with compiled trees. A runtime evaluates them for the player
(``value`` may read ``actor.x``) whenever state changes; the Godot panel is
``runtime/godot/scripts/hud_stats.gd``. Nested scenes' entries follow the outer scene's.
"""

from __future__ import annotations

from typing import Any

from . import conditions
from .core.node import SceneNode

META = "hud"
KEYS = {
    "stats": "[{label, value: <condition>, of: <number or condition>}]: rows of numbers or text",
    "banners": "[{when: <condition>, text, tone: info | good | bad}]: the first that holds is shown mid-screen",
}
STAT_KEYS = {"label", "value", "of"}
BANNER_KEYS = {"when", "text", "tone"}
TONES = ("info", "good", "bad")


class HudError(ValueError):
    """A ``hud:`` block that isn't the shape above."""


def declare(root: SceneNode, data: dict[str, Any], source: Any) -> None:
    """Record a scene's ``hud:`` on its root (checked for shape; compiled at export)."""
    spec = data.get("hud")
    if not spec:
        return
    where = f"{source}: hud" if source is not None else "hud"
    if not isinstance(spec, dict) or set(spec) - set(KEYS):
        raise HudError(f"{where}: hud is {{{', '.join(KEYS)}}}, got {spec!r}")
    for key, keys, required in (("stats", STAT_KEYS, ("label", "value")), ("banners", BANNER_KEYS, ("when", "text"))):
        entries = spec.get(key) or []
        if not isinstance(entries, list):
            raise HudError(f"{where}.{key}: a list of {{{', '.join(sorted(keys))}}}")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or set(entry) - keys or any(k not in entry for k in required):
                raise HudError(f"{where}.{key}.{index}: {{{', '.join(sorted(keys))}}} "
                               f"({', '.join(required)} required), got {entry!r}")
            if key == "banners" and entry.get("tone", "info") not in TONES:
                raise HudError(f"{where}.{key}.{index}: tone is one of {', '.join(TONES)}")
    root.meta[META] = {"source": str(source) if source is not None else None, "spec": spec}


def sources(root: SceneNode) -> list[tuple[dict[str, Any], Any, SceneNode]]:
    """(spec, source, node) of every scene below ``root`` that declares a HUD, outermost first."""
    return [(entry["spec"], entry.get("source"), node) for node in root.iter_nodes()
            for entry in [node.meta.get(META)] if entry]


def compile_hud(compiler: Any, entries: list[tuple[dict[str, Any], Any, SceneNode]]) -> dict[str, Any]:
    """The manifest's ``hud`` section; problems go to ``compiler`` (a ``quests._Compiler``)."""
    from .quests import QuestError, _Quest

    out: dict[str, list[dict[str, Any]]] = {"stats": [], "banners": []}
    for spec, source, node in entries:
        src = _Quest(compiler.uid(node), {}, source, node, node)
        if source is None:
            src.where = lambda *keys: "hud: " + ".".join(str(k) for k in keys)      # type: ignore[method-assign]

        def tree(value: Any, *at: Any) -> Any:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            try:
                compiled = compiler.condition(src, value)
                if conditions.event_fields(compiled):
                    raise QuestError("the HUD has no event to read: event.<field> is for rules")
                return compiled
            except QuestError as exc:
                compiler.bad(src.where("hud", *at), str(exc))
                return None

        for index, stat in enumerate(spec.get("stats") or []):
            entry = {"label": str(stat["label"]), "value": tree(stat["value"], "stats", index, "value")}
            if "of" in stat:
                entry["of"] = tree(stat["of"], "stats", index, "of")
            out["stats"].append(entry)
        for index, banner in enumerate(spec.get("banners") or []):
            out["banners"].append({"when": tree(banner["when"], "banners", index, "when"),
                                   "text": str(banner["text"]), "tone": banner.get("tone", "info")})
    return {key: value for key, value in out.items() if value}
