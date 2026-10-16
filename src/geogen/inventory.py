"""Things actors carry (geogen.18; "Items" in docs/rules.md).

What an actor carries is actor state, ``actor:<pkg>.item_<id>``: the quest compiler declares it
and compiles ``give`` / ``take`` effects and ``has()`` conditions (``quests._Compiler.item``).
This module puts items in the world.

A **pickup** is an object an actor takes: an asset's ``pickup: <item>`` or
``pickup: {item, count}`` (or the same on a placement, so any asset can be placed as one). It
gets a ``pickup`` interaction, ``here | taken``: using it hides the object (and its colliders,
in the runtime) and an asset rule gives the item to whoever used it. Nothing new runs in a
runtime: an interaction, a visibility motion and a rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import roots
from .core.node import SceneNode

PICKUP = "pickup"
STATES = ("here", "taken")
PICKUP_KEYS = {
    "item": "the item it is (items/<id>.yaml): whoever takes it carries one more",
    "count": "how many it gives (default 1)",
    "prompt": "what the E prompt says (default: Take the <item's title>)",
}


def parse_pickup(spec: Any, root: SceneNode, name: str, assets_dir: str | Path | None = None,
                 source: Any = None) -> dict[str, Any]:
    """Make ``root`` a pickup: add its ``pickup`` interaction and the rule that gives the item.
    Returns the normalised ``{item, count}``; ValueError for a bad spec or an unknown item."""
    from .flows import FlowError, carried
    from .layout.interactions import Interaction, Motion, State, apply_state
    from .wiring import META

    if isinstance(spec, str):
        spec = {"item": spec}
    if not isinstance(spec, dict) or set(spec) - set(PICKUP_KEYS) or not isinstance(spec.get("item"), str):
        raise ValueError(f"{name}: pickup is an item's id, or {{{', '.join(PICKUP_KEYS)}}}, got {spec!r}")
    item, count = spec["item"], spec.get("count", 1)
    if not item.isidentifier():
        raise ValueError(f"{name}: pickup item is an item's id (items/<id>.yaml), got {item!r}")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"{name}: pickup count is a whole number of items, got {count!r}")
    path = roots.find(f"items/{item}.yaml", assets_dir)
    if not path.is_file():
        known = [p.stem for p in roots.glob("items/*.yaml", assets_dir)]
        raise ValueError(f"{name}: pickup item '{item}' has no file items/{item}.yaml. Items: {', '.join(known) or 'none'}")
    try:
        title = carried(path)["title"]
    except FlowError as exc:
        raise ValueError(str(exc)) from None
    if any(i.name == PICKUP for i in root.interactions):
        raise ValueError(f"{name}: pickup adds the '{PICKUP}' interaction, which it already has")
    parts = [node for node in root.children if node.mesh is not None or any(True for _ in node.iter_meshes())]
    if not parts:
        raise ValueError(f"{name}: a pickup needs something to see and take")
    prompt = str(spec.get("prompt") or f"Take the {title[0].lower() + title[1:]}")
    hide = Motion(parts, "visible", np.zeros(3), np.zeros(3), {"here": 1.0, "taken": 0.0})
    interaction = Interaction(name=PICKUP, states={"here": State(next="taken", prompt=prompt), "taken": State()},
                              initial="here", motions=[hide], targets=list(parts), duration=0.0)
    apply_state(root, interaction, "here")
    root.interactions.append(interaction)
    rule = {"id": PICKUP, "on": {"use": f"self.{PICKUP}"}, "do": [{"give": item, **({"count": count} if count > 1 else {})}]}
    wiring = root.meta.get(META)
    if wiring:
        wiring = {**wiring, "spec": {**wiring["spec"], "rules": [*(wiring["spec"].get("rules") or []), rule]}}
    else:
        wiring = {"source": str(source) if source is not None else None, "asset": True, "spec": {"rules": [rule]}}
    root.meta[META] = wiring
    return {"item": item, "count": count}
