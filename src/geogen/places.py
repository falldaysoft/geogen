"""Named places: trigger volumes a scene or an asset declares.

Rooms and travel portals have volumes of their own; ``places:`` names anywhere
else something should notice an actor arriving: the lighthouse gallery, the
jetty, this platform. In a scene or an asset::

    places:
      gallery:                         # a box: `position` is the middle of its bottom
        box: [6.8, 2.6, 6.8]
        position: [0, 10, 0]           # scenes also take anchor + offset, like slots
        yaw: 0
      lawn:                            # a footprint in the frame, from `y` up `height`
        polygon: [[-4, -3], [4, -3], [4, 3], [-4, 3]]
        y: 0
        height: 2.5
      deck:                            # a box round something, by reference
        around: deck
        margin: 0.3                    # or [x, y, z]
        headroom: 2.2                  # more above it, so standing on it counts
        tags: [outside]

Each becomes an empty node named after the place, a child of the scene or asset
root, with ``meta.type: place`` and ``meta.place`` (``{id, size}`` or
``{id, polygon, height}``; the node is the middle of the volume's bottom, the
volume rises along its +Y). It gets a stable ID like anything else
(``island:lighthouse/gallery``), so references, ``signals: {x: {enter: gallery}}``
and rules name it. Exports list them in the manifest's ``places`` in world space.
The runtime gives each an area and reports actors entering and leaving.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .core.node import SceneNode
from .core.transform import Transform

TYPE = "place"
SHAPES = ("box", "polygon", "around")
DEFAULT_HEIGHT = 2.5              # of a polygon place: about a storey


def is_place(node: SceneNode) -> bool:
    return node.meta.get("type") == TYPE and isinstance(node.meta.get("place"), dict)


def _numbers(value: Any, count: int, where: str, what: str) -> list[float]:
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        out = []
    if len(out) != count:
        raise ValueError(f"{where}: {what} must be {count} numbers, got {value!r}")
    return out


def build_places(root: SceneNode, spec: Any, where: str,
                 locate: Callable[[str, dict[str, Any]], np.ndarray] | None = None,
                 resolve: Callable[[str, str], SceneNode] | None = None,
                 known: set[str] | None = None) -> list[SceneNode]:
    """Add a place node under ``root`` for each entry of a ``places:`` block.

    ``locate(name, entry)`` gives a box's position when the entry doesn't say
    ``position`` (scenes resolve anchor + offset); ``resolve(name, reference)``
    finds the node an ``around:`` names. ``where`` starts error messages.
    """
    from .layout.schema import PLACE_KEYS

    if not isinstance(spec, dict):
        raise ValueError(f"{where}: places must be {{name: {{box | polygon | around, ...}}}}")
    taken = {child.name for child in root.children}
    out = []
    for name, entry in spec.items():
        at = f"{where}: place '{name}'"
        if not isinstance(entry, dict):
            raise ValueError(f"{at} must be {{box | polygon | around, ...}}")
        unknown = set(entry) - set(known or PLACE_KEYS)
        if unknown:
            raise ValueError(f"{at}: unknown keys {sorted(unknown)}; known: {sorted(known or PLACE_KEYS)}")
        shapes = [k for k in SHAPES if k in entry]
        if len(shapes) != 1:
            raise ValueError(f"{at} needs exactly one of {', '.join(SHAPES)}" + (f", got {shapes}" if shapes else ""))
        if name in taken:
            raise ValueError(f"{at}: something here is already called '{name}'")
        taken.add(name)
        yaw = float(entry.get("yaw", 0.0))
        place: dict[str, Any] = {"id": str(name)}
        if "box" in entry:
            size = _numbers(entry["box"], 3, at, "box")
            if "position" in entry or locate is None:
                position = np.array(_numbers(entry.get("position", [0, 0, 0]), 3, at, "position"))
            else:
                position = np.asarray(locate(name, entry), dtype=np.float64)
        elif "polygon" in entry:
            points = [_numbers(p, 2, at, "a polygon point") for p in entry["polygon"]]
            if len(points) < 3:
                raise ValueError(f"{at}: a polygon needs at least 3 points")
            area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
            if abs(area) < 1e-9:
                raise ValueError(f"{at}: the polygon has no area")
            if area < 0:                    # one winding, whichever way it was written
                points.reverse()
            place["polygon"] = [[round(x, 6), round(z, 6)] for x, z in points]
            place["height"] = float(entry.get("height", DEFAULT_HEIGHT))
            position, yaw = np.array([0.0, float(entry.get("y", 0.0)), 0.0]), 0.0
        else:
            from .travel import _local_box

            if resolve is None:
                raise ValueError(f"{at}: around isn't available here")
            box = _local_box(root, resolve(name, str(entry["around"])))
            if box is None:
                raise ValueError(f"{at}: '{entry['around']}' has no geometry to put a place round")
            margin = entry.get("margin", 0.0)
            margin = np.array(_numbers(margin, 3, at, "margin") if isinstance(margin, (list, tuple))
                              else [float(margin)] * 3)
            lo, hi = box[0] - margin, box[1] + margin
            hi[1] += float(entry.get("headroom", 0.0))
            size = [float(v) for v in hi - lo]
            position, yaw = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]), 0.0
        if "polygon" not in place:
            if min(size) <= 0:
                raise ValueError(f"{at}: the box must have a size, got {size}")
            place["size"] = [round(v, 6) for v in size]
        elif place["height"] <= 0:
            raise ValueError(f"{at}: height must be positive")
        tags = [str(t) for t in entry.get("tags", [])]
        node = SceneNode(name=str(name),
                         transform=Transform(translation=position, rotation=np.array([0.0, np.radians(yaw), 0.0])),
                         tags=[TYPE, *tags], meta={"type": TYPE, "place": place})
        root.add_child(node)
        out.append(node)
    return out


def places_summary(root: SceneNode) -> list[dict[str, Any]]:
    """Every place under ``root`` in world space, for the manifest.

    Boxes give ``center`` (of the volume), ``size`` and ``yaw_deg``, like rooms;
    polygons their ``polygon`` ([x, z] points), base ``y`` and ``height``.
    """
    out = []
    for node in root.iter_nodes():
        if not is_place(node):
            continue
        place, world = node.meta["place"], node.world_transform()
        scale = np.linalg.norm(world[:3, :3], axis=0)
        entry: dict[str, Any] = {"id": place["id"], "node": node.name,
                                 **({"uid": node.meta["uid"]} if "uid" in node.meta else {})}
        tags = [t for t in node.tags if t != TYPE]
        if tags:
            entry["tags"] = tags
        if "polygon" in place:
            points = np.array([[x, 0.0, z] for x, z in place["polygon"]]) @ world[:3, :3].T + world[:3, 3]
            entry.update(shape="polygon", polygon=[[round(float(p[0]), 6), round(float(p[2]), 6)] for p in points],
                         y=round(float(world[1, 3]), 6), height=round(float(place["height"] * scale[1]), 6))
        else:
            size = np.asarray(place["size"]) * scale
            centre = world[:3, 3] + world[:3, 1] * place["size"][1] / 2
            entry.update(shape="box", center=[round(float(v), 6) for v in centre],
                         size=[round(float(v), 6) for v in size],
                         yaw_deg=round(float(np.degrees(np.arctan2(world[0, 2], world[2, 2]))), 6))
        out.append(entry)
    return out
