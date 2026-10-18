"""The node extras contract: which ``SceneNode.meta`` keys leave the generator.

``docs/schema/geogen-extras.v1.schema.json`` is the one declaration of what a
glTF node's ``extras.geogen`` may hold. A meta key is exported only when the
schema declares it (``declared_keys``); keys the generator keeps for itself
(QA, furnishing, what goes in the manifest instead) are registered here in
``BUILD_ONLY`` with the reason, and never exported. Anything else is a mistake:
``exported_meta`` raises ``ExtrasError`` naming the key and the node, so a new
meta key can't reach a runtime until someone has said what it is.

To add a key: declare it in the schema (with its properties) when a runtime is
meant to read it, and list the script that reads it in the table in
``runtime/godot/README.md``; otherwise add it to ``BUILD_ONLY``.
``tests/test_extras_contract.py`` keeps the schema, this table and the Godot
scene builder's ``KNOWN_KEYS`` in step.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .core.node import SceneNode

SCHEMA = "docs/schema/geogen-extras.v1.schema.json"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / SCHEMA

# Meta keys that stay in the generator: {key: why it isn't exported}.
BUILD_ONLY: dict[str, str] = {
    # Exported some other way
    "traffic": "the lane graph: merged into the manifest's traffic section (traffic.build_traffic)",
    "railway": "the line's points, stations and crossings: the manifest's traffic.railways",
    "declares": "state, signals and inputs as authored: compiled into the manifest's state (state.collect)",
    "quests": "quest specs as authored: compiled into the manifest's quests and rules (quests.collect)",
    "dialogues": "dialogue specs as authored: compiled into the manifest's dialogues and rules (geogen.dialogue)",
    "wiring": "rules, wires and bindings as authored: compiled into the manifest's rules",
    "hud": "HUD stats and banners as authored: compiled into the manifest's hud (geogen.hud)",
    "physics_ticks": "a scene's physics: {ticks}: exported as the manifest's physics",
    "travel_spec": "the travel block as authored; the resolved one is exported as 'travel'",
    "travel_source": "the file a travel block came from, for error messages",
    "switch": "the light a switch turns on and off: compiled to a binding of that fixture's lit state "
              "(state.switch_bindings)",
    # References (geogen.refs): only the resulting uid is exported
    "source": "the asset file and params a root was built from (inspection, prototype caching)",
    "scope": "marks a placed scene, recipe or NPC as its own namespace for references",
    "ref_key": "a refs index stand-in node's key",
    # Furnishing and layout QA
    "openings": "a room's doors and windows per wall, for the furnishing solver, facades and layout QA",
    "wall_inset": "how far a room's lining and skirting stand off the walls, for furnishing and QA",
    "clear_height": "a room's floor-to-ceiling height, for its finishes",
    "placed_by": "marks what the furnishing solver placed, for layout QA",
    "furnish_report": "the furnishing rules a plan couldn't meet (--inspect and tests read it)",
    "affordance_actor": "a body staged at an affordance by affordance QA (never part of a scene)",
    "npc_owns": "an NPC's owns / prefers / work as nodes: exported as stable IDs in npc.owns|prefers|work",
    # What a generator was asked for; the geometry is the result
    "facade": "the facade style a building was clad in",
    "storey": "a storey's index, elevation and height (the tag storey.<n> carries the index)",
    "stair_flights": "the flights leaving a stair room: {to, rise, steps} (the flight itself exports 'stairs')",
    "terrain": "terrain parameters, re-evaluated by scatter and QA (terrain_from_meta)",
    "recipe": "the building recipe and params a building was generated from (meta.building keeps its "
              "storeys, height and, placed by a city, its catalogue entry and zone)",
    "roof": "a pitched roof's style and rise, as the building generator was asked for them",
    "scatter": "the scatter placement, index and seed a copy came from (scatter and terrain tests read it)",
    "door_swings": "a room's door leaf swing arcs, which the furnishing solver, light switches and layout QA "
                   "keep clear; the door's interaction is what moves it at runtime",
    "tree": "the tree generator's style, seed and branch count",
    "humanoid": "the body's skeleton, height, pose and seat heights, for posing, clipping QA and tests",
    "preset": "the preset an asset was built with",
    # Traffic and trains: the runtime works from the lane graph and the train's own block
    "parked": "the kerb a city parked a car on (parked cars are static scenery in an export)",
    "crossing": "the level crossing a barrier guards; the manifest's crossings name their barriers",
    "train_car": "the train a car belongs to; the train's 'cars' name their nodes",
}


class ExtrasError(ValueError):
    """A node carries a meta key that is neither declared for export nor build-only."""


@lru_cache(maxsize=1)
def schema() -> dict:
    """The extras schema (``docs/schema/geogen-extras.v1.schema.json``)."""
    return json.loads(SCHEMA_PATH.read_text())


@lru_cache(maxsize=1)
def declared_keys() -> frozenset[str]:
    """Every key the schema allows under ``extras.geogen``."""
    return frozenset(schema()["properties"]["geogen"]["properties"])


def _path(node: SceneNode) -> str:
    names = []
    current: SceneNode | None = node
    while current is not None:
        names.append(current.name)
        current = current.parent
    return "/".join(reversed(names))


def exported_meta(node: SceneNode) -> dict:
    """The part of ``node.meta`` that is exported: its schema-declared keys.

    Build-only keys are left out; an unknown key raises ``ExtrasError``.
    """
    declared = declared_keys()
    out = {}
    for key, value in node.meta.items():
        if key in declared:
            out[key] = value.tolist() if hasattr(value, "tolist") else value
        elif key not in BUILD_ONLY:
            raise ExtrasError(
                f"node '{_path(node)}': meta key '{key}' is not part of the extras contract. Declare it in "
                f"{SCHEMA} if a runtime reads it, or add it to geogen.extras.BUILD_ONLY if it stays in the generator.")
    return out


def _plain(value):
    """numpy scalars and arrays as JSON (what the glTF writer does with them)."""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON")


def scene_problems(root: SceneNode, package: str = "scene") -> list[str]:
    """What exporting ``root`` would get wrong: meta keys outside the contract, and node
    extras the schema rejects. One line per distinct problem (the first node it shows on).

    Checks every node's ``export.exported_extras`` (IDs assigned as an export in
    ``package`` would) without building any geometry, so it is cheap on big scenes;
    colliders, LOD levels and light children are the exporter's own and are covered
    by ``tests/test_export.py``. Needs ``jsonschema``.
    """
    import jsonschema

    from . import refs
    from .export import exported_extras

    validator = jsonschema.Draft202012Validator(schema())
    refs.assign_ids(root, package)
    seen: set[str] = set()
    problems: dict[tuple, str] = {}
    try:
        for node in root.iter_nodes():
            try:
                extras = exported_extras(node)
            except ExtrasError as exc:
                problems.setdefault(("key", str(exc).split("': meta key ", 1)[-1]), str(exc))
                continue
            if not extras:
                continue
            extras = json.loads(json.dumps(extras, default=_plain))
            key = json.dumps({**extras["geogen"], "uid": "x:" if "uid" in extras["geogen"] else None}, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            for error in validator.iter_errors(extras):
                where = "/".join(str(p) for p in error.absolute_path if not isinstance(p, int))
                message = error.message if len(error.message) < 300 else error.message[:300] + "..."
                problems.setdefault((where, error.validator),
                                    f"node '{_path(node)}': extras.{where}: {message}")
    finally:
        for node in root.iter_nodes():
            node.meta.pop("uid", None)
    return list(problems.values())
