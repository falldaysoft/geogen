"""Scene composer for assembling multi-asset scenes from YAML definitions."""

import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import roots
from ..core.node import SceneNode
from ..core.transform import Transform
from .loader import LayoutLoader
from .anchors import resolve_anchor
from .attachments import parse_attachment
from .expressions import enabled
from .validation import suggest, validate_scene_yaml, pre_validate_scene_references
from .yaml_utils import safe_load, safe_load_path

logger = logging.getLogger("geogen.layout")



def _set_states(node: SceneNode, spec: Any, where: str) -> None:
    """Placement ``state: open`` (every interaction that has that state) or
    ``state: {swing: open}``: the object starts, and is exported, in that state.
    Names of variables the asset declares (``state: {lit: true}``) set their starting value."""
    from ..state import instance_state, set_values
    from .interactions import apply_state

    wanted = spec if isinstance(spec, dict) else None
    declared = instance_state(node)
    if wanted is not None and any(k in declared for k in wanted):
        set_values(node, {k: v for k, v in wanted.items() if k in declared}, f"state of {where}")
        wanted = {k: v for k, v in wanted.items() if k not in declared}
        if not wanted:
            return
    found = False
    for owner in node.iter_nodes():
        for interaction in owner.interactions:
            state = wanted.get(interaction.name) if wanted is not None else str(spec)
            if state is None or (wanted is None and state not in interaction.states):
                continue
            if state not in interaction.states:
                raise ValueError(f"state of {where}: interaction '{interaction.name}' has no state '{state}' "
                                 f"(states: {sorted(interaction.states)})")
            interaction.initial = state
            apply_state(owner, interaction, state)
            found = True
    if not found:
        names = sorted({i.name for n in node.iter_nodes() for i in n.interactions})
        raise ValueError(f"state of {where}: nothing takes {spec!r} (interactions: {names}"
                         + (f", state: {sorted(declared)})" if declared else ")"))



@dataclass
class _SceneBuild:
    """What the phases of one scene's build share."""

    root: SceneNode
    data: dict[str, Any]
    source: Path | None
    size: np.ndarray
    slots: dict[str, Transform]
    placements: dict[str, Any]      # the plain ones (no scatter, traffic or train), after `when:`
    scatters: dict[str, Any]
    fleets: dict[str, Any]
    refs: Any                       # refs.SceneRefs
    loaded: dict[str, SceneNode] = field(default_factory=dict)   # placement name -> its node


class SceneComposer:
    """Composes scenes by loading and connecting multiple assets.

    The composer reads a composition YAML file that specifies which assets
    to load and how to position them. Supports three positioning modes:

    1. **Slots**: Named positions defined in the scene with semantic meaning.
       Objects are placed at slots by name.

    2. **Attachments**: Objects attached to other objects' attachment points
       (e.g., chairs attached to table's seat points).

    3. **Nested scenes**: Composed scenes can reference other composed scenes,
       enabling hierarchical composition (street -> house -> dining_set).

    YAML format:
        name: scene_name
        size: [width, height, depth]  # Optional, for slot positioning

        # Define named slots for semantic positioning
        slots:
          slot_name:
            anchor: bottom_center      # Base anchor within scene bounds
            offset: [x, y, z]          # Offset in scene units (optional)
            facing: center|outward|north|south|east|west  # Direction (optional)

        # Place objects into the scene
        place:
          object_name:
            asset: path/to/asset.yaml  # Primitive asset
            # OR
            scene: path/to/scene.yaml  # Composed scene (nested)

            slot: slot_name            # Place at a named slot
            # OR
            attach_to: other_object    # Attach to another object
            at: attachment_point       # At this attachment point (or list of points)

    Example - Street scene:
        name: street
        size: [20, 0, 10]

        slots:
          sidewalk_lamp:
            anchor: bottom_front_left
            offset: [0.1, 0, 0.1]
          sidewalk_bench:
            anchor: bottom_front_center
            offset: [0, 0, 0.1]
            facing: south
          plot_left:
            anchor: bottom_back_left
            offset: [0.25, 0, 0.25]
          plot_right:
            anchor: bottom_back_right
            offset: [-0.25, 0, 0.25]

        place:
          lamp:
            asset: street_lamp.yaml
            slot: sidewalk_lamp

          bench:
            asset: bench.yaml
            slot: sidewalk_bench

          house_left:
            scene: house.yaml
            slot: plot_left

          house_right:
            scene: house.yaml
            slot: plot_right
    """

    def __init__(self, assets_dir: str | Path | None = None, cache_dir: str | Path | None = None) -> None:
        """Initialize the composer.

        Args:
            assets_dir: Base directory for asset files. Defaults to 'assets/' relative to project.
            cache_dir: Optional on-disk cache for placed assets/scenes (see ``PrototypeCache``);
                defaults to ``$GEOGEN_CACHE`` when set.
        """
        import os

        self._loader = LayoutLoader()
        self._prototypes: dict[str, SceneNode] = {}
        self._depth = 0                 # nesting of scenes being built (1: the top one)
        cache_dir = cache_dir or os.environ.get("GEOGEN_CACHE")
        self._disk_cache = None
        if cache_dir:
            from .cache import PrototypeCache

            self._disk_cache = PrototypeCache(cache_dir)
        self._assets_dir = Path(assets_dir) if assets_dir is not None else roots.CORE

    def compose(self, path: str | Path, params: dict[str, float] | None = None) -> SceneNode:
        """Load and compose a scene from a YAML file.

        Args:
            path: Path to the composition YAML file
            params: Overrides for the scene's declared ``params:`` (scenes
                use ``{expr}`` interpolation like assets)

        Returns:
            Root SceneNode of the composed scene
        """
        path = Path(path)
        logger.debug("Composing scene: %s", path)
        data = safe_load_path(path)

        # Validate YAML structure
        for warning in validate_scene_yaml(data):
            warnings.warn(warning, stacklevel=2)

        # Pre-validate references
        # attach_to targets are resolved (with file:line) when composing: see geogen.refs.
        ref_errors = pre_validate_scene_references(data, attach=False)
        if ref_errors:
            raise ValueError(
                f"Invalid references in {path.name}:\n  " + "\n  ".join(ref_errors)
            )

        # A scene folder (scenes/<name>/scene.yaml): its own files come first while it is composed.
        folder = roots.scene_folder(path) if path.name == roots.SCENE_FILE else None
        if folder is None:
            return self._build_scene(self._resolve_params(data, params), source=path)
        with roots.scene_scope(folder):
            return self._build_scene(self._resolve_params(data, params), source=path)

    @staticmethod
    def _resolve_params(data: dict[str, Any], overrides: dict[str, float] | None) -> dict[str, Any]:
        """Resolve a scene's ``params:``, presets, derived values and ``{expr}`` values (like LayoutLoader)."""
        from .expressions import resolve_document

        if not ({"params", "presets", "derived"} & set(data)) and not overrides:
            return data
        return resolve_document(data, overrides)[0]

    def compose_string(self, yaml_string: str) -> SceneNode:
        """Compose a scene from a YAML string.

        Args:
            yaml_string: YAML content as a string

        Returns:
            Root SceneNode of the composed scene
        """
        data = safe_load(yaml_string)
        return self._build_scene(self._resolve_params(data, None))

    def _build_scene(self, data: dict[str, Any], source: Path | None = None) -> SceneNode:
        """Build scene from parsed YAML data (``source``: its file, for diagnostics), then take
        parts that `move` to other placed things' attachments as far as this scene can see."""
        from ..physics import resolve_spawns
        from .interactions import resolve_moves

        self._depth += 1
        try:
            root = self._build_scene_here(data, source)
            resolve_moves(root, final=self._depth == 1)
            resolve_spawns(root, final=self._depth == 1)
            from ..flows import attach_templates

            attach_templates(root, self._loader, self._assets_dir)
            return root
        finally:
            self._depth -= 1

    def _build_scene_here(self, data: dict[str, Any], source: Path | None = None) -> SceneNode:
        name = data.get("name", "composed_scene")
        root = SceneNode(name)
        # Scene size for slot positioning, propagated to the root node.
        size = np.array(data.get("size", [1.0, 1.0, 1.0]), dtype=np.float64)
        root.size = size
        logger.debug("Building scene: %s (size=%s)", name, size)

        placements = dict(data.get("place") or {})
        # `when: "{expr}"` drops a placement when it resolves to 0 / false / "" (like parts).
        placements = {k: {kk: vv for kk, vv in v.items() if kk != "when"} if isinstance(v, dict) else v
                      for k, v in placements.items()
                      if not (isinstance(v, dict) and not enabled(v.get("when", True)))}
        if source is not None:      # where each travel was written, for export diagnostics
            from .yaml_utils import source_of

            placements = {
                k: ({**v, "_travel_where": source_of(source, ["place", k, "travel"])}
                    if isinstance(v, dict) and v.get("travel") else v)
                for k, v in placements.items()}
        # Scatter placements run once the objects they avoid exist; traffic and trains once
        # the streets and railways do.
        scatters = {k: v for k, v in placements.items() if isinstance(v, dict) and "scatter" in v}
        fleets = {k: v for k, v in placements.items() if isinstance(v, dict) and ("traffic" in v or "train" in v)}
        placements = {k: v for k, v in placements.items() if k not in scatters and k not in fleets}
        from ..refs import SceneRefs

        build = _SceneBuild(root, data, source, size, self._parse_slots(data.get("slots", {}), size), placements,
                            scatters, fleets, SceneRefs(root, source, [*placements, *scatters, *fleets]))
        for phase in self._PHASES:
            getattr(self, phase)(build)
        return root

    # The order matters: each phase may name what earlier ones placed.
    _PHASES = ("_declare", "_place_free", "_place_dependent", "_build_city", "_population", "_scatter", "_placement_extras",
               "_build_tracks", "_build_routes", "_build_railways", "_add_spawns", "_build_places", "_place_fleets",
               "_add_attachments")

    def _declare(self, b: _SceneBuild) -> None:
        """Scene-level declarations: state, wiring, HUD, physics ticks, quests and dialogues."""
        from ..hud import declare as declare_hud
        from ..state import declare_scene
        from ..wiring import declare as declare_wiring

        root, data, source = b.root, b.data, b.source
        declare_scene(root, data, source)
        declare_wiring(root, data, source, asset=False)
        declare_hud(root, data, source)
        if data.get("physics") is not None:
            physics = data["physics"]
            ticks = physics.get("ticks") if isinstance(physics, dict) else None
            if set(physics if isinstance(physics, dict) else ["?"]) - {"ticks"} or not isinstance(ticks, int) \
                    or not 30 <= ticks <= 480:
                raise ValueError(f"{source}: physics is {{ticks: 30..480}}, got {physics!r}")
            root.meta["physics_ticks"] = ticks
        if data.get("quests"):
            from ..quests import declare_scene as declare_quests

            declare_quests(root, data, source, self._assets_dir)
        if data.get("dialogues"):
            from ..dialogue import declare_scene as declare_dialogues

            declare_dialogues(root, data, source, self._assets_dir)

    def _place_free(self, b: _SceneBuild) -> None:
        """Placements that don't depend on others: at a slot, on one of the scene's own surfaces, or
        at the origin. (Object-to-object surfaces and ``attach_to`` wait for ``_place_dependent``.)"""
        for obj_name, obj_def in b.placements.items():
            if "attach_to" in obj_def:
                continue
            if "on" in obj_def and self._surface_target_is_object(obj_def["on"]):
                continue
            node = self._load_object(obj_def)
            node.name = obj_name
            if "slot" in obj_def:
                slot_name = obj_def["slot"]
                if slot_name not in b.slots:
                    available = sorted(b.slots.keys())
                    raise ValueError(f"Slot '{slot_name}' not defined.{suggest(slot_name, available)}"
                                     f" Available slots: {available}")
                logger.debug("Placing '%s' at slot '%s'", obj_name, slot_name)
                node.transform = b.slots[slot_name]
            elif "on" in obj_def:
                node.transform = self._resolve_surface_placement(obj_def, b.root, b.refs, ["place", obj_name])
            b.root.add_child(node)
            b.loaded[obj_name] = node

    def _place_dependent(self, b: _SceneBuild) -> None:
        """Placements on another object's surface or attachment points, each after the one it names."""
        root, refs = b.root, b.refs
        for obj_name in self._after_targets(b.placements, refs):
            obj_def = b.placements[obj_name]
            keys = ["place", obj_name]
            if "on" in obj_def and self._surface_target_is_object(obj_def["on"]):
                node = self._load_object(obj_def)
                node.name = obj_name
                node.transform = self._resolve_surface_placement(obj_def, root, refs, keys)
                root.add_child(node)
                b.loaded[obj_name] = node
                self._cut_host(node, obj_name, obj_def["on"], refs.surface(obj_def["on"], [*keys, "on"])[1])
                continue
            if "attach_to" not in obj_def:
                continue
            target_name = obj_def["attach_to"]
            target = refs.node(target_name, [*keys, "attach_to"])
            attachment_names = obj_def.get("at", [])
            if isinstance(attachment_names, str):
                attachment_names = [attachment_names]
            for i, attach_name in enumerate(attachment_names):
                node = self._load_object(obj_def)
                node.name = f"{obj_name}_{i}" if len(attachment_names) > 1 else obj_name
                attach_transform = target.get_attachment(attach_name)
                if attach_transform is None:
                    available = target.list_attachments()
                    raise refs.fail(
                        f"Attachment point '{attach_name}' not found on '{target_name}'.{suggest(attach_name, available)}"
                        f" Available: {available}", [*keys, "at"], str(attach_name))
                logger.debug("Attaching '%s' to '%s' at '%s'", obj_name, target_name, attach_name)
                node.transform = attach_transform
                root.add_child(node)
                b.loaded.setdefault(obj_name, node)

    def _build_city(self, b: _SceneBuild) -> None:
        if "city" in b.data:
            from .city import build_city

            build_city(b.root, b.data["city"], self._load_object, self._loader.material_loader)
            b.loaded["city"] = b.root.children[-1]

    def _population(self, b: _SceneBuild) -> None:
        """The scene's ``population:`` (geogen.npc.population_placements): NPC crowds on the
        places it names, added to the scatters."""
        if not b.data.get("population"):
            return
        from ..npc import population_placements

        crowds = population_placements(b.data["population"], b.refs, str(b.data.get("name", "scene")), ["population"])
        for name, placement in crowds.items():
            if name in b.placements or name in b.scatters:
                raise b.refs.fail(f"population: '{name}' is also a placement", ["population"])
            b.scatters[name] = placement
            b.refs.pending.add(name)

    def _scatter(self, b: _SceneBuild) -> None:
        if not b.scatters:
            return
        from .scatter import scatter

        occupied: list = []
        for obj_name, obj_def in b.scatters.items():
            keys = ["place", obj_name]
            placed = scatter(b.root, obj_name, obj_def, b.refs,
                             lambda d: self._load_object({k: v for k, v in d.items() if k != "scatter"}), occupied,
                             keys=[*keys, "scatter"])
            if "npc" in obj_def:          # a crowd: every copy shares the placement's home
                from ..npc import resolve_home

                for node in placed:
                    resolve_home(node, obj_def.get("home"), b.refs, [*keys, "home"])

    def _placement_extras(self, b: _SceneBuild) -> None:
        """Per-placement keys on the placed copy: pickup and readable (as the asset keys), an NPC's
        home, seed and quests, and extra tags. Then the scene root's own tags."""
        source = b.source
        for obj_name, obj_def in b.placements.items():
            node = b.loaded.get(obj_name)
            if node is None or not isinstance(obj_def, dict):
                continue
            where = f"{source}: place.{obj_name}"
            if obj_def.get("pickup"):
                from ..inventory import parse_pickup

                parse_pickup(obj_def["pickup"], node, where, self._assets_dir, source)
            if obj_def.get("readable"):
                from ..dialogue import parse_readable

                parse_readable(obj_def["readable"], node, where)
            if "npc" in obj_def:
                from ..npc import resolve_home

                resolve_home(node, obj_def.get("home"), b.refs, ["place", obj_name, "home"])
                if any(k in obj_def for k in ("owns", "prefers", "work")):
                    from ..npc import resolve_ownership

                    resolve_ownership(node, obj_def, b.refs, ["place", obj_name])
                if "seed" in obj_def:
                    node.meta["npc"]["seed"] = int(obj_def["seed"])  # type: ignore[index]
                if obj_def.get("quests"):         # quests this NPC holds from the start (quests.collect)
                    from ..quests import declare_scene as declare_quests

                    declare_quests(node, {"quests": obj_def["quests"]}, source, self._assets_dir)
            if obj_def.get("tags"):           # extra semantic tags, e.g. tags: [door.interior]
                node.tags = [*node.tags, *[t for t in obj_def["tags"] if t not in node.tags]]
        b.root.tags = list(b.data.get("tags", []))

    def _build_tracks(self, b: _SceneBuild) -> None:
        """Tracks spawned items follow (geogen.flows), in the scene frame."""
        if not b.data.get("tracks"):
            return
        from ..flows import scene_tracks

        for track_node in scene_tracks(b.data["tracks"], self._loader.material_loader):
            if track_node.name in b.loaded:
                raise ValueError(f"tracks.{track_node.name}: a placement already has that name")
            b.root.add_child(track_node)
            b.loaded[track_node.name] = track_node

    def _build_routes(self, b: _SceneBuild) -> None:
        """Explicit traffic routes (roads, bus loops) in the scene frame: a lane graph."""
        if not b.data.get("routes"):
            return
        from ..traffic import route_surface, routes_graph

        lanes = routes_graph(b.data["routes"])
        if lanes is not None:
            b.root.meta["traffic"] = lanes
        for route_name, route_spec in b.data["routes"].items():
            surface = route_surface(route_name, route_spec, self._loader.material_loader)
            if surface is not None:
                b.root.add_child(surface)
                b.loaded[surface.name] = surface

    def _build_railways(self, b: _SceneBuild) -> None:
        """Railways: track, stations, and barriers where they cross the roads built so far."""
        if not b.data.get("railways"):
            return
        from ..railway import build_railway, level_crossings
        from ..traffic import build_traffic

        root = b.root
        roads = build_traffic(root)
        for rail_name, rail_spec in b.data["railways"].items():
            if isinstance(rail_spec, dict) and "around" in rail_spec:
                rail_spec = self._railway_round_city(root, rail_name, rail_spec)
            rail_node = build_railway(rail_name, rail_spec, self._loader.material_loader,
                                      self._load_object, self._assets_dir)
            root.add_child(rail_node)
            b.loaded[rail_node.name] = rail_node
            level_crossings(root, rail_node, roads, self._load_object)

    @staticmethod
    def _railway_round_city(root: SceneNode, rail_name: str, rail_spec: dict[str, Any]) -> dict[str, Any]:
        """A loop round the district: ``{around: {margin, corner}}`` becomes the path."""
        from ..railway import loop_around

        city_node = next((n for n in root.iter_nodes() if isinstance(n.meta.get("city"), dict)), None)
        if city_node is None:
            raise ValueError(f"railway '{rail_name}': around needs a city: block")
        opts = rail_spec["around"] or {}
        return {**{k: v for k, v in rail_spec.items() if k != "around"},
                "path": loop_around(city_node.meta["city"]["extent"], float(opts.get("margin", 20.0)),
                                    float(opts.get("corner", 30.0))), "loop": True}

    def _add_spawns(self, b: _SceneBuild) -> None:
        """Spawn points: slot-style positions exported as empty nodes."""
        for spawn_name, transform in self._parse_slots(b.data.get("spawns", {}), b.size).items():
            b.root.add_child(SceneNode(name=spawn_name, transform=transform, tags=["spawn"],
                                       meta={"type": "spawn"}))

    def _build_places(self, b: _SceneBuild) -> None:
        """Named places: trigger volumes, positioned like slots or round something placed."""
        if not b.data.get("places"):
            return
        from ..places import build_places

        def place_position(place_name: str, entry: dict[str, Any]) -> np.ndarray:
            return (resolve_anchor(entry.get("anchor", "center"), b.size)
                    + np.array(entry.get("offset", [0, 0, 0]), dtype=np.float64))

        build_places(b.root, b.data["places"], str(b.source) if b.source is not None else b.root.name,
                     locate=place_position,
                     resolve=lambda place_name, ref: b.refs.node(ref, ["places", place_name, "around"]))

    def _place_fleets(self, b: _SceneBuild) -> None:
        """Traffic and trains last: they need the lanes and railways, and keep clear of the spawns."""
        if not b.fleets:
            return
        from ..traffic import build_traffic, place_traffic
        from ..trains import place_train

        graph = build_traffic(b.root)
        for obj_name, obj_def in b.fleets.items():
            if "train" in obj_def:
                node = place_train(b.root, obj_name, obj_def, self._load_object, self._assets_dir)
            else:
                node = place_traffic(b.root, obj_name, obj_def, self._load_object, graph, self._assets_dir)
            b.root.add_child(node)
            b.loaded[obj_name] = node

    def _add_attachments(self, b: _SceneBuild) -> None:
        """The scene's own attachment points, for parent scenes."""
        for attach_name, attach_def in b.data.get("attachments", {}).items():
            b.root.attachments[attach_name] = parse_attachment(attach_name, attach_def)

    def _load_object(self, obj_def: dict[str, Any]) -> SceneNode:
        """Load an object from asset or scene definition (instances of one prototype per definition)."""
        import json

        # The seed picks the person for archetypes and NPCs (their bodies may be archetypes).
        keys = ("asset", "scene", "recipe", "npc", "archetype", "params", "furnish")
        if "archetype" in obj_def or "npc" in obj_def:
            keys = (*keys, "seed")
        # (in a scene folder a name may be one of its private files)
        key = json.dumps({**{k: obj_def.get(k) for k in keys}, **({"in": str(roots.scope())} if roots.scope() else {})},
                         sort_keys=True,
                         default=str)
        prototype = self._prototypes.get(key)
        if prototype is None and self._disk_cache is not None:
            prototype = self._disk_cache.get(key)
        if prototype is None:
            prototype = self._load_prototype(obj_def)
            if self._disk_cache is not None:
                self._disk_cache.put(key, prototype)
        self._prototypes[key] = prototype
        node = prototype.instance()
        if "source" not in node.meta:
            node.meta["scope"] = True       # a placed scene / recipe / NPC: its own namespace (geogen.refs)
        if obj_def.get("travel"):
            from ..travel import apply_travel

            apply_travel(node, obj_def["travel"],
                         obj_def.get("_travel_where") or f"travel of {obj_def.get('asset') or obj_def.get('scene')}")
        if obj_def.get("state"):
            _set_states(node, obj_def["state"], obj_def.get("asset") or obj_def.get("scene") or "object")
        return node

    def _load_prototype(self, obj_def: dict[str, Any]) -> SceneNode:
        if "recipe" in obj_def:
            from .recipes import build_recipe

            return build_recipe(obj_def["recipe"], obj_def.get("params") or {}, self._loader)
        if "npc" in obj_def:
            from ..npc import build_npc, load_definition

            return build_npc(load_definition(roots.find(obj_def["npc"], self._assets_dir)), self._loader, self._assets_dir,
                             seed=int(obj_def.get("seed", 0)))
        if "archetype" in obj_def:
            from ..characters import load_character

            return load_character(self._loader, self._assets_dir, obj_def["archetype"], int(obj_def.get("seed", 0)),
                                  obj_def.get("params"))
        if "asset" in obj_def:
            asset_path = roots.find(obj_def["asset"], self._assets_dir)
            node = self._loader.load(asset_path, params=obj_def.get("params"))
            if obj_def.get("furnish"):
                from .furnish import furnish_plan

                seed = obj_def["furnish"] if isinstance(obj_def["furnish"], int) and obj_def["furnish"] is not True else 0
                furnish_plan(node, self._assets_dir, seed=seed, loader=self._loader)
            return node
        elif "scene" in obj_def:
            scene_path = roots.find(obj_def["scene"], self._assets_dir)
            return self.compose(scene_path, params=obj_def.get("params"))
        else:
            raise ValueError("Object must have 'asset', 'scene', 'recipe', 'npc' or 'archetype' specified")

    @staticmethod
    def _cut_host(node: SceneNode, obj_name: str, spec: str, surface: Any) -> None:
        """Subtract ``node``'s host cutters from the part that owns surface ``spec``
        (and any parts the surface export lists under ``cut:``), then line the
        opening's reveals if the surface declares a ``reveal`` finish."""
        if not node.host_cutters:
            return
        from ..core import csg

        host = surface.source if surface is not None else None
        if host is None or host.mesh is None:
            logger.warning("'%s' has openings but surface '%s' has no source part to cut", obj_name, spec)
            return
        parts = [p for p in [host, *surface.also_cut] if p.mesh is not None]
        uncut = {id(p): p.mesh for p in parts}
        for part in parts:
            to_part = np.linalg.inv(part.world_transform()) @ node.world_transform()
            cutters = [m.transform(to_part) for m in node.host_cutters]
            try:
                part.mesh = csg.difference(part.mesh, *cutters)
            except csg.CSGError as exc:
                raise ValueError(
                    f"Cutting opening for '{obj_name}' into '{spec}' ({part.name}) failed: {exc}"
                ) from exc

        if surface.reveal and node.host_reveals and host.parent is not None:
            SceneComposer._line_reveals(node, obj_name, surface.reveal, parts, uncut, host.parent)

    @staticmethod
    def _line_reveals(
        node: SceneNode,
        obj_name: str,
        reveal: dict[str, Any],
        parts: list[SceneNode],
        uncut: dict[int, Any],
        owner: SceneNode,
    ) -> None:
        """Add a ``<obj>_reveal`` part to ``owner`` lining the opening's sides.

        Each reveal box (in the opening's frame, running from behind its frame
        into the room) becomes a sleeve ``thickness`` thick around the opening,
        clipped to the material that was removed from the wall and lining.
        """
        from ..core import csg, uvmap
        from ..core.mesh import Mesh
        from ..generators.primitives import CubeGenerator

        def box(lo: np.ndarray, hi: np.ndarray) -> Mesh:
            size = hi - lo
            mesh = CubeGenerator(size_x=size[0], size_y=size[1], size_z=size[2], bevel=0).generate()
            move = np.eye(4)
            move[:3, 3] = (lo + hi) / 2
            return mesh.transform(move)

        t = reveal["thickness"]
        to_owner = np.linalg.inv(owner.world_transform())
        pieces = []
        for region, line_bottom in node.host_reveals:
            lo, hi = region.vertices.min(axis=0), region.vertices.max(axis=0)
            inner_lo = lo + np.array([t, t if line_bottom else -0.1, -0.1])
            inner_hi = hi - np.array([t, t, -0.1])
            sleeve = csg.difference(box(lo, hi), box(inner_lo, inner_hi))
            sleeve = sleeve.transform(to_owner @ node.world_transform())
            for part in parts:
                solid = uncut[id(part)].transform(to_owner @ part.world_transform())
                piece = csg.intersection(sleeve, solid)
                if len(piece.faces):
                    pieces.append(piece)
        if not pieces:
            return
        mesh = uvmap.box_project(csg.union(*pieces))
        mesh.material = reveal["material"]
        owner.add_child(SceneNode(name=f"{obj_name}_reveal", mesh=mesh))

    def _surface_target_is_object(self, spec: str) -> bool:
        """Return True when `on: <spec>` refers to `<object>.<surface>`."""
        return "." in spec

    def _after_targets(self, placements: dict[str, Any], refs: Any) -> list[str]:
        """Placement keys ordered so each comes after the placement its ``on:`` /
        ``attach_to:`` names (declaration order otherwise)."""
        from ..refs import RefError, parse

        def target(obj_def: Any) -> str | None:
            text = obj_def.get("attach_to") if "attach_to" in obj_def else obj_def.get("on")
            if not isinstance(text, str) or ("attach_to" not in obj_def and "." not in text):
                return None
            try:
                steps = parse(text, member="attach_to" not in obj_def).steps
            except RefError:
                return None                 # reported, with its source, when it is resolved
            return steps[0].name if steps else None

        order: list[str] = []
        state: dict[str, int] = {}

        def visit(name: str, chain: list[str]) -> None:
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                loop = " -> ".join([*chain[chain.index(name):], name])
                raise refs.fail(f"placements refer to each other in a circle: {loop}", ["place", name], name)
            state[name] = 1
            dep = target(placements[name])
            if dep in placements and dep != name:
                visit(dep, [*chain, name])
            state[name] = 2
            order.append(name)

        for name in placements:
            visit(name, [])
        return order

    def _resolve_surface_placement(
        self,
        obj_def: dict[str, Any],
        root: SceneNode,
        refs: Any,
        keys: list[str],
    ) -> Transform:
        """Resolve an `on: <surface>` + `at: {u,v,depth}` placement to a Transform.

        Forms:
          on: floor                 # root-level surface
          on: house.floor           # surface on a placed object
          on: plot/house.floor      # ... on something inside a placed scene (see geogen.refs)

        Objects face out of the surface (+Z along its normal); on horizontal
        surfaces that is +Z (north). ``facing:`` (north/south/east/west/
        center/outward) replaces the heading, ``yaw:`` (degrees) turns further.
        """
        spec = obj_def["on"]
        at = obj_def.get("at", {})
        if not isinstance(at, dict):
            raise ValueError(
                f"`at:` for surface placement must be a mapping (u/v/depth), got {type(at).__name__}"
            )
        u = at.get("u", 0.5)
        v = at.get("v", 0.5)
        depth = at.get("depth", 0.0)

        if "." in spec:
            target, surface = refs.surface(spec, [*keys, "on"])
            local = surface.resolve(u=u, v=v, depth=depth).to_matrix()
            return self._orient(Transform.from_matrix(target.world_transform() @ local), obj_def)

        # Root-level surface
        transform = root.get_surface(spec, u=u, v=v, depth=depth)
        if transform is None:
            available = root.list_surfaces()
            raise refs.fail(f"Surface '{spec}' not defined on scene root. Available: {available}",
                            [*keys, "on"], spec)
        return self._orient(transform, obj_def)

    def _orient(self, transform: Transform, obj_def: dict[str, Any]) -> Transform:
        """Apply a placement's ``facing:`` / ``yaw:`` to a surface transform."""
        if "facing" in obj_def:
            transform.rotation[1] = self._facing_to_rotation(obj_def["facing"], transform.translation)
        if "yaw" in obj_def:
            transform.rotation[1] += np.radians(float(obj_def["yaw"]))
        return transform

    def _parse_slots(
        self, slots_data: dict[str, Any], size: np.ndarray
    ) -> dict[str, Transform]:
        """Parse slot definitions into transforms.

        Slots can specify position in two ways:
        1. anchor + offset: offset is in ABSOLUTE units (meters), not fractions
        2. position: direct [x, y, z] coordinates relative to scene center
        """
        slots = {}

        for slot_name, slot_def in slots_data.items():
            facing = slot_def.get("facing", "north")

            if "position" in slot_def:
                # Direct position in local coordinates (meters from center)
                position = np.array(slot_def["position"], dtype=np.float64)
            else:
                # Anchor-based positioning
                anchor = slot_def.get("anchor", "center")
                offset = np.array(slot_def.get("offset", [0, 0, 0]), dtype=np.float64)

                # Resolve anchor to position
                position = resolve_anchor(anchor, size)
                # Apply offset in ABSOLUTE units (meters), not fractions
                position += offset

            # Compute rotation from facing direction
            rotation = self._facing_to_rotation(facing, position)

            slots[slot_name] = Transform(
                translation=position,
                rotation=np.array([0.0, rotation, 0.0], dtype=np.float64),
            )

        return slots

    def _facing_to_rotation(self, facing: str, position: np.ndarray) -> float:
        """Convert facing direction to Y-axis rotation in radians."""
        if facing == "center":
            # Face toward origin
            dx = -position[0]
            dz = -position[2]
            return float(np.arctan2(dx, dz))
        elif facing == "outward":
            # Face away from origin
            dx = position[0]
            dz = position[2]
            return float(np.arctan2(dx, dz))
        elif facing == "north":
            return 0.0
        elif facing == "south":
            return float(np.pi)
        elif facing == "east":
            return float(np.pi / 2)
        elif facing == "west":
            return float(-np.pi / 2)
        else:
            return 0.0
