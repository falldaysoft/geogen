# Layout: assets, scenes, placement and QA

How assets and scenes are written and composed (`src/geogen/layout/`): YAML formats, attachments, slots, surfaces, params and expressions, cities, scatter, references, places, projects, the registry, furnishing and the layout / affordance QA.

## Layout System (`src/geogen/layout/`)

- **Anchor** (`anchors.py`): Named anchor points using normalized coordinates (0-1). Examples: `bottom_center`, `top_front_left`. `resolve_anchor()` converts to world coordinates.

- **AttachmentPoint** (`attachments.py`): Named points for connecting objects. Specifies position via anchor + offset, and orientation via `facing` direction (`center`, `outward`, compass directions) or explicit rotation.

- **LayoutLoader** (`loader.py`): Loads composite objects from YAML. Profile primitives:
  ```yaml
  top:
    primitive: extrude
    shape: { rect: ["{width}", "{depth}"], radius: 0.05 }   # or {outer: ..., holes: [...]}
    fit: none          # profile in metres (default 'stretch' scales it to the part size)
    axis: y            # y: profile in XZ (slabs), z: profile in XY (panels), x
    bevel: 0.008
  leg:
    primitive: lathe
    profile: { spline: [[0.6, 0], [0.4, 0.5], [0.7, 1.0]] }  # (r, y); or [[r, y], ...] / {segments: [...]}
    segments: 40
  ```
  Base format:
  ```yaml
  name: object_name
  origin: bottom_center
  size: [x, y, z]
  parts:
    part_name:
      primitive: cube|cylinder|sphere|cone
      size: [x, y, z]
      anchor: bottom_center
      offset: [x, y, z]
      material: wood
  attachments:
    attach_name:
      anchor: bottom_front_center
      offset: [x, y, z]
      facing: center
  ```

- **SceneComposer** (`composer.py`): Composes scenes from YAML using semantic positioning. Supports:
  - **Slots**: Named positions with anchor + offset + facing direction
  - **Attachments**: Objects attached to other objects' attachment points
  - **Nested scenes**: Scenes can reference other composed scenes via `scene:` instead of `asset:`

  Scene format with slots:
  ```yaml
  name: street
  size: [5, 2, 3]

  slots:
    sidewalk_left:
      anchor: bottom_front_left
      offset: [0.1, 0, -0.1]
      facing: south
    sidewalk_center:
      anchor: bottom_front_center
      facing: south

  place:
    lamp:
      asset: street_lamp.yaml
      slot: sidewalk_left
    bench:
      asset: bench.yaml
      slot: sidewalk_center
    furniture:
      scene: scenes/dining_set.yaml  # Nested scene
      slot: room_center
  ```

  Placements take `when: "{expr}"` (dropped when 0 / false / ""), like parts.

  Parts take `array: {count, step: [x, y, z]}` to repeat as `<part>_0..n-1` (step in `offset` units; `count` may be an expression, e.g. `"{round(length / 3) + 1}"`); a `subtract:` naming an arrayed cutter takes every copy. Paths (`paths`, `fence`) accept `{circle: r, segments, center}`.

  Attachment-based composition:
  ```yaml
  name: dining_set
  place:
    table:
      asset: table.yaml
    chairs:
      asset: chair.yaml
      attach_to: table
      at: [seat_front, seat_back, seat_left, seat_right]
  ```

## Hierarchical Layout System - Semantic Connections

The layout system uses **semantic attachment points** instead of raw coordinates. This makes objects self-describing and composable.

## Two Levels of Composition

1. **LayoutLoader** (`loader.py`): Builds individual assets from primitives
2. **SceneComposer** (`composer.py`): Assembles assets into larger scenes

## Asset-Level: Part Attachments (`attach_to`, `at`, `from`)

Within a single asset, primitives connect to each other using named attachment points. Each primitive automatically gets standard attachment points based on its shape:

**Auto-generated attachment points per primitive:**
- Cylinders/Cones: `top`, `bottom`, `left`, `right`, `front`, `back` (radial at mid-height)
- Spheres: `top`, `bottom`, `left`, `right`, `front`, `back` (at surface)
- Cubes: `top`, `bottom`, `left`, `right`, `front`, `back` (face centers)

**Fire hydrant example** - shows hierarchical part-to-part attachment:
```yaml
parts:
  base:
    primitive: cylinder
    size: [0.9, 0.0625, 0.9]
    anchor: bottom_center       # Root part uses coordinate anchor

  barrel:
    primitive: cylinder
    size: [0.7, 0.375, 0.7]
    attach_to: base             # Connect to another part
    at: top                     # Parent's attachment point
    from: bottom                # Child's attachment point to align

  bulge:
    primitive: sphere
    attach_to: barrel
    at: top
    from: bottom

  left_outlet:
    primitive: cylinder
    attach_to: bulge
    at: left                    # Attach to side of sphere
    from: center                # Center outlet so half penetrates
    rotation: [0, 0, 90]        # Rotate to point outward
```

**Key concepts:**
- `attach_to`: Name of parent part (creates hierarchy)
- `at`: Which attachment point on the parent
- `from`: Which point on the child aligns to parent's point (default: `bottom`)
- `rotation`: Local rotation applied after attachment (degrees)
- Only the root part(s) use `anchor` + `offset` for coordinate positioning

## Asset-Level: Custom Attachment Points

Assets export named attachment points for scene-level composition:

```yaml
# table.yaml - defines where chairs can attach
attachments:
  seat_front:
    anchor: bottom_front_center
    offset: [0, 0, 0.4]         # Push out from table edge
    facing: center              # Chair faces toward table center

  seat_left:
    anchor: left_center
    offset: [-0.4, -0.5, 0]     # Push out and down to floor
    facing: center
```

**Facing directions:** `center` (toward origin), `outward` (away from origin), `north`, `south`, `east`, `west`

## Scene-Level: Slots and Asset Attachments

**Slots** are semantic positions within a scene:
```yaml
slots:
  sidewalk_lamp:
    position: [-4, 0, -10]      # Or use anchor + offset
    facing: east

place:
  lamp:
    asset: street_lamp.yaml
    slot: sidewalk_lamp         # Place at named slot
```

**Asset-to-asset attachment** (for furniture groupings):
```yaml
place:
  table:
    asset: table.yaml           # First object, no positioning needed

  chairs:
    asset: chair.yaml
    attach_to: table            # Connect to the table
    at: [seat_front, seat_back, seat_left, seat_right]  # Multiple instances!
```

## Hierarchy Composition

Scenes can nest other scenes, creating deep hierarchies:

```
street.yaml
├── road.yaml (asset)
├── street_side.yaml (scene)
│   ├── sidewalk.yaml (asset)
│   └── house_plot.yaml (scene)
│       └── house_simple.yaml (asset)
└── street_lamp.yaml (asset)
```

```yaml
place:
  left_side:
    scene: scenes/street_side.yaml    # Nested scene (not asset)
    slot: left_side
```

## City layout

A scene with a top-level `city:` block (`src/geogen/layout/city.py`, e.g. `assets/scenes/town.yaml`) builds a district:
- a street grid (`blocks`, `block_size`, `street_width`, `avenues: {ew: [..], ns: [..]}` + `avenue_width`), one asphalt slab with crosswalks at every block corner and dashed centre lines (`road_paint_white/yellow`);
- raised blocks (`curb_height`): a rounded concrete sidewalk ring plus lot slabs.

Lots are rows split into `lot_width` frontages. Zoning:
- `landmarks` blocks hold one building;
- `parks` blocks get trees;
- lots fronting an avenue are commercial, the rest residential.

Each lot takes a random fitting entry from `buildings: {zone: [{asset|scene, params, weight, setback}]}`. The building's +Z front faces the street. Buildings and furniture are loaded once and instanced (deep copies sharing meshes). `furniture:` lines the curbs (lamp/tree by `spacing`; bench/hydrant/trashcan `per_edge`, optional `zones`). Nodes carry `meta.lot` / `meta.building` / `meta.street_furniture`. Streets and sidewalks are walkable.

Catalogue entries can also be `recipe:` buildings (`layout/recipes.py`), generated to fit each lot:
- `detached_house`: 1–2 storeys, gable roof;
- `shop_row`: shop and stock room with flats above, reached by their own stair;
- `apartment_block`, `office`, `hotel`.

Recipe keys are `storeys` (a number or `[lo, hi]`), `interior`, `style` (one or a list), `max_width`/`max_depth`, `setback`, `side_gap` and `rear_gap`. `interior` is:
- `full`: furnish every storey;
- `lobby`: furnish only the ground floor;
- `shell`: rooms stripped, street doors locked, dark voids behind the windows.

Storeys are capped by what the stair hall can fit (`_cap_storeys`), and a recipe that doesn't fit is retried lower. Fitting uses the wall footprint (nodes tagged `wall`), so canopies may overhang the setback. The front is the side the entrance spawn is on (`recipes.front_of`), so a south-entrance hotel still faces its street. `building:` roofs take `style: gable|hip|shed` (+ `rise`, `overhang`, `gable_material`) as well as flat/parapet. The furnishing solver rejects placements whose own front can't be reached (it used to only protect already-placed items).

`scenes/town.yaml` is milestone M3:
- The district is exported with `--export-godot --stream`.
- The hotel lobby and shop floors are furnished; offices are shells.
- `tests/test_godot_runtime.py::test_m3_*` streams it in Godot, walks the avenue, opens a shop door and the hotel entrance, and checks the player ends up in `shop` / `lobby`.

Manifest spawns are ordered shallowest first, so a scene's own `spawns:` beat nested buildings' `entrance_spawn`s. Layout QA works in each room's parent (storey) frame, so moving or rotating a building never changes its result.

## Design Principles

1. **No magic numbers in connections**: Parts connect via named points (`top`, `bottom`, `seat_front`), not coordinates
2. **Self-describing objects**: Assets declare their own attachment points based on their semantics
3. **Composability**: Objects don't know about their containers; containers know about objects' attachment points
4. **Hierarchical transforms**: Child transforms are relative to parent, enabling grouped movement

## Parameters and expressions

Assets can declare parameters that control their geometry. Loaders resolve parameters before the hierarchy is built, so every part size/offset sees the post-substitution values.

```yaml
name: house
params:
  width:  { default: 8 }
  depth:  { default: 6 }
  height: { default: 4 }

size: ["{width}", "{height}", "{depth}"]

parts:
  shell:
    primitive: cube
    size: ["{1 - 0.05}", 0.6, 1]   # expressions allowed anywhere
    anchor: bottom_center
```

Assets whose container is just a unit (e.g. `size: ["{scale}", "{scale}", "{scale}"]` with part sizes in metres) set `bounds: geometry` so their reported `size` is the real extent. Overrides come in through `LayoutLoader.load(path, params={"width": 12})`, or per placement in scenes: `door: {asset: door.yaml, params: {width: 1.1}, on: ...}`. Scenes can declare `params:` and use `{expr}` too (`SceneComposer.compose(path, params=...)`, or `scene: plot.yaml, params: {...}` from a parent scene). Unknown params raise. The expression evaluator is AST-restricted: numeric and text literals, param references, `+ - * /`, parens, unit literals (`50cm`, `2m`, `20%`), comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`, chained, `in [..]`), `and`/`or`/`not`, `a if c else b`, and a fixed list of functions (`sind`, `cosd`, `min`, `max`, `abs`, `round`, `floor`, `ceil`). No other calls, no attribute access. A comparison is 1.0 or 0.0, so `when: "{paint == 'taxi' and seats >= 4}"` works, and `"{'brick' if storeys > 1 else 'wood'}"` picks text.

Relevant module: `src/geogen/layout/expressions.py`.

## Surfaces (2D regions on assets)

Where an `AttachmentPoint` is a single spot, a `Surface` is a 2D region with its own (u, v) coordinate system — a wall, a floor, the top of a table. Primitives expose surfaces automatically:
- **CubeGenerator** → `front`, `back`, `left`, `right`, `top`, `bottom`
- **PlaneGenerator** → `top`

For walls, `u` runs horizontally "left → right when facing the wall from outside" (or from inside, for room interior walls) and `v` runs vertical (+Y). Each surface's `normal` points outward from its home object.

Assets can re-export part surfaces at their root with a `surfaces:` block:

```yaml
surfaces:
  front_wall: { from: walls.front }
  roof:       { from: roof.top }
```

The re-export bakes the part's local-to-root transform into the surface so scene-level callers see surfaces in the asset's root frame. Add `cut: [part, ...]` to a surface export (`front_wall: { from: walls.front, cut: [lining] }`) so `cut_host` openings placed on it also cut those parts (`Surface.also_cut`). `flip: true` exports the same region facing the other way, with u still left to right for whoever faces it (`house_peaked`'s `front_inside` etc. are its lining cutter's faces seen from the room).

Relevant module: `src/geogen/layout/surfaces.py`. Runtime storage: `SceneNode.surfaces: dict[str, Surface]`, resolved via `SceneNode.get_surface(name, u, v, depth)`.

## Surface-based placement in scenes

Scenes can place assets on a surface of another placed object with `on:` + `at:`:

```yaml
place:
  house:
    asset: house.yaml
  mailbox:
    asset: mailbox.yaml
    on: house.front_wall          # <object>.<surface>
    at: { u: 0.25, v: 0.3 }       # fractional 0-1 by default
```

Coordinate forms for `u`/`v`/`depth`:
- Plain float (e.g. `0.5`): fractional for `u`/`v`, absolute metres for `depth`
- `{abs: 0.5}`: absolute metres
- `{frac: 0.25}`: explicit fractional

`depth` offsets along the surface normal (useful for pushing an object slightly off a wall so it's not z-fighting).

## YAML loader quirk

`yaml_utils.GeogenSafeLoader` disables YAML 1.1's `on`/`off`/`yes`/`no` boolean resolution so keys like `on:` parse as strings. Always load asset/scene YAML through `LayoutLoader` or `SceneComposer` — using `yaml.safe_load` directly will mangle `on:` into `True`.

## Layout QA

`layout/qa.py`: `check_layout(scene, player)` finds furniture overlaps (chairs may tuck under desks/tables), items in walls, items in door swing arcs, tall items in front of windows (sills below 1.2 m), doors narrower than `player.door_min_width`, and items whose front can't be reached from a door (2D occupancy grid eroded by the player radius, flood-filled from doorways). The furnishing solver uses `room_reachability` to reject placements that would wall off earlier items; the viewer inspector shows a "Layout check" section; `tests/test_asset_quality.py` requires every registered scene to pass.

Layout QA also requires every affordance's `approach` point on an item to be reachable.

Reachability starts from the ways in, not from every door. `check_layout` floods a storey from its entrances (doors from outdoors and the doors of `stair` / `lift_shaft` rooms, `qa.ARRIVAL_ROOM_TYPES`) through the doorways. `room_reachability` (one room, for the solver) starts from the doors people come into that room by (`qa._room_entrances`): a door to a dead end (a shop's stock room, a staff toilet off that) isn't one, so a till can't count on its staff side being reached from the stock room; the dead end's own door is its entrance. Floor-plan openings record the room beyond as `to` (None: outdoors). Rooms that no door joins to a way in keep every door.

`qa.object_overlaps(scene)` intersects (manifold3d) closed meshes of different placed objects (the nearest asset root, `meta.source`, else the top-level placement) and reports ones more than 2 cm deep, e.g. a car through a garage wall. It skips openings in their host (sills, steps), anything sunk into a walkable slab no higher than its top, skinned bodies and colliders.

`--inspect` (`inspection.py`) prints the resolved tree for checking a scene from numbers rather than renders:
- **Tree:** asset roots, rooms, doors, lights, spawns, NPCs and other tagged nodes, with world position, yaw, box, asset and params, a meta summary and interactions. Mesh parts fold into their object's line (`--parts` lists them). Identical siblings collapse to `×N` with each copy's position.
- **Checks:** `mesh`, `uvs` (metric), `layout`, `affordances`, `coplanar`, `overlaps`, `traffic`, `extras` (the closed contract) and `state` (state, wiring and quests compile) (`--checks`), one line per finding, with repeats folded.
- **Options:** `--depth`, `--select GLOB` (adds attachments and surfaces), `--detail`, `--json`.

## Affordance QA

`layout/affordance_qa.py` checks affordances the way the runtime uses them: walk to `approach`, face the anchor, and hold the action's `pose:` at the anchor.
- `stage_actors(scene, body)` poses a body at every affordance. Render with `-r out.png --affordances`, which also prints issues and uses the scene's NPC body if it has one.
- `check_affordances(scene)` reports:
  - `approach_blocked`: no standable floor within 0.3 m of the approach point. Floor is found by slicing collidable meshes between step height and head height, eroded by the actor radius. Interaction-moved parts such as door leaves don't block.
  - `approach_unreachable`: the approach can't be reached from spawns or NPCs.
  - `pose_clash`: the posed body sinks more than 12 L into the asset it uses, or more than 0.5 L into anything else. Feet may press 6 cm into the floor.
- `tests/test_asset_quality.py::test_affordances_are_usable` runs it on every scene. Unit tests are in `tests/test_affordance_qa.py`.

## Furnishing solver

`layout/furnish.py`: room archetypes in `assets/room_types/<room type>.yaml` (`hotel_bedroom`, `hotel_bathroom`, `corridor`) list items as rules (`against: wall|none`, `prefer: [...]`, `flank`, `front_of`, `under`, `on`, `mount: wall|window`, `count`, `optional`) — see the module docstring for the vocabulary. `furnish_plan(plan_root, assets_dir, seed)` places them greedily per floor-plan room, avoiding overlaps + declared clearances, door approach zones and swing arcs, and tall items in front of windows; unmet rules are returned and stored in `meta.furnish_report`. Floor plans record `meta.openings` per room for this. In scenes: `place: {suite: {asset: hotel_suite.yaml, furnish: true}}` (or a seed number) — see `scenes/hotel_room_auto.yaml`.

## Scatter placement

`layout/scatter.py`: a scene placement with `scatter: {seed, rect: [x0, z0, x1, z1] | path: [[x, z], ...] | on: <object>.<surface> | on_terrain: <object> (+ `height: [lo, hi]` above sea level, `slope: [lo, hi]` degrees), count, spacing, avoid: [objects], margin, yaw: [lo, hi], scale: [lo, hi], radius, offset, jitter}` places many seeded copies (Poisson disk in regions/on surfaces, evenly along paths), keeping off avoided objects' footprints and other scatter groups; `params:` values may be `{random: [lo, hi]}` or `{choice: [...]}`, drawn per copy. Scatter placements run after all other placements. Example: `scenes/park.yaml`.

## References and stable IDs

`geogen/refs.py` is the one way to name things in a scene; `on:`, `attach_to`, NPC `home:`, scatter `on:` / `on_terrain:` / `avoid:` and travel spawns all go through it.
- **Syntax:** `house` (a placement of this scene), `plot/house` (inside a placed scene), `town:park/bench` (in another package: a registry scene name; content packages name everything this way), `[tag=furniture.bench]`, `park/[tag=door, type=..., asset=..., name=glob]`. Keys that take a member add it after the first dot: `plot/house.front_wall`.
- **Scopes:** a path step names nodes by their authored name (placement key, part name, room id) within the scope of the step before it. Every asset and placed scene is a scope, so `door` in the cottage is its door placement, not a wardrobe's `door` part, and `suite/bed` skips the room in between. Filter steps search the whole subtree: use them for generated content (city lots, recipes), whose names are only stable per seed. Names may be globs; of same-named siblings the first is `chair`, the next `chair#2`.
- **API:** `refs.resolve` (exactly one) / `refs.select` (any number) raise `RefError` with near names and what exists at the failing step. `SceneRefs` is the composer's wrapper: it adds `file:line: place.x.on` (`yaml_utils.source_of`) and a hint when the target is a placement that isn't placed yet. Placements wait for the one their `on:` / `attach_to` names, whatever the order in the file.
- **Stable IDs:** `<package>:<authored path>` (`cottage:dining/chairs_2`). Exports write them as `extras.geogen.uid` on placed assets and scenes, tagged or typed nodes and anything with interactions (`refs.assign_ids`), on manifest spawns, rooms and travel points, and in `<name>.refs.json` beside the manifest (`geogen-refs` v1; `refs.load_index` gives a stand-in tree that references resolve on, which is how another package's refs are checked: `travel.check_travel`). The runtime never parses a selector: it looks IDs up (`WorldLoader.node_by_uid`) and keys saved state by them.
- **New content that names things** (quests, dialogue anchors, items) should store references in YAML, resolve them at export with `refs.resolve`/`select` and write the IDs to extras or the manifest.
- `python -m geogen.main -s cottage --resolve "house/[tag=furniture]"` prints what a reference matches (`"*"`: the top-level names). Tests: `tests/test_refs.py`.

## Named places

`geogen/places.py`: a scene or asset `places: {name: {...}}` declares trigger volumes for anywhere that isn't a room: the lighthouse `gallery`, the jetty's `on_deck`.
- **Shapes** (keys in `schema.PLACE_KEYS`): `box: [x, y, z]` standing on `position` (scenes also take `anchor` + `offset`; `yaw`), `polygon: [[x, z], ...]` from `y` up `height`, or `around: <reference>` (a placement, or a part in an asset) with `margin` and `headroom` (more above it, so standing on a deck counts).
- **Nodes:** each is an empty child of the scene or asset root named after the place, `meta.type: place`, `meta.place` (`{id, size}` or `{id, polygon, height}`; the node is the middle of the volume's bottom), tag `place` plus its `tags`. A place can't share a name with a placement or part.
- **IDs and wiring:** places get stable IDs (`island:lighthouse/gallery`), so references and `[type=place]` find them, and asset `signals: {occupied: {enter: plate}}` resolves to one (`enter` / `leave` only take a place or a room).
- **Export:** node extras (`place` in the extras schema), and the manifest's and chunk index's `places` in world space (`places_summary`). `--inspect` and `--describe NAME` list them.
- **Runtime:** `GeogenSceneBuilder` adds a `PlaceArea`; `WorldLoader` watches those and the room areas and emits `place_event` `{type: enter | leave, subject, place, kind: place | room, actor}` for the player and NPCs (printed as `place event: {...}`, API stream `place`). Seated actors stay in their place. See "Places" in `runtime/godot/README.md`. The event bus carries these as `enter` / `leave`.
- Tests: `tests/test_places.py`, `tests/test_showcases.py::test_island_places_report_the_player_arriving`.

## Projects and content roots

`geogen/roots.py`: a project that builds on geogen keeps its content outside `assets/`, in a directory laid out the same way (`scenes/`, `materials/`, `quests/`, `npcs/`, `room_types/`, ...), searched before the core.
- **Choosing one:** `--project DIR` (repeatable), `$GEOGEN_PROJECT` (path-separated), or the nearest `geogen.yaml` in or above the working directory. `DIR/geogen.yaml` is `{kind: project, name, assets: <dir, default assets>, overrides: [...]}`; without one, `DIR/assets` (else `DIR`) is the root.
- **Lookup:** anything named relative to `assets/` (`asset:`, `scene:`, `npc:`, `archetype:`, `traffic:`, `train:`, quests, room types, garments, skeletons, materials, `player.yaml`) resolves with `roots.find(name, assets_dir)`: the first root that has it. Listings use `roots.glob(pattern, assets_dir)`. New code that joins a name onto an assets directory should call these instead. An `assets_dir` that isn't a root (a test's temporary directory) stands alone.
- **Shadowing is explicit:** a project file with the same relative path as a core file is an error unless `overrides:` lists it; a project scene can't take a core scene's name any other way.
- The registry discovers scenes in every root, `--describe` lists them, the viewer hot-reloads them and the prototype cache's fingerprint covers them.
- **Scene folders:** `scenes/<name>/scene.yaml` in any root is scene `<name>`, with private files beside it laid out like a root (`safe.yaml`, `materials/brass.yaml`, `quests/open_safe.yaml`). `SceneComposer.compose` searches the folder first while that scene is composed (`roots.scene_scope`; `roots.scope()` is the folder), so the scene names them like anything else. No other scene can name them (`find` raises `RootsError`; a placed `scene: scenes/<name>/scene.yaml` is fine), `roots.glob` and `--describe` leave them out, and a private file may not share a relative name with a shared one. `MaterialLoader` caches by file, and the composer's prototype keys include the folder, so two folders can each have a `brass`.
- **A project's runtime catalogue:** `runtime_scenes.yaml` in its content root is its own list and default scene (core scenes may be listed; it is the one core file name a project may reuse without `overrides:`), and its exports go to `generated:` in `geogen.yaml` (default `<project>/generated`). `catalogue.active()` gives (catalogue file, generated dir): the first project's, else the core's; `load_catalogue()` and `--export-godot` / `--export-catalogue` / `--catalogue` use it, so with a project the core's catalogue and `runtime/godot/generated` are untouched. A project without a catalogue exports with no index (the runtime loads everything). Run it with `godot --path runtime/godot -- --generated=<project>/generated`. Stable IDs and save keys are `<scene name>:...`, and the registry refuses a project scene with a core scene's name, so they can't collide.
- **`geogen check`** (`geogen/check.py`): builds the active projects' own scenes (every scene without a project, or `--all`) and runs every `--inspect` check, plus `textures` (their materials tile; `check.seam_ratio` is what `tests/test_textures_tile.py` uses), and exits 1 on findings; `overlap_unmeasured` is only a note. The `--describe` size budget in `tests/test_describe.py` counts the core's entries only, so a project's assets don't break it. Tests: `tests/test_check.py`.
- `examples/sandwich/` is the worked example (`--project examples/sandwich -s sandwich_kitchen`; `--export-catalogue` writes `examples/sandwich/generated/`): a scene, the sandwich asset and its quest on the core's kitchen, counter, plate, loaf and jars. Tests: `tests/test_roots.py`.

## Scenes & Registry

- **`src/geogen/registry.py`**: `SceneRegistry.discover()` scans `assets/*.yaml` and `assets/scenes/*.yaml` to build the scene list. It peeks at each YAML and classifies it as a composed scene when a `place:` or `city:` key is present, otherwise as an asset; files with a top-level `kind:` (e.g. `player.yaml`) are data and skipped. Python-coded scenes are registered explicitly in `main._build_registry()` (`nature`, `skin_test`).

- **`src/geogen/scenes/*.py`**: Python-coded scenes, used when generation needs custom logic that YAML can't express. There are two, `nature.py` and `skin_test.py`, registered in `main._build_registry()`.

- **`assets/*.yaml`** vs **`assets/scenes/*.yaml`**: assets in the root directory can be either primitives-based assets (have `parts:`) or composed scenes (have `place:`). Files under `assets/scenes/` are always composed scenes. (Example: `assets/dining_set.yaml` is a scene, not an asset.)

## Vocabulary catalogue (`--describe`)

`describe.py` lists what there is to build with, from one set of dicts rendered as text, JSON and markdown.
- **Key vocabulary:** `layout/schema.py` documents every part key (common `PART_KEYS`, per-primitive `PRIMITIVES` with type, default, doc and an `example` part) and the asset, scene, slot and placement keys. `validation.py`'s known-key sets are derived from it, so a new YAML key is added there (with its meaning), not in validation.
- **`--describe`** (no name) reads only the YAML: primitives, keys, assets by category with params, scenes, kinds (`describe.KINDS`, keys from the loaders' `*_KEYS` sets), room types and materials. A one-line summary is the first sentence of the file's header comment, and a param's trailing comment is its note.
- **`--describe NAME...`** (primitive, asset name or file, scene, kind, or a data file) builds assets with default params and adds size, geometry bounds, attachment positions, surfaces, interaction states, affordances, light/vehicle/travel/portal meta and an example placement.
- **`docs/catalogue/`** is committed: `README.md` (thumbnail grid per category), `assets/<name>.md`, `primitives.md`, `vocabulary.md`, `scenes.md`, and `img/` (256 px palette PNGs; `img/hashes.json` holds each asset's YAML hash, so only changed assets re-render: `docs/make_catalogue.sh all` redoes them after a material or generator change).
- **HTML site:** `python docs/make_site.py [OUT]` (stdlib only, default `_site/`) turns those markdown pages into a static site: card grid with a name filter, one page per asset, source links to GitHub. `.github/workflows/pages.yml` builds and deploys it to GitHub Pages when `docs/catalogue/` changes on main, so the markdown stays the only committed copy. It reads just the markdown subset the pages use: a new construct in `describe.py`'s pages needs handling there (`tests/test_site.py` checks links and leftover markdown).
- `tests/test_describe.py` fails when the committed pages are stale (text only): run `docs/make_catalogue.sh` after changing an asset, a key or a header comment.
- Placements take `state: open` (every interaction with that state) or `state: {swing: open}`: the object starts, and is exported, in that state.

`LayoutLoader.load` records `meta.source = {asset, params}` on every asset root; it isn't exported. Erosion is by radius + half a cell, so a gap exactly one body wide counts as shut.
