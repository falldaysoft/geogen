# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geogen is a procedural 3D geometry generator for game assets. It provides a scene graph system with hierarchical transformations, primitive mesh generators, procedural textures, and PBR materials.

This is in active development, with an eventual goal of being able to generate complex, nested geometry like cities that contain neighborhoods that contain roads and houses.

Layout uses a container system that can position objects using anchors (like UI frameworks).

## Commands

```bash
# Run the demo (opens Qt-based interactive viewer)
python -m geogen.main

# Select a specific scene (any YAML in assets/ or assets/scenes/, plus nature; see --help)
python -m geogen.main -s dining_set

# Render to file and quit (for testing) — shadows + PBR maps, auto-framed
python -m geogen.main -r output.png
python -m geogen.main -s room -r output.png --resolution 1280x720
python -m geogen.main -s chair -r sheet.png --views            # iso/front/side/top contact sheet
python -m geogen.main -s chair -r sheet.png --views iso,back,top --resolution 700x700
python -m geogen.main -s chair -r out.png --view side --zoom 1.5 --no-ground
python -m geogen.main -s hotel_room_auto -r out.png --views iso,top --cutaway   # hide ceilings/roofs to see inside

# Export for game engines (hierarchy + PBR textures)
python -m geogen.main -s town --chunks out/town_chunks --cache    # streamable chunks + index
python -m geogen.main -s dining_set -e out/dining_set.glb     # .glb / .gltf / .obj
python -m geogen.main -s cottage --export-godot               # into runtime/godot/generated/

# Walk an export in the Godot runtime (macOS binary: /Applications/Godot.app/Contents/MacOS/Godot)
godot --path runtime/godot -- --scene cottage
python -m geogen.main -s town --export-godot --stream        # chunked export the runtime streams
godot --path runtime/godot -- --scene town --stream

# Regenerate the README screenshots (docs/images/)
docs/make_screenshots.sh

# Screenshot the interactive Qt viewer (display: lit|clay|normals|uv)
python -m geogen.main -s street --viewer-screenshot shot.png --display uv

# Install dependencies
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run a single test
pytest tests/test_scenes.py -k "test_name"
```

## Architecture

### Core System (`src/geogen/core/`)

- **SceneNode** (`node.py`): Hierarchical scene graph with parent-child relationships. Each node has a local Transform, optional Mesh, and children. Provides `world_transform()` for combined transformation matrices, `flatten()` to merge all geometry, and `iter_meshes()` for traversal. Supports attachment points via `get_attachment()`.

- **Mesh** (`mesh.py`): Geometry container storing vertices, faces, normals, UVs, and optional Material. Has `transform()` method, `merge()` classmethod for combining meshes, and conversion to/from trimesh.
  - **Material groups:** optional per-face `face_materials` index into `materials`, and `merge()` of differently-materialled meshes produces these groups. `groups()` splits them for the renderers, and the exporter writes one glTF primitive each.
  - **Vertex colours:** optional per-vertex RGBA `colors`, which tint the albedo in both renderers and export as `COLOR_0`.
  - **Propagation:** weld, normals, UV projection, subdivision, displacement and CSG/decimation all carry groups (CSG through a manifold vertex property); a difference's cut faces take the target's primary material. Derived meshes copy them with `with_attributes_of`.

- **Transform** (`transform.py`): TRS (Translation-Rotation-Scale) transformation. Rotation uses XYZ Euler angles in radians. Matrix order is Scale -> Rotate -> Translate.

- **geometry** (`geometry.py`): Helper functions for face winding and normal computation. Uses CCW winding convention.

- **meshops** (`meshops.py`): `compute_normals(mesh, crease_angle)` (smooth below the angle, split hard edges above; ignores UV seams), `ensure_normals`, `weld_vertices`, `compute_tangents` (glTF-style xyzw), and `validate(mesh) -> MeshReport` (degenerate faces, NaNs, boundary/non-manifold edges, inconsistent winding). Use `validate` whenever you touch a generator.

- **subdiv** (`subdiv.py`): `subdivide(mesh, levels, crease_angle)` (Loop, crease/boundary rules, sharp crease turns stay corners) and `displace(mesh, amplitude, scale, octaves, seed, ridged)` (fractal 3D gradient noise along smoothed normals; closed meshes stay closed). In YAML any part takes `subdivide: 3` / `{levels, crease}` (result is stretched back to the part box; `LayoutLoader(detail)` adds/removes levels) and `displace: {amplitude, scale, octaves, seed, ridged}`; UVs are box-projected by the undisplaced normals so noisy surfaces get clean seams. Used by rocks (`seed` param), bushes and bed pillows.

- **uvmap** (`uvmap.py`): Metric UV projection — `box_project`, `planar_project`, `cylindrical_project`, and `texel_density` (1.0 == metric).

- **profile** (`profile.py`): 2D shapes for extrude/lathe: `Shape(outer, holes)` (auto CCW/CW), `rect(w, h, radius)`, `circle`, `ellipse`, `regular_polygon`, `fillet`, `arc`, `bezier`, `catmull_rom`, `offset`/`Shape.difference` (shapely), `triangulate` (earcut), and YAML builders `loop_from_spec`/`shape_from_spec`/`polyline_from_spec`.

### Generators (`src/geogen/generators/`)

- **MeshGenerator** (`base.py`): Abstract base class for generators producing single meshes. Implement `generate() -> Mesh`.

- **CompositeGenerator** (`base.py`): Abstract base for generators producing scene hierarchies.

- **Primitives** (`primitives.py`): Dataclass-based generators for Cube, Sphere, Cylinder, Cone. Each declares its own attachment points via `get_attachment_points(size)`. `CubeGenerator` is a rounded box: `bevel` is the edge radius (default 0.02) and `bevel_segments` the arc steps per 45° (default 2), with analytic normals. Set `bevel: 0` in YAML for sharp cubes (do this for thin flat surfaces like roads/sidewalks to avoid disproportionate bevels).

- **ExtrudeGenerator / LatheGenerator** (`profiles.py`): Extrude a `Shape` (with holes) along x/y/z with an optional rounded cap bevel; revolve an (r, y) profile around Y (partial `sweep` supported, r=0 closes at the axis). Both are watertight, metric-UV'd, and use a crease angle (default 40°) for normals.

- **SweepGenerator** (`sweep.py`): `primitive: sweep` sweeps a `profile` shape (metres; x = right of travel, y = up) along a `path` of [x, y, z] points or `{spline: [...], samples}`, with rotation-minimising frames, mitred corners, `closed`, `twist` (deg), `scale: [start, end]`, caps on open paths. `size` is optional (taken from the path); `center: false` keeps the path in the asset frame (anchor ignored). Example: `assets/handrail.yaml`.

- **StairsGenerator** (`stairs.py`): `primitive: stairs` with `style: straight | l | spiral`, `rise`, `width`, `max_riser` (step count = ceil(rise / max_riser), so any rise stays climbable), `tread`, `turn`, `landing_at`, `railing: both|left|right|outer|none`, `railing_material`. Body is one watertight solid (walkable, trimesh collider); swept handrails on posts are a `<part>_railing` child. Climbs toward -Z. Example: `assets/staircase.yaml`.

- **RoofGenerator / PrismGenerator** (`architecture.py`): `primitive: roof` with `style: gable|hip|shed|flat`, `overhang`, `thickness`, `ridge_axis: auto|x|z`, `ridge_cap`. Size = wall-top footprint + rise; the part's bottom is the wall-top plane, eaves overhang beyond it. UVs are face-planar (shingle rows parallel to eaves). `primitive: prism` fills gable ends (`apex: center|back|front` for wedges/ramps).

- **CSG** (`core/csg.py`): `difference`, `union`, `intersection` on closed meshes via manifold3d; UVs survive, normals are recomputed. In YAML: `subtract: [part, ...]` on a target and `cutter: true` on helper parts (removed after cutting). `cut_host: true` marks an opening cutter that is applied to *whatever surface the asset is placed on* — `window.yaml` / `door.yaml` use this so `on: house.front_wall` cuts a real opening into the wall part (surfaces remember their `source` part).

- **Round shapes** (`round_shapes.py`): `primitive: torus` (fills x/z; `tube` radius, default size_y / 2), `primitive: capsule` (radius min(x, z) / 2, hemispherical ends), and `cylinder` with `bevel: <radius>` for rounded rims (cylinders stay sharp without it). Lathe-based, watertight, metric UVs.

- **EllipsoidGenerator** (`primitives.py`): Sphere stretched to all three size components (`sphere` keeps using `min(size)` for back-compat).

- **FloorPlan** (`floorplan.py`): Multi-room storeys from a YAML `floorplan:` block (an asset with `floorplan:` instead of `parts:`; `size` is computed). Rooms are grid rectangles on wall centre lines (`rect: [x, z, w, d]`); edges shared by two rooms become `interior_wall` (0.12) walls, outside edges `exterior_wall` (0.3), all centred on the edge. Segments are extended to their perpendicular walls and unioned in 2D (shapely) before one extrusion, so junctions are clean and `walls` is a single watertight mesh; `doors` (`between: [a, b]` or `room`+`side`) and `windows` (`room`, `side`, `at`, `width`, `height`, `sill`) are CSG-cut. Each room is a node (tags `room`, `room_type`) with `floor`/`ceiling` slabs (per-room `floor:`/`ceiling:` materials) and surfaces exported at the root as `<room>.<surface>`: `floor`, `ceiling`, `<side>_wall` (facing into the room) and `<side>_exterior` on outside walls, so `on: suite.bedroom.south_exterior` places a window asset that cuts the walls. North = +Z. The default 0.2 m ceiling slab is deliberate: thin slabs leak sunlight through shadow maps in Godot (ceilings tuck into walls; floors deliberately don't, so doorway thresholds don't z-fight). Doors: `style: door|archway|opening`, `swing: <room>|out` (default: second room of `between`, or into the room), `hinge: left|right` (seen from the swing side), `open: <deg>`; `DoorGenerator` (`generators/doors.py`) builds lining + architraves on both faces + a `leaf_pivot` node at the hinge (`meta.joint`) holding the leaf and chrome levers. Door nodes are tagged `door.interior|exterior` / `opening.archway`; the swing room gets `meta.door_swings` arcs. Example: `assets/hotel_suite.yaml`.

- **Buildings** (`building.py`): an asset with `building: {storeys: [{floorplan: {...}, repeat: n, furnish: true}], roof: {parapet, material}}` stacks floor plans (each storey's floor slab on the walls below; storey nodes tagged `storey.<n>` with `meta.storey`), numbers guest floors via the layout's `floor` param, links every `stair`-type room to the one above with a straight flight (floor above and ceiling below cut open), and adds a flat roof + parapet. Rooms of type `lift_shaft` stacked over storeys (both hotel layouts reserve one, centred) become a lift: the shaft is opened up, a `lift_car` rides it via a `lift` interaction (states `floor_<k>`, E goes to the next floor), and each floor's shaft doorway gets a gate (`meta.gate`: solid unless the car is at that floor; Godot toggles it). Buildings add an `entrance_spawn` outside the street door. Example: `assets/hotel.yaml` (lobby + 3 furnished guest floors). Layout QA checks reachability per storey.

- **Façades** (`facade.py`): `building: {facade: {style: brick_hotel | stucco | modern, balconies: bool}}` clads the structural walls (plan material `exterior`, separate from room linings), frames and glazes every exterior window (storefront mullions on low ground-floor windows), adds sills/lintels per style, string courses at each floor, a cornice, canopies over ground-floor exterior doors and optional Juliet balcony rails. Parts are merged per material under a `facade` node.

- **Plan layouts** (`plan_layouts.py`): `floorplan: {generate: hotel_corridor | hotel_lobby, ...params}` expands to rooms/doors/windows (plan keys like `wall_height`, `materials`, `finishes` pass through). `hotel_corridor`: stairs at the ends, lift lobby + service core mid-floor, double-loaded guest modules (`module`, `ensuite`, `bath_width`, ...). `hotel_lobby`: street entrance, lobby, reception, lounge, restaurant, restrooms, back-of-house. Assets `hotel_floor.yaml`, `hotel_lobby.yaml`; `scenes/hotel_floor_furnished.yaml` furnishes a whole floor. Public-room archetypes (`lobby`, `reception`, `lounge`, `restaurant`, `restroom`, `back_of_house`, `lift_lobby`) furnish the ground floor with `sofa`, `coffee_table`, `reception_desk`, `planter`, tables with chairs (`around: table`), etc.

- **Nature** (`nature.py`):
  - `primitive: tree` grows a space-colonisation skeleton in the part box: `style: deciduous` (ellipsoid crown) or `conifer` (cone, drooping boughs), with `seed`, `trunk_height`, `trunk_radius`, `attractors`, `step`, `leaf_size` and `foliage_material`.
  - Branch radii use the pipe model. The trunk mesh gets a hull collider; the `branches` and `foliage` children (noise-displaced leaf clusters on the twigs) have no collider.
  - `primitive: rock` is a random convex cage (`seed`, `points`, `levels`, `roughness`, `crease`), subdivided, displaced and flat-bottomed.
  - Generated child meshes inherit the part's `material` (bark).

- **RoomGenerator** (`room.py`): Generates rooms with walls, floor, ceiling, and openings (doors/windows). Supports `generate_parts()` for separate surface meshes with different materials. Uses `Opening` dataclass for doors/windows with wall position, size, and bottom offset.

### Textures (`src/geogen/textures/`)

- **TextureGenerator** (`base.py`): Abstract base class for procedural textures. Generates PIL Images and optional PBR maps (normal, roughness, AO). Uses numpy RNG with optional seed.

- **NoiseTextureGenerator**: Shared base for noise-based textures. Subclass implements `_get_noise_layers()`, `_compute_pattern()`, `_apply_color_shift()`. `BrickTextureGenerator` breaks this contract (returns a `(pattern, is_mortar)` tuple) and overrides `generate()`.

- Implementations: `wood`, `metal`, `floor`, `wall`, `asphalt`, `brick`, `concrete`, `dirt`, `grass`, `rock`, `roof` — each in its own module.

- **Interior pack** (`interior.py`): `fabric` (plain/twill weave, `threads` per repeat), `cut_pile_carpet`, `tile` (`tiles_x`/`tiles_y` per repeat, `grout_width`, `layout: grid|offset`), `marble`. Materials: `fabric_linen`, `fabric_velvet`, `bedding_white`, `carpet`, `tile_white`, `tile_subway`, `tile_floor_grey`, `marble`, `chrome`, `mirror`, `laminate`, `paint_{white,sage,navy,terracotta,greige}`. Any colour param in a material YAML may name a `PAINT_PALETTE` colour (`materials/loader.py`) instead of `[r, g, b]`.

### Materials (`src/geogen/materials/`)

- **Material** (`material.py`): Combines a TextureGenerator with PBR properties (roughness, metallic, normal_strength, ao_strength). Caches generated textures. `tile_size` (metres per texture repeat, YAML scalar or `[u, v]`) converts metric mesh UVs to texture space via `texture_uv_scale`; `gltf_images()` packs base colour / metallic-roughness / normal / occlusion for both the renderer and exporter.

- **MaterialLoader** (`loader.py`): Loads material definitions from YAML files in `assets/materials/`. `pbr: {opacity: 0.3}` makes a material alpha-blended (glass: see-through in renderer, viewer, glTF `BLEND` + double-sided, Godot); `pbr: {emissive: [r, g, b], emissive_strength}` makes it glow (`lamp_shade`, `screen`).

### Lighting (`src/geogen/lighting/`)

- **Light classes**: `DirectionalLight` (sun-like), `PointLight` (omni). Both have color and intensity.

- **SceneLighting**: Container with ambient color and lights list. Provides `get_shader_data()` for shader uniforms. Presets: `default()` and `room_lighting()`.

### Layout System (`src/geogen/layout/`)

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

### Furniture library

Parametric furniture assets in `assets/` (tags `furniture.*` / `bathroom.*` / `decor.*`, `clearance:` → `meta.footprint` + `meta.clearance` in extras): bed, nightstand (drawer interaction), wardrobe (door interaction), desk, desk_chair, armchair, bookshelf, floor_lamp, table_lamp, tv_console, luggage_rack, rug; wall-mounted (origin on the wall plane, depth along +Z): tv, wall_mirror, wall_art, curtains, towel_rail; bathroom: toilet (lid interaction), vanity (basin + tap), shower, bathtub. `scenes/hotel_room.yaml` furnishes `hotel_suite.yaml` by hand; `tests/test_furniture.py` checks nothing overlaps or blocks door swings. Soft goods are rounded boxes/ellipsoids until subdivision (geogen-o3s.14).

### Lighting

Fixtures carry `meta.light` (`{type: omni, color, energy, range, offset: [x, y, z]}`): assets declare it with a top-level `light:` block (`table_lamp`, `floor_lamp`), ceiling pendants from finishes have one. Exports add a KHR_lights_punctual point light per fixture on a `<node>_light` child (intensity = energy × `CANDELA_PER_ENERGY`); the Godot runtime tunes those imported lights (energy, range, shadows) rather than adding its own, and `--lights` prints them. Light switches have a `switch` interaction (on/off, rocker tips) and `meta.switch.light`; Godot hides the light when switched off. The viewer's **Night** toggle (N, `--night`) dims the sun/sky and lights the scene with its fixtures (up to 15 nearest, range-windowed; from the uncut scene so cutaways stay lit; the ground ignores them since point lights have no shadows).

### Layout QA

`layout/qa.py`: `check_layout(scene, player)` finds furniture overlaps (chairs may tuck under desks/tables), items in walls, items in door swing arcs, tall items in front of windows (sills below 1.2 m), doors narrower than `player.door_min_width`, and items whose front can't be reached from a door (2D occupancy grid eroded by the player radius, flood-filled from doorways). The furnishing solver uses `room_reachability` to reject placements that would wall off earlier items; the viewer inspector shows a "Layout check" section; `tests/test_asset_quality.py` requires every registered scene to pass.

### Interior finishes

`generators/finishes.py` (called from `FloorPlan.build`, `floorplan: {finishes: true | false | {lining, skirting, cornice, light, switches}}`): per room a 1 cm `lining` in the room's wall material (cut by openings; room wall surfaces sit on it), swept `skirting` broken at doorways, swept `cornice`, a pendant `<room>_light` with `meta.light` (Godot adds an `OmniLight3D`; fixture meshes don't cast shadows) and `<room>_switch_N` beside each door's latch side (`meta.switch.light`). Room materials resolve room keys (`floor`/`walls`/`ceiling`) > room type `finishes:` in `assets/room_types/<type>.yaml` > plan `materials`. Rooms record `meta.wall_inset` so the furnishing solver keeps floor items off the skirting.

### Furnishing solver

`layout/furnish.py`: room archetypes in `assets/room_types/<room type>.yaml` (`hotel_bedroom`, `hotel_bathroom`, `corridor`) list items as rules (`against: wall|none`, `prefer: [...]`, `flank`, `front_of`, `under`, `on`, `mount: wall|window`, `count`, `optional`) — see the module docstring for the vocabulary. `furnish_plan(plan_root, assets_dir, seed)` places them greedily per floor-plan room, avoiding overlaps + declared clearances, door approach zones and swing arcs, and tall items in front of windows; unmet rules are returned and stored in `meta.furnish_report`. Floor plans record `meta.openings` per room for this. In scenes: `place: {suite: {asset: hotel_suite.yaml, furnish: true}}` (or a seed number) — see `scenes/hotel_room_auto.yaml`.

### Interactions

`layout/interactions.py`: asset-level `interactions:` state machines (states with `next`/`then`/`emit`/`prompt`, `motions` that `rotate`/`translate` parts about a `pivot` in the asset frame, per-state `values`, `targets` the player aims at). `SceneNode.interactions` holds them; the loader poses parts for `initial`, `apply_state()` re-poses (absolute). Exported in the asset node's `extras.geogen.interactions` with exported node names (see the schema). States can auto-advance after a dwell (`open: {then: closed, after: 6}`); `lock: {key, locked}` makes an interaction refuse use while locked; interactions may have no motions (the TV's `power`). Floor-plan doors take `lock: <key id>` and `auto_close: <s>` (generated guest doors: `key_room_<n>`, 6 s). Assets declare `affordances:` (`[{type: sit|lie|use|stand, at: <attachment> | [x,y,z], facing: deg, height}]`: chairs, sofa, bench, bed) and `container: {capacity, part}` (wardrobe, nightstand), exported in extras. Part shorthand: `joint: {type: hinge, pivot: left|right|top|bottom|[x,y,z], limits: [0, 100], with: [parts]}` or `{type: slide, axis: z, range: [0, 0.3]}` expands to a closed/open interaction named after the part (hinges default to swinging out toward +Z). `door.yaml` has a `swing`; floor-plan doors (`DoorGenerator`) get one on `leaf_pivot`. Godot: `GeogenInteraction` (`runtime/godot/scripts/interaction.gd`) animates parts, honours locks (`--keys`, L to lock/unlock), `after` dwells, on/off emission, and snapshot/restore (`--save`/`--load`); E on furniture sits/lies on its nearest affordance; colliders under moving parts are `AnimatableBody3D`; look at a target within `reach` and press E; `--use=<asset>` / `--use=@aim` + `--wait=S` for headless tests; state arrivals print `interaction event: {...}`.

### Scatter placement

`layout/scatter.py`: a scene placement with `scatter: {seed, rect: [x0, z0, x1, z1] | path: [[x, z], ...] | on: <object>.<surface>, count, spacing, avoid: [objects], margin, yaw: [lo, hi], scale: [lo, hi], radius, offset, jitter}` places many seeded copies (Poisson disk in regions/on surfaces, evenly along paths), keeping off avoided objects' footprints and other scatter groups; `params:` values may be `{random: [lo, hi]}` or `{choice: [...]}`, drawn per copy. Scatter placements run after all other placements. Example: `scenes/park.yaml`.

### Scenes & Registry

- **`src/geogen/registry.py`**: `SceneRegistry.discover()` scans `assets/*.yaml` and `assets/scenes/*.yaml` to build the scene list. It peeks at each YAML and classifies it as a composed scene when a `place:` or `compose:` key is present (treated equivalently), otherwise as an asset; files with a top-level `kind:` (e.g. `player.yaml`) are data and skipped. Python-coded scenes are registered explicitly in `main._build_registry()` (only `nature`).

- **`src/geogen/scenes/*.py`**: Python-coded scenes, used when generation needs custom logic that YAML can't express. Only `nature.py` is registered; `chair.py`, `table.py`, `dining_set.py`, `room.py`, `street.py` are legacy thin wrappers around their YAML counterparts and aren't used by the registry.

- **`assets/*.yaml`** vs **`assets/scenes/*.yaml`**: assets in the root directory can be either primitives-based assets (have `parts:`) or composed scenes (have `place:`/`compose:`). Files under `assets/scenes/` are always composed scenes. (Example: `assets/dining_set.yaml` uses `compose:` and is a scene, not an asset.)

### Viewer (`src/geogen/viewer/`)

- **qt_viewer.py**: `ViewerWindow` — filterable scene list, node tree synced with viewport picking, inspector (size, tris, materials, attachments, surfaces, mesh validation), toolbar (reload, views, display mode, wireframe/grid/shadows/ground, screenshot), hot reload of `assets/**/*.yaml` via QFileSystemWatcher, errors shown in the status bar instead of crashing. Keys: 1/2/3/4/7/0 views, F frame selection, A frame all, W/G/S/H toggles, M display mode, Ctrl+R/S/F.
- **Section toolbar** (viewer): *Cutaway* (C) hides ceilings/roofs/ceiling lights, *Storey* shows building storeys up to N (façade per storey too), *Interactions* poses every interaction in a state (e.g. `open`) on a copy. CLI: `--viewer-screenshot out.png --cutaway --storey 1 --state open`. The inspector lists tags, room/storey/collider/joint/light/gate meta and interactions (states, motion axes and pivots).
- **gl_view.py**: `GLView` QOpenGLWidget — sun shadow map (PCF), hemisphere ambient + ACES tonemap, display modes Lit/Clay/Normals/UV checker (1 checker cell = 25 cm), selection highlight, CPU ray picking, zoom-to-cursor.
- **camera.py**: Pure-numpy `OrbitCamera` (framing, presets, pan/orbit/zoom, adaptive clip planes, pick rays) and `ray_mesh_intersect` — unit tested without GL.
- **shaders/**: `scene.vert/.frag` (PBR + shadows + debug modes), `depth.*` (shadow pass), `line.*` (grid/axes/wireframe).
- **viewer.py**: Legacy trimesh-based `Viewer` (still used by tests for trimesh scene conversion).

### Rendering & Export

- **render.py**: Offscreen pyrender renderer used by `-r`: `SceneRenderer`, `render_scene`, `render_views` (contact sheet), `RenderOptions`, `VIEWS` presets. Shares one offscreen context per process (macOS). Contains a `np.infty` shim for pyrender 0.1.45 on NumPy 2.
- **export.py**: `export_scene(root, path, player=None)` → GLB/glTF (node hierarchy with local transforms, PBR textures, texture-space UVs) or OBJ+MTL+PNG. GLBs carry a glTF animation per interaction transition (`<asset>/<interaction>/<from>-><to>`, sampled local TRS; animated nodes use TRS instead of `matrix`) for engines that don't read extras.geogen. glTF exports also write `<name>.manifest.json` (format `geogen-manifest` v1: model file, units, up axis, player spec, `rooms`, `spawns`) — the contract the Godot runtime reads.
- **LODs**: `meshops.decimate(mesh, ratio)` (manifold3d edge collapse; UVs and hard edges kept; open meshes unchanged); `export_scene(..., lods=[0.5, 0.25])` / `-e out.glb --lods 0.5,0.25` adds `<name>_LOD<n>` levels as glTF `MSFT_lod` (meshes with ≥ 200 tris; Godot drops them and uses its own auto-LOD). `LayoutLoader(detail=0.5)` scales curved primitives' tessellation.
- **Gameplay metadata**: `SceneNode.tags` (list, dotted: `room.hotel_bedroom`, `furniture.chair`) and `SceneNode.meta` (dict) are exported as glTF node extras `extras.geogen` (schema `docs/schema/geogen-extras.v1.schema.json`, validated in `tests/test_export.py`). YAML: asset/scene/placement `tags:`, part `collider: auto|none|box|hull|mesh` and `walkable: true`, scene `spawns:` (slot syntax; the spawn faces its `facing`). Every mesh gets a collider child named with Godot import suffixes (`-convcolonly` box/hull, `-colonly` trimesh); `auto` (export.resolve_collider) picks box if the mesh fills its bounds, hull if nearly convex, else trimesh, none below 3 cm. Floor plans add `<room>_volume` nodes (`type: room_volume`).
- **Scale** (`chunks.py`, `layout/cache.py`): `SceneNode.instance()` deep-copies a subtree sharing its meshes (interaction targets remapped). `LayoutLoader.load` and `SceneComposer._load_object` cache one prototype per (file, params[, furnish]) and return instances, so repeated furniture/buildings share meshes. The exporter writes each shared mesh (and its collider/LODs) once. `--cache [DIR]` / `$GEOGEN_CACHE` pickles prototypes on disk, keyed by a fingerprint of all assets and sources. `--chunks DIR` (`export_chunks`) writes per-block/building exterior GLBs, `_lod` stand-ins (decimated, merged per material, small parts dropped), per-building `_interior` GLBs (rooms + interior doors, for load-on-enter), `base.glb`, shared `textures/` (content-hashed, referenced by URI) and a `<name>.chunks.json` index (`geogen-chunks` v1). The Godot runtime streams them (`--export-godot --stream`, `scripts/chunk_streamer.gd`: full / LOD / interior radii, threaded loading, navmesh tiles baked around the player; see `runtime/godot/README.md`).
- **player.py**: `PlayerSpec` / `load_player_spec()` — player-scale constraints (capsule radius/height, eye and step height, max slope, min door opening, corridor width, reach) from `assets/player.yaml`. Use it for defaults instead of hard-coding clearances. YAML files with a top-level `kind:` (like `player.yaml`) are data, not assets; the registry skips them.

### Godot runtime (`runtime/godot/`)

Godot 4.7 reference runtime (Forward+, 1 unit = 1 m). `python -m geogen.main -s cottage --export-godot` writes `.glb` + manifest into `runtime/godot/generated/`; `godot --path runtime/godot -- --scene cottage` walks it in first person. `WorldLoader` loads exports at runtime (GLTFDocument, trimesh colliders, mipmaps) and hot-reloads when a manifest changes; `Player` is a cylinder `CharacterBody3D` sized from `PlayerSpec` with step-up. `--playtest[=N]` (run with `--headless --fixed-fps 60`) is the enterability check: every room volume and interaction target must be reachable on the navmesh from the export's spawn (doors opened and baked as obstacles first), and a bot walks N routes with the real player body; prints `playtest: {...}` and exits 1 on failure (`tests/test_godot_runtime.py`). Buildings add an `entrance_spawn` outside the street door. `addons/geogen/` (`GeogenSceneBuilder`, enabled editor plugin) turns extras into room Area3Ds, spawn markers, tag groups and a navmesh baked from colliders (moving parts excluded, so doorways stay navigable) — both on editor import of a geogen `.glb` and at runtime. Useful args after `--`: `--generated=DIR`, `--spawn=X,Y,Z`, `--walk=SECONDS` (prints the end position; used by `tests/test_godot_runtime.py`), `--nav=AX,AZ:BX,BZ` (prints a navmesh path), `--screenshot=out.png`, `--camera=overview`, `--colliders`. See `runtime/godot/README.md`. On macOS the binary is `/Applications/Godot.app/Contents/MacOS/Godot` (tests honour `$GODOT` and skip without it). Wrap ad-hoc Godot runs in a timeout: a GDScript parse error leaves the process running instead of exiting.

## Hierarchical Layout System - Semantic Connections

The layout system uses **semantic attachment points** instead of raw coordinates. This makes objects self-describing and composable.

### Two Levels of Composition

1. **LayoutLoader** (`loader.py`): Builds individual assets from primitives
2. **SceneComposer** (`composer.py`): Assembles assets into larger scenes

### Asset-Level: Part Attachments (`attach_to`, `at`, `from`)

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

### Asset-Level: Custom Attachment Points

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

### Scene-Level: Slots and Asset Attachments

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

### Hierarchy Composition

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

### City layout

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

### Buildings

`house_peaked.yaml` is a hollow brick shell (walls minus an `interior` cutter) with a gable `roof`, brick `prism` gables and a chimney. Inside it has a separate finish: `floor` (hardwood_floor), a hollow `lining` (wall_plaster, 1.5 cm) and a `ceiling` at the wall top. Its wall surfaces are exported with `cut: [lining]` and `reveal: {material: wall_plaster, thickness: 0.015}`, so openings cut through the brick and the lining, and the composer adds a `<object>_reveal` plaster sleeve behind the frame. Opening assets mark that region with a `reveal: true` part (`reveal: {bottom: false}` for doors), a box from the frame's back face into the room; it is clipped to the removed wall/lining material. `scenes/cottage.yaml` places `door.yaml` and `window.yaml` on its wall surfaces (each cuts its own opening) and furniture (dining set, `bookshelf.yaml`, `armchair.yaml`) on `house.floor`. Surface placements take `facing:` (compass/center/outward) and `yaw:` (degrees) to turn objects. Opening assets are authored with the origin at the bottom-centre of the opening on the wall face, +Z out of the wall, and a 1 m container depth so z offsets read as metres. `scenes/house_plot.yaml` (used by the street) places the cottage scene.

### Design Principles

1. **No magic numbers in connections**: Parts connect via named points (`top`, `bottom`, `seat_front`), not coordinates
2. **Self-describing objects**: Assets declare their own attachment points based on their semantics
3. **Composability**: Objects don't know about their containers; containers know about objects' attachment points
4. **Hierarchical transforms**: Child transforms are relative to parent, enabling grouped movement

## Parametric Assets, Surfaces, and Surface Placement

### Parameters and `{expr}` interpolation

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

Assets whose container is just a unit (e.g. `size: ["{scale}", "{scale}", "{scale}"]` with part sizes in metres) set `bounds: geometry` so their reported `size` is the real extent. Overrides come in through `LayoutLoader.load(path, params={"width": 12})`, or per placement in scenes: `door: {asset: door.yaml, params: {width: 1.1}, on: ...}`. Scenes can declare `params:` and use `{expr}` too (`SceneComposer.compose(path, params=...)`, or `scene: plot.yaml, params: {...}` from a parent scene). Unknown params raise. The expression evaluator is AST-restricted — only numeric literals, param references, `+ - * /`, parens, and unit literals (`50cm`, `2m`, `20%`). No function calls, no attribute access.

Relevant module: `src/geogen/layout/expressions.py`.

### Surfaces (2D regions on assets)

Where an `AttachmentPoint` is a single spot, a `Surface` is a 2D region with its own (u, v) coordinate system — a wall, a floor, the top of a table. Primitives expose surfaces automatically:
- **CubeGenerator** → `front`, `back`, `left`, `right`, `top`, `bottom`
- **PlaneGenerator** → `top`
- **RoomGenerator** → `north_wall`, `south_wall`, `east_wall`, `west_wall`, `floor`, `ceiling` (all interior-facing, normals point *into* the room)

For walls, `u` runs horizontally "left → right when facing the wall from outside" (or from inside, for room interior walls) and `v` runs vertical (+Y). Each surface's `normal` points outward from its home object.

Assets can re-export part surfaces at their root with a `surfaces:` block:

```yaml
surfaces:
  front_wall: { from: walls.front }
  roof:       { from: roof.top }
```

The re-export bakes the part's local-to-root transform into the surface so scene-level callers see surfaces in the asset's root frame. Add `cut: [part, ...]` to a surface export (`front_wall: { from: walls.front, cut: [lining] }`) so `cut_host` openings placed on it also cut those parts (`Surface.also_cut`).

Relevant module: `src/geogen/layout/surfaces.py`. Runtime storage: `SceneNode.surfaces: dict[str, Surface]`, resolved via `SceneNode.get_surface(name, u, v, depth)`.

### Surface-based placement in scenes

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

### YAML loader quirk

`yaml_utils.GeogenSafeLoader` disables YAML 1.1's `on`/`off`/`yes`/`no` boolean resolution so keys like `on:` parse as strings. Always load asset/scene YAML through `LayoutLoader` or `SceneComposer` — using `yaml.safe_load` directly will mangle `on:` into `True`.

## Key Conventions

- **UVs are metric** (1 UV unit = 1 m of surface). Generators must emit metric UVs (or run `uvmap.box_project`); materials set `tile_size`. Never emit 0–1-per-face UVs — texture density would vary with object size.
- **Procedural textures must tile**: noise is periodic (`perlin_noise` wraps its lattice); `tests/test_textures_tile.py` checks every material.
- Generators should produce closed meshes that pass `meshops.validate`; `tests/test_asset_quality.py` checks every registered asset/scene.
- For turned/curved/filleted parts prefer `lathe`/`extrude` over stacking primitives. YAML anchors+merge keys (`&leg` / `<<: *leg`) work for repeated parts.

- All meshes use counter-clockwise face winding for outward normals
- Transformations follow order: Scale -> Rotate -> Translate
- Geometry is centered at origin by default
- numpy arrays use `float64` for vertices/normals and `int64` for face indices
- Materials are optional on Mesh; viewer uses default gray when missing

## Testing

Always test changes by:
- Rendering a png of the update into /tmp
- Visually inspect the png
- If it's too small to see clearly, iterate until you get a good view.
- When you generate or update an object, render it from the front and side and make sure it looks correct and consistent.
- For runtime changes, export with `--export-godot` and screenshot the Godot view (`-- --scene X --screenshot=out.png`, add `--colliders` / `--camera=overview`); use `--walk=SECONDS` with `--spawn`/`--yaw` for headless movement checks (see `tests/test_godot_runtime.py`).
- `pytest` includes Godot runs (`tests/conftest.py` `run_godot` fixture); they skip if Godot isn't installed.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->
