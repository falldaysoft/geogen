# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geogen is a procedural 3D geometry generator for game assets. It provides a scene graph with hierarchical transforms, primitive and procedural mesh generators, procedural (tiling) textures and PBR materials, and a YAML layout system that composes assets into scenes using anchors, attachment points and surfaces (like a UI framework). The goal is complex nested worlds: cities that contain neighbourhoods that contain roads, buildings, furnished rooms, people and traffic, exported to game engines with the gameplay data (interactions, NPCs, rules, quests) a runtime needs. `runtime/godot/` is the reference runtime.

## Commands

```bash
python -m geogen.main                                  # Qt viewer; -s NAME picks a scene (any YAML in assets/, assets/scenes/; --help)
pip install -e ".[dev]"                                # install with dev dependencies

# Render to file and quit (shadows + PBR, auto-framed)
python -m geogen.main -s chair -r out.png [--resolution 1280x720]
python -m geogen.main -s chair -r sheet.png --views [iso,back,top]        # contact sheet
python -m geogen.main -s chair -r out.png --view side --zoom 1.5 --no-ground
python -m geogen.main -s portraits -r out.png --frame Head --view front --fov 10   # close-up on what a joint/node moves
python -m geogen.main -s hotel_room_auto -r out.png --views iso,top --cutaway   # hide ceilings/roofs
python -m geogen.main -s skin_test -r out.png --clip sway@1.0          # also --pose NAME, --filmstrip N, --affordances, --lanes
python -m geogen.main -s street --viewer-screenshot shot.png --display uv     # the Qt viewer (lit|clay|normals|uv)

# Check from numbers: the resolved tree (world bounds, assets, meta) and every QA check
python -m geogen.main -s hotel_room_auto --inspect
python -m geogen.main -s hotel --inspect --depth 2 --checks none      # big scenes: --depth / --select GLOB, --json
python -m geogen.main check --project examples/sandwich                # a project's QA; exit 1 on findings
python -m geogen.main -s cottage --resolve "[tag=furniture]"           # what a reference names, as stable IDs

# The vocabulary: primitives, assets (params, attachments, surfaces, interactions), scenes, kinds
python -m geogen.main --describe [door car cube npc] [--json]
docs/make_catalogue.sh                                 # regenerate docs/catalogue (commit it); `all` re-renders thumbnails
python docs/make_site.py /tmp/site                     # the catalogue as the static site GitHub Pages serves

# Export
python -m geogen.main -s dining_set -e out/dining_set.glb [--lods 0.5,0.25]   # .glb / .gltf / .obj
python -m geogen.main -s town --chunks out/town_chunks --cache          # streamable chunks + index
python -m geogen.main -s cottage --export-godot [--stream]              # into runtime/godot/generated/
python -m geogen.main --export-catalogue [all|showcase|test|NAME,NAME]  # assets/runtime_scenes.yaml; --catalogue rewrites the index

# Godot runtime (macOS binary: /Applications/Godot.app/Contents/MacOS/Godot)
godot --path runtime/godot -- --scene cottage [--stream]                # no --scene: the catalogue's default
python -m geogen.runtime_client --launch cottage                        # control API; then e.g. player.walk_to pos=0,0.3,0.5
docs/make_screenshots.sh                                                # README screenshots (docs/images/)

pytest [tests/test_x.py -k name] [--fast | --all]                       # see Testing
```

## Where things are

Detailed notes per subsystem are in `docs/dev/`; read the one for the area you are changing before you change it. The contracts with runtimes are the schemas in `docs/schema/`, `docs/rules.md` (state, rules, saves, dialogue, items), `docs/quests.md` and `runtime/godot/README.md`.

| Area | Code | Notes |
|---|---|---|
| Scene graph, meshes, mesh ops, subdivision, skinning, skeletons, UV projection, 2D profiles; textures, materials | `src/geogen/core/`, `textures/`, `materials/` | `docs/dev/core.md` |
| Primitives and generators: extrude/lathe/sweep, text, paths, fences, water, terrain, stairs, roofs, CSG, floor plans, buildings, facades, plan layouts, nature; furniture library, interior finishes | `src/geogen/generators/`, `assets/` | `docs/dev/generators.md` |
| YAML assets and scenes: parts, attachments, slots, surfaces, params and `{expr}`, cities and recipes, scatter, references and stable IDs, named places, projects and content roots, registry, `--describe`, furnishing solver, layout and affordance QA | `src/geogen/layout/`, `refs.py`, `places.py`, `roots.py`, `registry.py`, `describe.py` | `docs/dev/layout.md` |
| Humanoids: skeleton, ring-lofted skin, poses, IK, clips, face, hair, clothing and drape, archetypes, clipping QA | `generators/humanoid.py`, `ringloft.py`, `clips.py`, `drape.py`, `arms.py`, `characters.py`, `clipping.py` | `docs/dev/characters.md`, `docs/clothing.md` |
| Vehicles, the traffic lane graph and signals, moving traffic, railways and trains, pedestrians | `vehicles.py`, `traffic.py`, `railway.py`, `trains.py` | `docs/dev/transport.md` |
| Interactions, NPCs, time of day, light fixtures, physics props, spawned items (flows); conditions, state, wiring, rules, content packages, quests, items, dialogue, HUD | `layout/interactions.py`, `npc.py`, `physics.py`, `flows.py`, `conditions.py`, `state.py`, `rules.py`, `wiring.py`, `content.py`, `quests.py`, `dialogue.py`, `inventory.py`, `hud.py` | `docs/dev/gameplay.md`, `docs/rules.md`, `docs/quests.md` |
| Offscreen render, glTF/OBJ export, manifests, extras contract, LODs, chunks and caching, the Qt viewer, the Godot runtime and its control API | `render.py`, `export.py`, `extras.py`, `chunks.py`, `viewer/`, `runtime/godot/` | `docs/dev/runtime.md`, `runtime/godot/README.md` |

Worked examples of projects built on geogen: `examples/sandwich`, `examples/test_chamber`, `examples/marble_run`, `examples/tower_defense`.

### Files and formats

- `assets/*.yaml` is an asset (`parts:`, `floorplan:`, `building:`, ...) or a composed scene (`place:`); `assets/scenes/*.yaml` and `assets/scenes/<name>/scene.yaml` (a scene folder with private files) are always scenes. A YAML with a top-level `kind:` (`player.yaml`, npcs, quests, items, ...) is data, not a scene. Python scenes are registered in `main._build_registry()` (`nature`, `skin_test`).
- Two composition levels: `LayoutLoader` builds an asset from primitives (parts connect with `attach_to` / `at` / `from`, roots use `anchor` + `offset`); `SceneComposer` places assets and nested scenes (slots, `attach_to` + `at`, `on: <object>.<surface>` + `at: {u, v}`, scatter). Connections use named points and surfaces, not coordinates; objects don't know their containers.
- Always load asset and scene YAML through `LayoutLoader` / `SceneComposer` / `yaml_utils`: `GeogenSafeLoader` keeps `on:` / `off:` / `yes:` / `no:` as strings, and plain `yaml.safe_load` turns `on:` into `True`.
- Anything named relative to `assets/` resolves with `roots.find(name, assets_dir)` (and lists with `roots.glob`), so projects' content roots are searched first. Don't join names onto an assets directory yourself.
- Things in scenes are named by references (`refs.resolve` / `refs.select`: `plot/house`, `[tag=door]`, `town:park/bench`). New content that names things stores references in YAML, resolves them at export and writes stable IDs (`<package>:<path>`) to extras or the manifest; the runtime only looks IDs up.

### Changes that need a matching change elsewhere

- **A new YAML key** goes in `layout/schema.py` with its meaning (validation's known keys come from it). An asset, key or header-comment change makes `docs/catalogue/` stale: run `docs/make_catalogue.sh` (`tests/test_describe.py` fails otherwise). A new markdown construct in `describe.py`'s pages needs handling in `docs/make_site.py`.
- **A new `SceneNode.meta` key** must be declared in `docs/schema/geogen-extras.v1.schema.json` (with its properties; objects are closed), get a row in the "Node extras" table of `runtime/godot/README.md` and an entry in `scene_builder.gd` `KNOWN_KEYS`, or be listed in `extras.BUILD_ONLY` with the reason. Otherwise the export raises `ExtrasError` (`tests/test_extras_contract.py` keeps the lists in step).
- **Rules semantics** (`rules.py`) have a GDScript port (`runtime/godot/scripts/machine.gd`, `conditions.gd`). Change both; the conformance traces in `tests/data/rules/` (edited via `make_traces.py`) catch a mismatch.
- **A new showcase scene** needs a hub portal and a way back (`tests/test_hub.py`). NPC affordances must be within 1.5 m of the NPC home's floor.
- **New HUD panels** join `GeogenHud` (`hud.gd`, `add_panel`), not `main.gd`.
- **The clipping baseline** (`tests/data/clipping_baseline.json`) is regenerated with `python tests/test_clipping.py`.

### Godot runtime

- Godot 4.7, Forward+, Jolt physics (write physics code for Jolt; don't fake physics with kinematic tracks). 1 unit = 1 m.
- Wrap ad-hoc runs in a timeout: a GDScript parse error leaves the process running, and macOS has no `timeout` (`perl -e 'alarm shift; exec @ARGV' 300 godot ...`). After adding a `class_name` script, re-import (`godot --headless --path runtime/godot --import`) before launching.
- Drive the runtime through the control protocol (`docs/schema/geogen-control.v1.schema.json`, runtime-neutral: Godot is the first of several runtimes): live with `--api` (`geogen.runtime_client`, fixture `godot_runtime`), or as a script of steps (`--script`, `runtime_client.run_script`, fixture `godot_script`). Write runtime tests as `godot_script` scenarios, not Godot flags. A new API command goes in `api.gd` `COMMANDS`, then `python docs/schema/make_control_schema.py`. Screenshots need the windowed default, not `--headless`.
- Runtime path queries go through `GeogenWalker.find_path` (no polygon search limit). A whole export's navmesh syncs in step with physics frames (reproducible simulations); streamed exports sync asynchronously.

## Key Conventions

- **UVs are metric** (1 UV unit = 1 m of surface). Generators must emit metric UVs (or run `uvmap.box_project`); materials set `tile_size`. Never emit 0–1-per-face UVs: texture density would vary with object size. (The one exception is the painted face of a humanoid head.)
- **Procedural textures must tile**: noise is periodic (`perlin_noise` wraps its lattice); `tests/test_textures_tile.py` checks every material.
- Generators produce closed meshes that pass `meshops.validate`; `tests/test_asset_quality.py` checks every registered asset and scene.
- **No coplanar surfaces** between different meshes: `qa.coplanar_overlaps(scene)` finds horizontal faces from different meshes that lie in the same plane, face the same way and overlap (they z-fight in every renderer). Undersides at or below y = 0 are ignored. Fix a hit by offsetting one surface 3 mm (`floorplan.COPLANAR_GAP`) or by cutting one away.
- **Triangle budgets:** round YAML primitives get at most one radial segment per `loader.SEGMENT_EDGE` (2.5 cm) of circumference, never fewer than 12; lathe profiles are simplified to 2% of their radius; cubes under 12 cm default to `bevel_segments: 1`. Instanced props multiply: check a big scene's total with instances (a street tree is ~4k triangles; the city ~3.5M).
- Cubes are rounded boxes (`bevel` 0.02 by default): set `bevel: 0` on thin flat surfaces (roads, sidewalks, slabs). For turned, curved or filleted parts prefer `lathe` / `extrude` / `sweep` over stacked primitives. YAML anchors and merge keys (`&leg` / `<<: *leg`) work for repeated parts.
- Counter-clockwise winding for outward normals; transforms are Scale -> Rotate (XYZ Euler, radians) -> Translate; geometry is centred at the origin by default; numpy `float64` vertices and normals, `int64` faces; materials are optional on a Mesh.
- Use `PlayerSpec` / `load_player_spec()` (`assets/player.yaml`) for player-scale clearances instead of hard-coding them.

## Testing

Always test changes by:
- Rendering a png of the update into /tmp and looking at it; if it's too small to see clearly, iterate until you get a good view.
- When you generate or update an object, render it from the front and side and make sure it looks correct and consistent.
- For runtime changes, export with `--export-godot` and screenshot the Godot view (`-- --scene X --screenshot=out.png`, add `--colliders` / `--camera=overview`); for headless movement checks use `--spawn` / `--yaw` with `--walk=SECONDS --status` (shorthand for script steps; each result prints as `script: {...}`), see `tests/test_godot_runtime.py`.
- `pytest` includes Godot runs (`tests/conftest.py` `run_godot` fixture); they skip if Godot isn't installed (`$GODOT` overrides the binary).

**Tests are expensive: limit runs.** Waiting on them is the biggest drain on a session.
- Any edit under `src/geogen` or `assets` invalidates the shared build cache, so the next default-tier run rebuilds and re-exports every scene (about a minute more).
- **Policy: skip the suite for ordinary tasks.** Commit without running pytest unless the work is significant (a broad refactor, a core module many things use, a risky behaviour change: your call per task). A render or one narrow test file is fine when it is the quickest way to know the change works.
- Run the default tier after every few tasks, in the background, and find what broke from git history if it fails. Don't re-run a tier after a change to docs, comments or tests it doesn't select. Run `--all`, or an exhaustive-tier file, only when the change touches what that tier covers.

**Tiers** (don't run more than the change needs):
- The tests for what you changed, by file: `pytest tests/test_expressions.py` (seconds).
- `pytest --fast`: every cheap test (about 20 s). It skips Godot runs and any test, or group sharing a build, that took 1 s or more when last timed (`--fast=0.25` quicker, `--fast=5` wider; a test never timed is kept).
- `pytest`: the default tier, about 3.5 minutes; everything except the exhaustive tier.
- `pytest --all`: adds the exhaustive tier (about 10 minutes): before a release, or after a change to city layout, traffic at scale, streaming or the showcases. It is tests marked `@pytest.mark.exhaustive` and anything that builds a scene in `conftest.EXHAUSTIVE_SCENES` (city, town, main_street, street, street_side, hotel_showcase, restaurant, archetype_gallery, people_gallery). A new slow test of a big scene belongs there; core behaviour needs a small scene in the default tier (`crossroads` for cities and traffic, `cottage`, `hotel`, `park`, `island`).

**Fixtures and resources:**
- `built_scene(name)` and `scene_export(name)` (a read-only `--generated` directory holding the scene's GLB) build once and keep the result in `.pytest_cache/geogen-built/<fingerprint>/`, shared by every worker and later runs until a source or asset file changes (GDScript and test edits don't invalidate it). Use them rather than building or exporting a registry scene in a test. `GEOGEN_TEST_CACHE=off` builds afresh; `pytest --cache-clear` empties it.
- `--durations=30` shows what is slow (timings are recorded per machine by every run).
- `-n auto` (`pyproject.toml`) means 8 workers for the default tier (about 2 GB) and 4 for `--all` (up to ~4 GB each; 14 froze a 24 GB Mac). Don't raise them. Ad-hoc runs of big scenes (city, town) should be serial (`-n0`).
- Render tests need a display: with the screen asleep pyglet finds no screen and `test_scene_renders` fails with an IndexError in `pyglet/canvas/base.py`.

## Task Tracking

Work is tracked in the project's room on bots.fallday.ca (`.fdbot.json`) with the `fdbot` CLI; use the `fdbot` skill. Run `fdbot prime` at the start of a session (room, project notes, your tasks, ready work).

```bash
fdbot task ready                      # Find available work
fdbot task show 1.2                   # Details (refs are geogen.1.2; the key is optional)
fdbot task start 1.2                  # Claim work
fdbot task close 1.2 --reason "..."   # Complete work
fdbot note add "..." --key <slug>     # Durable project knowledge, shown by prime
```

- Use the room's task list for all task tracking — not TodoWrite, TaskCreate or markdown TODO lists.
- Mention the task ref in commit messages, e.g. `Fix stair railing (geogen.6.18)`.
- Tasks imported from Beads keep their old id in the body ("Imported from Beads geogen-o3s.13"), so older commit messages that cite Beads ids can still be traced.
- Do not commit or push unless asked. At handoff, report changed files, validation, and suggested next commands.
- When you change a subsystem, update its notes in `docs/dev/` (not this file, unless it changes what every session needs to know).
