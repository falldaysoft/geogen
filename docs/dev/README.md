# Developer notes

How each part of geogen works, for whoever changes it next. `CLAUDE.md` at the repo root is the short version every session reads: commands, conventions, testing policy and the changes that need matching changes elsewhere. The contracts with runtimes are `docs/schema/`, `docs/rules.md`, `docs/quests.md` and `runtime/godot/README.md`.

| File | Covers |
|---|---|
| [core.md](core.md) | Scene graph, meshes and mesh ops, subdivision, skinning, skeletons, UV projection, 2D profiles; textures and materials |
| [generators.md](generators.md) | Primitives and generators (profiles, sweeps, text, terrain, stairs, roofs, CSG, floor plans, buildings, facades, nature); the furniture library and interior finishes |
| [layout.md](layout.md) | Asset and scene YAML: attachments, slots, surfaces, params and expressions, cities, scatter, references, named places, projects, the registry, `--describe`, furnishing and layout QA |
| [characters.md](characters.md) | Humanoid bodies, clips, hair, clothing, archetypes, clipping QA (with [../clothing.md](../clothing.md)) |
| [transport.md](transport.md) | Vehicles, the traffic lane graph and signals, moving traffic, railways and trains, pedestrians |
| [gameplay.md](gameplay.md) | Interactions, NPCs, time of day, light fixtures, physics props, spawned items; conditions, state, rules, wiring, content packages, quests, items, dialogue, HUD |
| [runtime.md](runtime.md) | Rendering, export, extras, chunks and caching, the Qt viewer, the Godot runtime |

## Packages

| Package | Contents |
|---|---|
| `core/` | `SceneNode` hierarchy (with instancing), `Mesh` (material groups, vertex colours), `Transform`; `meshops` (normals, welding, tangents, decimation, `validate`), `uvmap` (metric projections), `profile` (2D shapes), `csg` (manifold3d booleans), `subdiv` (Loop subdivision, displacement) |
| `generators/` | Primitives, profiles, sweeps, stairs, roofs, doors, floor plans and plan layouts, multi-storey buildings, façades, interior finishes, trees and rocks |
| `textures/`, `materials/` | Tiling procedural texture generators; PBR materials loaded from `assets/materials/*.yaml` with a metric `tile_size` |
| `layout/` | `LayoutLoader` (assets) and `SceneComposer` (scenes); anchors, attachments, surfaces, the `{expr}` evaluator; interactions, the furnishing solver, layout QA, scatter, city layout, building recipes, the prototype cache |
| `viewer/` | Qt window, OpenGL view (shadows, PBR, debug display modes, picking, sections, night mode), pure-numpy orbit camera |
| `render.py`, `export.py`, `chunks.py` | Offscreen pyrender renderer; glTF/GLB/OBJ export (colliders, extras, lights, LODs, animations) and the runtime manifest; chunked export |
| `player.py` | `PlayerSpec`, loaded from `assets/player.yaml` |
| `characters.py`, `clipping.py` | Character archetypes; clothing and seat clipping QA (the humanoid generator, ring lofting and clips are in `generators/`, and the skeleton, skinning and IK in `core/`) |
| `npc.py` | NPC definitions, actions, routines and attention |
| `vehicles.py`, `traffic.py`, `railway.py`, `trains.py` | Vehicle metadata, lane graphs and signals, railways and level crossings, train timetables |
| `catalogue.py` | The runtime scene catalogue (`assets/runtime_scenes.yaml`) and its `catalogue.json` index |
| `describe.py` | `--describe`: the vocabulary (primitives, assets, scenes, kinds) as text, JSON and the [catalogue](../catalogue/README.md) |
| `registry.py` | Discovers YAML assets/scenes and Python-coded scenes |
| `runtime/godot/` | Godot 4.7 project: runtime glTF loading, first-person player, interactions, playtest, chunk streaming, NPCs, traffic, trains, the world clock |

## Tests

```bash
pytest                                   # everything
pytest tests/test_scenes.py -k chair     # a subset
```

The tests cover assets, textures, exports and (when Godot is installed) the runtime:
- Every asset and scene's meshes are valid, and every scene passes layout QA (including the z-fighting check).
- Textures tile, and exports round-trip.
- The Godot runtime is run headless: the player walks into the cottage (the wall blocks, the door step is climbed, the closed door blocks), rides the hotel lift, and playtests the hotel from the street.
- In the streamed town, the tests walk the avenue into a shop and the hotel lobby.
- NPCs, traffic, pedestrians and trains are simulated headless: residents meet their needs, groups stay together, traffic yields and doesn't deadlock, and trains close the level crossings in time.
- Dressed characters are checked for clipping against a recorded baseline.

The Godot tests are skipped when Godot isn't found.

`docs/make_screenshots.sh` regenerates the screenshots in the README and [the gallery](../gallery.md). `CLAUDE.md` has the test tiers (`--fast`, default, `--all`) and the policy on when to run them.
