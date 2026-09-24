# Geogen Godot runtime

Reference runtime for geogen exports (glTF + `extras.geogen`). See the beads
under epic `geogen-3cc` for the roadmap (player, world loader, import plugin).

- **Godot version:** 4.7-stable (pinned; `config/features` in `project.godot`)
- **Renderer:** Forward+
- **Units:** 1 unit = 1 metre, +Y up (same as geogen and glTF)

## Quick start

```bash
# 1. Export a scene into runtime/godot/generated/ (.glb + manifest)
python -m geogen.main -s cottage --export-godot

# 2. Walk around it
godot --path runtime/godot -- --scene cottage
```

Leave the game running and re-export: the runtime watches the manifests and
reloads the model within half a second (the player keeps their position).

Controls: WASD/arrows move, Shift sprint, Space jump, mouse look (click to
capture, Esc to release), F1 overlay, F2 collider wireframes, F3 fly/noclip
(Space/Ctrl up/down), F4 overview camera.

## Layout

| Path | Purpose |
|---|---|
| `project.godot` | Project settings; main scene is `scenes/main.tscn` |
| `scenes/main.tscn` | Sky, sun, fog, 400 m ground with collision, `World` loader, overview camera, overlay |
| `scripts/main.gd` | Root: parses user args, spawns the player, debug overlay |
| `scripts/world_loader.gd` | `WorldLoader`: loads `.glb` exports at runtime via `GLTFDocument`, builds static bodies from the exported collider nodes, collects room volumes (`room_at()`) and manifest spawns, adds texture mipmaps, hot-reloads |
| `scripts/chunk_streamer.gd` | `GeogenChunkStreamer`: streams a chunked export around the player (see below) |
| `scripts/player.gd` | `Player`: first-person `CharacterBody3D` sized from the player spec, with step-up |
| `scripts/player_spec.gd` | `PlayerSpec`: player radius/height/eye/step/slope/reach read from a geogen manifest |
| `addons/geogen/` | Editor plugin + `GeogenSceneBuilder`: turns `extras.geogen` into room `Area3D`s (`RoomArea`, group `geogen_room`), spawn `Marker3D`s (group `geogen_spawn`), tag groups (`furniture.bed` → `furniture.bed` + `furniture`) and a baked `NavigationRegion3D`; applied on editor import (post-import plugin) and by `WorldLoader` at runtime |
| `generated/` | Exports land here (git-ignored; `.gdignore` keeps the editor from importing them, the runtime loads them directly) |

Exports are loaded at runtime rather than imported by the editor so a
running game can reload them. Without `--scene` (e.g. pressing Play in the
editor) every export in `generated/` loads, laid out in a row along +X so
they don't overlap, and the player starts in front of the first one. geogen exports a collider child per mesh named
with Godot's import suffixes (`<name>-colonly` = trimesh, `<name>-convcolonly`
= box/convex hull); runtime glTF loading doesn't apply those suffixes, so
`WorldLoader` turns them into `StaticBody3D`s and drops their meshes (older
exports without collider nodes get a trimesh collider per mesh). Node extras
(`extras.geogen`, schema in `docs/schema/geogen-extras.v1.schema.json`) arrive
as `get_meta("extras")`; room volumes drive the overlay's `room:` readout and
the `room` field of `--walk` results. Without `--spawn`, the player starts at
the manifest's first spawn point. Interactions in node extras (e.g. the
cottage door's `swing`) become `GeogenInteraction` nodes (`scripts/interaction.gd`):
look at the door within reach and press E to open or close it (L locks/unlocks with a held key; timed states like self-closing guest doors advance on their own); colliders on
moving parts are `AnimatableBody3D`s so the open door lets you through. Looking at furniture with affordances (chairs, sofas, the bed) offers E: Sit / Lie down; E or walking stands up. The
player body is a
cylinder, not a capsule: a capsule's rounded bottom slides off the edge of
a step exactly `step_height` tall.

## Running

On macOS the binary is `/Applications/Godot.app/Contents/MacOS/Godot`; below it
is written as `godot`.

```bash
godot --editor --path runtime/godot                 # open in the editor
godot --path runtime/godot -- --scene cottage       # play one export
godot --path runtime/godot                          # play every export in generated/
godot --headless --path runtime/godot --import      # first run / CI: build the import cache
```

User args (after `--`):

| Arg | Effect |
|---|---|
| `--scene NAME` / `--scene=NAME` | Load `generated/NAME.glb` (default: every export) |
| `--generated=DIR` | Read exports from `DIR` instead of `res://generated` |
| `--stream` | Prefer the scene's chunked export (`generated/NAME_chunks/`) when a single-file one exists too |
| `--stream-radius=F,L,I` | Streaming radii in metres: full exterior, LOD, interiors (default 60, 400, 14) |
| `--spawn=X,Y,Z`, `--yaw=DEG` | Player start (default: 3 m in front (+Z) of the model, facing it) |
| `--camera=overview` | Start on the overview camera |
| `--colliders` | Show collider wireframes |
| `--walk=SECONDS` | Walk forward, print `walk result: {...}` and quit (used by tests) |
| `--use=ASSET`, `--use=@aim` | Use an asset's interactions (e.g. `door`) or whatever the player looks at, at start |
| `--nav=AX,AZ:BX,BZ` | Print the navigation path between two floor points (`nav path: {...}`) and quit |
| `--keys=K1,K2` | Keys the player holds; L locks/unlocks a focused door that takes one |
| `--lock=@aim` | Press L on whatever the player looks at |
| `--pitch=DEG` | Look up (+) / down (-) at spawn |
| `--save=PATH`, `--load=PATH` | Write interaction state (doors, drawers, switches, locks) on quit / restore it at start |
| `--status` | Print the player's pose, room and interaction states on quit |
| `--lights` | Print fixture lights on quit |
| `--wait=SECONDS` | Delay the `--walk` (let a door finish swinging) |
| `--screenshot=PATH` | Save a frame and quit (needs a GPU, not `--headless`) |
| `--quit-after=N` | Quit after N frames |
| `--manifest=PATH` | Use the player spec from this manifest |

`tests/test_godot_runtime.py` exports the cottage and walks the player into
it headless (wall blocks, door step is climbed, closed door blocks). The
`test_m3_*` tests stream the town district and walk into a shop and the
hotel lobby; `test_stream_*` check what loads at full, LOD and interior
detail as the player moves along a five-block street.

## Streaming large scenes

`python -m geogen.main -s town --export-godot --stream` writes a chunked
export to `generated/town_chunks/`:
- one GLB per city block or building;
- a decimated `_lod` stand-in for each;
- one `_interior` GLB per building;
- `base.glb`, holding the streets and everything else;
- a shared `textures/` folder;
- the `town.chunks.json` index (`geogen-chunks` v1).

`--scene town` loads it through `GeogenChunkStreamer`:
- `base` stays loaded.
- A chunk's full exterior loads within the full radius of its bounds, and its LOD out to the LOD radius. The LOD stays until the full version is in, so nothing pops.
- A building's interior loads when the player is within the interior radius of it.
- Pieces unload 10 m further out than they load.

Loading runs as follows:
- `prime()` loads what the spawn needs synchronously.
- After that, GLB parsing and mipmap generation run on the `WorkerThreadPool`; the main thread adds the nodes and runs `WorldLoader.setup_root`.
- An unloaded chunk is unregistered with `WorldLoader.forget`.

Navigation comes as 24 m tiles within 30 m of the player, baked from the loaded pieces that overlap them:
- Each tile is clipped to its own box with a border, so neighbouring tiles join edge to edge.
- Parsing happens on the main thread and baking runs async.
- A tile re-bakes when a chunk or interior overlapping it loads or unloads.
- `--nav` queries wait until the nearby tiles are baked.

The overlay shows full / LOD / interior counts, and `--walk` / `--status` results carry a `stream` report (including `nav_tiles`). `--playtest` still needs a single-file export, since it checks every room at once.

## Manifest

Every glTF export from geogen writes `<name>.manifest.json` next to the model:

```json
{"format": "geogen-manifest", "version": 1, "name": "chair", "model": "chair.glb",
 "units": "m", "up": "+Y",
 "player": {"kind": "player_spec", "version": 1, "radius": 0.3, "height": 1.8, "...": "..."}}
```

`player` comes from `assets/player.yaml` in the geogen repo, the single source
of truth for player scale. `tests/test_player.py::test_godot_reads_manifest`
checks that this runtime reads it back exactly.
