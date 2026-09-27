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

To export every runtime scene at once, use the scene catalogue
(`assets/runtime_scenes.yaml`): `python -m geogen.main --export-catalogue`
(or `showcase`, `test`, or `NAME,NAME`) exports its scenes and writes
`generated/catalogue.json`. Started without `--scene`, the runtime loads the
catalogue's `default` scene: the hub (`scenes/hub.yaml`), a plaza with a
portal to every showcase scene, each of which has a portal or travel door
back. Point `default` at whatever you're working on and run
`python -m geogen.main --catalogue` to rewrite the index.

## Switching scenes

F6 opens the scene list: the catalogue's scenes grouped showcase / test with
their descriptions (scenes not exported yet are greyed out). Picking one
unloads the current world (models, NPCs, traffic, trains, streamer,
navmesh, interactions) and loads the new one in place, with the player at
its first spawn; F7 / F8 step to the previous / next exported scene. The
last scene picked is remembered in `user://settings.cfg` (per generated
directory) and loads next time `--scene` isn't given (not in headless runs).
`--list-scenes` prints the list and quits; `--switch=NAME@S` switches
headlessly after S seconds and prints `switched: {...}` and, a few frames
later, `switch stats: {...}` (node, object, orphan, body and nav region
counts) so tests can check nothing is left behind.

Switching fades to a "Loading ..." screen and parses the new export's GLB on
a worker thread, so the window keeps drawing; the rest (materials, colliders,
navmesh) runs on the main thread. Baked navmeshes are cached as packed scenes
in `<generated>/.navcache/<name>-<key>.scn`, keyed by the export's content,
the player spec, the bake options and the baking scripts' source, so loading
an unchanged export again skips the bake (the town: 5.7 s -> 0.05 s).
Colliders share one shape per shared (instanced) mesh. Streamed exports bake
their own navigation tiles and aren't cached.

## Travel

Travel points (`geogen/travel.py`) are declared on assets or placements:

```yaml
place:
  hotel_door:
    asset: travel_door.yaml                # on: use (E on the leaf)
    params: { label: Hotel }
    travel: {scene: hotel_showcase, spawn: from_hub, prompt: Enter the hotel}
  to_town:
    asset: portal.yaml                     # on: enter (walk into its travel_volume)
    params: { label: Town }
    travel: {scene: town, spawn: from_hub}
  shortcut:
    asset: bookshelf.yaml
    travel: {spawn: attic}                 # no scene: move within this one (default on: use)
```

They're exported as `extras.geogen.travel` (`on: use` also gets a `travel`
interaction that emits `travel`, so aim + E works as for any interaction;
`on: enter` gets `volume: {center, size}`, turned into an Area3D). Using one
fades out, loads the target through the same path as the scene switcher (or
just moves the player for same-scene travel), puts the player at the named
spawn facing its way, prints `travelled: {...}` and fades back in. Walk-in
volumes stay quiet for a few physics frames after a load or move, so arriving
inside one doesn't bounce you back. The manifest lists every travel point
(`travel: [{node, scene, spawn, on}]`), and `--export-catalogue` fails if one
goes to a scene outside the catalogue or a spawn its (exported) target
doesn't have. Tests: `tests/test_travel.py`. Ready-made travel points:
`portal.yaml` (walk-in arch; `params: {label: Hotel}`) and `travel_door.yaml`
(press E on the leaf); `signpost.yaml` and `info_plinth.yaml` label the way.

Leave the game running and re-export: the runtime watches the manifests and
reloads the model within half a second (the player keeps their position).

Controls: WASD/arrows move, Shift sprint, Space jump, mouse look (click to
capture, Esc to release), F1 overlay, F2 collider wireframes, F3 fly/noclip
(Space/Ctrl up/down), F4 overview camera, F5 NPC labels, F6 scene list,
F7 / F8 previous / next scene, E use, L lock.

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
editor) the default scene from `generated/catalogue.json` loads. With
`--scene=all`, or when there's no catalogue, every export in `generated/`
loads, laid out in a row along +X so they don't overlap, and the player starts
in front of the first one. geogen exports a collider child per mesh named
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
godot --path runtime/godot                          # play the catalogue's default scene
godot --path runtime/godot -- --scene=all           # play every export in generated/ (slow)
godot --headless --path runtime/godot --import      # first run / CI: build the import cache
```

User args (after `--`):

| Arg | Effect |
|---|---|
| `--scene NAME` / `--scene=NAME` | Load `generated/NAME.glb` (default: `catalogue.json`'s default, else every export); `--scene=all` loads every export |
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
| `--list-scenes` | Print the scene catalogue (name, group, exported, description) and `scenes: {...}`, then quit |
| `--switch=NAME[@S]` | Switch to scene NAME after S s (default 1; repeatable); prints `switched:` / `switch stats:`, then quits unless `--walk` / `--status` / `--screenshot` follow |
| `--menu` | Open the scene list at start (for screenshots) |
| `--timings` | Print each load's phases as `load timings: {...}` (ms: glTF parse, materials, colliders, NPCs, traffic, navigation, ... and `nav_cache` hit/miss) |
| `--npc-trace` | Print every NPC decision (top-3 scores) and step as `npc: {...}` lines |
| `--timescale=N` | Run the world N times faster (physics ticks scale with it, so movement stays exact) |
| `--simulate=SECONDS` | Run SECONDS of world time, print `npc summary: [...]` and quit (with `--screenshot`, capture then quit) |
| `--npc-labels` | Show each NPC's current action and needs above it (F5 toggles) |
| `--camera=follow[:NAME]` | Watch an NPC (the first, or the one whose name starts with NAME) from a clear viewpoint |
| `--play=ANIM[@SECONDS]` | Loop every animation named `ANIM` or `<node>_ANIM` (skeletal clips, interaction animations); `@SECONDS` freezes it there |
| `--skeletons` | Print `skeletons: {...}` on quit: each Skeleton3D's bone positions, its skinned meshes' world bounds (CPU-skinned like the renderer) and the animations |

Skinned exports (glTF skins, e.g. `-s skin_test`) import as `Skeleton3D` +
skinned `MeshInstance3D` + `AnimationPlayer`; joint nodes keep their own
names (`Hips`, `Spine`, ...) in every character. Runtime `GLTFDocument`
resamples animations at 30 fps, so between those frames a pose differs
from the Python clip by the slerp-vs-chord error (a few mm on the test tube).
NPC bodies with skeletal clips (the humanoid) get their own AnimationPlayer
under the body, plus a `RESET` animation at the rest pose (the importer drops
tracks equal to the rest, the exported stand pose, so blends need it).
`npc.gd` cross-fades `pose_sit` / `pose_lie` when the NPC sits or lies, and
while standing plays `walk` (speed-scaled to the NPC's velocity over the
clip's recorded speed, so feet don't slide) or `idle`.
`tools/dump_skeleton_profile.gd` prints `SkeletonProfileHumanoid`
(saved as `assets/skeletons/humanoid_profile.json`).

`tests/test_godot_runtime.py` exports the cottage and walks the player into
it headless (wall blocks, door step is climbed, closed door blocks). The
`test_m3_*` tests stream the town district and walk into a shop and the
hotel lobby; `test_stream_*` check what loads at full, LOD and interior
detail as the player moves along a five-block street.

## NPCs

Scenes place NPCs like assets (`place: {resident: {npc: npcs/resident.yaml, on: house.floor, home: house.floor}}`).
Each one arrives as a node with `extras.geogen.type = "npc"`, the fully resolved definition in
`extras.geogen.npc`, and its body asset as a child. `WorldLoader` turns it into a `GeogenNpc`
(`scripts/npc.gd`): a `CharacterBody3D` (cylinder, layer 2, with the player's step-up) with the
body reparented under it. The script is a generic interpreter; nothing in it knows about chairs,
windows or doors:

- **Needs** fall at their declared rates. Finishing something adds its `advertises` to them.
- **Deciding**: every affordance in the NPC's home (+ `home_margin`) with a free slot, plus its
  own activities (wander, idle), scores
  `sum((1 - need) x advertised) x preference(tags) - distance x m - recency x e^(-age/memory) + noise x rand`,
  with the weights from the definition's `scoring`. The best one is reserved and run. Failures
  aren't retried for `scoring.retry` seconds.
- **Running**: the action's steps (`assets/npcs/actions.yaml`): `go_to` (navmesh path),
  `face`, `pose` (the body asset's `poses:`, placed at the affordance anchor), `wait`, `use`
  (drives an interaction to a state, as the player's E does), `play` (a body clip, once).
- **Doors**: exported `portal`s. When the path ahead crosses a portal whose interaction isn't
  open, the NPC runs the `pass` action first: stop clear of the leaf, open it, walk through,
  and close it if the NPC's `closes_doors` flag is set.
- **Traffic** (vehicles in group `geogen_vehicle`, see Traffic) isn't in the navmesh either.
  - An NPC waits at the kerb for a moving vehicle whose next 3 s of travel crosses its next steps.
  - After 3 s it steps out anyway (vehicles yield to people in their way).
  - It walks round a stopped vehicle past the nearer end: in front of a car that stopped for it, behind one that's queuing.
- **Other characters** (the player, other NPCs; group `geogen_character`) aren't in the
  navmesh, so paths bend around them on an arc of navmesh points. An NPC whose destination is
  occupied waits, then gives up.

- **The player** (the definition's `attention:`, on by default; `false` or a part `false` turns it off):
  - *look*: within `range` m and a `cone` ahead, a standing or seated skeletal NPC turns its neck
    and head toward the player's eyes, clamped to `yaw`/`pitch` degrees and eased
    (`scripts/head_look.gd`, a `SkeletonModifier3D` run after the clips).
  - *greet*: aiming at a standing NPC shows `E: <prompt>`. E runs its `greet` action on top of
    whatever it was doing: `face: player`, `play: wave` (the body's one-shot wave clip), a pause.
    Then the interrupted step starts again. It won't greet again for `cooldown` s.
  - *yield*: an NPC standing within `distance` m of the player walking straight at it runs
    `step_aside` (`go_to: aside`, `step` m off the player's line, then `face: player`).
  - The summary reports `greets`, `yields` and `looking` (seconds). `--greet=NAME[@S]` greets
    headlessly.

Pedestrians cross at crosswalks. The lane graph exports the painted crossings (`traffic.crosswalks`). A planned path that runs along road level inside the streets is rerouted over the crossing that makes the shortest detour: to one end, straight across, then on. The detour must be no longer than twice the direct path plus 25 m.

Crowds are scatters of an NPC (`npcs/pedestrian.yaml` in `town` and `crossroads`), each copy a
different person. `affordance_tags` limits which affordances an NPC considers. `wander.tags` makes
strolls go to random points on surfaces in those tag groups (sidewalks) rather than anywhere in
the home region.

The navmesh includes the runtime's ground plane around each model (`WorldLoader.ground_margin`),
so NPCs (and the playtest) can step outside. `tests/test_npc_runtime.py` simulates the cottage
resident at 8x and asserts on the summary: which affordances were used, door passes, stalls,
failures, and time spent outside home.

```bash
godot --path runtime/godot -- --scene cottage --camera=follow --npc-labels          # watch the resident
godot --headless --fixed-fps 60 --path runtime/godot -- --scene cottage \
    --timescale=8 --simulate=600 --npc-trace                                         # 10 minutes in ~20 s
```

## Traffic

Scenes with streets (`city: {traffic: ...}`) or `routes:` export a lane graph in the manifest's
(or chunk index's) `traffic` section (`docs/schema/geogen-traffic.v1.schema.json`). It contains
directed lanes resampled every 0.5 m with curvature-capped speeds, connectors through
intersections, conflict zones and crosswalk ranges. A traffic placement
(`place: {traffic: {traffic: traffic/town.yaml, seed: 3}}`) arrives as a node with
`extras.geogen.type = "traffic"` and the fleet's driving parameters in `extras.geogen.fleet`.
Its children are the starting vehicles, each on a lane (`extras.geogen.driving = {lane, s, factor}`).

`WorldLoader` hands both to a `GeogenTraffic` (`scripts/traffic.gd`). It's a generic interpreter
that keeps each vehicle as (lane, s, v), with the axles on the curve so it turns like a car:

- **Car following**: the Intelligent Driver Model against the vehicle ahead on its lane or its
  next lane. It slows in time for tighter connectors.
- **All-way stops**: vehicles stop at the stop line, wait `stop_wait`, and claim their connector
  once no conflicting movement is occupied or claimed. They're served first come, first served.
  They don't enter unless the lane beyond has room ("don't block the box"). After `give_up`
  seconds a vehicle picks another exit.
- **Traffic lights**: signalled intersections (`control: signals`) cycle their phases from world
  time (green, amber, all-red; the phases take turns). Vehicles go on green, or on amber if too
  close to stop, and still claim conflict zones, so turns yield. They wait at red. Signal heads
  (`extras.geogen.signal`) light the lamp for their phase.
- **Yielding**: the player and NPCs (group `geogen_character`) in the corridor ahead, or on a
  crosswalk about to be crossed, are obstacles to stop for.
- **Bodies**: vehicles are `AnimatableBody3D` boxes. They block the player, stay out of the
  navmesh, and never push anything. Wheels spin at v / r. Vehicles that drive off an open route
  come back at a route start.

`--simulate=S` also prints `traffic summary: [...]`: vehicles, distance, overlaps, idle and wait
times, claims, turns, yields and the closest stop to a person. `--traffic-trace` prints claims,
replans, respawns and overlaps. `tests/test_traffic_runtime.py` covers flow without overlaps or
gridlock, a player in the lane stopping traffic, determinism, and open routes.

```bash
python -m geogen.main -s crossroads --export-godot
godot --path runtime/godot -- --scene crossroads --camera=overview                   # watch it
godot --headless --fixed-fps 60 --path runtime/godot -- --scene crossroads \
    --timescale=8 --simulate=300 --traffic-trace                                     # 5 minutes in ~10 s
```

## Trains

Railways (a scene's `railways:` block, geogen/railway.py) are exported in the manifest's
`traffic.railways` section:
- the line resampled at the rail head;
- its length, loop flag and speed;
- `stations` (s, length);
- `crossings`, each with an id, the rail s-range to guard, barrier node names, and `lanes`, which are the road lanes and the s where vehicles wait.

A train placement arrives as a node with `extras.geogen.type = "train"`, its consist as children,
and `extras.geogen.train`, all precomputed by geogen/trains.py:
- `run`: the head's position every `dt` seconds after departure, station dwells included;
- `duration`, `stops`, `timetable` {headway, dwell, offset};
- `cars` (offset from the front, bogie positions);
- `closures`: each crossing's closed windows after departure.

`GeogenTrain` (`scripts/train.gd`) only looks these up.
- **Timing**: departure k leaves at `offset + k * headway` of world time (the clock's position in
  its day), so a train is where the timetable says whenever the world is loaded.
- **Placement**: each departure on the line gets a copy of the consist. Each car's bogies sit on
  the track, the body goes between them, the bogies turn to the rails, and the wheels spin.
- **Crossings**: during a closure window the crossing is closed. Its barriers' `barrier`
  interaction goes `down`, and road traffic (traffic.gd) waits at the lanes' stop points unless
  a vehicle is already past them.

Vehicles with `vehicle.sound` get synthesised audio from `scripts/sound.gd` (no files). The engine is a looping hum whose pitch follows speed, and trains sound their `horn` when a crossing closes ahead. `--simulate` prints `train summary`. `-s level_crossing` is the test scene; see
`tests/test_traffic_runtime.py` (crossings closed with no vehicle inside, timetable dwell).

## Time of day

`scripts/clock.gd` (`GeogenClock`) runs a world clock with physics time, so `--timescale` speeds it
up. `--time=HH:MM` sets the start (default 13:00) and `--day-length=S` sets the real seconds per
24 h (default 1440; 0 freezes it). The overlay shows the time.

- **Sun, sky and moon**: the sun's elevation, azimuth, colour and energy follow the hour, with
  warm light low in the sky. The sky and fog fade through a pink twilight to a blue night, and a
  faint shadowless moonlight keeps shapes readable.
- **Night**: fixtures exported with `light.auto: night` switch on after dark, together with
  their glowing glass. Street lamps are these, and have no shadows since there are many.
  Vehicle head and tail lamps brighten. `WorldLoader.set_night` handles the switch.
- **NPC routines**: definitions take `routine:` blocks `{from, to, activities, preferences, away}`.
  The active blocks multiply activity and tag scores. An `away` block runs the `leave` action:
  `go_to: exit` (the nearest building entrance spawn), then `vanish` (hidden, but not while the
  player is within 12 m). When the block ends the NPC reappears at an entrance or on its wander
  surfaces.
- **Traffic schedule**: a fleet's `schedule: [["HH:MM", share], ...]` (linear, wrapping) sets how
  much of it is on the road. Surplus vehicles park, and parked ones return, only 45 m or more
  from the player.

`--status` reports `clock`, `night` and `lamps_on`, and the NPC and traffic summaries include
`away` / `returns` and `share` / `parked`. See the night and morning tests in
`tests/test_traffic_runtime.py`.

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
