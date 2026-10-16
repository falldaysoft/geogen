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
back. Its Quests portal leads to the quest yard (`scenes/quest_yard.yaml`):
three quests for trying the quest HUD (see "Trying them" in `docs/quests.md`).
Point `default` at whatever you're working on and run
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
F7 / F8 previous / next scene, E use, L lock, Q track the next active quest,
J quest log.

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
look at the door within reach and press E to open or close it (L locks/unlocks with a held key; timed states like self-closing guest doors advance on their own; an interaction with a `when` offers nothing while the rules program says it is false, `GeogenRules.allows`); colliders on
moving parts are `AnimatableBody3D`s so the open door lets you through. A `visible` motion shows and hides
parts (`show: [states]` / `hide: [states]` in the asset): a part shows as soon as the interaction heads for a
state that shows it and hides on arriving at one that doesn't, with its colliders disabled while hidden. glTF has
no node visibility, so other engines see hidden parts unless they read the interaction. Looking at furniture with affordances (chairs, sofas, the bed) offers E: Sit / Lie down; E or walking stands up. The
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
| `--nav=AX,AZ:BX,BZ` | Print the navigation path between two floor points (`nav path: {...}`) and quit; `X,Y,Z` points snap from that height (terrain), and it waits for tiled navmeshes to settle |
| `--keys=K1,K2` | Keys the player holds; L locks/unlocks a focused door that takes one |
| `--lock=@aim` | Press L on whatever the player looks at |
| `--pitch=DEG` | Look up (+) / down (-) at spawn |
| `--save=PATH`, `--load=PATH` | Write the game's state on quit / restore it at start (before `--set` / `--quests`): a `geogen-save` (see "Saving" below) of every package visited. Keyed by stable ID (`cottage:door#swing`), so a save survives a re-export; entries for chunks that aren't loaded apply when they load |
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
| `--api[=PORT]` | Serve the control API on `127.0.0.1:PORT` (default 7878, 0 = a free port); see [Control API](#control-api) |
| `--skeletons` | Print `skeletons: {...}` on quit: each Skeleton3D's bone positions, its skinned meshes' world bounds (CPU-skinned like the renderer) and the animations |

Skinned exports (glTF skins, e.g. `-s skin_test`) import as `Skeleton3D` +
skinned `MeshInstance3D` + `AnimationPlayer`; joint nodes keep their own
names (`Hips`, `Spine`, ...) in every character. Runtime `GLTFDocument`
resamples animations at 30 fps, so between those frames a pose differs
from the Python clip by the slerp-vs-chord error (a few mm on the test tube).
NPC bodies with skeletal clips (the humanoid) get their own AnimationPlayer
under the body, plus a `RESET` animation at the rest pose (the importer drops
tracks equal to the rest, the exported stand pose, so blends need it).
When the NPC changes pose, `npc.gd` plays the body's transition clip for that
change (extras `from` / `to`: `sit_down`, `stand_up`, `lie_down`, `get_up`)
and moves the body along the clip's `root` path, so the feet stay where they
stood while the hips go to the seat; the path was made for the default approach
and seat height, and the difference to the actual ones is blended in over the
clip. Then it holds `pose_sit` / `pose_lie`. Bodies without a transition clip
cross-fade into the pose clip instead. While standing it plays `walk`
(speed-scaled to the NPC's velocity over the clip's recorded speed, so feet
don't slide) or `idle`.

`--simulate` also prints `npc cost: {npcs, script_us, physics_ms, physics_p95_ms,
process_ms, frame_ms, frame_p95_ms, frame_max_ms, draw_calls, primitives, objects}`
over the simulated time only: mean NPC script time per physics frame (also per NPC as
`cost_us` in the summary), every physics tick's callbacks timed directly (mean and 95th
percentile: all scripts and `move_and_slide`, not the physics server's own step), the
frame time (mean, 95th percentile, worst) and what was rendered per frame. Frame and
render numbers need a window and `--disable-vsync`. `process_ms` is Godot's TIME_PROCESS
monitor, which (like TIME_PHYSICS_PROCESS) is the slowest frame of the last second, so
don't read it as a mean.

#### Crowd budget

`-s crowd_test` is the crossroads grid with 50 pedestrians and its traffic
(geogen.11.5.17). `--npc-limit=N` spawns only the first N of an export's NPCs,
so one export gives the whole curve:

```bash
python -m geogen.main -s crowd_test --export-godot
godot --path runtime/godot --disable-vsync --resolution 1920x1080 -- \
    --scene crowd_test --simulate=30 --npc-limit=50 [--camera=overview]
```

Target: 50 walking NPCs at a sustained 60 fps, with the physics tick within 6 ms
so traffic, the player and rules have room. Measured 2026-10-15 on an M-series
Mac, 1920x1080, player view from the spawn (physics: the tick's callbacks, mean / p95):

| NPCs | before: script us / NPC | before: physics ms | now: script us / NPC | now: physics ms |
|---|---|---|---|---|
| 0  | -   | 0.15        | -  | 0.15        |
| 10 | 73  | 0.94 / 1.3  | 39 | 0.65 / 0.76 |
| 25 | 96  | 2.7 / 4.0   | 40 | 1.3 / 1.6   |
| 50 | 186 | 9.8 / 13.5  | 41 | 2.4 / 3.0   |

The cost per NPC grew with the crowd because every NPC scanned every character,
and moved its whole remaining path's waypoints off each one, every frame
(geogen.11.18). Now characters are gathered once a physics frame into a 2 m grid
(`npc.gd` `_characters`, `_characters_in`), waypoints are moved off characters
only over the next `DETOUR_HORIZON` (8 m) of path, and NPCs off screen (more than
`THROTTLE_NEAR` 10 m from the player) or beyond `THROTTLE_FAR` (60 m) step every
other frame over both frames' time (`_throttled`; not mid-transition, on a stair or
greeting; the summary counts `throttled` frames). About half of what is left is
`move_and_slide`.

Rendering is cheap: every NPC in view adds about 7 draw calls and 4.6k
primitives, shadows included (a dressed character is about 3.9k triangles in 6
surfaces). Draw calls at 50 NPCs ranged from 570 (overview) to 1200 depending on
who was in view; frames averaged 4.5 to 7 ms (p95 7 ms). Every run, with or
without NPCs, has one hitch of about 150 ms.

`tools/dump_skeleton_profile.gd` prints `SkeletonProfileHumanoid`
(saved as `assets/skeletons/humanoid_profile.json`).

`tests/test_godot_runtime.py` exports the cottage and walks the player into
it headless (wall blocks, door step is climbed, closed door blocks). The
`test_m3_*` tests stream the town district and walk into a shop and the
hotel lobby; `test_stream_*` check what loads at full, LOD and interior
detail as the player moves along a five-block street.

## Control API

`--api[=PORT]` (`scripts/api.gd`) keeps one world running and answers newline-delimited
JSON on a localhost socket, so tools, tests and agents ask many questions of one process
instead of launching one per `--walk` / `--nav` / `--status`. Requests are
`{"id", "cmd", "args"}`, replies `{"id", "ok", "result" | "error"}`; after
`events.subscribe` the connection also gets `{"event", "data"}` lines (streams `events`: the whole event bus, `interaction`,
`npc` traces, `room` enter/exit, `place` enter/leave by any actor, `travel`, `traffic` incl. trains). Points are `[x, y, z]`
metres, angles degrees. `help` lists every command; the groups are:

- `world.*`: `info`, `scenes`, `query` (objects by tag / type / name glob / room), `node`
  (transform, world bounds, extras, colliders, children, interactions with part bounds),
  `room_at`, `raycast` (hit node, asset, room, interaction), `nav_path`, `switch`, `reload`
  (re-read the export after re-exporting; the player stays put).
- `player.*`: `status` (pose, room, what's aimed at), `teleport`, `look` (`yaw`/`pitch` or `at`),
  `move`, `walk_to` (follows the navmesh with the real body, `GeogenWalker`, going round a spiral stair by its walking line where the path only has the stair link's two ends; returns status
  arrived/stuck/timeout, a 0.25 s position `trace` and `blocked_by`: the node, asset and
  interaction state of whatever stopped it), `use` (aimed, or a named asset), `sit`, `stand`, `keys`.
- `interactions.*`: `list`, `set` / `use` (`wait: true` returns once the parts arrive, with their
  world bounds), `lock`; `state.save` (`path` also writes the file) / `state.load` (`state` or `path`).
- `npcs.*`: `list` (reports), `options` (scored), `force` (the next decision takes an option),
  `greet`, `pause`, `resume`.
- `time.*`: `pause` / `resume` (the API keeps serving), `step` (exactly N physics frames),
  `advance` (S seconds of world time at a high scale, then NPC reports), `scale`, `clock`.
- `capture.*`: `screenshot` (current camera, a `from`/`at` pose, or framed on a `node` from a
  `view`: iso, front, back, left, right, top; `isolate` hides everything else; `zoom`), `views`
  (a contact sheet of those), `camera` (player / overview / follow an NPC). Needs a window, not
  `--headless`.
- `hud.quests`: what the HUD's quest tracker displays (see "HUD"); `track` and `linger` change it first.
- `hud.log`: the HUD's quest log (see "HUD"); `open`, `select` and `track` change it first.
- `hud.markers`: the HUD's world markers (see "HUD"): each one's kind, world and screen position, distance and whether it is clamped to the edge.
- `hud.notifications`: the HUD's quest toasts, on screen and waiting (see "HUD"); `hold`, `gap`, `limit`,
  `muted` and `clear` change them first.
- `input.key` (press a key by name, as the keyboard would), `window.size` (get or set it; headless
  runs start at 64 x 64, so set a size before checking a layout).
- `batch` (several commands, one reply), `wait_ready` (loaded and the navmesh answers), `quit`.

Python: `geogen.runtime_client.RuntimeClient` (`launch(scene, generated=...)`, `call(cmd, **args)`,
`wait_event(predicate)`); tests use the `godot_runtime` fixture (`tests/test_runtime_api.py`).
From a shell, keep one runtime in the background:

```bash
python -m geogen.runtime_client --launch cottage [--generated DIR] [--headless] [-- runtime args]
python -m geogen.runtime_client player.walk_to pos=0,0.3,0.5          # key=value or one JSON object
python -m geogen.runtime_client capture.views node=bookshelf isolate=true path=/tmp/shelf.png
python -m geogen.runtime_client --watch npc,interaction
python -m geogen.runtime_client --log | --stop
```

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
  occupied waits, then gives up. Only characters on the NPC's own level count (within 1.2 m in
  height: not someone a storey up, or a turn of a spiral stair above). An NPC held up for 1.5 s
  on such a way round plans its path again (twice per `go_to` at most).
- **Stairs**: a path through a spiral stair's navigation link has only the link's two ends, so
  `_plan` puts the stair's walking line in their place (`GeogenWalker.along_links`, as the
  player's `walk_to` does). On that line nothing is moved aside or walked around, and it is
  taken in single file: an NPC arriving within 2 m of a stair that another NPC is on, coming
  the other way, waits there until it is clear (`stuck: stair in use` after 45 s). Test scene:
  `lighthouse_keepers`, two NPCs who start at the wrong ends of the lighthouse stair.

- **The player** (the definition's `attention:`, on by default; `false` or a part `false` turns it off):
  - *look*: within `range` m and a `cone` ahead, a standing or seated skeletal NPC turns its neck
    and head toward the player's eyes, clamped to `yaw`/`pitch` degrees and eased
    (`scripts/head_look.gd`, a `SkeletonModifier3D` run after the clips).
  - *greet*: aiming at an NPC that is here (not `away`) shows `E: <prompt>`, in any pose and
    whatever it is doing. A seated or lying NPC has no collider, so the aim line passing its
    body (hips to head, `npc.aim_distance`) finds it. E is a greeting every time, except a
    second press within `debounce` s (0.5): it goes on the event bus as the signal `greeted`
    from the NPC (`attention.greet.signal`) with the greeter as the actor, so a rule can act
    on it. The cottage's resident gives the `lights_out` quest this way, from her armchair too.
    What the NPC does about it fits what it is doing:
    - standing: its `greet` action on top of whatever it was doing: `face: player`,
      `play: wave` (the body's one-shot wave clip), a pause. Then the interrupted step starts
      again.
    - seated: it keeps its seat and its action, looks at the greeter for `acknowledge` s (3)
      wherever they stand, and waves from the chair: the wave clip's right-arm tracks laid
      over the sit pose (`GeogenHeadLook.gesture`).
    - lying down, or part way through sitting down or getting up: nothing to see.

    It waves once per `cooldown` s (8); a greeting inside that is still a greeting.
  - *yield*: an NPC standing within `distance` m of the player walking straight at it runs
    `step_aside` (`go_to: aside`, `step` m off the player's line, then `face: player`).
  - The summary reports `greets` (greetings taken, each one a signal), `waves` (the ones it
    waved back at), `yields` and `looking` (seconds). `--greet=NAME[@S]` greets
    headlessly.

Pedestrians cross at crosswalks. The lane graph exports the painted crossings (`traffic.crosswalks`). A planned path that runs along road level inside the streets is rerouted over the crossing that makes the shortest detour: to one end, straight across, then on. The detour must be no longer than twice the direct path plus 25 m.

At a signalled crossing (`traffic.pedestrian_signals`, see Traffic) they wait for walk. The rule is where an NPC stands and where it is headed, not the path it planned: on a kerb of the crossing with its road within 0.7 m ahead, it stands still, facing across, until the signal shows walk. It doesn't start on the flashing don't-walk, and once on the road it carries on. Unsignalled crossings, and scenes with no traffic to run the signals, are walked as before. The summary reports `signal_waits` (and `signal_wait`, the longest, in seconds), `signal_crossings` (steps off a kerb onto a signalled crossing) and `signal_violations` (the ones made on a steady don't-walk); `--npc-trace` prints `signal_wait` and `signal_step` events with the state.

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
- **Pedestrian signals**: `traffic.pedestrian_signals` lists the crosswalks of signalled
  intersections that have pavement at both ends: `intersection`, the `phase` they are walked in
  (the one whose traffic runs beside the crossing, so the traffic across it is at red), `walk`
  and `clearance` seconds. From the start of that phase's green a crossing shows walk for `walk`
  seconds, then a flashing don't-walk for `clearance` (long enough to walk it at 1.2 m/s, ending
  with the amber), then don't-walk: `GeogenTraffic.pedestrian_state(id)` gives `walk | flash |
  dont` on the same clock as `signal_state`. Heads with `extras.geogen.signal.crossing`
  (`pedestrian_signal.yaml`, one at each end facing across) light `lamp_walk`, or `lamp_dont`
  (blinking once a second while it clears). Turning traffic of the walk phase isn't held: it
  yields to people on the crossing, as anywhere. NPCs wait for walk (see Pedestrians).
- **Yielding**: the player and NPCs (group `geogen_character`) in the corridor ahead, or on a
  crosswalk about to be crossed, are obstacles to stop for.
- **Bodies**: vehicles are `AnimatableBody3D` boxes. They block the player, stay out of the
  navmesh, and never push anything. Wheels spin at v / r. Vehicles that drive off an open route
  come back at a route start.

`--simulate=S` also prints `traffic summary: [...]`: vehicles, distance, overlaps, idle and wait
times, claims, turns, yields and the closest stop to a person, and for pedestrian signals
`pedestrian_heads`, how often they turned (`walks`, `flashes`, `dont_walks`) and `walk_conflicts`
(checks that found one not at don't-walk while the traffic across it wasn't at red). `--traffic-trace` prints claims,
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

While a train of the service stands at a station (between a stop's `arrive` and `depart`), the
train's `at_<station>` state is true: `GeogenTrain.standing` reports the change and `WorldLoader`
sets the variable in the rules program, so rules and quests can wait for a train or notice it
leaving (`quests/catch_train.yaml`). Each platform is a place (`station_<name>`).

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
- **Switches**: every other fixture's rules state `<uid>#lit` shows and hides its light and glow
  (`WorldLoader._fixtures`). A light switch drives it through a binding in the manifest's `rules`,
  and any rule may set it.
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

## Stable IDs

Nodes that matter to gameplay (placed assets and scenes, tagged or typed nodes, anything
with interactions) carry `extras.geogen.uid`: `<package>:<authored path>`, e.g.
`cottage:door` or `hotel_room_auto:suite/bedroom/wardrobe`. The package is the scene's
registry name and the path is made of the names things were authored with (placement keys,
part names, room ids), never the uniquified node names of the export, so an ID stays the
same when the scene is re-exported. geogen resolves references and selectors
(`geogen/refs.py`, `python -m geogen.main -s cottage --resolve "house/[tag=furniture]"`) to
these IDs at export; the runtime only ever compares IDs.

- `WorldLoader.nodes_by_uid` / `node_by_uid(uid)` find a loaded node (the scene's own
  package may be left off: `door`). Streamed chunks add and remove their nodes as they load.
- Saved state (`--save`, `state.save`) is keyed `<uid>#<interaction>`. Flat saves from before
  IDs (`<node name>/<interaction>`) still load.
- Manifest `spawns`, `rooms` and `travel` entries carry `uid`; a travel `spawn` may be a
  spawn's name or its path (`street/hotel/entrance_spawn`).
- API: `world.node` and every command that takes a node accept `uid`; `world.query` takes a
  `uid` glob and returns each node's `uid`.
- `<name>.refs.json` beside the manifest (`geogen-refs` v1) lists every ID with its tags,
  type and asset. geogen checks references from other packages against it; the runtime
  doesn't read it.

## Places

Scenes and assets declare named trigger volumes with `places:` (`geogen/places.py`:
a box, a footprint polygon, or the box round something). Each is exported as an
empty node with `extras.geogen.type: place`, a stable ID and `place: {id, size}`
or `{id, polygon, height}`, and listed in the manifest's `places` in world space.

`GeogenSceneBuilder` gives each a `PlaceArea` (`Area3D`, group `geogen_place`),
beside the `RoomArea` of every room volume. `WorldLoader` watches both and emits
`place_event` when the player or an NPC walks in or out:

```
place event: {"type":"enter","subject":"island:lighthouse/gallery","place":"gallery","kind":"place","actor":"player"}
```

- `kind` is `place` or `room`; a room's `subject` is the room's ID, not its volume's.
- `actor` is `player` or the NPC's stable ID (`island:resident_a`).
- An arrival counts once the body is still inside two physics frames on, so
  bodies passing through the origin while a world loads don't count.
- Sitting or lying turns an actor's collision shape off; they stay in the place
  until they walk out, or an NPC vanishes.
- Unloading a world or a streamed chunk raises no `leave`.

`main.gd` prints the line above, the control API streams them as `place`, and
`world.info` lists `places` with the actors in each. `WorldLoader.places` holds
`{id, uid, kind, node, area, actors, detect, things}`.

A place with `detect: [actors, props]` also counts physics props in it, with the
actors, in `things`, and sets its `<uid>#occupants` state to how many there are
(`WorldLoader._occupy`). Props raise no `enter` / `leave`.

## Physics props

The runtime uses Jolt (`physics/3d/physics_engine` in `project.godot`; geogen.49).
Write new physics code for it. Godot Physics snagged spheres on the internal edges of a
subdivided trimesh, so a ball couldn't roll down a swept trough. Two things changed
with the switch: the player's and NPCs' cylinders have a 1 mm `margin` (Jolt rounds the
rim by it, which tilted a step's contact) and step-up allows a landing 3 cm over
`step_height`. Chunk loads are high-priority `WorkerThreadPool` tasks: a low-priority
one was never run under Jolt.

At 60 steps a second Jolt lets a fast body sink into a curved trimesh wall: a 12 cm
ball going 2 m/s round the marble run's helix sat 2-4 cm into the trough (the contact is
there, with the right normal, and isn't pushed out; solver steps, penetration slop, CCD
and the contact cache made no difference). At 120 it stays within 3 mm. A scene asks
for that with `physics: {ticks: 120}`: the manifest's (and chunk index's) `physics`,
which `WorldLoader._use_physics` applies while the scene is loaded (unload puts the
project's rate back), so other scenes keep paying for 60.

An asset with `physics:` (`geogen/physics.py`) is exported with
`extras.geogen.physics` (`mass`, `friction`, `bounce`, `carry`, a box `size` and
`center`; with `shape`, a sphere `radius` or an upright capsule `radius` and
`height`; optional `damping`). `GeogenProps` (`scripts/props.gd`, `WorldLoader.props`)
wraps each in a `RigidBody3D` with that collider (`GeogenProps.make_shape`; the
parts' own colliders are dropped), on layer 1, never asleep. A sphere rolls where a
box would rest on its friction. Props aren't static colliders, so the navmesh is baked without them.

- **Carrying:** aiming at a prop with `carry` prompts `E: Pick up`; it is held
  `HOLD_DISTANCE` (1.4 m) in front of the eyes, chased there by velocity so it
  still collides, level and turning with the player, and never collides with the
  holder. E drops it; so does being snagged more than 1.4 m from the hold point
  for half a second. The player walking into a prop shoves it (`player.gd`
  `_push_props`).
- **Spawn:** a prop with `spawn: <thing>.<attachment>` has a `spawn` interaction
  (`stowed | live`) that rules drive. Stowed, it is hidden, frozen and out of
  collision; going live, it is put at `spawn_at` at rest and let go. A live prop
  more than 10 m below the lowest model is stowed, which rules hear as a change
  (`{changed: cube.spawn, to: stowed}`), so they can hand out another.
- **Saves** keep every live prop's pose in `bodies` (see "Saving" in `docs/rules.md`).
- Control API: `world.bodies`, `player.carry {prop?}`, `player.drop`; `player.use`
  picks up and drops too (`focus.kind`: `carry` / `drop`).

The example is `examples/test_chamber` (`tests/test_test_chamber.py`).

## Spawned items

Sources make copies of item templates while the game runs (`geogen/flows.py`;
`scripts/flows.gd`, `GeogenFlows`, `WorldLoader.flows`).

- **Load:** `flows.setup` runs first in `setup_root`: every node with
  `extras.geogen.item` (a template) is taken out of the world and kept by its
  source (the nearest ancestor with `flows.source`), so no uid, interaction, prop,
  collider or navmesh bake sees it.
- **Making items:** each physics frame a source whose `running` state is true makes
  `rate` items a second, and one at once for each its `queued` state asks for (rules
  raise it with the `emit` input), while it has fewer than `max_alive` in the world
  and has made fewer than `count`. The pick among its items is weighted and seeded
  per source. It drives `emitted` and `alive` and posts the signal `emitted {item}`.
  Sources start `SETTLE_FRAMES` after `world_loaded` (the player's body sits at the
  origin for a frame or two), and wait while their last item is still within two of
  its radii of the spawn point (two bodies made in one place fly apart).
- **Sinks:** a place with `sink` (`{accepts: [item names]}`; empty: any) takes in a
  flow item whose body enters its area (`WorldLoader._place_crossed` ->
  `GeogenFlows.take`): the item goes, the sink's `received` counts it, and it posts
  `received {item, ...the item's data}`.
- **Physics items** are clone props (`GeogenProps.add_clone`): rigid bodies like
  any prop, carryable, keyed `flow#<id>`, not saved with the props. Items live under
  `Flows/Items` in world space.
- **Taken away** when they outlive `lifetime`, when a physics item has rested
  `stuck` seconds (give rolling items `damping`: Godot has no rolling resistance),
  or when they fall out of the world.
- **Tracks** (`flows.track`: a conveyor's top, a scene's `tracks:`) carry belt and
  walk items. A source whose spawn point is within `JOIN` (0.5 m) of a track's start
  puts its items there, one a frame when the start is clear; each frame every running
  track (its `running` state) moves its items, leaders first: belt items at the track's
  speed, walk items at their own, none closer to the one ahead than `spacing` or the two
  radii, facing along the track. An item with `gait` (`{stride, bob, sway, waddle}`)
  steps as it goes: its body rises `bob` m twice a stride, rocks `sway` degrees and
  swings its heading `waddle` degrees, keyed to the distance it has walked (drawn
  only: positions and reports are the track's). At the end an item goes on to the track starting
  within `JOIN` (`end: auto`), is taken in (`received`, signal `received {item, ...}`),
  waits (`stop`) or falls off as a physics body (`drop`). Tracks and sources are stepped
  in position order, and sources pick among their items by smooth weighted round robin,
  so `geogen/flows.py` `simulate` gives the same items at the same distances
  (`tests/test_flow_tracks.py` checks it on `flow_test`).
- **Roll tracks** (`mode: roll`): gravity drives each item from the track's `speed`
  ((5/7) g sin(slope), less `damping` per second), it spins about its middle, and a
  source puts physics items on one too: balls down a chute, which fall off a `drop` end
  as bodies moving at their speed. Use it where an item must follow a set path; a ball
  that should bounce and roll freely is a plain physics item (`examples/marble_run`).
- **Effectors** (`flows.effector`: `turret.yaml`) step last each frame, in position
  order: one whose `enabled` state is true and that is ready (every `1 / rate` s) acts on
  the track item within `range` of its `at` point that it `targets` (`first`: furthest
  travelled, `last`, `nearest`, `strongest`: most hp). Damage lands as it fires (the
  tracer or bolt is the look of it): an item at hp 0 shrinks away and its source counts
  `killed` and raises `killed {item, ...data}`; `take` removes the item (`taken`, signal
  `took`). Aim parts (`aim.yaw` / `pitch`, exported names) turn toward the target at
  `turn_rate`; damaged items get a billboard health bar; shots play the `shot` chime.
- **Waves:** a source with `waves` (a `kind: waves` file, read at build) waits each
  wave's `delay`, sends its groups (`count` items `gap` s apart), and when the wave's
  items are all gone raises `wave_cleared {wave}` and sets its `running` false (or goes
  on with `auto`); after the last, `all_cleared`. `wave` counts them.
- **Saves** have a `flows` section (`GeogenFlows.save` / `restore`, after the rules
  program loads): the live items (track uid and distance, or a physics item's pose and
  velocity, with their data) and each source's wave phase, timers and pick, and each
  effector's cooldown. Counts come back from the rules state. Loading takes the items in
  the world away first, so a run carries on from a save exactly as it would have.
- Control API: `flows.list` (`ticks`, sources, tracks, effectors, items with `track`, `s`
  and `travelled`).
  Tests: `tests/test_flows.py`, `tests/test_flow_tracks.py`.

## Events

`WorldLoader.events` (`scripts/events.gd`, `GeogenEvents`) is the one stream of
what happens in the world, in the vocabulary of `docs/rules.md`. Every event is

```
event: {"type":"changed","subject":"cottage:door","name":"swing","actor":"player","package":"cottage","key":"cottage:door#swing","data":{"from":"closed","to":"open"}}
```

`subject` is a stable ID, `actor` is `player`, an NPC's ID or `""` for nobody,
and `package` is the scene the subject belongs to.

| Type | When | Carries |
|---|---|---|
| `use` | an actor used an interaction, before it moves | `name`: the interaction |
| `changed` | the state an interaction is heading to changed, or its lock was turned | `key`, `data.from`, `data.to` |
| `signal` | an interaction arrived at a state with an `emit`, a use was refused (`locked`), or a signal an asset declares was raised | `name` |
| `enter`, `leave` | an actor walked into or out of a place or room | `data.place`, `data.kind` |
| `travel` | an actor went through a travel point | `name`: the target scene, `data.spawn` |
| `clock` | dusk or dawn; never has an actor | `name`: `night` or `day` |
| `start` | a new game began for the player: the first world of a run, unless `--load` | |
| `arrived` | an interaction reached a state (runtime only) | `data.state`, `data.emit`, `data.asset` |
| `npc` | an NPC's trace: decide, step, done, ... (runtime only) | `name`, `data`: the trace |

- The actor of a move follows it: whoever used a door is the actor of its
  `changed`, its arrival and the signal it raises there. A move nobody is behind
  (a timer, `interactions.set`, `interactions.use` with no `actor`) has none.
- Jumping to a state (`interactions.set` with `instant`, loading a save) raises
  nothing: nothing happened.
- An event the rules program caused (a variable it set, a signal it raised,
  a declared asset signal) also has `origin: "rules"`.
- An event posted while another is being delivered waits its turn, so every
  listener sees one order.
- `events.listen(callable, types, subjects)` filters by type and subject ID
  (empty: any) and returns an id for `unlisten`. `events.history` keeps the last 256.

`--events` prints every event as above; `--events=use,signal` only those types.
The control API's `events` stream is the bus (`events.subscribe` takes `types`
and `subjects`), `events.history {since, types, subjects}` returns what already
happened, and the `place` and `npc` streams are views of it. The older console
lines (`interaction event:`, `place event:`, `travelled:`) are unchanged.

## Rules, state and quests

`WorldLoader.rules` (`scripts/rules.gd`, `GeogenRules`) runs the loaded
package's rules program: the manifest's `state` (variables, interactions) and
`rules` (bindings, rules) sections, with `quests` for reporting. The semantics
are `docs/rules.md`; nothing here knows what a quest is beyond printing it.
`--scene state_test` (catalogue test group) exercises all of it in one yard: two
copies of a lever puzzle, a jammed lever that says so, a barrier only rules
lift, a switch and bound lamps, a place and a quest.

- `scripts/machine.gd` (`GeogenMachine`) is a GDScript port of the reference
  evaluator, with `scripts/conditions.gd` for condition trees. It is pure data.
  `godot --headless --path runtime/godot -- --trace=FILE` replays a conformance
  trace (`tests/data/rules/*.json`) with no world loaded, prints
  `trace: {"name", "problems"}` and exits 1 on a mismatch.
- `GeogenRules` is its host. Loaded interactions stay the world's own: the
  machine reads their state from the nodes and drives them with `set_state`,
  and their arrivals come back as events. Every rule event on the bus is an
  input; what the machine changes or raises goes back on the bus. The world
  clock is its clock, checked each minute, and time passes with physics.
- Actors are `player` and each NPC by stable ID.
- A package's state is kept when its world unloads and resumes when it loads
  again, so progress survives a trip to another scene.
- **Content packages** ("Content packages" in `docs/rules.md`) are state, rules
  and quests that belong to no scene. `catalogue.json` lists the deployed ones
  under `packages`: `[{name, file, exported, scenes}]`, each `file` a
  `geogen-content` JSON beside it (`docs/schema/geogen-content.v1.schema.json`).
  `GeogenRules.content_packages()` reads them once (again when catalogue.json
  is rewritten). When a scene one of them lists loads, its variables and the
  rules and bindings with no `scene` or with that scene join the scene's
  program after the scene's own (`geogen: content package errand runs in camp`),
  and its quests are reported with the scene's, with their own `package`. Its
  values (`world:<package>.`, `actor:<package>.`, `once:` latches) are kept
  apart from any scene's and given to the next scene's program, so what a rule
  set in one scene a rule reads in the next, across `--switch`, F7 / F8 and
  travel. A scene with no state of its own still gets a program when a package
  runs in it.
- A rule can fire as a scene loads, before its interactions' nodes exist (a
  condition a content package's state already satisfies). `GeogenRules.adopt`
  gives each interaction that state as its node appears.
- **Saving.** `--save` / `--load` and `state.save` / `state.load` write and take up one
  `geogen-save` (`docs/schema/geogen-save.v1.schema.json`, "Saving" in `docs/rules.md`):
  `{format, version, package, now, clock, world, actors, actor_info, timers}`.
  `WorldLoader.save_state()` is the machine's save with every interaction added to `world`
  as the instance state it is (`"cottage:door#swing": "open"`, `"cottage:door#swing_locked": true`),
  loaded or last seen in a chunk that has since unloaded. Only what differs from how things
  start is stored, so `load_state()` puts everything a save doesn't mention back as it was
  exported. A flat save from before the format (`{"<uid>#<interaction>": {state, locked}}`,
  the rules state under `geogen:rules`) still loads, and only moves what it names.
  `--status` still prints `states` as `{key: {state, locked}}`.
  - The save also holds every other package visited this session, as it was left
    (`packages`), and the content packages' values (`carried`); loading takes them up
    (`GeogenRules.take_save`), so a fresh process resumes each scene when it is visited
    and the quest log lists their quests (from each one's manifest). A save made in
    another scene loads this one's part from `packages`, or starts it afresh.
  - Loading sets the world clock (`GeogenClock.hours`: sun, sky, night lights) to the
    save's, not only the rules program's.
  - `--load` is applied before `--set` and `--quests`, which act on what it restored.
  - While an interaction's chunk is unloaded the rules program holds its state
    (`GeogenRules.release`, `hold_absent` after a load): rules read it as it was left and
    may drive it, and its node takes the program's value when the chunk loads (`adopt`).
  - How far a move or an `after` dwell had got is kept as the save's `arrive` / `dwell`
    timers (`GeogenInteraction.progress` / `resume`; an `arrive` says where it set out
    `from`): a door saved half-open on its way loads half-open and carries on, one waiting
    to close itself closes when it would have. A chunk unloading hands the same timers to
    the rules program, and `adopt` gives them back to the node.

```
quest event: {"quest":"lighthouse_gallery","title":"The view from the top","actor":"player","event":"started","status":"active","from":"available"}
quest event: {"quest":"lighthouse_gallery","title":"The view from the top","actor":"player","event":"objective_done","objective":"gallery","text":"Climb to the lighthouse gallery"}
quest event: {"quest":"lighthouse_gallery","title":"The view from the top","actor":"player","event":"complete","status":"complete","from":"active"}
```

- `--quests` prints `quests: [...]` for the player once the world is loaded:
  each quest's status and objectives (`done | open | waiting`, `progress: [n, of]`,
  and `markers`: the world position of an open objective's subjects).
- `--set=KEY=VALUE` sets a variable at start (`world:pkg.name`, `actor:pkg.name`
  for the player, `<uid>#<state>`).
- Control API: `quests.list {actor}`, `state.get {key, actor}`, `state.set
  {key, value, actor}`, `state.list`, and the `quest` stream.
- The cottage has a quest given by an NPC: `godot --path runtime/godot --
  --scene cottage --quests`, press E on the resident (or `npcs.greet
  npc=resident`), then switch off the three light switches. `quests.list`
  reports `progress: [n, 3]` as they go off.

Not built: NPCs working on quests (15.8). (Interactions in chunks that aren't
loaded are held by the rules program: see "Saving".)

## HUD

`scripts/hud.gd` (`GeogenHud`, `main.hud`) is the game HUD: one `CanvasLayer`
under the debug overlay (so a scene switch's fade covers it) that panels join.
The host owns what panels share: a column per screen corner (`top_left`,
`top_right`, `bottom_left`, `bottom_right`; the top ones start under the debug
overlay's three lines), sizes that follow the window (`panel_width()`,
`slot_height(slot)`), a coalesced refresh, key presses and `report()`.

**Stats and banner.** A scene's `hud:` block (`geogen/hud.py`) is the manifest's
`hud` section: `stats` rows (`label`, a compiled `value`, optional `of`) and
`banners` (`when`, `text`, `tone`). `scripts/hud_stats.gd` (panel `stats`, top
left) shows the rows ("Lives 17", "Wave 3 / 8") and `scripts/hud_banner.gd`
(panel `banner`, across the middle) the first banner whose `when` holds, both
evaluated for the player by the loaded program (`GeogenRules.value_of`) on
`changed` / `clock` events and HUD refreshes. Control API `hud.stats`. Tests:
`tests/test_hud_stats.py`.

The HUD also has the aim point (`scripts/hud_crosshair.gd`): a dot in the middle
of the screen, with a gold ring while E would do something, and the E prompt
("E: Lights off") under it. Aiming is forgiving: when the aim line lands on
nothing usable, the usable interaction whose target is within `AIM_SLACK`
(20 cm) of the line, within reach and with nothing in the way, takes the focus
(`main._assisted_focus`), so a light switch doesn't need a pixel-exact aim.
`hud.aim` on the control API returns `{visible, active, focus}`.

A panel is a `Control` added with `hud.add_panel(panel, slot)`; its node name
is its name in `hud.report()` and `hud.panel(name)`. It may define:

| Method | Called |
|---|---|
| `hud_attach(hud)` | once, before it enters the tree: connect to what it shows (`hud.world.events`, `hud.world.rules`) and call `hud.queue_refresh()` from those handlers |
| `hud_refresh()` | once per frame at most, after `queue_refresh()`; and whenever the program or its state is replaced (a scene loaded or unloaded, a save restored: `GeogenRules.reset`) or the window resizes |
| `hud_key(keycode) -> bool` | a key was pressed (not while the scene list is open); true if the panel used it |
| `hud_report() -> Dictionary` | what it displays, for the control API and tests |

Panels hold presentation state only and read what the console reads, so a
panel never needs to know about scene switches.

The slot `screen` gives a panel the whole window instead of a corner, and
`hud.set_modal(panel, on)` gives it the keyboard: while one panel is modal
(`hud.modal`), keys go only to its `hud_key`, nothing else reaches the game,
and `modal_changed` has `main.gd` park the player and free the mouse (as the
scene list does), then put both back.

**Quest tracker** (`scripts/hud_quest_tracker.gd`, panel `quests`, top right):
the tracked quest's title and its objectives, from `GeogenRules.report("player")`.
Open objectives have a ring, done ones a tick, counted ones their progress
("3 of 5"), optional ones say so, and ones still waiting on another (`after`)
aren't listed. Text wraps to two lines and the panel clips at the bottom of a
small window.

- It redraws on quest events, on any `changed` event while a quest is shown
  (progress counts) and on the host's refreshes. Nothing is polled.
- The first active quest is tracked. **Q** tracks the next one (a line under
  the title says "1 of 2    Q: next quest" when there are several); the pick
  is remembered per package across scene switches. A quest that starts doesn't
  take over from the tracked one.
- A tracked quest that completes or fails stays up for 4 s marked "Quest
  complete" / "Quest failed", then the next active one replaces it. With no
  active quest the panel is hidden.
- Only the player's quests are shown.

`hud.quests {track?, linger?}` returns what is on screen:

```json
{"visible": true, "tracked": "lights_out", "package": "cottage", "title": "Lights out",
 "status": "active", "note": "1 of 2    Q: next quest", "active": ["lights_out", "errand"],
 "lines": [{"id": "lights", "text": "Switch off every light", "state": "open", "optional": false,
            "progress": [3, 5], "progress_text": "3 of 5"},
           {"id": "window", "text": "Close the window (optional)", "state": "open", "optional": true,
            "progress_text": ""}],
 "waiting": 1, "clipped": false, "rect": [948, 84, 320, 106], "viewport": [1280, 720]}
```

`status` is `active`, or `complete` / `failed` while a finished quest is still
up; `waiting` counts objectives not listed yet; `rect` is the panel on screen
(`[]` when hidden). `track: <quest id> | next | previous` picks the quest
first (an error if it isn't active), and `linger: <s>` sets how long a finished
quest stays (0 dismisses one that is up). Tests: `tests/test_hud_runtime.py`.

**Notifications** (`scripts/hud_notifications.gd`, panel `notifications`, bottom
left): a toast for each quest event of the player's (`GeogenRules.quest_event`):
"Quest started" and "Quest complete" / "Quest failed" with the quest's title,
"Objective complete" with the objective's text. Each has its kind's colour and
mark, and a short chime when it appears (`GeogenSound.chime`, synthesised in
`scripts/sound.gd` like the horns: `CHIMES` holds each kind's notes).

- A toast stays 4 s (`hold`) from when it appears, then fades. At most three
  (`limit`) are up at once, and one appears no sooner than 0.6 s (`gap`) after
  the one before. So a last objective, its quest completing and the next one
  starting arrive one by one in the order they happened, each with its chime;
  the rest wait in a queue. New toasts join below the ones being read.
- Only events raise them. A scene switch, a reload or a restored save resume
  state without events, so they raise none; a quest that starts with the game
  does (`main.gd` registers the panels before the first world loads). Toasts
  that are up or waiting carry over a switch. When the program or its state is
  replaced (`GeogenRules.reset`), a toast about the loaded package that its
  state no longer bears out (a save from before the quest was done) is dropped.
- Only the player's quests: an NPC's raise nothing.
- Feedback (`GeogenRules.feedback`, docs/rules.md "Feedback") for the player or
  for no actor: a rule's `notify`, or an interaction's `refused` text when its
  `when` turns the player away, is a `notice` toast (no heading, an "i" mark,
  no chime of its own) in the same queue; a `play` sounds its chime at once
  (`CHIMES` has every name in `rules.SOUNDS`), counted in `sounds`. Every piece
  of feedback prints `feedback: {kind, value, actor}`. While the player aims at
  an interaction whose `when` refuses them and that has `refused` text, the E
  prompt shows that text instead (the crosshair stays inactive; `hud.aim` focus
  kind `refused`), and E says it again.

`hud.notifications {hold?, gap?, limit?, muted?, clear?}` returns the queue:

```json
{"visible": true,
 "showing": [{"seq": 3, "kind": "objective_done", "package": "quest_yard", "quest": "yard_round",
              "objective": "third", "heading": "Objective complete", "text": "Walk to plinth 3", "age": 0.84}],
 "queued": [{"seq": 4, "kind": "complete", "package": "quest_yard", "quest": "yard_round",
             "heading": "Quest complete", "text": "The round"}],
 "raised": 4, "chimes": 3, "hold": 4.0, "gap": 0.6, "limit": 3,
 "rect": [12, 656, 320, 52], "viewport": [1280, 720]}
```

`kind` is `started | objective_done | complete | failed | notice` (`sounds`
lists what `play` feedback played); `showing` and
`queued` are oldest first; `raised` counts every toast so far (the last one's
`seq`) and `chimes` the ones that have sounded (counted in a headless run
too, where nothing is played). Tests set `hold` (0 dismisses
what is up; a large value keeps it), `gap` and `limit` to make the queue
deterministic, and `clear: true` drops everything.

**Quest log** (`scripts/hud_quest_log.gd`, panel `log`, slot `screen`): **J**
opens a screen over the game listing the player's quests in three sections,
Active, Completed and Failed, with the selected one's description and every
objective beside the list: a tick when done (with its count), a ring while
open, a dimmed dash for one that waits on another ("later") or was left undone
when the quest ended ("not done"). J or Esc closes it.

- While it is open the player doesn't move or look, E and the other game keys
  do nothing, and the mouse is free. The world keeps running.
- Up / Down (or W / S, or a click) select a quest. Enter (or Space, a double
  click, or the "Track this quest" button) makes the tracker follow it. Only an
  active quest can be tracked.
- It lists every package the session has loaded: the loaded one from
  `GeogenRules.report`, the others as they stood when their world unloaded
  (`GeogenRules.report_all`: `WorldLoader.unload` has the rules keep each
  actor's report first). Those rows carry the scene's name; tracking one sets
  what the tracker follows when that scene loads again.
- It redraws on quest events, on `changed` events while open and on the
  host's refreshes, so it stays right, and stays open, across a scene switch,
  a reload and a restored save. A save holds the loaded package only, so
  restoring one doesn't change what is listed of other scenes.
- Quests of packages not visited this session aren't listed (none of theirs
  has started), and neither are quests still `available` or `unavailable`.

`hud.log {open?, select?, track?}` returns what it holds, open or not:

```json
{"open": true, "tracked": "quest_yard/yard_round",
 "quests": [{"key": "quest_yard/yard_round", "package": "quest_yard", "id": "yard_round", "title": "The round",
             "status": "active", "loaded": true, "tracked": true, "done": 1, "of": 3}],
 "sections": {"active": ["quest_yard/yard_round", "quest_yard/yard_lamps"], "complete": [],
              "failed": ["quest_yard/yard_dash"]},
 "selected": {"key": "quest_yard/yard_lamps", "title": "Lamps out", "status": "active", "loaded": true,
              "tracked": false, "can_track": true, "description": "The steward wants ...",
              "objectives": [{"id": "lamps", "text": "Switch off every lamp at the board", "state": "open",
                              "optional": false, "progress": [2, 3], "progress_text": "2 of 3"},
                             {"id": "report", "text": "Tell the steward it's done", "state": "waiting",
                              "optional": false, "progress_text": ""}]},
 "hint": "Up / Down: select    Enter: track    J / Esc: close",
 "rect": [200, 100, 880, 520], "viewport": [1280, 720]}
```

`quests` is in display order; `done` / `of` count required objectives; `rect`
is the frame (`[]` while closed). `select` takes a quest id, `<package>/<id>`,
`next` or `previous` (an error if it isn't listed); `track: true` tracks the
selected quest (an error if it isn't active). Headless `--quests` and
`quests.list` print the same data as before. Tests: `tests/test_hud_log_runtime.py`.

**Markers** (`scripts/hud_markers.gd`, panel `markers`, slot `screen`, drawn
under the other panels): an indicator in the world over what each open
objective of the tracked quest is about, with the distance to it. They follow
the tracker: the quest it shows is the one marked (Q and the quest log change
both), objectives that are done or still waiting on another have none, and
neither has one with `marker: hidden`.

- Positions are the manifest's `quests.markers` (through
  `GeogenRules.report`), by stable ID, so a subject in a chunk that isn't
  loaded is marked (streamed scenes too: the chunk index has them). A place is
  marked at the middle of its volume; a subject the export didn't mark (a
  content package's) is found in the loaded world by stable ID.
- **Fade**: markers fade from 40 m to 160 m away (`FADE_FROM`, `FADE_TO`,
  down to 0.45), and one with something between it and the camera (rays a few
  times a second) is drawn at half strength. `alpha` and `occluded` are reported.
- **Counted objectives** (`count` / `all` / `any` over subjects): subjects that
  are done already (`subjects_done` in the report) aren't marked, so the last
  light switch is the only one left.
- **Edge**: an indicator that is off screen or behind the camera sits on the
  edge of the screen (`edge_margin` in: 7% of the smaller side, at most 44 px)
  with an arrow pointing the way to turn; one straight behind is on the bottom
  edge.
- **Area** (`marker: {area: r}`): a ring of that radius on the ground (its
  height taken from the colliders under it), and an indicator with the distance
  to the ring's edge until the player is inside it. The ring is not centred on
  the subject: its middle is 0.45 r off, in a direction fixed by the subject's
  id, so the subject is somewhere inside. Several subjects share one ring round
  their middle.
- **Portal**: a quest of another scene can't be marked where it is. Every
  travel point of the loaded scene that goes straight to that scene is marked
  instead (an arch, labelled with the scene's name), for one quest per package
  visited this session: the one tracked there (the quest log's pick), else its
  first active one, if it has an open objective with a marker. With the
  island's lighthouse quest going, the hub's Island portal is marked. With no
  travel point straight there, the first steps of the shortest routes over the
  exported scenes' travel points are marked (`route` lists the scenes on the
  way). A portal carries one marker, the tracked quest's.
- The list is rebuilt on quest events, `changed` events and the host's
  refreshes; only the projection runs per frame. They are hidden while a modal
  panel (the quest log) is up.

`hud.markers {enabled?}` returns what is displayed:

```json
{"visible": true, "package": "island", "tracked": "lighthouse_gallery", "viewport": [1280, 720],
 "edge_margin": 44.0,
 "markers": [{"quest": "lighthouse_gallery", "package": "island", "objective": "gallery",
              "subject": "island:lighthouse/gallery", "kind": "point", "position": [39.5, 15.454, 7.0],
              "screen": [1232.7, 676.0], "distance": 80.3, "distance_text": "80 m",
              "on_screen": false, "clamped": true, "behind": true, "turn_deg": 28.1}]}
```

`kind` is `point | area | portal`; `position` is what is pointed at (a ring's
middle, a portal's middle) and `distance` how far the camera is from it (from
a ring's edge); `screen` is where the indicator is drawn; `turn_deg` (clamped
ones) is the arrow's direction in screen axes. An area has `radius` and
`inside`; a portal has `scene`, `label` and the travel point's ID as
`subject`. `enabled: false` hides them. Not handled: a subject already done
within a counted objective (every light of "3 of 5" stays marked), and the
side-by-side layout of `--scene=all` for portals. Tests:
`tests/test_hud_markers_runtime.py`, `tests/test_hub.py` (the hub's portal).

## Node extras

`docs/schema/geogen-extras.v1.schema.json` is the whole contract for
`extras.geogen` on a glTF node, and it is closed: the exporter writes a node
meta key only when the schema declares it (`src/geogen/extras.py`; what the
generator keeps for itself is registered there as build-only), and a key that
is neither fails the export. `GeogenSceneBuilder.KNOWN_KEYS` is the same list
(`tests/test_extras_contract.py` keeps it, the schema and this table in step).
A new key needs a schema entry and a row here.

| Key | Read by | For |
|---|---|---|
| `version` | `addons/geogen/scene_builder.gd` | warns once when it isn't the supported version |
| `uid` | `scripts/world_loader.gd`, `scene_builder.gd`, `scripts/api.gd` | the stable ID: `node_by_uid`, event subjects, saved state |
| `tags` | `scene_builder.gd`, `scripts/api.gd` | node groups (`furniture.bed` and `furniture`) |
| `type` | `scene_builder.gd`, `scripts/world_loader.gd` | what kind of node it is: `room_volume`, `place`, `spawn`, `npc`, `traffic`, `train`, `light`, `lod`, `collider` |
| `shape` | `scripts/world_loader.gd` | collider nodes: convex (`box`, `hull`) or trimesh |
| `collider` | `scripts/api.gd` | only to tell mesh parts from things in `world.node` |
| `level`, `ratio` | - | `<node>_LOD<n>` nodes; the runtime drops them and uses Godot's own LODs |
| `walkable` | `scripts/api.gd` | as `collider`; the navmesh is baked from colliders |
| `room`, `size` | `scene_builder.gd`, `scripts/world_loader.gd` | room `Area3D`s and the room list |
| `nav` | `scripts/world_loader.gd` | rooms the playtest shouldn't expect to reach |
| `door`, `joint`, `door_swings` | - | floor-plan doors and their swing arcs (the door's interaction is what moves it) |
| `footprint`, `clearance`, `container` | - | furniture: floor area, free space, storage |
| `interactions` | `scripts/world_loader.gd` | a `GeogenInteraction` (`scripts/interaction.gd`) each |
| `clips` | `scripts/npc.gd` | body animations and transition clips |
| `light` | `scripts/world_loader.gd` | tunes the fixture's imported light; `auto: night` |
| `affordances` | `scripts/world_loader.gd` | sit / lie / use points for the player and NPCs |
| `portal` | `scripts/world_loader.gd` | doorways NPC paths open (`scripts/npc.gd`) |
| `place` | `scene_builder.gd` | `PlaceArea` trigger volumes |
| `physics` | `scripts/props.gd` | rigid-body props, carrying, spawn / stow |
| `flows` | `scripts/flows.gd` | spawned items: a source's rate, limits and spawn point |
| `item` | `scripts/flows.gd` | an item template: taken out of the world on load and copied by its source |
| `travel` | `scripts/world_loader.gd` | travel points |
| `npc` | `scripts/world_loader.gd` | a `GeogenNpc` (`scripts/npc.gd`) from the resolved definition |
| `poses` | - | on body assets; NPCs read the copy in `npc.body.poses` |
| `stairs` | `scene_builder.gd` | the `NavigationLink3D` and walking path of a spiral stair |
| `water` | `scripts/world_loader.gd` | parameters of `scripts/water.gdshader` |
| `vehicle` | `scripts/traffic.gd`, `scripts/train.gd` | wheels, lamps, clearance, sound |
| `driving` | `scripts/traffic.gd` | where a fleet vehicle starts on the lane graph |
| `fleet` | `scripts/world_loader.gd` | a `GeogenTraffic` (`scripts/traffic.gd`) |
| `signal` | `scripts/world_loader.gd`, `scripts/traffic.gd` | signal heads lit from their intersection's cycle (with `crossing`: pedestrian heads, lit from that pedestrian signal) |
| `train` | `scripts/world_loader.gd` | a `GeogenTrain` (`scripts/train.gd`) |
| `city`, `block`, `lot`, `building`, `street_furniture`, `recipe`, `roof`, `scatter`, `character` | - | where generated content came from |
| `lod`, `interior` | - | roots of a chunk's LOD stand-in and interior (the streamer works from the chunk index) |

A `-` means this runtime doesn't read the key: it describes the world for
other runtimes and tools.

Three more things a runtime relies on are in node names and children, not in extras:

- **Colliders** are child nodes named `<node>-colonly` (trimesh) or
  `<node>-convcolonly` (box or convex hull), Godot's import suffixes. Their
  extras say `type: collider` and `shape`. They are not to be drawn.
- **Lights**: a fixture with `light` has a child `<node>_light` carrying a
  `KHR_lights_punctual` point light (extras `type: light`).
- **LODs**: `<node>_LOD<n>` children are the levels of `<node>`'s mesh, listed
  in its `MSFT_lod` extension (extras `type: lod`, `level`, `ratio`).

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
