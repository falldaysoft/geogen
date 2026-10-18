# Transport: vehicles, traffic, railways, pedestrians

Vehicle assets, the lane graph and signals, moving traffic, railways and trains, and pedestrians in the street.

## Vehicles

`assets/vehicles/*.yaml` (like every `roots.ASSET_SUBDIRS` library) are registered by stem (`-s car`); `vehicle_lineup` is the review scene. The assets are:
- `car.yaml`, with presets sedan, hatchback, estate and taxi;
- `van.yaml`;
- `bus.yaml`.

They're built along +Z (the front) with the origin on the ground midway between the axles; left = +X. Parts are in metres (a unit container with `bounds: geometry`).
- **Construction:** the body is a side profile `extrude` with `axis: x`, so profile points are `[-z, y]`. Wheel arches are cut by cylinder cutters. Glass is `car_glass`. The body colour is a `paint` choice param applied as `tint: "{paint}"` over the neutral `car_paint` material.
- **Tints:** part `tint:` (a palette name or RGB) writes vertex colours after CSG. Palettes are `VEHICLE_PAINT` and `PAINT_PALETTE` in `materials/loader.py`.
- **Conditional parts:** part `when: "{expr}"` drops the part, and parts attached to it, when the expression is 0 or false (the taxi sign).
- **`vehicle:` block:** handled by `geogen/vehicles.py` and becomes `meta.vehicle` (`meta.type: vehicle`, schema-validated). Its keys:
  - `class`, `wheels`, `steer`, `paint` and `lamps {head, tail, brake}`, all naming parts;
  - `bogies` and `couplers` for rail;
  - `max_speed`, `turn_radius`, `accel`, `decel`;
  - `sound: {engine: {base, per_speed, volume_db, roughness}, horn: {tones, seconds, volume_db}}`, which the runtime synthesises (`scripts/sound.gd`).
- **Measured values:** wheel centres and radii, wheelbase, track, `clearance [length, width, height]`, and front/rear bumper z.
- **Export:** rewrites these part names to the exported (uniquified) node names.
- City `parking:` (`fleet` weighted with draw params, `fill`, `bay`, `drive`, `both_sides`, `clear`) puts static cars in kerbside bays.
  - Cars face the traffic on their kerb and keep clear of corners and hydrants.
  - Streets narrower than `both_sides` park on one kerb only.
  - Nodes carry `meta.parked`.
  - Tests: `tests/test_vehicles.py`. Traffic epic: geogen-r65.

## Traffic lanes

`geogen/traffic.py` builds a lane graph for vehicles. It is precomputed so the runtime only follows curves.
- **City lanes:** `city: {traffic: {drive, lanes, avenue_lanes, speed, avenue_speed, turns}}` gives:
  - directed lanes per street segment, laid across the carriageway that's left after parking and running between the crosswalks (a lane's end is its stop line);
  - Bézier connectors through each intersection (`turn` straight/left/right, with the kerb-side turn from the kerb lane).
- **Routes:** scene `routes: {name: {path | {spline}, lanes, drive, speed, loop, lane_width}}` declares explicit roads. Open routes `end: exit`.
- **Precomputed data:**
  - lanes resampled every 0.5 m, with the lane `speed` capped by curvature;
  - `conflicts` (connector pairs within 2.2 m in one intersection, with s-ranges);
  - connector `crosswalks` s-ranges;
  - `intersections` (`control: all_way`).
- **Storage and export:** graphs live in `meta.traffic` of the city/scene node (not exported as extras). `build_traffic(scene)` merges them in world space into the manifest's and chunk index's `traffic` section, with schema `docs/schema/geogen-traffic.v1.schema.json`.
- **QA:** `check_graph` finds dead ends and unknown successors. `check_clearance` sweeps car/van/bus boxes along every lane against geometry. Town street trees use `maple_tree` `height: 6.8, trunk: 0.6` so the crowns clear buses.
- **Review:** `-r out.png --lanes` overlays ribbons (lanes blue, straight green, left orange, right magenta) and prints issues. Tests: `tests/test_traffic.py`.
- **Traffic lights:** `city: {traffic: {signals: avenues | all | none, cycle: {green, amber, all_red}}}` signals intersections with three or more arms (with `exits`, an avenue's end at the edge of the grid counts four).
  - North/south approaches are phase 0 and east/west phase 1. Lanes carry `signal {intersection, phase}` and `kerb` (the distance to their right kerb). Intersections carry `control: signals`, `cycle` and a staggered `offset`.
  - The city places `signal_head.yaml` at each signalled stop line on the pavement (`meta.signal`).
  - The runtime runs the cycle from world time and lights `lamp_red|amber|green`. Vehicles claim on green, or on amber when too close to stop, and wait at red; only a lane's front vehicle claims (a queued one would hold its claim and lock out conflicting traffic). A car held on green longer than `give_up` picks another exit.
- Traffic placements take `count:` to size the fleet (`crossroads` uses 10).
  - **Pedestrian signals:** the graph's `pedestrian_signals` lists each crosswalk of a signalled intersection with pavement at both ends: `{id, intersection, arm, phase, crosswalk, length, walk, clearance}`. Crossings over the district's edge streets (no pavement beyond) aren't painted or walked, so every crossing at a signalled intersection has signals.
    - `phase` is the vehicle phase it is walked in: a crossing over the west or east arm is crossed by east/west traffic, so it is walked in phase 0, beside the north/south traffic. Turning traffic of that phase isn't held; it yields to people on the crossing.
    - `traffic.pedestrian_timing`: from the start of the phase's green, walk, then a flashing don't-walk `clearance` (length / `PEDESTRIAN_SPEED` 1.2 m/s, rounded up to 0.5 s) that ends with the amber, then don't-walk. Walk is what is left of green + amber, at least `MIN_WALK` 4 s; `check_graph` reports a cycle too short for both. `traffic.signal_state` / `pedestrian_state` are the reference for runtimes.
    - The city places `pedestrian_signal.yaml` (parts `lamp_walk`, `lamp_dont`) at both ends of each, at the kerb just outside the stripes, facing across, named `signal_walk_<id>_a|b` with `meta.signal = {intersection, phase, crossing}`. `traffic.gd` lights them (`pedestrian_state(id)`; the summary's `pedestrian_heads`, `walks`, `walk_conflicts`). The player on a kerb end or on the crossing sees WALK / DON'T WALK in the HUD (`hud_crossing.gd`, API `hud.crossing`).
- Stop lines are set back 3 m (`STOP_SETBACK`) and connectors within 3.6 m conflict, so buses' swept turns clear waiting cars.
- **Sweeps** (`traffic._sweeps`, graph `sweeps: [{connector, lane, hold}]`): a bus (`SWEEP_VEHICLE` 11.3 x 2.95 m, wheelbase 5.9) is swept along each turning connector, from 12 m before it (its rear still on the lane it leaves) to 12 m after, as a shapely union; every other lane into that intersection whose waiting car or van (`SWEEP_WAITING` 5.4 x 2.3 m) at the stop line would touch it gets a `hold`, how far back from the line that vehicle must stand to be clear. On two-lane avenues a bus's tail swings into the lane beside it; into a narrow street it cuts across the opposing stop line. The runtime holds either side (README "Swept turns").
- **Moving traffic:** `kind: traffic` definitions live in `assets/traffic/*.yaml`.
  - They declare the fleet (weighted assets with draw params, optional `lanes` id prefixes), `count`, `spacing`, `driving` (IDM `accel`, `decel`, `headway`, `gap`, `speed_factor`, `stop_wait`, `look_ahead`, `give_up`), `turns` weights and `yield: [player, npc]`.
  - A placement `{traffic: traffic/town.yaml, seed}` runs after everything else, including spawns. `place_traffic` puts the starting vehicles on lanes (`meta.driving = {lane, s, factor}`), 12 m clear of spawns and NPCs, under a node with `meta.type: traffic` and `meta.fleet`.
  - The Godot interpreter is `runtime/godot/scripts/traffic.gd`; see the Traffic section of `runtime/godot/README.md`. `-s crossroads` is the small test grid; `-s traffic_scale` (a 6 x 5 grid of 80 x 64 m park blocks, no furniture, 150 vehicles) is the scale test: far vehicles as MultiMesh instances, the 1 ms script budget (geogen.9.6). Runtime tests: `tests/test_traffic_runtime.py`. Ad-hoc Godot runs on macOS need a timeout wrapper: there's no `timeout` command, so use `perl -e 'alarm shift; exec @ARGV' 300 godot ...`.

## Railways and trains

- **`railways:`** is a scene key built by `geogen/railway.py`: `{name: {path | {spline}, loop, gauge, speed, stations: [{name, at (fraction | {s}), length, side, width}]}}`.
  - It sweeps a low track bed along the path (ballast, sleepers, rails; rail head `RAIL_TOP` = 0.16 above the ground, so road crossings are near flush).
  - It adds 0.3 m platforms (tagged `street.sidewalk`) with benches and lamps. Each platform is a named place, `railway_<line>/station_<name>` (`_l` / `_r` when a station has both), so rules and quests see who stands on it.
  - It records `meta.railway` (points at the rail head, stations), exported to the manifest's `traffic.railways`, not to extras.
- **Level crossings:** `level_crossings` finds where the line crosses road lanes of the lane graph. Each crossing lane gets a `crossing_barrier.yaml` (interaction `barrier`, up|down) on its kerb side, a stop point 4.5 m before the track, and `crossings[{id, s, barriers, lanes}]`.
- **Route surfaces:** routes now get an asphalt surface with a centre line (`surface: false` to skip it).
- **Rolling stock:** `vehicles/loco.yaml` and `vehicles/carriage.yaml` have bogies (with the wheels attached to them) and couplers. `meta.vehicle.bogie_centers` records where each bogie pivots.
- **Trains:** `kind: train` definitions (`assets/trains/commuter.yaml`) declare a `consist`, `speed`, `accel`, `decel`, `timetable {headway, dwell, offset}` and `warning`.
  - A placement is `{train: trains/commuter.yaml, railway: <name>}`. `trains.place_train` simulates the run once (`simulate_run`: stops with the train's middle at each station) and computes crossing `closures`.
  - On a loop a train goes round and round: each lap starts standing at its first station and ends back there, and takes exactly the `headway` (the spare time is shared out as longer dwells; a lap that can't be run that fast is an error). More trains on the loop means more placements with `offset`s (the city runs two, half a lap apart). Open lines run end to end, a fresh consist per departure.
  - `trains.check_spacing` (run by the composer) fails the build if two trains on one line would come within `SPACING` (20 m) of each other over an hour of world time.
  - The runtime is `runtime/godot/scripts/train.gd` (see the README's Trains section).
  - `-s level_crossing` is the test scene, and `town` has a line along its north edge. Tests: `tests/test_railway.py` and `tests/test_traffic_runtime.py`.

## Pedestrians and crowds

- An NPC placement with `scatter:` is a crowd. Each copy draws its own seed, so its archetype body and brain differ. Every copy shares the placement's `home:`.
- `npcs/pedestrian.yaml` (townsfolk bodies) strolls and pauses. NPC definitions take two keys for this:
  - `affordance_tags: [outside]` keeps pedestrians to benches, not people's chairs;
  - `wander: {tags: [street.sidewalk]}` makes `go_to: random` pick area-weighted points on the up-facing faces of nodes with those tag groups (npc.gd `_random_surface_point`).
- Scatter `on_tag: street.sidewalk` places copies on those surfaces.
- Pedestrians cross at the painted crossings (lane graph `crosswalks` rectangles; npc.gd `_use_crosswalk` reroutes road-level paths).
- At a signalled crossing they wait on the kerb, facing across, until walk (npc.gd `_wait_for_walk`: on a kerb with the crossing's road within `KERB_STEP` ahead, whatever the path). They don't start on the flashing don't-walk and finish once on the road. The summary reports `signal_waits`, `signal_wait`, `signal_crossings` and `signal_violations`. `tests/test_traffic_runtime.py::test_pedestrians_wait_for_the_walk_signal` covers it on `crossroads`.
- `town` has 10 pedestrians and `crossroads` has 6. Traffic yields to them. `tests/test_traffic_runtime.py::test_pedestrians_stroll_and_rest_while_traffic_yields` covers this.
