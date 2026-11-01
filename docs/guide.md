# Guide

How to render, view and export scenes, walk them in Godot, and write your own assets and scenes. For every primitive, asset and parameter, see [the catalogue](catalogue/README.md) or `python -m geogen.main --describe NAME`.

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For the runtime, install [Godot 4.7](https://godotengine.org/download). The tests look for `godot`/`godot4` on `PATH`, then `$GODOT`, then `/Applications/Godot.app`.

## Running scenes

Every YAML file in `assets/` and `assets/scenes/` is a scene you can pass to `-s` (for example `chair`, `dining_set`, `cottage`, `street`, `hotel`, `hotel_room_auto`, `park`, `town`), plus the Python-coded `nature` scene. Run `python -m geogen.main --help` for the full list.

```bash
# Interactive viewer
python -m geogen.main -s dining_set

# Render to a PNG and quit (shadows, PBR, auto-framed)
python -m geogen.main -s room -r out.png --resolution 1280x720
python -m geogen.main -s chair -r out.png --view side --zoom 1.5 --no-ground
python -m geogen.main -s cottage -r sheet.png --views                  # iso/front/side/top
python -m geogen.main -s cottage -r sheet.png --views iso,back,top
python -m geogen.main -s hotel_room_auto -r out.png --cutaway          # hide ceilings/roofs to see inside

# Screenshot the interactive viewer (display: lit | clay | normals | uv)
python -m geogen.main -s street --viewer-screenshot shot.png --display uv

# Export for game engines
python -m geogen.main -s dining_set -e out/dining_set.glb              # .glb / .gltf / .obj
python -m geogen.main -s hotel -e out/hotel.glb --lods 0.5,0.25        # with decimated LODs
python -m geogen.main -s town --chunks out/town_chunks --cache         # streamable chunks; cache generated assets
```

## Walking a scene in Godot

```bash
python -m geogen.main -s cottage --export-godot                        # -> runtime/godot/generated/
godot --path runtime/godot -- --scene cottage
```

The runtime's scenes are listed in [`assets/runtime_scenes.yaml`](../assets/runtime_scenes.yaml), grouped as `showcase` or `test`. It also names a `default`: the scene the runtime opens when you don't pass `--scene`.

```bash
python -m geogen.main --export-catalogue              # export every catalogue scene (or: showcase, test, NAME,NAME)
python -m geogen.main --catalogue                     # rewrite generated/catalogue.json after changing the default
godot --path runtime/godot                            # open the default scene
godot --path runtime/godot -- --scene=all             # load every export side by side (slow)
```

You spawn at the scene's spawn point (or in front of the model). The controls are:
- WASD moves, Shift sprints, Space jumps, and the mouse looks around (click to capture it, Esc to release).
- E uses what you're looking at (doors, drawers, switches, the lift) or sits/lies on furniture, and L locks or unlocks a door you hold the key for.
- F1 toggles the overlay, F2 the collider wireframes, F3 fly mode, and F4 an overview camera.

If you re-export while the game is running, it reloads the model within half a second.

Large scenes stream: the runtime loads nearby blocks in full, distant ones as LODs, and building interiors as you approach them, and bakes navigation tiles around you.

```bash
python -m geogen.main -s town --export-godot --stream                 # -> runtime/godot/generated/town_chunks/
godot --path runtime/godot -- --scene town --stream
```

Traffic, trains and pedestrians start moving when the scene loads (`-s crossroads` and `-s level_crossing` are small test scenes). `--time=21:00` starts at night, when the street lamps and headlights come on. NPCs placed in the scene come to life when it loads. F5 shows what each one is doing, `--camera=follow` watches one, and `--timescale=8 --simulate=600 --npc-trace` runs ten minutes of their day headless and prints every decision.

To check that a building is enterable, export it (`-s hotel --export-godot`) and run `godot --headless --fixed-fps 60 --path runtime/godot -- --scene hotel --playtest`. It reports any rooms or interaction targets the player can't reach from the spawn. See [`runtime/godot/README.md`](../runtime/godot/README.md) for all the runtime's options and the manifest and chunk index formats.

## Defining assets

An asset is a set of parts in a container of `size` metres, with its origin at the bottom centre. Parts are positioned by `anchor` + `offset` (as fractions of the container) or attached to another part with `attach_to`/`at`/`from`. Parameters can be overridden from code with `LayoutLoader().load(path, params={...})`.

```yaml
name: table
origin: bottom_center

params:
  width:  { default: 1.2 }
  height: { default: 0.75 }
  depth:  { default: 0.8 }
  top_thickness: { default: 0.035 }

size: ["{width}", "{height}", "{depth}"]

parts:
  top:                                   # rounded slab: 2D shape extruded along Y
    primitive: extrude
    shape: { rect: ["{width}", "{depth}"], radius: 0.05 }
    fit: none                            # shape is in metres
    axis: y
    bevel: 0.008
    size: [1.0, "{top_thickness / height}", 1.0]
    anchor: bottom_center
    offset: [0, "{1 - top_thickness / height}", 0]
    material: wood

  leg_front_left:                        # turned leg: (radius, height) profile revolved around Y
    primitive: lathe
    profile: { spline: [[0.62, 0.0], [0.42, 0.3], [0.72, 0.6], [1.0, 1.0]] }
    size: [0.05, "{1 - top_thickness / height}", 0.075]
    anchor: bottom_front_left
    offset: [0.08, 0, -0.08]
    material: wood

attachments:                             # where other objects can attach
  seat_front:
    anchor: bottom_front_center
    offset: [0, 0, 0.4]
    facing: center
```

Buildings use CSG and architectural primitives:

```yaml
parts:
  walls:
    primitive: cube
    subtract: [interior]                 # hollow shell
    material: brick
  interior: { primitive: cube, cutter: true, ... }
  roof:
    primitive: roof
    style: gable                         # gable | hip | shed | flat
    overhang: 0.4
    material: roof_shingle

surfaces:                                # 2D regions other objects can be placed on
  front_wall: { from: walls.front }
```

## Composing scenes

Scenes place assets (and other scenes) using slots, attachment points or surfaces:

```yaml
name: cottage
place:
  house:
    asset: house_peaked.yaml
  door:
    asset: door.yaml
    on: house.front_wall                 # cuts its own opening into the wall
    at: { u: 0.5, v: { abs: 0 } }
  window_front_left:
    asset: window.yaml
    on: house.front_wall
    at: { u: 0.2, v: { abs: 0.85 } }
```

```yaml
name: dining_set
place:
  table:
    asset: table.yaml
  chairs:
    asset: chair.yaml
    attach_to: table
    at: [seat_front, seat_back, seat_left, seat_right]   # one chair per attachment point
```

A district is a `city:` block. Buildings come from recipes generated to fit each lot:

```yaml
name: town
city:
  blocks: [3, 2]
  block_size: [44, 36]
  avenues: { ew: [1] }                   # the middle east-west street is an avenue
  landmarks: { "1,0": { recipe: hotel, storeys: 4, interior: lobby } }
  parks: ["2,1"]
  buildings:
    residential: [{ recipe: detached_house, storeys: [1, 2] }]
    commercial:  [{ recipe: shop_row, storeys: [2, 3], interior: lobby }]
  furniture:
    lamp: { asset: street_lamp.yaml, spacing: 20 }
    tree: { asset: maple_tree.yaml, spacing: 10 }
```
