# Geogen

Procedural 3D geometry generator for game assets. You describe objects and scenes in YAML (parts, profiles, cut-outs, roofs, materials, anchors) and geogen builds watertight, UV-mapped meshes with procedural PBR textures. It renders them, shows them in an interactive viewer, exports them to glTF/OBJ, and runs them in a Godot 4 runtime that you can walk around in first person.

The goal is nested, generated worlds: cities that contain neighbourhoods that contain streets, houses and furnished rooms.

## Examples

**Street**: a composed scene. Each side of the road is a nested `street_side` scene whose house plots place cottages, and the lamps, trees, hydrant and bench are placed in named slots.

![Street](docs/images/geogen_street.png)

**Cottage**: a hollow brick shell with a gable roof and chimney. The door and windows each cut their own opening into the wall they're placed on. Contact sheet from `--views` (iso, front, side, top):

![Cottage contact sheet](docs/images/geogen_cottage_views.png)

**Dining set**: turned (`lathe`) legs and spindles, a rounded (`extrude`) tabletop, and four chairs attached to the table's `seat_*` attachment points:

![Dining set](docs/images/geogen_dining_set.png)

| Chair | Room |
|---|---|
| ![Chair](docs/images/geogen_chair.png) | ![Room](docs/images/geogen_room.png) |

**Godot runtime**: the cottage exported with `--export-godot` and walked in first person (left); F2 shows the collision shapes (right):

| First person | Collider wireframes |
|---|---|
| ![Godot first person](docs/images/godot_first_person.png) | ![Godot colliders](docs/images/godot_colliders.png) |

**Interactive viewer** (Qt/OpenGL) viewport: shadows, PBR, grid, picking:

![Viewer](docs/images/geogen_viewer.png)

## Features

- **YAML assets and scenes**: parts positioned by anchors or attached to each other by named points; `{expr}` parameters (`"{width / 2}"`, `50cm`); scenes compose assets and other scenes through slots, attachments and surfaces (`on: house.front_wall`).
- **Geometry**: rounded-box cubes, cylinders, spheres, ellipsoids and cones; `extrude` (2D shapes with holes and a bevelled cap) and `lathe` (revolved profiles); gable, hip, shed and flat `roof`s and gable-end `prism`s; CSG (`subtract:`, `cutter:`), plus `cut_host:` so a window or door cuts its own opening into whatever wall it's placed on.
- **Quality checks**: meshes are closed and consistently wound, `meshops.validate` checks them, and a test runs it on every asset. UVs are metric (1 unit = 1 m), so texture density doesn't depend on object size.
- **Procedural PBR materials**: wood, metal, brick, concrete, asphalt, roof shingles, grass, rock, dirt, plaster and more, with base colour, normal, roughness and AO maps. Every texture tiles seamlessly.
- **Viewer**: scene list, node tree, inspector (sizes, triangle counts, materials, attachments, surfaces, validation), lit/clay/normals/UV-checker display modes, and hot reload of `assets/**/*.yaml`.
- **Rendering**: offscreen rendering with shadows and PBR, auto-framed, view presets, and multi-view contact sheets.
- **Export**: glTF/GLB with the node hierarchy, PBR textures and a JSON manifest, or OBJ+MTL+PNG.
- **Player spec**: one file (`assets/player.yaml`) defines the player's size (radius, height, eye and step height, max slope, minimum door and corridor clearances, reach). Generation and the runtime both read it, so "enterable" means the same thing on both sides.
- **Godot 4 runtime** (`runtime/godot/`): first-person player sized from the manifest, step-up, collision, fly mode, debug overlay, and live reload when an asset is re-exported.

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For the runtime, install [Godot 4.7](https://godotengine.org/download). The tests look for `godot`/`godot4` on `PATH`, then `$GODOT`, then `/Applications/Godot.app`.

## Usage

Every YAML file in `assets/` is a scene you can pass to `-s` (for example `chair`, `table`, `dining_set`, `room`, `cottage`, `house_peaked`, `street`, `street_lamp`), plus the Python-coded `nature` scene. Run `python -m geogen.main --help` for the full list.

```bash
# Interactive viewer
python -m geogen.main -s dining_set

# Render to a PNG and quit (shadows, PBR, auto-framed)
python -m geogen.main -s room -r out.png --resolution 1280x720
python -m geogen.main -s chair -r out.png --view side --zoom 1.5 --no-ground
python -m geogen.main -s cottage -r sheet.png --views                  # iso/front/side/top
python -m geogen.main -s cottage -r sheet.png --views iso,back,top

# Screenshot the interactive viewer (display: lit | clay | normals | uv)
python -m geogen.main -s street --viewer-screenshot shot.png --display uv

# Export for game engines
python -m geogen.main -s dining_set -e out/dining_set.glb              # .glb / .gltf / .obj
```

### Walking a scene in Godot

```bash
python -m geogen.main -s cottage --export-godot                        # -> runtime/godot/generated/
godot --path runtime/godot -- --scene cottage
```

You spawn in front of the model. WASD moves, Shift sprints, Space jumps, and the mouse looks around (click to capture it, Esc to release). F1 toggles the overlay, F2 the collider wireframes, F3 fly mode, and F4 an overview camera. If you re-export while the game is running, it reloads the model within half a second. See [`runtime/godot/README.md`](runtime/godot/README.md) for the runtime's options and the manifest format.

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

### Composing scenes

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

## Architecture

| Package | Contents |
|---|---|
| `core/` | `SceneNode` hierarchy, `Mesh`, `Transform` (scale → rotate → translate); `meshops` (normals, welding, tangents, `validate`), `uvmap` (metric box/planar/cylindrical projection), `profile` (2D shapes, splines, offsets, triangulation), `csg` (manifold3d booleans) |
| `generators/` | Primitives, `extrude`/`lathe` profiles, `roof`/`prism`, `RoomGenerator` |
| `textures/`, `materials/` | Tiling procedural texture generators; PBR materials loaded from `assets/materials/*.yaml` with a metric `tile_size` |
| `layout/` | `LayoutLoader` (assets), `SceneComposer` (scenes), anchors, attachments, surfaces, the `{expr}` evaluator, validation |
| `viewer/` | Qt window, OpenGL view (shadows, PBR, debug display modes, picking), pure-numpy orbit camera |
| `render.py`, `export.py` | Offscreen pyrender renderer; glTF/GLB/OBJ export and the runtime manifest |
| `player.py` | `PlayerSpec`, loaded from `assets/player.yaml` |
| `registry.py` | Discovers YAML assets/scenes and Python-coded scenes |
| `runtime/godot/` | Godot 4.7 project: runtime glTF loading, first-person player, debug overlay |

[`CLAUDE.md`](CLAUDE.md) has the detailed design notes and conventions.

## Tests

```bash
pytest                                   # everything
pytest tests/test_scenes.py -k chair     # a subset
```

The tests validate every asset's meshes, check that textures tile, round-trip exports, and (when Godot is installed) run the runtime headless. Those runs check that it reads the manifest, and walk the player into the cottage: the wall blocks, the door step is climbed, the closed door blocks. The Godot tests are skipped when Godot isn't found.

To regenerate the screenshots in this README, run `docs/make_screenshots.sh`.

## License

MIT
