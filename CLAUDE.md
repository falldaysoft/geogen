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

# Render to file and quit (for testing)
python -m geogen.main -r output.png
python -m geogen.main -r output.png --resolution 1280x720

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

- **Transform** (`transform.py`): TRS (Translation-Rotation-Scale) transformation. Rotation uses XYZ Euler angles in radians. Matrix order is Scale -> Rotate -> Translate.

- **geometry** (`geometry.py`): Helper functions for face winding and normal computation. Uses CCW winding convention.

### Generators (`src/geogen/generators/`)

- **MeshGenerator** (`base.py`): Abstract base class for generators producing single meshes. Implement `generate() -> Mesh`.

- **CompositeGenerator** (`base.py`): Abstract base for generators producing scene hierarchies.

- **Primitives** (`primitives.py`): Dataclass-based generators for Cube, Sphere, Cylinder, Cone.

- **RoomGenerator** (`room.py`): Generates rooms with walls, floor, ceiling, and openings (doors/windows). Supports `generate_parts()` for separate surface meshes with different materials. Uses `Opening` dataclass for doors/windows with wall position, size, and bottom offset.

### Textures (`src/geogen/textures/`)

- **TextureGenerator** (`base.py`): Abstract base class for procedural textures. Generates PIL Images and optional PBR maps (normal, roughness, AO). Uses numpy RNG with optional seed.

- Implementations: `WoodTexture`, `MetalTexture`, `FloorTexture`, `WallTexture`, plus noise utilities.

### Materials (`src/geogen/materials/`)

- **Material** (`material.py`): Combines a TextureGenerator with PBR properties (roughness, metallic, normal_strength, ao_strength). Caches generated textures. Meshes reference materials for rendering.

- **MaterialLoader** (`loader.py`): Loads material definitions from YAML files in `assets/materials/`.

### Lighting (`src/geogen/lighting/`)

- **Light classes**: `DirectionalLight` (sun-like), `PointLight` (omni). Both have color and intensity.

- **SceneLighting**: Container with ambient color and lights list. Provides `get_shader_data()` for shader uniforms. Presets: `default()` and `room_lighting()`.

### Layout System (`src/geogen/layout/`)

- **Anchor** (`anchors.py`): Named anchor points using normalized coordinates (0-1). Examples: `bottom_center`, `top_front_left`. `resolve_anchor()` converts to world coordinates.

- **AttachmentPoint** (`attachments.py`): Named points for connecting objects. Specifies position via anchor + offset, and orientation via `facing` direction (`center`, `outward`, compass directions) or explicit rotation.

- **LayoutLoader** (`loader.py`): Loads composite objects from YAML. Format:
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

### Viewer (`src/geogen/viewer/`)

- **qt_viewer.py**: PyQt6/OpenGL viewer with scene selection, node tree, and orbit camera. Uses modern shaders (PBR when available, Blinn-Phong fallback). `GLWidget` handles mesh rendering with VAOs/VBOs and texture binding. `ViewerWindow` provides scene selector UI.

- **shaders/**: GLSL vertex and fragment shaders. PBR shader supports albedo, normal, roughness, and AO maps.

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

### Design Principles

1. **No magic numbers in connections**: Parts connect via named points (`top`, `bottom`, `seat_front`), not coordinates
2. **Self-describing objects**: Assets declare their own attachment points based on their semantics
3. **Composability**: Objects don't know about their containers; containers know about objects' attachment points
4. **Hierarchical transforms**: Child transforms are relative to parent, enabling grouped movement

## Key Conventions

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