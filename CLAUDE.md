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

- **SceneComposer** (`composer.py`): Composes multi-asset scenes by attaching objects at attachment points:
  ```yaml
  name: dining_set
  compose:
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

