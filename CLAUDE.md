# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geogen is a procedural 3D geometry generator for game assets. It provides a scene graph system with hierarchical transformations and primitive mesh generators.

## Commands

```bash
# Run the demo (opens interactive viewer)
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
```

## Architecture

### Core System (`src/geogen/core/`)

- **SceneNode** (`node.py`): Hierarchical scene graph with parent-child relationships. Each node has a local Transform, optional Mesh, and children. Provides `world_transform()` for computing combined transformation matrices and `flatten()` to merge all geometry into a single mesh.

- **Mesh** (`mesh.py`): Geometry container storing vertices, faces, normals, and UVs as numpy arrays. Converts to/from trimesh for rendering. Has `transform()` method that handles both vertex and normal transformation correctly.

- **Transform** (`transform.py`): TRS (Translation-Rotation-Scale) transformation. Rotation uses XYZ Euler angles in radians. Matrix order is Scale -> Rotate -> Translate.

- **geometry** (`geometry.py`): Helper functions for face winding and normal computation. Uses counter-clockwise winding convention (CCW when viewed from outside = outward normal). Key functions: `make_cap_faces()`, `make_tube_faces()`, `make_cone_side_faces()`.

### Generators (`src/geogen/generators/`)

- **MeshGenerator** (`base.py`): Abstract base class for generators producing single meshes. Implement `generate() -> Mesh`. Has `to_node()` helper to wrap output in a SceneNode.

- **CompositeGenerator** (`base.py`): Abstract base for generators producing scene hierarchies (multi-part objects).

- **Primitives** (`primitives.py`): Dataclass-based generators for Cube, Sphere, Cylinder, Cone. All use the geometry helpers to ensure correct face winding.

### Layout System (`src/geogen/layout/`)

- **Anchor** (`anchors.py`): Named anchor points for positioning objects within bounding boxes. Uses normalized coordinates (0-1) for X/Y/Z. Key function: `resolve_anchor()` converts anchors to world coordinates.

- **LayoutLoader** (`loader.py`): Loads composite objects from YAML files. YAML format specifies a bounding box size and parts with fractional positioning using anchors. Parts are positioned relative to parent container.

### Scenes (`src/geogen/scenes/`)

- Scene factory functions that assemble complete scenes using generators and the layout system. Example: `create_chair_scene()` loads `assets/chair.yaml`.

### Asset Definitions (`assets/`)

- YAML files defining composite objects using the layout system. Parts use fractional sizes (0-1) relative to the parent bounding box.

### Viewer (`src/geogen/viewer/`)

- **Viewer** (`viewer.py`): Wraps trimesh's pyglet-based viewer. `add_scene_node()` iterates the hierarchy and adds world-transformed meshes.

## Key Conventions

- All meshes use counter-clockwise face winding for outward normals
- Transformations follow standard game engine order: Scale -> Rotate -> Translate
- Geometry is centered at origin by default
- numpy arrays use `float64` for vertices/normals and `int64` for face indices
