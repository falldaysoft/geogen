"""Nature scene with terrain, rocks, and vegetation."""

import warnings

from ..core.node import SceneNode
from ..core.transform import Transform
from ..generators.terrain import TerrainGenerator
from ..layout import LayoutLoader
from ..materials.loader import MaterialLoader
import numpy as np


def create_nature_scene() -> SceneNode:
    """Create a nature scene with terrain, rocks, bushes, and trees."""
    root = SceneNode("nature_scene")

    # Create terrain
    terrain_gen = TerrainGenerator(
        size_x=15.0,
        size_z=15.0,
        subdivisions_x=60,
        subdivisions_z=60,
        height_scale=1.5,
        octaves=4,
        scale=3.0,
        seed=42,
    )
    terrain_mesh = terrain_gen.generate()

    # Apply grass material to terrain
    material_loader = MaterialLoader()
    try:
        grass_material = material_loader.load("grass")
        terrain_mesh.material = grass_material
    except FileNotFoundError:
        warnings.warn("Grass material not found for terrain", stacklevel=2)

    terrain_node = SceneNode("terrain", mesh=terrain_mesh)
    terrain_node.transform.translation = np.array([0, 0, 0], dtype=np.float64)
    root.add_child(terrain_node)

    # Load rocks and bushes
    loader = LayoutLoader()

    # Place rocks at various positions (larger, more visible)
    rock_positions = [
        (-3, 0.2, -2, 2.0),
        (2, 0.4, -3, 1.5),
        (-1, 0.1, 3, 2.5),
        (4, 0.3, 1, 1.8),
    ]

    for i, (x, y, z, scale) in enumerate(rock_positions):
        try:
            rock_type = ["rock_small", "rock_medium", "rock_large"][i % 3]
            rock = loader.load(f"assets/{rock_type}.yaml")
            rock.name = f"rock_{i}"
            rock.transform.translation = np.array([x, y, z], dtype=np.float64)
            rock.transform.scale = np.array([scale, scale, scale], dtype=np.float64)
            root.add_child(rock)
        except FileNotFoundError:
            warnings.warn(f"Asset '{rock_type}' not found", stacklevel=2)

    # Place bushes (larger scale)
    bush_positions = [
        (-2, 0.2, 1, 2.0),
        (1, 0.3, 2, 1.8),
        (-4, 0.1, -1, 2.2),
        (3, 0.4, -2, 1.6),
        (-5, 0.1, 4, 2.0),
        (5, 0.2, -4, 1.5),
    ]

    for i, (x, y, z, scale) in enumerate(bush_positions):
        try:
            bush = loader.load("assets/bush.yaml")
            bush.name = f"bush_{i}"
            bush.transform.translation = np.array([x, y, z], dtype=np.float64)
            bush.transform.scale = np.array([scale, scale, scale], dtype=np.float64)
            root.add_child(bush)
        except FileNotFoundError:
            warnings.warn("Asset 'bush' not found", stacklevel=2)

    # Place trees
    tree_positions = [
        (-5, 0.1, -4, "maple_tree"),
        (4, 0.3, 4, "pine_tree"),
        (-2, 0.2, -5, "maple_tree"),
        (6, 0.1, -1, "pine_tree"),
        (0, 0.3, 5, "maple_tree"),
    ]

    for i, (x, y, z, tree_type) in enumerate(tree_positions):
        try:
            tree = loader.load(f"assets/{tree_type}.yaml")
            tree.name = f"tree_{i}"
            tree.transform.translation = np.array([x, y, z], dtype=np.float64)
            root.add_child(tree)
        except FileNotFoundError:
            warnings.warn(f"Asset '{tree_type}' not found", stacklevel=2)

    return root
