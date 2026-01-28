"""Nature scene with terrain, rocks, and vegetation."""

from ..core.node import SceneNode
from ..core.transform import Transform
from ..generators.terrain import TerrainGenerator
from ..layout import LayoutLoader
from ..materials.loader import MaterialLoader
import numpy as np


def create_nature_scene() -> SceneNode:
    """Create a nature scene with terrain, rocks, and bushes."""
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
        pass

    terrain_node = SceneNode("terrain", mesh=terrain_mesh)
    terrain_node.transform.translation = np.array([0, 0, 0], dtype=np.float64)
    root.add_child(terrain_node)

    # Load rocks and bushes
    loader = LayoutLoader()

    # Place rocks at various positions
    rock_positions = [
        (-3, 0.3, -2),
        (2, 0.5, -3),
        (-1, 0.2, 3),
        (4, 0.4, 1),
    ]

    for i, pos in enumerate(rock_positions):
        try:
            rock_type = ["rock_small", "rock_medium", "rock_large"][i % 3]
            rock = loader.load(f"assets/{rock_type}.yaml")
            rock.name = f"rock_{i}"
            rock.transform.translation = np.array(pos, dtype=np.float64)
            root.add_child(rock)
        except FileNotFoundError:
            pass

    # Place bushes
    bush_positions = [
        (-2, 0.3, 1),
        (1, 0.4, 2),
        (-4, 0.2, -1),
        (3, 0.5, -2),
    ]

    for i, pos in enumerate(bush_positions):
        try:
            bush = loader.load("assets/bush.yaml")
            bush.name = f"bush_{i}"
            bush.transform.translation = np.array(pos, dtype=np.float64)
            root.add_child(bush)
        except FileNotFoundError:
            pass

    return root
