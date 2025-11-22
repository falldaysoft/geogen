"""Main entry point for geogen."""

import numpy as np

from .core.node import SceneNode
from .generators import CubeGenerator, SphereGenerator, CylinderGenerator, ConeGenerator
from .viewer import Viewer


def main() -> None:
    """Run the geogen demo."""
    # Create a scene with various primitives
    root = SceneNode("root")

    # Add a cube
    cube = CubeGenerator(size_x=1.0, size_y=1.0, size_z=1.0)
    cube_node = cube.to_node("cube")
    cube_node.transform.translation = np.array([-2.0, 0.0, 0.0])
    root.add_child(cube_node)

    # Add a sphere
    sphere = SphereGenerator(radius=0.5, segments=32, rings=16)
    sphere_node = sphere.to_node("sphere")
    sphere_node.transform.translation = np.array([0.0, 0.0, 0.0])
    root.add_child(sphere_node)

    # Add a cylinder
    cylinder = CylinderGenerator(radius=0.4, height=1.2, segments=32)
    cylinder_node = cylinder.to_node("cylinder")
    cylinder_node.transform.translation = np.array([2.0, 0.0, 0.0])
    root.add_child(cylinder_node)

    # Add a cone
    cone = ConeGenerator(radius=0.5, height=1.0, segments=32)
    cone_node = cone.to_node("cone")
    cone_node.transform.translation = np.array([4.0, 0.0, 0.0])
    root.add_child(cone_node)

    # Display scene info
    print("Geogen - Procedural 3D Geometry Generator")
    print("=" * 40)
    print(f"Scene contains {len(list(root.iter_nodes()))} nodes:")
    for node in root.iter_nodes():
        indent = "  " * node.depth
        mesh_info = f" ({node.mesh.face_count} faces)" if node.mesh else ""
        print(f"{indent}- {node.name}{mesh_info}")

    # Show in viewer
    print("\nOpening viewer...")
    print("Controls: Left-drag to rotate, scroll to zoom, right-drag to pan")

    viewer = Viewer()
    viewer.add_scene_node(root, color=(0.7, 0.7, 0.8))
    viewer.show()


if __name__ == "__main__":
    main()
