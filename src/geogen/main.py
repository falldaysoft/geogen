"""Main entry point for geogen."""

import argparse
from pathlib import Path

import numpy as np
import pyrender
from PIL import Image

from .scenes import create_chair_scene
from .viewer import Viewer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Geogen - Procedural 3D Geometry Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-r", "--render",
        metavar="PATH",
        help="Render the scene to an image file and quit",
    )
    parser.add_argument(
        "--resolution",
        metavar="WxH",
        default="1920x1080",
        help="Render resolution (default: 1920x1080)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the geogen demo."""
    args = parse_args()

    root = create_chair_scene()

    # Display scene info
    print("Geogen - Procedural 3D Geometry Generator")
    print("=" * 40)
    print(f"Scene contains {len(list(root.iter_nodes()))} nodes:")
    for node in root.iter_nodes():
        indent = "  " * node.depth
        mesh_info = f" ({node.mesh.face_count} faces)" if node.mesh else ""
        print(f"{indent}- {node.name}{mesh_info}")

    # Use textures from materials; fallback color only for untextured meshes
    viewer = Viewer(root, color=(0.7, 0.7, 0.8))

    if args.render:
        # Render to file using pyrender for reliable offscreen rendering
        width, height = map(int, args.resolution.split("x"))
        output_path = Path(args.render)
        print(f"\nRendering to {output_path} ({width}x{height})...")

        # Build pyrender scene manually for better control
        pr_scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

        # Add each mesh from the trimesh scene
        for name, geom in viewer.scene.geometry.items():
            # Convert trimesh geometry to pyrender mesh (smooth=False for face colors)
            pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
            pr_scene.add(pr_mesh)

        # Add camera with good default positioning
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
        # Position camera zoomed out and rotated ~20 degrees from straight-on
        angle = np.radians(20)
        distance = 5.0
        cam_pos = np.array([np.sin(angle) * distance, 2.0, np.cos(angle) * distance])
        target = np.array([0.0, 0.4, 0.0])  # Look at chair center
        up = np.array([0.0, 1.0, 0.0])

        # Build look-at matrix
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward  # Camera looks down -Z
        camera_pose[:3, 3] = cam_pos
        pr_scene.add(camera, pose=camera_pose)

        # Add lighting
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        pr_scene.add(light, pose=camera_pose)

        # Render offscreen
        renderer = pyrender.OffscreenRenderer(width, height)
        color, _ = renderer.render(pr_scene)
        renderer.delete()

        # Save image
        img = Image.fromarray(color)
        img.save(str(output_path))
        print(f"Saved render to {output_path}")
    else:
        # Show interactive viewer
        print("\nOpening viewer...")
        print("Controls: Left-drag to rotate, scroll to zoom, right-drag to pan")
        # Pass a callback to force initial render (workaround for pyglet black screen)
        viewer.show(callback=lambda scene: None)


if __name__ == "__main__":
    main()
