"""Main entry point for geogen."""

import argparse
from pathlib import Path

import numpy as np
import pyrender
from PIL import Image

from .registry import SceneRegistry
from .scenes.nature import create_nature_scene
from .viewer import Viewer, run_viewer


def _build_registry() -> SceneRegistry:
    """Build the scene registry with auto-discovered and Python-coded scenes."""
    registry = SceneRegistry()
    registry.discover()
    # Python-coded scenes (not representable as pure YAML)
    registry.register("nature", create_nature_scene)
    return registry


def parse_args(registry: SceneRegistry) -> argparse.Namespace:
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
    parser.add_argument(
        "-s", "--scene",
        choices=registry.names(),
        default="chair",
        help="Scene to display (default: chair)",
    )
    parser.add_argument(
        "--camera",
        metavar="X,Y,Z",
        help="Camera position (default: auto-fit to scene)",
    )
    parser.add_argument(
        "--target",
        metavar="X,Y,Z",
        help="Camera target/look-at point (default: scene center)",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=45.0,
        help="Camera field of view in degrees (default: 45)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the geogen demo."""
    registry = _build_registry()
    args = parse_args(registry)

    root = registry[args.scene]()

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
        # Sky-blue background color
        pr_scene = pyrender.Scene(
            ambient_light=[0.3, 0.3, 0.3],
            bg_color=[0.55, 0.7, 0.85, 1.0],
        )

        # Add each mesh from the trimesh scene
        for name, geom in viewer.scene.geometry.items():
            # Convert trimesh geometry to pyrender mesh (smooth=False for face colors)
            pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
            pr_scene.add(pr_mesh)

        # Compute scene bounds for auto-fitting camera
        scene_bounds = viewer.scene.bounds
        scene_center = (scene_bounds[0] + scene_bounds[1]) / 2
        scene_size = np.linalg.norm(scene_bounds[1] - scene_bounds[0])

        # Parse camera position (or auto-fit)
        if args.camera:
            cam_pos = np.array([float(x) for x in args.camera.split(",")])
        else:
            # Auto-fit: distance based on scene size and FOV
            # Use the FOV to compute the distance needed to frame the scene
            half_fov = np.radians(args.fov) / 2
            distance = (scene_size * 0.5) / np.tan(half_fov) * 0.85
            distance = max(distance, scene_size * 0.6)

            angle = np.radians(35)
            cam_pos = scene_center + np.array([
                np.sin(angle) * distance,
                distance * 0.4,
                np.cos(angle) * distance
            ])

        # Parse target (or use scene center)
        if args.target:
            target = np.array([float(x) for x in args.target.split(",")])
        else:
            target = scene_center.copy()
            # Look slightly above the bottom of the scene
            target[1] = scene_bounds[0][1] + (scene_bounds[1][1] - scene_bounds[0][1]) * 0.35

        up = np.array([0.0, 1.0, 0.0])

        # Camera FOV
        camera = pyrender.PerspectiveCamera(yfov=np.radians(args.fov))

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

        # Add lighting - key light from camera direction
        key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        pr_scene.add(key_light, pose=camera_pose)

        # Fill light from above-behind to soften shadows
        fill_pose = np.eye(4)
        fill_dir = np.array([0.3, -0.8, -0.5])
        fill_dir = fill_dir / np.linalg.norm(fill_dir)
        fill_right = np.cross(fill_dir, np.array([0, 1, 0]))
        fill_right = fill_right / np.linalg.norm(fill_right)
        fill_up = np.cross(fill_right, fill_dir)
        fill_pose[:3, 0] = fill_right
        fill_pose[:3, 1] = fill_up
        fill_pose[:3, 2] = -fill_dir
        fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
        pr_scene.add(fill_light, pose=fill_pose)

        # Render offscreen
        renderer = pyrender.OffscreenRenderer(width, height)
        color, _ = renderer.render(pr_scene)
        renderer.delete()

        # Save image
        img = Image.fromarray(color)
        img.save(str(output_path))
        print(f"Saved render to {output_path}")
    else:
        # Show interactive viewer with scene selection menu
        print("\nOpening viewer...")
        print("Controls: Left-drag to rotate, scroll to zoom, right-drag to pan")
        run_viewer(scenes=registry.scenes, default_scene=args.scene)


if __name__ == "__main__":
    main()
