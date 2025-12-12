"""Main entry point for geogen."""

import argparse
from pathlib import Path

import numpy as np
import pyrender
from PIL import Image

from .core.node import SceneNode
from .layout import LayoutLoader, SceneComposer
from .scenes import create_chair_scene, create_dining_set_scene, create_room_scene, create_street_scene, create_table_scene


def _get_assets_dir() -> Path:
    """Get the assets directory path."""
    return Path(__file__).parent.parent.parent / "assets"


def _create_asset_scene(asset_name: str) -> SceneNode:
    """Create a scene from a single asset YAML file."""
    root = SceneNode("root")
    loader = LayoutLoader()
    asset = loader.load(_get_assets_dir() / f"{asset_name}.yaml")
    root.add_child(asset)
    return root


def _create_composed_scene(scene_path: str) -> SceneNode:
    """Create a scene from a composed scene YAML file."""
    assets_dir = _get_assets_dir()
    composer = SceneComposer(assets_dir)
    return composer.compose(assets_dir / scene_path)


def create_bench_scene() -> SceneNode:
    """Create a scene containing a park bench."""
    return _create_asset_scene("bench")


def create_fire_hydrant_scene() -> SceneNode:
    """Create a scene containing a fire hydrant."""
    return _create_asset_scene("fire_hydrant")


def create_mailbox_scene() -> SceneNode:
    """Create a scene containing a mailbox."""
    return _create_asset_scene("mailbox")


def create_street_lamp_scene() -> SceneNode:
    """Create a scene containing a street lamp."""
    return _create_asset_scene("street_lamp")


def create_trashcan_scene() -> SceneNode:
    """Create a scene containing a trash can."""
    return _create_asset_scene("trashcan")


def create_house_scene() -> SceneNode:
    """Create a scene containing a simple house."""
    return _create_asset_scene("house_simple")


def create_road_scene() -> SceneNode:
    """Create a scene containing a road section."""
    return _create_asset_scene("road")


def create_sidewalk_scene() -> SceneNode:
    """Create a scene containing a sidewalk."""
    return _create_asset_scene("sidewalk")


def create_ground_scene() -> SceneNode:
    """Create a scene containing a ground plane."""
    return _create_asset_scene("ground")


def create_house_plot_scene() -> SceneNode:
    """Create a scene containing a house plot."""
    return _create_composed_scene("scenes/house_plot.yaml")


def create_street_side_scene() -> SceneNode:
    """Create a scene containing one side of a street."""
    return _create_composed_scene("scenes/street_side.yaml")


from .viewer import Viewer, run_viewer


# Scene registry - maps scene names to factory functions
SCENES = {
    # Furniture
    "chair": create_chair_scene,
    "table": create_table_scene,
    "dining_set": create_dining_set_scene,
    "bench": create_bench_scene,
    # Street furniture
    "fire_hydrant": create_fire_hydrant_scene,
    "mailbox": create_mailbox_scene,
    "street_lamp": create_street_lamp_scene,
    "trashcan": create_trashcan_scene,
    # Buildings
    "house": create_house_scene,
    "room": create_room_scene,
    # Infrastructure
    "road": create_road_scene,
    "sidewalk": create_sidewalk_scene,
    "ground": create_ground_scene,
    # Composed scenes
    "house_plot": create_house_plot_scene,
    "street_side": create_street_side_scene,
    "street": create_street_scene,
}


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
    parser.add_argument(
        "-s", "--scene",
        choices=list(SCENES.keys()),
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
    args = parse_args()

    root = SCENES[args.scene]()

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

        # Compute scene bounds for auto-fitting camera
        scene_bounds = viewer.scene.bounds
        scene_center = (scene_bounds[0] + scene_bounds[1]) / 2
        scene_size = np.linalg.norm(scene_bounds[1] - scene_bounds[0])

        # Parse camera position (or auto-fit)
        if args.camera:
            cam_pos = np.array([float(x) for x in args.camera.split(",")])
        else:
            # Auto-fit: position camera at 45 degrees, distance based on scene size
            distance = scene_size * 0.8
            angle = np.radians(30)
            cam_pos = scene_center + np.array([
                np.sin(angle) * distance,
                distance * 0.5,
                np.cos(angle) * distance
            ])

        # Parse target (or use scene center)
        if args.target:
            target = np.array([float(x) for x in args.target.split(",")])
        else:
            target = scene_center.copy()
            target[1] = scene_center[1] + scene_size * 0.1  # Slightly above center

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
        # Show interactive viewer with scene selection menu
        print("\nOpening viewer...")
        print("Controls: Left-drag to rotate, scroll to zoom, right-drag to pan")
        run_viewer(scenes=SCENES, default_scene=args.scene)


if __name__ == "__main__":
    main()
