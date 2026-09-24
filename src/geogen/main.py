"""Main entry point for geogen."""

import argparse
from pathlib import Path

import numpy as np

from .registry import SceneRegistry
from .scenes.nature import create_nature_scene
from .render import VIEWS, RenderOptions, render_scene, render_views
from .viewer import run_viewer

GODOT_GENERATED = Path(__file__).parent.parent.parent / "runtime" / "godot" / "generated"


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
    parser.add_argument(
        "--view",
        choices=sorted(VIEWS),
        default="iso",
        help="Camera view preset for --render (default: iso)",
    )
    parser.add_argument(
        "--views",
        metavar="V1,V2,...",
        nargs="?",
        const="iso,front,side,top",
        help="Render a contact sheet of several views (default set: iso,front,side,top)",
    )
    parser.add_argument(
        "-e", "--export",
        metavar="PATH",
        help="Export the scene to .glb/.gltf/.obj and quit",
    )
    parser.add_argument(
        "--export-godot",
        action="store_true",
        help=f"Export the scene as <scene>.glb + manifest into {GODOT_GENERATED} and quit",
    )
    parser.add_argument(
        "--viewer-screenshot",
        metavar="PATH",
        help="Open the interactive viewer, save one frame to PATH and quit",
    )
    parser.add_argument(
        "--display",
        choices=["lit", "clay", "normals", "uv"],
        default="lit",
        help="Viewer display mode (default: lit)",
    )
    parser.add_argument("--zoom", type=float, default=1.0, help="Camera zoom factor for --render")
    parser.add_argument("--no-shadows", action="store_true", help="Disable shadows in --render")
    parser.add_argument("--no-ground", action="store_true", help="Don't add a ground plane in --render")
    parser.add_argument("--cutaway", action="store_true",
                        help="Remove ceilings, roofs and ceiling lights to see inside (--render and viewer)")
    parser.add_argument("--storey", type=int, default=None,
                        help="Viewer: show building storeys up to this index")
    parser.add_argument("--state", default=None,
                        help="Viewer: pose every interaction in this state (e.g. open)")
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

    if args.export or args.export_godot:
        from .export import export_scene

        targets = [args.export] if args.export else []
        if args.export_godot:
            targets.append(GODOT_GENERATED / f"{args.scene}.glb")
        for target in targets:
            path = export_scene(root, target)
            print(f"\nExported {args.scene} to {path}")
        if not args.render:
            return

    if args.render:
        if args.cutaway:
            from .render import cutaway

            root = cutaway(root)
        width, height = map(int, args.resolution.split("x"))
        output_path = Path(args.render)
        options = RenderOptions(
            width=width,
            height=height,
            fov=args.fov,
            shadows=not args.no_shadows,
            ground=not args.no_ground,
            camera=np.array([float(x) for x in args.camera.split(",")]) if args.camera else None,
            target=np.array([float(x) for x in args.target.split(",")]) if args.target else None,
            zoom=args.zoom,
        )
        if args.views:
            views = [v.strip() for v in args.views.split(",")]
            print(f"\nRendering {len(views)} views to {output_path} ({width}x{height} each)...")
            image = render_views(root, views, options)
        else:
            print(f"\nRendering to {output_path} ({width}x{height})...")
            image = render_scene(root, options, view=args.view)
        image.save(str(output_path))
        print(f"Saved render to {output_path}")
    else:
        # Show interactive viewer with scene selection menu
        print("\nOpening viewer...")
        print("Controls: Left-drag to rotate, scroll to zoom, right-drag to pan")
        run_viewer(
            scenes=registry.scenes,
            default_scene=args.scene,
            screenshot=args.viewer_screenshot,
            view=None if args.view == "iso" else {"side": "right"}.get(args.view, args.view),
            display_mode=["lit", "clay", "normals", "uv"].index(args.display),
            cutaway=args.cutaway,
            storey=args.storey,
            state=args.state,
        )


if __name__ == "__main__":
    main()
