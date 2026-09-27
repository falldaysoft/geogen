"""Main entry point for geogen."""

import argparse
from pathlib import Path

import numpy as np

from .registry import SceneRegistry
from .scenes.nature import create_nature_scene
from .scenes.skin_test import create_skin_test_scene
from .render import VIEWS, RenderOptions, render_scene, render_views
from .viewer import run_viewer

GODOT_GENERATED = Path(__file__).parent.parent.parent / "runtime" / "godot" / "generated"


def _build_registry() -> SceneRegistry:
    """Build the scene registry with auto-discovered and Python-coded scenes."""
    registry = SceneRegistry()
    registry.discover()
    # Python-coded scenes (not representable as pure YAML)
    registry.register("nature", create_nature_scene)
    registry.register("skin_test", create_skin_test_scene)
    return registry


def _refresh_catalogue_index() -> None:
    """Keep the runtime's catalogue.json in step with what's been exported."""
    from .catalogue import load_catalogue, write_index
    from .travel import check_travel

    catalogue = load_catalogue()
    write_index(catalogue, GODOT_GENERATED)
    for problem in check_travel(catalogue, GODOT_GENERATED):
        print(f"warning: {problem}")


def filmstrip(root, clip_name: str, frames: int, view: str):
    """``frames`` instances of ``root`` posed evenly across clip ``clip_name``, spaced across the view."""
    from .core.node import SceneNode

    clips = [c for n in root.iter_nodes() for c in n.clips if c.name == clip_name]
    if not clips:
        raise SystemExit(f"No clip named '{clip_name}'")
    duration = clips[0].duration
    lo, hi = np.min([m.vertices.min(axis=0) for _, m in root.iter_meshes()], axis=0), \
        np.max([m.vertices.max(axis=0) for _, m in root.iter_meshes()], axis=0)
    axis = 2 if view in ("side", "right", "left") else 0
    spacing = max(float(hi[axis] - lo[axis]) * 1.15, 0.8)
    strip = SceneNode(f"{root.name}_{clip_name}_filmstrip")
    from .core.skin import pose_clips

    for i in range(frames):
        copy = root.instance()
        pose_clips(copy, clip_name, duration * i / frames)
        offset = np.zeros(3)
        offset[axis] = i * spacing * (-1 if axis == 2 else 1)
        copy.transform.translation = copy.transform.translation + offset
        strip.add_child(copy)
    return strip


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
    parser.add_argument("--night", action="store_true", help="Viewer: night lighting (scene fixtures)")
    parser.add_argument("--lods", default=None, metavar="R1,R2",
                        help="Export: add decimated LODs at these triangle ratios (GLB, MSFT_lod), e.g. 0.5,0.25")
    parser.add_argument("--cache", nargs="?", const=".cache/geogen", default=None, metavar="DIR",
                        help="Cache generated assets on disk (default dir .cache/geogen); any source or asset "
                             "change invalidates it")
    parser.add_argument("--export-catalogue", nargs="?", const="all", default=None, metavar="SELECT",
                        help="Export runtime scenes from assets/runtime_scenes.yaml into the Godot runtime "
                             "and write its catalogue.json: all (default), showcase, test or NAME,NAME")
    parser.add_argument("--catalogue", action="store_true",
                        help="Only rewrite the Godot runtime's catalogue.json (e.g. after changing its default)")
    parser.add_argument("--stream", action="store_true",
                        help="With --export-godot: write a chunked export (<scene>_chunks/) the runtime streams")
    parser.add_argument("--chunks", default=None, metavar="DIR",
                        help="Export as streamable chunks (per block/building, exterior LODs, interiors) into DIR")
    parser.add_argument("--clip", default=None, metavar="NAME@SECONDS",
                        help="Render: pose every skeletal clip NAME at SECONDS (e.g. sway@1.0)")
    parser.add_argument("--pose", default=None, metavar="NAME",
                        help="Render: put every humanoid in skeletal pose NAME (its pose_NAME clip)")
    parser.add_argument("--affordances", action="store_true",
                        help="Render: pose a character at every affordance (sitting on seats, at windows...) "
                             "and print the affordance QA")
    parser.add_argument("--lanes", action="store_true",
                        help="Render: overlay the traffic lane graph (lanes blue, connectors by turn) "
                             "and print lane-graph and clearance issues")
    parser.add_argument("--filmstrip", type=int, default=0, metavar="N",
                        help="Render with --clip NAME: N copies posed across the clip, side by side")
    parser.add_argument("--state", default=None,
                        help="Viewer: pose every interaction in this state (e.g. open)")
    return parser.parse_args()


def main() -> None:
    """Run the geogen demo."""
    registry = _build_registry()
    args = parse_args(registry)
    if args.cache:
        import os

        os.environ["GEOGEN_CACHE"] = args.cache

    if args.export_catalogue or args.catalogue:
        from .catalogue import export_catalogue, load_catalogue, write_index

        catalogue = load_catalogue()
        if args.catalogue:
            print(f"Wrote {write_index(catalogue, GODOT_GENERATED)}")
            return
        index = export_catalogue(catalogue, lambda name: registry[name](), GODOT_GENERATED,
                                 catalogue.select(args.export_catalogue))
        print(f"Wrote {index} (default: {catalogue.default})")
        return

    root = registry[args.scene]()

    # Display scene info
    print("Geogen - Procedural 3D Geometry Generator")
    print("=" * 40)
    print(f"Scene contains {len(list(root.iter_nodes()))} nodes:")
    for node in root.iter_nodes():
        indent = "  " * node.depth
        mesh_info = f" ({node.mesh.face_count} faces)" if node.mesh else ""
        print(f"{indent}- {node.name}{mesh_info}")

    if args.chunks:
        from .chunks import export_chunks

        index = export_chunks(root, args.chunks, name=args.scene)
        print(f"\nExported {args.scene} chunks; index {index}")
        if not (args.render or args.export or args.export_godot):
            return

    if args.export_godot and args.stream:
        from .chunks import export_chunks

        index = export_chunks(root, GODOT_GENERATED / f"{args.scene}_chunks", name=args.scene)
        print(f"\nExported {args.scene} chunks for Godot; index {index}")
        _refresh_catalogue_index()
        if not (args.render or args.export):
            return

    if args.export or args.export_godot:
        from .export import export_scene

        targets = [args.export] if args.export else []
        if args.export_godot:
            targets.append(GODOT_GENERATED / f"{args.scene}.glb")
        for target in targets:
            lods = [float(v) for v in args.lods.split(",")] if args.lods else None
            path = export_scene(root, target, lods=lods)
            print(f"\nExported {args.scene} to {path}")
        if args.export_godot:
            _refresh_catalogue_index()
        if not args.render:
            return

    if args.render:
        if args.pose:
            from .core.skin import pose_clips

            if not pose_clips(root, f"pose_{args.pose}", 0.0):
                print(f"Warning: no pose '{args.pose}' in {args.scene}")
        if args.clip and args.filmstrip:
            root = filmstrip(root, args.clip.partition("@")[0], args.filmstrip, args.view)
        elif args.clip:
            from .core.skin import pose_clips

            clip_name, _, at = args.clip.partition("@")
            if not pose_clips(root, clip_name, float(at or 0)):
                print(f"Warning: no clip named '{clip_name}' in {args.scene}")
        if args.affordances:
            from .layout.affordance_qa import check_affordances, stage_actors

            npc = next((n for n in root.iter_nodes() if n.meta.get("type") == "npc"), None)
            body = npc.find("body") if npc is not None else None
            for issue in check_affordances(root):
                print(issue)
            root = stage_actors(root, body=body)
        if args.lanes:
            from .materials.loader import MaterialLoader
            from .traffic import build_traffic, check_clearance, check_graph, lane_overlay

            graph = build_traffic(root)
            if graph is None:
                print(f"Warning: {args.scene} has no traffic lanes")
            else:
                for issue in check_graph(graph) + check_clearance(root, graph):
                    print(issue)
                root.add_child(lane_overlay(graph, MaterialLoader()))
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
            night=args.night,
        )


if __name__ == "__main__":
    main()
