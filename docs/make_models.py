#!/usr/bin/env python3
"""Export the catalogue's assets and some scenes as GLBs for the site's 3D viewer (runtime/web).

    python docs/make_models.py OUT_DIR [--only NAME,NAME] [--scenes NAME,NAME]

Writes ``OUT_DIR/models/<asset>.glb`` (default params, as the thumbnails show
them) and ``OUT_DIR/models/<scene>.scene.glb`` with its manifest (spawns, the
player's size, light state and switch bindings, which walking needs), with the
textures in ``OUT_DIR/models/textures/``, shared by content hash, and
``OUT_DIR/models/index.json`` listing what was written. ``docs/make_site.py``
gives asset pages a 3D view when ``OUT_DIR/models`` is there and lists the scenes
on the scenes page. Anything that fails to build or export is reported and left out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Worlds small enough to download on a visit (a few MB to about 30): rooms, a house, open air.
SCENES = ("cottage", "hotel_room_auto", "wants_house", "quest_yard", "park")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path, help="site directory (models/ goes inside it)")
    parser.add_argument("--only", default="", help="comma-separated asset names (default: every asset)")
    parser.add_argument("--scenes", default=",".join(SCENES), help="comma-separated scenes (default: %(default)s)")
    args = parser.parse_args()

    from geogen.core.node import SceneNode
    from geogen.describe import catalogue, load_asset
    from geogen.export import export_scene
    from geogen.main import _build_registry

    models = args.out / "models"
    models.mkdir(parents=True, exist_ok=True)
    cat = catalogue()
    only = {name for name in args.only.split(",") if name}
    written, failed = [], []
    for name, entry in cat["assets"].items():
        if only and name not in only:
            continue
        try:
            root = SceneNode(name)
            root.add_child(load_asset(entry).instance())
            export_scene(root, models / f"{name}.glb", manifest=False, textures_dir=models / "textures")
            written.append(name)
        except Exception as exc:   # one broken asset shouldn't take the site down
            failed.append(name)
            print(f"{name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    registry = _build_registry()
    scenes = []
    for name in (name for name in args.scenes.split(",") if name):
        try:
            export_scene(registry[name](), models / f"{name}.scene.glb", textures_dir=models / "textures", package=name)
            scenes.append(name)
        except Exception as exc:   # as for assets
            failed.append(name)
            print(f"{name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    (models / "index.json").write_text(json.dumps({"assets": sorted(written), "scenes": scenes}, indent=1) + "\n")
    print(f"exported {len(written)} models and {len(scenes)} scenes to {models}"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
