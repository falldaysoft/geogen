#!/usr/bin/env python3
"""Export every catalogue asset as a GLB for the site's 3D viewer (runtime/web).

    python docs/make_models.py OUT_DIR [--only NAME,NAME]

Writes ``OUT_DIR/models/<asset>.glb`` (default params, as the thumbnails show
them) with the textures in ``OUT_DIR/models/textures/``, shared between assets
by content hash, and ``OUT_DIR/models/index.json`` listing what was written.
``docs/make_site.py`` gives asset pages a 3D view when ``OUT_DIR/models`` is
there. Assets that fail to build or export are reported and left out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path, help="site directory (models/ goes inside it)")
    parser.add_argument("--only", default="", help="comma-separated asset names (default: every asset)")
    args = parser.parse_args()

    from geogen.core.node import SceneNode
    from geogen.describe import catalogue, load_asset
    from geogen.export import export_scene

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
    (models / "index.json").write_text(json.dumps({"assets": sorted(written)}, indent=1) + "\n")
    print(f"exported {len(written)} models to {models}" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
