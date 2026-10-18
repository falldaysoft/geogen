"""Export the Godot runtime for the web (HTML5), with a set of scenes packed in.

    python runtime/godot/tools/export_web.py OUT_DIR [--scenes hub,cottage,...] [--godot BIN]

Builds the scenes into a staging copy of runtime/godot (its own generated/, so what is in
your runtime/godot/generated/ doesn't matter), imports it and exports the "Web" preset
(export_presets.cfg) to OUT_DIR/index.html. The build is the no-threads variant, so it runs
from a static host without cross-origin isolation headers (GitHub Pages); each scene's navmesh
is baked beforehand and packed. Needs Godot 4.7's web export templates. A hub portal to a scene left out of --scenes does nothing.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = "hub,cottage,farm,island,quest_yard"
DEFAULT_GODOT = "/Applications/Godot.app/Contents/MacOS/Godot"


def stage(scenes: list[str], into: Path) -> None:
    """Copy the runtime project to ``into`` and export ``scenes`` into its generated/."""
    from geogen.catalogue import export_catalogue, load_catalogue
    from geogen.main import _build_registry

    shutil.copytree(RUNTIME, into, ignore=shutil.ignore_patterns(".godot", "generated", "*.import"))
    generated = into / "generated"
    generated.mkdir()
    catalogue = load_catalogue()
    registry = _build_registry()
    export_catalogue(catalogue, lambda name: registry[name](), generated, catalogue.select(",".join(scenes)))
    # Packed as they are: the runtime parses the GLBs itself (no .gdignore here, or the export
    # would leave generated/ out, and "keep" so the editor doesn't import them as scenes).
    for glb in generated.rglob("*.glb"):
        glb.with_name(glb.name + ".import").write_text('[remap]\n\nimporter="keep"\n')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="directory for index.html and the files beside it")
    parser.add_argument("--scenes", default=DEFAULT_SCENES, help=f"catalogue scenes to pack (default {DEFAULT_SCENES})")
    parser.add_argument("--godot", default=os.environ.get("GODOT", DEFAULT_GODOT), help="Godot 4.7 binary ($GODOT)")
    parser.add_argument("--keep", action="store_true", help="keep the staging project (its path is printed)")
    args = parser.parse_args()

    staging = Path(tempfile.mkdtemp(prefix="geogen-web-")) / "runtime"
    try:
        scenes = [s for s in args.scenes.split(",") if s]
        stage(scenes, staging)
        subprocess.run([args.godot, "--headless", "--path", staging, "--import"], check=True)
        # Bake each scene's navmesh here (generated/.navcache, packed with it): the web build's
        # res:// is read-only, so it couldn't keep a bake of its own and would bake at every load.
        for scene in scenes:
            subprocess.run([args.godot, "--headless", "--path", staging, "--", f"--scene={scene}", "--quit-after=1"],
                           check=True, stdout=subprocess.DEVNULL)
        args.out.mkdir(parents=True, exist_ok=True)
        subprocess.run([args.godot, "--headless", "--path", staging, "--export-release", "Web",
                        (args.out / "index.html").resolve()], check=True)
    finally:
        if args.keep:
            print(f"staging project: {staging}")
        else:
            shutil.rmtree(staging.parent, ignore_errors=True)
    for f in sorted(args.out.iterdir()):
        print(f"{f.stat().st_size / 1e6:8.1f} MB  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
