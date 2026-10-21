"""Export the Godot runtime for the web (HTML5), with a set of scenes packed in.

    python runtime/godot/tools/export_web.py OUT_DIR [--scenes hub,cottage,...] [--godot BIN]
    python runtime/godot/tools/export_web.py OUT_DIR --project examples/sandwich [--engine DIR]

Builds the scenes into a staging copy of runtime/godot (its own generated/, so what is in
your runtime/godot/generated/ doesn't matter), imports it and exports the "Web" preset
(export_presets.cfg) to OUT_DIR/index.html. The build is the no-threads variant, so it runs
from a static host without cross-origin isolation headers (GitHub Pages); each scene's navmesh
is baked beforehand and packed. Needs Godot 4.7's web export templates. A hub portal to a scene
left out of --scenes does nothing.

With ``--project DIR`` the build is that project's (``geogen.roots``): its catalogue, all of
its scenes unless ``--scenes`` says otherwise, starting in its default scene. ``--engine DIR``
puts the engine (the wasm, its loader and the audio worklets, the same for every build of one
Godot version) in DIR instead of beside index.html, and the page loads it from there, so
several builds on one site share one download: the project's own files (the pack, icons) stay
in OUT_DIR. OUT_DIR/build.json describes the build (name, description, default scene) for the
catalogue site (docs/make_site.py), which lists the builds it finds.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = "hub,cottage,farm,island,quest_yard"
DEFAULT_GODOT = "/Applications/Godot.app/Contents/MacOS/Godot"
# What the exported page resolves off the engine's base path (godot.js ``locateFile`` and
# ``getModuleConfig``), so what a shared engine directory holds; the pack is named separately.
ENGINE_SUFFIXES = (".js", ".wasm", ".audio.worklet.js", ".audio.position.worklet.js", ".worker.js", ".side.wasm")
BUILD_INFO = "build.json"


def stage(scenes: list[str] | None, into: Path) -> tuple[list[str], dict]:
    """Copy the runtime project to ``into`` and export ``scenes`` (None: the catalogue's
    default set, see ``main``) into its generated/. Returns the scene names exported and the
    build's description for ``build.json``."""
    from geogen import roots
    from geogen.catalogue import export_catalogue, load_catalogue
    from geogen.main import _build_registry

    shutil.copytree(RUNTIME, into, ignore=shutil.ignore_patterns(".godot", "generated", "*.import"))
    generated = into / "generated"
    generated.mkdir()
    catalogue = load_catalogue()
    registry = _build_registry()
    entries = catalogue.select(",".join(scenes) if scenes else "all")
    export_catalogue(catalogue, lambda name: registry[name](), generated, entries)
    # Packed as they are: the runtime parses the GLBs itself (no .gdignore here, or the export
    # would leave generated/ out, and "keep" so the editor doesn't import them as scenes).
    for glb in generated.rglob("*.glb"):
        glb.with_name(glb.name + ".import").write_text('[remap]\n\nimporter="keep"\n')
    projects = roots.projects()
    if projects:
        info = {"name": projects[0].name, "description": projects[0].description}
    else:
        info = {"name": "showcases", "description": "The showcase scenes, from the hub"}
    info["default"] = catalogue.default
    info["scenes"] = {e.name: e.description for e in entries}
    return [e.name for e in entries], info


def share_engine(out: Path, engine: Path) -> list[Path]:
    """Move the engine files of the export in ``out`` to ``engine`` (checking they match any
    already there) and point ``out/index.html`` at them. Returns the files moved."""
    engine.mkdir(parents=True, exist_ok=True)
    moved = []
    for suffix in ENGINE_SUFFIXES:
        source = out / f"index{suffix}"
        if not source.is_file():
            continue
        target = engine / source.name
        if target.is_file() and _digest(target) != _digest(source):
            raise SystemExit(f"{target} is not the same engine as {source}: builds sharing an engine "
                             "must use the same Godot version and export options")
        shutil.move(str(source), target)
        moved.append(target)
    page = out / "index.html"
    page.write_text(point_at_engine(page.read_text(), os.path.relpath(engine, out).replace(os.sep, "/")))
    return moved


def point_at_engine(page: str, engine: str) -> str:
    """``page`` (an exported index.html) loading the engine from ``engine/`` (relative to the
    page) and its own pack from beside itself."""
    base = f"{engine}/index"
    match = re.search(r"const GODOT_CONFIG = (\{.*?\});\n", page)
    if not match:
        raise ValueError("no GODOT_CONFIG in the exported page")
    config = json.loads(match.group(1))
    executable = config.get("executable", "index")
    config["mainPack"] = config.get("mainPack") or f"{executable}.pck"
    config["executable"] = base
    sizes = config.get("fileSizes", {})
    if f"{executable}.wasm" in sizes:
        sizes[f"{base}.wasm"] = sizes.pop(f"{executable}.wasm")
    page = page[:match.start(1)] + json.dumps(config) + page[match.end(1):]
    tag = re.search(r'<script src="index\.js"></script>', page)
    if not tag:
        raise ValueError("no index.js script tag in the exported page")
    return page[:tag.start()] + f'<script src="{base}.js"></script>' + page[tag.end():]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="directory for index.html and the files beside it")
    parser.add_argument("--scenes", default=None,
                        help=f"catalogue scenes to pack (default {DEFAULT_SCENES}; with --project, all of its)")
    parser.add_argument("--project", type=Path, default=None, help="build this project (its geogen.yaml directory)")
    parser.add_argument("--engine", type=Path, default=None,
                        help="put the engine files here, shared with other builds, and load them from there")
    parser.add_argument("--godot", default=os.environ.get("GODOT", DEFAULT_GODOT), help="Godot 4.7 binary ($GODOT)")
    parser.add_argument("--keep", action="store_true", help="keep the staging project (its path is printed)")
    args = parser.parse_args()

    from geogen import roots
    roots.use(*([args.project] if args.project else []))     # the project named, or the core alone
    if args.scenes:
        scenes = [s for s in args.scenes.split(",") if s]
    else:
        scenes = None if args.project else DEFAULT_SCENES.split(",")

    staging = Path(tempfile.mkdtemp(prefix="geogen-web-")) / "runtime"
    try:
        scenes, info = stage(scenes, staging)
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
    (args.out / BUILD_INFO).write_text(json.dumps(info, indent=2) + "\n")
    if args.engine:
        share_engine(args.out, args.engine)
    for f in sorted(args.out.iterdir()):
        print(f"{f.stat().st_size / 1e6:8.1f} MB  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
