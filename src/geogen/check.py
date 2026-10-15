"""``geogen check``: the QA a project needs, without the core's test suite.

    geogen check [--project DIR] [SCENE ...] [--checks mesh,layout,...] [--all] [--json]

Builds each scene (by default the active projects' own; every registered scene when no
project is active, or with ``--all``) and runs every check ``--inspect`` has
(``inspection.CHECKS``: mesh validity, metric UVs, layout, affordances, coplanar faces,
object overlaps, traffic lanes, the extras contract, state and quest compilation), then
checks that the projects' own materials tile (``textures``). Prints one line per finding
and exits 1 when there is any, so a project's CI can run it. Notes (``NOTES``: what a check
couldn't measure) are printed but don't fail it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import roots
from .inspection import CHECKS, Finding, format_findings, run_checks

MATERIAL_CHECK = "textures"
# Finding kinds that are worth saying but don't fail a check (what couldn't be measured).
NOTES = {"overlap_unmeasured"}
SEAM_LIMIT = 1.1                  # a wrap edge may be up to this much sharper than any step inside


def seam_ratio(img: np.ndarray, axis: int) -> float:
    """Wrap-around edge difference relative to the largest interior row/column step.

    Patterned textures (plank gaps, mortar) legitimately have sharp steps, so the wrap
    edge only counts as a seam if it is sharper than any step inside the texture.
    """
    a = img.astype(np.float64)
    other = tuple(i for i in range(a.ndim) if i != axis)
    steps = np.abs(np.diff(a, axis=axis)).mean(axis=other)
    first = np.take(a, 0, axis=axis)
    last = np.take(a, -1, axis=axis)
    return float(np.abs(first - last).mean() / max(steps.max(), 1e-6))


def material_findings(name: str, size: int = 256) -> list[Finding]:
    """A material's albedo must tile: metric UVs repeat it across every surface."""
    from .materials.loader import MaterialLoader

    material = MaterialLoader().load(name)
    material.texture_size = (size, size)
    img = np.asarray(material.get_texture())
    return [Finding(MATERIAL_CHECK, "seam", f"{name}: a seam where the texture wraps along "
                    f"{'v' if axis == 0 else 'u'} ({ratio:.2f}x its sharpest inner step)", (name,))
            for axis in (0, 1) for ratio in [seam_ratio(img, axis)] if ratio >= SEAM_LIMIT]


def own_scenes(registry: Any, everything: bool = False) -> list[str]:
    """The scenes to check: the projects' own, or all of them (no project, or ``everything``)."""
    if everything or not roots.projects():
        return registry.names()
    mine = [p.root for p in roots.projects()]
    return [name for name in registry.names()
            if (path := registry.path(name)) is not None and any(path.resolve().is_relative_to(r) for r in mine)]


def own_materials(everything: bool = False) -> list[str]:
    """Material names the projects define (every material when there is no project, or ``everything``)."""
    paths = roots.glob("materials/*.yaml")
    if not everything and roots.projects():
        mine = [p.root for p in roots.projects()]
        paths = [p for p in paths if any(p.resolve().is_relative_to(r) for r in mine)]
    return sorted({p.stem for p in paths})


def check(scenes: list[str], registry: Any, checks: tuple[str, ...] = (*CHECKS, MATERIAL_CHECK),
          materials: list[str] | None = None, progress: Any = None) -> dict[str, list[Finding]]:
    """{scene or "materials": findings}; a scene that fails to build is one ``build`` finding."""
    out: dict[str, list[Finding]] = {}
    scene_checks = tuple(c for c in checks if c != MATERIAL_CHECK)
    for name in scenes:
        if progress:
            progress(name)
        try:
            root = registry[name]()
        except Exception as exc:  # noqa: BLE001 - one broken scene shouldn't hide the rest
            out[name] = [Finding("build", "error", f"{type(exc).__name__}: {exc}")]
            continue
        out[name] = run_checks(root, scene_checks, package=name) if scene_checks else []
    if MATERIAL_CHECK in checks and materials:
        found: list[Finding] = []
        for name in materials:
            try:
                found.extend(material_findings(name))
            except Exception as exc:  # noqa: BLE001
                found.append(Finding(MATERIAL_CHECK, "error", f"{name}: {type(exc).__name__}: {exc}", (name,)))
        out["materials"] = found
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geogen check", description=__doc__.split("\n\n")[0])
    parser.add_argument("scenes", nargs="*", metavar="SCENE", help="scenes to check (default: the project's own)")
    parser.add_argument("--project", action="append", metavar="DIR", help="a project to check (repeatable)")
    parser.add_argument("--checks", default=",".join((*CHECKS, MATERIAL_CHECK)), metavar="C1,C2",
                        help=f"checks to run (default: all of {','.join((*CHECKS, MATERIAL_CHECK))})")
    parser.add_argument("--all", action="store_true", help="every registered scene and material, the core's too")
    parser.add_argument("--json", action="store_true", help="findings as JSON")
    args = parser.parse_args(argv)
    if args.project:
        try:
            roots.use(*args.project)
        except roots.RootsError as exc:
            parser.error(str(exc))
    checks = tuple(c.strip() for c in args.checks.split(",") if c.strip())
    unknown = set(checks) - {*CHECKS, MATERIAL_CHECK}
    if unknown:
        parser.error(f"unknown checks {', '.join(sorted(unknown))} (choose from {', '.join((*CHECKS, MATERIAL_CHECK))})")

    from .main import _build_registry

    registry = _build_registry()
    missing = [n for n in args.scenes if n not in registry]
    if missing:
        parser.error(f"no scene {', '.join(missing)}")
    scenes = args.scenes or own_scenes(registry, args.all)
    materials = own_materials(args.all) if not args.scenes or args.all else []
    progress = None if args.json else (lambda name: print(f"checking {name} ...", file=sys.stderr, flush=True))
    results = check(scenes, registry, checks, materials, progress)
    failing = {k: [f for f in v if f.kind not in NOTES] for k, v in results.items()}
    notes = {k: [f for f in v if f.kind in NOTES] for k, v in results.items()}
    total = sum(len(f) for f in failing.values())
    if args.json:
        print(json.dumps({"findings": {k: [f.__dict__ for f in v] for k, v in failing.items()},
                          "notes": {k: [f.__dict__ for f in v] for k, v in notes.items() if v}, "total": total},
                         indent=1, default=str))
    else:
        for name in results:
            if failing[name]:
                print(f"{name}:")
                print("\n".join("  " + line for line in format_findings(failing[name])))
            for note in notes[name]:
                print(f"{name}: note: {note.message}")
        materials_note = f", {len(materials)} materials" if MATERIAL_CHECK in checks and materials else ""
        print(f"geogen check: {len(scenes)} scenes{materials_note}: "
              + (f"{total} findings in {sum(1 for f in failing.values() if f)}" if total else "no findings"))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
