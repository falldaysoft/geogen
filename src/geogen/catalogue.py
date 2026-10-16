"""The runtime scene catalogue (``assets/runtime_scenes.yaml``; a project has its own, see ``active``).

A ``kind: scene_catalogue`` file lists the scenes the Godot runtime offers,
each a registered scene name with a ``group`` (``showcase`` or ``test``), a
one-line ``description`` and export options (``stream``, ``lods``), plus the
``default`` scene the runtime loads when it isn't given ``--scene``. Its
``packages: [name, ...]`` are the content packages deployed with them
(``geogen.content``: ``packages/<name>.yaml`` beside the catalogue).

``export_catalogue`` exports entries into a generated directory and
``write_index`` writes ``catalogue.json`` there (format ``geogen-catalogue``
v1) for the runtime, which has no YAML parser. ``export_packages`` compiles
the content packages against the exported scenes (``<name>.content.json``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import roots
from .layout.yaml_utils import safe_load_path

CATALOGUE_PATH = roots.CORE / "runtime_scenes.yaml"
INDEX_NAME = "catalogue.json"
INDEX_FORMAT = "geogen-catalogue"
GROUPS = ("showcase", "test")


@dataclass
class CatalogueEntry:
    name: str
    group: str = "test"
    description: str = ""
    stream: bool = False
    lods: list[float] | None = None


@dataclass
class Catalogue:
    default: str
    entries: dict[str, CatalogueEntry] = field(default_factory=dict)
    packages: dict[str, Path] = field(default_factory=dict)    # deployed content packages: name -> its file

    def select(self, selector: str | None = None) -> list[CatalogueEntry]:
        """Entries named by ``selector``: all (None/'all'), a group, or comma-separated names."""
        if selector in (None, "", "all"):
            return list(self.entries.values())
        if selector in GROUPS:
            return [e for e in self.entries.values() if e.group == selector]
        names = [n.strip() for n in selector.split(",") if n.strip()]
        unknown = [n for n in names if n not in self.entries]
        if unknown:
            raise ValueError(f"not in the catalogue: {', '.join(unknown)} (known: {', '.join(self.entries)})")
        return [self.entries[n] for n in names]


CORE_GENERATED = Path(__file__).parent.parent.parent / "runtime" / "godot" / "generated"


def active() -> tuple[Path | None, Path]:
    """What exports are for: (the catalogue file, the generated directory). With a project
    (``geogen.roots``) they are its own ``runtime_scenes.yaml`` (None if it has none: its
    exports then have no index, and the runtime loads them all) and ``generated`` directory;
    otherwise the core's."""

    projects = roots.projects()
    if projects:
        return projects[0].catalogue, projects[0].generated
    return CATALOGUE_PATH, CORE_GENERATED


def load_catalogue(path: str | Path | None = None) -> Catalogue:
    """The catalogue in ``path`` (default: the active one, see ``active``)."""
    if path is None:
        path = active()[0]
        if path is None:
            raise ValueError("this project has no runtime catalogue: add runtime_scenes.yaml "
                             "(kind: scene_catalogue) to its assets directory")
    data = safe_load_path(path) or {}
    if data.get("kind") != "scene_catalogue":
        raise ValueError(f"{path}: expected kind: scene_catalogue")
    unknown_top = set(data) - {"kind", "default", "scenes", "packages"}
    if unknown_top:
        raise ValueError(f"{path}: unknown keys {sorted(unknown_top)}")
    entries = {}
    for name, spec in (data.get("scenes") or {}).items():
        spec = spec or {}
        unknown = set(spec) - {"group", "description", "stream", "lods"}
        if unknown:
            raise ValueError(f"{path}: scene '{name}': unknown keys {sorted(unknown)}")
        group = spec.get("group", "test")
        if group not in GROUPS:
            raise ValueError(f"{path}: scene '{name}': group must be one of {GROUPS}, got {group!r}")
        lods = spec.get("lods")
        entries[name] = CatalogueEntry(name, group, str(spec.get("description", "")), bool(spec.get("stream", False)),
                                       [float(v) for v in lods] if lods else None)
    default = data.get("default")
    if default not in entries:
        raise ValueError(f"{path}: default {default!r} is not one of its scenes")
    listed = data.get("packages") or []
    if not isinstance(listed, list) or not all(isinstance(name, str) for name in listed):
        raise ValueError(f"{path}: packages is a list of content package names (packages/<name>.yaml)")
    packages = {}
    for name in listed:
        from .content import DIRECTORY

        packages[name] = Path(path).parent / DIRECTORY / f"{name}.yaml"
        if name in entries or not packages[name].is_file():
            raise ValueError(f"{path}: package '{name}': " + ("is also a scene" if name in entries
                                                             else f"no file {packages[name]}"))
    return Catalogue(default, entries, packages)


def entry_manifest(entry: CatalogueEntry) -> str:
    """The manifest path the runtime loads for an entry, relative to the generated directory."""
    if entry.stream:
        return f"{entry.name}_chunks/{entry.name}.chunks.json"
    return f"{entry.name}.manifest.json"


def write_index(catalogue: Catalogue, out_dir: str | Path) -> Path:
    """Write ``catalogue.json`` (every entry, with whether its export exists yet) into ``out_dir``.
    ``packages`` lists the deployed content packages as ``export_packages`` left them."""
    from .content import index_entries

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "format": INDEX_FORMAT,
        "version": 1,
        "default": catalogue.default,
        "scenes": [
            {
                "name": e.name,
                "group": e.group,
                "description": e.description,
                "stream": e.stream,
                "manifest": entry_manifest(e),
                "exported": (out_dir / entry_manifest(e)).exists(),
            }
            for e in catalogue.entries.values()
        ],
        "packages": index_entries(catalogue, out_dir),
    }
    path = out_dir / INDEX_NAME
    path.write_text(json.dumps(index, indent=2) + "\n")
    return path


def export_entry(entry: CatalogueEntry, root, out_dir: str | Path) -> Path:
    """Export one built scene the way its entry asks (chunked or a single .glb + manifest)."""
    out_dir = Path(out_dir)
    if entry.stream:
        from .chunks import export_chunks

        return Path(export_chunks(root, out_dir / f"{entry.name}_chunks", name=entry.name))
    from .export import export_scene

    return Path(export_scene(root, out_dir / f"{entry.name}.glb", lods=entry.lods))


def export_catalogue(catalogue: Catalogue, build: Callable[[str], object], out_dir: str | Path,
                     entries: Iterable[CatalogueEntry] | None = None, log=print) -> Path:
    """Build and export ``entries`` (default: all) with ``build(name)``, then write the index.

    Raises ValueError (after writing everything) if a travel point goes to a
    scene outside the catalogue or to a spawn its target doesn't have, or a
    content package doesn't fit the scenes it names.
    """
    for entry in list(entries) if entries is not None else list(catalogue.entries.values()):
        log(f"exporting {entry.name}{' (streamed)' if entry.stream else ''}...")
        path = export_entry(entry, build(entry.name), out_dir)
        log(f"  -> {path}")
    from .content import export_packages
    from .travel import check_travel

    content = export_packages(catalogue, out_dir, log)
    index = write_index(catalogue, out_dir)
    problems = check_travel(catalogue, out_dir)
    if problems:
        raise ValueError("bad travel targets:\n  " + "\n  ".join(problems))
    if content:
        raise ValueError("bad content packages:\n  " + "\n  ".join(content))
    return index
