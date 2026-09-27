"""The runtime scene catalogue (``assets/runtime_scenes.yaml``).

A ``kind: scene_catalogue`` file lists the scenes the Godot runtime offers,
each a registered scene name with a ``group`` (``showcase`` or ``test``), a
one-line ``description`` and export options (``stream``, ``lods``), plus the
``default`` scene the runtime loads when it isn't given ``--scene``.

``export_catalogue`` exports entries into a generated directory and
``write_index`` writes ``catalogue.json`` there (format ``geogen-catalogue``
v1) for the runtime, which has no YAML parser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .layout.yaml_utils import safe_load_path

CATALOGUE_PATH = Path(__file__).parent.parent.parent / "assets" / "runtime_scenes.yaml"
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


def load_catalogue(path: str | Path = CATALOGUE_PATH) -> Catalogue:
    data = safe_load_path(path) or {}
    if data.get("kind") != "scene_catalogue":
        raise ValueError(f"{path}: expected kind: scene_catalogue")
    unknown_top = set(data) - {"kind", "default", "scenes"}
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
    return Catalogue(default, entries)


def entry_manifest(entry: CatalogueEntry) -> str:
    """The manifest path the runtime loads for an entry, relative to the generated directory."""
    if entry.stream:
        return f"{entry.name}_chunks/{entry.name}.chunks.json"
    return f"{entry.name}.manifest.json"


def write_index(catalogue: Catalogue, out_dir: str | Path) -> Path:
    """Write ``catalogue.json`` (every entry, with whether its export exists yet) into ``out_dir``."""
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
    """Build and export ``entries`` (default: all) with ``build(name)``, then write the index."""
    for entry in list(entries) if entries is not None else list(catalogue.entries.values()):
        log(f"exporting {entry.name}{' (streamed)' if entry.stream else ''}...")
        path = export_entry(entry, build(entry.name), out_dir)
        log(f"  -> {path}")
    return write_index(catalogue, out_dir)
