"""Content roots: where asset files are looked for.

Geogen's own ``assets/`` is the core root. A project that builds on geogen
keeps its content in a directory of its own, laid out the same way
(``scenes/``, ``materials/``, ``quests/``, ``npcs/``, ``room_types/``, ...),
and that directory is searched before the core, so a project's scenes use
core assets and its own by the same relative names (``asset: door.yaml``,
``asset: puzzles/safe.yaml``, ``material: brass``).

A project is chosen, in this order, by ``--project DIR``, by
``$GEOGEN_PROJECT`` (several directories separated like ``$PATH``, earlier
ones first), or by the nearest ``geogen.yaml`` in or above the working
directory. ``DIR`` holds a ``geogen.yaml``::

    kind: project
    name: sandwich            # default: the directory's name
    assets: assets            # the content root, relative to this file (default: assets)
    overrides: [player.yaml]  # core files this project replaces on purpose

Without a ``geogen.yaml``, ``DIR/assets`` is the root if there is one, else
``DIR`` itself.

A project file with the same relative path as a core file would silently
change core assets that load it, so it is an error unless ``overrides`` lists
it. Everything that takes a file name relative to ``assets/`` resolves it with
``find``; listings go through ``glob``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CORE = Path(__file__).resolve().parents[2] / "assets"
CONFIG = "geogen.yaml"
ENV = "GEOGEN_PROJECT"
CONFIG_KEYS = {"kind", "name", "assets", "overrides", "description"}


class RootsError(ValueError):
    """A project that can't be used: no such directory, a bad geogen.yaml, files that shadow the core."""


@dataclass(frozen=True)
class Project:
    name: str
    directory: Path                 # where geogen.yaml is (or would be)
    root: Path                      # its content root
    overrides: frozenset[str] = field(default_factory=frozenset)


_projects: list[Project] | None = None      # None: not looked for yet


def load_project(directory: str | Path) -> Project:
    """The project in ``directory`` (see the module docstring), checked against the core."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise RootsError(f"project directory {directory} doesn't exist")
    config = directory / CONFIG
    spec: dict = {}
    if config.is_file():
        spec = yaml.safe_load(config.read_text()) or {}
        if not isinstance(spec, dict) or spec.get("kind", "project") != "project":
            raise RootsError(f"{config}: expected a mapping with kind: project")
        unknown = sorted(set(spec) - CONFIG_KEYS)
        if unknown:
            raise RootsError(f"{config}: unknown key(s) {', '.join(unknown)}; keys: {', '.join(sorted(CONFIG_KEYS))}")
    if "assets" in spec:
        root = (directory / str(spec["assets"])).resolve()
        if not root.is_dir():
            raise RootsError(f"{config}: assets directory {root} doesn't exist")
    else:
        root = directory / "assets" if (directory / "assets").is_dir() else directory
    overrides = spec.get("overrides") or []
    if not isinstance(overrides, list) or not all(isinstance(o, str) for o in overrides):
        raise RootsError(f"{config}: overrides is a list of file names relative to the assets directory")
    project = Project(str(spec.get("name") or directory.name), directory, root, frozenset(overrides))
    if root == CORE:
        raise RootsError(f"{directory}: a project's assets can't be geogen's own")
    shadowed = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                      if p.is_file() and p.name != CONFIG and (CORE / p.relative_to(root)).is_file()
                      and p.relative_to(root).as_posix() not in project.overrides)
    if shadowed:
        raise RootsError(f"project '{project.name}' ({root}) has files that are also in geogen's assets: "
                         f"{', '.join(shadowed)}. Rename them, or list them under overrides: in {CONFIG} "
                         f"to replace the core's on purpose")
    missing = sorted(o for o in project.overrides if not (root / o).is_file())
    if missing:
        raise RootsError(f"{config}: overrides names file(s) the project doesn't have: {', '.join(missing)}")
    return project


def use(*directories: str | Path) -> list[Project]:
    """Make these the projects (earlier ones searched first); none: the core alone."""
    global _projects
    projects = [load_project(d) for d in directories]
    _projects = projects
    _changed()
    return projects


def _discover() -> list[Project]:
    listed = [d for d in os.environ.get(ENV, "").split(os.pathsep) if d]
    if listed:
        return [load_project(d) for d in listed]
    here = Path.cwd().resolve()
    for directory in (here, *here.parents):
        if (directory / CONFIG).is_file():
            return [load_project(directory)]
    return []


def projects() -> list[Project]:
    global _projects
    if _projects is None:
        _projects = _discover()
    return list(_projects)


def roots() -> list[Path]:
    """Every content root, in search order: projects, then the core."""
    return [*[p.root for p in projects()], CORE]


def _layered(base: str | Path | None) -> bool:
    """Whether names relative to ``base`` are looked up in every root. A directory that
    isn't a root (a test's temporary assets) stands alone."""
    return base is None or Path(base).resolve() in roots()


def find(name: str | Path, base: str | Path | None = None) -> Path:
    """The file ``name`` (relative to an assets directory) in the first root that has
    it. When none has, the path it would have in ``base`` (default: the core), so the
    caller's own "no such file" error names somewhere sensible."""
    name = Path(name)
    if name.is_absolute():
        return name
    if _layered(base):
        for root in roots():
            if (root / name).exists():
                return root / name
    return Path(base if base is not None else CORE) / name


def glob(pattern: str, base: str | Path | None = None) -> list[Path]:
    """Files matching ``pattern`` in every root, sorted by relative name; a file in an
    earlier root hides the same relative name in a later one."""
    found: dict[str, Path] = {}
    for root in (roots() if _layered(base) else [Path(base)]):
        for path in root.glob(pattern):
            found.setdefault(path.relative_to(root).as_posix(), path)
    return [found[name] for name in sorted(found)]


def root_of(path: str | Path) -> Path | None:
    """The root ``path`` is under, if any."""
    path = Path(path).resolve()
    return next((root for root in roots() if path.is_relative_to(root)), None)


def relative(path: str | Path, base: str | Path | None = None) -> str:
    """``path`` as written in YAML: relative to its root (or to ``base``)."""
    root = root_of(path)
    if root is not None:
        return Path(path).resolve().relative_to(root).as_posix()
    return Path(path).relative_to(base if base is not None else CORE).as_posix()


def _changed() -> None:
    """Forget what was derived from the old roots."""
    from .layout import cache

    cache.source_fingerprint.cache_clear()
