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
    generated: generated      # where its runtime exports go, relative to this file (default: generated)

Its runtime scene catalogue is ``runtime_scenes.yaml`` in its content root (``geogen.catalogue``):
the project's own list and default scene, which may include core scenes. With one,
``--export-godot`` and ``--export-catalogue`` write to the project's ``generated`` directory
(run it with ``godot --path runtime/godot -- --generated=DIR``); the core's catalogue and
``runtime/godot/generated`` are left alone.

Without a ``geogen.yaml``, ``DIR/assets`` is the root if there is one, else
``DIR`` itself.

A scene can also be a folder, ``scenes/<name>/scene.yaml`` in any root, with
files of its own beside it laid out like a root (``safe.yaml``,
``materials/brass.yaml``, ``quests/open_safe.yaml``). While that scene is being
composed its folder is searched first (``scene_scope``), so it names them as it
names anything else. They are private: no other scene can name them, listings
leave them out, and they may not share a relative name with a file in a root.
Only ``scene.yaml`` itself is public (``scene: scenes/<name>/scene.yaml``).

A project file with the same relative path as a core file would silently
change core assets that load it, so it is an error unless ``overrides`` lists
it. Everything that takes a file name relative to ``assets/`` resolves it with
``find``; listings go through ``glob``.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CORE = Path(__file__).resolve().parents[2] / "assets"
CONFIG = "geogen.yaml"
ENV = "GEOGEN_PROJECT"
CONFIG_KEYS = {"kind", "name", "assets", "overrides", "description", "generated"}
CATALOGUE = "runtime_scenes.yaml"   # a project's own runtime catalogue: beside, not instead of, the core's
SCENES = "scenes"
SCENE_FILE = "scene.yaml"           # scenes/<name>/scene.yaml: a scene folder
ASSET_SUBDIRS = ("vehicles", "characters")   # asset libraries: <dir>/<name>.yaml is asset <name>
SCENE_KEYS = ("place", "city")      # a top-level YAML with one of these is a composed scene


class RootsError(ValueError):
    """A project that can't be used: no such directory, a bad geogen.yaml, files that shadow the core."""


@dataclass(frozen=True)
class Project:
    name: str
    directory: Path                 # where geogen.yaml is (or would be)
    root: Path                      # its content root
    overrides: frozenset[str] = field(default_factory=frozenset)
    generated: Path | None = None   # where its runtime exports go
    description: str = ""           # its one-line description (geogen.yaml)

    @property
    def catalogue(self) -> Path | None:
        """Its runtime scene catalogue, if it has one."""
        path = self.root / CATALOGUE
        return path if path.is_file() else None


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
    project = Project(str(spec.get("name") or directory.name), directory, root, frozenset(overrides),
                      (directory / str(spec.get("generated", "generated"))).resolve(),
                      str(spec.get("description") or ""))
    if root == CORE:
        raise RootsError(f"{directory}: a project's assets can't be geogen's own")
    shadowed = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                      if p.is_file() and p.name != CONFIG and p != root / CATALOGUE and (CORE / p.relative_to(root)).is_file()
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
    """Every content root, in search order: the folder of the scene being composed (if it
    has one), projects, then the core."""
    return [*reversed(_scopes), *[p.root for p in projects()], CORE]


_scopes: list[Path] = []            # folders of the scenes being composed, outermost first


def scene_folder(path: str | Path) -> Path | None:
    """The scene folder ``path`` is in (or is): the nearest ``scenes/<name>/`` holding a ``scene.yaml``."""
    path = Path(path).resolve()
    return next((d for d in (path, *path.parents) if d.parent.name == SCENES and (d / SCENE_FILE).is_file()), None)


def _shared(base: str | Path | None) -> list[Path]:
    """The roots every scene sees: projects and the core, or ``base`` alone when it isn't one of them."""
    everyone = [*[p.root for p in projects()], CORE]
    return everyone if base is None or Path(base).resolve() in roots() else [Path(base)]


@contextlib.contextmanager
def scene_scope(folder: str | Path):
    """Search scene folder ``folder`` first while its scene is composed. Its files may not
    take a name a root already has: a core asset the scene places would load them instead of
    the core's."""
    folder = Path(folder).resolve()
    shared = _shared(folder.parent.parent if root_of(folder) is None else None)
    taken = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                   if p.is_file() and p != folder / SCENE_FILE
                   and any((root / p.relative_to(folder)).is_file() for root in shared))
    if taken:
        raise RootsError(f"scene folder '{folder.name}' ({folder}) has files with the names of shared ones: "
                         f"{', '.join(taken)}. Rename them: a scene's private files can't replace shared assets")
    _scopes.append(folder)
    try:
        yield folder
    finally:
        _scopes.pop()


def scope() -> Path | None:
    """The folder of the scene being composed, if it has one."""
    return _scopes[-1] if _scopes else None


def _private(path: Path) -> Path | None:
    """The scene folder ``path`` is private to, unless that scene is the one being composed."""
    folder = scene_folder(path)
    if folder is None or folder in _scopes or path.resolve() == folder / SCENE_FILE:
        return None
    return folder


def find(name: str | Path, base: str | Path | None = None) -> Path:
    """The file ``name`` (relative to an assets directory) in the first root that has
    it. When none has, the path it would have in ``base`` (default: the core), so the
    caller's own "no such file" error names somewhere sensible. A file in another scene's
    folder is that scene's own: naming it raises ``RootsError``."""
    name = Path(name)
    if name.is_absolute():
        return name
    for root in (*reversed(_scopes), *_shared(base)):
        if (root / name).exists():
            owner = _private(root / name)
            if owner is not None:
                raise RootsError(f"'{name.as_posix()}' is private to the scene folder '{owner.name}': only "
                                 f"{SCENES}/{owner.name}/{SCENE_FILE} can name it. Move it out of the folder to share it")
            return root / name
    return Path(base if base is not None else CORE) / name


def glob(pattern: str, base: str | Path | None = None) -> list[Path]:
    """Files matching ``pattern`` in every root, sorted by relative name; a file in an
    earlier root hides the same relative name in a later one. Scene folders' private
    files are left out (but for the scene being composed)."""
    found: dict[str, Path] = {}
    for root in (*reversed(_scopes), *_shared(base)):
        for path in root.glob(pattern):
            if _private(path) is None:
                found.setdefault(path.relative_to(root).as_posix(), path)
    return [found[name] for name in sorted(found)]


def classify(path: str | Path) -> str | None:
    """What a content file is: ``"scene"``, ``"asset"``, or None for data (a ``kind:``) and non-mappings.

    Files under ``scenes/`` are always scenes; elsewhere a ``place:`` or ``city:`` key makes one.
    """
    from .layout.yaml_utils import safe_load_path

    path = Path(path)
    data = safe_load_path(path)
    if not isinstance(data, dict) or "kind" in data:
        return None
    if SCENES in path.parent.parts[-2:] or any(k in data for k in SCENE_KEYS):
        return "scene"
    return "asset"


def content_files(base: str | Path | None = None) -> dict[str, tuple[str, Path]]:
    """Every asset and scene in the roots: ``{name: ("asset" | "scene", path)}``.

    ``*.yaml`` (assets or scenes), ``<ASSET_SUBDIRS>/*.yaml`` (assets), ``scenes/*.yaml`` and
    scene folders ``scenes/<name>/scene.yaml`` (named by the folder). The first of a name wins.
    """
    out: dict[str, tuple[str, Path]] = {}
    patterns = ["*.yaml", *(f"{sub}/*.yaml" for sub in ASSET_SUBDIRS), f"{SCENES}/*.yaml", f"{SCENES}/*/{SCENE_FILE}"]
    for pattern in patterns:
        for path in glob(pattern, base):
            if path.name == CONFIG:
                continue
            kind = classify(path)
            if kind is not None:
                out.setdefault(path.parent.name if path.name == SCENE_FILE else path.stem, (kind, path))
    return out


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
