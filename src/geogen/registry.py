"""Scene registry with auto-discovery of YAML assets and scenes."""

from pathlib import Path
from typing import Callable

from . import roots
from .core.node import SceneNode
from .layout import LayoutLoader, SceneComposer


SceneFactory = Callable[[], SceneNode]


class SceneRegistry:
    """Registry that maps scene names to factory functions.

    Supports auto-discovery of YAML assets and composed scenes,
    plus manual registration of Python-coded scenes.
    """

    def __init__(self, assets_dir: Path | None = None) -> None:
        self._factories: dict[str, SceneFactory] = {}
        self._assets_dir = assets_dir or roots.CORE
        # Every content root (a project's, then the core's) unless one directory was asked for.
        self._roots = [Path(assets_dir)] if assets_dir is not None else roots.roots()
        self._files: dict[str, Path] = {}

    @property
    def scenes(self) -> dict[str, SceneFactory]:
        """Return a copy of the scene name -> factory mapping."""
        return dict(self._factories)

    def names(self) -> list[str]:
        """Return sorted list of registered scene names."""
        return sorted(self._factories.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def __getitem__(self, name: str) -> SceneFactory:
        return self._factories[name]

    def register(self, name: str, factory: SceneFactory) -> None:
        """Register a Python-coded scene factory."""
        self._factories[name] = factory

    def discover(self) -> None:
        """Scan assets directory for YAML files and register them.

        - ``assets/*.yaml`` are loaded as individual assets via LayoutLoader
        - ``assets/scenes/*.yaml`` are loaded as composed scenes via SceneComposer
        - ``assets/scenes/<name>/scene.yaml`` is scene ``<name>``, a folder with private files
        - ``assets/<sub>/*.yaml`` for ``roots.ASSET_SUBDIRS`` (vehicles, characters) are assets too

        With a project (``geogen.roots``) its directory is scanned the same way, first.
        A project scene may only take a core scene's name by replacing that very file
        (``overrides:`` in its geogen.yaml).
        """
        for index, root in enumerate(self._roots):
            earlier = {name for name, path in self._files.items() if not path.is_relative_to(root)}
            self._discover_root(root, earlier if index else set())

    def _discover_root(self, assets_dir: Path, earlier: set[str]) -> None:
        def free(yaml_path: Path, name: str | None = None) -> bool:
            name = name or yaml_path.stem
            if name in earlier:
                first = self._files[name]
                if roots.relative(first) != yaml_path.relative_to(assets_dir).as_posix():
                    raise roots.RootsError(
                        f"scene name '{name}' is both {first} and {yaml_path}: rename the project's")
                return False                    # the project's file replaces this one (an override)
            return name not in self._factories

        # Discover assets and composed scenes in root assets dir
        for yaml_path in sorted(assets_dir.glob("*.yaml")):
            name = yaml_path.stem
            kind = None if yaml_path.name == roots.CONFIG else roots.classify(yaml_path)
            if kind is not None and free(yaml_path):
                if kind == "scene":
                    self._register_composed_scene(name, yaml_path, assets_dir)
                else:
                    self._register_asset(name, yaml_path)

        # Asset libraries in subdirectories (vehicles/sedan.yaml -> "sedan").
        for sub in roots.ASSET_SUBDIRS:
            for yaml_path in sorted((assets_dir / sub).glob("*.yaml")):
                name = yaml_path.stem
                if roots.classify(yaml_path) == "asset" and free(yaml_path):
                    self._register_asset(name, yaml_path)

        # Discover composed scenes
        scenes_dir = assets_dir / "scenes"
        if scenes_dir.is_dir():
            for yaml_path in sorted(scenes_dir.glob("*.yaml")):
                name = yaml_path.stem
                if free(yaml_path):
                    self._register_composed_scene(name, yaml_path, assets_dir)
            # Scene folders: scenes/<name>/scene.yaml, with private files beside it (geogen.roots).
            for yaml_path in sorted(scenes_dir.glob(f"*/{roots.SCENE_FILE}")):
                name = yaml_path.parent.name
                if free(yaml_path, name):
                    self._register_composed_scene(name, yaml_path, assets_dir)

    def path(self, name: str) -> Path | None:
        """The YAML file a discovered scene is built from (None for Python-coded ones)."""
        return self._files.get(name)

    def _register_asset(self, name: str, yaml_path: Path) -> None:
        """Register a YAML asset that loads via LayoutLoader."""
        def factory(path: Path = yaml_path) -> SceneNode:
            root = SceneNode("root")
            loader = LayoutLoader()
            asset = loader.load(path)
            root.add_child(asset)
            return root
        self._factories[name] = factory
        self._files[name] = yaml_path

    def _register_composed_scene(self, name: str, yaml_path: Path, assets_dir: Path | None = None) -> None:
        """Register a YAML composed scene that loads via SceneComposer."""
        def factory(path: Path = yaml_path) -> SceneNode:
            composer = SceneComposer(assets_dir or self._assets_dir)
            return composer.compose(path)
        self._factories[name] = factory
        self._files[name] = yaml_path
