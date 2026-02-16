"""Scene registry with auto-discovery of YAML assets and scenes."""

from pathlib import Path
from typing import Callable

import yaml

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
        self._assets_dir = assets_dir or self._default_assets_dir()

    @staticmethod
    def _default_assets_dir() -> Path:
        return Path(__file__).parent.parent.parent / "assets"

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
        """
        # Discover assets and composed scenes in root assets dir
        for yaml_path in sorted(self._assets_dir.glob("*.yaml")):
            name = yaml_path.stem
            if name not in self._factories:
                if self._is_composed_scene(yaml_path):
                    self._register_composed_scene(name, yaml_path)
                else:
                    self._register_asset(name, yaml_path)

        # Discover composed scenes
        scenes_dir = self._assets_dir / "scenes"
        if scenes_dir.is_dir():
            for yaml_path in sorted(scenes_dir.glob("*.yaml")):
                name = yaml_path.stem
                if name not in self._factories:
                    self._register_composed_scene(name, yaml_path)

    def _is_composed_scene(self, yaml_path: Path) -> bool:
        """Check if a YAML file is a composed scene (has place: or compose:)."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return False
        return "place" in data or "compose" in data

    def _register_asset(self, name: str, yaml_path: Path) -> None:
        """Register a YAML asset that loads via LayoutLoader."""
        def factory(path: Path = yaml_path) -> SceneNode:
            root = SceneNode("root")
            loader = LayoutLoader()
            asset = loader.load(path)
            root.add_child(asset)
            return root
        self._factories[name] = factory

    def _register_composed_scene(self, name: str, yaml_path: Path) -> None:
        """Register a YAML composed scene that loads via SceneComposer."""
        def factory(path: Path = yaml_path) -> SceneNode:
            composer = SceneComposer(self._assets_dir)
            return composer.compose(path)
        self._factories[name] = factory
