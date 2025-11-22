"""Load materials from YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .material import Material
from ..textures.wood import WoodTextureGenerator
from ..textures.metal import MetalTextureGenerator, MetalType


# Registry of texture generator types
TEXTURE_GENERATORS = {
    "wood": WoodTextureGenerator,
    "metal": MetalTextureGenerator,
}


class MaterialLoader:
    """Loads material definitions from YAML files.

    YAML format:
    ```yaml
    name: oak_wood
    texture:
      type: wood
      params:
        color_light: [210, 170, 120]
        color_dark: [140, 90, 50]
        ring_count: 8
        grain_strength: 0.3
    size: [512, 512]
    shininess: 0.2
    ```
    """

    def __init__(self, search_paths: list[Path] | None = None) -> None:
        """Initialize loader with search paths.

        Args:
            search_paths: Directories to search for material YAML files.
                         Defaults to ['assets/materials/'] relative to project root.
        """
        if search_paths is None:
            # Default to assets/materials/ relative to this file
            project_root = Path(__file__).parent.parent.parent.parent
            self.search_paths = [project_root / "assets" / "materials"]
        else:
            self.search_paths = search_paths

        self._cache: dict[str, Material] = {}

    def load(self, name: str) -> Material:
        """Load a material by name.

        Searches for {name}.yaml in search paths.

        Args:
            name: Material name (without .yaml extension)

        Returns:
            Material instance

        Raises:
            FileNotFoundError: If material YAML not found
            ValueError: If YAML format is invalid
        """
        if name in self._cache:
            return self._cache[name]

        # Find the YAML file
        yaml_path = self._find_yaml(name)
        if yaml_path is None:
            raise FileNotFoundError(
                f"Material '{name}' not found in search paths: {self.search_paths}"
            )

        # Load and parse
        material = self._load_yaml(yaml_path)
        self._cache[name] = material
        return material

    def _find_yaml(self, name: str) -> Path | None:
        """Find YAML file for material name."""
        for search_path in self.search_paths:
            yaml_path = search_path / f"{name}.yaml"
            if yaml_path.exists():
                return yaml_path
        return None

    def _load_yaml(self, path: Path) -> Material:
        """Load material from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return self._parse_material(data)

    def _parse_material(self, data: dict[str, Any]) -> Material:
        """Parse material definition from YAML data."""
        name = data.get("name", "unnamed")

        # Parse texture generator
        texture_data = data.get("texture", {})
        texture_type = texture_data.get("type", "wood")
        texture_params = texture_data.get("params", {})

        generator_class = TEXTURE_GENERATORS.get(texture_type)
        if generator_class is None:
            raise ValueError(f"Unknown texture type: {texture_type}")

        # Handle special parameter conversions
        texture_params = self._convert_params(texture_type, texture_params)

        # Create generator
        generator = generator_class(**texture_params)

        # Parse other material properties
        size = data.get("size", [512, 512])
        shininess = data.get("shininess", 0.3)
        tint = data.get("tint")
        if tint is not None:
            tint = tuple(tint)

        return Material(
            name=name,
            texture_generator=generator,
            texture_size=tuple(size),
            shininess=shininess,
            tint=tint,
        )

    def _convert_params(self, texture_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """Convert YAML params to generator constructor args."""
        converted = dict(params)

        # Convert color lists to tuples
        for key in ["color_light", "color_dark", "base_color", "highlight_color"]:
            if key in converted and isinstance(converted[key], list):
                converted[key] = tuple(converted[key])

        # Convert metal_type string to enum
        if texture_type == "metal" and "metal_type" in converted:
            converted["metal_type"] = MetalType(converted["metal_type"])

        return converted

    def clear_cache(self) -> None:
        """Clear the material cache."""
        self._cache.clear()
