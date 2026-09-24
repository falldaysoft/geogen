"""Load materials from YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .material import Material
from ..textures.asphalt import AsphaltTextureGenerator
from ..textures.brick import BrickTextureGenerator
from ..textures.concrete import ConcreteTextureGenerator
from ..textures.dirt import DirtTextureGenerator
from ..textures.floor import HardwoodFloorTextureGenerator, CarpetTextureGenerator
from ..textures.grass import GrassTextureGenerator
from ..textures.interior import (
    CutPileCarpetTextureGenerator,
    FabricTextureGenerator,
    MarbleTextureGenerator,
    TileTextureGenerator,
)
from ..textures.metal import MetalTextureGenerator, MetalType
from ..textures.rock import RockTextureGenerator
from ..textures.roof import RoofTextureGenerator
from ..textures.wall import PlasterTextureGenerator, PaintedWallTextureGenerator
from ..textures.wood import WoodTextureGenerator


# Registry of texture generator types
TEXTURE_GENERATORS = {
    "wood": WoodTextureGenerator,
    "metal": MetalTextureGenerator,
    "plaster": PlasterTextureGenerator,
    "painted_wall": PaintedWallTextureGenerator,
    "hardwood_floor": HardwoodFloorTextureGenerator,
    "carpet": CarpetTextureGenerator,
    "grass": GrassTextureGenerator,
    "rock": RockTextureGenerator,
    "dirt": DirtTextureGenerator,
    "asphalt": AsphaltTextureGenerator,
    "brick": BrickTextureGenerator,
    "concrete": ConcreteTextureGenerator,
    "roof": RoofTextureGenerator,
    "fabric": FabricTextureGenerator,
    "cut_pile_carpet": CutPileCarpetTextureGenerator,
    "tile": TileTextureGenerator,
    "marble": MarbleTextureGenerator,
}

# Named paint colours; any colour param may use one of these instead of [r, g, b].
PAINT_PALETTE: dict[str, tuple[int, int, int]] = {
    "white": (244, 242, 236),
    "warm_white": (240, 234, 222),
    "greige": (200, 190, 175),
    "stone": (184, 178, 166),
    "sage": (158, 172, 148),
    "duck_egg": (170, 196, 196),
    "navy": (46, 58, 84),
    "charcoal": (62, 64, 66),
    "terracotta": (190, 110, 80),
    "mustard": (206, 160, 60),
    "blush": (222, 186, 176),
    "forest": (52, 84, 64),
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
            available = self._list_available()
            import difflib
            suggestion = difflib.get_close_matches(name, available, n=1, cutoff=0.5)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            raise FileNotFoundError(
                f"Material '{name}' not found.{hint}"
                f" Available materials: {available}"
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

        # Parse PBR properties (with fallback to legacy shininess)
        pbr = data.get("pbr", {})
        roughness = pbr.get("roughness", 1.0 - shininess)  # Convert shininess to roughness
        metallic = pbr.get("metallic", 0.0)
        normal_strength = pbr.get("normal_strength", 1.0)
        ao_strength = pbr.get("ao_strength", 1.0)

        # UV tiling
        uv_scale = tuple(data.get("uv_scale", [1.0, 1.0]))
        tile_raw = data.get("tile_size", 1.0)
        tile_size = (float(tile_raw), float(tile_raw)) if isinstance(tile_raw, (int, float)) else tuple(tile_raw)

        return Material(
            name=name,
            texture_generator=generator,
            texture_size=tuple(size),
            roughness=roughness,
            metallic=metallic,
            normal_strength=normal_strength,
            ao_strength=ao_strength,
            uv_scale=uv_scale,
            tile_size=tile_size,
            shininess=shininess,
            tint=tint,
        )

    def _convert_params(self, texture_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """Convert YAML params to generator constructor args."""
        converted = dict(params)

        # Convert color lists to tuples
        color_keys = [
            "color_light", "color_dark", "color_base", "color_variation",
            "base_color", "highlight_color", "mortar_color",
            "weft_color", "tile_color", "grout_color", "vein_color", "gap_color",
        ]
        for key in color_keys:
            value = converted.get(key)
            if isinstance(value, list):
                converted[key] = tuple(value)
            elif isinstance(value, str):
                if value not in PAINT_PALETTE:
                    raise ValueError(
                        f"Unknown colour '{value}' for {key}; palette: {sorted(PAINT_PALETTE)}"
                    )
                converted[key] = PAINT_PALETTE[value]

        # Convert metal_type string to enum
        if texture_type == "metal" and "metal_type" in converted:
            converted["metal_type"] = MetalType(converted["metal_type"])

        return converted

    def _list_available(self) -> list[str]:
        """List all available material names from search paths."""
        names = []
        for search_path in self.search_paths:
            if search_path.exists():
                for yaml_path in search_path.glob("*.yaml"):
                    names.append(yaml_path.stem)
        return sorted(names)

    def clear_cache(self) -> None:
        """Clear the material cache."""
        self._cache.clear()
