"""Material class for texture and rendering properties."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from ..textures.base import TextureGenerator


@dataclass
class Material:
    """Material definition with texture and PBR rendering properties.

    A material combines a procedurally generated texture with
    PBR (Physically Based Rendering) properties for realistic lighting.

    Attributes:
        name: Material identifier
        texture_generator: Generator to create the diffuse/albedo texture
        texture_size: Size of generated texture (width, height)
        roughness: Surface roughness (0=smooth/shiny, 1=rough/matte)
        metallic: Metalness (0=dielectric/non-metal, 1=metal)
        normal_strength: Normal map intensity multiplier
        ao_strength: Ambient occlusion strength multiplier
        shininess: Legacy specular shininess (maps to roughness)
        tint: Optional color tint multiplier (R, G, B) normalized 0-1
    """

    name: str
    texture_generator: TextureGenerator
    texture_size: tuple[int, int] = (512, 512)

    # PBR properties
    roughness: float = 0.5
    metallic: float = 0.0
    normal_strength: float = 1.0
    ao_strength: float = 1.0

    # Legacy property (converted to roughness if roughness not explicitly set)
    shininess: float = 0.3
    tint: tuple[float, float, float] | None = None

    # Cached textures
    _cached_albedo: Image.Image | None = field(default=None, repr=False)
    _cached_normal: Image.Image | None = field(default=None, repr=False)
    _cached_roughness: Image.Image | None = field(default=None, repr=False)
    _cached_ao: Image.Image | None = field(default=None, repr=False)

    def get_texture(self) -> Image.Image:
        """Generate or return cached albedo texture image.

        Returns:
            PIL Image in RGB mode
        """
        if self._cached_albedo is None:
            self.texture_generator.width = self.texture_size[0]
            self.texture_generator.height = self.texture_size[1]
            self._cached_albedo = self.texture_generator.generate()

            if self.tint is not None:
                self._cached_albedo = self._apply_tint(self._cached_albedo)

        return self._cached_albedo

    def get_normal_map(self) -> Image.Image | None:
        """Generate or return cached normal map.

        Returns:
            PIL Image in RGB mode (tangent-space normal map) or None
        """
        if self._cached_normal is None:
            self.texture_generator.width = self.texture_size[0]
            self.texture_generator.height = self.texture_size[1]
            self._cached_normal = self.texture_generator.generate_normal_map()

        return self._cached_normal

    def get_roughness_map(self) -> Image.Image | None:
        """Generate or return cached roughness map.

        Returns:
            PIL Image in L (grayscale) mode or None
        """
        if self._cached_roughness is None:
            self.texture_generator.width = self.texture_size[0]
            self.texture_generator.height = self.texture_size[1]
            self._cached_roughness = self.texture_generator.generate_roughness_map()

        return self._cached_roughness

    def get_ao_map(self) -> Image.Image | None:
        """Generate or return cached ambient occlusion map.

        Returns:
            PIL Image in L (grayscale) mode or None
        """
        if self._cached_ao is None:
            self.texture_generator.width = self.texture_size[0]
            self.texture_generator.height = self.texture_size[1]
            self._cached_ao = self.texture_generator.generate_ao_map()

        return self._cached_ao

    def get_pbr_maps(self) -> dict[str, Image.Image | None]:
        """Get all PBR texture maps.

        Returns:
            Dictionary with keys: 'albedo', 'normal', 'roughness', 'ao'
        """
        return {
            'albedo': self.get_texture(),
            'normal': self.get_normal_map(),
            'roughness': self.get_roughness_map(),
            'ao': self.get_ao_map(),
        }

    def _apply_tint(self, image: Image.Image) -> Image.Image:
        """Apply color tint to texture."""
        arr = np.array(image, dtype=np.float64)
        tint_arr = np.array(self.tint)
        arr *= tint_arr
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode='RGB')

    def get_texture_array(self) -> np.ndarray:
        """Get albedo texture as numpy array.

        Returns:
            HxWx3 uint8 array in RGB format
        """
        return np.array(self.get_texture())

    def get_normal_array(self) -> np.ndarray | None:
        """Get normal map as numpy array.

        Returns:
            HxWx3 uint8 array in RGB format or None
        """
        normal = self.get_normal_map()
        return np.array(normal) if normal is not None else None

    def get_roughness_array(self) -> np.ndarray | None:
        """Get roughness map as numpy array.

        Returns:
            HxW uint8 array or None
        """
        roughness = self.get_roughness_map()
        return np.array(roughness) if roughness is not None else None

    def get_ao_array(self) -> np.ndarray | None:
        """Get AO map as numpy array.

        Returns:
            HxW uint8 array or None
        """
        ao = self.get_ao_map()
        return np.array(ao) if ao is not None else None

    def invalidate_cache(self) -> None:
        """Clear all cached textures, forcing regeneration on next access."""
        self._cached_albedo = None
        self._cached_normal = None
        self._cached_roughness = None
        self._cached_ao = None
