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

    # UV tiling. Mesh UVs are metric, so tile_size is the size in metres that
    # one repeat of the texture covers; uv_scale is an extra multiplier.
    uv_scale: tuple[float, float] = (1.0, 1.0)
    tile_size: tuple[float, float] = (1.0, 1.0)

    # Transparency (1 = opaque; below 1 is alpha-blended: glass) and emission
    # (linear RGB 0-1 times strength: lamp shades, screens).
    opacity: float = 1.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_strength: float = 1.0

    # Legacy property (converted to roughness if roughness not explicitly set)
    shininess: float = 0.3
    tint: tuple[float, float, float] | None = None

    # Cached textures
    _cached_albedo: Image.Image | None = field(default=None, repr=False)
    _cached_normal: Image.Image | None = field(default=None, repr=False)
    _cached_roughness: Image.Image | None = field(default=None, repr=False)
    _cached_ao: Image.Image | None = field(default=None, repr=False)

    @property
    def transparent(self) -> bool:
        return self.opacity < 1.0

    @property
    def emissive_factor(self) -> tuple[float, float, float]:
        """glTF emissiveFactor (clamped to 0-1; strength beyond 1 goes to KHR_materials_emissive_strength)."""
        return tuple(float(min(1.0, c * min(self.emissive_strength, 1.0))) for c in self.emissive)

    @property
    def texture_uv_scale(self) -> tuple[float, float]:
        """Multiplier from mesh (metric) UVs to texture-space UVs."""
        return (
            self.uv_scale[0] / self.tile_size[0],
            self.uv_scale[1] / self.tile_size[1],
        )

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

    def gltf_images(self) -> dict[str, Image.Image]:
        """Texture images packed the way glTF metallic-roughness expects.

        Returns a dict with ``base_color`` (RGB), ``metallic_roughness``
        (G = roughness, B = metallic, absolute values so factors can be 1.0),
        and optionally ``normal`` (strength baked in) and ``occlusion``.
        Shared by the offscreen renderer and the glTF exporter.
        """
        albedo = self.get_texture().convert("RGB")
        width, height = albedo.size
        images: dict[str, Image.Image] = {"base_color": albedo}

        rough = self.get_roughness_map()
        if rough is not None:
            rough_arr = np.asarray(rough.convert("L").resize((width, height)), dtype=np.float64) / 255.0
            rough_arr = rough_arr * (self.roughness / max(rough_arr.mean(), 1e-3))
        else:
            rough_arr = np.full((height, width), self.roughness)
        mr = np.zeros((height, width, 3), dtype=np.uint8)
        mr[..., 1] = np.clip(rough_arr * 255, 0, 255).astype(np.uint8)
        mr[..., 2] = int(np.clip(self.metallic, 0, 1) * 255)
        images["metallic_roughness"] = Image.fromarray(mr)

        normal = self.get_normal_map()
        if normal is not None and self.normal_strength > 0:
            n = np.asarray(normal.convert("RGB"), dtype=np.float64) / 127.5 - 1.0
            n[..., :2] *= self.normal_strength
            n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
            images["normal"] = Image.fromarray(np.clip((n + 1.0) * 127.5, 0, 255).astype(np.uint8))

        ao = self.get_ao_map()
        if ao is not None and self.ao_strength > 0:
            ao_arr = np.asarray(ao.convert("L"), dtype=np.float64) / 255.0
            ao_arr = 1.0 - (1.0 - ao_arr) * self.ao_strength
            images["occlusion"] = Image.fromarray(np.clip(ao_arr * 255, 0, 255).astype(np.uint8)).convert("RGB")
        return images

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
