"""Base class for procedural texture generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass
class TextureGenerator(ABC):
    """Abstract base class for procedural texture generators.

    Subclasses implement generate() to create texture images from noise
    and other procedural techniques. Can also generate PBR maps (normal,
    roughness, AO) for enhanced rendering.
    """

    width: int = 512
    height: int = 512
    seed: int | None = None

    def __post_init__(self) -> None:
        """Initialize random state if seed is provided."""
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)
        else:
            self._rng = np.random.default_rng()

    @abstractmethod
    def generate(self) -> Image.Image:
        """Generate the texture image.

        Returns:
            PIL Image in RGB mode
        """
        pass

    def generate_array(self) -> NDArray[np.uint8]:
        """Generate texture as numpy array.

        Returns:
            HxWx3 uint8 array in RGB format
        """
        return np.array(self.generate())

    def generate_normal_map(self) -> Image.Image | None:
        """Generate a normal map from the texture.

        Default implementation derives normals from luminance as height.
        Subclasses can override for custom normal generation.

        Returns:
            PIL Image in RGB mode (tangent-space normal map) or None
        """
        albedo = self.generate_array()
        # Convert to grayscale as height map
        height = np.mean(albedo, axis=2).astype(np.float64) / 255.0
        return self._height_to_normal(height, strength=0.5)

    def generate_roughness_map(self) -> Image.Image | None:
        """Generate a roughness map.

        Default implementation returns None (use uniform roughness).
        Subclasses can override for spatially-varying roughness.

        Returns:
            PIL Image in L (grayscale) mode or None
        """
        return None

    def generate_ao_map(self) -> Image.Image | None:
        """Generate an ambient occlusion map.

        Default implementation returns None (no baked AO).
        Subclasses can override to bake AO into textures.

        Returns:
            PIL Image in L (grayscale) mode or None
        """
        return None

    def _height_to_normal(
        self,
        height: NDArray[np.float64],
        strength: float = 1.0,
    ) -> Image.Image:
        """Convert a height map to a tangent-space normal map.

        Uses Sobel-like gradient computation to derive surface normals.

        Args:
            height: 2D array of height values (0-1 range)
            strength: Normal map intensity multiplier

        Returns:
            PIL Image in RGB mode
        """
        # Compute gradients using central differences
        # Pad for edge handling
        padded = np.pad(height, 1, mode='edge')

        # Sobel-like kernels for smoother gradients
        dx = (
            padded[1:-1, 2:] - padded[1:-1, :-2] +
            2 * (padded[1:-1, 2:] - padded[1:-1, :-2]) +
            padded[1:-1, 2:] - padded[1:-1, :-2]
        ) / 8.0

        dy = (
            padded[2:, 1:-1] - padded[:-2, 1:-1] +
            2 * (padded[2:, 1:-1] - padded[:-2, 1:-1]) +
            padded[2:, 1:-1] - padded[:-2, 1:-1]
        ) / 8.0

        # Scale gradients by strength
        dx *= strength
        dy *= strength

        # Normal = normalize([-dx, -dy, 1])
        dz = np.ones_like(dx)
        length = np.sqrt(dx * dx + dy * dy + dz * dz)

        nx = -dx / length
        ny = -dy / length
        nz = dz / length

        # Convert from [-1, 1] to [0, 255]
        normal_rgb = np.stack([
            ((nx + 1.0) * 0.5 * 255).astype(np.uint8),
            ((ny + 1.0) * 0.5 * 255).astype(np.uint8),
            ((nz + 1.0) * 0.5 * 255).astype(np.uint8),
        ], axis=-1)

        return Image.fromarray(normal_rgb, mode='RGB')

    def _create_roughness_from_variation(
        self,
        base_roughness: float,
        variation: NDArray[np.float64],
        variation_strength: float = 0.2,
    ) -> Image.Image:
        """Create a roughness map from a variation pattern.

        Args:
            base_roughness: Base roughness value (0-1)
            variation: 2D array of variation values (0-1)
            variation_strength: How much the variation affects roughness

        Returns:
            PIL Image in L (grayscale) mode
        """
        roughness = base_roughness + (variation - 0.5) * variation_strength
        roughness = np.clip(roughness, 0.0, 1.0)
        return Image.fromarray((roughness * 255).astype(np.uint8), mode='L')

    def save(self, path: str) -> None:
        """Generate and save texture to file.

        Args:
            path: Output file path (e.g., 'texture.png')
        """
        self.generate().save(path)


@dataclass
class NoiseLayer:
    """Configuration for a single noise layer in a NoiseTextureGenerator.

    Attributes:
        name: Identifier for this layer (used for caching/access)
        octaves: Number of fractal noise octaves
        persistence: Amplitude falloff per octave (0-1)
        scale: Base noise frequency
        seed_offset: Offset added to base seed for this layer
        weight: Contribution weight when combining layers
    """
    name: str
    octaves: int = 4
    persistence: float = 0.5
    scale: float = 4.0
    seed_offset: int = 0
    weight: float = 1.0


@dataclass
class NoiseTextureGenerator(TextureGenerator):
    """Base class for noise-based procedural textures.

    Extracts the common noise-to-color pattern shared by most texture generators:
    1. Generate multiple noise layers (via _get_noise_layers)
    2. Combine them into a 0-1 pattern (via _compute_pattern)
    3. Map pattern to colors via interpolation between color_a and color_b
    4. Add optional color shift noise

    Subclasses must define:
        color_a: First color (mapped to low pattern values)
        color_b: Second color (mapped to high pattern values)

    Subclasses override:
        _get_noise_layers() -> list of NoiseLayer configs
        _compute_pattern(layers) -> 0-1 pattern array (optional)
        _apply_color_shift(rgb) -> rgb with color modifications (optional)
    """

    color_a: tuple[int, int, int] = (100, 100, 100)
    color_b: tuple[int, int, int] = (200, 200, 200)
    color_shift_scale: float = 2.0
    color_shift_strength: float = 0.0  # 0 = no shift

    def _get_noise_layers(self) -> list[NoiseLayer]:
        """Define noise layers for this texture.

        Subclasses override to specify their noise configuration.
        """
        return [NoiseLayer(name="base", octaves=4, scale=4.0, weight=1.0)]

    def _generate_noise_layers(self) -> dict[str, NDArray[np.float64]]:
        """Generate all noise layers and cache them.

        Returns:
            Dict mapping layer name to noise array (values in [-1, 1])
        """
        from .noise import fractal_noise

        layers = {}
        for layer_cfg in self._get_noise_layers():
            seed = (self.seed + layer_cfg.seed_offset) if self.seed else layer_cfg.seed_offset
            noise = fractal_noise(
                self.width, self.height,
                octaves=layer_cfg.octaves,
                persistence=layer_cfg.persistence,
                scale=layer_cfg.scale,
                seed=seed,
            )
            layers[layer_cfg.name] = noise
        return layers

    def _compute_pattern(self, layers: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Combine noise layers into a 0-1 pattern.

        Default: weighted sum of normalized layers.
        Subclasses override for custom combination logic.
        """
        configs = self._get_noise_layers()
        total_weight = sum(c.weight for c in configs)
        pattern = np.zeros((self.height, self.width), dtype=np.float64)
        for cfg in configs:
            normalized = (layers[cfg.name] + 1.0) / 2.0  # [-1,1] -> [0,1]
            pattern += normalized * (cfg.weight / total_weight)
        return np.clip(pattern, 0, 1)

    def _apply_color_shift(
        self, rgb: NDArray[np.float64], layers: dict[str, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """Apply color shift noise to the RGB image.

        Default: adds fractal noise-based color variation.
        """
        if self.color_shift_strength <= 0:
            return rgb

        from .noise import fractal_noise
        color_noise = fractal_noise(
            self.width, self.height,
            octaves=2,
            scale=self.color_shift_scale,
            seed=(self.seed + 900) if self.seed else 900,
        )
        shift = color_noise * self.color_shift_strength
        for i in range(3):
            rgb[:, :, i] = rgb[:, :, i] + shift
        return rgb

    def generate(self) -> Image.Image:
        """Generate texture using the noise pipeline."""
        # Generate noise layers
        layers = self._generate_noise_layers()

        # Compute pattern
        pattern = self._compute_pattern(layers)

        # Map to colors
        a = np.array(self.color_a, dtype=np.float64)
        b = np.array(self.color_b, dtype=np.float64)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for i in range(3):
            rgb[:, :, i] = a[i] + (b[i] - a[i]) * pattern

        # Apply color shift
        rgb = self._apply_color_shift(rgb, layers)

        # Clip and convert
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode='RGB')
