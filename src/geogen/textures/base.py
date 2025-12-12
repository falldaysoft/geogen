"""Base class for procedural texture generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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
