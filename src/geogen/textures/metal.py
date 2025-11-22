"""Metal texture generator using procedural noise."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

from .base import TextureGenerator
from .noise import perlin_noise, fractal_noise


class MetalType(Enum):
    """Types of metal finishes."""
    BRUSHED = "brushed"
    SCRATCHED = "scratched"
    POLISHED = "polished"


@dataclass
class MetalTextureGenerator(TextureGenerator):
    """Generates procedural metal textures.

    Creates realistic metal surfaces with:
    - Brushed metal effect (directional scratches)
    - Random scratches and wear
    - Subtle color variation

    Attributes:
        width: Texture width in pixels
        height: Texture height in pixels
        seed: Random seed for reproducibility
        base_color: Base metal color as (R, G, B) tuple, 0-255
        highlight_color: Highlight/reflection color
        metal_type: Type of metal finish (brushed, scratched, polished)
        brush_direction: Angle of brush strokes in degrees (0 = horizontal)
        brush_density: Density of brush strokes (higher = finer)
        scratch_amount: Amount of random scratches (0-1)
        roughness: Surface roughness affecting color variation (0-1)
    """

    base_color: tuple[int, int, int] = (160, 165, 170)
    highlight_color: tuple[int, int, int] = (220, 225, 230)
    metal_type: MetalType = MetalType.BRUSHED
    brush_direction: float = 0.0
    brush_density: float = 50.0
    scratch_amount: float = 0.3
    roughness: float = 0.4

    def generate(self) -> Image.Image:
        """Generate a metal texture."""
        if self.metal_type == MetalType.BRUSHED:
            return self._generate_brushed()
        elif self.metal_type == MetalType.SCRATCHED:
            return self._generate_scratched()
        else:  # POLISHED
            return self._generate_polished()

    def _generate_brushed(self) -> Image.Image:
        """Generate brushed metal texture with directional streaks."""
        # Create directional noise by stretching in brush direction
        angle_rad = np.radians(self.brush_direction)

        # Generate base noise at different scales
        # Stretch noise perpendicular to brush direction
        noise_fine = self._directional_noise(
            scale_along=self.brush_density,
            scale_across=self.brush_density * 0.1,
            angle=angle_rad,
            seed_offset=0,
        )

        noise_medium = self._directional_noise(
            scale_along=self.brush_density * 0.3,
            scale_across=self.brush_density * 0.05,
            angle=angle_rad,
            seed_offset=100,
        )

        # Combine noise layers
        combined = noise_fine * 0.6 + noise_medium * 0.4

        # Add some random scratches
        if self.scratch_amount > 0:
            scratches = self._generate_scratches()
            combined = combined * (1 - self.scratch_amount * 0.5) + scratches * self.scratch_amount * 0.5

        # Map to colors
        return self._apply_metal_colors(combined)

    def _generate_scratched(self) -> Image.Image:
        """Generate scratched metal with random directional scratches."""
        # Base smooth metal
        base = fractal_noise(
            self.width, self.height,
            octaves=2,
            scale=2.0,
            seed=self.seed,
        ) * 0.3 + 0.5

        # Add multiple scratch layers at different angles
        scratches = np.zeros((self.height, self.width), dtype=np.float64)
        n_scratch_layers = 4

        for i in range(n_scratch_layers):
            angle = np.random.default_rng(self.seed + i if self.seed else i).uniform(0, 180)
            layer = self._directional_noise(
                scale_along=30.0,
                scale_across=2.0,
                angle=np.radians(angle),
                seed_offset=200 + i * 50,
            )
            # Threshold to create distinct scratches
            layer = np.clip((layer - 0.3) * 3, 0, 1)
            scratches = np.maximum(scratches, layer * 0.3)

        combined = base + scratches
        return self._apply_metal_colors(combined)

    def _generate_polished(self) -> Image.Image:
        """Generate polished metal with subtle variation."""
        # Very subtle noise for polished look
        noise = fractal_noise(
            self.width, self.height,
            octaves=3,
            persistence=0.3,
            scale=4.0,
            seed=self.seed,
        )

        # Normalize to small range
        combined = noise * 0.15 * self.roughness + 0.5

        return self._apply_metal_colors(combined)

    def _directional_noise(
        self,
        scale_along: float,
        scale_across: float,
        angle: float,
        seed_offset: int,
    ) -> np.ndarray:
        """Generate noise stretched in a specific direction."""
        # Create coordinate grids
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        xv, yv = np.meshgrid(x, y)

        # Rotate coordinates
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = xv * cos_a - yv * sin_a
        y_rot = xv * sin_a + yv * cos_a

        # Scale differently along each axis
        x_scaled = x_rot * scale_along
        y_scaled = y_rot * scale_across

        # Sample noise at transformed coordinates
        seed = (self.seed + seed_offset) if self.seed else seed_offset
        rng = np.random.default_rng(seed)

        # Generate noise using the transformed coordinates
        # We'll use a simple approach: generate 2D noise and sample it
        noise = perlin_noise(
            self.width, self.height,
            scale=scale_along,
            offset_x=rng.uniform(0, 100),
            offset_y=rng.uniform(0, 100),
            seed=seed,
        )

        # Add stretched component
        noise_stretched = perlin_noise(
            self.width, self.height,
            scale=scale_across,
            offset_x=rng.uniform(0, 100),
            offset_y=rng.uniform(0, 100),
            seed=seed + 1000,
        )

        # Blend based on direction
        result = noise * abs(cos_a) + noise_stretched * abs(sin_a)
        return result * 0.5 + 0.5  # Normalize to [0, 1]

    def _generate_scratches(self) -> np.ndarray:
        """Generate random scratch marks."""
        scratches = np.zeros((self.height, self.width), dtype=np.float64)
        rng = np.random.default_rng(self.seed if self.seed else 42)

        n_scratches = int(20 * self.scratch_amount)
        for i in range(n_scratches):
            # Random scratch line
            x1 = rng.integers(0, self.width)
            y1 = rng.integers(0, self.height)
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.integers(20, min(self.width, self.height) // 2)

            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))

            # Draw line with anti-aliasing approximation
            self._draw_scratch_line(scratches, x1, y1, x2, y2, rng.uniform(0.3, 0.8))

        return scratches

    def _draw_scratch_line(
        self,
        img: np.ndarray,
        x1: int, y1: int,
        x2: int, y2: int,
        intensity: float,
    ) -> None:
        """Draw a scratch line on the image."""
        # Bresenham's line algorithm with thickness
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1
        while True:
            # Draw with slight blur
            for ox in range(-1, 2):
                for oy in range(-1, 2):
                    px, py = x + ox, y + oy
                    if 0 <= px < self.width and 0 <= py < self.height:
                        falloff = 1.0 / (1 + abs(ox) + abs(oy))
                        img[py, px] = max(img[py, px], intensity * falloff)

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _apply_metal_colors(self, pattern: np.ndarray) -> Image.Image:
        """Apply metal coloring to a grayscale pattern."""
        pattern = np.clip(pattern, 0, 1)

        base = np.array(self.base_color, dtype=np.float64)
        highlight = np.array(self.highlight_color, dtype=np.float64)

        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for i in range(3):
            rgb[:, :, i] = base[i] + (highlight[i] - base[i]) * pattern

        # Add subtle color noise for realism
        color_noise = fractal_noise(
            self.width, self.height,
            octaves=2,
            scale=3.0,
            seed=(self.seed + 500) if self.seed else 500,
        )
        rgb += color_noise[:, :, np.newaxis] * 5 * self.roughness

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode='RGB')
