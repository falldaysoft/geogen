"""Floor texture generators for hardwood, tile, and carpet."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import TextureGenerator
from .noise import fractal_noise, perlin_noise


@dataclass
class HardwoodFloorTextureGenerator(TextureGenerator):
    """Generates procedural hardwood floor textures with planks.

    Creates realistic hardwood flooring with:
    - Individual plank patterns with wood grain
    - Plank seams/gaps
    - Natural color variation between planks

    Attributes:
        width: Texture width in pixels
        height: Texture height in pixels
        seed: Random seed for reproducibility
        color_light: Light wood color as (R, G, B) tuple, 0-255
        color_dark: Dark wood color (grain lines) as (R, G, B) tuple
        plank_width: Width of each plank in texture fraction (0-1)
        plank_length: Length of planks (how often they stagger)
        grain_scale: Scale of wood grain pattern
        color_variation: How much planks vary in color (0-1)
        gap_width: Width of gaps between planks in pixels
        gap_color: Color of gaps between planks
    """

    color_light: tuple[int, int, int] = (180, 140, 90)  # Medium oak
    color_dark: tuple[int, int, int] = (120, 80, 45)
    plank_width: float = 0.1  # 10% of texture width per plank
    plank_length: float = 0.5  # Planks stagger at 50%
    grain_scale: float = 8.0
    color_variation: float = 0.2
    gap_width: int = 2
    gap_color: tuple[int, int, int] = (40, 30, 20)

    def generate(self) -> Image.Image:
        """Generate a hardwood floor texture."""
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)

        # Calculate plank dimensions in pixels
        plank_w_px = int(self.width * self.plank_width)
        plank_h_px = int(self.height * self.plank_length)

        if plank_w_px < 1:
            plank_w_px = 1
        if plank_h_px < 1:
            plank_h_px = 1

        # Number of planks
        n_planks_x = int(np.ceil(self.width / plank_w_px))
        n_planks_y = int(np.ceil(self.height / plank_h_px))

        # Generate each plank
        rng = np.random.default_rng(self.seed)

        for px in range(n_planks_x):
            # Stagger odd columns by half a plank
            y_offset = (plank_h_px // 2) if px % 2 == 1 else 0

            for py in range(-1, n_planks_y + 1):
                # Plank bounds
                x_start = px * plank_w_px
                x_end = min((px + 1) * plank_w_px, self.width)
                y_start = py * plank_h_px + y_offset
                y_end = y_start + plank_h_px

                # Skip if completely out of bounds
                if y_end < 0 or y_start >= self.height:
                    continue

                # Clamp to image bounds
                y_start_clamped = max(0, y_start)
                y_end_clamped = min(self.height, y_end)
                x_start_clamped = max(0, x_start)
                x_end_clamped = min(self.width, x_end)

                if x_end_clamped <= x_start_clamped or y_end_clamped <= y_start_clamped:
                    continue

                # Generate unique grain for this plank
                plank_seed = self.seed + px * 1000 + py if self.seed else px * 1000 + py
                self._render_plank(
                    rgb,
                    x_start_clamped, y_start_clamped,
                    x_end_clamped, y_end_clamped,
                    plank_seed,
                    rng.random() * self.color_variation
                )

        # Draw gaps between planks
        self._draw_gaps(rgb, plank_w_px, plank_h_px, n_planks_x, n_planks_y)

        # Clamp and convert
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode='RGB')

    def generate_ao_map(self) -> Image.Image:
        """Generate AO map that darkens plank gaps."""
        ao = np.ones((self.height, self.width), dtype=np.float64)

        # Calculate plank dimensions in pixels
        plank_w_px = int(self.width * self.plank_width)
        plank_h_px = int(self.height * self.plank_length)

        if plank_w_px < 1:
            plank_w_px = 1
        if plank_h_px < 1:
            plank_h_px = 1

        n_planks_x = int(np.ceil(self.width / plank_w_px))
        n_planks_y = int(np.ceil(self.height / plank_h_px))

        # AO settings
        gap_ao = 0.4  # How dark the gaps are (0 = black, 1 = white)
        falloff = 3  # Pixels of falloff around gaps

        # Vertical gaps
        for px in range(1, n_planks_x):
            x = px * plank_w_px
            for offset in range(-falloff, falloff + 1):
                xo = x + offset
                if 0 <= xo < self.width:
                    # Smooth falloff
                    t = 1.0 - abs(offset) / (falloff + 1)
                    darken = t * (1.0 - gap_ao)
                    ao[:, xo] = np.minimum(ao[:, xo], 1.0 - darken)

        # Horizontal gaps with stagger
        for px in range(n_planks_x):
            y_offset = (plank_h_px // 2) if px % 2 == 1 else 0
            x_start = px * plank_w_px
            x_end = min((px + 1) * plank_w_px, self.width)

            for py in range(n_planks_y + 2):
                y = py * plank_h_px + y_offset
                for offset in range(-falloff, falloff + 1):
                    yo = y + offset
                    if 0 <= yo < self.height:
                        t = 1.0 - abs(offset) / (falloff + 1)
                        darken = t * (1.0 - gap_ao)
                        ao[yo, x_start:x_end] = np.minimum(
                            ao[yo, x_start:x_end], 1.0 - darken
                        )

        return Image.fromarray((ao * 255).astype(np.uint8), mode='L')

    def generate_normal_map(self) -> Image.Image:
        """Generate normal map with wood grain relief and gap edges."""
        # Generate height map from wood grain
        albedo = self.generate_array()
        height = np.mean(albedo, axis=2).astype(np.float64) / 255.0

        # Add depth at gaps
        plank_w_px = int(self.width * self.plank_width)
        plank_h_px = int(self.height * self.plank_length)

        if plank_w_px < 1:
            plank_w_px = 1
        if plank_h_px < 1:
            plank_h_px = 1

        n_planks_x = int(np.ceil(self.width / plank_w_px))
        n_planks_y = int(np.ceil(self.height / plank_h_px))

        half_gap = self.gap_width // 2

        # Lower height at gaps
        for px in range(1, n_planks_x):
            x = px * plank_w_px
            x_start = max(0, x - half_gap)
            x_end = min(self.width, x + half_gap + 1)
            height[:, x_start:x_end] -= 0.2

        for px in range(n_planks_x):
            y_offset = (plank_h_px // 2) if px % 2 == 1 else 0
            x_start = px * plank_w_px
            x_end = min((px + 1) * plank_w_px, self.width)

            for py in range(n_planks_y + 2):
                y = py * plank_h_px + y_offset
                if 0 <= y < self.height:
                    y_start = max(0, y - half_gap)
                    y_end = min(self.height, y + half_gap + 1)
                    height[y_start:y_end, x_start:x_end] -= 0.2

        return self._height_to_normal(height, strength=1.5)

    def _render_plank(
        self,
        rgb: np.ndarray,
        x_start: int, y_start: int,
        x_end: int, y_end: int,
        plank_seed: int,
        color_offset: float,
    ) -> None:
        """Render a single plank with wood grain."""
        h = y_end - y_start
        w = x_end - x_start

        if h <= 0 or w <= 0:
            return

        # Generate wood grain running along the plank (Y direction)
        grain = fractal_noise(
            w, h,
            octaves=3,
            persistence=0.5,
            scale=self.grain_scale,
            seed=plank_seed,
        )

        # Subtle cross-grain
        cross = fractal_noise(
            w, h,
            octaves=2,
            scale=self.grain_scale * 3,
            seed=plank_seed + 500,
        )

        # Combine for final pattern
        pattern = grain * 0.8 + cross * 0.2
        pattern = pattern * 0.5 + 0.5  # Normalize to 0-1

        # Interpolate colors with variation
        light = np.array(self.color_light, dtype=np.float64)
        dark = np.array(self.color_dark, dtype=np.float64)

        # Apply color offset for variation between planks
        light = light * (1.0 + (color_offset - 0.5) * 0.3)
        dark = dark * (1.0 + (color_offset - 0.5) * 0.3)

        for i in range(3):
            rgb[y_start:y_end, x_start:x_end, i] = (
                dark[i] + (light[i] - dark[i]) * pattern
            )

    def _draw_gaps(
        self,
        rgb: np.ndarray,
        plank_w_px: int, plank_h_px: int,
        n_planks_x: int, n_planks_y: int,
    ) -> None:
        """Draw gaps between planks."""
        gap_color = np.array(self.gap_color, dtype=np.float64)
        half_gap = self.gap_width // 2

        # Vertical gaps (between columns)
        for px in range(1, n_planks_x):
            x = px * plank_w_px
            x_start = max(0, x - half_gap)
            x_end = min(self.width, x + half_gap + 1)
            for i in range(3):
                rgb[:, x_start:x_end, i] = gap_color[i]

        # Horizontal gaps (between plank rows, with stagger)
        for px in range(n_planks_x):
            y_offset = (plank_h_px // 2) if px % 2 == 1 else 0
            x_start = px * plank_w_px
            x_end = min((px + 1) * plank_w_px, self.width)

            for py in range(n_planks_y + 2):
                y = py * plank_h_px + y_offset
                if 0 <= y < self.height:
                    y_gap_start = max(0, y - half_gap)
                    y_gap_end = min(self.height, y + half_gap + 1)
                    for i in range(3):
                        rgb[y_gap_start:y_gap_end, x_start:x_end, i] = gap_color[i]


@dataclass
class CarpetTextureGenerator(TextureGenerator):
    """Generates procedural carpet textures.

    Creates carpet-like surfaces with:
    - Fine fiber texture
    - Subtle pattern/weave

    Attributes:
        width: Texture width in pixels
        height: Texture height in pixels
        seed: Random seed for reproducibility
        base_color: Carpet color as (R, G, B) tuple, 0-255
        fiber_scale: Scale of fiber texture
        pattern_strength: How visible the pattern is (0-1)
    """

    base_color: tuple[int, int, int] = (80, 70, 65)  # Dark gray carpet
    fiber_scale: float = 50.0
    pattern_strength: float = 0.15

    def generate(self) -> Image.Image:
        """Generate a carpet texture."""
        # Fine fiber noise
        fiber_noise = fractal_noise(
            self.width, self.height,
            octaves=4,
            persistence=0.6,
            scale=self.fiber_scale,
            seed=self.seed,
        )

        # Coarser pattern
        pattern_noise = fractal_noise(
            self.width, self.height,
            octaves=2,
            scale=10.0,
            seed=(self.seed + 100) if self.seed else 100,
        )

        # Combine
        combined = fiber_noise * 0.7 + pattern_noise * 0.3

        # Create RGB
        base = np.array(self.base_color, dtype=np.float64)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)

        brightness = combined * self.pattern_strength * 60

        for i in range(3):
            rgb[:, :, i] = base[i] + brightness

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode='RGB')
