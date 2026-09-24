"""Interior texture generators: woven fabric, ceramic tile and marble.

All patterns are built on a [0, 1) lattice (endpoint excluded) with integer
repeat counts, so every texture tiles seamlessly under metric UVs.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import TextureGenerator
from .noise import fractal_noise, turbulence


def _unit_grid(width: int, height: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pixel-centre coordinates in [0, 1) that wrap cleanly at the edges."""
    x = np.arange(width, dtype=np.float64) / width
    y = np.arange(height, dtype=np.float64) / height
    xv, yv = np.meshgrid(x, y)
    return xv, yv


def _to_image(rgb: NDArray[np.float64]) -> Image.Image:
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _to_gray(values: NDArray[np.float64]) -> Image.Image:
    return Image.fromarray((np.clip(values, 0, 1) * 255).astype(np.uint8), mode="L")


def _seed(seed: int | None, offset: int) -> int:
    return (seed or 0) + offset


@dataclass
class FabricTextureGenerator(TextureGenerator):
    """Woven upholstery / bedding fabric.

    A plain (``weave: plain``) or twill (``weave: twill``) weave of warp and
    weft threads, with slub noise so the cloth doesn't look printed.

    Attributes:
        base_color: Cloth colour (the tint) as (R, G, B), 0-255
        weft_color: Optional second thread colour; defaults to base_color
        threads: Threads per texture repeat in each direction (integer)
        weave: 'plain' or 'twill'
        thread_contrast: Brightness difference between thread crowns and gaps (0-1)
        slub: Strength of irregular thread thickness variation (0-1)
    """

    base_color: tuple[int, int, int] = (120, 130, 150)
    weft_color: tuple[int, int, int] | None = None
    threads: int = 96
    weave: str = "plain"
    thread_contrast: float = 0.35
    slub: float = 0.3

    def _weave(self) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Return (height 0-1, is_warp_on_top) for every pixel."""
        xv, yv = _unit_grid(self.width, self.height)
        n = max(1, int(self.threads))
        cx = xv * n
        cy = yv * n
        ix = np.floor(cx).astype(np.int64)
        iy = np.floor(cy).astype(np.int64)
        fx = cx - ix
        fy = cy - iy

        if self.weave == "twill":
            warp_top = ((ix - iy) % 4) < 2
        else:
            warp_top = ((ix + iy) % 2) == 0

        # Rounded thread cross-sections: warp runs along y, weft along x.
        warp_profile = np.sin(np.pi * fx)
        weft_profile = np.sin(np.pi * fy)
        height = np.where(warp_top, warp_profile, weft_profile)

        # Slub: threads thicken and thin along their length.
        if self.slub > 0:
            slub = fractal_noise(self.width, self.height, octaves=3, scale=8.0,
                                 seed=_seed(self.seed, 11))
            height = height * (1.0 + slub * self.slub)
        return np.clip(height, 0, 1), warp_top

    def generate(self) -> Image.Image:
        height, warp_top = self._weave()
        warp = np.array(self.base_color, dtype=np.float64)
        weft = np.array(self.weft_color or self.base_color, dtype=np.float64)
        colour = np.where(warp_top[..., None], warp, weft)

        shade = 1.0 - self.thread_contrast + self.thread_contrast * height
        mottle = fractal_noise(self.width, self.height, octaves=3, scale=4.0,
                               seed=_seed(self.seed, 23))
        shade = shade * (1.0 + mottle * 0.05)
        return _to_image(colour * shade[..., None])

    def generate_normal_map(self) -> Image.Image:
        height, _ = self._weave()
        return self._height_to_normal(height, strength=2.0)

    def generate_roughness_map(self) -> Image.Image:
        height, _ = self._weave()
        return _to_gray(0.95 - height * 0.1)

    def generate_ao_map(self) -> Image.Image:
        height, _ = self._weave()
        return _to_gray(0.7 + height * 0.3)


@dataclass
class CutPileCarpetTextureGenerator(TextureGenerator):
    """Cut-pile carpet: dense fibre tufts with soft tonal shading.

    Attributes:
        base_color: Carpet colour as (R, G, B), 0-255
        tufts: Tufts per texture repeat in each direction (integer)
        pile_contrast: Brightness variation between fibre tips (0-1)
        shading: Strength of broad pile-direction shading (0-1)
    """

    base_color: tuple[int, int, int] = (110, 95, 85)
    tufts: int = 160
    pile_contrast: float = 0.35
    shading: float = 0.12

    def _pile(self) -> NDArray[np.float64]:
        rng = np.random.default_rng(_seed(self.seed, 5))
        n = max(1, int(self.tufts))
        tips = rng.random((n, n))
        # Sample the tuft lattice with wrap-around bilinear filtering.
        xv, yv = _unit_grid(self.width, self.height)
        gx, gy = xv * n, yv * n
        x0, y0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
        tx, ty = gx - x0, gy - y0
        x1, y1 = (x0 + 1) % n, (y0 + 1) % n
        x0, y0 = x0 % n, y0 % n
        top = tips[y0, x0] * (1 - tx) + tips[y0, x1] * tx
        bot = tips[y1, x0] * (1 - tx) + tips[y1, x1] * tx
        return top * (1 - ty) + bot * ty

    def generate(self) -> Image.Image:
        pile = self._pile()
        broad = fractal_noise(self.width, self.height, octaves=3, scale=3.0,
                              seed=_seed(self.seed, 17))
        shade = 1.0 + (pile - 0.5) * self.pile_contrast + broad * self.shading
        return _to_image(np.array(self.base_color, dtype=np.float64) * shade[..., None])

    def generate_normal_map(self) -> Image.Image:
        return self._height_to_normal(self._pile(), strength=1.5)

    def generate_ao_map(self) -> Image.Image:
        return _to_gray(0.8 + self._pile() * 0.2)


@dataclass
class TileTextureGenerator(TextureGenerator):
    """Ceramic / porcelain tile with grout joints.

    One texture repeat holds ``tiles_x`` x ``tiles_y`` tiles, so the tile size
    in metres is the material's ``tile_size`` divided by those counts.

    Attributes:
        tile_color: Tile colour as (R, G, B), 0-255
        grout_color: Grout colour as (R, G, B), 0-255
        tiles_x: Tiles across one texture repeat (integer)
        tiles_y: Tiles down one texture repeat (integer)
        grout_width: Grout joint width as a fraction of a tile
        layout: 'grid' (stacked) or 'offset' (running bond, e.g. subway tile)
        color_variation: Per-tile brightness variation (0-1)
        glaze_noise: Subtle mottling in the glaze (0-1)
        marble: Blend marble veining into each tile (0-1)
    """

    tile_color: tuple[int, int, int] = (228, 226, 220)
    grout_color: tuple[int, int, int] = (170, 168, 160)
    tiles_x: int = 2
    tiles_y: int = 2
    grout_width: float = 0.025
    layout: str = "grid"
    color_variation: float = 0.04
    glaze_noise: float = 0.3
    marble: float = 0.0

    def _layout(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Return (edge distance in tile units, tile index) per pixel."""
        xv, yv = _unit_grid(self.width, self.height)
        nx, ny = max(1, int(self.tiles_x)), max(1, int(self.tiles_y))
        gy = yv * ny
        row = np.floor(gy).astype(np.int64)
        shift = (row % 2) * 0.5 if self.layout == "offset" and ny % 2 == 0 else 0.0
        gx = xv * nx + shift
        col = np.floor(gx).astype(np.int64)
        fx, fy = gx - col, gy - row
        # Aspect-correct edge distance so joints are equal width both ways.
        aspect = (1.0 / nx) / (1.0 / ny)
        dx = np.minimum(fx, 1 - fx) * aspect
        dy = np.minimum(fy, 1 - fy)
        edge = np.minimum(dx, dy)
        index = (row % ny) * nx + (col % nx)
        return edge, index

    def _grout_mask(self, edge: NDArray[np.float64]) -> NDArray[np.float64]:
        """1 in the grout, 0 on the tile face, with a narrow soft bevel."""
        half = self.grout_width / 2
        bevel = max(half * 0.6, 1.5 / max(self.width, self.height) * max(self.tiles_y, 1))
        return np.clip(1.0 - (edge - half) / bevel, 0, 1)

    def _veins(self) -> NDArray[np.float64]:
        return MarbleTextureGenerator(width=self.width, height=self.height,
                                      seed=self.seed)._veins()

    def generate(self) -> Image.Image:
        edge, index = self._layout()
        grout = self._grout_mask(edge)

        rng = np.random.default_rng(_seed(self.seed, 3))
        per_tile = 1.0 + (rng.random(int(index.max()) + 1) - 0.5) * 2 * self.color_variation
        glaze = fractal_noise(self.width, self.height, octaves=4, scale=6.0,
                              seed=_seed(self.seed, 31))
        tile = np.array(self.tile_color, dtype=np.float64)[None, None, :]
        shade = per_tile[index] * (1.0 + glaze * 0.06 * self.glaze_noise)
        face = tile * shade[..., None]
        if self.marble > 0:
            veins = self._veins()
            face = face * (1.0 - veins[..., None] * 0.45 * self.marble)

        grout_rgb = np.array(self.grout_color, dtype=np.float64)[None, None, :]
        rgb = face * (1 - grout[..., None]) + grout_rgb * grout[..., None]
        return _to_image(rgb)

    def generate_normal_map(self) -> Image.Image:
        edge, _ = self._layout()
        height = 1.0 - self._grout_mask(edge) * 0.6
        return self._height_to_normal(height, strength=3.0)

    def generate_roughness_map(self) -> Image.Image:
        edge, _ = self._layout()
        grout = self._grout_mask(edge)
        return _to_gray(0.12 + grout * 0.75)

    def generate_ao_map(self) -> Image.Image:
        edge, _ = self._layout()
        return _to_gray(1.0 - self._grout_mask(edge) * 0.45)


@dataclass
class MarbleTextureGenerator(TextureGenerator):
    """Polished marble with turbulent veining.

    Attributes:
        base_color: Stone body colour as (R, G, B), 0-255
        vein_color: Vein colour as (R, G, B), 0-255
        vein_frequency: Veins per texture repeat (integer)
        vein_sharpness: Higher = thinner, crisper veins
        turbulence_strength: How much the veins wander
        cloudiness: Broad tonal clouding in the stone body (0-1)
    """

    base_color: tuple[int, int, int] = (232, 230, 226)
    vein_color: tuple[int, int, int] = (120, 118, 118)
    vein_frequency: int = 3
    vein_sharpness: float = 6.0
    turbulence_strength: float = 1.2
    cloudiness: float = 0.4

    def _veins(self) -> NDArray[np.float64]:
        xv, yv = _unit_grid(self.width, self.height)
        k = max(1, int(self.vein_frequency))
        warp = fractal_noise(self.width, self.height, octaves=5, persistence=0.55,
                             scale=2.0, seed=_seed(self.seed, 41))
        # Integer diagonal frequencies keep the phase periodic in x and y.
        phase = np.pi * (k * xv + k * yv + warp * self.turbulence_strength)
        ridge = 1.0 - np.abs(np.sin(phase))
        # Soft halo around each vein plus a crisp core.
        veins = 0.35 * ridge ** (self.vein_sharpness * 0.5) + 0.65 * ridge ** (self.vein_sharpness * 4)
        # Secondary hairline veins crossing the main ones.
        warp2 = fractal_noise(self.width, self.height, octaves=4, scale=3.0,
                              seed=_seed(self.seed, 53))
        hair = 1.0 - np.abs(np.sin(np.pi * (2 * k * xv - k * yv + warp2 * 2.0)))
        veins = veins + 0.35 * hair ** (self.vein_sharpness * 6)
        # Veins fade in and out along their length.
        fade = fractal_noise(self.width, self.height, octaves=2, scale=2.0,
                             seed=_seed(self.seed, 67)) * 0.5 + 0.6
        return np.clip(veins * fade, 0, 1)

    def generate(self) -> Image.Image:
        veins = self._veins()
        cloud = fractal_noise(self.width, self.height, octaves=4, scale=2.0,
                              seed=_seed(self.seed, 61))
        base = np.array(self.base_color, dtype=np.float64)
        body = base[None, None, :] * (1.0 + cloud[..., None] * 0.08 * self.cloudiness)
        vein = np.array(self.vein_color, dtype=np.float64)[None, None, :]
        return _to_image(body * (1 - veins[..., None]) + vein * veins[..., None])

    def generate_normal_map(self) -> Image.Image:
        # Polished stone: nearly flat, veins very slightly etched.
        return self._height_to_normal(1.0 - self._veins() * 0.1, strength=0.5)

    def generate_roughness_map(self) -> Image.Image:
        return _to_gray(0.1 + self._veins() * 0.15)
