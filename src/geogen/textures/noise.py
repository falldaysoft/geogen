"""Noise functions for procedural texture generation.

Implements Perlin noise and fractal (fBm) variants using numpy.
"""

import numpy as np
from numpy.typing import NDArray


def _fade(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Smoothstep fade function: 6t^5 - 15t^4 + 10t^3"""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: NDArray[np.float64], b: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear interpolation."""
    return a + t * (b - a)


def _generate_permutation(rng: np.random.Generator | None = None) -> NDArray[np.int32]:
    """Generate a permutation table for noise generation."""
    if rng is None:
        rng = np.random.default_rng(0)
    p = np.arange(256, dtype=np.int32)
    rng.shuffle(p)
    return np.concatenate([p, p])


def _grad2d(hash_val: NDArray[np.int32], x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute gradient dot product for 2D Perlin noise."""
    h = hash_val & 3
    # 4 gradient vectors: (1,1), (-1,1), (1,-1), (-1,-1)
    u = np.where(h < 2, x, y)
    v = np.where(h < 2, y, x)
    return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)


def perlin_noise(
    width: int,
    height: int,
    scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate 2D Perlin noise.

    Args:
        width: Output width in pixels
        height: Output height in pixels
        scale: Noise scale (higher = more zoomed out)
        offset_x: X offset for tiling/variation
        offset_y: Y offset for tiling/variation
        seed: Random seed for reproducibility

    Returns:
        2D array of noise values in range [-1, 1]

    The noise is seamlessly tileable: the lattice wraps with an integer
    period (``scale`` rounded to the nearest whole number of cells), and
    samples exclude the end point so the right edge flows into the left.
    """
    rng = np.random.default_rng(seed if seed is not None else 0)
    perm = _generate_permutation(rng)
    period = max(1, int(round(scale)))

    # Generate coordinate grids (endpoint excluded so the texture tiles)
    x = np.linspace(offset_x, offset_x + period, width, endpoint=False, dtype=np.float64)
    y = np.linspace(offset_y, offset_y + period, height, endpoint=False, dtype=np.float64)
    xv, yv = np.meshgrid(x, y)

    # Integer and fractional parts; lattice indices wrap every `period` cells
    x0 = np.floor(xv).astype(np.int64)
    y0 = np.floor(yv).astype(np.int64)
    xf = xv - x0
    yf = yv - y0
    xi, xi1 = (x0 % period) & 255, ((x0 + 1) % period) & 255
    yi, yi1 = (y0 % period) & 255, ((y0 + 1) % period) & 255

    # Fade curves
    u = _fade(xf)
    v = _fade(yf)

    # Hash coordinates of cube corners
    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yi1]
    ba = perm[perm[xi1] + yi]
    bb = perm[perm[xi1] + yi1]

    # Gradient dot products
    g_aa = _grad2d(aa, xf, yf)
    g_ba = _grad2d(ba, xf - 1, yf)
    g_ab = _grad2d(ab, xf, yf - 1)
    g_bb = _grad2d(bb, xf - 1, yf - 1)

    # Bilinear interpolation
    x1 = _lerp(g_aa, g_ba, u)
    x2 = _lerp(g_ab, g_bb, u)
    result = _lerp(x1, x2, v)

    return result


def directional_noise(
    width: int,
    height: int,
    along: float,
    across: float,
    angle: float,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """2D Perlin noise in streaks: ``along`` cells per tile in the direction ``angle`` (radians,
    0 = the image's x axis) and ``across`` cells per tile at right angles to it, so
    ``along`` < ``across`` gives streaks that run along the direction. Values in [-1, 1].

    It tiles at any angle. The noise coordinates are an integer matrix times the tile's unit
    coordinates (the frequencies rounded to whole cells), so crossing the tile moves a sample by
    a whole lattice vector, and corner hashes are taken modulo that lattice.
    """
    rng = np.random.default_rng(seed if seed is not None else 0)
    perm = _generate_permutation(rng)
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    # u runs across the streaks, v along them: (u, v) = [[a, b], [c, d]] @ (x, y), x and y in [0, 1).
    a, b = int(round(-across * sin_a)), int(round(across * cos_a))
    c, d = int(round(along * cos_a)), int(round(along * sin_a))
    if (a, b) == (0, 0):                                # under half a cell across: one cell
        a, b = (0, 1) if abs(cos_a) >= abs(sin_a) else (1, 0)
    if a * d - b * c == 0:                              # rounding made the rows parallel
        g = int(np.gcd(a, b))
        c, d = b // g, -a // g
    det = abs(a * d - b * c)
    # The lattice of tile translations, (a, c) and (b, d), has the basis (det / g, 0) and (e, g).
    g = int(np.gcd(c, d))
    _, k, m = _extended_gcd(c, d)
    e, period = k * a + m * b, det // g

    x, y = np.meshgrid(np.arange(width) / width, np.arange(height) / height)
    u, v = a * x + b * y, c * x + d * y
    u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    uf, vf = u - u0, v - v0

    def corner(i: NDArray[np.int64], j: NDArray[np.int64]) -> NDArray[np.int32]:
        q = np.floor_divide(j, g)
        return perm[(perm[np.mod(i - q * e, period) & 255] + (j - q * g)) & 255]

    fu, fv = _fade(uf), _fade(vf)
    low = _lerp(_grad2d(corner(u0, v0), uf, vf), _grad2d(corner(u0 + 1, v0), uf - 1, vf), fu)
    high = _lerp(_grad2d(corner(u0, v0 + 1), uf, vf - 1), _grad2d(corner(u0 + 1, v0 + 1), uf - 1, vf - 1), fu)
    return _lerp(low, high, fv)


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """(g, k, m) with k * a + m * b == g == gcd(a, b) > 0 (b may be 0, not both)."""
    if b == 0:
        return (abs(a), 1 if a > 0 else -1, 0)
    g, k, m = _extended_gcd(b, a % b)
    return (g, m, k - (a // b) * m)


def fractal_noise(
    width: int,
    height: int,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    scale: float = 4.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate fractal Brownian motion (fBm) noise.

    Combines multiple octaves of Perlin noise for more natural-looking results.

    Args:
        width: Output width in pixels
        height: Output height in pixels
        octaves: Number of noise layers to combine
        persistence: Amplitude multiplier per octave (typically 0.5)
        lacunarity: Frequency multiplier per octave (typically 2.0)
        scale: Base noise scale
        offset_x: X offset for variation
        offset_y: Y offset for variation
        seed: Random seed for reproducibility

    Returns:
        2D array of noise values, normalized to approximately [-1, 1]
    """
    result = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for i in range(octaves):
        noise = perlin_noise(
            width, height,
            scale=scale * frequency,
            offset_x=offset_x * frequency,
            offset_y=offset_y * frequency,
            seed=seed + i if seed is not None else i,
        )
        result += noise * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize to approximately [-1, 1]
    return result / max_amplitude


def turbulence(
    width: int,
    height: int,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    scale: float = 4.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate turbulence noise (absolute value of fractal noise).

    Useful for creating veiny or marble-like patterns.

    Args:
        width: Output width in pixels
        height: Output height in pixels
        octaves: Number of noise layers
        persistence: Amplitude multiplier per octave
        lacunarity: Frequency multiplier per octave
        scale: Base noise scale
        seed: Random seed

    Returns:
        2D array of noise values in range [0, 1]
    """
    result = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for i in range(octaves):
        noise = perlin_noise(
            width, height,
            scale=scale * frequency,
            seed=seed + i if seed is not None else i,
        )
        result += np.abs(noise) * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return result / max_amplitude
