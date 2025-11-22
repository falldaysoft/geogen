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
    """
    rng = np.random.default_rng(seed if seed is not None else 0)
    perm = _generate_permutation(rng)

    # Generate coordinate grids
    x = np.linspace(offset_x, offset_x + scale, width, dtype=np.float64)
    y = np.linspace(offset_y, offset_y + scale, height, dtype=np.float64)
    xv, yv = np.meshgrid(x, y)

    # Integer and fractional parts
    xi = xv.astype(np.int32) & 255
    yi = yv.astype(np.int32) & 255
    xf = xv - np.floor(xv)
    yf = yv - np.floor(yv)

    # Fade curves
    u = _fade(xf)
    v = _fade(yf)

    # Hash coordinates of cube corners
    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]

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
