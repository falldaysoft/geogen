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
    """Material definition with texture and rendering properties.

    A material combines a procedurally generated texture with
    rendering properties like shininess and color tinting.

    Attributes:
        name: Material identifier
        texture_generator: Generator to create the diffuse texture
        texture_size: Size of generated texture (width, height)
        shininess: Specular shininess (0-1, higher = more shiny)
        tint: Optional color tint multiplier (R, G, B) normalized 0-1
    """

    name: str
    texture_generator: TextureGenerator
    texture_size: tuple[int, int] = (512, 512)
    shininess: float = 0.3
    tint: tuple[float, float, float] | None = None

    _cached_texture: Image.Image | None = field(default=None, repr=False)

    def get_texture(self) -> Image.Image:
        """Generate or return cached texture image.

        Returns:
            PIL Image in RGB mode
        """
        if self._cached_texture is None:
            # Update generator size
            self.texture_generator.width = self.texture_size[0]
            self.texture_generator.height = self.texture_size[1]
            self._cached_texture = self.texture_generator.generate()

            # Apply tint if specified
            if self.tint is not None:
                self._cached_texture = self._apply_tint(self._cached_texture)

        return self._cached_texture

    def _apply_tint(self, image: Image.Image) -> Image.Image:
        """Apply color tint to texture."""
        arr = np.array(image, dtype=np.float64)
        tint_arr = np.array(self.tint)
        arr *= tint_arr
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode='RGB')

    def get_texture_array(self) -> np.ndarray:
        """Get texture as numpy array.

        Returns:
            HxWx3 uint8 array in RGB format
        """
        return np.array(self.get_texture())

    def invalidate_cache(self) -> None:
        """Clear cached texture, forcing regeneration on next access."""
        self._cached_texture = None
