"""Base class for procedural texture generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass
class TextureGenerator(ABC):
    """Abstract base class for procedural texture generators.

    Subclasses implement generate() to create texture images from noise
    and other procedural techniques.
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

    def save(self, path: str) -> None:
        """Generate and save texture to file.

        Args:
            path: Output file path (e.g., 'texture.png')
        """
        self.generate().save(path)
