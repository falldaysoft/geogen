#!/usr/bin/env python
"""Keep the screenshots in docs/images small: at most MAX_WIDTH wide, PNGs reduced to a 256-colour palette.

Every regenerated image stays in the repository's history for good, so make_screenshots.sh runs this
on what it writes. Usage: python docs/shrink_screenshots.py [image ...] (default: every PNG there)."""

import sys
from pathlib import Path

from PIL import Image

MAX_WIDTH = 1280
IMAGES = Path(__file__).parent / "images"


def shrink(path: Path) -> None:
    image = Image.open(path).convert("RGB")
    if image.width > MAX_WIDTH:
        image = image.resize((MAX_WIDTH, round(image.height * MAX_WIDTH / image.width)), Image.LANCZOS)
    image.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(path, optimize=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or sorted(IMAGES.glob("*.png")):
        shrink(Path(name))
