"""What several test modules share (import as ``from helpers import ...``; tests/ is on sys.path)."""

import json
import struct
from pathlib import Path


def gltf_json(path) -> dict:
    """The JSON chunk of a ``.glb``."""
    data = Path(path).read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])
