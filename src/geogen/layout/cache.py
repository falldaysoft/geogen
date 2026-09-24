"""On-disk cache of generated prototypes (placed assets and nested scenes).

``SceneComposer(cache_dir=...)`` (or ``$GEOGEN_CACHE``, or ``--cache`` on
the command line) pickles each placement's prototype, keyed by its
definition (asset/scene, params, furnish) and a fingerprint of every file
under ``assets/`` and ``src/geogen/``: any edit to an asset, material or
generator invalidates the whole cache, so a stale entry can't be served.

Materials in unpickled prototypes are deduplicated by content, so the same
material from different cache entries generates its textures once.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import pickle
from pathlib import Path

from ..core.node import SceneNode

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]


@functools.lru_cache(maxsize=1)
def source_fingerprint() -> str:
    """Hash of every asset YAML and geogen source file (paths and contents)."""
    digest = hashlib.sha1()
    for base, pattern in ((_ROOT / "assets", "*.yaml"), (_ROOT / "src" / "geogen", "*.py")):
        for path in sorted(base.rglob(pattern)):
            digest.update(str(path.relative_to(_ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class PrototypeCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._materials: dict[str, object] = {}

    def _path(self, key: str) -> Path:
        name = hashlib.sha1(f"{source_fingerprint()}|{key}".encode()).hexdigest()
        return self.directory / f"{name}.pickle"

    def get(self, key: str) -> SceneNode | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            node = pickle.loads(path.read_bytes())
        except Exception as exc:   # a corrupt or incompatible entry is just a miss
            logger.warning("Ignoring unreadable cache entry %s: %s", path.name, exc)
            return None
        self._share_materials(node)
        logger.debug("Cache hit: %s", key)
        return node

    def put(self, key: str, node: SceneNode) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_bytes(pickle.dumps(node, protocol=pickle.HIGHEST_PROTOCOL))
            tmp.replace(path)
        except Exception as exc:   # unpicklable content: skip caching, don't fail the build
            logger.warning("Not caching %s: %s", key, exc)
            tmp.unlink(missing_ok=True)

    def _share_materials(self, node: SceneNode) -> None:
        for n in node.iter_nodes():
            material = n.mesh.material if n.mesh is not None else None
            if material is None:
                continue
            fingerprint = hashlib.sha1(pickle.dumps(material, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
            n.mesh.material = self._materials.setdefault(fingerprint, material)
