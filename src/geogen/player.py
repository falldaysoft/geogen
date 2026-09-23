"""Shared player spec: the body the world is generated for.

``assets/player.yaml`` is the single source of truth for player-scale
constraints (capsule, eye and step height, walkable slope, door/corridor
clearances, reach). Generators and layout checks use it for defaults, the
exporter writes it into the manifest, and the Godot controller reads it back
from there, so "enterable" means the same thing on both sides.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path

from .layout.yaml_utils import safe_load_path

SPEC_KIND = "player_spec"
SPEC_VERSION = 1
DEFAULT_PATH = Path(__file__).parent.parent.parent / "assets" / "player.yaml"


@dataclass(frozen=True)
class PlayerSpec:
    radius: float = 0.3
    height: float = 1.8
    eye_height: float = 1.65
    step_height: float = 0.3
    max_slope_deg: float = 40.0
    door_min_width: float = 0.85
    door_min_height: float = 2.0
    corridor_min_width: float = 1.2
    reach: float = 1.5

    def __post_init__(self) -> None:
        for f in fields(self):
            if getattr(self, f.name) <= 0:
                raise ValueError(f"player spec: {f.name} must be positive")
        if self.height < 2 * self.radius:
            raise ValueError("player spec: height must be at least 2 * radius")
        if self.eye_height >= self.height:
            raise ValueError("player spec: eye_height must be below height")
        if self.step_height >= self.height - self.radius:
            raise ValueError("player spec: step_height too tall for the capsule")
        if not self.max_slope_deg < 90:
            raise ValueError("player spec: max_slope_deg must be under 90")
        if self.door_min_width < 2 * self.radius or self.door_min_height < self.height:
            raise ValueError("player spec: door_min_* must fit the capsule")
        if self.corridor_min_width < 2 * self.radius:
            raise ValueError("player spec: corridor_min_width must fit the capsule")

    @classmethod
    def from_dict(cls, data: dict) -> PlayerSpec:
        data = dict(data)
        kind = data.pop("kind", SPEC_KIND)
        version = data.pop("version", SPEC_VERSION)
        if kind != SPEC_KIND:
            raise ValueError(f"player spec: expected kind '{SPEC_KIND}', got '{kind}'")
        if version != SPEC_VERSION:
            raise ValueError(f"player spec: unsupported version {version}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"player spec: unknown keys {unknown}")
        return cls(**{k: float(v) for k, v in data.items()})

    def to_dict(self) -> dict:
        return {"kind": SPEC_KIND, "version": SPEC_VERSION, **asdict(self)}


def load_player_spec(path: str | Path | None = None) -> PlayerSpec:
    """Load a player spec; with no path, the project's ``assets/player.yaml``."""
    if path is None:
        return _default_spec()
    return PlayerSpec.from_dict(safe_load_path(path) or {})


@lru_cache(maxsize=1)
def _default_spec() -> PlayerSpec:
    return PlayerSpec.from_dict(safe_load_path(DEFAULT_PATH) or {})
