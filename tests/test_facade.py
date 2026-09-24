"""Building façades."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.layout import LayoutLoader

BUILDING = """
name: b
building:
  storeys:
    - floorplan: {{ generate: hotel_lobby, length: 30, depth: 15, wall_height: 3.6, finishes: false }}
    - floorplan: {{ generate: hotel_corridor, length: 30, depth: 15, finishes: false }}
      repeat: 2
  facade: {{ style: {style}, balconies: {balconies} }}
"""


def _build(style="brick_hotel", balconies="false"):
    return LayoutLoader().load_string(BUILDING.format(style=style, balconies=balconies))


@pytest.mark.parametrize("style, cladding, frame", [
    ("brick_hotel", "brick", "trim_white"), ("stucco", "wall_plaster", "plastic_black"), ("modern", "concrete", "metal"),
])
def test_styles_clad_walls_and_frame_windows(style, cladding, frame):
    root = _build(style)
    for storey in ("storey_0", "storey_1", "storey_2"):
        assert root.find(storey).find("walls").mesh.material.name == cladding
    facade = root.find("facade")
    assert facade.tags == ["facade", f"facade.{style}"]
    assert facade.find("facade_frame").mesh.material.name == frame
    assert facade.find("facade_glass").mesh.material.name == "glass"
    assert [c.meta["storey"]["index"] for c in facade.children] == [0, 1, 2]
    for part in facade.iter_nodes():
        if part.mesh is not None:
            assert meshops.validate(part.mesh).nan_values == 0


def test_every_exterior_window_gets_glass_in_its_opening():
    root = _build()
    # Separate panes: 8 guest + 2 stair windows per guest floor, 2 storefronts on the lobby.
    import trimesh
    panes = 0
    for node in root.find("facade").iter_nodes():
        if node.name == "facade_glass":
            m = node.mesh
            panes += len(trimesh.Trimesh(m.vertices, m.faces, process=True).split(only_watertight=False))
    assert panes == 2 * (8 + 2) + 2


def test_canopy_over_street_door_and_juliet_balconies():
    plain = _build()
    assert plain.find("facade_canopy") is not None and plain.find("facade_rail") is None
    assert plain.find("facade_canopy").meta["collider"] == "box"
    with_balconies = _build(balconies="true")
    assert with_balconies.find("facade_rail") is not None


def test_bad_style():
    with pytest.raises(ValueError, match="facade style"):
        _build(style="gothic")
