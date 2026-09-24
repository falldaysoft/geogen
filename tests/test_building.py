"""Multi-storey buildings."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.layout import LayoutLoader

SMALL = """
name: small_building
building:
  storeys:
    - floorplan: { generate: hotel_lobby, length: 30, depth: 15, wall_height: 3.6, finishes: false }
    - floorplan: { generate: hotel_corridor, length: 30, depth: 15, finishes: false }
      repeat: 2
  roof: { parapet: 1.0 }
"""


@pytest.fixture(scope="module")
def building():
    return LayoutLoader().load_string(SMALL)


def test_storeys_stack_on_the_walls_below(building):
    storeys = [c for c in building.children if "storey" in c.tags]
    assert [s.name for s in storeys] == ["storey_0", "storey_1", "storey_2"]
    assert [s.meta["storey"]["elevation"] for s in storeys] == pytest.approx([0.0, 3.65, 6.4])
    assert storeys[1].tags == ["storey", "storey.1"]
    # Guest rooms are numbered by storey.
    assert storeys[1].find("room_101") is not None and storeys[2].find("room_201") is not None
    assert building.meta["building"] == {"storeys": 3, "height": pytest.approx(9.1)}


def test_stair_cores_connect_every_storey(building):
    for lower, upper in (("storey_0", "storey_1"), ("storey_1", "storey_2")):
        lo, hi = building.find(lower), building.find(upper)
        for side in ("stair_west", "stair_east"):
            room = lo.find(side)
            stairs = room.find("stairs")
            assert stairs is not None, (lower, side)
            assert meshops.validate(stairs.mesh).watertight
            pts = (stairs.world_transform() @ np.c_[stairs.mesh.vertices, np.ones(len(stairs.mesh.vertices))].T)[1]
            # From this floor to the next one's floor level.
            assert pts.min() == pytest.approx(lo.meta["storey"]["elevation"], abs=1e-6)
            assert pts.max() == pytest.approx(hi.meta["storey"]["elevation"], abs=1e-6)
            # The floor above is cut open over the flight.
            floor_above = hi.find(side).find("floor")
            assert meshops.validate(floor_above.mesh).watertight
            area = np.ptp(floor_above.mesh.vertices[:, 0]) * np.ptp(floor_above.mesh.vertices[:, 2])
            v = floor_above.mesh.vertices[floor_above.mesh.faces]
            top_faces = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])[:, 1] > 0
            top_area = 0.5 * np.cross(v[top_faces, 1] - v[top_faces, 0], v[top_faces, 2] - v[top_faces, 0])[:, 1].sum()
            assert top_area < area - 1.0 * 5.0  # at least a 1 x 5 m stairwell removed


def test_roof_caps_the_top(building):
    roof = building.find("roof")
    assert meshops.validate(roof.mesh).watertight
    assert roof.mesh.vertices[:, 1].min() == pytest.approx(6.4 + 2.7)


def test_building_errors():
    with pytest.raises(ValueError, match="storey"):
        LayoutLoader().load_string("name: b\nbuilding: { storeys: [] }\n")
    with pytest.raises(ValueError, match="floorplan"):
        LayoutLoader().load_string("name: b\nbuilding: { storeys: [ { repeat: 2 } ] }\n")
