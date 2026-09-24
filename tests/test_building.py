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


def test_entrance_spawn_outside_the_street_door(building):
    spawn = building.find("entrance_spawn")
    assert spawn.meta["type"] == "spawn"
    m = spawn.world_transform()
    assert m[1, 3] == 0.0 and m[2, 3] < -7.5          # in front of the south façade
    assert m[:3, 2] == pytest.approx([0, 0, 1], abs=1e-9)  # facing the building


def test_roof_caps_the_top(building):
    roof = building.find("roof")
    assert meshops.validate(roof.mesh).watertight
    assert roof.mesh.vertices[:, 1].min() == pytest.approx(6.4 + 2.7)


def test_building_errors():
    with pytest.raises(ValueError, match="storey"):
        LayoutLoader().load_string("name: b\nbuilding: { storeys: [] }\n")
    with pytest.raises(ValueError, match="floorplan"):
        LayoutLoader().load_string("name: b\nbuilding: { storeys: [ { repeat: 2 } ] }\n")


def test_lift_serves_every_storey(building):
    lift = building.find("lift_lift_shaft")
    (it,) = lift.interactions
    assert list(it.states) == ["floor_0", "floor_1", "floor_2"]
    assert it.states["floor_2"].next == "floor_0"          # wraps to the lobby
    assert it.motions[0].values == pytest.approx({"floor_0": 0.0, "floor_1": 3.65, "floor_2": 6.4})
    gates = [c for c in lift.children if "lift.gate" in c.tags]
    assert [g.meta["gate"]["open_in"] for g in gates] == ["floor_0", "floor_1", "floor_2"]
    # The shaft is open: only the bottom room keeps its floor, only the top its ceiling.
    shafts = [building.find(f"storey_{k}").find("lift_shaft") for k in range(3)]
    assert [s.find("floor") is not None for s in shafts] == [True, False, False]
    assert [s.find("ceiling") is not None for s in shafts] == [False, False, True]
    assert all(s.find("lift_shaft_volume").meta["nav"] is False for s in shafts)
