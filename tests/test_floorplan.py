"""FloorPlan: wall segments, clean junctions, openings, room nodes and surfaces."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.generators.floorplan import FloorPlan
from geogen.layout import LayoutLoader, SceneComposer

SUITE = {
    "rooms": {
        "bedroom": {"rect": [0, 0, 4.4, 4.6], "type": "hotel_bedroom"},
        "bathroom": {"rect": [4.4, 0, 2.4, 2.6], "type": "hotel_bathroom"},
        "corridor": {"rect": [4.4, 2.6, 2.4, 2.0]},
    },
}


def _volume(mesh):
    v = mesh.vertices[mesh.faces]
    return np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6


def test_segments_classify_interior_and_exterior():
    segs = FloorPlan.from_spec(SUITE).segments()
    interior = [s for s in segs if not s.exterior]
    assert len(interior) == 3  # bedroom|bathroom, bedroom|corridor, bathroom|corridor
    assert all(s.thickness == 0.12 for s in interior)
    assert all(s.thickness == 0.3 for s in segs if s.exterior)
    perimeter = sum(s.length for s in segs if s.exterior)
    assert perimeter == pytest.approx(2 * (6.8 + 4.6))


def test_wall_footprint_has_no_overlaps_or_gaps():
    plan = FloorPlan.from_spec(SUITE)
    footprint = plan._wall_footprint(plan.segments())
    # Outer boundary = centre-line bounds grown by half the exterior wall.
    assert footprint.bounds == pytest.approx((-0.15, -0.15, 6.95, 4.75))
    # Holes are exactly the rooms' clear rectangles (no wall stubs inside rooms).
    clear = sorted(round(h.area, 6) for h in [type(footprint)(i) for i in footprint.interiors])
    expected = sorted(round(a, 6) for a in [
        (4.4 - 0.15 - 0.06) * (4.6 - 0.3),        # bedroom
        (2.4 - 0.06 - 0.15) * (2.6 - 0.15 - 0.06),  # bathroom
        (2.4 - 0.06 - 0.15) * (2.0 - 0.06 - 0.15),  # corridor
    ])
    assert clear == pytest.approx(expected)


def test_walls_are_watertight_with_openings():
    spec = dict(SUITE, doors=[{"between": ["bedroom", "corridor"], "width": 0.9}],
                windows=[{"room": "bedroom", "side": "south", "width": 2.0, "height": 1.5, "sill": 0.7}])
    plain = FloorPlan.from_spec(SUITE).build("plain")
    cut = FloorPlan.from_spec(spec).build("cut")
    before, after = plain.find("walls").mesh, cut.find("walls").mesh
    assert meshops.validate(after).watertight
    removed = _volume(before) - _volume(after)
    assert removed == pytest.approx(0.9 * 2.1 * 0.12 + 2.0 * 1.5 * 0.3, rel=1e-3)


def test_room_nodes_have_slabs_surfaces_and_tags():
    root = FloorPlan.from_spec(SUITE).build("suite")
    bedroom = root.find("bedroom")
    assert bedroom.tags == {"room": "bedroom", "room_type": "hotel_bedroom"}
    assert {c.name for c in bedroom.children} == {"floor", "ceiling"}
    assert {"floor", "ceiling", "north_wall", "west_exterior"} <= set(bedroom.surfaces)
    assert "east_exterior" not in bedroom.surfaces  # east side is shared with other rooms
    floor = root.surfaces["bedroom.floor"]
    assert floor.u_extent == pytest.approx(4.4 - 0.15 - 0.06)
    # Interior wall surface sits on the wall face, normal into the room.
    north = root.surfaces["bedroom.north_wall"]
    assert north.normal == pytest.approx([0, 0, -1])
    assert north.origin[2] == pytest.approx(4.6 - 0.15 - 2.3)


@pytest.mark.parametrize("spec, match", [
    ({"rooms": {"a": {"rect": [0, 0, 2, 2]}, "b": {"rect": [1, 1, 2, 2]}}}, "overlap"),
    ({"rooms": {"a": {"rect": [0, 0, 2.05, 2]}}}, "grid"),
    ({"rooms": {"a": {"rect": [0, 0, 2, 2]}}, "doors": [{"between": ["a", "zz"]}]}, "unknown room"),
    ({"rooms": {"a": {"rect": [0, 0, 2, 2]}, "b": {"rect": [3, 0, 2, 2]}},
      "doors": [{"between": ["a", "b"]}]}, "don't share"),
    ({"rooms": {"a": {"rect": [0, 0, 2, 2]}}, "windows": [{"room": "a", "side": "north", "width": 3}]},
     "doesn't fit"),
])
def test_invalid_plans_raise(spec, match):
    with pytest.raises(ValueError, match=match):
        FloorPlan.from_spec(spec).build()


def test_window_asset_on_exterior_surface_cuts_walls():
    composer = SceneComposer()
    scene = composer.compose_string("""
name: s
place:
  suite: { asset: hotel_suite.yaml }
  extra_window:
    asset: window.yaml
    on: suite.bedroom.west_exterior
    at: { u: 0.5, v: { abs: 1.0 } }
""")
    plain = LayoutLoader().load("assets/hotel_suite.yaml")
    walls = scene.find("walls").mesh
    assert meshops.validate(walls).watertight
    removed = _volume(plain.find("walls").mesh) - _volume(walls)
    assert removed == pytest.approx(1.0 * 1.3 * 0.3, rel=1e-3)
