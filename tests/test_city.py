"""City layout (layout/city.py): street grid, lots, zoning, instancing and furniture."""

import numpy as np
import pytest

from geogen.layout.city import CityBuilder
from geogen.layout.composer import SceneComposer

CITY = """
name: test_city
city:
  seed: {seed}
  blocks: [2, 2]
  block_size: [40, 32]
  street_width: 8
  avenues: {{ ew: [1] }}
  avenue_width: 12
  sidewalk: 3
  lot_width: [10, 14]
  parks: ["1,1"]
  buildings:
    residential:
      - {{ asset: house_simple.yaml, params: {{ width: 7, depth: 7 }} }}
    commercial:
      - {{ asset: house_simple.yaml, params: {{ width: 9, depth: 9 }}, setback: 0 }}
  furniture:
    lamp: {{ asset: street_lamp.yaml, spacing: 20 }}
    tree: {{ asset: maple_tree.yaml, spacing: 10 }}
    hydrant: {{ asset: fire_hydrant.yaml, per_edge: 1 }}
    bench: {{ asset: bench.yaml, per_edge: 1, zones: [commercial] }}
"""


@pytest.fixture(scope="module")
def city():
    return SceneComposer().compose_string(CITY.format(seed=3))


def _lots(root):
    return [n for n in root.iter_nodes() if "lot" in n.meta]


def _footprint(node):
    pts = []
    for n in node.iter_nodes():
        if n.mesh is not None and len(n.mesh.vertices):
            v = n.mesh.vertices
            pts.append((n.world_transform() @ np.c_[v, np.ones(len(v))].T).T[:, :3])
    pts = np.vstack(pts)
    return pts.min(axis=0), pts.max(axis=0)


def test_grid_dimensions_and_walkable_streets(city):
    # 3 streets (8 + 12 + 8) and 2 blocks along z; 3 x 8 m streets and 2 blocks along x.
    np.testing.assert_allclose(city.size[[0, 2]], [8 * 3 + 80, 8 + 12 + 8 + 64])
    streets = next(n for n in city.iter_nodes() if n.name == "streets")
    assert streets.meta["walkable"]
    _, hi = _footprint(streets)
    assert hi[1] == pytest.approx(0.0)
    sidewalks = [n for n in city.iter_nodes() if "street.sidewalk" in n.tags]
    assert len(sidewalks) == 4 and all(n.meta["walkable"] for n in sidewalks)
    assert _footprint(sidewalks[0])[1][1] == pytest.approx(0.15)


def test_lots_tile_each_block():
    builder = CityBuilder({"blocks": [2, 2], "block_size": [40, 32], "avenues": {"ew": [1]}, "seed": 3,
                           "parks": ["1,1"], "sidewalk": 3, "lot_width": [10, 14]}, None, None)
    lots = builder.lots()
    for i in range(2):
        for j in range(2):
            area = sum(float(np.prod(lt.hi - lt.lo)) for lt in lots if lt.block == (i, j))
            assert area == pytest.approx((40 - 6) * (32 - 6))
    widths = [lt.frontage for lt in lots if lt.zone != "park"]
    assert min(widths) >= 14 / 2 - 1e-9 and max(widths) <= 14 + 1e-9   # slivers are split in half


def test_zoning_follows_the_avenue(city):
    lots = [n.meta["lot"] for n in _lots(city)]
    # Blocks j=0 front the avenue (index 1) on their north side, blocks j=1 on their south side.
    for lot in lots:
        if lot["zone"] == "park":
            assert lot["block"] == [1, 1]
            continue
        fronts_avenue = (lot["block"][1] == 0 and lot["side"] == "north") or \
                        (lot["block"][1] == 1 and lot["side"] == "south")
        assert (lot["zone"] == "commercial") == fronts_avenue


def test_buildings_fit_their_lots_and_face_the_street(city):
    count = 0
    for lot_node in _lots(city):
        lot = lot_node.meta["lot"]
        for building in (c for c in lot_node.children if "building" in c.meta):
            count += 1
            lo, hi = _footprint(building)
            x0, z0, x1, z1 = lot["rect"]
            assert x0 - 1e-6 <= lo[0] and hi[0] <= x1 + 1e-6 and z0 - 1e-6 <= lo[2] and hi[2] <= z1 + 1e-6
            forward = building.world_transform()[:3, 2]
            expected = {"north": [0, 0, 1], "south": [0, 0, -1], "east": [1, 0, 0], "west": [-1, 0, 0]}
            np.testing.assert_allclose(forward, expected[lot["side"]], atol=1e-9)
            if lot["zone"] == "commercial":   # built to the sidewalk
                edge = {"north": z1, "south": z0}[lot["side"]]
                front = hi[2] if lot["side"] == "north" else lo[2]
                assert front == pytest.approx(edge, abs=0.05)
    assert count >= 8


def test_buildings_share_meshes(city):
    houses = [n for n in city.iter_nodes() if "building" in n.meta
              and n.meta["building"]["zone"] == "residential"]
    assert len(houses) >= 2
    mesh_ids = [{id(m.mesh) for m in h.iter_nodes() if m.mesh is not None} for h in houses]
    assert mesh_ids[0] == mesh_ids[1]


def test_street_furniture_stays_on_sidewalks_and_apart(city):
    furniture = [n for n in city.iter_nodes() if "street_furniture" in n.meta]
    kinds = {n.meta["street_furniture"] for n in furniture}
    assert {"lamp", "tree", "hydrant", "bench"} <= kinds
    pts = np.array([n.transform.translation[[0, 2]] for n in furniture])
    assert all(n.transform.translation[1] == pytest.approx(0.15) for n in furniture)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(len(pts)) * 99
    assert d.min() >= 1.2 - 1e-9
    # Benches only along blocks with commercial lots (their frontage and side streets), never the park.
    benches = [n for n in furniture if n.meta["street_furniture"] == "bench"]
    assert 1 <= len(benches) <= 3 * 3
    park = next(n for n in _lots(city) if n.meta["lot"]["zone"] == "park").meta["lot"]["rect"]
    for bench in benches:
        x, z = bench.transform.translation[[0, 2]]
        assert not (park[0] - 3.5 <= x <= park[2] + 3.5 and park[1] - 3.5 <= z <= park[3] + 3.5)


def test_markings_and_park(city):
    names = {n.name for n in city.iter_nodes()}
    assert {"markings_road_paint_white", "markings_road_paint_yellow"} <= names
    park = next(n for n in _lots(city) if n.meta["lot"]["zone"] == "park")
    assert sum(c.name.startswith("park_tree") for c in park.children) >= 5


def test_seeded(city):
    again = SceneComposer().compose_string(CITY.format(seed=3))
    other = SceneComposer().compose_string(CITY.format(seed=9))

    def rects(root):
        return [tuple(n.meta["lot"]["rect"]) for n in _lots(root)]

    assert rects(again) == rects(city)
    assert rects(other) != rects(city)


def test_unknown_keys_raise():
    with pytest.raises(ValueError, match="unknown keys"):
        SceneComposer().compose_string("name: x\ncity: { blocks: [1, 1], blok_size: [10, 10] }\n")


def test_registry_lists_town():
    from geogen.registry import SceneRegistry

    registry = SceneRegistry()
    registry.discover()
    assert "town" in registry.names()


def test_manifest_prefers_the_scenes_own_spawn():
    from geogen.core.node import SceneNode
    from geogen.core.transform import Transform
    from geogen.export import gameplay_summary

    root = SceneNode("root")
    building = SceneNode("building")
    building.add_child(SceneNode("entrance_spawn", meta={"type": "spawn"}))
    root.add_child(building)
    root.add_child(SceneNode("start", transform=Transform(translation=np.array([1.0, 0, 2])),
                             meta={"type": "spawn"}))
    assert [s["name"] for s in gameplay_summary(root)["spawns"]] == ["start", "entrance_spawn"]
