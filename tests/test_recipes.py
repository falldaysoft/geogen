"""Building recipes (layout/recipes.py), pitched roofs, and recipe fitting in cities."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.layout.composer import SceneComposer
from geogen.layout.qa import check_layout
from geogen.layout.recipes import RecipeError, _cap_storeys, front_of, recipe_spec


@pytest.fixture(scope="module")
def composer():
    return SceneComposer()


def build(composer, name, **params):
    return composer._load_object({"recipe": name, "params": params})


def _rooms(node):
    return [n for n in node.iter_nodes() if isinstance(n.meta.get("room"), dict) and n.meta.get("type") != "room_volume"]


@pytest.mark.parametrize("name, params", [
    ("detached_house", {"width": 9, "depth": 8, "storeys": 1}),
    ("detached_house", {"width": 8, "depth": 9, "storeys": 2}),
    ("shop_row", {"width": 10, "depth": 12, "storeys": 3}),
    ("office", {"width": 14, "depth": 12, "storeys": 3}),
    ("apartment_block", {"width": 26, "depth": 15, "storeys": 3, "interior": "lobby"}),
])
def test_recipes_build_clean(composer, name, params):
    building = build(composer, name, **params)
    assert "building" in building.tags and f"building.{name}" in building.tags
    assert building.meta["building"]["storeys"] == params["storeys"]
    for node, mesh in building.iter_meshes():
        assert not meshops.validate(mesh).issues, node.name
    assert [str(i) for i in check_layout(building)] == []
    # Multi-storey recipes get working stairs.
    if params["storeys"] > 1:
        assert any("stairs" in n.meta for n in building.iter_nodes())
    # Footprint follows the requested size (outer wall skin adds a wall thickness).
    walls = [n for n in building.iter_nodes() if "wall" in n.tags and n.mesh is not None]
    pts = np.vstack([(n.world_transform() @ np.c_[n.mesh.vertices, np.ones(len(n.mesh.vertices))].T).T
                     for n in walls])
    extent = pts.max(axis=0) - pts.min(axis=0)
    np.testing.assert_allclose(extent[[0, 2]], [params["width"] + 0.3, params["depth"] + 0.3], atol=0.05)


def test_house_has_a_gable_roof(composer):
    house = build(composer, "detached_house", width=9, depth=8, storeys=1)
    roof = next(n for n in house.iter_nodes() if n.name == "roof")
    assert "roof.gable" in roof.tags and roof.meta["roof"]["rise"] > 1.5
    gables = roof.find("gables")
    assert gables is not None and meshops.validate(gables.mesh).ok
    assert house.size[1] > 2.7 + 1.5


def test_front_follows_the_entrance(composer):
    assert list(front_of(build(composer, "detached_house", width=9, depth=8))) == [0.0, 1.0]
    assert list(front_of(build(composer, "hotel", width=30, depth=15, storeys=2, interior="shell"))) == [0.0, -1.0]


def test_too_small_raises(composer):
    with pytest.raises(RecipeError):
        build(composer, "office", width=6, depth=6, storeys=2)


def test_storeys_capped_by_stair_length():
    assert _cap_storeys(4, 9.0, 3.4, 2.7) == 2
    assert _cap_storeys(4, 14.0, 3.4, 2.7) == 4
    assert len(recipe_spec("shop_row", {"width": 10, "depth": 10, "storeys": 3})["storeys"]) == 2


def test_interior_modes(composer):
    full = build(composer, "shop_row", width=10, depth=12, storeys=2, interior="full")
    lobby = build(composer, "shop_row", width=10, depth=12, storeys=2, interior="lobby")
    shell = build(composer, "shop_row", width=10, depth=12, storeys=2, interior="shell")

    def furniture_per_storey(node):
        counts = {}
        for storey in (n for n in node.children if "storey" in n.tags):
            counts[storey.name] = sum(1 for n in storey.iter_nodes() if n.meta.get("placed_by") == "furnish")
        return counts

    assert all(v > 0 for v in furniture_per_storey(full).values())
    counts = furniture_per_storey(lobby)
    assert counts["storey_0"] > 0 and counts["storey_1"] == 0
    assert _rooms(lobby)                                   # upper rooms still exist, just empty
    assert not _rooms(shell) and "building.shell" in shell.tags
    assert sum(1 for n in shell.iter_nodes() if "shell.void" in n.tags) == 2
    locks = [i.lock for n in shell.iter_nodes() for i in n.interactions]
    assert locks and all(lock["locked"] for lock in locks)


CITY = """
name: recipe_city
city:
  seed: 2
  blocks: [2, 1]
  block_size: [36, 32]
  avenues: { ew: [0] }
  lot_width: [11, 14]
  buildings:
    residential: [{ recipe: detached_house, storeys: [1, 2], interior: shell }]
    commercial: [{ recipe: shop_row, storeys: 2, interior: shell }]
"""


def test_city_fits_recipes_to_lots():
    root = SceneComposer().compose_string(CITY)
    lots = [n for n in root.iter_nodes() if "lot" in n.meta]
    placed = 0
    for lot_node in lots:
        lot = lot_node.meta["lot"]
        building = next((c for c in lot_node.children if "building" in c.meta and "zone" in c.meta["building"]),
                        None)
        assert building is not None, lot_node.name
        placed += 1
        x0, z0, x1, z1 = lot["rect"]
        walls = [n for n in building.iter_nodes() if "wall" in n.tags and n.mesh is not None]
        pts = np.vstack([(n.world_transform() @ np.c_[n.mesh.vertices, np.ones(len(n.mesh.vertices))].T).T
                         for n in walls])
        assert pts[:, 0].min() >= x0 - 1e-6 and pts[:, 0].max() <= x1 + 1e-6
        assert pts[:, 2].min() >= z0 - 1e-6 and pts[:, 2].max() <= z1 + 1e-6
        # The entrance is on the street side of the building.
        spawn = next(n for n in building.iter_nodes() if n.name == "entrance_spawn").world_transform()[[0, 2], 3]
        centre = (pts.min(axis=0) + pts.max(axis=0))[[0, 2]] / 2
        out = {"north": [0, 1], "south": [0, -1], "east": [1, 0], "west": [-1, 0]}[lot["side"]]
        assert float(np.dot(spawn - centre, out)) > 0
        # Commercial frontage is built up to the sidewalk (walls, not the canopy).
        if lot["zone"] == "commercial":
            edge = {"north": z1, "south": z0}[lot["side"]]
            front = pts[:, 2].max() if lot["side"] == "north" else pts[:, 2].min()
            assert front == pytest.approx(edge, abs=0.05)
    assert placed == len(lots) and placed >= 6
