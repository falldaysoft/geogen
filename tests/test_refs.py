"""References: syntax, scoped resolution, selectors, stable IDs and their index."""

import pytest

from geogen import refs
from geogen.core.node import SceneNode
from geogen.refs import RefError


def _asset(name, asset, *children, tags=()):
    node = SceneNode(name, tags=list(tags), meta={"source": {"asset": f"{asset}.yaml", "params": {}}})
    for child in children:
        node.add_child(child)
    return node


@pytest.fixture
def town():
    """town: house (asset: rooms with furniture), a nested park scene, two lamps."""
    root = SceneNode("town")
    bedroom = SceneNode("bedroom", tags=["room", "room.bedroom"])
    bedroom.add_child(_asset("bed", "bed", SceneNode("frame"), tags=["furniture.bed"]))
    bedroom.add_child(_asset("wardrobe", "wardrobe", SceneNode("door"), tags=["furniture.wardrobe"]))
    kitchen = SceneNode("kitchen", tags=["room", "room.kitchen"])
    kitchen.add_child(_asset("chair", "chair", tags=["furniture.chair"]))
    kitchen.add_child(_asset("chair", "chair", tags=["furniture.chair"]))
    root.add_child(_asset("house", "house", SceneNode("walls"), bedroom, kitchen))
    root.add_child(_asset("door", "door", SceneNode("leaf_pivot", children=[]), tags=["door.exterior"]))
    root.children[-1].children[0].add_child(SceneNode("leaf"))
    park = SceneNode("park", meta={"scope": True})
    park.add_child(_asset("bench", "bench", tags=["furniture.bench"]))
    park.add_child(SceneNode("arrival", tags=["spawn"], meta={"type": "spawn"}))
    root.add_child(park)
    root.add_child(_asset("lamp", "street_lamp"))
    root.add_child(_asset("lamp", "street_lamp"))
    return root


def test_parse_forms():
    ref = refs.parse("town:park/[tag=furniture.bench]")
    assert ref.package == "town"
    assert [str(s) for s in ref.steps] == ["park", "[tag=furniture.bench]"]
    assert refs.parse("house").package is None
    ref = refs.parse("suite.bedroom.south_wall", member=True)
    assert (ref.steps[0].name, ref.member) == ("suite", "bedroom.south_wall")
    ref = refs.parse("town:plot/house[asset=house].floor", member=True)
    assert (ref.steps[1].name, ref.steps[1].filters, ref.member) == ("house", {"asset": "house"}, "floor")
    assert refs.parse("a.b").steps[0].name == "a.b"           # no member unless the key takes one


@pytest.mark.parametrize("text, message", [
    ("", "expected a reference"),
    ("a//b", "can't read step"),
    ("a/[tag]", "should be key=value"),
    ("[colour=red]", "unknown filter 'colour'"),
    ("house.", "nothing after"),
])
def test_parse_errors(text, message):
    with pytest.raises(RefError, match=message):
        refs.parse(text, member=True)


def test_names_resolve_within_a_scope(town):
    assert refs.resolve(town, "house").name == "house"
    # 'door' is the door placement, not the wardrobe's door part inside the house.
    assert refs.path_of(refs.resolve(town, "door")) == "door"
    assert refs.path_of(refs.resolve(town, "house/wardrobe/door")) == "house/bedroom/wardrobe/door"
    # Steps skip the nodes in between that belong to the same scope.
    assert refs.path_of(refs.resolve(town, "house/bed")) == "house/bedroom/bed"
    assert refs.path_of(refs.resolve(town, "door/leaf")) == "door/leaf_pivot/leaf"
    assert refs.resolve(town, "town:park/bench", package="town").name == "bench"


def test_duplicate_siblings_get_numbered_keys(town):
    assert refs.resolve(town, "lamp") is town.children[-2]     # the first keeps the plain name
    with pytest.raises(RefError, match=r"matches 2 things: 'lamp', 'lamp#2'"):
        refs.resolve(town, "lamp*")
    second = refs.resolve(town, "lamp#2")
    assert second is town.children[-1]
    assert refs.node_id(second, "town") == "town:lamp#2"
    assert len(refs.select(town, "lamp*")) == 2
    assert len(refs.select(town, "house/chair*")) == 2


def test_tag_selectors_cross_scopes(town):
    assert {n.name for n in refs.select(town, "[tag=furniture]")} == {"bed", "wardrobe", "chair", "bench"}
    assert [n.name for n in refs.select(town, "park/[tag=furniture]")] == ["bench"]
    assert [n.name for n in refs.select(town, "[type=spawn]")] == ["arrival"]
    assert len(refs.select(town, "[asset=street_lamp]")) == 2
    assert [n.name for n in refs.select(town, "house/[tag=room, name=kit*]")] == ["kitchen"]
    assert refs.select(town, "[tag=furniture.sofa]") == []
    assert refs.resolve(town, "house/[tag=furniture.bed]").name == "bed"


def test_missing_names_suggest_and_list(town):
    with pytest.raises(RefError) as err:
        refs.resolve(town, "hous")
    assert "no 'hous' in 'town'. Did you mean 'house'?" in str(err.value)
    assert "It has: door, house, lamp, lamp#2, park" in str(err.value)
    with pytest.raises(RefError, match=r"no 'sofa' in 'house' in 'town'.*It has: bed, bedroom, chair"):
        refs.resolve(town, "house/sofa")
    # Something of that name exists, but in another scope: say where.
    with pytest.raises(RefError, match=r"no 'bench' in 'town'\. There is one at 'park/bench'"):
        refs.resolve(town, "bench")
    with pytest.raises(RefError, match=r"nothing under 'town' matches \[tag=furniture.benhc\]. Did you mean 'furniture.bench'"):
        refs.select(town, "[tag=furniture.benhc]", optional=False)


def test_other_packages(town):
    other = SceneNode("hotel")
    other.add_child(SceneNode("arrival", meta={"type": "spawn"}))
    with pytest.raises(RefError, match="package 'hotel' isn't available"):
        refs.resolve(town, "hotel:arrival", package="town")
    found = refs.resolve(town, "hotel:arrival", package="town", packages={"hotel": other}.get)
    assert found is other.children[0]


def test_ids_round_trip_through_the_index(town):
    ids = refs.assign_ids(town, "town")
    assert ids["town:"] is town
    assert ids["town:house/bedroom/bed"].name == "bed"
    assert "town:house/walls" not in ids                      # plain parts have no ID
    for ident, node in ids.items():
        assert refs.resolve(town, ident, package="town") is node
    index = refs.build_index(town, "town", names=lambda n: n.name)
    stub = refs.index_tree(index)
    # The index answers the same references as the tree it was built from.
    for text in ("park/bench", "house/bed", "lamp#2", "house/kitchen/chair#2", "park/arrival"):
        assert refs.resolve(stub, text).meta["id"] == refs.resolve(town, text).meta["id"]
    assert {n.meta["id"] for n in refs.select(stub, "[tag=furniture]")} == \
        {n.meta["id"] for n in refs.select(town, "[tag=furniture]")}
    with pytest.raises(RefError, match="There is one at 'park/bench'"):
        refs.resolve(stub, "bench")


def test_surface_errors_name_near_ones():
    from geogen.layout.loader import LayoutLoader
    from pathlib import Path

    house = LayoutLoader().load(Path(__file__).parent.parent / "assets" / "house_peaked.yaml")
    assert refs.surface_of(house, "floor") is house.surfaces["floor"]
    with pytest.raises(RefError, match="has no surface 'front_wal'. Did you mean 'front_wall'"):
        refs.surface_of(house, "front_wal")
