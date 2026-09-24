"""Furniture library: metadata, and the hand-furnished hotel room keeps clear."""

from pathlib import Path

import numpy as np
import pytest

from geogen.layout import LayoutLoader, SceneComposer

FLOOR_FURNITURE = ["bed", "nightstand", "wardrobe", "desk", "desk_chair", "armchair", "floor_lamp",
                   "tv_console", "luggage_rack", "toilet", "vanity", "shower", "bathtub", "bookshelf"]
WALL_ITEMS = ["tv", "wall_mirror", "wall_art", "curtains", "towel_rail"]


@pytest.mark.parametrize("name", FLOOR_FURNITURE + WALL_ITEMS + ["table_lamp", "rug"])
def test_asset_is_tagged(name):
    root = LayoutLoader().load(f"assets/{name}.yaml")
    assert root.tags and all("." in t for t in root.tags), root.tags


@pytest.mark.parametrize("name", FLOOR_FURNITURE)
def test_floor_furniture_declares_footprint_and_clearance(name):
    root = LayoutLoader().load(f"assets/{name}.yaml")
    assert root.meta["footprint"] == pytest.approx([root.size[0], root.size[2]])
    assert root.meta["clearance"]


@pytest.mark.parametrize("name", WALL_ITEMS)
def test_wall_items_sit_in_front_of_the_wall(name):
    root = LayoutLoader().load(f"assets/{name}.yaml")
    zs = np.concatenate([
        (n.world_transform() @ np.c_[n.mesh.vertices, np.ones(len(n.mesh.vertices))].T)[2]
        for n in root.iter_nodes() if n.mesh is not None])
    assert zs.min() >= -1e-6  # nothing behind the wall plane (z = 0)


def _footprint(node):
    pts = np.concatenate([
        (n.world_transform() @ np.c_[n.mesh.vertices, np.ones(len(n.mesh.vertices))].T)[[0, 2]].T
        for n in node.iter_nodes() if n.mesh is not None])
    return pts.min(axis=0), pts.max(axis=0)


def test_hotel_room_furniture_does_not_overlap_or_block_doors():
    scene = SceneComposer().compose(Path("assets/scenes/hotel_room.yaml"))
    names = ["bed", "nightstand_south", "nightstand_north", "tv_console", "wardrobe", "desk",
             "armchair", "floor_lamp", "shower", "toilet", "vanity", "luggage_rack"]
    boxes = {n: _footprint(scene.find(n)) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (alo, ahi), (blo, bhi) = boxes[a], boxes[b]
            overlap = np.minimum(ahi, bhi) - np.maximum(alo, blo)
            assert not (overlap > 0.01).all(), f"{a} overlaps {b}"

    # No furniture inside a door's swing arc (sampled quarter discs).
    suite = scene.find("suite")
    for room in ("bedroom", "bathroom", "corridor"):
        room_node = suite.find(room)
        to_world = room_node.world_transform()
        for swing in room_node.meta.get("door_swings", []):
            hinge = to_world @ np.r_[swing["hinge"], 1.0]
            lo, hi = sorted((swing["from_deg"], swing["to_deg"]))
            angles = np.radians(np.linspace(lo, hi, 12))
            radii = np.linspace(0.1, swing["radius"], 6)
            pts = np.array([[hinge[0] + r * np.cos(t), hinge[2] + r * np.sin(t)] for r in radii for t in angles])
            for n, (blo, bhi) in boxes.items():
                inside = ((pts > blo + 0.01) & (pts < bhi - 0.01)).all(axis=1)
                assert not inside.any(), f"{n} is in the swing of {swing['door']}"
