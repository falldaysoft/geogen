"""Furnishing solver: rooms get furnished from archetypes, cleanly and repeatably."""

from pathlib import Path

import numpy as np
import pytest

from geogen.layout import LayoutLoader, SceneComposer
from geogen.layout.furnish import furnish_plan

ASSETS = Path(__file__).parent.parent / "assets"
FLOOR_ONLY = {"rug", "curtains"}  # placed on the floor but not obstacles


def _rect(node):
    pts = np.concatenate([
        (n.world_transform() @ np.c_[n.mesh.vertices, np.ones(len(n.mesh.vertices))].T)[[0, 2]].T
        for n in node.iter_nodes() if n.mesh is not None])
    return pts.min(axis=0), pts.max(axis=0)


def _furnished_suite(seed=0):
    plan = LayoutLoader().load(ASSETS / "hotel_suite.yaml")
    report = furnish_plan(plan, ASSETS, seed=seed)
    return plan, report


def _placed(room):
    return [c for c in room.children if c.meta.get("placed_by") == "furnish"]


def test_suite_is_furnished_without_unsatisfied_rules():
    plan, report = _furnished_suite()
    assert report == []
    bedroom = [c.name for c in _placed(plan.find("bedroom"))]
    for item in ("bed", "nightstand", "nightstand_2", "lamp", "lamp_2", "wardrobe", "tv_console", "tv",
                 "rug", "curtains"):
        assert item in bedroom
    bathroom = [c.name for c in _placed(plan.find("bathroom"))]
    for item in ("shower", "toilet", "vanity", "mirror"):
        assert item in bathroom


def test_furniture_does_not_overlap_or_block_doors():
    plan, _ = _furnished_suite()
    for room_name in ("bedroom", "bathroom", "corridor"):
        room = plan.find(room_name)
        floor_items = [c for c in _placed(room)
                       if abs(c.transform.translation[1]) < 1e-9 and c.name not in FLOOR_ONLY]
        boxes = {c.name: _rect(c) for c in floor_items}
        names = list(boxes)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                (alo, ahi), (blo, bhi) = boxes[a], boxes[b]
                assert not ((np.minimum(ahi, bhi) - np.maximum(alo, blo)) > 0.01).all(), f"{a} overlaps {b}"
        to_world = room.world_transform()
        for swing in room.meta.get("door_swings", []):
            hinge = to_world @ np.r_[swing["hinge"], 1.0]
            lo, hi = sorted((swing["from_deg"], swing["to_deg"]))
            pts = np.array([[hinge[0] + r * np.cos(t), hinge[2] + r * np.sin(t)]
                            for r in np.linspace(0.1, swing["radius"], 6)
                            for t in np.radians(np.linspace(lo, hi, 12))])
            for name, (blo, bhi) in boxes.items():
                assert not ((pts > blo + 0.01) & (pts < bhi - 0.01)).all(axis=1).any(), (name, swing["door"])


def test_bed_rules_hold():
    plan, _ = _furnished_suite()
    room = plan.find("bedroom")
    bed = room.find("bed")
    # Headboard against a wall with no door and no window; nightstands flank it.
    back = -np.array([np.sin(bed.transform.rotation[1]), np.cos(bed.transform.rotation[1])])
    half = room.size[[0, 2]] / 2
    head = bed.transform.translation[[0, 2]] + back * bed.size[2] / 2
    assert np.isclose(np.abs(head), half, atol=0.02).any()
    tv = room.find("tv_console")
    facing_tv = np.array([np.sin(tv.transform.rotation[1]), np.cos(tv.transform.rotation[1])])
    facing_bed = np.array([np.sin(bed.transform.rotation[1]), np.cos(bed.transform.rotation[1])])
    assert facing_tv @ facing_bed == pytest.approx(-1)  # the TV faces the bed


def test_furnishing_is_deterministic_per_seed():
    a, _ = _furnished_suite(seed=3)
    b, _ = _furnished_suite(seed=3)
    pose = lambda plan: [(n.name, tuple(np.round(n.transform.translation, 6))) for n in plan.iter_nodes()
                         if n.meta.get("placed_by")]
    assert pose(a) == pose(b)


def test_unsatisfiable_rules_are_reported(tmp_path):
    (tmp_path / "room_types").mkdir()
    (tmp_path / "room_types" / "closet.yaml").write_text(
        "name: closet\nitems:\n  bed: { asset: bed.yaml, against: wall }\n"
        "  chair: { asset: armchair.yaml, against: none, optional: true }\n")
    for name in ("bed.yaml", "armchair.yaml"):
        (tmp_path / name).write_text((ASSETS / name).read_text())
    from geogen.generators.floorplan import FloorPlan
    plan = FloorPlan.from_spec({"rooms": {"closet": {"rect": [0, 0, 1.2, 1.2], "type": "closet"}}}).build()
    report = furnish_plan(plan, tmp_path)
    assert [str(r) for r in report] == ["closet: bed: placed 0 of 1"]
    assert plan.meta["furnish_report"] == ["closet: bed: placed 0 of 1"]


def test_scene_placement_furnishes_plan():
    scene = SceneComposer().compose(ASSETS / "scenes/hotel_room_auto.yaml")
    assert scene.find("suite").find("bed") is not None
