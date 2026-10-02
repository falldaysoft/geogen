"""--inspect: the scene dump and its checks (geogen.7)."""

import json
import subprocess
import sys

import pytest

from geogen.export import node_extras
from geogen.inspection import build_tree, inspect_scene
from geogen.layout.composer import SceneComposer
from geogen.layout.qa import object_overlaps


def _garage(car_z: float):
    """A cottage shell as a 4 x 6 m garage with a car parked along its length."""
    return SceneComposer().compose_string(f"""
name: garage_test
slots:
  bay: {{position: [0, 0.32, {car_z}]}}
place:
  garage: {{asset: house_peaked.yaml, params: {{width: 4, depth: 6}}}}
  car: {{asset: vehicles/car.yaml, slot: bay}}
""")


def test_car_through_the_garage_wall_is_reported_with_its_depth():
    issues = object_overlaps(_garage(-0.75))      # the rear bumper ends 30 cm into the back wall
    body = [i for i in issues if set(i.items) == {"garage/walls", "car/body"}]
    assert body, [str(i) for i in issues]
    assert "0.30 m deep" in body[0].message


def test_parked_car_has_no_overlaps():
    assert object_overlaps(_garage(0.2)) == []


def test_openings_and_ground_are_not_overlaps(built_scene):
    # Windows' sills sit in their wall, door steps in the foundation, houses in the ground.
    for issue in object_overlaps(built_scene("street")):
        assert not any(name in issue.message for name in ("window", "door", "ground_plane", "road")), issue


def test_hotel_room_dump_is_short_and_complete(built_scene):
    text = inspect_scene(built_scene("hotel_room_auto"))
    assert len(text) < 12000
    bed = next(line for line in text.splitlines() if line.strip().startswith("bed "))
    assert "(bed.yaml)" in bed and "#furniture.bed" in bed and "box [" in bed and "affords lie" in bed
    assert "nightstand ×2" in text
    assert text.rstrip().endswith("checks: no findings")


def test_repeated_siblings_collapse(built_scene):
    tree = build_tree(built_scene("dining_set"))
    chairs = [e for e in tree.children if e.node.name.startswith("chair")]
    assert len(chairs) == 1 and chairs[0].count == 4


def test_select_and_json(built_scene):
    root = built_scene("hotel_room_auto")
    text = inspect_scene(root, pattern="bed", checks=())
    assert text.startswith("bed (bed.yaml)")
    assert "@" in text                      # attachments in the selected subtree's detail
    data = json.loads(inspect_scene(root, as_json=True, pattern="*nightstand*", checks=("layout",)))
    assert [s["name"] for s in data["selected"]] == ["nightstand", "nightstand_2"]
    assert data["findings"] == []


def test_source_is_recorded_but_not_exported(built_scene):
    bed = next(n for n in built_scene("hotel_room_auto").iter_nodes() if n.name == "bed")
    assert bed.meta["source"] == {"asset": "bed.yaml", "params": {}}
    assert "source" not in node_extras(bed).get("geogen", {})


def test_cli_inspect():
    out = subprocess.run([sys.executable, "-m", "geogen.main", "-s", "dining_set", "--inspect", "--depth", "1"],
                         capture_output=True, text=True, check=True).stdout
    assert out.startswith("dining_set")
    assert "chair" in out and "checks:" in out


def test_cli_rejects_unknown_checks():
    run = subprocess.run([sys.executable, "-m", "geogen.main", "-s", "dining_set", "--inspect", "--checks", "nope"],
                         capture_output=True, text=True)
    assert run.returncode != 0 and "unknown nope" in run.stderr


@pytest.mark.parametrize("depth, shown, hidden", [(0, None, "suite"), (1, "suite", "bedroom"),
                                                  (2, "bedroom", "nightstand")])
def test_depth_limits_the_tree(built_scene, depth, shown, hidden):
    text = inspect_scene(built_scene("hotel_room_auto"), depth=depth, checks=())
    assert "more below" in text
    assert shown is None or f" {shown} " in text
    assert f" {hidden} " not in text
