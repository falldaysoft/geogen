"""HUD markers, follow-ups (geogen.37): the way to a quest two scenes off, counted objectives
that drop the subjects already done, and markers that fade with distance and dim when hidden."""

import time

import pytest

from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from test_state import LEVER
from test_travel import PORTAL
from test_wiring import PLATE

pytestmark = pytest.mark.xdist_group("test_hud_markers_routes_runtime")

WALL = """
name: wall
origin: bottom_center
size: [6, 3, 0.3]
parts:
  slab: {primitive: cube, size: [1, 1, 1], anchor: bottom_center, bevel: 0}
"""
START = """
name: start
size: [12, 3, 12]
slots:
  p: {position: [0, 0, -4]}
place:
  to_middle: {asset: portal_test.yaml, slot: p, travel: {scene: middle, spawn: start}}
spawns:
  start: {position: [0, 0, 2], facing: south}
"""
MIDDLE = """
name: middle
size: [12, 3, 12]
slots:
  p: {position: [-4, 0, -4]}
  q: {position: [4, 0, -4]}
place:
  to_far: {asset: portal_test.yaml, slot: p, travel: {scene: far, spawn: start}}
  to_start: {asset: portal_test.yaml, slot: q, travel: {scene: start, spawn: start}}
spawns:
  start: {position: [0, 0, 2], facing: south}
"""
FAR = """
name: far
size: [240, 3, 240]
quests: [quests/levers.yaml, quests/look.yaml]
slots:
  l1: {position: [3, 0, 3]}
  l2: {position: [5, 0, 3]}
  l3: {position: [7, 0, 3]}
  wall: {position: [0, 0, -6]}
  behind: {position: [0, 0, -9]}
  away: {position: [60, 0, -80]}                # well off to the side: nothing in the way
  p: {position: [-8, 0, 4]}
place:
  lever_1: {asset: lever.yaml, slot: l1, tags: [lever]}
  lever_2: {asset: lever.yaml, slot: l2, tags: [lever]}
  lever_3: {asset: lever.yaml, slot: l3, tags: [lever]}
  wall: {asset: wall.yaml, slot: wall}
  near_plate: {asset: plate.yaml, slot: behind}
  far_plate: {asset: plate.yaml, slot: away}
  to_middle: {asset: portal_test.yaml, slot: p, travel: {scene: middle, spawn: start}}
spawns:
  start: {position: [0, 0, 0], facing: south}
"""
LEVERS = """
kind: quest
version: 1
title: Three levers
start: auto
objectives:
  - {id: all, text: Pull every lever, done: 'all("[tag=lever]", it.pull == "down")'}
"""
LOOK = """
kind: quest
version: 1
title: A look round
start: auto
objectives:
  - {id: behind, text: Look behind the wall, done: {enter: near_plate/top}}
  - {id: away, text: Walk out to the far plate, done: {enter: far_plate/top}}
"""


@pytest.fixture(scope="module")
def rt(tmp_path_factory, godot_runtime):
    assets = tmp_path_factory.mktemp("routes_assets")
    for name, text in (("plate", PLATE), ("portal_test", PORTAL), ("lever", LEVER), ("wall", WALL),
                       ("start", START), ("middle", MIDDLE), ("far", FAR),
                       ("quests/levers", LEVERS), ("quests/look", LOOK)):
        path = assets / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    out = tmp_path_factory.mktemp("generated_routes")
    for scene in ("start", "middle", "far"):
        export_scene(SceneComposer(assets_dir=assets).compose(assets / f"{scene}.yaml"), out / f"{scene}.glb")
    client = godot_runtime("far", out)
    client.call("wait_ready")
    client.call("window.size", size=[1280, 720])
    return client


def markers(rt, want=lambda found: True, timeout=10.0):
    end = time.time() + timeout
    while True:
        found = rt.call("hud.markers")["markers"]
        if want(found) or time.time() > end:
            return found
        time.sleep(0.05)


def test_a_counted_objective_marks_only_what_is_left(rt):
    assert rt.call("hud.quests", track="levers")["tracked"] == "levers"
    assert sorted(m["subject"] for m in markers(rt, lambda f: len(f) == 3)) == ["far:lever_1", "far:lever_2", "far:lever_3"]
    rt.call("state.set", key="far:lever_2#pull", value="down")
    assert sorted(m["subject"] for m in markers(rt, lambda f: len(f) == 2)) == ["far:lever_1", "far:lever_3"]
    (quest,) = [q for q in rt.call("quests.list") if q["id"] == "levers"]
    assert quest["objectives"][0]["subjects_done"] == ["far:lever_2"]


def test_markers_fade_with_distance_and_dim_when_hidden(rt):
    rt.call("hud.quests", track="look")
    found = {m["subject"]: m for m in markers(rt, lambda f: any(m.get("occluded") for m in f))}
    behind, away = found["far:near_plate/top"], found["far:far_plate/top"]
    assert behind["occluded"] and behind["alpha"] == pytest.approx(0.5)          # the wall is in the way
    assert not away["occluded"] and 0.45 <= away["alpha"] < 0.6                  # 100 m off: faded


def test_a_quest_two_scenes_off_is_marked_at_the_first_step(rt):
    rt.call("world.switch", scene="start")
    rt.call("wait_ready")
    (portal,) = markers(rt, lambda f: f)
    assert (portal["kind"], portal["subject"], portal["quest"]) == ("portal", "start:to_middle", "look")
    assert portal["route"] == ["middle", "far"] and portal["scene"] == "middle"
    rt.call("world.switch", scene="middle")                     # one step on: straight there
    rt.call("wait_ready")
    (portal,) = markers(rt, lambda f: f)
    assert (portal["subject"], portal["route"]) == ("middle:to_far", ["far"])


def test_a_streamed_scene_marks_subjects_in_chunks_that_arent_loaded(tmp_path, godot_runtime):
    """Markers come from the chunk index's positions: the far end of a streamed street is marked
    while the blocks between are only LOD stand-ins."""
    from geogen import roots
    from geogen.chunks import export_chunks
    from test_godot_runtime import STRIP

    project = tmp_path / "project"
    (project / "assets" / "quests").mkdir(parents=True)
    (project / "geogen.yaml").write_text("kind: project\nname: streamed\n")
    (project / "assets" / "quests" / "far_end.yaml").write_text(
        "kind: quest\nversion: 1\ntitle: To the far end\nstart: auto\n"
        "objectives:\n  - {id: there, text: Walk to the far end of the street, done: {enter: far_end}}\n")
    scene = STRIP.replace("name: strip", "name: strip\nquests: [quests/far_end.yaml]\n"
                          "places:\n  far_end: {box: [4, 2.4, 4], position: [60, 0.1, -20]}")
    roots.use(project)
    try:
        root = SceneComposer().compose_string(scene)
    finally:
        roots.use()
    out = tmp_path / "generated"
    export_chunks(root, out / "strip_chunks", name="strip")
    rt = godot_runtime("strip", out, "--stream-radius=20,1000,6")
    rt.call("wait_ready")
    (marker,) = markers(rt, lambda f: f)
    assert (marker["kind"], marker["subject"]) == ("point", "strip:far_end")
    assert marker["position"] == pytest.approx([60, 1.3, -20], abs=0.05) and marker["distance"] > 120
