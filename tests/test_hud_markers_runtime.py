"""The HUD's world markers (runtime/godot/scripts/hud_markers.gd): an indicator over what the
tracked quest's open objectives are about, clamped to the screen's edge when it is off screen or
behind, a ring for `marker: {area}`, nothing for `marker: hidden`, and the portal to another
scene's quest. `hud.markers` returns what is displayed."""

import json
import math
import time

import pytest

from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from test_travel import PORTAL
from test_wiring import PLATE

pytestmark = pytest.mark.xdist_group("test_hud_markers_runtime")

# Four plates on a field and a portal to the annex. `sights` marks one plate exactly, one with a
# ring, one not at all, and one only once the first is done; `walk` is a second quest to track.
FIELD = """
name: field
size: [40, 3, 40]
quests: [quests/sights.yaml, quests/walk.yaml]
slots:
  a: {position: [0, 0, -10]}
  b: {position: [12, 0, 0]}
  c: {position: [-12, 0, 0]}
  d: {position: [0, 0, 12]}
  p: {position: [-8, 0, -8]}
place:
  plate:   {asset: plate.yaml, slot: a}
  plate_2: {asset: plate.yaml, slot: b}
  plate_3: {asset: plate.yaml, slot: c}
  plate_4: {asset: plate.yaml, slot: d}
  to_annex:
    asset: portal_test.yaml
    slot: p
    travel: {scene: annex, spawn: start}
spawns:
  start: {position: [0, 0, 0], facing: south}
"""

SIGHTS = """
kind: quest
version: 1
title: The sights
start: auto
objectives:
  - {id: post, text: Stand on the far plate, done: {enter: plate/top}}
  - {id: search, text: Find the plate out east, done: {enter: plate_2/top}, marker: {area: 4}}
  - {id: secret, text: Find the one nobody mentions, optional: true, done: {enter: plate_3/top}, marker: hidden}
  - {id: back, text: Come back to the near plate, after: post, done: {enter: plate_4/top}}
"""

WALK = """
kind: quest
version: 1
title: A walk
start: auto
objectives:
  - {id: west, text: Walk to the west plate, done: {enter: plate_3/top}}
"""

# The annex: a quest of its own, a portal back to the field and one that stays in the annex.
ANNEX = """
name: annex
size: [16, 3, 16]
quests: [quests/stroll.yaml]
slots:
  a: {position: [2, 0, -2]}
  p: {position: [-4, 0, -4]}
  q: {position: [5, 0, 5]}
place:
  plate: {asset: plate.yaml, slot: a}
  to_field:
    asset: portal_test.yaml
    slot: p
    travel: {scene: field, spawn: start}
  round:
    asset: portal_test.yaml
    slot: q
    travel: {spawn: start}
spawns:
  start: {position: [0, 0, 3], facing: south}
"""

STROLL = """
kind: quest
version: 1
title: A stroll
start: auto
objectives:
  - {id: plate, text: Stand on the plate, done: {enter: plate/top}}
"""

SIZE = [1280, 720]


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    assets = tmp_path_factory.mktemp("marker_assets")
    for name, text in (("plate", PLATE), ("portal_test", PORTAL), ("field", FIELD), ("annex", ANNEX),
                       ("quests/sights", SIGHTS), ("quests/walk", WALK), ("quests/stroll", STROLL)):
        path = assets / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    out = tmp_path_factory.mktemp("generated_markers")
    for scene in ("field", "annex"):
        export_scene(SceneComposer(assets_dir=assets).compose(assets / f"{scene}.yaml"), out / f"{scene}.glb")
    return out


@pytest.fixture(scope="module")
def rt(generated, godot_runtime):
    client = godot_runtime("field", generated)
    client.call("wait_ready")
    client.call("window.size", size=SIZE)
    return client


def markers(rt, want=lambda found: True, timeout=10.0):
    """`hud.markers`, by (quest, objective), once it matches `want`."""
    end = time.time() + timeout
    while True:
        hud = rt.call("hud.markers")
        found = {(m["quest"], m["objective"]): m for m in hud["markers"]}
        if want(found) or time.time() > end:
            return found
        time.sleep(0.05)


def test_the_manifest_says_where_each_subject_is_and_how_it_is_marked(generated):
    quests = json.loads((generated / "field.manifest.json").read_text())["quests"]
    by_id = {o["id"]: o for o in quests["quests"]["sights"]["objectives"]}
    assert by_id["post"]["subjects"] == ["field:plate/top"] and "marker" not in by_id["post"]
    assert by_id["search"]["subjects"] == ["field:plate_2/top"] and by_id["search"]["marker"] == {"area": 4.0}
    assert by_id["secret"]["subjects"] == [] and by_id["secret"]["marker"] == {"hidden": True}
    x, y, z = quests["markers"]["field:plate/top"]
    assert (x, z) == (0, -10) and 0.5 < y < 1.5          # the middle of the place's volume


def test_open_objectives_of_the_tracked_quest_are_marked(rt, generated):
    where = json.loads((generated / "field.manifest.json").read_text())["quests"]["markers"]
    found = markers(rt, lambda f: len(f) == 2)
    # the hidden one and the one that waits on another have none; nor has the quest not tracked
    assert set(found) == {("sights", "post"), ("sights", "search")}
    assert rt.call("hud.quests")["tracked"] == "sights"
    post = found[("sights", "post")]
    assert post["kind"] == "point" and post["subject"] == "field:plate/top" and post["package"] == "field"
    assert post["position"] == pytest.approx(where["field:plate/top"], abs=1e-3)
    eye = rt.call("player.status")["eye"]
    assert post["distance"] == pytest.approx(math.dist(eye, post["position"]), abs=0.05)
    assert post["distance_text"] == "10 m"
    # the spawn faces it: it is on screen, near the middle, where it projects
    assert post["on_screen"] and not post["clamped"] and not post["behind"]
    assert abs(post["screen"][0] - SIZE[0] / 2) < 5 and 0 < post["screen"][1] < SIZE[1]
    hud = rt.call("hud.markers")
    assert hud["visible"] and hud["tracked"] == "sights" and hud["viewport"] == SIZE


def test_a_marker_off_screen_or_behind_is_clamped_to_the_edge(rt):
    hud = rt.call("hud.markers")
    margin = hud["edge_margin"]
    assert 4 <= margin <= 44

    rt.call("player.look", yaw=180, pitch=0)                # the plate is straight behind
    post = markers(rt)[("sights", "post")]
    assert post["behind"] and post["clamped"] and not post["on_screen"]
    assert post["screen"][1] == pytest.approx(SIZE[1] - margin, abs=0.5)        # the bottom edge: turn round
    assert margin - 0.5 <= post["screen"][0] <= SIZE[0] - margin + 0.5
    assert post["distance_text"] == "10 m"                  # still the distance to it

    rt.call("player.look", yaw=-90, pitch=0)                # facing east: the plate (south) is to the left
    post = markers(rt)[("sights", "post")]
    assert post["clamped"] and not post["on_screen"]
    assert post["screen"][0] == pytest.approx(margin, abs=0.5) and abs(post["turn_deg"]) > 90
    rt.call("player.look", yaw=90, pitch=0)                 # facing west: to the right
    post = markers(rt)[("sights", "post")]
    assert post["clamped"] and post["screen"][0] == pytest.approx(SIZE[0] - margin, abs=0.5)
    assert abs(post["turn_deg"]) < 90

    rt.call("player.look", yaw=0, pitch=80)                 # looking at the sky: below the screen
    post = markers(rt)[("sights", "post")]
    assert post["clamped"] and post["screen"][1] == pytest.approx(SIZE[1] - margin, abs=0.5)

    rt.call("player.look", at=post["position"])
    post = markers(rt)[("sights", "post")]
    assert post["on_screen"] and not post["clamped"] and "turn_deg" not in post
    assert post["screen"] == pytest.approx([SIZE[0] / 2, SIZE[1] / 2], abs=2)

    # a window of another size clamps to its own edge
    rt.call("window.size", size=[640, 400])
    rt.call("player.look", yaw=90, pitch=0)
    hud = rt.call("hud.markers")
    post = [m for m in hud["markers"] if m["objective"] == "post"][0]
    assert hud["viewport"] == [640, 400] and post["screen"][0] == pytest.approx(640 - hud["edge_margin"], abs=0.5)
    rt.call("window.size", size=SIZE)
    rt.call("player.look", yaw=0, pitch=0)


def test_an_area_marker_is_a_ring_that_does_not_give_the_spot_away(rt, generated):
    where = json.loads((generated / "field.manifest.json").read_text())["quests"]["markers"]["field:plate_2/top"]
    ring = markers(rt)[("sights", "search")]
    assert ring["kind"] == "area" and ring["radius"] == 4 and not ring["inside"]
    off = math.dist(ring["position"][::2], where[::2])
    assert 1.0 < off < 4.0                                   # the subject is in the ring, not at its middle
    eye = rt.call("player.status")["eye"]
    assert ring["distance"] == pytest.approx(math.dist(eye[::2], ring["position"][::2]) - 4, abs=0.05)

    rt.call("player.teleport", pos=[ring["position"][0], 0.1, ring["position"][2] + 1])
    ring = markers(rt)[("sights", "search")]
    assert ring["inside"] and ring["distance"] == 0
    rt.call("player.teleport", pos=[0, 0.1, 0], yaw=0)


def test_markers_follow_the_objectives_and_the_tracked_quest(rt):
    rt.call("hud.quests", linger=0)
    rt.call("player.teleport", pos=[0, 0.1, -10])           # onto the far plate
    found = markers(rt, lambda f: ("sights", "back") in f)
    assert set(found) == {("sights", "search"), ("sights", "back")}     # done: gone; the next one: marked
    assert found[("sights", "back")]["subject"] == "field:plate_4/top"
    rt.call("player.teleport", pos=[0, 0.1, 0], yaw=0)

    assert rt.call("hud.quests", track="walk")["tracked"] == "walk"
    found = markers(rt)
    assert set(found) == {("walk", "west")} and found[("walk", "west")]["position"][0] == -12

    # hidden while a screen has the keyboard, and when switched off
    rt.call("hud.log", open=True)
    assert not rt.call("hud.markers")["visible"]
    rt.call("hud.log", open=False)
    assert rt.call("hud.markers")["visible"]
    assert not rt.call("hud.markers", enabled=False)["visible"]
    assert rt.call("hud.markers", enabled=True)["visible"]
    rt.call("hud.quests", track="sights")


def test_a_quest_of_another_scene_is_marked_at_the_portal_that_leads_there(rt):
    assert all(m["kind"] != "portal" for m in markers(rt).values())      # the annex has nothing going yet
    rt.call("world.switch", scene="annex")
    rt.call("wait_ready")
    found = markers(rt, lambda f: len(f) == 2)
    # the annex's own quest, and the field's tracked one at the portal back
    assert set(found) == {("stroll", "plate"), ("sights", "search")}
    assert found[("stroll", "plate")]["kind"] == "point"
    portal = found[("sights", "search")]
    assert portal["kind"] == "portal" and portal["scene"] == "field" and portal["package"] == "field"
    assert portal["subject"] == "annex:to_field" and portal["label"] == "Field"
    x, y, z = portal["position"]
    assert (x, z) == pytest.approx((-4, -4), abs=0.2) and 0.8 < y < 1.8        # the middle of the arch
    # only one portal goes to the field: the one that stays in the annex isn't marked
    assert [m["subject"] for m in rt.call("hud.markers")["markers"] if m["kind"] == "portal"] == ["annex:to_field"]

    # the quest log picks another of the field's quests: the portal is marked for that one
    rt.call("hud.log", open=True)
    rt.call("hud.log", select="field/walk", track=True)
    rt.call("hud.log", open=False)
    found = markers(rt, lambda f: ("walk", "west") in f)
    assert found[("walk", "west")]["kind"] == "portal" and ("sights", "search") not in found

    # back in the field its own markers are exact again, and the annex's quest is at the portal there
    rt.call("world.switch", scene="field")
    rt.call("wait_ready")
    found = markers(rt, lambda f: len(f) == 2)
    assert set(found) == {("walk", "west"), ("stroll", "plate")}
    assert found[("walk", "west")]["kind"] == "point"
    there = found[("stroll", "plate")]
    assert there["kind"] == "portal" and there["scene"] == "annex" and there["subject"] == "field:to_annex"
