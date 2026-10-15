"""A content package's quest in the HUD (geogen.34): the tracker and toasts know it by its own
package, its markers have positions in the scenes it runs in, a portal leads to the scene of its
next step, and a scene without a program has no package."""

import time

import pytest

from test_content import CAMP, deploy, write_assets
from test_travel import PORTAL

pytestmark = pytest.mark.xdist_group("test_content_hud_runtime")

# The camp of test_content, with a portal to the annex.
CAMP_PORTAL = CAMP.replace("place:\n", "  p: {position: [-3, 0, -2]}\nplace:\n  to_annex:\n"
                           "    asset: portal_test.yaml\n    slot: p\n    travel: {scene: annex, spawn: start}\n")


@pytest.fixture(scope="module")
def rt(tmp_path_factory, godot_runtime):
    assets = write_assets(tmp_path_factory.mktemp("content_hud_assets"))
    (assets / "camp.yaml").write_text(CAMP_PORTAL)
    (assets / "portal_test.yaml").write_text(PORTAL)
    out = tmp_path_factory.mktemp("generated_content_hud")
    deploy(assets, out)
    client = godot_runtime("camp", out)
    client.call("wait_ready")
    return client


def eventually(call, want, timeout=10.0):
    end = time.time() + timeout
    while True:
        got = call()
        if want(got) or time.time() > end:
            return got
        time.sleep(0.05)


def test_the_tracker_and_toasts_know_the_quest_by_its_package(rt):
    shown = eventually(lambda: rt.call("hud.quests"), lambda q: q["tracked"] == "errand")
    assert (shown["tracked"], shown["package"]) == ("errand", "errand")
    toasts = rt.call("hud.notifications")
    started = [t for t in toasts["showing"] + toasts["queued"] if t["kind"] == "started"]
    assert [(t["quest"], t["package"]) for t in started] == [("errand", "errand")]
    log = rt.call("hud.log")
    assert [(q["key"], q["tracked"]) for q in log["quests"]] == [("errand/errand", True)]


def test_its_subjects_are_marked_where_they_are(rt):
    hud = eventually(lambda: rt.call("hud.markers"), lambda h: h["markers"])
    (marker,) = hud["markers"]
    assert (marker["kind"], marker["subject"]) == ("point", "camp:lever")
    lever = rt.call("world.node", uid="camp:lever")
    assert all(abs(a - b) < 0.6 for a, b in zip(marker["position"], lever["position"]))


def test_its_next_step_in_another_scene_is_marked_at_the_portal(rt):
    assert rt.call("player.use", asset="lever")["used"][0]["ok"]
    hud = eventually(lambda: rt.call("hud.markers"), lambda h: any(m["kind"] == "portal" for m in h["markers"]))
    (portal,) = [m for m in hud["markers"] if m["kind"] == "portal"]
    assert (portal["objective"], portal["scene"], portal["package"]) == ("plate", "annex", "errand")
    assert portal["subject"] == "camp:to_annex"


def test_the_pick_follows_the_quest_into_the_next_scene(rt):
    rt.call("world.switch", scene="annex")
    rt.call("wait_ready")
    shown = eventually(lambda: rt.call("hud.quests"), lambda q: q["tracked"] == "errand")
    assert shown["package"] == "errand"
    hud = eventually(lambda: rt.call("hud.markers"), lambda h: h["markers"])
    assert [(m["kind"], m["subject"]) for m in hud["markers"]] == [("point", "annex:plate/top")]
