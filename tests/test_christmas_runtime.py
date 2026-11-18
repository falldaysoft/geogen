"""Two people decorate a tree because they both want it decorated (geogen.110): nobody is told to
fetch anything. Tom leaves the errands to Nell, and then both hang the ornaments."""

import pytest

pytestmark = [pytest.mark.godot, pytest.mark.xdist_group("test_christmas_runtime")]

ERRANDS = {"fir#0", "tree#0", "cupboard#0", "tree#1"}


def person(rt, name):
    return next(n for n in rt.call("npcs.list") if n["npc"].split(":")[-1].startswith(name))


@pytest.mark.parametrize("scene", ["christmas_test", "christmas_eve"])
def test_two_people_decorate_the_tree(godot_runtime, scene_export, scene):
    rt = godot_runtime(scene, scene_export(scene))
    rt.call("wait_ready")
    rt.call("time.pause")
    most_carried = 0
    for _ in range(150):
        rt.call("time.advance", seconds=5, scale=4)
        nell, tom = person(rt, "nell"), person(rt, "tom")
        most_carried = max(most_carried, len(nell["carrying"]), len(tom["carrying"]))
        if nell["needs"]["tree_decorated"] + tom["needs"]["tree_decorated"] >= 1.07:
            break
    nell, tom = person(rt, "nell"), person(rt, "tom")
    assert nell["needs"]["tree_decorated"] + tom["needs"]["tree_decorated"] == pytest.approx(1.08, abs=0.01)
    assert nell["needs"]["tree_decorated"] >= 0.18 and tom["needs"]["tree_decorated"] >= 0.18   # both hung some
    assert most_carried == 1
    assert not ERRANDS & set(tom["used"]), tom["used"]
    assert ERRANDS <= set(nell["used"]), nell["used"]
