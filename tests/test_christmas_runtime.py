"""Two people decorate a tree because they both want it decorated (geogen.110): nobody is told to
fetch anything. Tom leaves the errands to Nell, both hang the ornaments, the lights go on and they
stand back to look."""

import pytest

pytestmark = [pytest.mark.godot, pytest.mark.xdist_group("test_christmas_runtime")]

ERRANDS = {"fir#0", "tree#0", "cupboard#0", "tree#1"}
LIGHTS, LOOK = "tree#11", {"tree#12", "tree#13"}


def hung_by(who):
    return sum(n for option, n in who["used"].items() if ".hang_" in option)


def taken_by(who):
    return sum(n for option, n in who["used"].items() if option in ("tree#2", "tree#3"))


def person(rt, name):
    return next(n for n in rt.call("npcs.list") if n["name"] == name)


@pytest.mark.parametrize("scene", ["christmas_test", "christmas_eve"])
def test_two_people_decorate_the_tree(godot_runtime, scene_export, scene):
    rt = godot_runtime(scene, scene_export(scene))
    rt.call("wait_ready")
    rt.call("time.pause")
    most_carried = 0
    for _ in range(300):
        rt.call("time.advance", seconds=5, scale=4)
        nell, tom = person(rt, "Nell"), person(rt, "Tom")
        most_carried = max(most_carried, len(nell["carrying"]), len(tom["carrying"]))
        if LOOK & set(nell["used"]) and LOOK & set(tom["used"]):
            break
    hung = {who["name"]: hung_by(who) for who in (nell, tom)}
    assert sum(hung.values()) == 20 and min(hung.values()) >= 4, hung      # all twenty on, by both of them
    assert taken_by(nell) + taken_by(tom) == 20                            # and the box is empty
    assert most_carried == 1
    assert ERRANDS <= set(nell["used"]) and not ERRANDS & set(tom["used"]), (nell["used"], tom["used"])
    assert LIGHTS in nell["used"] and LIGHTS not in tom["used"]
