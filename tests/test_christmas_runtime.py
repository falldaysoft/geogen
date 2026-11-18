"""Two people decorate a tree because they both want it decorated (geogen.110): nobody is told to
fetch anything. Tom leaves the errands to Nell, both hang the ornaments, the lights go on and they
stand back to look."""

import pytest

pytestmark = [pytest.mark.godot, pytest.mark.xdist_group("test_christmas_runtime")]

ERRANDS = {"fir#0", "tree#0", "cupboard#0", "tree#1"}
HOOKS = {f"tree#{i}" for i in range(4, 16)}
LIGHTS, LOOK = "tree#16", {"tree#17", "tree#18"}


def person(rt, name):
    return next(n for n in rt.call("npcs.list") if n["name"] == name)


@pytest.mark.parametrize("scene", ["christmas_test", "christmas_eve"])
def test_two_people_decorate_the_tree(godot_runtime, scene_export, scene):
    rt = godot_runtime(scene, scene_export(scene))
    rt.call("wait_ready")
    rt.call("time.pause")
    most_carried = 0
    for _ in range(200):
        rt.call("time.advance", seconds=5, scale=4)
        nell, tom = person(rt, "Nell"), person(rt, "Tom")
        most_carried = max(most_carried, len(nell["carrying"]), len(tom["carrying"]))
        if LOOK & set(nell["used"]) and LOOK & set(tom["used"]):
            break
    hung = {who["name"]: sum(n for k, n in who["used"].items() if k in HOOKS) for who in (nell, tom)}
    assert sum(hung.values()) == 12 and min(hung.values()) >= 2, hung      # both hung some
    assert most_carried == 1
    assert ERRANDS <= set(nell["used"]) and not ERRANDS & set(tom["used"]), (nell["used"], tom["used"])
    assert LIGHTS in nell["used"] and LIGHTS not in tom["used"]
