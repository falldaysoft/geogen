"""A life (geogen.111): what stays changed about a person. Walter's seed tin is out of his reach;
the same seed run twice, the only difference is whether someone lifts it down."""

from pathlib import Path

import pytest

from geogen import npc as npc_module

EXAMPLE = Path(__file__).parent.parent / "examples" / "dollhouse"
SCENE = "walters_week"

pytestmark = pytest.mark.xdist_group("test_life")


def _person(rt, who):
    return next(n for n in rt.call("npcs.list") if n["npc"].removesuffix("_npc") == who)


def _launch(godot_runtime, generated):
    rt = godot_runtime(SCENE, generated)
    rt.call("wait_ready")
    rt.call("time.pause")
    rt.call("time.clock", time="09:00")
    return rt


def _week(rt, visits=4, seconds=300):
    """Each visit: some hours with them, then away for two days (the letterbox's skip)."""
    log = []
    for visit in range(visits):
        for _ in range(seconds // 30):
            rt.call("time.advance", seconds=30, scale=8)
        walter, ada = _person(rt, "walter"), _person(rt, "ada")
        log.append((visit, walter["life"]["chapter"], walter["life"]["practice"], ada["life"]["chapter"],
                    walter["relations"], walter["position"], ada["position"], walter["doing"], ada["doing"]))
        print(log[-1], flush=True)
        rt.call("time.skip", days=1)
        rt.call("time.skip", until="09:00")     # what the letterbox does
    return log


def test_a_life_is_checked_when_it_is_loaded():
    needs = {"social": npc_module._need({}, {}, "test")}
    life = npc_module._life({"practice": {"craft": {"from": ["work"]}}, "chapters": [
        {"name": "start", "until": {"practice": {"craft": 0.5}}},
        {"name": "after", "becomes": {"needs": {"social": {"decay": 0.01}}, "home": {"rects": [[0, 0, 1, 1]]}}},
    ]}, needs, "test")["life"]
    assert life["practice"]["craft"]["per_use"] == 0.05
    assert life["chapters"][1]["becomes"]["needs"]["social"] == {"decay": 0.01}     # only what it changes
    for bad in ({"chapters": []},
                {"chapters": [{"name": "a", "until": {"practice": {"unknown": 1}}}]},
                {"chapters": [{"name": "a", "becomes": {"needs": {"hunger": {"decay": 1}}}}]},
                {"chapters": [{"name": "a", "becomes": {"home": {"rects": [[0, 1]]}}}]},
                {"chapters": [{"story": "no name"}]}):
        with pytest.raises(ValueError):
            npc_module._life(bad, needs, "test")


@pytest.mark.godot
def test_lifting_the_tin_down_changes_his_life(godot_runtime, project_export):
    generated = project_export(EXAMPLE, SCENE)
    rt = _launch(godot_runtime, generated)
    rt.call("player.carry", prop="*tin*")
    rt.call("player.teleport", pos=[-14.6, 0.05, -6.2])
    rt.call("time.advance", seconds=0.5, scale=1)       # (it is carried along with the player)
    rt.call("player.drop")
    _week(rt)
    walter, ada = _person(rt, "walter"), _person(rt, "ada")
    assert walter["life"]["chapter"] == "a_neighbour" and ada["life"]["chapter"] == "over_the_fence"
    assert [line["chapter"] for line in walter["life"]["story"]] == ["first_shoots", "the_garden_again", "a_neighbour"]
    assert any("sow" in option for option in walter["used"]) and any("admire" in option for option in ada["used"])
    assert walter["relations"]["walters_week:ada"] >= 0.34
    # Neighbours, not housemates: neither has used anything in the other's house.
    assert not any(option.startswith(("fridge_2", "range_2")) for option in walter["used"])
    assert not any(option.startswith(("fridge#", "range#")) for option in ada["used"])
    # Who he has become is in the save, and comes back with it.
    saved = rt.call("state.save")
    assert saved["npcs"][walter["uid"]]["life"]["chapter"] == 3
    rt.call("state.load", state=saved)
    rt.call("time.advance", seconds=1, scale=1)
    again = _person(rt, "walter")
    assert again["life"]["chapter"] == "a_neighbour" and len(again["life"]["story"]) == 3


@pytest.mark.godot
def test_left_alone_he_stays_as_he_was(godot_runtime, project_export):
    generated = project_export(EXAMPLE, SCENE)
    rt = _launch(godot_runtime, generated)
    _week(rt)
    walter = _person(rt, "walter")
    assert walter["life"]["chapter"] == "keeping_to_himself" and walter["life"]["story"] == []
    assert not any("sow" in option for option in walter["used"])
