"""The state test scene (assets/scenes/state_test/): every piece of the rules layer, walked
through in the reference machine and in the Godot runtime (geogen.3.7)."""

import pytest

from geogen.quests import collect
from geogen.rules import Machine

P = "state_test"
A, B = f"{P}:puzzle_a", f"{P}:puzzle_b"
LEVER_A, LEVER_B = f"{A}/lever#pull", f"{B}/lever#pull"
QUEST = f"actor:{P}.quest_two_levers"


@pytest.fixture(scope="module")
def found(built_scene):
    return collect(built_scene(P), P)


def test_the_state_test_scene_in_the_reference_machine(found):
    m = Machine(found.program(), {"player": {"kind": "player"}})
    m.post({"type": "start", "subject": "", "actor": "player"})
    assert m.get(f"{B}/lever#enabled") is False                          # jammed on a new game
    assert found.status(m, "two_levers", "player") == "active"

    m.post({"type": "enter", "subject": f"{P}:middle", "actor": "player"})
    m.post({"type": "enter", "subject": f"{P}:middle", "actor": "player"})
    assert m.get(f"actor:{P}.greeted", "player") is True
    assert [f for f in m.feedback if f[0] == "notify"] == [["notify", "Two levers: the left one frees the right",
                                                           "player"]]   # once
    m.feedback.clear()

    assert m.use(LEVER_B, "player") is False                             # says why
    assert m.feedback == [["notify", "The lever is jammed", "player"], ["play", "refused", "player"]]
    assert m.get(f"{P}:barrier_lamp#lit") is False                       # midday, nothing solved
    assert m.use(LEVER_A, "player") is True
    m.advance(0.7)
    assert (m.get(f"{A}#solved"), m.get(f"{B}#solved")) == (True, False)  # one copy each
    assert m.get(f"{B}/lever#enabled") is True and m.get(f"{P}:barrier_lamp#lit") is True
    assert m.get(f"actor:{P}.pulls", "player") == 1                      # the package's, shared by the copies
    assert m.use(LEVER_B, "player") is True
    m.advance(0.7)
    assert m.get(f"{P}:barrier#barrier") == "down"                       # two seconds later
    m.advance(2.0)
    assert m.get(f"{P}:barrier#barrier") == "up"
    m.set_clock(time="22:00", night=True)
    m.set(f"world:{P}.power", False)
    assert m.get(f"{P}:barrier_lamp#lit") is False                       # the bind reads both
    m.post({"type": "enter", "subject": f"{P}:beyond", "actor": "player"})
    assert found.status(m, "two_levers", "player") == "complete" and m.errors == []

    assert m.use(f"{P}:reset_lever#pull", "player")
    m.advance(0.7)                                                         # pulled: all of it, as it started
    assert (m.get(f"{A}#solved"), m.get(f"{B}#solved"), m.get(LEVER_A)) == (False, False, "up")
    assert m.get(f"{B}/lever#enabled") is False and m.get(f"{P}:barrier#barrier") == "down"
    assert found.status(m, "two_levers", "player") in ("available", "active") and m.errors == []


def test_the_state_test_scene_in_godot(scene_export, godot_runtime):
    rt = godot_runtime(P, scene_export(P))
    rt.call("wait_ready")
    rt.call("interactions.use", asset="lever_2", actor="player", wait=True)
    assert rt.call("state.get", key=f"{B}#solved")["value"] is False     # jammed, and it says so
    assert "The lever is jammed" in toasts(rt)
    rt.call("interactions.use", asset="lever", actor="player", wait=True)
    rt.call("time.advance", seconds=0.5)
    rt.call("interactions.use", asset="lever_2", actor="player", wait=True)
    rt.call("time.advance", seconds=7, scale=4)
    assert rt.call("state.get", key=f"{P}:barrier#barrier")["value"] == "up"
    assert "The barrier lifts" in toasts(rt)


def toasts(rt):
    shown = rt.call("hud.notifications")
    return [t["text"] for t in shown["showing"] + shown["queued"]]
