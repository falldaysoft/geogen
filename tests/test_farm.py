"""Farming (geogen.70): plots planted by a dropped packet of seeds, growing on the game clock,
harvested into the inventory."""

import pytest

from geogen.layout.composer import SceneComposer
from geogen.quests import collect
from geogen.rules import Machine

PLOT = "plot_test:plot"


@pytest.fixture
def plot_machine():
    root = SceneComposer(assets_dir="assets").compose("assets/scenes/plot_test.yaml")
    program = collect(root, "plot_test").program()
    assert program["interactions"][f"{PLOT}#crop"]["states"]["wheat_planted"]["after"] == {"minutes": 360.0}
    machine = Machine(program, {"player": {"kind": "player"}})
    machine.set_clock(time="08:00", day=0)
    return machine


def drop(machine, item):
    machine.post({"type": "signal", "subject": f"{PLOT}/bed", "name": "received", "actor": "player",
                  "data": {"item": item}})


def test_a_packet_plants_the_plot_it_grows_and_is_harvested(plot_machine):
    m = plot_machine
    drop(m, "carrot_seeds")
    assert m.get(f"{PLOT}#crop") == "carrot_planted"
    drop(m, "wheat_seeds")                                   # already growing: the packet is lost
    assert m.get(f"{PLOT}#crop") == "carrot_planted"
    m.set_clock(time="14:00")
    assert m.get(f"{PLOT}#crop") == "carrot_sprouting"
    m.set_clock(time="20:00")
    assert m.get(f"{PLOT}#crop") == "carrot_grown"
    m.advance(1)
    assert m.use(f"{PLOT}#crop", "player")
    m.advance(1)                                             # harvested, then empty
    assert m.get(f"{PLOT}#crop") == "empty"
    assert m.get("actor:plot_test.item_carrot", "player") == 6
    assert ["notify", "Harvested: Carrots", "player"] in m.feedback


def test_it_grows_while_nobody_watches(plot_machine):
    m = plot_machine
    drop(m, "pumpkin_seeds")
    m.set_clock(time="08:00", day=2)                          # two days on, in one jump
    assert m.get(f"{PLOT}#crop") == "pumpkin_grown"
    assert m.use(f"{PLOT}#crop", "player")


def test_in_godot_a_bought_packet_dropped_on_the_plot_grows_and_is_harvested(godot_runtime, scene_export):
    rt = godot_runtime("plot_test", scene_export("plot_test"))
    rt.call("interactions.use", asset="cabinet", interaction="a", actor="player")     # wheat seeds, free here
    rt.call("time.advance", seconds=3, scale=1)
    (packet,) = rt.call("flows.list")["items"]
    x, y, z = packet["position"]
    rt.call("player.teleport", pos=[x - 0.75, 0.1, z], yaw=-90)
    rt.call("time.advance", seconds=0.3, scale=1)
    rt.call("player.look", at=[x, y, z])
    rt.call("time.step", frames=3)
    rt.call("time.resume")
    assert rt.call("player.use")["kind"] == "carry"
    rt.call("player.walk_to", pos=[0.0, 0.1, 1.3])                                    # the plot's near edge
    rt.call("player.look", at=[0.0, 0.3, 2.6])
    rt.call("time.advance", seconds=2, scale=1)
    plot = lambda: rt.call("interactions.list", asset="plot")[0]["state"]            # noqa: E731
    assert plot() == "empty"                                                          # held over it: not yet
    rt.call("player.drop")
    rt.call("time.advance", seconds=2, scale=1)
    assert plot() == "wheat_planted" and rt.call("flows.list")["items"] == []         # the soil took it in
    rt.call("time.skip", hours=6)
    rt.call("time.advance", seconds=0.5, scale=1)
    assert plot() == "wheat_sprouting"
    rt.call("time.skip", until="08:00")                                               # overnight: grown
    rt.call("time.advance", seconds=0.5, scale=1)
    assert plot() == "wheat_grown"
    rt.call("interactions.use", asset="plot", interaction="crop", actor="player")
    rt.call("time.advance", seconds=1, scale=1)
    assert plot() == "empty"
    carried = {i["id"]: i["count"] for i in rt.call("inventory.list")["items"]}
    assert carried == {"wheat": 4}


STALL = """
name: stall
size: [4, 3, 4]
state:
  gold: {scope: actor, type: number, default: 1}
place:
  counter: {asset: produce_counter.yaml}
"""


def test_the_counter_buys_crops_for_gold(tmp_path):
    path = tmp_path / "stall.yaml"
    path.write_text(STALL)
    found = collect(SceneComposer(assets_dir="assets").compose(path), "stall")
    m = Machine(found.program(), {"player": {"kind": "player"}})
    m.set("actor:stall.item_carrot", 2, "player")
    sell = lambda crop: m.use(f"stall:counter#sell_{crop}", "player")         # noqa: E731
    gold = lambda: m.get("actor:stall.gold", "player")                         # noqa: E731
    for _ in range(2):
        assert sell("carrot")
        m.advance(1)
    assert gold() == 1 + 2 * 2 and m.get("actor:stall.item_carrot", "player") == 0
    assert ["notify", "Sold: Carrots (+2 gold)", "player"] in m.feedback
    assert not sell("carrot") and ["notify", "No carrots to sell", "player"] in m.feedback
    assert not sell("pumpkin")
