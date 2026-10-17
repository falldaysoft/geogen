"""The vending machine (vending_machine.yaml): its buttons wired to the chutes' `emit`, and in
Godot (the quest yard's machine) every press giving another product: one stays in the tray, many
pile up, and the player carries one off, puts it down and picks it up off the ground."""

import pytest

from geogen.quests import collect
from geogen.rules import Machine
from geogen.runtime_client import step

pytestmark = pytest.mark.xdist_group("test_quest_yard")

PKG = "quest_yard"
CHUTES = {"a": "cola", "b": "lemonade", "c": "crisps", "d": "chocolate"}


def test_each_button_queues_its_own_product(built_scene):
    machine = Machine(collect(built_scene("quest_yard"), PKG).program(), {"player": {"kind": "player"}})
    for _ in range(3):
        machine.post({"type": "signal", "subject": f"{PKG}:vending/cabinet", "name": "c", "actor": "player"})
    assert {name: machine.get(f"{PKG}:vending/{name}#queued") for name in CHUTES.values()} == \
        {"cola": 0, "lemonade": 0, "crisps": 3, "chocolate": 0}


def press(button, n=1, gap=0.25) -> list[dict]:
    return [step(name, **args) for _ in range(n) for name, args in
            (("interactions.use", {"asset": "cabinet", "interaction": button, "actor": "player"}),
             ("wait", {"seconds": gap}))]


def yard(godot_script, scene_export, steps):
    return godot_script([step("wait_ready"), *steps], "--scene=quest_yard", f"--generated={scene_export('quest_yard')}")


def bodies(run, key="world.bodies") -> list[dict]:
    out = run[key]
    return out if isinstance(out, list) else out["bodies"]


# The machine stands at x -3.6, z 9.6 facing -Z: its tray floor is 0.41 up (on the yard's 0.1
# paving), from the cabinet's front (z 9.2) back under the hood, and out onto the ledge to z 9.02.

def test_one_product_stays_in_the_tray(godot_script, scene_export):
    for button in "abcd":
        run = yard(godot_script, scene_export, [*press(button), step("wait", seconds=3), step("flows.list")])
        [item] = run["flows.list"]["items"]
        assert item["position"][1] > 0.38 and 9.02 < item["position"][2] < 9.66, item


def test_presses_keep_coming_and_pile_up(godot_script, scene_export):
    # Twenty presses in quick succession: none waits, and all are in the tray or in front of it.
    run = yard(godot_script, scene_export,
               [*[s for i in range(20) for s in press("abcd"[i % 4])], step("wait", seconds=3), step("flows.list")])
    flows = run["flows.list"]
    machine = {s["uid"].split("/")[-1]: s for s in flows["sources"] if s["uid"].startswith(f"{PKG}:vending/")}
    assert {k: (s["emitted"], s["queued"]) for k, s in machine.items()} == {k: (5, 0) for k in CHUTES.values()}
    assert all(8.0 < it["position"][2] < 9.66 for it in flows["items"])


def test_a_product_is_carried_off_put_down_and_picked_up_off_the_ground(godot_script, scene_export):
    # Runs repeat the same steps, so the same pile (physics runs in step): each finds where
    # the last left things and goes one step further.
    made = [*press("a", 4, 0.3), step("wait", seconds=3)]
    tray = bodies(yard(godot_script, scene_export, [*made, step("world.bodies")]))
    x, y, z = min(tray, key=lambda b: b["center"][2])["center"]          # the frontmost
    carry = [*made, step("player.teleport", pos=[x, 0.1, z - 0.75], yaw=0), step("wait", seconds=0.3),
             step("player.look", at=[x, y, z]), step("wait", frames=3), step("player.use"),
             step("player.walk_to", pos=[0, 0.1, 5]), step("wait", seconds=5), step("player.drop"),
             step("wait", seconds=2), step("world.bodies")]
    run = yard(godot_script, scene_export, carry)
    held = run["player.use"]
    assert held["kind"] == "carry" and run["player.drop"]["dropped"] == held["prop"]
    put = next(b for b in bodies(run) if b["uid"] == held["prop"])
    assert abs(put["position"][0]) < 1.0 and 4.0 < put["position"][2] < 6.0 and put["center"][1] < 0.2
    # On the paving, 0.9 m ahead and below the eyes: further than `reach` (1.5 m) from them,
    # but within bending down for it.
    x, y, z = put["center"]
    run = yard(godot_script, scene_export, [*carry, step("player.teleport", pos=[x, 0.1, z + 0.9], yaw=180),
                                            step("wait", seconds=0.3), step("player.look", at=[x, y, z]),
                                            step("wait", frames=3), step("player.use")])
    assert run.all("player.use")[-1] == {"kind": "carry", "prompt": "Pick up", "prop": held["prop"]}
