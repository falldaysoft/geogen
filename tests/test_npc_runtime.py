"""The cottage resident in the Godot runtime (headless, deterministic, sped up).

The NPC is driven entirely by data (assets/npcs/, affordances, the door's
portal); these tests assert on its ``npc summary`` after a simulated span.
"""

import json

import pytest

from geogen.export import export_scene
from geogen.main import _build_registry

FAST = ("--fixed-fps", "60")   # unthrottled: frames run back to back


@pytest.fixture(scope="module")
def cottage_dir(scene_export):
    return scene_export("cottage")


def simulate(run_godot, generated, seconds, *extra, scene="cottage") -> dict:
    out = run_godot("--scene", scene, f"--generated={generated}", "--timescale=8",
                    f"--simulate={seconds}", *extra, engine_args=FAST)
    line = next(l for l in out.splitlines() if l.startswith("npc summary: "))
    (report,) = json.loads(line.removeprefix("npc summary: "))
    return report


def test_resident_idles_before_deciding(run_godot, cottage_dir):
    # Straight after load the resident stands where the scene placed it.
    report = simulate(run_godot, cottage_dir, 0.02)
    assert report["npc"] == "resident_npc"
    assert report["pose"] == "stand"
    assert report["position"][1] == pytest.approx(0.32, abs=0.03)   # on the floorboards


def test_humanoid_resident_plays_clips(run_godot, cottage_dir):
    # The resident is a skinned humanoid: she walks to her first window (walk clip), idles
    # there, sits down in the armchair ~14 s in (the sit_down transition clip, then pose_sit on
    # her own AnimationPlayer) and later stands up (stand_up) and idles again.
    assert simulate(run_godot, cottage_dir, 0.9)["clip"] == "walk"
    assert simulate(run_godot, cottage_dir, 4)["clip"] == "idle"
    lowering = simulate(run_godot, cottage_dir, 15)
    assert (lowering["pose"], lowering["clip"], lowering["transitions"]) == ("sit", "sit_down", 1)
    sitting = simulate(run_godot, cottage_dir, 20)
    assert (sitting["pose"], sitting["clip"]) == ("sit", "pose_sit")
    later = simulate(run_godot, cottage_dir, 100)
    assert (later["pose"], later["clip"], later["transitions"]) == ("stand", "idle", 2)
    assert later["cost_us"] > 0                           # script time per physics frame


def test_resident_sits_facing_the_seat_in_a_turned_cottage(run_godot, cottage_dir, scene_export):
    # house_plot turns the cottage to face east, so the NPC's parent is rotated: poses and
    # turns must use world headings (she used to sit sideways here, backwards on the island).
    sitting = simulate(run_godot, scene_export("house_plot"), 20, scene="house_plot")
    assert (sitting["pose"], sitting["doing"]) == ("sit", "armchair#0: wait duration")
    assert sitting["pose_yaw_error"] == pytest.approx(0, abs=1)
    plain = simulate(run_godot, cottage_dir, 20)
    assert sitting["yaw"] == pytest.approx(plain["yaw"] + 90, abs=1)


def test_resident_lives_in_the_cottage(run_godot, cottage_dir):
    report = simulate(run_godot, cottage_dir, 600)
    used = report["used"]
    kinds = {key.split("#")[0].split("_")[0] for key in used}
    # Sits (armchair / dining chairs), looks out of windows, steps out of the door.
    assert {"armchair", "chairs", "window", "door"} <= kinds, used
    assert report["distinct"] >= 6
    assert report["passes"] >= 2          # out through the front door and back in
    assert report["failures"] == []
    assert report["max_stall"] < 2.0      # never pinned against geometry
    assert report["outside"] == 0.0       # stayed home (the doorstep is within the margin)


def test_resident_walks_around_the_player(run_godot, cottage_dir):
    # The player stands between the table and the armchair, on the armchair's
    # approach, for the whole run: paths bend around them, and only the
    # armchair (whose approach they block) is given up on.
    report = simulate(run_godot, cottage_dir, 300, "--spawn=1.5,0.32,1.0")
    assert report["distinct"] >= 6
    assert report["max_stall"] < 2.0
    assert all("destination occupied" in f for f in report["failures"]), report["failures"]


def test_resident_greets_and_carries_on(run_godot, cottage_dir):
    # The player, behind her at the window, presses E on her: she turns, waves, and goes back to
    # her day; she keeps an eye on the player while they're close and ahead of her.
    out = run_godot("--scene", "cottage", f"--generated={cottage_dir}", "--spawn=1.5,0.32,0.03",
                    "--yaw=-90", "--greet=resident@3", "--simulate=12", engine_args=FAST)
    assert 'greeted: {"npc":"resident_npc","ok":true}' in out
    line = next(l for l in out.splitlines() if l.startswith("npc summary: "))
    (report,) = json.loads(line.removeprefix("npc summary: "))
    assert report["greets"] == 1
    assert report["looking"] > 1.0
    assert report["failures"] == []
    assert report["decisions"] >= 1 and report["clip"] != "wave"


def test_resident_steps_aside_for_the_player(run_godot, cottage_dir):
    # The player walks straight at her while she looks out of the window: she steps aside.
    out = run_godot("--scene", "cottage", f"--generated={cottage_dir}", "--spawn=1.0,0.32,0.03",
                    "--yaw=-90", "--wait=3", "--walk=2.5", "--npc-trace", engine_args=FAST)
    assert '"event":"yield"' in out
    steps = [json.loads(l.removeprefix("npc: ")) for l in out.splitlines() if l.startswith("npc: ")]
    aside = [s for s in steps if s.get("action") == "step_aside" and s.get("step") == {"face": "player"}]
    assert aside and abs(aside[0]["pos"][2] - 0.03) > 0.2          # moved off the player's line


# --- spiral stairs (scenes/lighthouse_keepers.yaml) --------------------------------------------

def keepers(run_godot, generated, seconds) -> dict:
    out = run_godot("--scene", "lighthouse_keepers", f"--generated={generated}", "--timescale=8",
                    f"--simulate={seconds}", engine_args=FAST)
    line = next(l for l in out.splitlines() if l.startswith("npc summary: "))
    return {r["npc"].removesuffix("_npc"): r for r in json.loads(line.removeprefix("npc summary: "))}


def test_npcs_take_a_spiral_stair_by_its_walking_line(run_godot, scene_export):
    # The keeper starts on the ground and lives on the lighthouse gallery, 10 m up; the lookout
    # starts up there and lives outside the door. The navmesh path between has the stair's
    # navigation link in it: two points, a straight line through the treads. Walking that left
    # each stuck against the stair; they follow the link's walking line round instead.
    generated = scene_export("lighthouse_keepers")
    axis = lambda r: (r["position"][0] ** 2 + r["position"][2] ** 2) ** 0.5

    # 14 s in the lookout is half way down, on the treads (0.09 .. 1.19 m from the axis). The
    # stair is one body wide, so the keeper waits at its foot rather than meet them on it.
    early = keepers(run_godot, generated, 14)
    assert 2.0 < early["lookout"]["position"][1] < 8.0 and 0.3 < axis(early["lookout"]) < 1.0, early["lookout"]
    assert early["keeper"]["position"][1] < 0.3 and axis(early["keeper"]) > 1.3, early["keeper"]
    assert early["keeper"]["doing"] == "self/wander: go_to random"

    # A minute in they have changed places, and neither was ever stuck.
    late = keepers(run_godot, generated, 60)
    assert late["keeper"]["position"][1] == pytest.approx(10.0, abs=0.1), late["keeper"]
    assert 1.2 < axis(late["keeper"]) < 3.4                               # on the deck, off the stair well
    assert late["lookout"]["position"][1] == pytest.approx(0.0, abs=0.1) and late["lookout"]["position"][2] > 3.0
    for report in late.values():
        assert report["failures"] == [] and report["unstuck"] == 0, report
        assert report["max_stall"] < 2.0, report
        assert report["used"].get("self/wander", 0) >= 1
