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
def cottage_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("generated_npc")
    export_scene(_build_registry()["cottage"](), out / "cottage.glb")
    return out


def simulate(run_godot, generated, seconds, *extra) -> dict:
    out = run_godot("--scene", "cottage", f"--generated={generated}", "--timescale=8",
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
    # there, sits in the armchair ~14 s in (pose_sit on her own AnimationPlayer) and later
    # stands and idles again.
    assert simulate(run_godot, cottage_dir, 0.9)["clip"] == "walk"
    assert simulate(run_godot, cottage_dir, 4)["clip"] == "idle"
    sitting = simulate(run_godot, cottage_dir, 20)
    assert (sitting["pose"], sitting["clip"]) == ("sit", "pose_sit")
    later = simulate(run_godot, cottage_dir, 100)
    assert (later["pose"], later["clip"]) == ("stand", "idle")


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
