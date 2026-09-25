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
