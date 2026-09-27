"""Showcase scenes (assets/runtime_scenes.yaml, group showcase) in the Godot runtime."""

import json

import pytest

from geogen.export import export_scene


def _json_line(out: str, prefix: str):
    return json.loads(next(l for l in out.splitlines() if l.startswith(prefix)).removeprefix(prefix))


@pytest.fixture(scope="module")
def hotel_dir(tmp_path_factory, built_scene):
    out = tmp_path_factory.mktemp("generated")
    export_scene(built_scene("hotel_showcase"), out / "hotel_showcase.glb")
    return out


def test_hotel_showcase_playtest_from_the_street(run_godot, hotel_dir):
    out = run_godot("--scene=hotel_showcase", f"--generated={hotel_dir}", "--playtest=2",
                    engine_args=("--fixed-fps", "60"))
    result = _json_line(out, "playtest: ")
    assert result["ok"] and not result["unreachable"] and not result["unreachable_targets"], result


def test_hotel_showcase_staff_and_guests(run_godot, hotel_dir):
    out = run_godot("--scene=hotel_showcase", f"--generated={hotel_dir}", "--timescale=8", "--simulate=240")
    npcs = {n["npc"]: n for n in _json_line(out, "npc summary: ")}
    assert any(k.startswith("desk") for k in npcs["receptionist_npc"]["used"]), npcs["receptionist_npc"]
    # Guests keep to their own storey (floors at 3.65 and 6.4) and never leave their rooms' affordances.
    for name, floor in (("guest_101_npc", 3.65), ("guest_207_npc", 6.4)):
        guest = npcs[name]
        assert abs(guest["position"][1] - floor) < 1.2, guest
        assert guest["used"], guest
    pedestrians = [n for k, n in npcs.items() if k.startswith("pedestrians")]
    assert len(pedestrians) == 4 and all(p["decisions"] > 2 for p in pedestrians)


@pytest.fixture(scope="module")
def park_dir(tmp_path_factory, built_scene):
    out = tmp_path_factory.mktemp("generated")
    export_scene(built_scene("park"), out / "park.glb")
    return out


def test_park_visitors_keep_to_the_paths_and_sit_down(run_godot, park_dir):
    out = run_godot("--scene=park", f"--generated={park_dir}", "--timescale=8", "--simulate=300", "--time=16:00")
    visitors = _json_line(out, "npc summary: ")
    assert len(visitors) == 8
    used = {k.split("#")[0].rstrip("_0123456789") for v in visitors for k in v["used"]}
    assert {"bench", "picnic"} <= used, used
    for v in visitors:
        assert v["outside"] == 0.0, v                 # never left the railed park
        x, _, z = v["position"]
        assert abs(x) < 35 and abs(z) < 28, v
        assert len(v["failures"]) <= 2, v["failures"]
