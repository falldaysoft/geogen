"""A meal eaten in the runtime (geogen.105): Ana's planned chain in household_test ends at the
chair, where the plate is set out on the table and eaten down (full, half, empty), and then the
dirty plate is what her hand carries."""

import json

import pytest
from geogen.runtime_client import step

pytestmark = [pytest.mark.godot]


def test_the_meal_is_set_out_eaten_down_and_the_plate_carried_off(godot_script, scene_export):
    run = godot_script([step("time.simulate", seconds=45)], "--scene=household_test",
                       f"--generated={scene_export('household_test')}", "--npc-trace", start={"timescale": 4})
    events = [json.loads(line[5:]) for line in run.output.splitlines() if line.startswith("npc: ")]
    set_out = [e for e in events if e.get("event") == "set_out"]
    assert set_out and set_out[0]["item"] == "meal"
    assert set_out[0]["at"][1] > 0.6, "on the table top, not the floor"
    assert [e["show"] for e in events if e.get("event") == "show"][:2] == [{"meal": "half"}, {"meal": "empty"}]
    (ana,) = run["time.simulate"]["npcs"]
    assert ana["carrying"] == ["dirty_plate"] and ana["holding"] == "dirty_plate"
