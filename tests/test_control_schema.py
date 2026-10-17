"""The control protocol's contract (docs/schema/geogen-control.v1.schema.json) agrees with the Godot
runtime's command table, and checks scripts: every test scenario is validated against it
(conftest.godot_script), so a runtime implementing the schema can run them."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from geogen.runtime_client import step

ROOT = Path(__file__).parent.parent
SCHEMA = ROOT / "docs" / "schema" / "geogen-control.v1.schema.json"


def test_schema_is_current_with_the_runtime(tmp_path):
    # The generator writes the schema from api.gd's COMMANDS: run it on a copy and compare.
    before = SCHEMA.read_text()
    try:
        subprocess.run([sys.executable, str(ROOT / "docs" / "schema" / "make_control_schema.py")], cwd=ROOT, check=True,
                       capture_output=True)
        assert SCHEMA.read_text() == before, "run python docs/schema/make_control_schema.py and commit the schema"
    finally:
        SCHEMA.write_text(before)


def test_scripts_validate(control_validator):
    control_validator({"start": {"spawn": [0, 0, 6], "yaw": 90, "time": "21:30", "keys": ["key_suite"]},
                       "steps": [step("player.teleport", pos=[1, 0, 2]), step("player.move", seconds=1.5),
                                 step("player.use", retry=2, id="door"), step("npcs.greet", npc="clerk", wait=5),
                                 step("state.load", state={"format": "geogen-save", "version": 1, "now": 0,
                                                           "clock": {"time": "08:00", "hour": 8, "night": False,
                                                                     "day": 0},
                                                           "world": {}, "actors": {}, "timers": []})]})


@pytest.mark.parametrize("script, problem", [
    ({"steps": [{"cmd": "player.fly"}]}, "player.fly"),                                    # no such command
    ({"steps": [step("player.move", second=1)]}, "second"),                              # a misspelt argument
    ({"steps": [step("player.teleport", pos="0,0,1")]}, "0,0,1"),                       # a point is a list
    ({"steps": [step("wait", frames=1)], "start": {"spawns": [0, 0, 0]}}, "spawns"),     # an unknown start
    ({"steps": [{"cmd": "wait", "args": {}, "retries": 2}]}, "retries"),                 # an unknown step key
])
def test_scripts_with_mistakes_are_rejected(control_validator, script, problem):
    with pytest.raises(Exception, match=problem):
        control_validator(script)


def test_every_command_is_documented():
    schema = json.loads(SCHEMA.read_text())
    commands = schema["$defs"]["commands"]["const"]
    assert set(commands) == set(schema["$defs"]["args"])
    assert all(text for name, text in commands.items() if name != "time.resume")
