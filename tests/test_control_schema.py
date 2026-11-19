"""The control protocol's contract (docs/schema/geogen-control.v1.schema.json) agrees with the Godot
runtime's command table, and checks scripts: every test scenario is validated against it
(conftest.godot_script), so a runtime implementing the schema can run them."""

import json
import re
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


# --- api.gd's records against its handlers --------------------------------------------------------

API = ROOT / "runtime" / "godot" / "scripts" / "api.gd"


def _functions(src: str) -> dict[str, str]:
    """GDScript function name -> its text (a function runs until the next top-level func)."""
    parts = re.split(r"^(?=(?:static )?func \w+\()", src, flags=re.M)
    return {re.match(r"(?:static )?func (\w+)", p).group(1): p for p in parts if re.match(r"(?:static )?func ", p)}


def _reads(text: str, functions: dict[str, str], seen: set[str]) -> set[str]:
    """The argument names ``text`` reads (args.get("x"), args.has("x"), args["x"]), following calls that hand
    args (or a copy of it) to another function of api.gd; minus what the code sets on its copies itself."""
    alias = "|".join(["args"] + re.findall(r"var (\w+) :?= args\.duplicate\(\)", text))
    names = set(re.findall(rf'\b(?:{alias})(?:\.get\(|\.has\(|\[)"(\w+)"', text))
    for callee in re.findall(rf"\b(_\w+)\((?:[^()]*, )?(?:{alias})\b", text):
        if callee in functions and callee not in seen:
            seen.add(callee)
            names |= _reads(functions[callee], functions, seen)
    return names - set(re.findall(rf'\b(?!args\b)(?:{alias})\["(\w+)"\] =', text))


def test_every_command_has_a_handler_that_reads_only_declared_arguments():
    src = API.read_text()
    start = src.index("const COMMANDS := {") + len("const COMMANDS := ")
    commands = eval(src[start:src.index("\n}\n", start) + 2], {"__builtins__": {}})
    functions = _functions(src)
    arms = re.split(r'^        ("[\w.]+"(?:, "[\w.]+")*):\s*$', functions["_run"], flags=re.M)
    handled = {}
    for names, body in zip(arms[1::2], arms[2::2]):
        handled[tuple(re.findall(r'"([\w.]+)"', names))] = _reads(body, functions, set())
    flat = {c for group in handled for c in group}
    assert flat == set(commands), (f"commands without a match arm in _run: {sorted(set(commands) - flat)}; "
                                   f"arms without a command: {sorted(flat - set(commands))}")
    for group, read in handled.items():     # an arm several commands share reads the union of their arguments
        declared = set().union(*(commands[c]["args"] for c in group))
        assert read <= declared, f"{', '.join(group)} read undeclared arguments {sorted(read - declared)}"
