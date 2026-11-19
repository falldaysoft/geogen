"""Write docs/schema/geogen-control.v1.schema.json from runtime/godot/scripts/api.gd's COMMANDS: one record
per command, its description and its arguments' types (TYPES below says what each type name means). Run from
the repository root after adding or changing a command: python docs/schema/make_control_schema.py
(tests/test_control_schema.py checks they agree, and that api.gd handles and declares what it reads)."""

import json
from pathlib import Path

s = Path("runtime/godot/scripts/api.gd").read_text()
start = s.index("const COMMANDS := {") + len("const COMMANDS := ")
# The table is a GDScript literal of strings, dicts and lists joined with +: a Python expression too.
COMMANDS = eval(s[start:s.index("\n}\n", start) + 2], {"__builtins__": {}})

POINT = {"$ref": "#/$defs/point"}
NUM, INT, STR = {"type": "number"}, {"type": "integer"}, {"type": "string"}
TYPES = {   # api.gd's argument types
    "point": POINT, "number": NUM, "integer": INT, "day": {"type": "integer", "minimum": 0},
    "bool": {"type": "boolean"}, "string": STR, "strings": {"type": "array", "items": STR}, "any": {},
    "clock": {"type": "string", "pattern": "^[0-9]{1,2}:[0-9]{2}$"},
    "pixel": {"type": "array", "items": NUM, "minItems": 2, "maxItems": 2},
    "size": {"type": "array", "items": INT, "minItems": 2, "maxItems": 2},
    "save": {"$ref": "geogen-save.v1.schema.json"},
    "box": {"type": "object", "properties": {"min": POINT, "max": POINT}},
    "commands": {"type": "array", "items": {"$ref": "#/$defs/command"}},
    "streams": {"type": "array", "items": {"enum": ["events", "quest", "interaction", "npc", "room", "place", "travel",
                                                    "traffic"]}},
    "needs": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1}},
    "params": {"type": "object", "additionalProperties": {}},
}


def arg_type(command, name, t):
    if isinstance(t, list):
        return {"enum": t}
    if t not in TYPES:
        raise SystemExit(f"{command}.{name}: unknown type {t!r} (one of {', '.join(TYPES)} or a list)")
    return TYPES[t]


commands = {name: record["doc"] for name, record in COMMANDS.items()}
args_defs = {name: {"type": "object", "additionalProperties": False,
                    "properties": {a: arg_type(name, a, t) for a, t in record["args"].items()}}
             for name, record in COMMANDS.items()}

schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/falldaysoft/geogen/docs/schema/geogen-control.v1.schema.json",
    "title": "geogen control v1",
    "description": (
        "How tools, tests and agents drive a geogen runtime, whichever engine runs it (Godot is the reference: "
        "runtime/godot/scripts/api.gd). The same commands arrive two ways. Live: newline-delimited JSON over a "
        "localhost socket, requests {id, cmd, args} answered {id, ok, result | error}, plus {event, data} lines "
        "for subscribed streams. Scripted: a script document (this schema's root) whose steps run in order once "
        "the world is loaded, each printing a stepResult line; then a scriptDone line, and the runtime exits "
        "(status 1 if a step failed). Points are [x, y, z] world metres ([x, z]: on the ground); angles are degrees; "
        "times are world seconds (they speed up with time.scale) and frames are physics frames. A runtime may "
        "add commands; one it doesn't have answers an error naming it."),
    "type": "object",
    "additionalProperties": False,
    "required": ["steps"],
    "properties": {
        "start": {"$ref": "#/$defs/start"},
        "steps": {"type": "array", "items": {"$ref": "#/$defs/step"}},
        "stop_on_error": {"type": "boolean", "default": True,
                          "description": "Stop at the first step that fails (default); otherwise run on."},
    },
    "$defs": {
        "point": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 3,
                  "description": "[x, y, z] in world metres, or [x, z] on the ground."},
        "start": {
            "description": "How the world starts: applied before it loads (so a spawn primes what streams in "
                           "first, and a load replaces the new game's start).",
            "type": "object", "additionalProperties": False,
            "properties": {
                "scene": {"type": "string", "description": "The scene to load (default: the catalogue's)."},
                "spawn": {"$ref": "#/$defs/point", "description": "Where the player's feet start."},
                "yaw": {"type": "number"}, "pitch": {"type": "number"},
                "time": {"type": "string", "pattern": "^[0-9]{1,2}:[0-9]{2}$", "description": "World clock."},
                "day_length": {"type": "number", "description": "Real seconds per 24 h of world time (0 stops it)."},
                "timescale": {"type": "number", "description": "World speed (physics ticks scale too)."},
                "keys": {"type": "array", "items": {"type": "string"}, "description": "Lock keys the player holds."},
                "load": {"type": "string", "description": "A geogen-save to start from instead of a new game."},
                "set": {"type": "object", "description": "{state key: value} set as the world loads (after a load)."},
                "give": {"type": "object", "additionalProperties": {"type": "integer"},
                         "description": "{item: count} given to the player as the world loads."},
                "stream_radius": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                                  "description": "Streaming radii (m): full exterior, LOD, interiors."},
                "open_doors": {"type": "string", "description": "Open doors matching this glob before the navmesh "
                                                                "bakes, and hold them."},
                "npc_limit": {"type": "integer", "description": "Spawn at most this many NPCs."},
            },
        },
        "command": {
            "type": "object", "required": ["cmd"],
            "properties": {"cmd": {"enum": sorted(commands)}, "args": {"type": "object"}},
            "allOf": [{"if": {"properties": {"cmd": {"const": c}}},
                       "then": {"properties": {"args": {"$ref": f"#/$defs/args/{c}"}}}} for c in sorted(commands)],
        },
        "step": {
            "description": "One command, with what a script adds: retry (world seconds to keep trying a step that "
                           "fails, a physics frame apart) and an id to find its result by.",
            "allOf": [{"$ref": "#/$defs/command"}],
            "properties": {"cmd": True, "args": True, "retry": {"type": "number", "minimum": 0},
                           "id": {"type": "string"}},
            "unevaluatedProperties": False,
        },
        "stepResult": {
            "description": "A scripted runtime prints one per step run: `script: {...}`.",
            "type": "object", "required": ["step", "cmd", "ok"],
            "properties": {"step": {"type": "integer"}, "cmd": {"type": "string"}, "id": {"type": "string"},
                           "ok": {"type": "boolean"}, "result": {}, "error": {"type": "string"}},
        },
        "event": {
            "description": "A subscribed stream's event: `script event: {...}` in a script, a line on the socket.",
            "type": "object", "required": ["event", "data"],
            "properties": {"event": {"type": "string"}, "data": {"type": "object"}},
        },
        "scriptDone": {
            "description": "Printed after the last step run: `script done: {...}`.",
            "type": "object", "required": ["steps", "ok"],
            "properties": {"steps": {"type": "integer"}, "ok": {"type": "boolean"}},
        },
        "commands": {"description": "What each command does (the runtime's `help`).",
                     "type": "object", "const": commands},
        "args": args_defs,
    },
}
Path("docs/schema/geogen-control.v1.schema.json").write_text(json.dumps(schema, indent=2) + "\n")
print(len(commands), "commands")
