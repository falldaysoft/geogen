class_name GeogenTraceRunner
extends RefCounted
## Replays conformance traces (tests/data/rules/*.json, docs/rules.md "Traces")
## against GeogenMachine: the GDScript twin of geogen/rules.py `run_trace`.
##
## A trace is {name, program, actors, steps}. A step is one of
##   {"use": key, "actor", "used"}     {"event": {...}}
##   {"set": key, "to": value, "actor"}   {"advance": seconds}   {"clock": {...}}
##   {"save_load": true}               a round trip through a save (as JSON)
##   {"expect": {state, actors, signals, signal_data, fired, feedback, clock, timers, errors}}
## `signals` ([subject, name] pairs) and `fired` (rule ids) are exactly what
## happened since the last expect; `errors` is a count. JSON null and "" are
## the same subject.
##
## `godot --headless --path runtime/godot -- --trace=FILE` (repeatable) prints
## one line per file, `trace: {"name": ..., "problems": [...]}`, and exits 1 if
## any has problems. No world is loaded.


## Replay a trace; the mismatches ([] when the machine conforms).
static func run(trace: Dictionary) -> Array[String]:
    var problems: Array[String] = []
    var program: Dictionary = trace.get("program", {})
    var actors: Dictionary = trace["actors"] if trace.get("actors") is Dictionary else {}
    var machine := GeogenMachine.new(program, actors)
    var steps: Array = trace.get("steps", [])
    for index in steps.size():
        var step: Dictionary = steps[index]
        var where := "step %d" % (index + 1)
        var actor := "" if step.get("actor") == null else str(step["actor"])
        if step.has("use"):
            var used := machine.use(step["use"], actor)
            if step.has("used") and used != GeogenConditions.truthy(step["used"]):
                problems.append("%s: use %s gave %s, expected %s" % [where, step["use"], used, step["used"]])
        elif step.has("event"):
            machine.post(step["event"])
        elif step.has("set"):
            if not machine.set_value(step["set"], step.get("to"), actor):
                problems.append("%s: set %s: %s" % [where, step["set"], machine.last_error])
        elif step.has("advance"):
            machine.advance(float(step["advance"]))
        elif step.has("clock"):
            machine.set_clock(step["clock"])
        elif step.has("save_load"):
            var saved: Variant = JSON.parse_string(JSON.stringify(machine.save(), "", false, true))
            var fresh := GeogenMachine.new(program, actors)
            if saved is Dictionary and step["save_load"] is Dictionary:
                for section in step["save_load"].get("without", []):
                    saved.erase(section)
            if not saved is Dictionary or not fresh.load_save(saved):
                problems.append("%s: the save didn't load: %s" % [where, fresh.last_error])
                continue
            fresh.signals = machine.signals     # loading is not an input: nothing fires
            fresh.signal_data = machine.signal_data
            fresh.fired = machine.fired
            fresh.feedback = machine.feedback
            fresh.errors = machine.errors
            machine = fresh
        elif step.has("expect"):
            var expect: Dictionary = step["expect"]
            var state: Dictionary = expect.get("state", {})
            for key in state:
                var got: Variant = machine.get_value(key)
                if not _same(got, state[key]):
                    problems.append("%s: %s is %s, expected %s" % [where, key, _shown(got), _shown(state[key])])
            var per_actor: Dictionary = expect.get("actors", {})
            for who in per_actor:
                for key in per_actor[who]:
                    var got: Variant = machine.get_value(key, who)
                    if not _same(got, per_actor[who][key]):
                        problems.append("%s: %s %s is %s, expected %s" % [where, who, key, _shown(got),
                            _shown(per_actor[who][key])])
            if expect.has("signals"):
                var raised: Array = machine.signals.map(func(s: Array) -> Array: return s.slice(0, 2))
                if not _same(raised, expect["signals"]):
                    problems.append("%s: signals %s, expected %s" % [where, _shown(raised), _shown(expect["signals"])])
            if expect.has("signal_data"):
                var carried: Array = []
                for k in machine.signals.size():
                    carried.append([machine.signals[k][0], machine.signals[k][1], machine.signal_data[k]])
                if not _same(carried, expect["signal_data"]):
                    problems.append("%s: signal data %s, expected %s" % [where, _shown(carried), _shown(expect["signal_data"])])
            if expect.has("fired"):
                var ran: Array = machine.fired.map(func(f: Array) -> Variant: return f[0])
                if not _same(ran, expect["fired"]):
                    problems.append("%s: fired %s, expected %s" % [where, _shown(ran), _shown(expect["fired"])])
            if expect.has("feedback") and not _same(machine.feedback, expect["feedback"]):
                problems.append("%s: feedback %s, expected %s" % [where, _shown(machine.feedback),
                    _shown(expect["feedback"])])
            var clock_want: Dictionary = expect.get("clock", {})
            for field in clock_want:
                if not _same(machine.clock.get(field), clock_want[field]):
                    problems.append("%s: clock %s is %s, expected %s" % [where, field, _shown(machine.clock.get(field)),
                        _shown(clock_want[field])])
            if expect.has("timers") and machine.timers.size() != int(expect["timers"]):
                problems.append("%s: %d timers pending, expected %d" % [where, machine.timers.size(), int(expect["timers"])])
            if expect.has("errors") and machine.errors.size() != int(expect["errors"]):
                problems.append("%s: %d errors, expected %d: %s" % [where, machine.errors.size(),
                    int(expect["errors"]), _shown(machine.errors)])
            machine.signals = []
            machine.signal_data = []
            machine.fired = []
            machine.feedback = []
            machine.errors = []
        else:
            problems.append("%s: unknown step %s" % [where, _shown(step.keys())])
    return problems


## Replay the trace in a JSON file: {"name", "problems"}.
static func run_file(path: String) -> Dictionary:
    var name := path.get_file().get_basename()
    if not FileAccess.file_exists(path):
        return {"name": name, "problems": ["no such file: %s" % path]}
    var trace: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
    if not trace is Dictionary or not trace.has("program") or not trace.has("steps"):
        return {"name": name, "problems": ["not a trace: %s" % path]}
    return {"name": str(trace.get("name", name)), "problems": run(trace)}


## Run each file, print its `trace: {...}` line; the exit code (1 if any has problems).
static func run_files(paths: Array) -> int:
    var code := 0
    for path in paths:
        var result := run_file(str(path))
        print("trace: %s" % JSON.stringify(result))
        if not (result["problems"] as Array).is_empty():
            code = 1
    return code


## Equal as a trace means it: numbers by value, null as "", lists item by item.
static func _same(got: Variant, want: Variant) -> bool:
    if got is Array or want is Array:
        if not (got is Array and want is Array) or got.size() != want.size():
            return false
        for i in got.size():
            if not _same(got[i], want[i]):
                return false
        return true
    if got == null:
        got = ""
    if want == null:
        want = ""
    return GeogenConditions.same(got, want)


static func _shown(value: Variant) -> String:
    return JSON.stringify(value)
