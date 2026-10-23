class_name GeogenMachine
extends RefCounted
## A rules program running: its state, its clock and its pending timers.
##
## The GDScript twin of geogen/rules.py `Machine`; docs/rules.md is the contract
## and tests/data/rules/*.json the traces both must pass (`--trace=FILE`,
## trace_runner.gd). It is pure data: no nodes, no scene tree, no signals.
##
## Feed it inputs, each handled to completion before the next:
##   post(event)                 an event from outside (a place entered, a signal, a new game)
##   use(key, actor)             an actor uses an interaction; false when it can't be used now
##   set_value(key, value, actor)  set a variable from outside (a console, a test)
##   set_clock({time, night, day})   move the clock; dusk and dawn are events
##   advance(seconds)            let time pass: arrivals, dwells and delayed rules, in time order
## Clock timers (an `after` of {minutes: n}: game-clock time) run when the clock
## reaches them, at the end of whichever input moved it (set_clock, a skip effect).
## and read `get_value(key, actor)`, `value(tree, actor)` and the logs `signals`
## ([subject, name, actor]; `signal_data` holds each one's event data), `fired` ([rule id, actor]), `feedback` ([notify | play,
## text or sound, actor]: what rules and refused uses tell an actor) and `errors` (messages).
## `save()` / `load_save(data)` carry everything but bindings, which are recomputed.
##
## "No actor" is "" (Python's None), and so is "no subject". Numbers are floats
## when they come from JSON: 1 and 1.0 are one value, but true is not 1.
##
## Where Python raises, this reports: a failing rule effect or binding goes in
## `errors` exactly as in the reference; a bad call from outside (an undeclared
## key, a value of the wrong type, actor state with no actor) returns false or
## null, sets `last_error` and, for inputs, adds "<input>: <message>" to `errors`.
## Programs are assumed sound (export runs rules.check_program).
##
## Host seam. On its own the machine simulates interactions as the reference
## does: the variable changes at once and an arrival timer runs `duration`
## later (emit, then / after dwell). A live runtime, whose nodes really move,
## gives Callables instead (in `_init`'s `host`, so the first settle sees them,
## or by assigning the properties):
##   host_get(key: String) -> Variant
##       the current value of a key the host owns, or null for one it doesn't
##       (the machine's own store answers then). Instance and world keys only:
##       actor state and bound variables always stay with the machine.
##   host_set(key: String, value: Variant, actor: String) -> bool
##       asked only for keys host_get answers, when a use, a rule or set_value
##       changes one. The value is already checked against the variable's type
##       and differs from host_get's. Return true once the change is under way
##       (false is reported as an error). The machine then queues the `changed`
##       event itself, so rules cascade in the reference's order, stores nothing
##       and schedules no arrival: the host posts what happens next, e.g.
##       post({"type": "signal", "subject": thing, "name": emit, "actor": actor})
##       on arrival. A change the machine did not ask for (a `then` move, a
##       node driven directly) is posted as a `changed` event, the bus's shape:
##       {"type": "changed", "key": key, "subject": thing, "name": name,
##        "actor": actor, "data": {"from": old, "to": new}}.
## Hooks, called for what the machine itself causes (never for what was posted
## to it, or for a change handed to host_set, which the host already knows):
##   on_changed(key: String, old: Variant, new: Variant, actor: String)
##       a stored or bound value changed; called at the moment it does.
##   on_signal(subject: String, name: String, actor: String, data: Dictionary)
##       a rule's emit (with the data it gives), a simulated arrival's emit or a
##       refused use's "locked"; called when the signal is processed (when it joins `signals`).
##   on_feedback(kind: String, value: String, actor: String)
##       a rule's notify / play, or the `refused` text of a use its `when` stopped;
##       called when it joins `feedback`.
##   on_clock(clock: Dictionary)
##       a skip effect moved the clock forward (the world's clock should follow);
##       called at the moment it does.
## A hook or host callable may call `post`: the event waits until the current
## input is finished. It must not call the other inputs (they are refused).

const MAX_DEPTH := 16               # how many events deep one input may cascade
const SAVE_FORMAT := "geogen-save"
const SAVE_VERSION := 1
const DEFAULT_DURATION := 0.9       # seconds an interaction takes when it doesn't say

var program: Dictionary
## Every variable's {type, default, choices}: the program's, plus each interaction's two.
var variables := {}
var interactions := {}
var rules: Array = []
## Values that differ from their defaults (and world latches).
var world := {}
## Actor ID -> its stored values (and latches).
var actors := {}
## Actor ID -> {kind, room, ...}.
var actor_info := {}
## Binding values: never saved.
var derived := {}
var clock := {"time": "12:00", "hour": 12.0, "night": false, "day": 0}
var now := 0.0
## Pending {at | clock, seq, kind, ...}, kept in firing order: those in seconds, then clock ones.
var timers: Array = []
## [subject, name, actor] of every signal processed.
var signals: Array = []
## The event data of each of `signals`, in step.
var signal_data: Array = []
## [rule id, actor] of every rule fired.
var fired: Array = []
## [notify | play, text or sound, actor]: never saved.
var feedback: Array = []
var errors: Array = []
## Why the last call from outside failed ("" when it didn't).
var last_error := ""

var host_get := Callable()
var host_set := Callable()
var on_changed := Callable()
var on_signal := Callable()
var on_feedback := Callable()
var on_clock := Callable()

var _bindings: Array = []           # in dependency order
var _bound := {}                    # key -> true
var _reads_actor := {}              # rule id -> whether its `when` reads actor state
var _reads_event := {}              # rule id -> whether it reads its event's data (a delayed one keeps it)
var _seq := 0
var _fail := ""                     # set by _read / _look / _value when they return null
var _busy := false
var _silent := false                # no hooks: loading a save
var _inbox: Array = []              # events posted while an input was being handled
var _base: Variant = null           # the clock minute the clock timer firing now was due
var _every_pending: Array = []      # [rule, actor] periodic rules (`every`) not yet on a timer
var _touches_actor := {}            # rule id -> reads or writes actor state (a periodic rule runs per actor)
var _catching := false


## `actors` maps actor IDs to {"kind": "player" | "npc", ...}. `host` may hold
## any of host_get, host_set, on_changed, on_signal, on_feedback (see the class comment).
func _init(rules_program: Dictionary, with_actors: Dictionary = {}, host: Dictionary = {}) -> void:
    program = rules_program
    variables = (program.get("variables", {}) as Dictionary).duplicate()
    interactions = program.get("interactions", {})
    for key in interactions:
        var spec: Dictionary = interactions[key]
        variables[key] = {"type": "choice", "choices": (spec["states"] as Dictionary).keys(),
            "default": spec["initial"]}
        variables["%s_locked" % key] = {"type": "bool", "default": GeogenConditions.truthy(spec.get("locked", false))}
    rules = program.get("rules", [])
    for rule in rules:
        _reads_actor[rule["id"]] = rule.has("when") and _tree_reads_actor(rule["when"])
        _reads_event[rule["id"]] = _rule_reads_event(rule)
        _touches_actor[rule["id"]] = _rule_touches_actor(rule)
        if rule.has("every") and not _touches_actor[rule["id"]]:
            _every_pending.append([rule, ""])
    _bindings = _binding_order()
    for binding in _bindings:
        _bound[binding["set"]] = true
    host_get = host.get("host_get", Callable())
    host_set = host.get("host_set", Callable())
    on_changed = host.get("on_changed", Callable())
    on_signal = host.get("on_signal", Callable())
    on_feedback = host.get("on_feedback", Callable())
    on_clock = host.get("on_clock", Callable())
    for actor in with_actors:
        var info: Dictionary = with_actors[actor] if with_actors[actor] is Dictionary else {}
        add_actor(str(actor), str(info.get("kind", "npc")), info)
    _input([])                          # bindings and watches see the starting state


# -- reading -------------------------------------------------------------------------------

func add_actor(actor: String, kind := "npc", info: Dictionary = {}) -> void:
    if not actors.has(actor):
        actors[actor] = {}
        for rule in rules:                  # new here: its periodic rules start with it
            if rule.has("every") and _touches_actor[rule["id"]] and str(rule.get("actor", kind)) == kind:
                _every_pending.append([rule, actor])
    var entry := {"kind": kind, "room": ""}
    for field in info:
        if field != "kind":
            entry[field] = info[field]
    actor_info[actor] = entry


## The value of a variable (`actor` for "actor:" keys); null with `last_error`
## when the key isn't declared or actor state is asked for without an actor.
func get_value(key: String, actor := "") -> Variant:
    _fail = ""
    var found: Variant = _read(key, actor)
    last_error = _fail
    return found


## The value of a compiled condition tree; null with `last_error` when it can't be evaluated.
func value(tree: Variant, actor := "") -> Variant:
    var found: Variant = _value(tree, actor)
    last_error = _fail
    return found


static func is_actor_key(key: String) -> bool:
    return key.begins_with("actor:")


## The variable key a resolved condition term reads.
static func key_of(term: Dictionary) -> String:
    if term["get"] == "instance":
        return "%s#%s" % [term["id"], term["name"]]
    return "%s:%s.%s" % [term["get"], term["package"], term["name"]]


# -- inputs --------------------------------------------------------------------------------

## An event from outside: {type, subject, name, actor, key, data}. Posted from
## a hook, it is handled once the current input is finished.
## Put the periodic rules that are waiting on timers: at the first input after the machine or
## an actor arrives (not `set_clock`, so a host that sets the clock first counts from that).
func _start_every() -> void:
    for pair in _every_pending:
        var rule: Dictionary = pair[0]
        _schedule(rule["every"], {"kind": "every", "rule": rule["id"], "actor": pair[1]})
    _every_pending = []


func post(event: Dictionary) -> void:
    _start_every()
    var copy := event.duplicate()
    copy["actor"] = _text(event.get("actor"))
    copy["subject"] = _text(event.get("subject"))
    copy["depth"] = 0
    copy.erase("own")
    if _busy:
        _inbox.append(copy)
        return
    _input([copy])


## An actor uses an interaction; false when it can't be used now (no `next`, or locked).
func use(key: String, actor := "") -> bool:
    if _refused("use"):
        return false
    if not interactions.has(key):
        return _bad("use", "'%s' isn't an interaction" % key)
    _start_every()
    var thing := key.substr(0, key.rfind("#"))
    var interaction_name := key.substr(key.rfind("#") + 1)
    var state: Dictionary = interactions[key]["states"].get(_text(_read(key, "")), {})
    if state.get("next") == null:
        return false
    if GeogenConditions.truthy(_read("%s_locked" % key, "")):
        _input([{"type": "signal", "subject": thing, "name": "locked", "actor": actor, "depth": 0, "own": true}])
        return false
    if not allows(key, actor):
        if str(interactions[key].get("refused", "")) != "":
            _tell("notify", str(interactions[key]["refused"]), actor)
            _tell("play", "refused", actor)
        return false
    var queue: Array = [{"type": "use", "subject": thing, "name": interaction_name, "actor": actor, "depth": 0}]
    var problem := _write(key, state["next"], actor, queue, 0)
    _input(queue)
    if problem != "":
        return _bad("use", problem)
    return true


## Set a variable from outside (a test, a console, a loaded tool); false with
## `last_error` when it can't hold that value.
func set_value(key: String, to: Variant, actor := "") -> bool:
    if _refused("set"):
        return false
    _start_every()
    var queue: Array = []
    var problem := _write(key, to, actor, queue, 0)
    _input(queue)
    if problem != "":
        return _bad("set", problem)
    last_error = ""
    return true


## Move the clock (`time` "HH:MM", `night`, `day`); dusk and dawn are events.
## Bindings and watches that read the clock are judged again either way.
func set_clock(values: Dictionary) -> void:
    if _refused("clock"):
        return
    var was_night: bool = GeogenConditions.truthy(clock["night"])
    clock.merge(values, true)
    if values.has("time"):
        var parts := str(values["time"]).split(":")
        clock["hour"] = int(parts[0]) + int(parts[1]) / 60.0
    var queue: Array = []
    var night := GeogenConditions.truthy(clock["night"])
    if night != was_night:
        queue.append({"type": "clock", "subject": "", "name": "night" if night else "day", "actor": "", "depth": 0})
    _input(queue)


## Let `seconds` pass: interactions arrive, dwells and delayed rules run, in
## time order (ties to the one scheduled first), each as its own input.
func advance(seconds: float) -> void:
    _start_every()
    if _refused("advance"):
        return
    var end := now + seconds
    while not timers.is_empty() and timers[0].has("at") and float(timers[0]["at"]) <= end + 1e-9:
        var timer: Dictionary = timers.pop_front()
        now = maxf(now, float(timer["at"]))
        _input(_timer(timer))
    now = end


# -- the semantics -------------------------------------------------------------------------

func _input(queue: Array) -> void:
    _busy = true
    _settle(queue)
    while not _inbox.is_empty():
        _settle([_inbox.pop_front()])
    _busy = false


func _refused(input: String) -> bool:
    if _busy:
        _bad(input, "called while another input is being handled (only post may be)")
    return _busy


func _bad(input: String, message: String) -> bool:
    last_error = message
    errors.append("%s: %s" % [input, message])
    return false


static func _text(value: Variant) -> String:
    return "" if value == null else str(value)


func _read(key: String, actor: String) -> Variant:
    if derived.has(key):
        return derived[key]
    if not variables.has(key):
        _fail = "'%s' isn't declared" % key
        return null
    var fallback: Variant = variables[key]["default"]
    if is_actor_key(key):
        if actor == "":
            _fail = "'%s' is actor state: whose?" % key
            return null
        return (actors.get(actor, {}) as Dictionary).get(key, fallback)
    if host_get.is_valid():
        var held: Variant = host_get.call(key)
        if held != null:
            return held
    return world.get(key, fallback)


func _look(term: Dictionary, actor: String, data: Variant) -> Variant:
    if term["get"] == "clock":
        return clock.get(term["name"])
    if term["get"] == "event":
        if not data is Dictionary:
            _fail = "'event.%s': there is no event here" % term["name"]
            return null
        if not (data as Dictionary).has(term["name"]):
            var fields := ", ".join(PackedStringArray((data as Dictionary).keys()))
            _fail = "'event.%s': the event has no '%s' (it has %s)" % [term["name"], term["name"],
                fields if fields != "" else "no data"]
            return null
        return data[term["name"]]
    if term["get"] == "actor" and not term.has("package"):
        if actor == "":
            _fail = "'actor.%s' read without an actor" % term["name"]
            return null
        return actor if term["name"] == "id" else (actor_info.get(actor, {}) as Dictionary).get(term["name"], "")
    return _read(key_of(term), actor)


## `data` is the triggering event's data (null: no event).
func _value(tree: Variant, actor: String, data: Variant = null) -> Variant:
    _fail = ""
    var problems: Array = []
    var found: Variant = GeogenConditions.evaluate(tree, _look.bind(actor, data), problems)
    if not problems.is_empty():
        if _fail == "":
            _fail = str(problems[0])
        return null
    return found


static func _fits(spec: Dictionary, value: Variant) -> bool:
    match spec.get("type"):
        "bool":
            return typeof(value) == TYPE_BOOL
        "number":
            return GeogenConditions.is_number(value)
        "text":
            return GeogenConditions.is_text(value)
    for choice in spec.get("choices", []):
        if GeogenConditions.equal(choice, value):
            return true
    return false


## The clock as minutes since day 0 began: what clock timers are due at.
func clock_minute() -> float:
    return float(clock.get("day", 0)) * 1440.0 + snappedf(float(clock["hour"]) * 60.0, 0.000001)


## The game-clock minutes of an `after` ({"minutes": n}), or -1 for seconds.
static func clock_minutes(after: Variant) -> float:
    return float(after["minutes"]) if after is Dictionary else -1.0


## A timer `delay` seconds from now, or, for clock time ({"minutes": n}), n clock minutes
## from now: from when the clock timer firing now was due, so a chain of them catches up.
func _schedule(delay: Variant, timer: Dictionary) -> void:
    _seq += 1
    var entry := {"seq": _seq}
    var minutes := clock_minutes(delay)
    if minutes < 0.0:
        entry["at"] = snappedf(now + float(delay), 0.000001)
    else:
        var base: float = _base if _base != null else clock_minute()
        entry["clock"] = snappedf(base + minutes, 0.000001)
    entry.merge(timer)
    timers.append(entry)
    timers.sort_custom(_sooner)


static func _sooner(a: Dictionary, b: Dictionary) -> bool:
    if a.has("at") != b.has("at"):
        return a.has("at")
    var field := "at" if a.has("at") else "clock"
    if float(a[field]) != float(b[field]):
        return float(a[field]) < float(b[field])
    return float(a["seq"]) < float(b["seq"])


## An interaction entered `state`: a clock-time dwell counts from now (not from its
## arrival, which takes seconds), and one waiting in the state it left is dropped.
func _clock_dwell(key: String, state: Variant, actor: String) -> void:
    var kept: Array = []
    for timer in timers:
        if not (timer.has("clock") and timer["kind"] == "dwell" and timer.get("key") == key):
            kept.append(timer)
    timers = kept
    var spec: Dictionary = interactions[key]["states"].get(_text(state), {})
    if spec.get("then") != null and clock_minutes(spec.get("after")) >= 0.0:
        _schedule(spec["after"], {"kind": "dwell", "key": key, "state": _text(state), "actor": actor})


## Run the clock timers the clock has reached, in order, each as its own input.
func _catch_up() -> void:
    if _catching:
        return
    _catching = true
    while true:
        var due: Variant = null
        for timer in timers:
            if timer.has("clock") and float(timer["clock"]) <= clock_minute() + 0.000001:
                due = timer
                break
        if due == null:
            break
        timers.erase(due)
        _base = float(due["clock"])
        _settle(_timer(due))
        _base = null
    _catching = false


## Move the clock forward (a skip effect); what comes due on the way runs after this input.
func _skip(skip: Dictionary) -> void:
    var start := clock_minute()
    var target := 0.0
    if skip.has("to"):
        var parts := str(skip["to"]).split(":")
        target = start - fmod(start, 1440.0) + int(parts[0]) * 60 + int(parts[1])
        if target <= start + 0.000001:
            target += 1440.0
    else:
        target = start + float(skip["minutes"])
    var total := int(roundf(target))
    var minute := total % 1440
    clock["day"] = total / 1440
    clock["time"] = "%02d:%02d" % [minute / 60, minute % 60]
    clock["hour"] = minute / 60.0
    if not _silent and on_clock.is_valid():
        on_clock.call(clock.duplicate())


func _timer(timer: Dictionary) -> Array:
    var queue: Array = []
    var kind: String = timer["kind"]
    var key := _text(timer.get("key"))
    var actor := _text(timer.get("actor"))
    if kind == "rule":
        for rule in rules:
            if rule["id"] == timer["rule"]:
                _effects(rule, actor, queue, 0, timer.get("data"))
                break
    elif kind == "every":
        for rule in rules:
            if rule["id"] == timer["rule"]:     # runs if its condition holds now; the next is a period on
                if _holds(rule, actor):
                    fired.append([rule["id"], actor])
                    _effects(rule, actor, queue, 0)
                _schedule(rule["every"], {"kind": "every", "rule": rule["id"], "actor": actor})
                break
    elif kind == "arrive" and _text(_read(key, "")) == timer["state"]:    # not reversed on the way
        var state: Dictionary = interactions[key]["states"][timer["state"]]
        if GeogenConditions.truthy(state.get("emit")):
            queue.append({"type": "signal", "subject": key.substr(0, key.rfind("#")), "name": state["emit"],
                "actor": actor, "depth": 0, "own": true})
        if state.get("then") != null:
            if clock_minutes(state.get("after")) >= 0.0:
                pass                            # counting since it was entered (_clock_dwell)
            elif GeogenConditions.truthy(state.get("after")):
                _schedule(float(state["after"]), {"kind": "dwell", "key": key, "state": timer["state"], "actor": actor})
            else:
                _report_timer(key, _write(key, state["then"], actor, queue, 0))
    elif kind == "dwell" and _text(_read(key, "")) == timer["state"]:
        _report_timer(key, _write(key, interactions[key]["states"][timer["state"]]["then"], actor, queue, 0))
    return queue


func _report_timer(key: String, problem: String) -> void:
    if problem != "":
        errors.append("timer of %s: %s" % [key, problem])


## Changes a value; "" or why it can't be done (what Python raises).
func _write(key: String, to: Variant, actor: String, queue: Array, depth: int) -> String:
    if not variables.has(key):
        return "'%s' isn't declared" % key
    if _bound.has(key):
        return "'%s' is bound and can't be set" % key
    var spec: Dictionary = variables[key]
    if spec["type"] == "bool" and typeof(to) != TYPE_BOOL:
        to = GeogenConditions.truthy(to)
    if not _fits(spec, to):
        var choices := " (%s)" % ", ".join(PackedStringArray(spec["choices"])) if spec["type"] == "choice" else ""
        return "'%s' is a %s%s: can't be %s" % [key, spec["type"], choices, GeogenConditions.shown(to)]
    var store: Dictionary = world
    if is_actor_key(key):
        if actor == "":
            return "'%s' is actor state, and this has no actor" % key
        if not actors.has(actor):
            actors[actor] = {}
        store = actors[actor]
    elif host_get.is_valid():
        var held: Variant = host_get.call(key)
        if held != null:                # the host's: it makes the change, and says what follows
            if GeogenConditions.same(held, to):
                return ""
            if not host_set.is_valid() or not host_set.call(key, to, actor):
                return "'%s' is the host's, and it wouldn't take %s" % [key, GeogenConditions.shown(to)]
            _changed(key, held, to, actor, queue, depth, false)
            return ""
    var old: Variant = store.get(key, spec["default"])
    if GeogenConditions.same(old, to):
        return ""                       # setting a value it already has is not a change
    if GeogenConditions.same(to, spec["default"]):
        store.erase(key)
    else:
        store[key] = to
    _changed(key, old, to, actor, queue, depth)
    if interactions.has(key):
        _schedule(float(interactions[key].get("duration", DEFAULT_DURATION)),
            {"kind": "arrive", "key": key, "state": to, "actor": actor})
    return ""


func _changed(key: String, old: Variant, new: Variant, actor: String, queue: Array, depth: int, tell := true) -> void:
    var at := key.rfind("#")
    queue.append({"type": "changed", "key": key, "subject": key.substr(0, at) if at >= 0 else "",
        "name": key.substr(at + 1) if at >= 0 else key, "actor": actor, "data": {"from": old, "to": new},
        "depth": depth + 1})
    if tell and not _silent and on_changed.is_valid():
        on_changed.call(key, old, new, actor)


func _matches(on: Dictionary, event: Dictionary) -> bool:
    if on["type"] != event["type"]:
        return false
    var subject: Variant = on.get("subject")
    if subject != null:
        var wanted: Array = subject if subject is Array else [subject]
        if not wanted.has(event.get("subject", "")):
            return false
    for field in ["name", "key"]:
        if on.get(field) != null and not GeogenConditions.equal(on[field], event.get(field)):
            return false
    if on.has("to") and not GeogenConditions.equal((event.get("data", {}) as Dictionary).get("to"), on["to"]):
        return false
    var who := str(on.get("actor", "any"))
    if who == "any":
        return true
    var actor := _text(event.get("actor"))
    if who == "none":
        return actor == ""
    return actor != "" and _text((actor_info.get(actor, {}) as Dictionary).get("kind")) == who


func _fire(rule: Dictionary, actor: String, queue: Array, depth: int, data: Variant = null) -> void:
    if GeogenConditions.truthy(rule.get("once")):
        var latch := "once:%s" % rule["id"]
        var store: Dictionary = world
        if actor != "":
            if not actors.has(actor):
                actors[actor] = {}
            store = actors[actor]
        if GeogenConditions.truthy(store.get(latch)):
            return
        store[latch] = true
    fired.append([rule["id"], actor])
    if GeogenConditions.truthy(rule.get("after")):
        var timer := {"kind": "rule", "rule": rule["id"], "actor": actor}
        if data != null and _reads_event.get(rule["id"], false):
            timer["data"] = data
        _schedule(rule["after"], timer)
    else:
        _effects(rule, actor, queue, depth, data)


func _effects(rule: Dictionary, actor: String, queue: Array, depth: int, data: Variant = null) -> void:
    for effect in rule["do"]:
        if effect.has("set"):
            var to: Variant = _value(effect["to"], actor, data)
            var problem := _fail if to == null else _write(effect["set"], to, actor, queue, depth)
            if problem != "":           # reported and skipped: the rule's other effects still run
                errors.append("rule %s: %s" % [rule["id"], problem])
        elif effect.has("notify"):
            _tell("notify", str(effect["notify"]), actor)
        elif effect.has("play"):
            _tell("play", str(effect["play"]), actor)
        elif effect.has("skip"):
            _skip(effect["skip"])
        else:
            var sent := {}
            if GeogenConditions.truthy(effect.get("forward")) and data is Dictionary:
                sent = (data as Dictionary).duplicate()
            var given: Dictionary = effect.get("data", {})
            var problem := ""
            for field in given:
                var v: Variant = _value(given[field], actor, data)
                if v == null:
                    problem = _fail
                    break
                sent[field] = v
            if problem != "":
                errors.append("rule %s: %s" % [rule["id"], problem])
                continue
            var signal_event := {"type": "signal", "subject": _text(effect.get("subject")), "name": effect["emit"],
                "actor": actor, "depth": depth + 1, "own": true}
            if not sent.is_empty():
                signal_event["data"] = sent
            queue.append(signal_event)


func _tell(kind: String, value: String, actor: String) -> void:
    feedback.append([kind, value, actor])
    if not _silent and on_feedback.is_valid():
        on_feedback.call(kind, value, actor)


## Whether the interaction's `when` holds for `actor` now (true without one). It only gates
## use: rules and set_value drive the interaction whatever it says.
func allows(key: String, actor := "") -> bool:
    var when: Variant = interactions.get(key, {}).get("when")
    if when == null:
        return true
    var found: Variant = _value(when, actor)
    if found == null:
        errors.append("%s when: %s" % [key, _fail])
        return false
    return GeogenConditions.truthy(found)


func _holds(rule: Dictionary, actor: String, data: Variant = null) -> bool:
    if not rule.has("when"):
        return true
    var found: Variant = _value(rule["when"], actor, data)
    if found == null:
        errors.append("rule %s: %s" % [rule["id"], _fail])
        return false
    return GeogenConditions.truthy(found)


## Process events until nothing is left: rules, then bindings, then watches; then the
## clock timers the clock has reached (a skip, or a clock moved on while nobody ran them).
func _settle(queue: Array) -> void:
    _process(queue)
    _catch_up()


func _process(queue: Array) -> void:
    for _round in MAX_DEPTH + 1:
        while not queue.is_empty():
            var event: Dictionary = queue.pop_front()
            var depth := int(event.get("depth", 0))
            if depth > MAX_DEPTH:
                var what := _text(event.get("key"))
                if what == "":
                    what = _text(event.get("name"))
                errors.append(("cascade deeper than %d events dropped at %s %s" % [MAX_DEPTH, event["type"], what])
                    .strip_edges(false, true))
                continue
            var actor := _text(event.get("actor"))
            var subject := _text(event.get("subject"))
            var data: Dictionary = event.get("data", {}) if event.get("data") is Dictionary else {}
            if event["type"] == "signal":
                signals.append([subject, event["name"], actor])
                signal_data.append(data)
                if event.get("own", false) and on_signal.is_valid():
                    on_signal.call(subject, str(event["name"]), actor, data)
            elif (event["type"] == "enter" or event["type"] == "leave") and actor_info.has(actor) \
                    and GeogenConditions.truthy((event.get("data", {}) as Dictionary).get("room")):
                actor_info[actor]["room"] = subject if event["type"] == "enter" else ""
            elif event["type"] == "changed" and interactions.has(_text(event.get("key"))):
                _clock_dwell(str(event["key"]), data.get("to"), actor)
            for rule in rules:
                if rule.get("on") != null and _matches(rule["on"], event) and _holds(rule, actor, data):
                    _fire(rule, actor, queue, depth, data)
            _rebind(queue, depth)
        _rebind(queue, 0)
        _watch(queue)
        if queue.is_empty():
            return
    errors.append("watched conditions kept triggering each other: stopped after %d rounds" % MAX_DEPTH)


func _rebind(queue: Array, depth: int) -> void:
    for binding in _bindings:
        var key: String = binding["set"]
        var spec: Dictionary = variables[key]
        var found: Variant = _value(binding["to"], "")
        var problem := _fail if found == null else ""
        if problem == "":
            if spec["type"] == "bool":
                found = GeogenConditions.truthy(found)
            if not _fits(spec, found):
                problem = "a %s can't be %s" % [spec["type"], GeogenConditions.shown(found)]
        if problem != "":
            errors.append("binding of %s: %s" % [key, problem])
            continue
        var old: Variant = derived.get(key, spec["default"])
        derived[key] = found
        if not GeogenConditions.equal(old, found):
            _changed(key, old, found, "", queue, depth)


## Rules without `on` fire when their condition becomes true (per actor when it
## reads actor state). Every condition is judged against the same settled state
## before any fires.
func _watch(queue: Array) -> void:
    var newly: Array = []
    for rule in rules:
        if rule.get("on") != null or rule.has("every"):
            continue
        var latch := "watch:%s" % rule["id"]
        var subjects: Array = actors.keys() if _reads_actor[rule["id"]] else [""]
        for actor in subjects:
            var store: Dictionary = actors[actor] if actor != "" else world
            var truth := _holds(rule, actor)
            if truth and not GeogenConditions.truthy(store.get(latch)):
                store[latch] = true
                newly.append([rule, actor])
            elif not truth:
                store.erase(latch)
    for pair in newly:
        _fire(pair[0], pair[1], queue, 0)


# -- the program ---------------------------------------------------------------------------

static func _rule_reads_event(rule: Dictionary) -> bool:
    if _tree_has_get(rule.get("when"), "event"):
        return true
    for effect in rule.get("do", []):
        if GeogenConditions.truthy(effect.get("forward")) or _tree_has_get(effect.get("to"), "event"):
            return true
        for field in effect.get("data", {}):
            if _tree_has_get(effect["data"][field], "event"):
                return true
    return false


static func _tree_has_get(tree: Variant, kind: String) -> bool:
    if typeof(tree) != TYPE_DICTIONARY:
        return false
    if tree.has("get"):
        return tree["get"] == kind
    for arg in tree.get("args", []):
        if _tree_has_get(arg, kind):
            return true
    return _tree_has_get(tree.get("where"), kind)


## Whether a rule reads or writes actor state (a periodic rule that does runs per actor).
static func _rule_touches_actor(rule: Dictionary) -> bool:
    if _tree_reads_actor(rule.get("when")):
        return true
    for effect in rule.get("do", []):
        if is_actor_key(str(effect.get("set", ""))) or _tree_reads_actor(effect.get("to")):
            return true
        for field in effect.get("data", {}):
            if _tree_reads_actor(effect["data"][field]):
                return true
    return false


static func _tree_reads_actor(tree: Variant) -> bool:
    if not tree is Dictionary:
        return false
    if tree.get("get") == "actor":
        return true
    for arg in tree.get("args", []):
        if _tree_reads_actor(arg):
            return true
    return tree.has("where") and _tree_reads_actor(tree["where"])


## Variable keys a condition tree reads ("it" terms as each thing's key).
static func _reads(tree: Variant, over: Array, out: Array) -> void:
    if not tree is Dictionary:
        return
    if tree.has("get"):
        if tree["get"] == "it":
            for thing in over:
                out.append("%s#%s" % [thing, tree["name"]])
        elif tree["get"] == "instance" or tree["get"] == "world" or (tree["get"] == "actor" and tree.has("package")):
            out.append(key_of(tree))
        return
    for arg in tree.get("args", []):
        _reads(arg, over, out)
    if tree.has("where"):
        _reads(tree["where"], tree.get("over", []), out)


## Bindings ordered so each comes after the ones it reads.
func _binding_order() -> Array:
    var by_key := {}
    for binding in program.get("bindings", []):
        by_key[binding["set"]] = binding
    var order: Array = []
    var done := {}                      # key -> finished (false while on the stack)
    for key in by_key:
        _order_binding(key, by_key, done, order)
    return order


func _order_binding(key: String, by_key: Dictionary, done: Dictionary, order: Array) -> void:
    if not by_key.has(key) or done.get(key, false):
        return
    if done.has(key):
        errors.append("bindings form a cycle at %s" % key)
        return
    done[key] = false
    var reads: Array = []
    _reads(by_key[key]["to"], [], reads)
    for read in reads:
        _order_binding(read, by_key, done, order)
    done[key] = true
    order.append(by_key[key])


# -- saving --------------------------------------------------------------------------------

## Everything needed to carry on later. World and actor state stay apart;
## bindings aren't saved (nor is anything the host owns).
func save() -> Dictionary:
    return {"format": SAVE_FORMAT, "version": SAVE_VERSION, "now": now, "clock": clock, "world": world,
        "actors": actors, "actor_info": actor_info, "timers": timers}.duplicate(true)


## Take up a save. Loading is not an input: bindings are recomputed, silently,
## since nothing changed while saved. Sections it leaves out are empty; actors it
## doesn't describe stay as they were. False when `data` isn't a save.
func load_save(data: Dictionary) -> bool:
    if _refused("load"):
        return false
    if data.get("format") != SAVE_FORMAT:
        last_error = "not a %s file" % SAVE_FORMAT
        return false
    if int(data.get("version", SAVE_VERSION)) != SAVE_VERSION:
        last_error = "a %s of version %s, not %d" % [SAVE_FORMAT, data["version"], SAVE_VERSION]
        return false
    var copy := data.duplicate(true)
    now = float(copy.get("now", 0.0))
    clock = copy.get("clock", clock)
    world = copy.get("world", {})
    actors = copy.get("actors", {})
    var known := actor_info
    var saved: Dictionary = copy.get("actor_info", {})
    actor_info = {}
    for actor in actors.keys() + saved.keys():
        actor_info[actor] = saved.get(actor, known.get(actor, {"kind": "npc", "room": ""}))
    timers = copy.get("timers", [])
    timers.sort_custom(_sooner)
    _every_pending = []                 # the save holds the periodic timers
    _seq = 0
    for timer in timers:
        _seq = maxi(_seq, int(timer["seq"]))
    derived = {}
    _silent = true
    _rebind([], 0)
    _silent = false
    return true
