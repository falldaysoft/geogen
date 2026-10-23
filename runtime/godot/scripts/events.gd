class_name GeogenEvents
extends Node
## The world's one stream of events (docs/rules.md "Events").
##
## Everything that happens is posted here as
##   {type, subject, name, actor, package, data}  (+ key for "changed")
## with `subject` a stable ID, `actor` "player", an NPC's ID or "" for nobody,
## and `package` the scene the subject belongs to.
##
## Types rules can trigger on:
##   signal   a thing raised `name`: an interaction arriving at a state with an
##            emit, a refused use ("locked"), an NPC an actor greeted ("greeted", from
##            the NPC, with the greeter as the actor), or one the rules program raises
##            (a rule's emit, a signal an asset declares)
##   changed  state took a new value: `key`, data.from, data.to
##   use      an actor used an interaction (`name`), before it moves
##   enter, leave   an actor entered or left a place or room (data.kind, data.place)
##   travel   an actor went through a travel point (`name`: the target scene)
##   clock    dusk or dawn (`name`: night | day); never has an actor
##   start    a new game began for an actor
## Runtime-only types, for tools and tests (no rule sees them):
##   arrived  an interaction reached a state (`name`; data.state, data.emit, data.asset)
##   npc      an NPC's trace (`name`: decide, step, done, fail, greet, ...; data: the trace)
##
## An event the rules program caused (a variable it set, a signal it raised)
## also has `origin: "rules"`.
##
## Listeners get events in the order they were posted. An event posted while
## another is being delivered waits its turn, so every listener sees the same
## order. `--events[=type,...]` prints each as `event: {...}`.

signal posted(event: Dictionary)

const RULE_TYPES := ["signal", "changed", "use", "enter", "leave", "travel", "clock", "start", "use_item"]
const RUNTIME_TYPES := ["arrived", "npc"]
const HISTORY := 256

## Used for events whose subject has no package of its own (the clock, a new game).
var package := ""
## Types to print (empty: none; ["*"]: all).
var print_types: Array = []
## The last HISTORY events, oldest first.
var history: Array[Dictionary] = []
## How many events were ever posted: history[-1] is number `count`.
var count := 0

var _listeners := {}                    # id -> {call, types, subjects}
var _next_id := 1
var _queue: Array[Dictionary] = []
var _delivering := false


func post(type: String, subject := "", event_name := "", actor := "", data := {}, origin := "") -> void:
    var event := {"type": type, "subject": subject, "name": event_name, "actor": actor,
        "package": subject.get_slice(":", 0) if ":" in subject else package, "data": data}
    if origin != "":
        event["origin"] = origin        # "rules": caused by the rules program, not by the world
    if type == "changed" and data.has("key"):
        event["key"] = data["key"]
        data.erase("key")
    _queue.append(event)
    if _delivering:
        return
    _delivering = true
    while not _queue.is_empty():
        _deliver(_queue.pop_front())
    _delivering = false


## Call `callable(event)` for events of `types` (empty: any) about `subjects`
## (stable IDs; empty: any). Returns an id for `unlisten`.
func listen(callable: Callable, types: Array = [], subjects: Array = []) -> int:
    var id := _next_id
    _next_id += 1
    _listeners[id] = {"call": callable, "types": types, "subjects": subjects}
    return id


func unlisten(id: int) -> void:
    _listeners.erase(id)


## Whether `event` is one a rule's or signal's `on` asks for (docs/rules.md "Rules").
static func matches(on: Dictionary, event: Dictionary) -> bool:
    if on.get("type") != event["type"]:
        return false
    if on.has("subject"):
        var wanted = on["subject"]
        if not (wanted is Array and event["subject"] in wanted) and not (wanted is String and wanted == event["subject"]):
            return false
    for field in ["name", "key"]:
        if on.has(field) and on[field] != event.get(field):
            return false
    if on.has("to"):
        var to = event["data"].get("to")
        if typeof(to) != typeof(on["to"]) or to != on["to"]:
            return false
    return true


func _deliver(event: Dictionary) -> void:
    count += 1
    history.append(event)
    if history.size() > HISTORY:
        history.pop_front()
    if "*" in print_types or event["type"] in print_types:
        print("event: %s" % JSON.stringify(event))
    posted.emit(event)
    for id in _listeners.keys():
        var entry: Dictionary = _listeners.get(id, {})
        if entry.is_empty():
            continue                    # unlistened by an earlier listener
        if not entry["types"].is_empty() and not event["type"] in entry["types"]:
            continue
        if not entry["subjects"].is_empty() and not event["subject"] in entry["subjects"]:
            continue
        entry["call"].call(event)
