class_name GeogenRules
extends Node
## Runs the loaded package's rules program (docs/rules.md) against the live world.
##
## A GeogenMachine (machine.gd, the port of the reference evaluator) holds the
## state and runs the rules; this node is its host:
## - the manifest's `state` and `rules` sections make the program, `quests` what
##   to report about it;
## - interactions that are loaded are the world's own: the machine reads their
##   state from the nodes and drives them through `set_state`, and their
##   arrivals come back as events. One in a chunk that isn't loaded is simulated;
## - every event on the bus (events.gd) is an input, and what the machine
##   changes or raises goes back on the bus (`origin: "rules"`), so listeners
##   see one stream;
## - the world clock is the machine's clock, and time passes with physics.
##
## The player is the actor "player"; NPCs are actors by stable ID. A package's
## state is kept when its world unloads and comes back when it loads again, so
## a quest survives a trip to another scene.
##
## Quests are only reported here (docs/quests.md): `quest event: {...}` whenever
## a quest's state or an objective's latch changes, and `report(actor)`.

signal quest_event(event: Dictionary)
## The program or all of its state was replaced (a package loaded or unloaded, a save
## restored): nothing fired, so whoever shows state re-reads it.
signal reset

var world: WorldLoader
var machine: GeogenMachine = null
var package := ""
## The manifest's `quests` section ({} when the package has none).
var quests := {}

var _saves := {}            # package -> its machine's save, while another world is loaded
var _reports := {}          # package -> {actor: its report()} as it unloaded: what the quest log lists of it
var _watched := {}          # variable key -> {quest, objective?}: what quest events come from
var _expected := {}         # interaction key -> [values we asked the node for]: their echoes aren't inputs
var _minute := -1
var _listener := 0


func _ready() -> void:
    _listener = world.events.listen(_on_event, GeogenEvents.RULE_TYPES)


## Start running a package: called by WorldLoader when its manifest is read.
func load_package(name_: String, manifest: Dictionary) -> void:
    var state: Dictionary = manifest.get("state", {})
    var rules: Dictionary = manifest.get("rules", {})
    if machine != null or (state.is_empty() and rules.is_empty()):
        return
    package = name_
    quests = manifest.get("quests", {})
    _watched.clear()
    for id in quests.get("quests", {}):
        var quest: Dictionary = quests["quests"][id]
        _watched[quest["status"]] = {"quest": id}
        for objective in quest["objectives"]:
            _watched[objective["done"]] = {"quest": id, "objective": objective["id"]}
    var program := {"format": "geogen-rules", "version": 1, "variables": state.get("variables", {}),
        "interactions": state.get("interactions", {}), "bindings": rules.get("bindings", []),
        "rules": rules.get("rules", [])}
    var saved: Dictionary = _saves.get(package, {})
    machine = GeogenMachine.new(program, {},
        {"host_get": _host_get, "host_set": _host_set, "on_changed": _on_changed, "on_signal": _on_signal})
    if not saved.is_empty():
        machine.load_save(saved)
    # The world's time before anyone is there: the machine starts at noon, and a quest that
    # opens at a clock time would open at once.
    _sync_clock()
    var actors := _actors()
    for actor in actors:
        if saved.is_empty():
            add_actor(actor, actors[actor]["kind"])        # a fresh package: its watches fire for them
        else:
            machine.add_actor(actor, actors[actor]["kind"])    # one coming back resumes
    _minute = -1
    _report_errors()
    reset.emit()


## Keep what the quest log lists of this package once it is gone (`report_all`). WorldLoader
## calls it before it frees the world: counts over interactions are read from their nodes.
func remember_reports() -> void:
    if machine == null:
        return
    _reports[package] = {}
    for actor in machine.actor_info:
        _reports[package][actor] = _described(report(actor), false)


func unload() -> void:
    if machine != null:
        _saves[package] = machine.save()
        _saves[package]["timers"] = []          # arrivals of things that no longer exist
    machine = null
    quests = {}
    _watched.clear()
    _expected.clear()
    reset.emit()


func _actors() -> Dictionary:
    var out := {"player": {"kind": "player"}}
    for npc in world.npcs:
        if is_instance_valid(npc):
            out[WorldLoader.actor_id(npc)] = {"kind": "npc"}
    return out


## NPCs spawn after the manifest is read (and with streamed chunks): each becomes an actor.
func add_actor(actor: String, kind: String) -> void:
    if machine != null and not machine.actor_info.has(actor):
        machine.add_actor(actor, kind)
        machine.post({"type": "actor", "subject": actor, "actor": actor})   # no rule's event: it settles the watches for them
        _report_errors()


func _physics_process(delta: float) -> void:
    if machine == null:
        return
    machine.advance(delta)
    var clock: GeogenClock = world.clock
    if clock != null:
        var minute := int(clock.hours * 60.0)
        if minute != _minute:
            _minute = minute
            _sync_clock()
    _report_errors()


func _sync_clock() -> void:
    var clock: GeogenClock = world.clock
    if machine == null or clock == null:
        return
    var total := int(clock.hours * 60.0) % 1440
    machine.set_clock({"time": "%02d:%02d" % [total / 60, total % 60], "night": clock.night})


# -- the host seam ---------------------------------------------------------------------------

func _interaction(key: String) -> GeogenInteraction:
    var it = world.interactions_by_key.get(key)
    return it if it != null and is_instance_valid(it) else null


func _host_get(key: String) -> Variant:
    var it := _interaction(key)
    if it != null:
        return it.target
    if key.ends_with("_locked"):
        it = _interaction(key.trim_suffix("_locked"))
        if it != null:
            return it.locked
    return null


func _host_set(key: String, value: Variant, actor: String) -> bool:
    var it := _interaction(key)
    if it != null:
        _expected[key] = _expected.get(key, []) + [value]
        it.set_state(str(value), false, actor)
        return true
    it = _interaction(key.trim_suffix("_locked"))
    if it == null:
        return false
    _expected[key] = _expected.get(key, []) + [value]
    it.set_locked(bool(value), actor)
    return true


func _on_changed(key: String, old: Variant, new: Variant, actor: String) -> void:
    var subject := key.get_slice("#", 0) if "#" in key else ""
    var var_name := key.get_slice("#", 1) if "#" in key else key
    world.events.post("changed", subject, var_name, actor, {"key": key, "from": old, "to": new}, "rules")


func _on_signal(subject: String, signal_name: String, actor: String) -> void:
    world.events.post("signal", subject, signal_name, actor, {}, "rules")


# -- the bus ---------------------------------------------------------------------------------

func _on_event(event: Dictionary) -> void:
    if event["type"] == "changed" and _watched.has(event.get("key", "")):
        _quest_changed(event)
    if machine == null or event.get("origin", "") == "rules":
        return
    match event["type"]:
        "clock":
            _sync_clock()        # the machine raises its own night / day from its clock
        "changed":
            var waiting: Array = _expected.get(event.get("key", ""), [])
            var at := waiting.find(event["data"].get("to"))
            if at >= 0:
                waiting.remove_at(at)    # the move we asked for: the machine queued it already
            else:
                machine.post(event)
        "enter", "leave":
            var data: Dictionary = event["data"].duplicate()
            data["room"] = data.get("kind") == "room"
            machine.post({"type": event["type"], "subject": event["subject"], "name": event["name"],
                "actor": event["actor"], "data": data})
        _:
            machine.post(event)
    _report_errors()


func _report_errors() -> void:
    if machine == null or machine.errors.is_empty():
        return
    for message in machine.errors:
        push_warning("geogen rules: %s" % message)
    machine.errors.clear()


# -- saving ------------------------------------------------------------------------------------

func save() -> Dictionary:
    return {} if machine == null else machine.save()


## Resume from `save()`'s data (docs/rules.md "Saving"): nothing fires, since nothing happened.
func load_save(data: Dictionary) -> void:
    if machine == null or data.is_empty():
        return
    var actors := machine.actor_info.duplicate(true)
    if machine.load_save(data):
        for actor in actors:
            if not machine.actor_info.has(actor):
                machine.add_actor(actor, str(actors[actor].get("kind", "npc")))
    _report_errors()
    reset.emit()


# -- state, for tools --------------------------------------------------------------------------

func get_value(key: String, actor := "") -> Variant:
    return null if machine == null else machine.get_value(key, actor)


## Set a variable from outside (the console, the API). "" when done, else why not.
func set_value(key: String, value: Variant, actor := "") -> String:
    if machine == null:
        return "no rules program is loaded"
    machine.last_error = ""
    var ok := machine.set_value(key, value, actor)
    machine.errors.clear()
    return "" if ok else machine.last_error


## Stored values that differ from their defaults: {world: {...}, actors: {id: {...}}} (latches left out).
func stored() -> Dictionary:
    if machine == null:
        return {"world": {}, "actors": {}}
    var out := {"world": _visible(machine.world), "actors": {}}
    for actor in machine.actors:
        out["actors"][actor] = _visible(machine.actors[actor])
    return out


static func _visible(values: Dictionary) -> Dictionary:
    var out := {}
    for key in values:
        if not (key.begins_with("once:") or key.begins_with("watch:")):
            out[key] = values[key]
    return out


# -- quests ------------------------------------------------------------------------------------

## Every quest's state for an actor: what a quest log shows. Objectives are
## done | open | waiting, with `progress: [n, of]` and `markers` (world positions of
## an open objective's subjects) where there are any.
func report(actor := "player") -> Array:
    var out := []
    if machine == null:
        return out
    var markers: Dictionary = quests.get("markers", {})
    for id in quests.get("quests", {}):
        var quest: Dictionary = quests["quests"][id]
        var status = machine.get_value(quest["status"], actor)
        var done := {}
        for objective in quest["objectives"]:
            done[objective["id"]] = bool(machine.get_value(objective["done"], actor))
        var objectives := []
        for objective in quest["objectives"]:
            var state := "waiting"
            if done[objective["id"]]:
                state = "done"
            elif status == "active" and objective["after"].all(func(a): return done[a]):
                state = "open"
            var entry := {"id": objective["id"], "text": objective["text"], "optional": objective["optional"],
                "state": state}
            if objective.has("progress"):
                var n = machine.value(objective["progress"]["value"], actor)
                entry["progress"] = [int(n) if n != null else 0, int(objective["progress"]["of"])]
            if state == "open" and not objective.get("marker", {}).get("hidden", false):
                var points := []
                for subject in objective["subjects"]:
                    if markers.has(subject):
                        points.append({"subject": subject, "position": markers[subject]})
                if not points.is_empty():
                    entry["markers"] = points
                    if objective.get("marker", {}).has("area"):
                        entry["area"] = objective["marker"]["area"]
            objectives.append(entry)
        out.append({"id": id, "package": package, "title": quest["title"], "status": status,
            "objectives": objectives})
    machine.errors.clear()
    return out


## Every quest of every package this session has loaded, for an actor: the loaded package's
## `report()`, then the others' as they stood when they unloaded. Each entry also has
## `loaded` (its package is the one running) and `description`.
func report_all(actor := "player") -> Array:
    var out := _described(report(actor), true)
    for other in _reports:
        if other != package or machine == null:
            out.append_array(_reports[other].get(actor, []).duplicate(true))
    return out


func _described(entries: Array, loaded: bool) -> Array:
    for entry in entries:
        entry["loaded"] = loaded
        entry["description"] = str(quests.get("quests", {}).get(entry["id"], {}).get("description", ""))
        for objective in entry["objectives"]:
            objective.erase("markers")          # positions in a world that may not be the loaded one
            objective.erase("area")
    return entries


func _quest_changed(event: Dictionary) -> void:
    var what: Dictionary = _watched[event["key"]]
    var quest: Dictionary = quests["quests"][what["quest"]]
    var out := {"quest": what["quest"], "title": quest["title"], "actor": event["actor"]}
    if what.has("objective"):
        if not event["data"]["to"]:
            return
        out["objective"] = what["objective"]
        out["event"] = "objective_done"
        for objective in quest["objectives"]:
            if objective["id"] == what["objective"]:
                out["text"] = objective["text"]
    else:
        out["status"] = event["data"]["to"]
        out["from"] = event["data"]["from"]
        out["event"] = {"active": "started", "complete": "complete", "failed": "failed"}.get(out["status"], out["status"])
    print("quest event: %s" % JSON.stringify(out))
    quest_event.emit(out)
