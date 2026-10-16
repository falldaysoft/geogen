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
## Content packages (docs/rules.md "Content packages": the `packages` of
## catalogue.json, each a geogen-content file) belong to no scene. They are read
## once; when a scene one of them names loads, its variables and the rules and
## bindings that run there join the scene's program, after the scene's own, and
## its quests join the scene's. Their values are kept apart from any scene's
## (`_carried`), so what a rule set in one scene is read by a rule in the next.
##
## Quests are only reported here (docs/quests.md): `quest event: {...}` whenever
## a quest's state or an objective's latch changes, and `report(actor)`.

signal quest_event(event: Dictionary)
## Feedback for an actor (docs/rules.md "Feedback"): `notify` (text) or `play` (a sound in
## GeogenSound.CHIMES), from a rule's effect or a use an interaction's `when` refused. The HUD
## toasts and plays the player's (hud_notifications.gd).
signal feedback(kind: String, value: String, actor: String)
## The program or all of its state was replaced (a package loaded or unloaded, a save
## restored): nothing fired, so whoever shows state re-reads it.
signal reset

var world: WorldLoader
var machine: GeogenMachine = null
var package := ""
## The manifest's `quests` section ({} when the package has none).
var quests := {}
## The manifest's `hud` section (geogen/hud.py: {stats, banners}; {} when the scene has none).
var hud := {}
var items := {}                 # the manifest's items: id -> {title, tags?, opens?} (geogen.18)

var _saves := {}            # package -> its machine's save, while another world is loaded
var _content := {}          # content package name -> its geogen-content file, as deployed (catalogue.json)
var _content_stamp := ""    # the catalogue.json they were read from, and when it was written
var _applied: Array = []    # the content packages running with the loaded scene
var _carried := {"world": {}, "actors": {}}     # content packages' values, whatever scene is loaded
var _at_load := {}          # the machine's stored world values as the package loaded (see `adopt`)
var _reports := {}          # package -> {actor: its report()} as it unloaded: what the quest log lists of it
var _watched := {}          # variable key -> {quest, objective?}: what quest events come from
var _expected := {}         # interaction key -> [values we asked the node for]: their echoes aren't inputs
var _holding := {}          # interaction key -> true: its node is gone (an unloaded chunk), the machine holds it
var _minute := -1
var _listener := 0


func _ready() -> void:
    _listener = world.events.listen(_on_event, GeogenEvents.RULE_TYPES)


## How many of item ``id`` ``actor`` carries (0 for an item this package doesn't name).
func held(id: String, actor := "player") -> int:
    if machine == null or not items.has(id):
        return 0
    var n = get_value("actor:%s.item_%s" % [package, id], actor)
    return int(n) if n != null else 0


## The lock keys ``actor`` holds by what it carries: each carried item's id and its `opens`.
func carried_keys(actor := "player") -> Array:
    var out := []
    for id in items:
        if held(id, actor) > 0:
            out.append(id)
            out.append_array(items[id].get("opens", []))
    return out


## Start running a package: called by WorldLoader when its manifest is read.
func load_package(name_: String, manifest: Dictionary) -> void:
    var state: Dictionary = manifest.get("state", {})
    var rules: Dictionary = manifest.get("rules", {})
    var content: Array = []
    if machine == null:
        hud = manifest.get("hud", {})
        items = manifest.get("items", {})
        for other in content_packages().values():
            if name_ in other.get("scenes", []):
                content.append(other)
    if machine != null or (state.is_empty() and rules.is_empty() and content.is_empty()):
        return
    package = name_
    quests = manifest.get("quests", {}).duplicate(true)
    # The scene's program, then what each content package adds to it here: its variables, and
    # the rules and bindings that touch this scene or none.
    var variables: Dictionary = state.get("variables", {}).duplicate()
    var bindings: Array = rules.get("bindings", []).duplicate()
    var program_rules: Array = rules.get("rules", []).duplicate()
    _applied = content
    for other in content:
        variables.merge(other.get("variables", {}))
        for section in [["rules", program_rules], ["bindings", bindings]]:
            for entry in other.get(section[0], []):
                if entry.get("scene", name_) == name_:
                    section[1].append(entry)
        var theirs: Dictionary = other.get("quests", {}).get("quests", {})
        if not theirs.is_empty() and not quests.has("quests"):
            quests["quests"] = {}
        for id in theirs:
            var quest: Dictionary = theirs[id].duplicate(true)
            quest["package"] = other["package"]
            quests["quests"][id] = quest
        print("geogen: content package %s runs in %s" % [other["package"], name_])
    _watched.clear()
    for id in quests.get("quests", {}):
        var quest: Dictionary = quests["quests"][id]
        _watched[quest["status"]] = {"quest": id}
        for objective in quest["objectives"]:
            _watched[objective["done"]] = {"quest": id, "objective": objective["id"]}
    var program := {"format": "geogen-rules", "version": 1, "variables": variables,
        "interactions": state.get("interactions", {}), "bindings": bindings, "rules": program_rules}
    var saved: Dictionary = _saves.get(package, {})
    var resumed := not saved.is_empty()
    var actors := _actors()
    saved = _with_carried(saved, actors.keys())
    machine = GeogenMachine.new(program, {},
        {"host_get": _host_get, "host_set": _host_set, "on_changed": _on_changed, "on_signal": _on_signal,
            "on_feedback": tell})
    if not saved.is_empty():
        machine.load_save(saved)
        for key in machine.interactions:        # their nodes are about to load, as they start
            machine.world.erase(key)
            machine.world.erase(key + "_locked")
    _at_load = machine.world.duplicate()
    # The world's time before anyone is there: the machine starts at noon, and a quest that
    # opens at a clock time would open at once.
    _sync_clock()
    for actor in actors:
        machine.add_actor(actor, actors[actor]["kind"])
        # A fresh package: its watches fire for them (no rule's event: it settles the watches).
        # One coming back resumes, but a content package's rules here haven't seen what its
        # state became in the scenes between.
        if not resumed or not content.is_empty():
            machine.post({"type": "actor", "subject": actor, "actor": actor})
    _minute = -1
    _report_errors()
    reset.emit()


## The value of a compiled condition for `actor` now (null when it can't be read: no program,
## or what it reads isn't there). What the HUD's stats and banners show.
func value_of(tree: Variant, actor := "player") -> Variant:
    if typeof(tree) != TYPE_DICTIONARY:
        return tree
    if machine == null:
        return null
    return machine.value(tree, actor if machine.actor_info.has(actor) else "")


## Keep what the quest log lists of this package once it is gone (`report_all`). WorldLoader
## calls it before it frees the world: counts over interactions are read from their nodes.
func remember_reports() -> void:
    if machine == null:
        return
    _reports[package] = {}
    for other in _applied:
        _reports[other["package"]] = {}
    for actor in machine.actor_info:
        _reports[package][actor] = []
        for entry in _described(report(actor), false):      # each quest under the package it belongs to
            var theirs: Dictionary = _reports[entry["package"]]
            theirs[actor] = theirs.get(actor, []) + [entry]


func unload() -> void:
    if machine != null:
        _saves[package] = machine.save()
        _saves[package]["timers"] = []          # arrivals of things that no longer exist
        _carry(_saves[package])
    machine = null
    package = ""                # a scene without a program loads next, maybe: it has no package
    _applied = []
    quests = {}
    hud = {}
    _watched.clear()
    _expected.clear()
    _holding.clear()
    reset.emit()


# -- content packages ------------------------------------------------------------------------

## The deployed content packages by name (catalogue.json's `packages` that are exported),
## read once per catalogue: again only when catalogue.json is rewritten.
func content_packages() -> Dictionary:
    var path := "%s/catalogue.json" % world.generated_dir
    var stamp := "%s@%d" % [path, FileAccess.get_modified_time(path) if FileAccess.file_exists(path) else 0]
    if stamp == _content_stamp:
        return _content
    _content_stamp = stamp
    _content = {}
    var catalogue = JSON.parse_string(FileAccess.get_file_as_string(path)) if FileAccess.file_exists(path) else null
    if not catalogue is Dictionary:
        return _content
    for entry in catalogue.get("packages", []):
        if not entry.get("exported", false):
            continue
        var file := "%s/%s" % [world.generated_dir, entry.get("file", "")]
        var data = JSON.parse_string(FileAccess.get_file_as_string(file)) if FileAccess.file_exists(file) else null
        if data is Dictionary and data.get("format") == "geogen-content" and int(data.get("version", 0)) == 1:
            _content[str(data["package"])] = data
        else:
            push_warning("geogen: bad content package %s" % file)
    return _content


## The package a variable key or a hidden latch belongs to ("world:<pkg>.x", "actor:<pkg>.x",
## "once:<pkg>:<rule>", "watch:<pkg>:<rule>", "<pkg>:thing#state"); "" when it says none.
static func package_of(key: String) -> String:
    if not ":" in key:
        return ""
    var head := key.get_slice(":", 0)
    var rest := key.substr(head.length() + 1)
    if not "#" in key and head in ["world", "actor"]:
        return rest.get_slice(".", 0)
    if head in ["once", "watch"]:
        return rest.get_slice(":", 0) if ":" in rest else ""
    return head


## Keep the applied content packages' values from a save of the scene that is unloading.
## A watch latch of a rule tied to this scene is left behind: the scene's things start
## over when it loads again, so the condition is judged afresh.
func _carry(saved: Dictionary) -> void:
    _carry_into(_carried, saved)


func _carry_into(carried: Dictionary, saved: Dictionary) -> void:
    if _applied.is_empty():
        return
    var names := {}
    var afresh := {}
    for other in _applied:
        names[other["package"]] = true
        for rule in other.get("rules", []):
            if rule.has("scene"):
                afresh["watch:%s" % rule["id"]] = true
    _swap(carried["world"], saved.get("world", {}), names, afresh)
    for actor in saved.get("actors", {}):
        if not carried["actors"].has(actor):
            carried["actors"][actor] = {}
        _swap(carried["actors"][actor], saved["actors"][actor], names, afresh)


static func _swap(kept: Dictionary, values: Dictionary, names: Dictionary, afresh: Dictionary) -> void:
    for key in kept.keys():
        if names.has(package_of(key)):
            kept.erase(key)                 # back at its default: a save only holds what differs
    for key in values:
        if names.has(package_of(key)) and not afresh.has(key):
            kept[key] = values[key]


## The save the loading scene's program starts from: its own from last time ({}: never
## loaded) with the applied content packages' values as they are now, for the actors here.
func _with_carried(saved: Dictionary, here: Array) -> Dictionary:
    if _applied.is_empty():
        return saved
    var names := {}
    for other in _applied:
        names[other["package"]] = true
    var out := saved.duplicate(true)
    var world_values := {}
    for key in out.get("world", {}):
        if not names.has(package_of(key)):
            world_values[key] = out["world"][key]
    for key in _carried["world"]:
        if names.has(package_of(key)):
            world_values[key] = _carried["world"][key]
    var people := {}
    for actor in out.get("actors", {}):
        people[actor] = {}
        for key in out["actors"][actor]:
            if not names.has(package_of(key)):
                people[actor][key] = out["actors"][actor][key]
    for actor in here:
        var theirs: Dictionary = _carried["actors"].get(actor, {})
        for key in theirs:
            if names.has(package_of(key)):
                if not people.has(actor):
                    people[actor] = {}
                people[actor][key] = theirs[key]
    if out.is_empty():
        if world_values.is_empty() and people.is_empty():
            return {}
        out = {"format": GeogenMachine.SAVE_FORMAT, "version": GeogenMachine.SAVE_VERSION}
    out["world"] = world_values
    out["actors"] = people
    return out


## An interaction's node now exists (WorldLoader calls this as a model or chunk loads).
## The program runs from the moment the manifest is read, so a rule may already have driven
## it (a condition true as the scene loads: state a content package brought along): that is
## its state, taken up silently, as the machine has processed the change.
func adopt(key: String, it: GeogenInteraction) -> void:
    if machine == null:
        return
    var lock := key + "_locked"
    if _holding.has(key):           # the machine had it while its chunk was away: what it says now
        _holding.erase(key)
        it.locked = bool(machine.world.get(lock, machine.variables[lock]["default"]))
        it.set_state(str(machine.world.get(key, machine.variables[key]["default"])), true)
        # Its timers (on its way, waiting to move on) are the node's again.
        for timer in machine.timers.duplicate():
            if timer.get("key") == key and timer.get("kind") in ["arrive", "dwell"]:
                machine.timers.erase(timer)
                it.resume(timer, float(timer["at"]) - machine.now)
        return
    if machine.world.has(lock) and (not _at_load.has(lock) or _at_load[lock] != machine.world[lock]):
        it.locked = bool(machine.world[lock])
    if machine.world.has(key) and (not _at_load.has(key) or _at_load[key] != machine.world[key]):
        it.set_state(str(machine.world[key]), true)


## An interaction's node is going (its chunk unloads): the machine holds its state, as the
## node left it, until it comes back (`adopt`), so rules read and drive it meanwhile. Silent:
## nothing changed. False when the program doesn't know it.
## ``progress`` (GeogenInteraction.progress) becomes the machine's timer, so a door on its way
## arrives, and one waiting to close itself closes, while its chunk is away.
func release(key: String, snapshot: Dictionary, progress := {}) -> bool:
    if machine == null or not machine.interactions.has(key):
        return false
    for entry in [[key, snapshot.get("state")], [key + "_locked", snapshot.get("locked")]]:
        if entry[1] == null or entry[1] == machine.variables[entry[0]]["default"]:
            machine.world.erase(entry[0])
        else:
            machine.world[entry[0]] = entry[1]
    if not progress.is_empty():
        var timer := {"kind": progress["kind"], "key": key, "state": progress["state"], "actor": progress["actor"]}
        if progress.has("from"):
            timer["from"] = progress["from"]
        machine._schedule(float(progress["in"]), timer)
    _holding[key] = true
    return true


## After a save is restored: the machine holds every interaction whose node isn't loaded.
func hold_absent() -> void:
    _holding.clear()
    if machine == null:
        return
    for key in machine.interactions:
        if _interaction(key) == null:
            _holding[key] = true


## Whether the machine holds an interaction's state (its node isn't loaded).
func holds(key: String) -> bool:
    return _holding.has(key)


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


func _on_signal(subject: String, signal_name: String, actor: String, data: Dictionary = {}) -> void:
    world.events.post("signal", subject, signal_name, actor, data, "rules")


## Tell an actor something (a rule's notify / play, a refused use): prints `feedback: {...}`.
func tell(kind: String, value: String, actor: String) -> void:
    print("feedback: ", JSON.stringify({"kind": kind, "value": value, "actor": actor}))
    feedback.emit(kind, value, actor)


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


## What a save holds besides the loaded package (docs/rules.md "Saving"): `packages`, the
## other packages visited this session as they were left, and `carried`, the content
## packages' values as they are now. Empty sections are left out.
func save_others() -> Dictionary:
    var out := {}
    var others := {}
    for name_ in _saves:
        if machine == null or name_ != package:
            others[name_] = _saves[name_].duplicate(true)
    if not others.is_empty():
        out["packages"] = others
    var carried := _carried.duplicate(true)
    if machine != null:
        _carry_into(carried, machine.save())
    if not carried["world"].is_empty() or carried["actors"].values().any(func(v): return not v.is_empty()):
        out["carried"] = carried
    return out


## Take up a whole save before the loaded package's part of it is restored: the other
## packages it holds become what they resume from (and what the quest log lists of them),
## the content packages' values are carried again. Returns the part for ``current`` (the
## loaded package, or the scene's name when it has no program): the save's own sections if
## it was made there, else what it holds of ``current`` under `packages`, or a fresh start;
## a save made elsewhere goes under `packages` for when that scene is visited.
func take_save(data: Dictionary, current: String) -> Dictionary:
    var others: Dictionary = (data.get("packages", {}) as Dictionary).duplicate(true)
    var own := data.duplicate(true)
    own.erase("packages")
    own.erase("carried")
    var saved_in := str(data.get("package", current))
    var mine := own
    if saved_in != current:
        own["timers"] = []                      # arrivals of things that aren't here
        others[saved_in] = own
        mine = others.get(current, {"format": GeogenMachine.SAVE_FORMAT, "version": GeogenMachine.SAVE_VERSION,
            "now": 0.0, "world": {}, "actors": {}, "timers": []})
        if own.get("clock") is Dictionary:
            mine["clock"] = own["clock"]        # the world's time is the save's
    others.erase(current)
    _saves = others
    _carried = (data.get("carried", {"world": {}, "actors": {}}) as Dictionary).duplicate(true)
    for section in ["world", "actors"]:
        if not _carried.has(section):
            _carried[section] = {}
    if saved_in != current and machine != null:
        mine = _with_carried(mine, machine.actor_info.keys())
    _reports = {}
    for name_ in _saves:
        _reports[name_] = _saved_reports(name_, _saves[name_])
    return mine


## {actor: report} of a package that isn't loaded, from its manifest and a save of it.
func _saved_reports(name_: String, saved: Dictionary) -> Dictionary:
    var manifest := _manifest_of(name_)
    var listed: Dictionary = manifest.get("quests", {})
    if listed.get("quests", {}).is_empty():
        return {}
    var state: Dictionary = manifest.get("state", {})
    var rules: Dictionary = manifest.get("rules", {})
    var program := {"format": "geogen-rules", "version": 1, "variables": state.get("variables", {}),
        "interactions": state.get("interactions", {}), "bindings": rules.get("bindings", []),
        "rules": rules.get("rules", [])}
    var other := GeogenMachine.new(program, {"player": {"kind": "player"}})
    other.load_save(saved)
    var out := {}
    for actor in other.actor_info:
        out[actor] = _described(_report_of(other, listed, name_, actor), false, listed)
    return out


## The manifest (or chunk index) a package was exported in: {} when there is none.
func _manifest_of(name_: String) -> Dictionary:
    for path in ["%s/%s.manifest.json" % [world.generated_dir, name_],
            "%s/%s_chunks/%s.chunks.json" % [world.generated_dir, name_, name_]]:
        if FileAccess.file_exists(path):
            var data = JSON.parse_string(FileAccess.get_file_as_string(path))
            if data is Dictionary:
                return data
    return {}


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

## Whether an interaction's `when` lets ``actor`` use it now (true with no program or no `when`).
func allows(key: String, actor := "") -> bool:
    return machine == null or not machine.interactions.has(key) or machine.allows(key, actor)


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
    return [] if machine == null else _report_of(machine, quests, package, actor, _locate)


## The packages running now: the loaded scene's and the content packages applied to it.
func running() -> Array:
    return [] if machine == null else [package] + _applied.map(func(other): return other["package"])


## The package a quest of the loaded program belongs to: a content package's, or the scene's.
func quest_package(id: String) -> String:
    return str(quests.get("quests", {}).get(id, {}).get("package", package))


## Where a subject the export didn't mark is in the loaded world (a content package's quest
## names things in the scenes it runs in): its node's position, or null.
func _locate(subject: String) -> Variant:
    var node = world.node_by_uid(subject)
    if not node is Node3D:
        return null
    var box := WorldLoader._aabb(node)
    var at: Vector3 = box.get_center() if box.size != Vector3.ZERO else (node as Node3D).global_position
    return [at.x, at.y, at.z]


static func _report_of(machine: GeogenMachine, quests: Dictionary, package: String, actor: String,
        locate := Callable()) -> Array:
    var out := []
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
            var scenes := []        # the scenes its subjects are in: where a portal leads to them
            for subject in objective["subjects"]:
                var scene := str(subject).get_slice(":", 0)
                if ":" in str(subject) and not scene in scenes:
                    scenes.append(scene)
            var entry := {"id": objective["id"], "text": objective["text"], "optional": objective["optional"],
                "state": state, "scenes": scenes}
            if not objective.get("goals", []).is_empty():
                entry["goals"] = objective["goals"]     # what an NPC holder does for it
            if objective.has("progress"):
                var n = machine.value(objective["progress"]["value"], actor)
                entry["progress"] = [int(n) if n != null else 0, int(objective["progress"]["of"])]
            # A counted objective (count / all / any over its subjects): which subjects are done
            # already, judged one at a time, so markers leave them out.
            var done_subjects := []
            var counted = objective.get("progress", {}).get("value")
            if counted is Dictionary and counted.get("op") in ["count", "all", "any"] and counted.has("where"):
                for subject in counted.get("over", []):
                    var one = machine.value({"op": "count", "over": [subject], "where": counted["where"]}, actor)
                    if one != null and float(one) >= 1.0:
                        done_subjects.append(subject)
                entry["subjects_done"] = done_subjects
            if state == "open" and not objective.get("marker", {}).get("hidden", false):
                var points := []
                for subject in objective["subjects"]:
                    if subject in done_subjects:
                        continue
                    var at: Variant = markers.get(subject)
                    if at == null and locate.is_valid():
                        at = locate.call(subject)
                    if at != null:
                        points.append({"subject": subject, "position": at})
                if not points.is_empty():
                    entry["markers"] = points
                    if objective.get("marker", {}).has("area"):
                        entry["area"] = objective["marker"]["area"]
            objectives.append(entry)
        out.append({"id": id, "package": str(quest.get("package", package)), "title": quest["title"],
            "status": status, "objectives": objectives})
    machine.errors.clear()
    return out


## Every quest of every package this session has loaded, for an actor: the loaded package's
## `report()`, then the others' as they stood when they unloaded. Each entry also has
## `loaded` (its package is the one running) and `description`.
func report_all(actor := "player") -> Array:
    var out := _described(report(actor), true)
    var running := [package] + _applied.map(func(other): return other["package"])
    for other in _reports:
        if machine == null or not other in running:
            out.append_array(_reports[other].get(actor, []).duplicate(true))
    return out


func _described(entries: Array, loaded: bool, listed: Variant = null) -> Array:
    var source: Dictionary = quests if listed == null else listed
    for entry in entries:
        entry["loaded"] = loaded
        entry["description"] = str(source.get("quests", {}).get(entry["id"], {}).get("description", ""))
        for objective in entry["objectives"]:
            if objective.has("markers"):
                objective["marked"] = true      # it has somewhere to point (hud_markers.gd: the portal there)
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
