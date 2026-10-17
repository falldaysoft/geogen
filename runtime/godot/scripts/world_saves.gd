class_name GeogenWorldSaves
extends RefCounted
## Saving and restoring a world (WorldLoader.saves; docs/rules.md "Saving",
## docs/schema/geogen-save.v1.schema.json): the rules program's sections plus every interaction,
## physics prop and spawned item, and the state of interactions in chunks that aren't loaded.

var world: WorldLoader
## Saved interaction state for things that aren't loaded right now (streamed chunks), by save key.
var pending: Dictionary = {}


## Keep ``it``'s state while its chunk is away (when the rules program doesn't hold it).
func remember(it: GeogenInteraction) -> void:
    pending[state_key(it)] = it.snapshot()


const SAVE_FORMAT := "geogen-save"       # docs/schema/geogen-save.v1.schema.json
const RULES_SAVE := "geogen:rules"       # where saves from before that format kept the rules program's state


static func state_key(it: GeogenInteraction) -> String:
    var uid := WorldLoader.uid_of(it.asset)
    if uid == "":
        return "%s/%s" % [it.asset.name, it.interaction_name]
    return "%s#%s" % [uid, it.interaction_name]


## {state, locked} of every interaction, by ``state_key``: the loaded ones, plus what was
## last known of those in chunks that aren't loaded.
func interaction_states() -> Dictionary:
    var data := pending.duplicate(true)
    for it in world.interactions:
        data[state_key(it)] = it.snapshot()
    return data


## The game as a save (docs/rules.md "Saving"): the rules program's world and actor sections,
## clock and timers, with every interaction in ``world`` as the instance state it is ("<key>":
## its state, "<key>_locked": its lock). Only what differs from how things start is there.
func save() -> Dictionary:
    var data := world.rules.save()
    if data.is_empty():                  # no rules program: only interactions
        data = {"format": SAVE_FORMAT, "version": 1, "now": 0.0, "clock": _clock_now(), "world": {},
            "actors": {}, "actor_info": {}, "timers": []}
    data["package"] = world.rules.package if world.rules.machine != null else world.scene_name
    data.merge(world.rules.save_others())      # the other packages visited, and content packages' values
    var stored: Dictionary = data["world"]
    var live := _live_interactions()
    var states := interaction_states()
    for key in states:
        var snap: Dictionary = states[key]
        var it: GeogenInteraction = live.get(key)
        _store(stored, key, snap.get("state"), it.initial if it != null else _default_of(key))
        _store(stored, key + "_locked", snap.get("locked"),
            it.initial_locked if it != null else _default_of(key + "_locked"))
    # How far loaded interactions had got (a door on its way, one waiting to close itself), as
    # the timers the rules program keeps for the ones it holds.
    var timers: Array = data.get("timers", [])
    var now := float(data.get("now", 0.0))
    var seq := 0
    for timer in timers:
        seq = maxi(seq, int(timer["seq"]))
    for key in live:
        var progress: Dictionary = live[key].progress()
        if progress.is_empty():
            continue
        seq += 1
        var timer := {"at": snappedf(now + float(progress["in"]), 0.000001), "seq": seq,
            "kind": progress["kind"], "key": key, "state": progress["state"], "actor": progress["actor"]}
        if progress.has("from"):
            timer["from"] = progress["from"]
        timers.append(timer)
    timers.sort_custom(GeogenMachine._sooner)
    data["timers"] = timers
    var bodies := world.props.save()
    if not bodies.is_empty():
        data["bodies"] = bodies
    var items := world.flows.save()
    if not items.is_empty():
        data["flows"] = items
    return data


func _live_interactions() -> Dictionary:
    var live := {}
    for it in world.interactions:
        live[state_key(it)] = it
    return live


static func _store(stored: Dictionary, key: String, value: Variant, default: Variant) -> void:
    if value == null or (default != null and value == default):
        stored.erase(key)
    else:
        stored[key] = value


## What the package says a variable starts as (null: it doesn't say).
func _default_of(key: String) -> Variant:
    if world.rules.machine == null:
        return null
    return world.rules.machine.variables.get(key, {}).get("default")


func _clock_now() -> Dictionary:
    if world.clock == null:
        return {"time": "12:00", "hour": 12.0, "night": false, "day": 0}
    var total := int(world.clock.hours * 60.0) % 1440
    return {"time": "%02d:%02d" % [total / 60, total % 60], "hour": total / 60.0, "night": world.clock.night, "day": 0}


## Restore a save: whatever it doesn't mention goes back to how it started. Entries for
## things that aren't loaded yet apply when they load. A save from before the format (a flat
## {key: {state, locked}}, the rules state under "geogen:rules") still loads; it only moves
## what it names.
func load(data: Dictionary) -> void:
    if data.get("format") != SAVE_FORMAT:
        pending = {}
        for key in data:
            if key != RULES_SAVE and data[key] is Dictionary:
                pending[key] = data[key].duplicate(true)
        for it in world.interactions:
            restore_pending(it)
        if data.get(RULES_SAVE) is Dictionary:
            world.rules.load_save(data[RULES_SAVE])
        return
    # Every package it holds: the loaded one's part is restored here, the rest wait for their
    # scene. A save made in another scene loads this one as that save left it.
    data = world.rules.take_save(data, world.rules.package if world.rules.machine != null else world.scene_name)
    if world.clock != null and data.get("clock") is Dictionary and data["clock"].has("hour"):
        world.clock.hours = fposmod(float(data["clock"]["hour"]), 24.0)
        world.clock.apply()
    # The interactions in the world section. A loaded one is its node's to hold; the others
    # wait for their chunk, and the rules program simulates them meanwhile.
    var live := _live_interactions()
    var for_rules: Dictionary = data.duplicate(true)
    var stored: Dictionary = for_rules.get("world", {})
    pending = {}
    for key in stored.keys():
        var owner_key := _saved_interaction(key, stored[key])
        if owner_key == "":
            continue
        if not pending.has(owner_key):
            pending[owner_key] = {}
        pending[owner_key]["state" if owner_key == key else "locked"] = stored[key]
        if live.has(owner_key) or world.rules.machine == null or not world.rules.machine.variables.has(key):
            stored.erase(key)
    for key in live:
        var it: GeogenInteraction = live[key]
        var snap: Dictionary = pending.get(key, {})
        pending.erase(key)
        it.restore({"state": snap.get("state", it.initial), "locked": snap.get("locked", it.initial_locked)})
    world.props.restore(data.get("bodies", {}))     # after their spawn interactions: where they had got to
    # Loaded interactions carry on from their own timers; the program keeps the rest (rule
    # timers, and those of interactions it holds: ones it doesn't know are dropped).
    var kept := []
    var now := float(for_rules.get("now", 0.0))
    for timer in for_rules.get("timers", []):
        var key := str(timer.get("key", ""))
        if timer.get("kind") in ["arrive", "dwell"] and live.has(key):
            live[key].resume(timer, float(timer["at"]) - now)
        elif timer.get("kind") == "rule" or (world.rules.machine != null and world.rules.machine.interactions.has(key)):
            kept.append(timer)
    for_rules["timers"] = kept
    world.rules.load_save(for_rules)           # after the nodes: bindings are recomputed from them
    world.rules.hold_absent()                  # the program has the rest until their chunks load
    world.flows.restore(data.get("flows", {}))     # spawned items, after the program has its counts
    for key in pending.keys():
        if world.rules.holds(key):
            pending.erase(key)


## Write a save as one JSON file; "" when done, else why not.
static func write(path: String, data: Dictionary) -> String:
    var f := FileAccess.open(path, FileAccess.WRITE)
    if f == null:
        return "can't write %s: %s" % [path, error_string(FileAccess.get_open_error())]
    f.store_string(JSON.stringify(data, "  ", true, true))     # floats round-trip exactly
    f.close()
    return ""


## The save in a file (a Dictionary), or null.
static func read(path: String) -> Variant:
    if not FileAccess.file_exists(path):
        return null
    return JSON.parse_string(FileAccess.get_file_as_string(path))


## The interaction a key of a save's world section belongs to ("" for a variable or a
## latch): the key itself for its state, the key less "_locked" for its lock.
func _saved_interaction(key: String, value: Variant) -> String:
    var base := key.trim_suffix("_locked")
    if world.rules.machine != null:
        if world.rules.machine.interactions.has(key):
            return key
        if base != key and world.rules.machine.interactions.has(base):
            return base
        if world.rules.machine.variables.has(key):
            return ""
    if key.begins_with("world:") or key.begins_with("once:") or key.begins_with("watch:"):
        return ""
    for it in world.interactions:              # an export without stable IDs or a state section
        var own := state_key(it)
        if own == key or own == base:
            return own
    if world.rules.machine != null:
        return ""
    return base if value is bool and base != key else key


func restore_pending(it: GeogenInteraction) -> void:
    for key in [state_key(it), "%s/%s" % [it.asset.name, it.interaction_name]]:   # the second: older saves
        if pending.has(key):
            it.restore(pending[key])
            pending.erase(key)
            return
