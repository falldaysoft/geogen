class_name GeogenMemory
extends RefCounted
## Memory (geogen.99.2, npc.py MEMORY_DEFAULTS): what an NPC with `memory:` knows of the props
## around it. It knows a prop it has seen (where, and what it rested on then) or heard land (a
## stimulus it reacts to: only where), for `forget` s after it last perceived it. Every `every` s
## it looks round: what it sees it remembers afresh, and a remembered spot it can see where the
## thing no longer is it forgets. Its wants about props are then about what it remembers
## (npc_wants.gd); a thing it only heard, or that isn't where it was, it goes to look for first.
##
## An entry: {at: Vector3 (the thing's middle), clock, by: seen | heard, resting: floor | thing |
## held | "" (not known: heard, or seen on the move)}.

const MOVED := 0.3          ## m: a thing further than this from where it is remembered isn't there


## Whether ``npc`` has a memory (else its wants know everything in its home region, as before).
static func has(npc: GeogenNpc) -> bool:
    return npc.definition.has("memory")


## Called every step: look round when it is time, and let old entries go.
static func step(npc: GeogenNpc) -> void:
    var cfg: Dictionary = npc.definition["memory"]
    if npc.clock < npc._looked_at + float(cfg["every"]) or npc.world.props == null:
        return
    npc._looked_at = npc.clock
    var seen := {}
    for uid in npc.world.props.props:
        var at: Vector3 = npc._prop_centre(uid)
        if npc.world.props.holder_of(uid) == npc or npc._sees(at, [npc.world.props.props[uid]["body"].get_rid()]):
            remember(npc, uid, at, "seen", _resting(npc, uid))     # (what it holds it knows of)
            seen[uid] = true
    var forget := float(cfg["forget"])
    for uid in npc._memory.keys():
        var entry: Dictionary = npc._memory[uid]
        if npc.clock - float(entry["clock"]) > forget or not npc.world.props.props.has(uid):
            npc._memory.erase(uid)
        elif not seen.has(uid) and npc._sees(entry["at"], [npc.world.props.props[uid]["body"].get_rid()]):
            npc._memory.erase(uid)              # it can see the spot, and the thing isn't there
            npc._trace({"event": "forget", "subject": uid, "at": GeogenUtil.cm(entry["at"])})


## What prop ``uid`` rests on, as far as one looking at it can tell.
static func _resting(npc: GeogenNpc, uid: String) -> String:
    if npc.world.props.holder_of(uid) != null:
        return "held"
    var body: RigidBody3D = npc.world.props.props[uid]["body"]
    if body.linear_velocity.length() > 0.1:
        return ""                                # on the move: where it ends up isn't known yet
    return str(GeogenPlaceSearch.resting_on(npc.world, uid).get("kind", ""))


static func remember(npc: GeogenNpc, uid: String, at: Vector3, by: String, resting := "") -> void:
    var was: Dictionary = npc._memory.get(uid, {})
    npc._memory[uid] = {"at": at, "clock": npc.clock, "by": by, "resting": resting}
    if was.is_empty():
        npc._trace({"event": "remember", "subject": uid, "by": by, "at": GeogenUtil.cm(at), "resting": resting})


## A stimulus it noticed (npc.gd `_on_event`) whose subject is a prop: heard, it knows only where
## the noise was; if it can see the thing now, it knows what it rests on as well.
static func noticed(npc: GeogenNpc, uid: String, point: Vector3) -> void:
    if npc.world.props == null or not npc.world.props.props.has(uid):
        return
    var at: Vector3 = npc._prop_centre(uid)
    if npc._sees(at, [npc.world.props.props[uid]["body"].get_rid()]):
        remember(npc, uid, at, "seen", _resting(npc, uid))
    else:
        remember(npc, uid, point, "heard", "")


## Whether a want about props has to go and look before it can act on what ``entry`` says of
## prop ``uid``: only heard, or seen somewhere it no longer is.
static func needs_look(npc: GeogenNpc, uid: String, entry: Dictionary) -> bool:
    return entry["resting"] == "" or npc._prop_centre(uid).distance_to(entry["at"]) > MOVED


## The context of the `search` action: go to where the thing is remembered and look there.
static func search_context(npc: GeogenNpc, entry: Dictionary) -> Dictionary:
    var at: Vector3 = entry["at"]
    var stand = npc.stand_for(at)
    if stand == null:
        stand = NavigationServer3D.map_get_closest_point(npc.get_world_3d().navigation_map, at)
    return {"anchor": at, "yaw_deg": rad_to_deg(npc._yaw()), "approach": stand}


## A search is over: what it saw there is in its memory now. What it still can't act on (only
## heard of, or not where it was: out of sight from there) it lets go.
static func searched(npc: GeogenNpc, uid: String) -> void:
    var entry: Dictionary = npc._memory.get(uid, {})
    if not entry.is_empty() and npc.world.props.props.has(uid) and needs_look(npc, uid, entry):
        npc._memory.erase(uid)
        npc._trace({"event": "forget", "subject": uid, "at": GeogenUtil.cm(entry["at"])})


## The report's `memory`: {uid: {at, by, resting, age}}.
static func report(npc: GeogenNpc) -> Dictionary:
    var out := {}
    for uid in npc._memory:
        var entry: Dictionary = npc._memory[uid]
        out[uid] = {"at": GeogenUtil.cm(entry["at"]), "by": entry["by"], "resting": entry["resting"],
            "age": snappedf(npc.clock - float(entry["clock"]), 0.1)}
    return out


## A save's `memory` (npc.gd save_state): {uid: {at: [x, y, z], by, resting, age}}, by the props'
## stable IDs, so a loaded game knows what it knew. Ages count on from the loading NPC's clock.
static func save(npc: GeogenNpc) -> Dictionary:
    var out := {}
    for uid in npc._memory:
        var entry: Dictionary = npc._memory[uid]
        var at: Vector3 = entry["at"]
        out[uid] = {"at": [snappedf(at.x, 0.001), snappedf(at.y, 0.001), snappedf(at.z, 0.001)],
            "by": entry["by"], "resting": entry["resting"], "age": snappedf(npc.clock - float(entry["clock"]), 0.1)}
    return out


## Take up a save's `memory`, in place of what it remembered (props the world no longer has are left out).
static func load_save(npc: GeogenNpc, saved: Dictionary) -> void:
    npc._memory = {}
    for uid in saved:
        if npc.world.props != null and not npc.world.props.props.has(uid):
            continue
        var entry: Dictionary = saved[uid]
        var at: Array = entry.get("at", [0, 0, 0])
        npc._memory[uid] = {"at": Vector3(at[0], at[1], at[2]), "clock": npc.clock - float(entry.get("age", 0.0)),
            "by": str(entry.get("by", "seen")), "resting": str(entry.get("resting", ""))}
