class_name GeogenWants
extends RefCounted
## Wants (geogen.97, npc.py WANT_KEYS): conditions on the world an NPC puts right with whatever
## is there. Each subject a want is unhappy about is one more option in the NPC's decision,
## scored beside its affordances and activities. Nothing here names a behaviour: a want says
## what it is about, when that is a problem, and the action to run.
##
## A want about props (`of: prop`): choosing it, the NPC picks a place for the subject among
## the spots the world offers (GeogenPlaceSearch) by its own weights.
## A want about things (`of` a reference; the export wrote `subjects`: [{subject, interaction}]):
## a subject is a problem while its interaction shows `when.state` (and, with `room: empty`,
## nobody is in the room it is in); the action goes to the thing and works it to `to`.

const ROOM_PROBE := 1.0          ## m above the floor a point is asked which room it is in
const TRIED_NEAR := 0.3          ## m: a spot this close to one already used for the subject


## The options ``npc``'s wants add now: one per subject that is a problem, in its home region,
## that it can carry and nobody else is seeing to.
static func options(npc: GeogenNpc) -> Array[Dictionary]:
    var out: Array[Dictionary] = []
    var wants: Dictionary = npc.definition.get("wants", {})
    if wants.is_empty() or npc.prop_in_hand() != "":
        return out
    var margin := float(npc.definition.get("home_margin", 1.0))
    npc._wants_left = []
    for want_name in wants:
        var want: Dictionary = wants[want_name]
        if str(want.get("of", "")) != "prop":
            _thing_options(npc, want_name, want, margin, out)
            continue
        if npc.world.props == null:
            continue
        for uid in npc.world.props.props:
            var id := "want/%s/%s" % [want_name, uid]
            if npc.clock < float(npc._retry_at.get(id, -1.0)) or not npc.world.props.can_take(uid):
                continue
            if npc.world.npc_reserved(_subject_key(uid), npc) > 0 or npc._carry_limit(uid) != "":
                continue
            var at: Vector3 = npc._prop_centre(uid)
            if npc.home_distance(at) > margin or absf(at.y - npc.home_y) > GeogenNpc.FLOOR_TOLERANCE:
                continue
            if not _is_problem(npc, uid, want.get("when", {})):
                continue
            # Reluctance: each time it has had this subject in hand takes some of the want away,
            # and with none left the subject is left be (not a failure: it has had enough).
            var base := float(want.get("weight", 0.5)) - reluctance(npc, want, uid)
            if base <= 0.0:
                npc._wants_left.append(id)
                continue
            out.append({"id": id, "action": want["action"], "want": want_name, "subject": uid,
                "advertises": want.get("advertises", {}), "tags": [], "at": at, "base": base})
    return out


## A want about things: one option for each of its subjects that is a problem now and whose
## place to stand is in the NPC's home region.
static func _thing_options(npc: GeogenNpc, want_name: String, want: Dictionary, margin: float, out: Array[Dictionary]) -> void:
    var when: Dictionary = want.get("when", {})
    for entry in want.get("subjects", []):
        var key := "%s#%s" % [entry["subject"], entry["interaction"]]
        var id := "want/%s/%s" % [want_name, key]
        if npc.clock < float(npc._retry_at.get(id, -1.0)) or npc.world.npc_reserved(_subject_key(key), npc) > 0:
            continue
        var it = npc.world.interactions_by_key.get(key)
        if not it is GeogenInteraction or not is_instance_valid(it):
            continue
        if it.state != str(when.get("state", "")) or it.target != it.state or it.locked:
            continue                    # not showing the state, on its way to another, or not its to work
        var ctx := npc._goal_context({"subject": entry["subject"], "interaction": entry["interaction"], "state": want["to"]})
        if ctx.is_empty():
            continue
        var at: Vector3 = ctx["approach"]
        if npc.home_distance(at) > margin or absf(at.y - npc.home_y) > GeogenNpc.FLOOR_TOLERANCE:
            continue
        if when.has("room") and not _room_is(npc, at, str(when["room"])):
            continue
        var base := float(want.get("weight", 0.5)) - reluctance(npc, want, key)
        if base <= 0.0:
            npc._wants_left.append(id)
            continue
        out.append({"id": id, "action": want["action"], "want": want_name, "subject": key, "thing": ctx,
            "advertises": want.get("advertises", {}), "tags": [], "at": at, "base": base})


## Whether the room a thing is worked from (``at``: where one stands to use it) is ``how``:
## `empty` is no character in that room volume, this NPC included. Outside every room: no.
static func _room_is(npc: GeogenNpc, at: Vector3, how: String) -> bool:
    var room := npc.world.room_at(at + Vector3.UP * ROOM_PROBE)
    if room == "" or how != "empty":
        return false
    for c in npc._characters():
        if npc.world.room_at((c["pos"] as Vector3) + Vector3.UP * ROOM_PROBE) == room:
            return false
    return true


## How many times ``npc`` has picked prop ``uid`` up, each fading to nothing over ``forget`` s.
static func handled(npc: GeogenNpc, uid: String, forget: float) -> float:
    var total := 0.0
    for t in npc._handled.get(uid, []):
        total += clampf(1.0 - (npc.clock - float(t)) / maxf(forget, 1.0), 0.0, 1.0)
    return total


## The utility ``want`` has lost for prop ``uid``: its `reluctance` for each of those times.
static func reluctance(npc: GeogenNpc, want: Dictionary, uid: String) -> float:
    return float(want.get("reluctance", 0.1)) * handled(npc, uid, float(want.get("forget", 600.0)))


## ``npc`` has prop ``uid`` in hand (for a want or anything else), or has worked a thing for a
## want (``uid``: `<thing>#<interaction>`): one more handling.
static func note_handled(npc: GeogenNpc, uid: String) -> void:
    var times: Array = npc._handled.get(uid, [])
    times.append(npc.clock)
    npc._handled[uid] = times.slice(-16)


static func _is_problem(npc: GeogenNpc, uid: String, when: Dictionary) -> bool:
    if when.has("resting_on"):
        return GeogenPlaceSearch.resting_on(npc.world, uid).get("kind", "none") == str(when["resting_on"])
    return false


static func _subject_key(uid: String) -> String:
    return "subject/%s" % uid


## Take ``option`` (one of `options`): choose where its subject goes and keep others off it.
## Returns the action's context, or {} when the world offers no place.
static func begin(npc: GeogenNpc, option: Dictionary) -> Dictionary:
    if option.has("thing"):
        var about: Dictionary = npc.definition["wants"][option["want"]]
        npc.world.npc_reserve(_subject_key(option["subject"]), npc)
        npc._want_doing = {"want": option["want"], "subject": option["subject"], "to": about["to"],
            "at": GeogenNpc._v(option["thing"]["anchor"]),
            "reluctance": snappedf(reluctance(npc, about, str(option["subject"])), 0.001)}
        var thing_event := {"event": "want"}
        thing_event.merge(npc._want_doing)
        npc._trace(thing_event)
        return option["thing"]
    var ranked := places(npc, str(option["want"]), str(option["subject"]))
    if ranked.is_empty():
        return {}
    var place: Dictionary = ranked[0]
    option["place"] = place
    npc.world.npc_reserve(_subject_key(option["subject"]), npc)
    var want: Dictionary = npc.definition["wants"][option["want"]]
    npc._want_doing = {"want": option["want"], "subject": option["subject"], "place": place["uid"],
        "at": GeogenNpc._v(place["position"]), "score": snappedf(place["score"], 0.001), "of": ranked.size(),
        "reluctance": snappedf(reluctance(npc, want, str(option["subject"])), 0.001)}
    var event := {"event": "want"}
    event.merge(npc._want_doing)
    npc._trace(event)
    return {"anchor": npc.global_position, "yaw_deg": rad_to_deg(npc._yaw()), "approach": npc.global_position,
        "subject": option["subject"], "place": place}


## The option is over (``ok``: its action ran to the end): free its subject, and remember the spot
## it was put at, so a spot the subject fell off isn't the first choice again.
static func finish(npc: GeogenNpc, option: Dictionary, ok: bool) -> void:
    npc.world.npc_release(_subject_key(option["subject"]), npc)
    npc._want_doing = {}
    if ok and option.has("thing"):
        note_handled(npc, str(option["subject"]))     # worked it: once more, and it tires of this one
        npc.stats["wants"] = int(npc.stats.get("wants", 0)) + 1
    if ok and option.has("place"):
        var tried: Array = npc._tried.get(option["subject"], [])
        tried.append(option["place"]["position"])
        npc._tried[option["subject"]] = tried.slice(-8)
        npc.stats["wants"] = int(npc.stats.get("wants", 0)) + 1


## The places ``npc`` could put prop ``uid`` for its want ``want_name``, best first: the spots that
## hold its box in the NPC's home region (or `place.radius` m round it), at heights its hands get
## to, with somewhere to stand in reach: [{position, approach, uid, tags, height, score}].
static func places(npc: GeogenNpc, want_name: String, uid: String) -> Array:
    var want: Dictionary = (npc.definition.get("wants", {}) as Dictionary).get(want_name, {})
    var cfg: Dictionary = want.get("place", {})
    var body: RigidBody3D = npc.world.props.props[uid]["body"]
    var box: Vector3 = npc.world.props.box_of(uid)["size"]
    var side := maxf(box.x, box.z)                       # it may be set down turned any way
    var heights: Array = (npc.definition.get("reach", {}) as Dictionary).get("heights", [0.0, 1.5])
    var feet := npc.home_y if npc.home.size() >= 3 else npc.global_position.y
    var exclude: Array[RID] = [body.get_rid()]
    var spots := GeogenPlaceSearch.spots(npc.world, npc.get_world_3d().direct_space_state,
        npc.get_world_3d().navigation_map, Vector3(side, box.y, side), npc.global_position, npc.home,
        float(cfg.get("radius", 6.0)), feet + float(heights[0]), feet + float(heights[1]), exclude)
    var prefer: Dictionary = cfg.get("prefer", {})
    var tried: Array = npc._tried.get(uid, [])
    var from: Vector3 = npc._prop_centre(uid)
    var out := []
    for spot in spots:
        var approach = npc.stand_for(spot["position"])
        if approach == null:
            continue
        var score := 1.0
        for key in prefer:
            if (spot["tags"] as Array).any(func(t): return _tag_is(str(t), str(key))):
                score *= float(prefer[key])
        score -= float(cfg.get("distance", 0.1)) * from.distance_to(approach)
        score -= float(cfg.get("off_height", 0.3)) * absf(float(spot["height"]) - feet - float(cfg.get("height", 0.9)))
        if tried.any(func(p): return (p as Vector3).distance_to(spot["position"]) < TRIED_NEAR):
            score -= float(cfg.get("tried", 1.0))
        score += float(cfg.get("noise", 0.1)) * npc.rng.randf()
        spot["approach"] = approach
        spot["score"] = score
        out.append(spot)
    out.sort_custom(func(a, b): return a["score"] > b["score"])
    return out


## For the NPC's report: the problems its wants saw at its last decision ({id, score}, best
## first), the ones it has had enough of (ids), the one it is putting right now (the `want`
## trace event: {want, subject, place, at, score, of, reluctance}; {} for none) and how many
## times lately it has handled each prop ({uid: n}, fading).
static func report(npc: GeogenNpc) -> Dictionary:
    var forget := 600.0
    for want in (npc.definition.get("wants", {}) as Dictionary).values():
        forget = maxf(forget, float(want.get("forget", 600.0)))
    var held := {}
    for uid in npc._handled:
        var n := handled(npc, uid, forget)
        if n > 0.0:
            held[uid] = snappedf(n, 0.01)
    return {"seen": npc._wants_seen, "left": npc._wants_left, "doing": npc._want_doing, "handled": held}


## Whether a dotted tag (`furniture.table`) is, or is a kind of, ``key`` (`table`, `furniture`).
static func _tag_is(tag: String, key: String) -> bool:
    return tag == key or tag.begins_with(key + ".") or tag.ends_with("." + key) or ("." + key + ".") in tag
