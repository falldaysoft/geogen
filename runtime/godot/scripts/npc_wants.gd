class_name GeogenWants
extends RefCounted
## Wants (geogen.97, npc.py WANT_KEYS): conditions on the world an NPC puts right with whatever
## is there. Each subject a want is unhappy about is one more option in the NPC's decision,
## scored beside its affordances and activities; choosing it, the NPC picks a place for the
## subject among the spots the world offers (GeogenPlaceSearch) by its own weights. Nothing
## here names a behaviour: a want says what it is about, when that is a problem, and the
## action to run.

const TRIED_NEAR := 0.3          ## m: a spot this close to one already used for the subject


## The options ``npc``'s wants add now: one per subject that is a problem, in its home region,
## that it can carry and nobody else is seeing to.
static func options(npc: GeogenNpc) -> Array[Dictionary]:
    var out: Array[Dictionary] = []
    var wants: Dictionary = npc.definition.get("wants", {})
    if wants.is_empty() or npc.world.props == null or npc.prop_in_hand() != "":
        return out
    var margin := float(npc.definition.get("home_margin", 1.0))
    for want_name in wants:
        var want: Dictionary = wants[want_name]
        if str(want.get("of", "")) != "prop":
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
            out.append({"id": id, "action": want["action"], "want": want_name, "subject": uid,
                "advertises": want.get("advertises", {}), "tags": [], "at": at, "base": float(want.get("weight", 0.5))})
    return out


static func _is_problem(npc: GeogenNpc, uid: String, when: Dictionary) -> bool:
    if when.has("resting_on"):
        return GeogenPlaceSearch.resting_on(npc.world, uid).get("kind", "none") == str(when["resting_on"])
    return false


static func _subject_key(uid: String) -> String:
    return "subject/%s" % uid


## Take ``option`` (one of `options`): choose where its subject goes and keep others off it.
## Returns the action's context, or {} when the world offers no place.
static func begin(npc: GeogenNpc, option: Dictionary) -> Dictionary:
    var ranked := places(npc, str(option["want"]), str(option["subject"]))
    if ranked.is_empty():
        return {}
    var place: Dictionary = ranked[0]
    option["place"] = place
    npc.world.npc_reserve(_subject_key(option["subject"]), npc)
    npc._trace({"event": "want", "want": option["want"], "subject": option["subject"], "place": place["uid"],
        "at": GeogenNpc._v(place["position"]), "score": snappedf(place["score"], 0.001), "of": ranked.size()})
    return {"anchor": npc.global_position, "yaw_deg": rad_to_deg(npc._yaw()), "approach": npc.global_position,
        "subject": option["subject"], "place": place}


## The option is over (``ok``: its action ran to the end): free its subject, and remember the spot
## it was put at, so a spot the subject fell off isn't the first choice again.
static func finish(npc: GeogenNpc, option: Dictionary, ok: bool) -> void:
    npc.world.npc_release(_subject_key(option["subject"]), npc)
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


## Whether a dotted tag (`furniture.table`) is, or is a kind of, ``key`` (`table`, `furniture`).
static func _tag_is(tag: String, key: String) -> bool:
    return tag == key or tag.begins_with(key + ".") or tag.ends_with("." + key) or ("." + key + ".") in tag
