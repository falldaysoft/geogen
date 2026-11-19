class_name GeogenLife
extends RefCounted
## A life (geogen.111, npc.py `_life`): what stays changed about a person. Needs fill and empty
## again; this doesn't. What they finish doing adds to their `practice`, and their `chapters` are
## the stretches of their life: when a chapter's `until` holds the next begins, and its `becomes`
## changes the numbers their decisions already read (preferences, need curves, their home), for
## good. Nothing here decides anything: a person who has changed simply chooses differently.
##
## `npc.life`: {chapter, practice: {name: 0..1}, story: [{day, time, chapter, text}],
## today: {what: times}, habits: {what: times}, day}. `what` is a practice's name or
## `with/<stable ID>` for someone they spent time with. `habits` is their last full day: while
## the clock is skipped they carry on as they were, so each day skipped counts those again.

const DAILY_CAP := 3            ## times a day doing something counts: a skill takes days, not one long afternoon


static func has(npc: GeogenNpc) -> bool:
    return npc.definition.has("life")


## Called once the NPC is ready: the start of their life, before anything has happened to them.
static func begin(npc: GeogenNpc) -> void:
    if not has(npc):
        return
    npc._life_base = {"preferences": (npc.definition.get("preferences", {}) as Dictionary).duplicate(true),
        "needs": (npc.definition.get("needs", {}) as Dictionary).duplicate(true), "home": npc.home.duplicate()}
    npc.life = {"chapter": 0, "practice": {}, "story": [], "today": {}, "habits": {}, "day": _day(npc)}
    _become(npc)


## What a save keeps.
static func save(npc: GeogenNpc) -> Dictionary:
    return npc.life.duplicate(true)


## Take up a save (none: back to the start of their life).
static func load_save(npc: GeogenNpc, data: Dictionary) -> void:
    if not has(npc):
        return
    npc.life = {"chapter": int(data.get("chapter", 0)), "practice": (data.get("practice", {}) as Dictionary).duplicate(true),
        "story": (data.get("story", []) as Array).duplicate(true), "today": (data.get("today", {}) as Dictionary).duplicate(true),
        "habits": (data.get("habits", {}) as Dictionary).duplicate(true), "day": int(data.get("day", _day(npc)))}
    _become(npc)


## The chapter they are in now.
static func chapter(npc: GeogenNpc) -> Dictionary:
    var chapters: Array = npc.definition["life"]["chapters"]
    return chapters[clampi(int(npc.life.get("chapter", 0)), 0, chapters.size() - 1)]


## What the chapter they are in makes ``option`` worth, whatever their needs.
static func worth(npc: GeogenNpc, option: Dictionary) -> float:
    if not has(npc):
        return 0.0
    var pursues: Dictionary = chapter(npc).get("pursues", {})
    var best := 0.0
    for tag in option.get("tags", []):
        best = maxf(best, float(pursues.get(tag, 0.0)))
    if best > 0.0 and _done_for_today(npc, option):
        return 0.0                  # enough for one day
    return best


## Whether what ``option`` is practice of has counted as often as it can today.
static func _done_for_today(npc: GeogenNpc, option: Dictionary) -> bool:
    _roll(npc)
    var practice: Dictionary = npc.definition["life"]["practice"]
    for name in practice:
        var from: Array = practice[name]["from"]
        if (option.get("tags", []) as Array).any(func(t): return t in from) and int(npc.life["today"].get(name, 0)) < DAILY_CAP:
            return false
    return true


## They finished ``option``: it counts toward whatever it is practice of, and it is what they did today.
static func did(npc: GeogenNpc, option: Dictionary) -> void:
    if not has(npc):
        return
    _roll(npc)
    var practice: Dictionary = npc.definition["life"]["practice"]
    for name in practice:
        var from: Array = practice[name]["from"]
        if (option.get("tags", []) as Array).any(func(t): return t in from):
            if int(npc.life["today"].get(name, 0)) < DAILY_CAP:
                _add(npc, name, float(practice[name]["per_use"]))
            npc.life["today"][name] = int(npc.life["today"].get(name, 0)) + 1
    if option.has("person") and is_instance_valid(option["person"]):
        var what := "with/%s" % WorldLoader.npc_uid(option["person"])
        npc.life["today"][what] = int(npc.life["today"].get(what, 0)) + 1
    check(npc)


## The clock was moved on over days ``from_day`` .. ``to_day``: they kept doing what they do.
static func days_passed(npc: GeogenNpc, from_day: int, to_day: int) -> void:
    if not has(npc) or to_day <= from_day:
        return
    var usual: Dictionary = npc.life["habits"].duplicate()
    for what in npc.life["today"]:
        usual[what] = maxi(int(usual.get(what, 0)), int(npc.life["today"][what]))
    for day in range(from_day + 1, to_day + 1):
        npc.life["day"] = day
        for what in usual:
            _again(npc, str(what), mini(int(usual[what]), DAILY_CAP))
        check(npc, day)
    npc.life["habits"] = usual
    npc.life["today"] = {}


## Move on through every chapter whose `until` now holds.
static func check(npc: GeogenNpc, day := -1) -> void:
    if not has(npc):
        return
    var chapters: Array = npc.definition["life"]["chapters"]
    while int(npc.life["chapter"]) < chapters.size() - 1 and _ready_to_leave(npc, chapter(npc)):
        npc.life["chapter"] = int(npc.life["chapter"]) + 1
        _become(npc)
        var now := chapter(npc)
        for name in now.get("tends", {}):
            _tend(npc, str(name), str(now["tends"][name]))
        var entry := {"day": day if day >= 0 else _day(npc), "time": _time(npc), "chapter": now["name"]}
        if now.has("story"):
            entry["text"] = now["story"]
            npc.life["story"].append(entry)
        var event := {"event": "chapter"}
        event.merge(entry)
        npc._trace(event)


## The report's `life`.
static func report(npc: GeogenNpc) -> Dictionary:
    var now := chapter(npc)
    var practice := {}
    for name in npc.life["practice"]:
        practice[name] = snappedf(float(npc.life["practice"][name]), 0.001)
    return {"chapter": now["name"], "index": int(npc.life["chapter"]), "hopes": now.get("hopes", ""),
        "practice": practice, "story": npc.life["story"].duplicate(true)}


static func _ready_to_leave(npc: GeogenNpc, now: Dictionary) -> bool:
    var until: Dictionary = now.get("until", {})
    if until.is_empty():
        return false
    for name in until.get("practice", {}):
        if float(npc.life["practice"].get(name, 0.0)) < float(until["practice"][name]):
            return false
    for person in until.get("knows", {}):
        var other = _person(npc, str(person))
        if other == null or npc.knows(other) < float(until["knows"][person]):
            return false
    for name in until.get("world", {}):
        if not _shows(npc, str(name), str(until["world"][name])):
            return false
    return true


## Someone by their character file's name, or the end of their stable ID.
static func _person(npc: GeogenNpc, who: String) -> GeogenNpc:
    for other in npc.world.npcs:
        if other != npc and is_instance_valid(other) and (str(other.definition.get("character", "")) == who \
                or WorldLoader.npc_uid(other).get_file() == who or String(other.name) == who):
            return other
    return null


## Whether any interaction called ``name`` is in ``state``.
static func _shows(npc: GeogenNpc, name: String, state: String) -> bool:
    for key in npc.world.interactions_by_key:
        if str(key).ends_with("#" + name) and npc.world.interactions_by_key[key] is GeogenInteraction \
                and (npc.world.interactions_by_key[key] as GeogenInteraction).state == state:
            return true
    return false


static func _tend(npc: GeogenNpc, name: String, state: String) -> void:
    for key in npc.world.interactions_by_key:
        if str(key).ends_with("#" + name) and npc.world.interactions_by_key[key] is GeogenInteraction:
            (npc.world.interactions_by_key[key] as GeogenInteraction).set_state(state, false, WorldLoader.actor_id(npc))


## Make the definition what the chapters up to this one have made it, from how they started.
static func _become(npc: GeogenNpc) -> void:
    npc.definition["preferences"] = (npc._life_base["preferences"] as Dictionary).duplicate(true)
    npc.definition["needs"] = (npc._life_base["needs"] as Dictionary).duplicate(true)
    npc.home = (npc._life_base["home"] as PackedVector2Array).duplicate()
    var chapters: Array = npc.definition["life"]["chapters"]
    for i in int(npc.life["chapter"]) + 1:
        var changes: Dictionary = chapters[i].get("becomes", {})
        (npc.definition["preferences"] as Dictionary).merge(changes.get("preferences", {}), true)
        for need in changes.get("needs", {}):
            (npc.definition["needs"][need] as Dictionary).merge(changes["needs"][need], true)
        for rect in changes.get("home", {}).get("rects", []):
            npc.home = _joined(npc.home, rect)


## ``home`` with a rectangle joined on (it must touch it: a home is one region).
static func _joined(home: PackedVector2Array, rect: Array) -> PackedVector2Array:
    var box := PackedVector2Array([Vector2(rect[0], rect[1]), Vector2(rect[2], rect[1]),
        Vector2(rect[2], rect[3]), Vector2(rect[0], rect[3])])
    if home.size() < 3:
        return box
    var merged := Geometry2D.merge_polygons(home, box)
    return merged[0] if merged.size() == 1 else home      # (apart from it: it can't be joined on)


static func _add(npc: GeogenNpc, name: String, amount: float) -> void:
    npc.life["practice"][name] = clampf(float(npc.life["practice"].get(name, 0.0)) + amount, 0.0, 1.0)


## A day of doing ``what`` ``times`` times, unseen.
static func _again(npc: GeogenNpc, what: String, times: int) -> void:
    if what.begins_with("with/"):
        var uid := what.trim_prefix("with/")
        for other in npc.world.npcs:
            if is_instance_valid(other) and WorldLoader.npc_uid(other) == uid:
                for i in times:
                    npc._know_better(other, 1.0)
        return
    var practice: Dictionary = npc.definition["life"]["practice"]
    if practice.has(what):
        _add(npc, what, float(practice[what]["per_use"]) * times)


## A new day has come by itself: what they did yesterday is what they usually do.
static func _roll(npc: GeogenNpc) -> void:
    var day := _day(npc)
    if day != int(npc.life["day"]):
        if not (npc.life["today"] as Dictionary).is_empty():
            npc.life["habits"] = npc.life["today"]
        npc.life["today"] = {}
        npc.life["day"] = day


static func _day(npc: GeogenNpc) -> int:
    return npc.world.clock.day if npc.world != null and npc.world.clock != null else 0


static func _time(npc: GeogenNpc) -> String:
    return npc.world.clock.label() if npc.world != null and npc.world.clock != null else ""
