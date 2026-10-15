class_name GeogenFlows
extends Node
## Spawned items (geogen/flows.py): sources make copies of item templates while the game
## runs, and the copies are taken away again.
##
## On load (`setup`, before anything else looks at the export) every item template
## (extras.geogen.item) is taken out of the world and kept; every source (extras.geogen.flows
## .source) is collected. Each physics frame a running source makes items `rate` a second,
## and at once for each one `queued`, while it has fewer than `max_alive` in the world and
## has made fewer than `count` (0: no limit). It reads `running` and `queued` from the rules
## program and drives `emitted` and `alive` there, and raises `emitted {item}` on the event bus.
##
## A source waits while its last item is still within two of its radii of the spawn point (two
## bodies made in one place would fly apart).
##
## A physics item is a clone prop (props.gd `add_clone`): carried like any prop. An item is
## taken away when a sink takes it in (`take`), when it outlives `lifetime`, when a physics
## item has rested `stuck` seconds, or when it falls out of the world.
##
## Tracks (extras.geogen.flows.track) carry belt and walk items. A source whose spawn point is
## within JOIN m of a track's start puts its items there (one a frame, when the start is clear);
## each frame every running track moves its items, leaders first: belt items at the track's
## speed, walk items at their own, none closer to the one ahead than the gap (`spacing`, or
## the two radii). At the end an item goes on to the track starting there (`end: auto`), is
## taken in (a sink: the track's `received`, signal `received {item, ...data}`), waits
## (`stop`) or falls off as a physics body (`drop`). geogen/flows.py `simulate` is the
## reference: sources and tracks are stepped in the same order (by position), and a source's
## pick among its items is the same smooth weighted round robin, so both agree.
##
## Owned by WorldLoader (`world.flows`); the control API's `flows.list` reports it.

## An item was made or taken away (why: sink | lifetime | stuck | fell | cleared).
signal item_added(id: int)
signal item_removed(id: int, why: String)

const STILL_SPEED := 0.05       ## m/s: slower than this, a physics item is at rest
## Physics frames sources wait after the world loads: bodies (the player's) sit at the origin
## for a frame or two before they go where they start.
const SETTLE_FRAMES := 5
const JOIN := 0.5                ## m: a source or track end this close to a track's start feeds it

var world: WorldLoader
## uid -> {uid, node, data, templates: {item name: Node3D}, rng, due, made, at (world), track}
var sources := {}
## [{uid, node, data, points (world), cum, length, next, received}], in stepping order
var tracks: Array = []
## Physics frames the sources and tracks have run (simulate's `ticks`).
var ticks := 0
## id -> {id, source, item, mode, node (the item's root), body, data, born, still}
var items := {}
var _next_id := 1
var _layer: Node3D               # where items live, in world space
var _settle := -1                # frames until sources run (-1: the world isn't loaded)


func _init(owner_world: WorldLoader) -> void:
    world = owner_world
    name = "Flows"


func _ready() -> void:
    _layer = Node3D.new()
    _layer.name = "Items"
    add_child(_layer)
    world.props.clone_fell.connect(func(uid: String): remove(_id_of(uid), "fell"), CONNECT_DEFERRED)
    world.world_loaded.connect(func(_aabb: AABB): _settle = SETTLE_FRAMES)


## Take the item templates out of a freshly loaded export (before uids, interactions,
## props and colliders are gathered: a template is not part of the world) and collect its
## sources.
func setup(root: Node) -> void:
    var found: Array[Node3D] = []
    for node: Node3D in root.find_children("*", "Node3D", true, false):
        var g := WorldLoader.geogen_extras(node)
        if (g.get("flows", {}) as Dictionary).get("source") is Dictionary:
            found.append(node)
    for node: Node3D in root.find_children("*", "Node3D", true, false):
        var track: Variant = (WorldLoader.geogen_extras(node).get("flows", {}) as Dictionary).get("track")
        if track is Dictionary:
            var points := PackedVector3Array()
            var cum := PackedFloat64Array([0.0])
            for p in track["points"]:
                points.append(node.global_transform * _vec(p))
                if points.size() > 1:
                    cum.append(cum[cum.size() - 1] + points[points.size() - 2].distance_to(points[points.size() - 1]))
            tracks.append({"uid": WorldLoader.uid_of(node), "node": node, "data": track, "points": points,
                "cum": cum, "length": cum[cum.size() - 1], "next": -1, "received": 0})
    for node in found:
        var uid := WorldLoader.uid_of(node)
        var data: Dictionary = WorldLoader.geogen_extras(node)["flows"]["source"]
        var entry := {"uid": uid, "node": node, "data": data, "templates": {}, "due": 0.0, "made": 0,
            "rng": RandomNumberGenerator.new()}
        entry["rng"].seed = hash(uid) + int(data.get("seed", 0))
        for child: Node3D in node.find_children("*", "Node3D", true, false):
            var item: Variant = WorldLoader.geogen_extras(child).get("item")
            if item is Dictionary and child.get_parent() != null:
                child.get_parent().remove_child(child)
                child.visible = true
                entry["templates"][str(item["name"])] = child
        entry["at"] = node.global_transform * _vec(data.get("at", [0, 0, 0]))
        sources[uid] = entry
    _join()


## Order tracks and sources by position (as geogen/flows.py `network` does) and join them:
## a track's end to the track starting within JOIN of it, a source to the track its spawn
## point starts.
func _join() -> void:
    for track in tracks:
        track.erase("ref")
    tracks.sort_custom(func(a, b): return _before(a["points"][0], b["points"][0]))
    for k in tracks.size():
        tracks[k]["index"] = k
    for track in tracks:
        track["next"] = -1
        if track["data"].get("loop", false) or track["data"].get("end", "auto") != "auto":
            continue
        var end: Vector3 = track["points"][track["points"].size() - 1]
        var best := -1
        for other in tracks:
            if other != track and other["points"][0].distance_to(end) <= JOIN and (best < 0 or
                    other["points"][0].distance_to(end) < tracks[best]["points"][0].distance_to(end)):
                best = other["index"]
        track["next"] = best
    var ordered: Array = sources.values()
    ordered.sort_custom(func(a, b): return _before(a["at"], b["at"]))
    var by_position := {}
    for entry in ordered:
        entry["track"] = -1
        for track in tracks:
            if track["points"][0].distance_to(entry["at"]) <= JOIN and (entry["track"] < 0 or
                    track["points"][0].distance_to(entry["at"]) < tracks[entry["track"]]["points"][0].distance_to(entry["at"])):
                entry["track"] = track["index"]
        by_position[entry["uid"]] = entry
    sources = by_position


static func _before(a: Vector3, b: Vector3) -> bool:
    for k in 3:
        var x := snappedf(a[k], 0.001)
        var y := snappedf(b[k], 0.001)
        if x != y:
            return x < y
    return false


## Forget sources under ``root`` and take their items away (it is being unloaded).
func forget(root: Node) -> void:
    for uid in sources.keys():
        var entry: Dictionary = sources[uid]
        if is_instance_valid(entry["node"]) and not root.is_ancestor_of(entry["node"]):
            continue
        for id in items.keys():
            if items[id]["source"] == uid:
                remove(id, "cleared")
        _free_templates(entry)
        sources.erase(uid)
    for track in tracks.duplicate():
        if not is_instance_valid(track["node"]) or root.is_ancestor_of(track["node"]):
            for id in items.keys():
                if items[id].get("track", -1) == track["index"]:
                    remove(id, "cleared")
            tracks.erase(track)
    _join()


## Forget everything (WorldLoader.unload).
func clear() -> void:
    _settle = -1
    for id in items.keys():
        remove(id, "cleared")
    for entry in sources.values():
        _free_templates(entry)
    sources.clear()
    tracks.clear()
    ticks = 0


func _free_templates(entry: Dictionary) -> void:
    for template in entry["templates"].values():
        if is_instance_valid(template):
            template.free()


func _physics_process(delta: float) -> void:
    if _settle != 0:
        _settle = maxi(_settle - 1, -1) if _settle > 0 else -1
        return
    ticks += 1
    for uid in sources:
        if sources[uid]["track"] >= 0 and _on_track(sources[uid]):
            _feed(sources[uid], delta)
        else:
            _run(sources[uid], delta)
    var moved := {}
    for track in tracks:
        _move(track, delta, moved)
    var gone: Array = []
    for id in items:
        var item: Dictionary = items[id]
        item["age"] += delta
        var lifetime := float(item["def"].get("lifetime", 0.0))
        if lifetime > 0.0 and item["age"] > lifetime:
            gone.append([id, "lifetime"])
            continue
        var body: RigidBody3D = item["body"]
        if body != null and is_instance_valid(body):
            var resting := body.linear_velocity.length() < STILL_SPEED and body.angular_velocity.length() < STILL_SPEED * 4.0
            item["still"] = item["still"] + delta if resting and world.props.held != item["prop"] else 0.0
            var stuck := float(item["def"].get("stuck", 10.0))
            if stuck > 0.0 and item["still"] > stuck:
                gone.append([id, "stuck"])
    for entry in gone:
        remove(entry[0], entry[1])


func _run(entry: Dictionary, delta: float) -> void:
    var data: Dictionary = entry["data"]
    var uid: String = entry["uid"]
    var queued := int(_read(uid, "queued", 0))
    var running := GeogenConditions.truthy(_read(uid, "running", data.get("initial", "running") == "running"))
    var rate := float(data.get("rate", 1.0))
    if running and rate > 0.0:
        entry["due"] = minf(entry["due"] + delta * rate, 1.0)
    else:
        entry["due"] = 0.0
    var count := int(data.get("count", 0))
    while (queued > 0 or entry["due"] >= 1.0) and alive(uid) < int(data.get("max_alive", 10)) \
            and (count == 0 or entry["made"] < count) and _clear(entry):
        if queued > 0:
            queued -= 1
            _write(uid, "queued", queued)
        else:
            entry["due"] -= 1.0
        if spawn(entry) < 0:
            break


func _on_track(entry: Dictionary) -> bool:
    for template in entry["templates"].values():
        if str(WorldLoader.geogen_extras(template)["item"].get("mode", "physics")) != "physics":
            return true
    return false


## A source that feeds a track: at most one item a frame, when the track's start is clear.
func _feed(entry: Dictionary, delta: float) -> void:
    var data: Dictionary = entry["data"]
    var uid: String = entry["uid"]
    var queued := int(_read(uid, "queued", 0))
    var running := GeogenConditions.truthy(_read(uid, "running", data.get("initial", "running") == "running"))
    var rate := float(data.get("rate", 1.0))
    entry["due"] = minf(entry["due"] + delta * rate, 1.0) if running and rate > 0.0 else 0.0
    var count := int(data.get("count", 0))
    if (queued > 0 or entry["due"] >= 1.0) and alive(uid) < int(data.get("max_alive", 10)) \
            and (count == 0 or entry["made"] < count):
        var track: Dictionary = tracks[entry["track"]]
        var pick := _pick(entry)
        var radius := float(WorldLoader.geogen_extras(entry["templates"][pick])["item"].get("radius", 0.25))
        for item in items.values():
            if item.get("track", -1) == track["index"] and item["s"] < maxf(float(track["data"].get("spacing", 0.0)),
                    radius + float(item["def"].get("radius", 0.25))):
                entry["pending"] = pick            # the same pick waits for room
                return
        entry.erase("pending")
        if queued > 0:
            _write(uid, "queued", queued - 1)
        else:
            entry["due"] -= 1.0
        spawn(entry, pick)


## Step a track: its items, leaders first (geogen/flows.py `simulate` is the reference).
func _move(track: Dictionary, delta: float, moved: Dictionary) -> void:
    if not GeogenConditions.truthy(_read(track["uid"], "running", true)):
        return
    var index: int = track["index"]
    var data: Dictionary = track["data"]
    var order: Array = items.values().filter(func(it): return it.get("track", -1) == index and not moved.has(it["id"]))
    order.sort_custom(func(a, b): return a["s"] > b["s"])
    var length: float = track["length"]
    var loop := bool(data.get("loop", false))
    for j in order.size():
        var it: Dictionary = order[j]
        var v: float = float(data.get("speed", 1.0)) if data.get("mode", "belt") == "belt" \
            else float(it["def"].get("speed", data.get("speed", 1.0)))
        var target: float = it["s"] + v * delta
        var limit := INF
        if j > 0:
            limit = order[j - 1]["s"] - _gap(it, order[j - 1], data)
        elif loop and order.size() > 1:
            limit = order[order.size() - 1]["s"] + length - _gap(it, order[order.size() - 1], data)
        elif track["next"] >= 0:
            var lead := {}
            for other in items.values():
                if other.get("track", -1) == track["next"] and (lead.is_empty() or other["s"] < lead["s"]):
                    lead = other
            if not lead.is_empty():
                limit = length + lead["s"] - _gap(it, lead, tracks[track["next"]]["data"])
        target = maxf(it["s"], minf(target, limit))
        if loop:
            it["s"] = fmod(target, length)
        elif target >= length:
            var end := str(data.get("end", "auto"))
            var accepts: Array = data.get("accepts", [])
            if track["next"] >= 0:
                it["track"] = track["next"]
                it["s"] = target - length
                moved[it["id"]] = true
            elif end == "stop":
                it["s"] = length
            elif (end == "auto" or end == "sink") and (accepts.is_empty() or it["item"] in accepts):
                _received(track, it)
                continue
            elif end == "drop":
                _drop(track, it)
                continue
            else:
                it["s"] = length
        else:
            it["s"] = target
        _place_on(it)


func _gap(a: Dictionary, b: Dictionary, track_data: Dictionary) -> float:
    return maxf(float(track_data.get("spacing", 0.0)), float(a["def"].get("radius", 0.25)) + float(b["def"].get("radius", 0.25)))


## Put an item where its track says, facing along it (+Z forward).
func _place_on(it: Dictionary) -> void:
    var track: Dictionary = tracks[it["track"]]
    var at := _at(track, it["s"])
    var ahead := _at(track, minf(it["s"] + 0.2, track["length"]))
    if ahead.distance_to(at) < 0.01:
        ahead = at + (at - _at(track, maxf(it["s"] - 0.2, 0.0)))
    var forward := ahead - at
    forward.y = 0.0
    var node: Node3D = it["node"]
    if not is_instance_valid(node):
        return
    if forward.length() > 0.001:
        node.global_transform = Transform3D(Basis.looking_at(-forward.normalized(), Vector3.UP), at)
    else:
        node.global_position = at


## The world point ``s`` m along a track.
func _at(track: Dictionary, s: float) -> Vector3:
    var cum: PackedFloat64Array = track["cum"]
    var points: PackedVector3Array = track["points"]
    s = clampf(s, 0.0, cum[cum.size() - 1])
    var k := clampi(cum.bsearch(s, false) - 1, 0, points.size() - 2)
    var seg := cum[k + 1] - cum[k]
    return points[k].lerp(points[k + 1], 0.0 if seg <= 0.0 else (s - cum[k]) / seg)


## A track's end takes an item in: `received`, and the signal `received {item, ...data}`.
func _received(track: Dictionary, it: Dictionary) -> void:
    remove(it["id"], "sink")
    track["received"] += 1
    _write(track["uid"], "received", track["received"])
    var data: Dictionary = it["data"].duplicate()
    data["item"] = it["item"]
    world.events.post("signal", track["uid"], "received", "", data)


## An item falls off a track's end (`drop`): a physics body moving on along the track, if its
## body has physics; else it is taken away.
func _drop(track: Dictionary, it: Dictionary) -> void:
    var template: Node3D = sources[it["source"]]["templates"].get(it["item"]) if sources.has(it["source"]) else null
    var physics: Dictionary = WorldLoader.geogen_extras(template).get("physics", {}) if template != null else {}
    if physics.is_empty():
        remove(it["id"], "dropped")
        return
    var node: Node3D = it["node"]
    var xform: Transform3D = node.global_transform
    var length: float = track["length"]
    var tangent := (_at(track, length) - _at(track, maxf(length - 0.25, 0.0))).normalized()
    _layer.remove_child(node)
    it.erase("track")
    it["mode"] = "physics"
    it["prop"] = "flow#%d" % it["id"]
    it["body"] = world.props.add_clone(node, physics, it["prop"], _layer, xform,
        tangent * float(track["data"].get("speed", 1.0)))


## The next item a source makes: its pick waiting for room, else a smooth weighted round
## robin over its items (the same as geogen/flows.py `_pick`).
func _pick(entry: Dictionary) -> String:
    if entry.has("pending"):
        return entry["pending"]
    var choices: Array = entry["data"].get("items", [])
    if not entry.has("current"):
        entry["current"] = []
        entry["current"].resize(choices.size())
        entry["current"].fill(0.0)
    var total := 0.0
    for k in choices.size():
        total += float(choices[k].get("weight", 1.0))
        entry["current"][k] += float(choices[k].get("weight", 1.0))
    var best := 0
    for k in choices.size():
        if entry["current"][k] > entry["current"][best]:
            best = k
    entry["current"][best] -= total
    return str(choices[best].get("name", ""))


## Whether a source's spawn point is free of the last item it made.
func _clear(entry: Dictionary) -> bool:
    var last: Dictionary = items.get(entry.get("last", -1), {})
    if last.is_empty():
        return true
    var at: Node3D = last["body"] if last["body"] != null else last["node"]
    if not is_instance_valid(at):
        return true
    var spawn_point: Vector3 = entry["node"].global_transform * _vec(entry["data"].get("at", [0, 0, 0]))
    var centre := at.global_position
    if last["body"] != null:
        centre = at.global_transform * _vec(WorldLoader.geogen_extras(entry["templates"][last["item"]]).get("physics", {})
            .get("center", [0, 0, 0]))
        spawn_point += entry["node"].global_transform.basis * _vec(WorldLoader.geogen_extras(
            entry["templates"][last["item"]]).get("physics", {}).get("center", [0, 0, 0]))
    return centre.distance_to(spawn_point) > 2.0 * float(last["def"].get("radius", 0.25)) + 0.01


## Make one item of a source (its weighted pick); the item's id, or -1.
func spawn(entry: Dictionary, item_name := "") -> int:
    var data: Dictionary = entry["data"]
    var choices: Array = data.get("items", [])
    if item_name == "" and not choices.is_empty():
        item_name = _pick(entry)
    var template: Node3D = entry["templates"].get(item_name)
    if template == null or not is_instance_valid(entry["node"]):
        return -1
    var def: Dictionary = WorldLoader.geogen_extras(template)["item"]
    var at: Vector3 = _vec(data.get("at", [0, 0, 0]))
    var spread := float(data.get("spread", 0.0))
    if spread > 0.0:
        var angle: float = entry["rng"].randf() * TAU
        var r: float = sqrt(entry["rng"].randf()) * spread
        at += Vector3(cos(angle) * r, 0.0, sin(angle) * r)
    var source_xform: Transform3D = entry["node"].global_transform
    var xform := Transform3D(source_xform.basis.orthonormalized(), source_xform * at)
    var id := _next_id
    _next_id += 1
    var copy := template.duplicate() as Node3D
    copy.name = "%s_%d" % [item_name, id]
    for mi: MeshInstance3D in copy.find_children("*", "MeshInstance3D", true, false):
        if WorldLoader._is_collider(mi):         # an item's collider is its body's (or none)
            mi.get_parent().remove_child(mi)
            mi.free()
    var item := {"id": id, "source": entry["uid"], "item": item_name, "mode": str(def.get("mode", "physics")),
        "def": def, "data": (def.get("data", {}) as Dictionary).duplicate(), "node": copy, "body": null,
        "prop": "", "age": 0.0, "still": 0.0}
    if entry.get("track", -1) >= 0 and item["mode"] != "physics":
        item["track"] = entry["track"]
        item["s"] = 0.0
        _layer.add_child(copy)
        _place_on(item)
    elif item["mode"] == "physics":
        var physics: Dictionary = WorldLoader.geogen_extras(template).get("physics", {})
        item["prop"] = "flow#%d" % id
        var velocity := source_xform.basis.orthonormalized() * _vec(data.get("velocity", [0, 0, 0]))
        item["body"] = world.props.add_clone(copy, physics, item["prop"], _layer, xform, velocity)
    else:
        _layer.add_child(copy)
        copy.global_transform = xform
    items[id] = item
    entry["last"] = id
    entry["made"] += 1
    _write(entry["uid"], "emitted", entry["made"])
    _write(entry["uid"], "alive", alive(entry["uid"]))
    world.events.post("signal", entry["uid"], "emitted", "", {"item": item_name})
    item_added.emit(id)
    return id


## A sink (a place entry of WorldLoader.places with `sink`) takes an item in, if it accepts
## it: the item goes, the sink's `received` counts it and it raises `received {item, ...data}`.
func take(id: int, sink: Dictionary) -> bool:
    var item: Dictionary = items.get(id, {})
    var accepts: Array = (sink.get("sink", {}) as Dictionary).get("accepts", [])
    if item.is_empty() or (not accepts.is_empty() and not item["item"] in accepts):
        return false
    remove(id, "sink")
    sink["received"] = int(sink.get("received", 0)) + 1
    var key := "%s#received" % sink["uid"]
    if world.rules.machine != null and world.rules.machine.variables.has(key):
        world.rules.set_value(key, float(sink["received"]))
    var data: Dictionary = item["data"].duplicate()
    data["item"] = item["item"]
    world.events.post("signal", sink["uid"], "received", "", data)
    return true


## Take an item away (why: sink, lifetime, stuck, fell, cleared). False if there is no such item.
func remove(id: int, why: String) -> bool:
    var item: Dictionary = items.get(id, {})
    if item.is_empty():
        return false
    items.erase(id)
    if item["prop"] != "":
        world.props.remove_clone(item["prop"])
    elif is_instance_valid(item["node"]):
        item["node"].queue_free()
    if why != "cleared":
        _write(item["source"], "alive", alive(item["source"]))
    item_removed.emit(id, why)
    return true


## The item a prop clone or a collider belongs to (-1: none).
func item_of(body: Object) -> int:
    if body == null:
        return -1
    return _id_of(world.props.prop_of(body))


func _id_of(prop: String) -> int:
    return int(prop.substr(5)) if prop.begins_with("flow#") else -1


## How many of a source's items are in the world.
func alive(source: String) -> int:
    var n := 0
    for item in items.values():
        if item["source"] == source:
            n += 1
    return n


func _read(uid: String, name_: String, fallback: Variant) -> Variant:
    var machine: GeogenMachine = world.rules.machine
    var key := "%s#%s" % [uid, name_]
    if machine == null or not machine.variables.has(key):
        return fallback
    var found: Variant = machine.get_value(key)
    return fallback if found == null else found


func _write(uid: String, name_: String, value: Variant) -> void:
    var key := "%s#%s" % [uid, name_]
    if world.rules.machine != null and world.rules.machine.variables.has(key):
        world.rules.set_value(key, float(value))


## {sources: [{uid, running, queued, emitted, alive, items: [names]}], items: [{id, source, item,
## mode, position, data, age, track?, s?}], tracks: [{uid, index, length, next, received,
## running}], ticks}: what flows.list returns.
func report() -> Dictionary:
    var out := {"sources": [], "items": []}
    for uid in sources:
        var entry: Dictionary = sources[uid]
        out["sources"].append({"uid": uid, "running": GeogenConditions.truthy(_read(uid, "running",
            entry["data"].get("initial", "running") == "running")), "queued": int(_read(uid, "queued", 0)),
            "emitted": entry["made"], "alive": alive(uid), "items": entry["templates"].keys()})
    for id in items:
        var item: Dictionary = items[id]
        var at: Node3D = item["body"] if item["body"] != null else item["node"]
        var p: Vector3 = at.global_position if is_instance_valid(at) else Vector3.ZERO
        var entry := {"id": id, "source": item["source"], "item": item["item"], "mode": item["mode"],
            "position": [snappedf(p.x, 0.001), snappedf(p.y, 0.001), snappedf(p.z, 0.001)],
            "data": item["data"], "age": snappedf(item["age"], 0.01)}
        if item.has("track"):
            entry["track"] = item["track"]
            entry["s"] = item["s"]
        out["items"].append(entry)
    out["tracks"] = tracks.map(func(t): return {"uid": t["uid"], "index": t["index"], "length": t["length"],
        "next": t["next"], "received": t["received"], "running": GeogenConditions.truthy(_read(t["uid"], "running", true))})
    out["ticks"] = ticks
    return out


static func _vec(a) -> Vector3:
    return Vector3(float(a[0]), float(a[1]), float(a[2]))
