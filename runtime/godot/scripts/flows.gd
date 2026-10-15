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
## A physics item is a clone prop (props.gd `add_clone`): carried like any prop. An item is
## taken away when a sink takes it in (`take`), when it outlives `lifetime`, when a physics
## item has rested `stuck` seconds, or when it falls out of the world.
##
## Owned by WorldLoader (`world.flows`); the control API's `flows.list` reports it.

## An item was made or taken away (why: sink | lifetime | stuck | fell | cleared).
signal item_added(id: int)
signal item_removed(id: int, why: String)

const STILL_SPEED := 0.05       ## m/s: slower than this, a physics item is at rest

var world: WorldLoader
## uid -> {uid, node, data, templates: {item name: Node3D}, rng, due, made}
var sources := {}
## id -> {id, source, item, mode, node (the item's root), body, data, born, still}
var items := {}
var _next_id := 1
var _layer: Node3D               # where items live, in world space


func _init(owner_world: WorldLoader) -> void:
    world = owner_world
    name = "Flows"


func _ready() -> void:
    _layer = Node3D.new()
    _layer.name = "Items"
    add_child(_layer)
    world.props.clone_fell.connect(func(uid: String): remove(_id_of(uid), "fell"), CONNECT_DEFERRED)


## Take the item templates out of a freshly loaded export (before uids, interactions,
## props and colliders are gathered: a template is not part of the world) and collect its
## sources.
func setup(root: Node) -> void:
    var found: Array[Node3D] = []
    for node: Node3D in root.find_children("*", "Node3D", true, false):
        var g := WorldLoader.geogen_extras(node)
        if (g.get("flows", {}) as Dictionary).get("source") is Dictionary:
            found.append(node)
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
        sources[uid] = entry


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


## Forget everything (WorldLoader.unload).
func clear() -> void:
    for id in items.keys():
        remove(id, "cleared")
    for entry in sources.values():
        _free_templates(entry)
    sources.clear()


func _free_templates(entry: Dictionary) -> void:
    for template in entry["templates"].values():
        if is_instance_valid(template):
            template.free()


func _physics_process(delta: float) -> void:
    for uid in sources:
        _run(sources[uid], delta)
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
            and (count == 0 or entry["made"] < count):
        if queued > 0:
            queued -= 1
            _write(uid, "queued", queued)
        else:
            entry["due"] -= 1.0
        if spawn(entry) < 0:
            break


## Make one item of a source (its weighted pick); the item's id, or -1.
func spawn(entry: Dictionary, item_name := "") -> int:
    var data: Dictionary = entry["data"]
    var choices: Array = data.get("items", [])
    if item_name == "" and not choices.is_empty():
        var total := 0.0
        for choice in choices:
            total += float(choice.get("weight", 1.0))
        var pick: float = entry["rng"].randf() * total
        for choice in choices:
            pick -= float(choice.get("weight", 1.0))
            item_name = str(choice.get("name", ""))
            if pick <= 0.0:
                break
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
    if item["mode"] == "physics":
        var physics: Dictionary = WorldLoader.geogen_extras(template).get("physics", {})
        item["prop"] = "flow#%d" % id
        var velocity := source_xform.basis.orthonormalized() * _vec(data.get("velocity", [0, 0, 0]))
        item["body"] = world.props.add_clone(copy, physics, item["prop"], _layer, xform, velocity)
    else:
        _layer.add_child(copy)
        copy.global_transform = xform
    items[id] = item
    entry["made"] += 1
    _write(entry["uid"], "emitted", entry["made"])
    _write(entry["uid"], "alive", alive(entry["uid"]))
    world.events.post("signal", entry["uid"], "emitted", "", {"item": item_name})
    item_added.emit(id)
    return id


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
## mode, position, data, age}]}: what flows.list returns.
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
        out["items"].append({"id": id, "source": item["source"], "item": item["item"], "mode": item["mode"],
            "position": [snappedf(p.x, 0.001), snappedf(p.y, 0.001), snappedf(p.z, 0.001)],
            "data": item["data"], "age": snappedf(item["age"], 0.01)})
    return out


static func _vec(a) -> Vector3:
    return Vector3(float(a[0]), float(a[1]), float(a[2]))
