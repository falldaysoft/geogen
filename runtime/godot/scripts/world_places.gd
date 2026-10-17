class_name GeogenWorldPlaces
extends RefCounted
## Who is in each named place and room (WorldLoader.places): the areas the scene builder made
## report actors walking in and out (enter / leave events, WorldLoader.place_event), and places
## that detect props count their occupants.

var world: WorldLoader
var _held: Array[Dictionary] = []   # {entry, body, actor}: occupants whose body stopped colliding (seated, lying)


## Forget the places and held occupants under what ``inside`` says belongs to a freed chunk.
func forget(inside: Callable) -> void:
    world.places = world.places.filter(func(p): return not inside.call(p.get("node")))
    _held = _held.filter(func(h): return not inside.call(h["entry"].get("node")))


func clear() -> void:
    _held.clear()


## Place and room areas (made by the scene builder) report the actors that walk
## in and out: the player and NPCs.
func collect(root: Node) -> void:
    for area: Area3D in root.find_children("*", "Area3D", true, false):
        var is_room := area.is_in_group("geogen_room")
        if not (is_room or area.is_in_group("geogen_place")) or area.has_meta("geogen_wired"):
            continue
        area.set_meta("geogen_wired", true)
        var node := area.get_parent()
        var entry := {"kind": "room" if is_room else "place", "node": node, "area": area, "actors": {}}
        if is_room:      # the room is the volume's parent; rules name it, not its volume
            var room: Dictionary = area.get_meta("geogen_room", {})
            entry["id"] = str(room.get("id", node.name))
            entry["uid"] = WorldLoader.uid_of(node.get_parent()) if node.get_parent() != null else ""
            if entry["uid"] == "":
                entry["uid"] = WorldLoader.uid_of(node)
        else:
            var place: Dictionary = area.get_meta("geogen_place", {})
            entry["id"] = str(place.get("id", node.name))
            entry["uid"] = str(place.get("uid", ""))
            entry["detect"] = place.get("detect", ["actors"])
            entry["things"] = {}
            if place.get("sink") is Dictionary:
                entry["sink"] = place["sink"]
        world.places.append(entry)
        area.body_entered.connect(func(body: Node3D): crossed(entry, body, true))
        area.body_exited.connect(func(body: Node3D): crossed(entry, body, false))


func crossed(entry: Dictionary, body: Node3D, entered: bool) -> void:
    var actor := WorldLoader.actor_id(body)
    if actor == "" and entered and entry.has("sink") and world.flows.item_of(body) >= 0:
        world.flows.take(world.flows.item_of(body), entry)
        return
    if actor == "":
        var prop := world.props.prop_of(body)
        if prop != "" and "props" in entry.get("detect", []):
            occupy(entry, prop, entered)
        return
    if entered:
        # Bodies sit at the origin for a frame while a world loads or they spawn:
        # an arrival counts once the body is still inside two frames on.
        for i in 2:
            await world.get_tree().physics_frame
        var area = entry["area"]
        if not is_instance_valid(body) or not is_instance_valid(area) or not area.overlaps_body(body) \
                or entry["actors"].has(actor):
            return
        entry["actors"][actor] = true
    elif not entry["actors"].has(actor):
        return
    elif parked(body):
        _held.append({"entry": entry, "body": body, "actor": actor})
        return
    else:
        entry["actors"].erase(actor)
    post_place("enter" if entered else "leave", entry, actor)


## Seated and lying actors turn their collision shape off, which an area reads as
## walking out: they stay in the place until the shape is back (or they vanish).
func parked(body: Node3D) -> bool:
    if not is_instance_valid(body) or not body.is_inside_tree() or not body.visible:
        return false
    for child in body.get_children():
        if child is CollisionShape3D and not child.disabled:
            return false
    return true


## Occupants whose shape is back (stood up): gone, unless they're still inside. Every physics frame.
func release_held() -> void:
    for held in _held.duplicate():
        if parked(held["body"]):
            continue
        _held.erase(held)
        leave_unless_inside(held)


func leave_unless_inside(held: Dictionary) -> void:
    for i in 2:      # the area sees a shape that came back on the next physics frame
        await world.get_tree().physics_frame
    var entry: Dictionary = held["entry"]
    var body = held["body"]
    var area = entry["area"]
    if is_instance_valid(body) and is_instance_valid(area) and body.visible and area.overlaps_body(body):
        return
    if entry["actors"].erase(held["actor"]):
        post_place("leave", entry, held["actor"])


func post_place(type: String, entry: Dictionary, actor: String) -> void:
    if "props" in entry.get("detect", []):
        occupy(entry, actor, type == "enter")
    world.events.post(type, entry["uid"], "", actor, {"place": entry["id"], "kind": entry["kind"]})
    world.place_event.emit({"type": type, "subject": entry["uid"], "place": entry["id"], "kind": entry["kind"],
        "actor": actor})


## A place that detects props counts what is in it (actors and props) as its
## `occupants` (declared with `driven: place`), which rules read.
func occupy(entry: Dictionary, thing: String, inside: bool) -> void:
    var things: Dictionary = entry["things"]
    if inside == things.has(thing):
        return
    if inside:
        things[thing] = true
    else:
        things.erase(thing)
    var key := "%s#occupants" % entry["uid"]
    if world.rules.machine != null and world.rules.machine.variables.has(key):
        world.rules.set_value(key, float(things.size()))
