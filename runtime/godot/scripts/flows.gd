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
## (`stop`) or falls off as a physics body (`drop`). On a `roll` track items are driven by
## gravity: they speed up down its slope ((5/7) g sin(slope)), slow by its `damping`, and spin
## as they go (balls down a chute; physics items too: they fall off the end as bodies). geogen/flows.py `simulate` is the
## reference: sources and tracks are stepped in the same order (by position), and a source's
## pick among its items is the same smooth weighted round robin, so both agree.
##
## Effectors (extras.geogen.flows.effector) come last each frame, in position order: one that
## is `enabled` and ready (every 1 / rate s) acts on the track item in range it `targets`.
## Damage lands as it fires (the projectile is the look of it): at hp 0 the item is destroyed
## (it shrinks away) and its source counts `killed` and raises `killed {item, ...data}`; take
## removes it (`taken`, signal `took`). Aim parts turn toward the target at `turn_rate`.
## A waves source (`waves`) sends its waves' groups in turn while running and stops when a
## wave is cleared (or goes on, `auto`), raising wave_started / wave_cleared / all_cleared.
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
const ROLL_GRAVITY := 9.81 * 5.0 / 7.0   ## a solid ball rolling without slipping: (5/7) g sin(slope)

var world: WorldLoader
## uid -> {uid, node, data, templates: {item name: Node3D}, rng, due, made, at (world), track}
var sources := {}
## [{uid, node, data, points (world), cum, length, next, received}], in stepping order
var tracks: Array = []
## [{uid, node, data, at (world), cool, taken, target, yaw, pitch, ...}], in stepping order
var effectors: Array = []
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
    for node: Node3D in root.find_children("*", "Node3D", true, false):
        var effector: Variant = (WorldLoader.geogen_extras(node).get("flows", {}) as Dictionary).get("effector")
        if effector is Dictionary:
            var entry := {"uid": WorldLoader.uid_of(node), "node": node, "data": effector, "cool": 0.0, "taken": 0,
                "target": -1, "at": node.global_transform * _vec(effector.get("at", [0, 0, 0]))}
            var aim: Dictionary = effector.get("aim", {})
            for key in ["yaw", "pitch"]:
                if aim.has(key):
                    var part := node.find_child(str(aim[key]), true, false) as Node3D
                    if part != null:
                        entry[key] = part
                        entry[key + "_rest"] = part.basis
                        entry[key + "_angle"] = 0.0
            var sound := AudioStreamPlayer3D.new()
            sound.stream = GeogenSound.chime_stream("shot")
            sound.unit_size = 6.0
            sound.position = _vec(effector.get("at", [0, 0, 0]))
            node.add_child(sound)
            entry["sound"] = sound
            effectors.append(entry)
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
        entry.merge({"killed": 0, "wave": 0, "phase": "idle", "timer": 0.0, "group": 0, "left": 0})
        sources[uid] = entry
    _join()


## Order tracks and sources by position (as geogen/flows.py `network` does) and join them:
## a track's end to the track starting within JOIN of it, a source to the track its spawn
## point starts.
func _join() -> void:
    for track in tracks:
        track.erase("ref")
    tracks.sort_custom(func(a, b): return _before(a["points"][0], b["points"][0]))
    effectors.sort_custom(func(a, b): return _before(a["at"], b["at"]))
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
    for effector in effectors.duplicate():
        if not is_instance_valid(effector["node"]) or root.is_ancestor_of(effector["node"]):
            effectors.erase(effector)
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
    effectors.clear()
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
        var entry: Dictionary = sources[uid]
        if entry["track"] >= 0 and _on_track(entry):
            if entry["data"].get("waves") is Dictionary:
                _waves(entry, delta)
            else:
                _feed(entry, delta)
        else:
            _run(entry, delta)
    var moved := {}
    for track in tracks:
        _move(track, delta, moved)
    for effector in effectors:
        _act(effector, delta)
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
            var resting := body.linear_velocity.length() < STILL_SPEED    # creeping counts (balls spin on)
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
    if entry["track"] >= 0 and str(tracks[entry["track"]]["data"].get("mode", "belt")) == "roll":
        return true                                 # gravity carries physics items down it
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
        var pick := _pick(entry)
        if not _start_clear(entry, pick):
            entry["pending"] = pick                # the same pick waits for room
            return
        entry.erase("pending")
        if queued > 0:
            _write(uid, "queued", queued - 1)
        else:
            entry["due"] -= 1.0
        spawn(entry, pick)


## Whether a source's track has room at its start for an item ``item_name``.
func _start_clear(entry: Dictionary, item_name: String) -> bool:
    var track: Dictionary = tracks[entry["track"]]
    var radius := float(WorldLoader.geogen_extras(entry["templates"][item_name])["item"].get("radius", 0.25))
    for item in items.values():
        if item.get("track", -1) == track["index"] and item["s"] < maxf(float(track["data"].get("spacing", 0.0)),
                radius + float(item["def"].get("radius", 0.25))):
            return false
    return true


## A waves source (geogen/flows.py `simulate`'s `waves` is the reference).
func _waves(entry: Dictionary, delta: float) -> void:
    var plan: Dictionary = entry["data"]["waves"]
    var list: Array = plan.get("list", [])
    var uid: String = entry["uid"]
    if entry["phase"] == "clearing":
        if alive(uid) == 0:
            world.events.post("signal", uid, "wave_cleared", "", {"wave": float(entry["wave"])})
            entry["phase"] = "idle"
            if entry["wave"] >= list.size():
                world.events.post("signal", uid, "all_cleared", "", {"wave": float(entry["wave"])})
                _write(uid, "running", false)
            elif not plan.get("auto", false):
                _write(uid, "running", false)
        return
    if not GeogenConditions.truthy(_read(uid, "running", entry["data"].get("initial", "running") == "running")):
        return
    match entry["phase"]:
        "idle":
            if entry["wave"] < list.size():
                entry["phase"] = "delay"
                entry["timer"] = float(list[entry["wave"]].get("delay", 2.0))
        "delay":
            entry["timer"] -= delta
            if entry["timer"] <= 0.0:
                entry["wave"] += 1
                _write(uid, "wave", entry["wave"])
                world.events.post("signal", uid, "wave_started", "", {"wave": float(entry["wave"])})
                entry["phase"] = "spawning"
                entry["group"] = 0
                entry["timer"] = 0.0
                entry["left"] = int(list[entry["wave"] - 1]["items"][0]["count"])
        "spawning":
            var groups: Array = list[entry["wave"] - 1]["items"]
            entry["timer"] -= delta
            if entry["timer"] > 0.0 or alive(uid) >= int(entry["data"].get("max_alive", 10)):
                return
            var group: Dictionary = groups[entry["group"]]
            if not _start_clear(entry, str(group["name"])):
                return
            spawn(entry, str(group["name"]))
            entry["left"] -= 1
            entry["timer"] = float(group.get("gap", 1.0))
            if entry["left"] <= 0:
                entry["group"] += 1
                if entry["group"] >= groups.size():
                    entry["phase"] = "clearing"
                else:
                    entry["left"] = int(groups[entry["group"]]["count"])


## An effector's frame: aim, and act when ready (geogen/flows.py `simulate` is the reference).
func _act(e: Dictionary, delta: float) -> void:
    var data: Dictionary = e["data"]
    _aim(e, delta)
    if not GeogenConditions.truthy(_read(e["uid"], "enabled", data.get("enabled", true))):
        e["target"] = -1
        return
    e["cool"] -= delta
    if e["cool"] > 0.0:
        return
    var effect: Dictionary = data.get("effect", {})
    var damage := effect.has("damage")
    var accepts: Array = data.get("accepts", [])
    var best := {}
    var best_d := 0.0
    var mode := str(data.get("targets", "first"))
    for id in items:
        var it: Dictionary = items[id]
        if not it.has("track") or (not accepts.is_empty() and not it["item"] in accepts) \
                or (damage and not it["data"].has("hp")):
            continue
        var d := _at(tracks[it["track"]], it["s"]).distance_to(e["at"])
        if d > float(data.get("range", 5.0)):
            continue
        if best.is_empty() or _better(mode, it, d, best, best_d):
            best = it
            best_d = d
    if best.is_empty():
        e["cool"] = 0.0
        e["target"] = -1
        return
    e["cool"] = 1.0 / float(data.get("rate", 1.0))
    e["target"] = best["id"]
    var hit_at := _at(tracks[best["track"]], best["s"]) + Vector3.UP * float(best["def"].get("radius", 0.25))
    _shot(e, hit_at)
    if damage:
        best["data"]["hp"] = float(best["data"]["hp"]) - float(effect["damage"])
        if float(best["data"]["hp"]) <= 0.0:
            _killed(best)
        else:
            _health(best)
    else:
        remove(best["id"], "taken")
        e["taken"] += 1
        _write(e["uid"], "taken", e["taken"])
        var sent: Dictionary = best["data"].duplicate()
        sent["item"] = best["item"]
        world.events.post("signal", e["uid"], "took", "", sent)


## Whether ``it`` (``d`` m away) is a better target than ``best`` (``best_d``) for ``mode``.
static func _better(mode: String, it: Dictionary, d: float, best: Dictionary, best_d: float) -> bool:
    match mode:
        "last":
            return it["travelled"] < best["travelled"] or (it["travelled"] == best["travelled"] and it["id"] < best["id"])
        "nearest":
            return d < best_d or (d == best_d and it["id"] < best["id"])
        "strongest":
            var a := float(it["data"]["hp"])
            var b := float(best["data"]["hp"])
            if a != b:
                return a > b
            if it["travelled"] != best["travelled"]:
                return it["travelled"] > best["travelled"]
            return it["id"] < best["id"]
    return it["travelled"] > best["travelled"] or (it["travelled"] == best["travelled"] and it["id"] < best["id"])


## Turn an effector's aim parts toward its target (the look of it: firing doesn't wait).
func _aim(e: Dictionary, delta: float) -> void:
    if not e.has("yaw") and not e.has("pitch"):
        return
    var target: Dictionary = items.get(e["target"], {})
    if target.is_empty() or not target.has("track"):
        return
    var at := _at(tracks[target["track"]], target["s"]) + Vector3.UP * float(target["def"].get("radius", 0.25))
    var step := deg_to_rad(float(e["data"].get("aim", {}).get("turn_rate", 360.0))) * delta
    if e.has("yaw") and is_instance_valid(e["yaw"]):
        var part: Node3D = e["yaw"]
        var local: Vector3 = part.get_parent().to_local(at) - part.position
        var want := atan2(local.x, local.z)
        e["yaw_angle"] = e["yaw_angle"] + clampf(wrapf(want - e["yaw_angle"], -PI, PI), -step, step)
        part.basis = Basis(Vector3.UP, e["yaw_angle"]) * e["yaw_rest"]
    if e.has("pitch") and is_instance_valid(e["pitch"]):
        var part: Node3D = e["pitch"]
        var from: Vector3 = part.global_position
        var want := atan2(at.y - from.y, Vector2(at.x - from.x, at.z - from.z).length())
        e["pitch_angle"] = e["pitch_angle"] + clampf(want - e["pitch_angle"], -step, step)
        part.basis = Basis(Vector3.RIGHT, -e["pitch_angle"]) * e["pitch_rest"]


## What a shot looks like: a tracer flash, or a bolt that flies to the target.
func _shot(e: Dictionary, to: Vector3) -> void:
    var projectile: Dictionary = e["data"].get("projectile", {})
    var headless := DisplayServer.get_name() == "headless"
    var sound: AudioStreamPlayer3D = e.get("sound")
    if sound != null and is_instance_valid(sound) and not headless:
        sound.play()
    var kind := str(projectile.get("kind", "tracer"))
    if kind == "instant" or headless:
        return
    var from: Vector3 = e["at"]
    var c: Array = projectile.get("color", [1.0, 0.75, 0.3])
    var material := StandardMaterial3D.new()
    material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
    material.albedo_color = Color(float(c[0]), float(c[1]), float(c[2]))
    var mesh := MeshInstance3D.new()
    mesh.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
    _layer.add_child(mesh)
    if kind == "tracer":
        var box := BoxMesh.new()
        box.size = Vector3(0.03, 0.03, maxf(from.distance_to(to), 0.01))
        box.material = material
        mesh.mesh = box
        if from.distance_to(to) > 0.01:
            mesh.global_transform = Transform3D(Basis.looking_at(to - from, Vector3.UP), (from + to) / 2.0)
        get_tree().create_timer(0.08, false, true).timeout.connect(mesh.queue_free)
    else:
        var ball := SphereMesh.new()
        ball.radius = 0.06
        ball.height = 0.12
        ball.material = material
        mesh.mesh = ball
        mesh.global_position = from
        var tween := mesh.create_tween()
        tween.tween_property(mesh, "global_position", to, from.distance_to(to) / maxf(float(projectile.get("speed", 20.0)), 0.1))
        tween.tween_callback(mesh.queue_free)


## An item's hp ran out: its source counts and signals it, and it shrinks away.
func _killed(it: Dictionary) -> void:
    var source: Dictionary = sources.get(it["source"], {})
    if not source.is_empty():
        source["killed"] += 1
        _write(it["source"], "killed", source["killed"])
    var sent: Dictionary = it["data"].duplicate()
    sent["item"] = it["item"]
    remove(it["id"], "killed")
    world.events.post("signal", it["source"], "killed", "", sent)


## Show (or refresh) the health bar over a damaged item.
func _health(it: Dictionary) -> void:
    var node: Node3D = it["node"]
    if not is_instance_valid(node):
        return
    var full := float((it["def"].get("data", {}) as Dictionary).get("hp", 1.0))
    var bar := node.get_node_or_null("health") as MeshInstance3D
    if bar == null:
        bar = MeshInstance3D.new()
        bar.name = "health"
        var quad := QuadMesh.new()
        quad.size = Vector2(0.6, 0.08)
        bar.mesh = quad
        bar.material_override = _bar_material()
        bar.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
        bar.position = Vector3.UP * (float(it["def"].get("radius", 0.25)) * 1.6 + 0.2)
        node.add_child(bar)
    bar.set_instance_shader_parameter("fill", clampf(float(it["data"]["hp"]) / maxf(full, 0.001), 0.0, 1.0))


static var _bar: ShaderMaterial


static func _bar_material() -> ShaderMaterial:
    if _bar == null:
        var shader := Shader.new()
        shader.code = "shader_type spatial;\nrender_mode unshaded, cull_disabled, shadows_disabled;\n" \
            + "instance uniform float fill = 1.0;\n" \
            + "void vertex() {\n    MODELVIEW_MATRIX = VIEW_MATRIX * mat4(INV_VIEW_MATRIX[0], INV_VIEW_MATRIX[1], " \
            + "INV_VIEW_MATRIX[2], MODEL_MATRIX[3]);\n}\n" \
            + "void fragment() {\n    ALBEDO = UV.x < fill ? vec3(0.35, 0.9, 0.35) : vec3(0.25, 0.06, 0.05);\n}\n"
        _bar = ShaderMaterial.new()
        _bar.shader = shader
    return _bar


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
        var roll := str(data.get("mode", "belt")) == "roll"
        var v: float
        if roll:
            it["v"] = maxf(0.0, float(it.get("v", 0.0)) + (ROLL_GRAVITY * _fall(track, it["s"])
                - float(data.get("damping", 0.1)) * float(it.get("v", 0.0))) * delta)
            v = it["v"]
        else:
            v = float(data.get("speed", 1.0)) if data.get("mode", "belt") == "belt" \
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
        if roll and target - it["s"] < v * delta:
            it["v"] = (target - it["s"]) / delta    # held up behind the one ahead
        it["travelled"] += target - it["s"]
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


## How steeply a track falls at ``s``: -dy/ds of the segment there (geogen/flows.py `_fall`).
func _fall(track: Dictionary, s: float) -> float:
    var cum: PackedFloat64Array = track["cum"]
    var points: PackedVector3Array = track["points"]
    var k := clampi(cum.bsearch(s, false) - 1, 0, points.size() - 2)
    var seg := cum[k + 1] - cum[k]
    return 0.0 if seg <= 0.0 else (points[k].y - points[k + 1].y) / seg


## Put an item where its track says, facing along it (+Z forward); a rolling one spins.
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
    if forward.length() <= 0.001:
        node.global_position = at
        return
    var basis := Basis.looking_at(-forward.normalized(), Vector3.UP)
    if str(track["data"].get("mode", "belt")) == "roll":
        var r := maxf(float(it["def"].get("radius", 0.1)), 0.01)
        basis = basis * Basis(Vector3.RIGHT, float(it["travelled"]) / r)
        var centre := at + Vector3.UP * r              # it turns about its middle, not its origin
        node.global_transform = Transform3D(basis, centre - basis * (Vector3.UP * r))
    elif it["def"].has("gait"):
        # A walking body steps: up twice a stride, rocking and swinging its heading once.
        var gait: Dictionary = it["def"]["gait"]
        var phase := float(it["travelled"]) / float(gait["stride"]) * TAU
        basis = basis * Basis(Vector3.UP, deg_to_rad(float(gait["waddle"])) * sin(phase)) \
            * Basis(Vector3.BACK, deg_to_rad(float(gait["sway"])) * sin(phase))
        node.global_transform = Transform3D(basis, at + Vector3.UP * float(gait["bob"]) * absf(sin(phase)))
    else:
        node.global_transform = Transform3D(basis, at)


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
    var speed := float(it["v"]) if it.has("v") else float(track["data"].get("speed", 1.0))
    it["body"] = world.props.add_clone(node, physics, it["prop"], _layer, xform, tangent * speed)


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
    var item := _new_item(entry, template, id)
    var copy: Node3D = item["node"]
    if entry.get("track", -1) >= 0 and (item["mode"] != "physics"
            or str(tracks[entry["track"]]["data"].get("mode", "belt")) == "roll"):
        item["track"] = entry["track"]
        item["s"] = 0.0
        item["v"] = float(tracks[entry["track"]]["data"].get("speed", 1.0))
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


## An item of ``entry``'s from ``template``, numbered ``id``: its node (a copy of the template
## without colliders, not yet in the tree) and its record.
func _new_item(entry: Dictionary, template: Node3D, id: int) -> Dictionary:
    var def: Dictionary = WorldLoader.geogen_extras(template)["item"]
    var copy := template.duplicate() as Node3D
    copy.name = "%s_%d" % [def["name"], id]
    for mi: MeshInstance3D in copy.find_children("*", "MeshInstance3D", true, false):
        if WorldLoader._is_collider(mi):         # an item's collider is its body's (or none)
            mi.get_parent().remove_child(mi)
            mi.free()
    return {"id": id, "source": entry["uid"], "item": str(def["name"]), "mode": str(def.get("mode", "physics")),
        "def": def, "data": (def.get("data", {}) as Dictionary).duplicate(), "node": copy, "body": null,
        "prop": "", "age": 0.0, "still": 0.0, "travelled": 0.0}


## A sink (a place entry of WorldLoader.places with `sink`) takes an item in, if it accepts
## it: the item goes, the sink's `received` counts it and it raises `received {item, ...data}`.
func take(id: int, sink: Dictionary) -> bool:
    var item: Dictionary = items.get(id, {})
    var accepts: Array = (sink.get("sink", {}) as Dictionary).get("accepts", [])
    if item.is_empty() or (not accepts.is_empty() and not item["item"] in accepts):
        return false
    remove(id, "sink")
    sink["received"] = int(_read(sink["uid"], "received", sink.get("received", 0))) + 1
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
        if why == "killed" and DisplayServer.get_name() != "headless":
            var node: Node3D = item["node"]          # it shrinks and sinks away
            var tween := node.create_tween().set_parallel()
            tween.tween_property(node, "scale", Vector3.ONE * 0.05, 0.35)
            tween.tween_property(node, "global_position", node.global_position + Vector3.DOWN * 0.2, 0.35)
            tween.chain().tween_callback(node.queue_free)
        else:
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
        world.rules.set_value(key, value if typeof(value) == TYPE_BOOL else float(value))


# -- saving ----------------------------------------------------------------------------------------

## What a save keeps (its `flows` section; {} when there is nothing): live items (on a track by
## its uid and distance, or a physics item's pose and velocity) and how far each source and
## effector had got. Counts (emitted, killed, wave, received, taken) are rules state, saved
## with the rest of it.
func save() -> Dictionary:
    if sources.is_empty() and tracks.is_empty() and effectors.is_empty():
        return {}
    var out := {"next_id": _next_id, "ticks": ticks, "sources": {}, "effectors": {}, "items": []}
    for uid in sources:
        var entry: Dictionary = sources[uid]
        var kept := {"due": entry["due"], "phase": entry["phase"], "timer": entry["timer"], "group": entry["group"],
            "left": entry["left"]}
        for key in ["current", "pending", "last"]:
            if entry.has(key):
                kept[key] = entry[key]
        out["sources"][uid] = kept
    for e in effectors:
        out["effectors"][e["uid"]] = {"cool": e["cool"]}
    for id in items:
        var it: Dictionary = items[id]
        var kept := {"id": id, "source": it["source"], "item": it["item"], "data": it["data"], "age": it["age"],
            "travelled": it["travelled"]}
        if it.has("track"):
            kept["track"] = tracks[it["track"]]["uid"]
            kept["s"] = it["s"]
            if it.has("v"):
                kept["v"] = it["v"]
        elif it["body"] != null and is_instance_valid(it["body"]):
            var body: RigidBody3D = it["body"]
            kept["position"] = _arr(body.global_position)
            kept["rotation"] = _arr(body.global_rotation_degrees)
            kept["velocity"] = _arr(body.linear_velocity)
        else:
            continue
        out["items"].append(kept)
    return out


## Take up a save's `flows` section (after the rules program has its state): the items in the
## world now go, the saved ones come back where they were, and counts are the rules'.
func restore(data: Dictionary) -> void:
    for id in items.keys():
        remove(id, "cleared")
    _next_id = int(data.get("next_id", _next_id))  # every item now is gone: ids go on as saved
    ticks = int(data.get("ticks", ticks))
    var kept_sources: Dictionary = data.get("sources", {})
    for uid in sources:
        var entry: Dictionary = sources[uid]
        var kept: Dictionary = kept_sources.get(uid, {})
        entry.merge({"due": 0.0, "phase": "idle", "timer": 0.0, "group": 0, "left": 0}, true)
        for key in ["current", "pending", "last"]:
            entry.erase(key)
        for key in kept:
            entry[key] = kept[key]
        for key in ["group", "left", "last"]:
            if entry.has(key):
                entry[key] = int(entry[key])
        entry["made"] = int(_read(uid, "emitted", 0))
        entry["killed"] = int(_read(uid, "killed", 0))
        entry["wave"] = int(_read(uid, "wave", 0))
    var kept_effectors: Dictionary = data.get("effectors", {})
    for e in effectors:
        e["cool"] = float(kept_effectors.get(e["uid"], {}).get("cool", 0.0))
        e["taken"] = int(_read(e["uid"], "taken", 0))
        e["target"] = -1
    var track_of := {}
    for track in tracks:
        track["received"] = int(_read(track["uid"], "received", 0))
        track_of[track["uid"]] = track["index"]
    for kept in data.get("items", []):
        var entry: Dictionary = sources.get(str(kept.get("source", "")), {})
        var template: Node3D = entry.get("templates", {}).get(str(kept.get("item", ""))) if not entry.is_empty() else null
        if template == null:
            continue
        var item := _new_item(entry, template, int(kept["id"]))
        item["data"] = (kept.get("data", {}) as Dictionary).duplicate()
        item["age"] = float(kept.get("age", 0.0))
        item["travelled"] = float(kept.get("travelled", 0.0))
        if kept.has("track") and track_of.has(str(kept["track"])):
            item["track"] = track_of[str(kept["track"])]
            item["s"] = float(kept.get("s", 0.0))
            if kept.has("v"):
                item["v"] = float(kept["v"])
            _layer.add_child(item["node"])
            _place_on(item)
        elif kept.has("position") and item["mode"] == "physics":
            var physics: Dictionary = WorldLoader.geogen_extras(template).get("physics", {})
            item["prop"] = "flow#%d" % item["id"]
            var xform := Transform3D(Basis.from_euler(_vec(kept.get("rotation", [0, 0, 0])) * (PI / 180.0)),
                _vec(kept["position"]))
            item["body"] = world.props.add_clone(item["node"], physics, item["prop"], _layer, xform,
                _vec(kept.get("velocity", [0, 0, 0])))
        else:
            item["node"].free()
            continue
        items[item["id"]] = item
        if float(item["data"].get("hp", 1.0)) < float((item["def"].get("data", {}) as Dictionary).get("hp", 1.0)):
            _health(item)
    for uid in sources:
        _write(uid, "alive", alive(uid))


static func _arr(v: Vector3) -> Array:
    return [snappedf(v.x, 0.0001), snappedf(v.y, 0.0001), snappedf(v.z, 0.0001)]


## {sources: [{uid, running, queued, emitted, alive, items: [names]}], items: [{id, source, item,
## mode, position, data, age, track?, s?}], tracks: [{uid, index, length, next, received,
## running}], ticks}: what flows.list returns.
func report() -> Dictionary:
    var out := {"sources": [], "items": []}
    for uid in sources:
        var entry: Dictionary = sources[uid]
        out["sources"].append({"uid": uid, "running": GeogenConditions.truthy(_read(uid, "running",
            entry["data"].get("initial", "running") == "running")), "queued": int(_read(uid, "queued", 0)),
            "emitted": entry["made"], "alive": alive(uid), "items": entry["templates"].keys(),
            "killed": entry["killed"], "wave": entry["wave"], "phase": entry["phase"]})
    for id in items:
        var item: Dictionary = items[id]
        var at: Node3D = item["body"] if item["body"] != null else item["node"]
        var p: Vector3 = at.global_position if is_instance_valid(at) else Vector3.ZERO
        var entry := {"id": id, "source": item["source"], "item": item["item"], "mode": item["mode"],
            "position": [snappedf(p.x, 0.001), snappedf(p.y, 0.001), snappedf(p.z, 0.001)],
            "data": item["data"], "age": snappedf(item["age"], 0.01)}
        if at is RigidBody3D and is_instance_valid(at):
            var v: Vector3 = at.linear_velocity
            entry["velocity"] = [snappedf(v.x, 0.001), snappedf(v.y, 0.001), snappedf(v.z, 0.001)]
            for child in at.get_children():
                if child is CollisionShape3D:            # the shape's middle: the origin may be off it
                    var c: Vector3 = child.global_position
                    entry["center"] = [snappedf(c.x, 0.001), snappedf(c.y, 0.001), snappedf(c.z, 0.001)]
                    break
        if item.has("track"):
            entry["track"] = item["track"]
            entry["s"] = item["s"]
            entry["travelled"] = item["travelled"]
        out["items"].append(entry)
    out["tracks"] = tracks.map(func(t): return {"uid": t["uid"], "index": t["index"], "length": t["length"],
        "next": t["next"], "received": t["received"], "running": GeogenConditions.truthy(_read(t["uid"], "running", true))})
    out["effectors"] = effectors.map(func(e): return {"uid": e["uid"], "taken": e["taken"], "target": e["target"],
        "enabled": GeogenConditions.truthy(_read(e["uid"], "enabled", e["data"].get("enabled", true)))})
    out["ticks"] = ticks
    return out


static func _vec(a) -> Vector3:
    return Vector3(float(a[0]), float(a[1]), float(a[2]))
