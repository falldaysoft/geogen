class_name GeogenProps
extends Node
## Physics props (extras.geogen.physics, geogen/physics.py): each becomes a RigidBody3D
## wrapping the asset node, with a box collider from the extras (the parts' own colliders
## are dropped). A prop with a `spawn` interaction is handed out by rules: stowed, it is
## hidden, frozen and out of collision; going live, it is put at `spawn_at` at rest and let
## go. A live prop that falls below the world is stowed (or, with no spawn, put back where
## it started). The player picks up and carries props with `carry` (E; E again drops).
##
## Owned by WorldLoader (`world.props`); main.gd asks it what is aimed at and drives carrying.

const LAYER := 1                 ## props collide with the world and the player
const KILL_DEPTH := 10.0         ## m below the lowest loaded model: fallen out of the world
const HOLD_SPEED := 12.0         ## m/s at most toward the hold point
const HOLD_GAIN := 14.0          ## 1/s: how hard a held prop chases the hold point
const DROP_DISTANCE := 1.4       ## m from the hold point for SNAG_SECONDS: a snagged prop is let go
const SNAG_SECONDS := 0.5

var world: WorldLoader
## uid -> {uid, node, body, data, home: Transform3D, spawn: GeogenInteraction or null}
var props := {}
var _by_body := {}               ## RigidBody3D -> uid
var held := ""                   ## uid of the prop the player carries, or ""
var _holder: Node3D = null       ## the player body carrying it
var _hold_point := Callable()    ## () -> Vector3: where the held prop goes
var _snagged := 0.0              ## s the held prop has been too far from the hold point


func _init(owner_world: WorldLoader) -> void:
    world = owner_world
    name = "Props"


## Make bodies of the physics props under ``root`` (a freshly loaded export, before
## WorldLoader adds colliders: the parts' collider nodes are dropped here).
func setup(root: Node) -> void:
    var found: Array[Node3D] = []
    for node: Node3D in root.find_children("*", "Node3D", true, false):
        if WorldLoader.geogen_extras(node).get("physics") is Dictionary and not node.get_parent() is RigidBody3D:
            found.append(node)
    for node in found:
        var data: Dictionary = WorldLoader.geogen_extras(node)["physics"]
        for mi: MeshInstance3D in node.find_children("*", "MeshInstance3D", true, false):
            if WorldLoader._is_collider(mi):
                mi.get_parent().remove_child(mi)
                mi.free()
        var uid := WorldLoader.uid_of(node)
        var home := node.global_transform
        var body := RigidBody3D.new()
        body.name = "%s_body" % node.name
        body.mass = float(data.get("mass", 10.0))
        body.collision_layer = LAYER
        body.collision_mask = 1 | 2
        body.continuous_cd = true
        body.can_sleep = false        # resting on a hatch that swings away, it must notice
        var material := PhysicsMaterial.new()
        material.friction = float(data.get("friction", 0.8))
        material.bounce = float(data.get("bounce", 0.0))
        body.physics_material_override = material
        var shape := CollisionShape3D.new()
        var box := BoxShape3D.new()
        box.size = _vec(data.get("size", [0.5, 0.5, 0.5]))
        shape.shape = box
        shape.position = _vec(data.get("center", [0, 0.25, 0]))
        body.add_child(shape)
        body.set_meta("geogen_prop", uid)
        var parent := node.get_parent()
        parent.add_child(body)
        body.global_transform = home
        node.reparent(body, true)
        var entry := {"uid": uid, "node": node, "body": body, "data": data, "home": home, "spawn": null}
        props[uid] = entry
        _by_body[body] = uid
        var spawn: GeogenInteraction = world.interactions_by_key.get("%s#spawn" % uid)
        if spawn != null and data.has("spawn_at"):
            entry["spawn"] = spawn
            # Rules can stow and hand it out again in one go, which never arrives anywhere
            # new: follow where it is heading (and instant sets, which only say they arrived).
            spawn.target_changed.connect(func(_from: String, to: String, _actor: String): _spawn_state(entry, to))
            spawn.state_entered.connect(func(state: String, _event: String): _spawn_state(entry, state))
            _spawn_state(entry, spawn.target if spawn.target != "" else spawn.state)


## Forget props under ``root`` (it is being unloaded).
func forget(root: Node) -> void:
    for uid in props.keys():
        var body: RigidBody3D = props[uid]["body"]
        if not is_instance_valid(body) or root.is_ancestor_of(body):
            if held == uid:
                held = ""
            _by_body.erase(body)
            props.erase(uid)


## Forget every prop (WorldLoader.unload freed the models).
func clear() -> void:
    props.clear()
    _by_body.clear()
    held = ""
    _holder = null


## The uid of the prop ``body`` is (a hit collider), or "".
func prop_of(body: Object) -> String:
    return str(_by_body.get(body, ""))


func can_carry(uid: String) -> bool:
    var entry: Dictionary = props.get(uid, {})
    return not entry.is_empty() and bool(entry["data"].get("carry", false)) and _live(entry) and held == ""


## Start carrying prop ``uid`` with ``holder`` (the player body); ``point`` gives the hold point.
func pick(uid: String, holder: PhysicsBody3D, point: Callable) -> bool:
    if not can_carry(uid):
        return false
    var body: RigidBody3D = props[uid]["body"]
    held = uid
    _snagged = 0.0
    _holder = holder
    _hold_point = point
    body.gravity_scale = 0.0
    body.angular_damp = 8.0
    body.axis_lock_angular_x = true       # it stays level in hand
    body.axis_lock_angular_z = true
    body.add_collision_exception_with(holder)
    return true


## Let go of what is carried (it falls from where it is).
func drop() -> void:
    if held == "" or not props.has(held):
        held = ""
        return
    var body: RigidBody3D = props[held]["body"]
    body.gravity_scale = 1.0
    body.angular_damp = 0.0
    body.axis_lock_angular_x = false
    body.axis_lock_angular_z = false
    body.linear_velocity = body.linear_velocity.limit_length(3.0)
    if is_instance_valid(_holder):
        body.remove_collision_exception_with(_holder)
    held = ""
    _holder = null


func _physics_process(delta: float) -> void:
    if held != "":
        _carry(delta)
    var floor_y := _kill_height()
    for uid in props:
        var entry: Dictionary = props[uid]
        var body: RigidBody3D = entry["body"]
        if is_instance_valid(body) and _live(entry) and body.global_position.y < floor_y:
            if uid == held:
                drop()
            if entry["spawn"] != null:
                entry["spawn"].set_state("stowed", false, "none")   # the rules hear it go
                _freeze(entry)
            else:
                _place(entry, entry["home"])


func _carry(delta: float) -> void:
    var entry: Dictionary = props.get(held, {})
    if entry.is_empty() or not _live(entry) or not _hold_point.is_valid():
        drop()
        return
    var body: RigidBody3D = entry["body"]
    var goal: Vector3 = _hold_point.call()
    var centre := body.global_transform * _vec(entry["data"].get("center", [0, 0, 0]))
    var gap := goal - centre
    _snagged = _snagged + delta if gap.length() > DROP_DISTANCE else 0.0
    if _snagged > SNAG_SECONDS:
        drop()
        return
    body.linear_velocity = (gap * HOLD_GAIN).limit_length(HOLD_SPEED)
    # Turn it to face the way the holder does (about Y only), so it doesn't tumble in hand.
    if is_instance_valid(_holder):
        var want := _holder.global_rotation.y
        var have := body.global_rotation.y
        body.angular_velocity = Vector3(0, wrapf(want - have, -PI, PI) * 6.0, 0)


## Stowed: hidden, frozen and out of collision. Live: put at the spawn point and let go.
func _spawn_state(entry: Dictionary, state: String, force := false) -> void:
    if entry.get("applied", "") == state and not force:
        return
    entry["applied"] = state
    var body: RigidBody3D = entry["body"]
    var shape := body.get_child(0) as CollisionShape3D
    if state == "live":
        var at := _vec(entry["data"]["spawn_at"])
        var home: Transform3D = entry["home"]
        _place(entry, Transform3D(home.basis, home * at))
        body.freeze = false
        shape.set_deferred("disabled", false)
    else:
        _freeze(entry)


func _freeze(entry: Dictionary) -> void:
    var body: RigidBody3D = entry["body"]
    if held == entry["uid"]:
        drop()
    body.freeze = true
    (body.get_child(0) as CollisionShape3D).set_deferred("disabled", true)
    _place(entry, entry["home"])


func _place(entry: Dictionary, xform: Transform3D) -> void:
    var body: RigidBody3D = entry["body"]
    body.linear_velocity = Vector3.ZERO
    body.angular_velocity = Vector3.ZERO
    body.global_transform = xform
    PhysicsServer3D.body_set_state(body.get_rid(), PhysicsServer3D.BODY_STATE_TRANSFORM, xform)


func _live(entry: Dictionary) -> bool:
    var spawn = entry["spawn"]
    return spawn == null or spawn.target == "live"


func _kill_height() -> float:
    var lowest := INF
    for box in world.model_aabbs:
        lowest = minf(lowest, box.position.y)
    return (lowest if lowest != INF else 0.0) - KILL_DEPTH


## Where the live props are, for a save: {uid: {position, rotation (Euler, degrees), velocity}}.
## Carried props are saved where they are, and load dropped.
func save() -> Dictionary:
    var out := {}
    for uid in props:
        var entry: Dictionary = props[uid]
        var body: RigidBody3D = entry["body"]
        if not is_instance_valid(body) or not _live(entry):
            continue
        out[uid] = {"position": _arr(body.global_position), "rotation": _arr(body.global_rotation_degrees),
            "velocity": _arr(body.linear_velocity)}
    return out


## Put the props a save names where it says (the others stay as they are: those it doesn't
## name are stowed, or at home, by their interactions).
func restore(bodies: Dictionary) -> void:
    if held != "":
        drop()
    for uid in bodies:
        var entry: Dictionary = props.get(uid, {})
        var saved = bodies[uid]
        if entry.is_empty() or not saved is Dictionary or not _live(entry):
            continue
        var rotation := Basis.from_euler(_vec(saved.get("rotation", [0, 0, 0])) * (PI / 180.0))
        _place(entry, Transform3D(rotation, _vec(saved.get("position", [0, 0, 0]))))
        (entry["body"] as RigidBody3D).linear_velocity = _vec(saved.get("velocity", [0, 0, 0]))


static func _arr(v: Vector3) -> Array:
    return [snappedf(v.x, 0.0001), snappedf(v.y, 0.0001), snappedf(v.z, 0.0001)]


## Where every prop is (the control API's world.bodies, saves).
func report() -> Array:
    var out := []
    for uid in props:
        var entry: Dictionary = props[uid]
        var body: RigidBody3D = entry["body"]
        if not is_instance_valid(body):
            continue
        var p := body.global_position
        out.append({"uid": uid, "node": String(entry["node"].name), "position": [snappedf(p.x, 0.001),
            snappedf(p.y, 0.001), snappedf(p.z, 0.001)], "yaw_deg": snappedf(rad_to_deg(body.global_rotation.y), 0.1),
            "live": _live(entry), "held": held == uid, "sleeping": body.sleeping,
            "speed": snappedf(body.linear_velocity.length(), 0.01)})
    return out


static func _vec(a) -> Vector3:
    return Vector3(float(a[0]), float(a[1]), float(a[2]))
