class_name GeogenMechanisms
extends Node
## Mechanisms (extras.geogen.mechanism, geogen/mechanism.py): an asset whose parts form
## several rigid bodies held by joints, some motorised (a carousel's platform, a Ferris
## wheel and its swinging gondolas, a drop tower's carriage). Each body becomes a
## RigidBody3D beside the asset node, in the asset's frame, with the body's parts moved
## under it and their exported collider meshes turned into convex shapes (a dynamic body
## can't be concave). Each joint is a Godot joint at the body's pivot: HingeJoint3D (its
## local Z the axis), Generic6DOFJoint3D locked to one linear axis (a slide; SliderJoint3D
## has no motor) or PinJoint3D; the two bodies it holds don't collide with each other, and a
## joint with no `to` holds the body to the world (the asset's static parts are world
## colliders). A body never collides with its own asset's static parts, the frame it is
## mounted on (a deck's hull encloses the mast it turns about, a car's its tower): those are
## collision exceptions, added once WorldLoader has built the colliders (`exclude_frames`).
## The asset's interactions run the motors with `drive` motions (interaction.gd calls `drive`).
##
## Standing on a body carries the player with it (platform velocity, as the test chamber's
## platform does). An affordance with a `body` (`ride`) is kept where its body is, and the
## player's pose follows it (player.gd), so a seated rider goes round too.
##
## The navmesh is baked from static colliders, so each body also gets a static shadow under
## the asset node (`nav_shadows`, on no layer): a box over each of its shapes, at least
## OBSTACLE_HEIGHT tall, which world_navigation.gd puts on layer 1 for the bake only. So a
## ride is an obstacle NPCs path round, never floor they walk across (a deck that turns):
## they reach its seats from the seats' approach points, off the body.
##
## Owned by WorldLoader (`world.mechanisms`). Bodies aren't saved: a load puts a mechanism
## back at its exported pose and the interactions' restored states run the motors again.

const LAYER := 1                 ## bodies collide with the world, props and NPCs (not the player: see Player.LAYER)
## Godot's hinge motor and limits turn the other way from the right-hand rule about the
## joint's Z axis; this keeps `drive` speeds and `limits` in the convention of `rotate`
## motions (degrees about the axis, right-handed).
const HINGE_SIGN := -1.0
const OBSTACLE_HEIGHT := 2.0     ## m: a body's navmesh shadow is at least this tall (never floor)

var world: WorldLoader
## asset uid -> {node, bodies: {name: RigidBody3D}, joints: {name: Joint3D}}
var mechanisms := {}
var _by_body := {}               ## RigidBody3D -> [uid, body name]
## Affordances riding on bodies: [{entry, body, local: Transform3D}]
var _riding: Array[Dictionary] = []
## Static copies of the bodies' shapes at their exported pose, for the navmesh bake.
var _shadows: Array[StaticBody3D] = []


func _init(owner_world: WorldLoader) -> void:
    world = owner_world
    name = "Mechanisms"


## Make bodies and joints of the mechanisms under ``root`` (a freshly loaded export, before
## WorldLoader adds colliders: the bodies' parts' collider meshes become their shapes here).
func setup(root: Node) -> void:
    var found: Array[Node3D] = []
    for node: Node3D in root.find_children("*", "Node3D", true, false):
        if WorldLoader.geogen_extras(node).get("mechanism") is Dictionary:
            found.append(node)
    for node in found:
        _setup_one(node)


func _setup_one(asset: Node3D) -> void:
    var data: Dictionary = WorldLoader.geogen_extras(asset)["mechanism"]
    var uid := WorldLoader.uid_of(asset)
    var parent := asset.get_parent()
    var frame := asset.global_transform
    var specs: Dictionary = data.get("bodies", {})
    var bodies := {}
    var joints := {}
    for bname in specs:
        var spec: Dictionary = specs[bname]
        var body := RigidBody3D.new()
        body.name = "%s_%s" % [asset.name, bname]
        body.mass = float(spec.get("mass", 100.0))
        body.collision_layer = LAYER
        body.collision_mask = 1 | 2      # not Player.LAYER: the player stands on and rides a body it can't block
        body.continuous_cd = true
        body.can_sleep = false
        var material := PhysicsMaterial.new()
        material.friction = float(spec.get("friction", 0.8))
        material.bounce = float(spec.get("bounce", 0.0))
        body.physics_material_override = material
        var damping: Dictionary = spec.get("damping", {})
        if not damping.is_empty():
            body.linear_damp_mode = RigidBody3D.DAMP_MODE_REPLACE
            body.angular_damp_mode = RigidBody3D.DAMP_MODE_REPLACE
            body.linear_damp = float(damping.get("linear", 0.0))
            body.angular_damp = float(damping.get("angular", 0.0))
        parent.add_child(body)
        body.global_transform = frame
        var shapes := 0
        for n in spec.get("nodes", []):
            var part := GeogenInteraction._find(asset, str(n))
            if part == null:
                push_warning("geogen: mechanism %s body %s: no node '%s'" % [asset.name, bname, n])
                continue
            var meshes: Array[MeshInstance3D] = []
            if part is MeshInstance3D:
                meshes.append(part)
            for mi: MeshInstance3D in part.find_children("*", "MeshInstance3D", true, false):
                meshes.append(mi)
            for mi in meshes:
                if not WorldLoader._is_collider(mi):
                    continue
                var shape: Shape3D = mi.mesh.create_convex_shape(true, false) if mi.mesh != null else null
                if shape != null:
                    var col := CollisionShape3D.new()
                    col.shape = shape
                    body.add_child(col)
                    col.global_transform = mi.global_transform
                    shapes += 1
                mi.get_parent().remove_child(mi)
                mi.free()
            part.reparent(body, true)
        if shapes == 0:
            push_warning("geogen: mechanism %s body %s has no collider: it has no shape" % [asset.name, bname])
        var shadow := StaticBody3D.new()
        shadow.name = "%s_%s_nav" % [asset.name, bname]
        shadow.collision_layer = 0
        shadow.collision_mask = 0
        asset.add_child(shadow)
        shadow.global_transform = body.global_transform
        for col: CollisionShape3D in body.find_children("*", "CollisionShape3D", false, false):
            var box := col.shape.get_debug_mesh().get_aabb()      # the shape's bounds, in its own frame
            if box.size.length() < 1e-6:
                continue
            var copy := CollisionShape3D.new()
            var block := BoxShape3D.new()
            block.size = Vector3(box.size.x, maxf(box.size.y, OBSTACLE_HEIGHT), box.size.z)
            copy.shape = block
            copy.transform = col.transform * Transform3D(Basis.IDENTITY, box.position + Vector3(box.size.x, block.size.y, box.size.z) / 2.0)
            shadow.add_child(copy)
        _shadows.append(shadow)
        body.set_meta("geogen_mechanism", [uid, bname])
        bodies[bname] = body
        _by_body[body] = [uid, bname]
    for bname in bodies:
        var spec: Dictionary = specs[bname]
        if not spec.get("joint") is Dictionary:
            continue
        var j: Dictionary = spec["joint"]
        var joint := _make_joint(j)
        joint.name = "%s_%s_joint" % [asset.name, bname]
        joint.set_meta("geogen_joint", j)
        parent.add_child(joint)
        var axis := _vec(j.get("axis", [0, 1, 0])).normalized()
        var basis := _hinge_basis(axis) if j["type"] == "hinge" else _slide_basis(axis)
        joint.global_transform = frame * Transform3D(basis, _vec(j.get("pivot", [0, 0, 0])))
        joint.node_a = bodies[bname].get_path()
        if j.has("to") and bodies.has(j["to"]):
            joint.node_b = bodies[j["to"]].get_path()
        _set_limits(joint, j)
        var motor: Dictionary = j.get("motor", {})
        if not motor.is_empty():
            set_motor(joint, 0.0, true)        # braked until an interaction drives it
        joints[bname] = joint
    # Copies of one arrayed body (chair_0 .. chair_7, hung from the same body) don't collide
    # with each other: swing chairs stopped abruptly hooked their chains together.
    var names: Array = bodies.keys()
    for i in names.size():
        for k in range(i + 1, names.size()):
            var a: String = names[i]
            var b: String = names[k]
            if a.rsplit("_", true, 1)[0] == b.rsplit("_", true, 1)[0] \
                    and specs[a].get("joint", {}).get("to") == specs[b].get("joint", {}).get("to"):
                (bodies[a] as RigidBody3D).add_collision_exception_with(bodies[b])
    mechanisms[uid] = {"node": asset, "bodies": bodies, "joints": joints}
    asset.set_meta("geogen_mechanism", uid)
    asset.set_meta("geogen_joints", joints)
    asset.set_meta("geogen_bodies", bodies)


static func _make_joint(j: Dictionary) -> Joint3D:
    match str(j.get("type", "hinge")):
        "slide":
            var slide := Generic6DOFJoint3D.new()
            # Only linear X (the axis) is free; the rest is locked.
            for setter in [slide.set_flag_y, slide.set_flag_z]:
                setter.call(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_LIMIT, true)
            for setter in [slide.set_param_y, slide.set_param_z]:
                setter.call(Generic6DOFJoint3D.PARAM_LINEAR_LOWER_LIMIT, 0.0)
                setter.call(Generic6DOFJoint3D.PARAM_LINEAR_UPPER_LIMIT, 0.0)
            for setter in [slide.set_flag_x, slide.set_flag_y, slide.set_flag_z]:
                setter.call(Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_LIMIT, true)
            for setter in [slide.set_param_x, slide.set_param_y, slide.set_param_z]:
                setter.call(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, 0.0)
                setter.call(Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, 0.0)
            slide.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_LIMIT, false)
            return slide
        "pin":
            return PinJoint3D.new()
    return HingeJoint3D.new()


static func _set_limits(joint: Joint3D, j: Dictionary) -> void:
    if not j.get("limits") is Array:
        return
    var lo := float(j["limits"][0])
    var hi := float(j["limits"][1])
    if joint is HingeJoint3D:
        var hinge := joint as HingeJoint3D
        hinge.set_flag(HingeJoint3D.FLAG_USE_LIMIT, true)
        # The sign flips the interval: [lo, hi] right-handed is [-hi, -lo] for Godot.
        hinge.set_param(HingeJoint3D.PARAM_LIMIT_LOWER, deg_to_rad(minf(HINGE_SIGN * lo, HINGE_SIGN * hi)))
        hinge.set_param(HingeJoint3D.PARAM_LIMIT_UPPER, deg_to_rad(maxf(HINGE_SIGN * lo, HINGE_SIGN * hi)))
    elif joint is Generic6DOFJoint3D:
        var slide := joint as Generic6DOFJoint3D
        slide.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_LIMIT, true)
        slide.set_param_x(Generic6DOFJoint3D.PARAM_LINEAR_LOWER_LIMIT, lo)
        slide.set_param_x(Generic6DOFJoint3D.PARAM_LINEAR_UPPER_LIMIT, hi)


## A frame whose Z is ``axis`` (a HingeJoint3D turns about its local Z).
static func _hinge_basis(axis: Vector3) -> Basis:
    var ref := Vector3.UP if absf(axis.dot(Vector3.UP)) < 0.9 else Vector3.RIGHT
    var x := ref.cross(axis).normalized()
    var y := axis.cross(x).normalized()
    return Basis(x, y, axis)


## A frame whose X is ``axis`` (the free linear axis of a slide).
static func _slide_basis(axis: Vector3) -> Basis:
    var ref := Vector3.UP if absf(axis.dot(Vector3.UP)) < 0.9 else Vector3.RIGHT
    var z := axis.cross(ref).normalized()
    var y := z.cross(axis).normalized()
    return Basis(axis, y, z)


## Run a joint's motor at ``speed`` (deg/s for a hinge, m/s for a slide; 0 brakes), or turn
## it off (``on`` false: the body moves as physics takes it). The force is the joint's.
static func set_motor(joint: Joint3D, speed: float, on: bool) -> void:
    var j: Dictionary = joint.get_meta("geogen_joint", {})
    var force := float(j.get("motor", {}).get("force", 10000.0))
    if joint is HingeJoint3D:
        var hinge := joint as HingeJoint3D
        hinge.set_flag(HingeJoint3D.FLAG_ENABLE_MOTOR, on)
        hinge.set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, HINGE_SIGN * deg_to_rad(speed))
        hinge.set_param(HingeJoint3D.PARAM_MOTOR_MAX_IMPULSE, force)
    elif joint is Generic6DOFJoint3D:
        var slide := joint as Generic6DOFJoint3D
        slide.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_MOTOR, on)
        slide.set_param_x(Generic6DOFJoint3D.PARAM_LINEAR_MOTOR_TARGET_VELOCITY, speed)
        slide.set_param_x(Generic6DOFJoint3D.PARAM_LINEAR_MOTOR_FORCE_LIMIT, force)


## A `drive` motion's effect (interaction.gd): the motor of ``asset``'s body ``body_name``.
static func drive(asset: Node3D, body_name: String, speed: float, on: bool) -> void:
    var joints: Dictionary = asset.get_meta("geogen_joints", {})
    var joint: Joint3D = joints.get(body_name)
    if joint != null:
        set_motor(joint, speed, on)


## An affordance ``entry`` (WorldLoader._collect_affordances) on body ``body_name`` of
## ``asset``: from now on its position, yaw and approach follow the body, `boardable` says
## whether the ride is still at the ground, and a pose on it (the player's, an NPC's) rides
## along (`follow` and `local` in the entry). Returns whether the body was found.
func ride(entry: Dictionary, asset: Node3D, body_name: String) -> bool:
    var bodies: Dictionary = asset.get_meta("geogen_bodies", {})
    var body: RigidBody3D = bodies.get(body_name)
    if body == null:
        return false
    var seat := Transform3D(Basis(Vector3.UP, deg_to_rad(float(entry["yaw_deg"]))), entry["position"])
    var local := body.global_transform.affine_inverse() * seat
    entry["follow"] = body
    entry["local"] = local
    entry["boardable"] = true
    _riding.append({"entry": entry, "body": body, "local": local, "low_y": float((entry["position"] as Vector3).y)})
    return true


## Where a riding affordance is now: its seat transform in the world.
static func seat_now(entry: Dictionary) -> Transform3D:
    var body: RigidBody3D = entry.get("follow")
    if body == null or not is_instance_valid(body):
        return Transform3D(Basis(Vector3.UP, deg_to_rad(float(entry["yaw_deg"]))), entry["position"])
    return body.global_transform * (entry["local"] as Transform3D)


func _physics_process(_delta: float) -> void:
    for r in _riding:
        var body: RigidBody3D = r["body"]
        if not is_instance_valid(body):
            continue
        var seat: Transform3D = body.global_transform * (r["local"] as Transform3D)
        var entry: Dictionary = r["entry"]
        var step := seat.origin - (entry["position"] as Vector3)
        var yaw := rad_to_deg(seat.basis.get_euler().y)
        var turn := angle_difference(deg_to_rad(float(entry["yaw_deg"])), deg_to_rad(yaw))
        entry["position"] = seat.origin
        entry["yaw_deg"] = yaw
        entry["npc_yaw_deg"] = float(entry.get("npc_yaw_deg", 0.0)) + rad_to_deg(turn)
        entry["approach"] = seat.origin + Basis(Vector3.UP, turn) * ((entry["approach"] as Vector3) - seat.origin + step)
        # Boardable while the ride is still and the seat is as low as it gets (the lowest it
        # has been seen: a gondola exported high up qualifies once the wheel has brought it
        # down): NPCs only take a seat then, and stay in it until then (npc.gd).
        r["low_y"] = minf(float(r["low_y"]), seat.origin.y)
        entry["boardable"] = body.linear_velocity.length() < 0.05 and body.angular_velocity.length() < 0.02 \
            and seat.origin.y - float(r["low_y"]) < 0.5


## The bodies' static shadows (their shapes at the exported pose) for the navmesh bake.
func nav_shadows() -> Array[StaticBody3D]:
    _shadows = _shadows.filter(func(s): return is_instance_valid(s))
    return _shadows


## After WorldLoader._add_collision: each body ignores the collider bodies of its own
## asset's static parts (the frame), which its parts are authored to fit round.
func exclude_frames() -> void:
    for uid in mechanisms:
        var m: Dictionary = mechanisms[uid]
        var asset: Node3D = m["node"]
        if not is_instance_valid(asset):
            continue
        var frame: Array[Node] = asset.find_children("*", "PhysicsBody3D", true, false)
        for body: RigidBody3D in m["bodies"].values():
            for col in frame:
                body.add_collision_exception_with(col)


## Forget mechanisms under ``root`` (it is being unloaded; their nodes go with it).
func forget(root: Node) -> void:
    for uid in mechanisms.keys():
        var node: Node3D = mechanisms[uid]["node"]
        if not is_instance_valid(node) or root.is_ancestor_of(node):
            for body in mechanisms[uid]["bodies"].values():
                _by_body.erase(body)
            mechanisms.erase(uid)
    _riding = _riding.filter(func(r): return is_instance_valid(r["body"]) and not root.is_ancestor_of(r["body"]))


func clear() -> void:
    mechanisms.clear()
    _by_body.clear()
    _riding.clear()
    _shadows.clear()


## The uid and body name of a mechanism body, or [] for any other object.
func body_of(body: Object) -> Array:
    return _by_body.get(body, [])


## Every mechanism body: uid (<asset uid>/<body>), node, position (its centre of mass, world),
## yaw_deg, tilt_deg (how far its up axis leans from vertical: a gondola hanging level reads 0),
## speed (m/s), angular_speed (deg/s), and the motor's state (on, speed) when it has one.
func report() -> Array:
    var out := []
    for uid in mechanisms:
        var m: Dictionary = mechanisms[uid]
        for bname in m["bodies"]:
            var body: RigidBody3D = m["bodies"][bname]
            if not is_instance_valid(body):
                continue
            var state := PhysicsServer3D.body_get_direct_state(body.get_rid())
            var com := body.global_position + (state.center_of_mass if state != null else Vector3.ZERO)
            var row := {
                "uid": "%s/%s" % [uid, bname], "node": String(m["node"].name), "body": bname,
                "position": _arr(com), "yaw_deg": rad_to_deg(body.rotation.y),
                "tilt_deg": rad_to_deg(body.global_basis.y.normalized().angle_to(Vector3.UP)),
                "speed": body.linear_velocity.length(), "angular_speed": rad_to_deg(body.angular_velocity.length()),
            }
            var joint: Joint3D = m["joints"].get(bname)
            if joint != null and joint.has_meta("geogen_joint") and joint.get_meta("geogen_joint").has("motor"):
                if joint is HingeJoint3D:
                    row["motor"] = {"on": (joint as HingeJoint3D).get_flag(HingeJoint3D.FLAG_ENABLE_MOTOR),
                        "speed": HINGE_SIGN * rad_to_deg((joint as HingeJoint3D).get_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY))}
                elif joint is Generic6DOFJoint3D:
                    var slide := joint as Generic6DOFJoint3D
                    row["motor"] = {"on": slide.get_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_MOTOR),
                        "speed": slide.get_param_x(Generic6DOFJoint3D.PARAM_LINEAR_MOTOR_TARGET_VELOCITY)}
            out.append(row)
    return out


static func _arr(v: Vector3) -> Array:
    return [snappedf(v.x, 0.001), snappedf(v.y, 0.001), snappedf(v.z, 0.001)]


static func _vec(a) -> Vector3:
    return Vector3(float(a[0]), float(a[1]), float(a[2]))
