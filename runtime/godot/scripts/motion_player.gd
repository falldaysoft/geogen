class_name GeogenMotionPlayer
extends SkeletonModifier3D
## Runs motion files (motions.json beside the exports; `kind: motion`, assets/motions/) on a
## character body: a GeogenMotionRig built from the body's `extras.geogen.rig`, stepped on the
## physics tick at the rig's fixed rate (it pauses with the world), its bones written over
## whatever animates the skeleton while a motion runs. The last modifier on the skeleton.
## The body's node is the rig's body frame (the character stands at its origin).
##
## Things: a motion's roles bind to nodes in the world by name ("cup", "table.top": the grip
## points their `extras.geogen.grips` give). What a hand takes goes with it: a physics prop
## (props.gd) is taken out of the physics while held (`GeogenProps.take`) and let go of where the
## hand leaves it (a break lets it fall); any other node is moved with the hand.
##
## Control API: motion.start / motion.stop / motion.status (api.gd).

const FPS := 30

static var _motions := {}        # generated dir -> {name: spec}
static var _catalogues := {}     # generated dir -> [catalogue entry] (motion.motion_catalogue)

var rig: GeogenMotionRig
var body: Node3D
var _bones := PackedInt32Array()  # rig bone index -> skeleton bone index (-1: the skeleton lacks it)
var _clock := 0.0
var _active := false              # writing bones (a motion runs, or one just ended and is easing out)
var _things := {}                 # node -> its GeogenMotionRig.Thing


## The player on `body` (a node whose extras carry `rig`), made and attached on first use;
## null when the body has no rig or no skeleton.
static func on(body_: Node3D) -> GeogenMotionPlayer:
    var data = WorldLoader.geogen_extras(body_).get("rig")
    if not data is Dictionary:
        return null
    var skeletons := body_.find_children("*", "Skeleton3D", true, false)
    if skeletons.is_empty():
        return null
    var skeleton: Skeleton3D = skeletons[0]
    for child in skeleton.get_children():
        if child is GeogenMotionPlayer:
            return child
    var player := GeogenMotionPlayer.new()
    player.name = "Motions"
    player.body = body_
    player.rig = GeogenMotionRig.from_extras(data, FPS)
    player.rig.body_world = body_.global_transform
    player.rig.resolver = player.thing_named
    player.rig.attached.connect(player._taken)
    player.rig.released.connect(player._let_go)
    for name in player.rig.names:
        player._bones.append(skeleton.find_bone(name))
    skeleton.add_child(player)        # last: over the animation and the other modifiers
    return player


## Every motion in `generated`/motions.json by name ({} when there is none).
static func motions(generated: String) -> Dictionary:
    if not _motions.has(generated):
        var found := {}
        var path := "%s/motions.json" % generated
        if FileAccess.file_exists(path):
            var data = JSON.parse_string(FileAccess.get_file_as_string(path))
            if data is Dictionary and data.get("motions") is Dictionary:
                found = data["motions"]
        _motions[generated] = found
    return _motions[generated]


## The motion catalogue in `generated`/motions.json ([] when there is none): every motion's
## name, category, summary, params, roles and whether it needs only the body
## (geogen.motion.motion_catalogue), by category then name.
static func catalogue(generated: String) -> Array:
    if not _catalogues.has(generated):
        var found := []
        var path := "%s/motions.json" % generated
        if FileAccess.file_exists(path):
            var data = JSON.parse_string(FileAccess.get_file_as_string(path))
            if data is Dictionary and data.get("catalogue") is Array:
                found = data["catalogue"]
        _catalogues[generated] = found
    return _catalogues[generated]


## The thing `node` is to a motion (its grip points from its extras), the same one each time.
func thing_of(node: Node3D) -> GeogenMotionRig.Thing:
    if not _things.has(node) or not is_instance_valid(node):
        _things[node] = GeogenMotionRig.Thing.of(node, WorldLoader.geogen_extras(node).get("grips", {}))
    return _things[node]


## A thing in the body's world by node name (a gameplay object before a same-named part), or null.
func thing_named(name: String):
    var world := _world()
    var root: Node = world if world != null else body.get_tree().current_scene
    if root == null:
        return null
    var best: Node3D = null
    for node: Node3D in root.find_children(name, "Node3D", true, false):
        if node == body or body.is_ancestor_of(node):
            continue
        if not WorldLoader.geogen_extras(node).is_empty():
            return thing_of(node)
        if best == null:
            best = node
    return thing_of(best) if best != null else null


## Start `motion` with `values` for its params and roles (a role's value: a node, a Thing, or a
## name `thing_named` finds); "" or why it didn't.
func play(motion: Dictionary, values := {}) -> String:
    var given := {}
    for k in values:
        given[k] = thing_of(values[k]) if values[k] is Node3D else values[k]
    rig.body_world = body.global_transform
    var run := rig.start(motion, given)
    if run == null:
        return "motion '%s' can't run on this body (see the log)" % motion.get("name", "?")
    _active = true
    return ""


func stop(name: String, how := "yield") -> String:
    return rig.interrupt(name, how)


## A break: every motion drops at once and the body inertializes into `reaction` (a motion, or
## null to just stop); "" or why the reaction didn't start.
func break_(reaction = null, values := {}) -> String:
    if rig.break_(reaction, values) == null and reaction is Dictionary:
        return "motion '%s' can't run on this body (see the log)" % reaction.get("name", "?")
    _active = true
    return ""


## What the rig is doing; `bones` are where it puts them (world), which is what the modifier
## writes (a skeleton's own poses don't show a modifier's work outside its update).
func report() -> Dictionary:
    var bones := {}
    var world := rig.world()
    for b in ["Hips", "Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot"]:
        if rig.index.has(b):
            var p: Vector3 = (body.global_transform * (world[rig.index[b]] as Transform3D)).origin
            bones[b] = [snappedf(p.x, 0.0001), snappedf(p.y, 0.0001), snappedf(p.z, 0.0001)]
    var things := {}                  # every thing a motion has bound: where it is, who holds it
    for node in _things:
        if is_instance_valid(node):
            var thing: GeogenMotionRig.Thing = _things[node]
            var p := thing.world().origin
            things[String(node.name)] = {"position": [snappedf(p.x, 0.0001), snappedf(p.y, 0.0001), snappedf(p.z, 0.0001)],
                "held": thing.held}
    return {"running": rig.running(), "time": snappedf(rig.time, 0.001), "posture": rig.posture, "things": things,
        "events": rig.trace.map(func(e): return {"t": e["t"], "event": e["event"], "motion": e["motion"], "beat": e.get("beat")}),
        "bones": bones}


func _physics_process(delta: float) -> void:
    if not _active:
        return
    _clock += delta
    var steps := 0
    while _clock >= rig.dt - 1e-6 and steps < 8:
        _clock -= rig.dt
        rig.body_world = body.global_transform
        rig.step()
        steps += 1
    if steps > 0:
        _move_held()
    # Keep writing for a moment after the last motion ends: the rig eases its channels back.
    if rig.running().is_empty() and rig._offsets.is_empty():
        _active = false


# -- what the hands hold ------------------------------------------------------------------------

func _world() -> WorldLoader:
    var n: Node = body
    while n != null:
        if n is WorldLoader:
            return n
        n = n.get_parent()
    return null


## The physics prop `thing` is ({uid, props} or {}), and the transform of its body for `at` (the
## thing's own node sits in the prop's body: `at * node.transform.inverse()`).
func _prop(thing: GeogenMotionRig.Thing) -> Dictionary:
    var world := _world()
    if thing.node == null or world == null or world.props == null:
        return {}
    var uid := WorldLoader.uid_of(thing.node)
    if uid == "" or not world.props.props.has(uid) or world.props.props[uid]["node"] != thing.node:
        return {}
    return {"uid": uid, "props": world.props}


func _body_at(thing: GeogenMotionRig.Thing) -> Transform3D:
    return thing.xform * thing.node.transform.affine_inverse()


func _taken(thing: GeogenMotionRig.Thing, _end: String) -> void:
    var prop := _prop(thing)
    if not prop.is_empty():
        (prop["props"] as GeogenProps).take(prop["uid"], body, _body_at.bind(thing))


func _let_go(thing: GeogenMotionRig.Thing, to_physics: bool) -> void:
    var prop := _prop(thing)
    if not prop.is_empty():
        # Let go where the hand left it (put down: it settles; a break: it falls from there).
        (prop["props"] as GeogenProps).release(prop["uid"], _body_at(thing))
    elif thing.node != null and is_instance_valid(thing.node) and not to_physics:
        thing.node.global_transform = thing.xform


## Held things where the hands have them now.
func _move_held() -> void:
    for role in rig.attached_:
        var thing: GeogenMotionRig.Thing = rig.attached_[role]["thing"]
        if thing.node == null or not is_instance_valid(thing.node):
            continue
        var prop := _prop(thing)
        if prop.is_empty():
            thing.node.global_transform = thing.xform
        else:
            var b: RigidBody3D = (prop["props"] as GeogenProps).props[prop["uid"]]["body"]
            b.global_transform = _body_at(thing)


func _process_modification() -> void:
    if not _active and rig.running().is_empty():
        return
    var skeleton := get_skeleton()
    for i in _bones.size():
        var bone := _bones[i]
        if bone < 0:
            continue
        var local := rig.local(i)
        skeleton.set_bone_pose_position(bone, local.origin)
        skeleton.set_bone_pose_rotation(bone, local.basis.get_rotation_quaternion())
