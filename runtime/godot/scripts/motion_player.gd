class_name GeogenMotionPlayer
extends SkeletonModifier3D
## Runs motion files (motions.json beside the exports; `kind: motion`, assets/motions/) on a
## character body: a GeogenMotionRig built from the body's `extras.geogen.rig`, stepped on the
## physics tick at the rig's fixed rate (it pauses with the world), its bones written over
## whatever animates the skeleton while a motion runs. The last modifier on the skeleton.
## The body's node is the rig's body frame (the character stands at its origin); a motion that
## holds the root (sit) moves that node. A posture change plays the body's transition clip
## (sit_down: its exported animation, with its root path from `extras.geogen.clips`) and its drape
## correctives (blend shapes) follow the pose.
##
## Things: a motion's roles bind to nodes in the world by name ("cup", "table.top": the grip
## points their `extras.geogen.grips` give). What a hand takes goes with it: a physics prop
## (props.gd) is taken out of the physics while held (`GeogenProps.take`) and let go of where the
## hand leaves it (a break lets it fall); any other node is moved with the hand.
##
## Layered (an NPC's body): only the channels a motion holds are written, eased in and out over
## whatever the body's clips and other modifiers put there, so a hand presses a button or carries a
## plate while the legs go on walking.
##
## Control API: motion.start / motion.stop / motion.status (api.gd).

const FPS := 30
const LAYER_EASE := 0.15          # s for a channel to come onto, or go off, what the clips are doing
const TRACE_KEEP := 256           # events a long-lived body keeps of its rig's trace

static var _motions := {}        # generated dir -> {name: spec}
static var _catalogues := {}     # generated dir -> [catalogue entry] (motion.motion_catalogue)

var rig: GeogenMotionRig
var body: Node3D
var _bones := PackedInt32Array()  # rig bone index -> skeleton bone index (-1: the skeleton lacks it)
var _clock := 0.0
var _active := false              # writing bones (a motion runs, or one just ended and is easing out)
var _things := {}                 # node -> its GeogenMotionRig.Thing
var _shapes := {}                 # "node:target" -> [MeshInstance3D, blend shape index]
var layered := false              # write only what motions hold, over the body's animation
var walking := false              # layered: the body is walking (a motion's `while` can quiet its free arms)
var holder: Node3D = null         # who holds what the hands take (GeogenProps.take); null: the body
var _weights := PackedFloat32Array()   # layered: how much of each rig bone is the rig's, 0..1


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
    player._weights.resize(player.rig.names.size())
    player.rig.add_transitions(transitions_of(body_))
    for key in player.rig.morph_keys:
        var at := String(key).rfind(":")
        var mesh := body_.get_node_or_null(NodePath(String(key).substr(0, at))) as MeshInstance3D
        if mesh != null:
            var i := mesh.find_blend_shape_by_name(StringName(String(key).substr(at + 1)))
            if i >= 0:
                player._shapes[key] = [mesh, i]
    skeleton.add_child(player)        # last: over the animation and the other modifiers
    return player


## The body's transition clips (`extras.geogen.clips` with `from` / `to` and a root path, their
## keys from its exported animations) and pose correctives (each `pose_<name>` animation's blend
## shapes), as GeogenMotionRig.add_transitions takes them (motion.transition_data). Morph keys are
## "<mesh node path from the body>:<blend shape>".
static func transitions_of(body_: Node3D) -> Dictionary:
    var anims := GeogenNpcAnimator._body_clips(body_)
    var transitions := []
    var pose_morphs := {}
    for clip in WorldLoader.geogen_extras(body_).get("clips", []):
        var anim: Animation = anims.get(clip["name"])
        if anim == null:
            continue
        var name := String(clip["name"])
        if name.begins_with("pose_"):
            var weights := {}
            for t in anim.get_track_count():
                if anim.track_get_type(t) == Animation.TYPE_BLEND_SHAPE and anim.track_get_key_count(t) > 0:
                    weights[String(anim.track_get_path(t))] = float(anim.track_get_key_value(t, 0))
            pose_morphs[name.substr(5)] = weights
            continue
        if not (clip.get("from") and clip.get("to")) or clip.get("loop", false):
            continue
        var tracks := {}
        var morphs := {}
        for t in anim.get_track_count():
            var kind := anim.track_get_type(t)
            var path := anim.track_get_path(t)
            var times := []
            var values := []
            for k in anim.track_get_key_count(t):
                times.append(anim.track_get_key_time(t, k))
                values.append(anim.track_get_key_value(t, k))
            if kind == Animation.TYPE_BLEND_SHAPE:
                morphs[String(path)] = {"times": times, "values": values}
                continue
            if kind != Animation.TYPE_ROTATION_3D and kind != Animation.TYPE_POSITION_3D:
                continue
            var bone := String(path.get_concatenated_subnames())
            var entry: Dictionary = tracks.get(bone, {})
            if kind == Animation.TYPE_ROTATION_3D:
                entry["times"] = times
                entry["rotations"] = values.map(func(q: Quaternion): return [q.x, q.y, q.z, q.w])
            else:
                entry["translations"] = values.map(func(v: Vector3): return [v.x, v.y, v.z])
                if not entry.has("times"):
                    entry["times"] = times
            tracks[bone] = entry
        transitions.append({"from": clip["from"], "to": clip["to"], "duration": float(clip.get("duration", anim.length)),
            "tracks": tracks, "root": clip.get("root"), "morphs": morphs})
    return {"transitions": transitions, "pose_morphs": pose_morphs}


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
        var extras := WorldLoader.geogen_extras(node)
        _things[node] = GeogenMotionRig.Thing.of(node, extras.get("grips", {}), extras.get("affordances", []))
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
    if start(motion, values) == null:
        return "motion '%s' can't run on this body (see the log)" % motion.get("name", "?")
    return ""


## `play`, handing back the run (null when it can't start).
func start(motion: Dictionary, values := {}) -> GeogenMotionRig.Run:
    var given := {}
    for k in values:
        given[k] = thing_of(values[k]) if values[k] is Node3D else values[k]
    rig.body_world = body.global_transform
    var run := rig.start(motion, given)
    if run != null:
        _active = true
    return run


func stop(name: String, how := "yield") -> String:
    return rig.interrupt(name, how)


## A break: every motion drops at once and the body inertializes into `reaction` (a motion, or
## null to just stop); "" or why the reaction didn't start.
func break_(reaction = null, values := {}) -> String:
    if rig.break_(reaction, values) == null and reaction is Dictionary:
        return "motion '%s' can't run on this body (see the log)" % reaction.get("name", "?")
    _active = true
    return ""


## Stop everything and stand the body at `at` at once (no easing): a gallery putting people back.
func reset(at: Transform3D) -> void:
    rig.break_(null)
    rig.reset_standing(at)
    body.global_transform = at
    _active = true                    # write the standing pose once


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
    var at := body.global_position
    return {"running": rig.running(), "time": snappedf(rig.time, 0.001), "posture": rig.posture, "things": things,
        "root": [snappedf(at.x, 0.0001), snappedf(at.y, 0.0001), snappedf(at.z, 0.0001)],
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
        if rig.root_moved:
            body.global_transform = rig.body_world
        steps += 1
    if steps > 0:
        _move_held()
        # (Layered, the drape follows the clips unless a motion has the hips.)
        if not layered or rig.owners.get("hips") != null:
            for key in _shapes:
                var shape: Array = _shapes[key]
                if is_instance_valid(shape[0]):
                    (shape[0] as MeshInstance3D).set_blend_shape_value(shape[1], float(rig.morphs.get(key, 0.0)))
        if rig.trace.size() > TRACE_KEEP * 2:
            rig.trace = rig.trace.slice(-TRACE_KEEP)
    var eased := not layered or _ease_layers(delta)
    # Keep writing for a moment after the last motion ends: the rig eases its channels back.
    if rig.running().is_empty() and rig._offsets.is_empty() and eased:
        _active = false


## Each bone's share toward 1 while a motion holds its channel, else toward 0; whether all are off.
func _ease_layers(delta: float) -> bool:
    var off := true
    var swing := float(rig.coupling("walk")["arm_swing"])
    for i in _weights.size():
        var channel = rig.channel_of.get(rig.names[i])
        var held: bool = channel != null and rig.owners.get(channel) != null
        var goal := 1.0 if held else 0.0
        if not held and walking and channel in ["left_hand", "right_hand"]:
            goal = 1.0 - swing               # towards the rig's hanging arm: the walk's swing, less
        _weights[i] = move_toward(_weights[i], goal, delta / LAYER_EASE)
        off = off and _weights[i] <= 0.0
    return off


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
        (prop["props"] as GeogenProps).take(prop["uid"], _holder(), _body_at.bind(thing))


func _let_go(thing: GeogenMotionRig.Thing, to_physics: bool) -> void:
    var prop := _prop(thing)
    if not prop.is_empty():
        # Let go where the hand left it (put down: it settles; a break: it falls from there).
        if (prop["props"] as GeogenProps).holder_of(prop["uid"]) == _holder():
            (prop["props"] as GeogenProps).release(prop["uid"], _body_at(thing))
    elif thing.node != null and is_instance_valid(thing.node) and not to_physics:
        thing.node.global_transform = thing.xform


func _holder() -> Node3D:
    return holder if holder != null else body


## Held things where the hands have them now. One someone else has taken from the hand (the
## player) is gone: the motions holding it stop.
func _move_held() -> void:
    for role in rig.attached_.keys():
        var thing: GeogenMotionRig.Thing = rig.attached_[role]["thing"]
        if thing.node == null or not is_instance_valid(thing.node):
            continue
        var prop := _prop(thing)
        if prop.is_empty():
            thing.node.global_transform = thing.xform
        elif (prop["props"] as GeogenProps).holder_of(prop["uid"]) != _holder():
            rig.break_(null)
            return
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
        var weight := _weights[i] if layered else 1.0
        if weight <= 0.0:
            continue
        var local := rig.local(i)
        var at := local.origin
        var turn := local.basis.get_rotation_quaternion()
        if weight < 1.0:
            at = skeleton.get_bone_pose_position(bone).lerp(at, weight)
            turn = skeleton.get_bone_pose_rotation(bone).slerp(turn, weight)
        skeleton.set_bone_pose_position(bone, at)
        skeleton.set_bone_pose_rotation(bone, turn)
