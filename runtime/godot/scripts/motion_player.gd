class_name GeogenMotionPlayer
extends SkeletonModifier3D
## Runs motion files (motions.json beside the exports; `kind: motion`, assets/motions/) on a
## character body: a GeogenMotionRig built from the body's `extras.geogen.rig`, stepped on the
## physics tick at the rig's fixed rate (it pauses with the world), its bones written over
## whatever animates the skeleton while a motion runs. The last modifier on the skeleton.
## The body's node is the rig's body frame (the character stands at its origin).
##
## Control API: motion.start / motion.stop / motion.status (api.gd).

const FPS := 30

static var _motions := {}        # generated dir -> {name: spec}

var rig: GeogenMotionRig
var body: Node3D
var _bones := PackedInt32Array()  # rig bone index -> skeleton bone index (-1: the skeleton lacks it)
var _clock := 0.0
var _active := false              # writing bones (a motion runs, or one just ended and is easing out)


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


## Start `motion` with `values` for its params; "" or why it didn't.
func play(motion: Dictionary, values := {}) -> String:
    var run := rig.start(motion, values)
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
    return {"running": rig.running(), "time": snappedf(rig.time, 0.001), "posture": rig.posture,
        "events": rig.trace.map(func(e): return {"t": e["t"], "event": e["event"], "motion": e["motion"], "beat": e.get("beat")}),
        "bones": bones}


func _physics_process(delta: float) -> void:
    if not _active:
        return
    _clock += delta
    var steps := 0
    while _clock >= rig.dt - 1e-6 and steps < 8:
        _clock -= rig.dt
        rig.step()
        steps += 1
    # Keep writing for a moment after the last motion ends: the rig eases its channels back.
    if rig.running().is_empty() and rig._offsets.is_empty():
        _active = false


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
