class_name GeogenFeet
extends SkeletonModifier3D
## A standing skeletal NPC's feet on the ground that is really under them: steps, slopes, kerbs.
##
## The clips are made on flat ground at the body's floor (the NPC's origin). Each physics frame
## each foot looks down for the ground under it (a ray from well above the animated foot); while
## a foot is planted (within PLANTED of the bottom of its swing) it is moved by how far that
## ground is from the floor the clip assumed, eased (so it doesn't jump as a step comes and goes),
## up to MAX_STEP either way. The hips drop by the lowest foot's offset, so a foot going down
## can reach. A TwoBoneIK3D after this modifier bends the legs; on flat ground nothing moves.
##
## Only while standing (`standing` callable: no seat, no transition clip) and within RANGE of
## the camera, so a crowd far away costs nothing; on flat ground (both feet's last probes level
## with the floor) it probes every FLAT_EVERY frames.

const MAX_STEP := 0.35           # m: ground further from the floor than this isn't the foot's
const PLANTED := 0.06            # m: a foot this close to its lowest point is on the ground
const PROBE_UP := 0.6            # m above the floor the ray starts
const PROBE_DOWN := 0.6          # m below it the ray ends
const SPEED := 12.0              # 1/s easing of the foot offsets
const RANGE := 25.0              # m from the camera beyond which feet are left to the clip
const FLAT_EVERY := 4            # physics frames between probes while the ground is flat
const LEGS := {"LeftFoot": ["LeftUpperLeg", "LeftLowerLeg"], "RightFoot": ["RightUpperLeg", "RightLowerLeg"]}

## Whether the body stands (true) or sits, lies or is between poses (false).
var standing := Callable()
## The physics body whose collider the rays ignore (the NPC).
var exclude: RID

var _ik: TwoBoneIK3D = null
var _targets := {}
var _poles := {}
var _bones := {}                 # foot -> [upper, lower, foot] bone indices
var _hips := -1
var _animated := {}              # foot -> where the animation put it (world)
var _ground := {}                # foot -> ground height under it (world y), from the last probe
var _offset := {}                # foot -> how far it is moved up (eased, m)
var _lowest := {}                # foot -> lowest height above the floor it has been at recently (m)
var _active := false
var _flat := false               # the last probes found the floor under both feet
var _phase := 0                  # which frame of FLAT_EVERY this one probes on


static func attach(skeleton: Skeleton3D, standing_: Callable, exclude_: RID) -> GeogenFeet:
    var out := GeogenFeet.new()
    out.name = "Feet"
    out.standing = standing_
    out.exclude = exclude_
    skeleton.add_child(out)
    out._hips = skeleton.find_bone("Hips")
    out._phase = randi() % FLAT_EVERY
    out._ik = TwoBoneIK3D.new()
    out._ik.name = "FootIK"
    out._ik.influence = 0.0
    skeleton.add_child(out._ik)
    var feet := LEGS.keys().filter(func(f): return skeleton.find_bone(f) >= 0)
    out._ik.set_setting_count(feet.size())
    for i in feet.size():
        var foot: String = feet[i]
        var target := Marker3D.new()
        var pole := Marker3D.new()
        out.add_child(target)
        out.add_child(pole)
        out._targets[foot] = target
        out._poles[foot] = pole
        out._bones[foot] = [skeleton.find_bone(LEGS[foot][0]), skeleton.find_bone(LEGS[foot][1]),
            skeleton.find_bone(foot)]
        out._ik.set_root_bone_name(i, LEGS[foot][0])
        out._ik.set_middle_bone_name(i, LEGS[foot][1])
        out._ik.set_end_bone_name(i, foot)
        out._ik.set_target_node(i, out._ik.get_path_to(target))
        out._ik.set_pole_node(i, out._ik.get_path_to(pole))
    return out


func report() -> Dictionary:
    var out := {}
    for foot in _offset:
        out[foot] = {"offset": snappedf(float(_offset[foot]), 0.001),
            "ground": snappedf(float(_ground[foot]), 0.001) if _ground.has(foot) else null}
    return {"active": _active, "flat": _flat, "feet": out}


func _physics_process(delta: float) -> void:
    var skeleton := get_skeleton()
    var camera := get_viewport().get_camera_3d() if get_viewport() != null else null
    _active = skeleton != null and standing.is_valid() and bool(standing.call()) \
        and (camera == null or camera.global_position.distance_to(skeleton.global_position) < RANGE)
    if not _active:
        for foot in _offset:
            _offset[foot] = move_toward(float(_offset[foot]), 0.0, delta * SPEED * MAX_STEP)
        return
    if _flat and (Engine.get_physics_frames() + _phase) % FLAT_EVERY != 0:
        return
    var floor_y := skeleton.global_position.y
    var space := get_world_3d().direct_space_state
    var flat := true
    for foot in _bones:
        if not _animated.has(foot):
            continue
        var at: Vector3 = _animated[foot]
        var query := PhysicsRayQueryParameters3D.create(Vector3(at.x, floor_y + PROBE_UP, at.z),
            Vector3(at.x, floor_y - PROBE_DOWN, at.z))
        query.exclude = [exclude]
        var hit := space.intersect_ray(query)
        var goal := 0.0
        if not hit.is_empty() and (hit["normal"] as Vector3).y > 0.7:
            _ground[foot] = (hit["position"] as Vector3).y
            var step: float = float(_ground[foot]) - floor_y
            flat = flat and absf(step) < 0.01
            # Only a planted foot is put on the ground; a swinging one is left to the clip.
            var height := at.y - floor_y
            var lowest: float = minf(float(_lowest.get(foot, height)) + delta * 0.05, height)
            _lowest[foot] = lowest
            if absf(step) <= MAX_STEP and height - lowest < PLANTED:
                goal = step
        else:
            _ground.erase(foot)
        _offset[foot] = lerpf(float(_offset.get(foot, 0.0)), goal, 1.0 - exp(-SPEED * delta))
        flat = flat and absf(float(_offset[foot])) < 0.002
    _flat = flat


func _process_modification() -> void:
    var skeleton := get_skeleton()
    if skeleton == null or _ik == null:
        return
    var to_world := skeleton.global_transform
    for foot in _bones:
        _animated[foot] = to_world * skeleton.get_bone_global_pose(_bones[foot][2]).origin
    var drop := 0.0
    for foot in _offset:
        drop = minf(drop, float(_offset[foot]))
    var moved := false
    for foot in _offset:
        moved = moved or absf(float(_offset[foot])) > 0.002
    if not moved:
        _ik.influence = 0.0
        return
    if _hips >= 0 and drop < 0.0:      # lower the hips so the leg going down can reach
        var pose := skeleton.get_bone_global_pose(_hips)
        pose.origin += to_world.basis.inverse() * Vector3(0.0, drop, 0.0)
        skeleton.set_bone_global_pose(_hips, pose)
    for foot in _bones:
        var b: Array = _bones[foot]
        var hip := to_world * skeleton.get_bone_global_pose(b[0]).origin
        var knee := to_world * skeleton.get_bone_global_pose(b[1]).origin
        var ankle := to_world * skeleton.get_bone_global_pose(b[2]).origin
        # After the hip drop the animated foot is `drop` lower: put it back up by its own offset.
        var target: Vector3 = (_animated[foot] as Vector3) + Vector3.UP * float(_offset.get(foot, 0.0))
        (_targets[foot] as Marker3D).global_position = target
        var bend := knee - (hip + ankle) * 0.5
        var forward := to_world.basis.z.normalized()
        (_poles[foot] as Marker3D).global_position = knee + ((bend.normalized() if bend.length() > 1e-3 else forward) * 0.4)
    _ik.influence = 1.0
