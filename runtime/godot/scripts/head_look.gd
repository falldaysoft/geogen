class_name GeogenHeadLook
extends SkeletonModifier3D
## Turns a skeletal NPC's neck and head toward `target` (a world point), after the animation
## has posed the skeleton: the yaw and pitch are clamped to the NPC's attention limits and eased
## in and out with `want` (0 = don't look, 1 = look). The turn is split between the neck and the
## head. Geogen bodies face +Z in the skeleton's frame.
##
## It can also lay a one-shot gesture over the pose the clips hold (`gesture()`): a seated NPC
## waves back with its arm without leaving its seat. Only the named bones take the clip's
## rotations, eased in and out, so the rest of the body keeps its pose.

var target := Vector3.ZERO
var want := 0.0
var yaw_limit := deg_to_rad(70.0)
var pitch_limit := deg_to_rad(25.0)
var speed := 4.0                  # 1/s easing rate
var neck_share := 0.4

var _yaw := 0.0
var _pitch := 0.0
var _neck := -1
var _head := -1
var _gesture: Animation = null    # the clip laid over the pose, or null
var _gesture_tracks := {}         # bone index -> its rotation track in _gesture
var _gesture_t := 0.0
const GESTURE_EASE := 0.3         # s to blend a gesture in and out


func _ready() -> void:
    var skeleton := get_skeleton()
    if skeleton != null:
        _neck = skeleton.find_bone("Neck")
        _head = skeleton.find_bone("Head")


## How far the head is turned now (radians), for reports.
func turned() -> float:
    return absf(_yaw) + absf(_pitch)


## Play `clip` once on `bones` only, over whatever pose the body is in. False if the clip
## moves none of them.
func gesture(clip: Animation, bones: Array) -> bool:
    var skeleton := get_skeleton()
    if skeleton == null or clip == null:
        return false
    var tracks := {}
    for t in clip.get_track_count():
        if clip.track_get_type(t) != Animation.TYPE_ROTATION_3D:
            continue
        var bone_name := String(clip.track_get_path(t).get_concatenated_subnames())
        if bone_name in bones and skeleton.find_bone(bone_name) >= 0:
            tracks[skeleton.find_bone(bone_name)] = t
    if tracks.is_empty():
        return false
    _gesture = clip
    _gesture_tracks = tracks
    _gesture_t = 0.0
    return true


func gesturing() -> bool:
    return _gesture != null


func stop_gesture() -> void:
    _gesture = null


func _process_modification() -> void:
    var skeleton := get_skeleton()
    if skeleton == null:
        return
    var delta := get_process_delta_time()
    if delta <= 0.0:
        delta = get_physics_process_delta_time()
    if _gesture != null:
        _gesture_t += delta
        if _gesture_t >= _gesture.length:
            _gesture = null
        else:
            var w := clampf(minf(_gesture_t, _gesture.length - _gesture_t) / GESTURE_EASE, 0.0, 1.0)
            w = w * w * (3.0 - 2.0 * w)
            for bone in _gesture_tracks:
                var q: Quaternion = _gesture.rotation_track_interpolate(_gesture_tracks[bone], _gesture_t)
                skeleton.set_bone_pose_rotation(bone, skeleton.get_bone_pose_rotation(bone).slerp(q, w))
    if _head < 0:
        return
    var head_pose := skeleton.get_bone_global_pose(_head)
    var goal_yaw := 0.0
    var goal_pitch := 0.0
    if want > 0.0:
        var local := skeleton.global_transform.affine_inverse() * target - head_pose.origin
        goal_yaw = clampf(atan2(local.x, local.z), -yaw_limit, yaw_limit)
        goal_pitch = clampf(atan2(local.y, Vector2(local.x, local.z).length()), -pitch_limit, pitch_limit)
    var k := 1.0 - exp(-speed * delta)
    _yaw = lerpf(_yaw, goal_yaw * want, k)
    _pitch = lerpf(_pitch, goal_pitch * want, k)
    if absf(_yaw) + absf(_pitch) < 1e-4:
        return
    if _neck >= 0:
        _turn(skeleton, _neck, neck_share)
    _turn(skeleton, _head, 1.0 - neck_share)


## Rotate a bone (about its own origin, skeleton-space axes) by its share of the look.
func _turn(skeleton: Skeleton3D, bone: int, share: float) -> void:
    var pose := skeleton.get_bone_global_pose(bone)
    var turn := Basis(Vector3.UP, _yaw * share) * Basis(Vector3.RIGHT, -_pitch * share)
    pose.basis = turn * pose.basis
    skeleton.set_bone_global_pose(bone, pose)
