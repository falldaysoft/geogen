class_name GeogenHeadLook
extends SkeletonModifier3D
## Turns a skeletal NPC's neck and head toward `target` (a world point), after the animation
## has posed the skeleton: the yaw and pitch are clamped to the NPC's attention limits and eased
## in and out with `want` (0 = don't look, 1 = look). The turn is split between the neck and the
## head. Geogen bodies face +Z in the skeleton's frame.

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


func _ready() -> void:
    var skeleton := get_skeleton()
    if skeleton != null:
        _neck = skeleton.find_bone("Neck")
        _head = skeleton.find_bone("Head")


## How far the head is turned now (radians), for reports.
func turned() -> float:
    return absf(_yaw) + absf(_pitch)


func _process_modification() -> void:
    var skeleton := get_skeleton()
    if skeleton == null or _head < 0:
        return
    var delta := get_process_delta_time()
    if delta <= 0.0:
        delta = get_physics_process_delta_time()
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
