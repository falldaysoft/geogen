## Print Godot's SkeletonProfileHumanoid as JSON (source of assets/skeletons/humanoid_profile.json):
##   godot --headless --path runtime/godot --script res://tools/dump_skeleton_profile.gd | grep ^PROFILE
extends SceneTree

func _init() -> void:
    var p := SkeletonProfileHumanoid.new()
    var bones := []
    for i in p.bone_size:
        var t: Transform3D = p.get_reference_pose(i)
        var b := t.basis
        var q := b.get_rotation_quaternion()
        bones.append({"name": String(p.get_bone_name(i)), "parent": String(p.get_bone_parent(i)),
            "tail": String(p.get_bone_tail(i)), "group": String(p.get_group(i)),
            "tail_direction": p.get_tail_direction(i), "required": p.is_required(i),
            "reference_pose": {"translation": [t.origin.x, t.origin.y, t.origin.z], "rotation": [q.x, q.y, q.z, q.w]}})
    var groups := []
    for g in p.group_size:
        groups.append(String(p.get_group_name(g)))
    var out := {"profile": "SkeletonProfileHumanoid", "godot": Engine.get_version_info().string,
        "root_bone": String(p.root_bone), "scale_base_bone": String(p.scale_base_bone), "groups": groups, "bones": bones}
    print("PROFILE " + JSON.stringify(out, "", false))
    quit()
