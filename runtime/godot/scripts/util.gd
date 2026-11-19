class_name GeogenUtil
extends RefCounted
## Helpers the runtime's scripts share: vectors to and from JSON lists, at the precisions the
## reports use; the bounds of a subtree's meshes; and load-phase timings.


## [x, y, z] to the millimetre: positions in reports and traces.
static func mm(v: Vector3) -> Array:
    return [snappedf(v.x, 0.001), snappedf(v.y, 0.001), snappedf(v.z, 0.001)]


## [x, y, z] to the centimetre: NPC reports and memories.
static func cm(v: Vector3) -> Array:
    return [snappedf(v.x, 0.01), snappedf(v.y, 0.01), snappedf(v.z, 0.01)]


## [x, y, z] to a tenth of a millimetre: what is saved and restored (props, flows).
static func fine(v: Vector3) -> Array:
    return [snappedf(v.x, 0.0001), snappedf(v.y, 0.0001), snappedf(v.z, 0.0001)]


## A Vector3 from a JSON [x, y, z].
static func vec(a) -> Vector3:
    return Vector3(float(a[0]), float(a[1]), float(a[2]))


## Global bounds of the meshes in ``node``'s subtree, ``node`` itself included, that ``keep``
## (MeshInstance3D -> bool) accepts; the debug collider meshes never count. AABB() if none do.
static func mesh_bounds(node: Node, keep := Callable()) -> AABB:
    var out := AABB()
    var first := true
    var meshes: Array = node.find_children("*", "MeshInstance3D", true, false)
    if node is MeshInstance3D:
        meshes.append(node)
    for mi: MeshInstance3D in meshes:
        if mi.is_in_group("geogen_collider_debug") or (keep.is_valid() and not keep.call(mi)):
            continue
        var box: AABB = mi.global_transform * mi.get_aabb()
        out = box if first else out.merge(box)
        first = false
    return out


## Add the milliseconds since ``since`` (Time.get_ticks_usec) to ``timings[phase]``; returns now.
static func lap(timings: Dictionary, phase: String, since: int) -> int:
    var now := Time.get_ticks_usec()
    timings[phase] = snappedf(float(timings.get(phase, 0.0)) + (now - since) / 1000.0, 0.1)
    return now
