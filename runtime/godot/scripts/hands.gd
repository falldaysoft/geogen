class_name GeogenHands
extends SkeletonModifier3D
## A skeletal NPC's hands on the things that are really there, by arm IK over the animation.
##
## Resting (`rests`): a pose clip with hands on a table top or a sill says which (extras.geogen.clips
## `rests`), made for a default table or sill height and reach; tables, sills and how far the seat
## is from them differ. While such a pose plays, each resting hand looks down for what is under
## it, and, finding nothing, a little further ahead (rays in the physics frame, up to REACH forward
## along the body's facing): it goes onto the first top it finds within MAX_SHIFT of its height.
## Eased in and out with the pose.
##
## Reaching (`reach`): one hand goes to a point on a moving node (a door's handle) for a while,
## easing in over REACH_EASE, following the node (the handle swings with the leaf), easing out.
##
## A TwoBoneIK3D after this modifier moves the hands, keeping the elbows bent the way the
## animation bends them. This runs after the animation (and the head look) and reads the animated
## hands, so it must come before the TwoBoneIK3D among the skeleton's children (`attach`).

const MAX_SHIFT := 0.15          # m: further up or down than this, it isn't the hand's support
const REACH := [0.0, 0.06, 0.12, 0.18, 0.24]   # m ahead of the hand the probes look
const INSET := 0.05              # m in from a top's edge found ahead, where the hand rests
const PALM := 0.015              # m from the hand bone's origin down to the palm
const PROBE_UP := 0.2            # m above the animated hand the ray starts
const PROBE_DOWN := 0.35         # m below it the ray ends
const SPEED := 6.0               # 1/s easing rate of resting
const REACH_EASE := 0.3          # s to reach out, and to let go
const GRIP := 0.03               # m short of the point the hand bone stops (the fingers are round it)
const ARMS := {"LeftHand": ["LeftUpperArm", "LeftLowerArm"], "RightHand": ["RightUpperArm", "RightLowerArm"]}

## Pose clip name -> its resting hands (extras `rests`).
var rests := {}
## Returns the clip playing (GeogenNpcAnimator.current).
var current := Callable()
## The physics body whose collider the rays ignore (the NPC).
var exclude: RID

var _ik: TwoBoneIK3D = null
var _targets := {}               # hand -> Marker3D the IK reaches for
var _poles := {}                 # hand -> Marker3D the elbow bends toward
var _bones := {}                 # hand -> [upper, lower, hand] bone indices
var _animated := {}              # hand -> where the animation put it (world), last modification
var _support := {}               # hand -> where on its support it goes (world), from the last probe
var _want := 0.0                 # resting weight
var _raw := {}                   # hand -> from it to its support (world), last resting
var _shift := {}                 # hand -> how far it is being moved now (m, for reports)
var _reach := {}                 # {hand, node, local, t, seconds}: a hand reaching for a point
var _reaches := 0                # how many reaches have started (for reports)


## Make the modifier and the IK it drives under `skeleton` (after whatever is there).
static func attach(skeleton: Skeleton3D, rests_: Dictionary, current_: Callable, exclude_: RID) -> GeogenHands:
    var out := GeogenHands.new()
    out.name = "Hands"
    out.rests = rests_
    out.current = current_
    out.exclude = exclude_
    skeleton.add_child(out)
    out._ik = TwoBoneIK3D.new()
    out._ik.name = "HandIK"
    out._ik.influence = 0.0
    skeleton.add_child(out._ik)
    var hands := ARMS.keys().filter(func(h): return skeleton.find_bone(h) >= 0)
    out._ik.set_setting_count(hands.size())
    for i in hands.size():
        var hand: String = hands[i]
        var target := Marker3D.new()
        var pole := Marker3D.new()
        target.name = hand + "Target"
        pole.name = hand + "Pole"
        out.add_child(target)
        out.add_child(pole)
        out._targets[hand] = target
        out._poles[hand] = pole
        out._bones[hand] = [skeleton.find_bone(ARMS[hand][0]), skeleton.find_bone(ARMS[hand][1]),
            skeleton.find_bone(hand)]
        out._ik.set_root_bone_name(i, ARMS[hand][0])
        out._ik.set_middle_bone_name(i, ARMS[hand][1])
        out._ik.set_end_bone_name(i, hand)
        out._ik.set_target_node(i, out._ik.get_path_to(target))
        out._ik.set_pole_node(i, out._ik.get_path_to(pole))
    return out


## Reach `hand` for `local` (a point in `node`'s frame, followed as it moves) for `seconds`,
## easing in and out. The hand on that side of the body when `hand` is "": the point's side.
## Returns the hand used ("" when the body has no such arm).
func reach(hand: String, node: Node3D, local: Vector3, seconds: float) -> String:
    var skeleton := get_skeleton()
    if skeleton == null or node == null:
        return ""
    if hand == "":
        var point := skeleton.global_transform.affine_inverse() * (node.global_transform * local)
        hand = "LeftHand" if point.x > 0.0 else "RightHand"   # bodies' +X is their left
    if not _bones.has(hand):
        return ""
    _reach = {"hand": hand, "node": node, "local": local, "t": 0.0, "seconds": maxf(seconds, 2.0 * REACH_EASE)}
    _reaches += 1
    return hand


## The hands resting now: where each is (the animation's), where on its support it goes, and how
## far it is being moved there (m); the hand reaching now, and for what.
func report() -> Dictionary:
    var out := {}
    for hand in _resting():
        var at: Vector3 = _animated.get(hand, Vector3.ZERO)
        var on: Variant = _support.get(hand)
        out[hand] = {"support": _v(on) if on != null else null, "shift": snappedf(float(_shift.get(hand, 0.0)), 0.001),
            "at": _v(at)}
    var reaching := {}
    if not _reach.is_empty() and is_instance_valid(_reach["node"]):
        var goal: Vector3 = (_reach["node"] as Node3D).global_transform * (_reach["local"] as Vector3)
        reaching = {"hand": _reach["hand"], "weight": snappedf(_reach_weight(), 0.01), "goal": _v(goal),
            "at": _v(_animated.get(_reach["hand"], Vector3.ZERO))}
    return {"weight": snappedf(_want, 0.01), "hands": out, "reaching": reaching, "reaches": _reaches}


static func _v(p: Vector3) -> Array:
    return [snappedf(p.x, 0.001), snappedf(p.y, 0.001), snappedf(p.z, 0.001)]


func _resting() -> Array:
    if not current.is_valid():
        return []
    return rests.get(str(current.call()), [])


func _reach_weight() -> float:
    if _reach.is_empty():
        return 0.0
    var t: float = _reach["t"]
    var w := clampf(minf(t, float(_reach["seconds"]) - t) / REACH_EASE, 0.0, 1.0)
    return w * w * (3.0 - 2.0 * w)


func _physics_process(delta: float) -> void:
    if not _reach.is_empty():
        _reach["t"] = float(_reach["t"]) + delta
        if float(_reach["t"]) >= float(_reach["seconds"]) or not is_instance_valid(_reach["node"]):
            _reach = {}
    var resting := _resting()
    var goal := 1.0 if not resting.is_empty() else 0.0
    _want = move_toward(_want, goal, delta * SPEED)
    if resting.is_empty():
        return
    var space := get_world_3d().direct_space_state
    var ahead := get_skeleton().global_basis.z if get_skeleton() != null else Vector3.BACK
    ahead = Vector3(ahead.x, 0.0, ahead.z).normalized()       # bodies face +Z in the skeleton's frame
    for hand in resting:
        if not _animated.has(hand):
            continue
        var at: Vector3 = _animated[hand]
        _support.erase(hand)
        for d in REACH:
            var top: Variant = _top(space, at + ahead * float(d), at.y)
            if top != null:
                # Found ahead of the hand: an edge. Rest a little way in from it, if the top goes on.
                var inside: Variant = _top(space, at + ahead * (float(d) + INSET), at.y) if d > 0.0 else null
                _support[hand] = inside if inside != null and absf((inside as Vector3).y - (top as Vector3).y) < 0.01 else top
                break


## The top under `probe` within MAX_SHIFT of a hand at height `y` (world point), or null.
func _top(space: PhysicsDirectSpaceState3D, probe: Vector3, y: float) -> Variant:
    var query := PhysicsRayQueryParameters3D.create(probe + Vector3.UP * PROBE_UP, probe + Vector3.DOWN * PROBE_DOWN)
    query.exclude = [exclude]
    var hit := space.intersect_ray(query)
    if hit.is_empty() or (hit["normal"] as Vector3).y < 0.7 or absf((hit["position"] as Vector3).y + PALM - y) > MAX_SHIFT:
        return null
    return hit["position"]


func _process_modification() -> void:
    var skeleton := get_skeleton()
    if skeleton == null or _ik == null:
        return
    var to_world := skeleton.global_transform
    var resting := _resting()
    var reach_w := _reach_weight()
    var weight := maxf(_want, reach_w)
    var moved := false
    for hand in _bones:
        var b: Array = _bones[hand]
        var shoulder := to_world * skeleton.get_bone_global_pose(b[0]).origin
        var elbow := to_world * skeleton.get_bone_global_pose(b[1]).origin
        var wrist := to_world * skeleton.get_bone_global_pose(b[2]).origin
        _animated[hand] = wrist
        # Resting on something: onto it. Not (the pose is ending): where it was put, fading out.
        var move: Vector3 = _raw.get(hand, Vector3.ZERO)
        if hand in resting:
            move = (_support[hand] as Vector3) + Vector3.UP * PALM - wrist if _support.has(hand) else Vector3.ZERO
            _raw[hand] = move
        move *= _want / weight if weight > 0.0 else 0.0
        if not _reach.is_empty() and _reach["hand"] == hand and is_instance_valid(_reach["node"]):
            var goal: Vector3 = (_reach["node"] as Node3D).global_transform * (_reach["local"] as Vector3)
            goal -= (goal - shoulder).normalized() * GRIP
            move = (goal - wrist) * (reach_w / weight)
        _shift[hand] = move.length() * weight
        moved = moved or move.length() > 0.001
        (_targets[hand] as Marker3D).global_position = wrist + move
        # The elbow keeps bending the way the animation bends it.
        var bend := elbow - (shoulder + wrist) * 0.5
        (_poles[hand] as Marker3D).global_position = elbow + (bend.normalized() if bend.length() > 1e-4 else Vector3.BACK) * 0.3
    _ik.influence = weight if moved else 0.0
    if _want <= 0.0:
        _raw.clear()
