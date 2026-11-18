class_name GeogenMotionRig
extends RefCounted
## Runs motions (`kind: motion`, motions.json) on one body: the runtime port of the reference rig,
## geogen/motion.py MotionRig (docs/dev/characters.md "Motions"). Pure data: built from the body's
## `extras.geogen.rig` (bones and their rests, params, channels, hand shapes, pose library), it
## steps at a fixed rate and gives every bone's local transform (`local()`), which
## GeogenMotionPlayer writes to the Skeleton3D. Conformance traces (tests/data/motion/) hold the
## two rigs to the same bone positions.
##
## Ported so far: beats (pose, bones, hips, ik onto body-frame points or points on things with
## hand shapes and pole, look, lean, plant: feet, over, ease, after, emit, then, loop, interrupt
## and exit), roles bound to things (`Thing`: a node with `extras.geogen.grips`, or a fixed
## transform) and their contacts, `<role>@start`, `self.<Bone>`, attach / release and `holding`,
## channel arbitration, pausing under an overlay and `resume`, a break into a reaction (what the
## hands hold goes to physics), inertialization, the reach bend, free arms hanging under a bent
## spine, root targets (moving the body: sit), a posture change playing the body's transition
## clip (sit_down: `add_transitions`) with its root path and drape correctives (`morphs`), and the
## trace. The rig works in the body's frame; `body_world` is where the body is, which a run that
## holds the root moves (GeogenMotionPlayer puts the body's node there). Things are read and moved
## in the world.

const REACH_SPEED := 0.9
const MIN_OVER := 0.35
const TRAVEL_EPS := 1e-4         # m: closer than this a beat's end or root is already there
const HALF_LIFE := {"gaze": 0.08, "left_hand": 0.12, "right_hand": 0.12}
const DEFAULT_HALF_LIFE := 0.18
const REACH_SLACK := 0.96
const BEND_BONES := ["Spine", "Chest"]
const MAX_BEND := 75.0
const HANG_SHARE := 0.7          # of the way a free arm swings back to hanging plumb (motion.HANG_SHARE)
const TIP_LENGTH := {"Index": 0.02, "Middle": 0.021, "Ring": 0.02, "Little": 0.016}   # generators/hand.py
const LOOK_IN := 0.3
const LOOK_OUT := 0.4            # s: a look that ends eases off in this long
const LOOK_FOLLOW := 0.1
const YAW_LIMIT := deg_to_rad(70.0)
const PITCH_LIMIT := deg_to_rad(35.0)
const NECK_SHARE := 0.4
const STRENGTHS := {"prefer": 1, "claim": 2, "require": 3}
const FINGERS := ["Thumb", "Index", "Middle", "Ring", "Little"]
const LOOK_BONES := ["Neck", "Head"]
const NOT_PORTED := ["when", "until", "do", "bend"]
const MORPH_HALF_LIFE := 0.12    # s: drape correctives settle onto a new owner's pose
const ROOT_LEAD := 0.7           # over planted feet the root arrives by this fraction of the beat
const SHARE_TURN := 2.0          # s a sharer of weight 1 keeps a shared channel before passing it on
## Grips that are not a hand holding the thing: `grip: any` on a role takes the first other one.
const NOT_HELD := ["sit", "lie", "stand", "use", "look", "press", "put", "rest"]
const HANDS := {"hand": "RightHand", "right": "RightHand", "left": "LeftHand",
    "right_hand": "RightHand", "left_hand": "LeftHand"}

## A thing took hold of / let go of by a hand (`to_physics`: let go of by a break, to fall).
signal attached(thing: Thing, end: String)
signal released(thing: Thing, to_physics: bool)
signal emitted(motion: String, what: String)
## Two-bone limbs by end bone: upper, lower, rest direction, rest pole (core/ik.py LIMBS).
const LIMBS := {
    "LeftFoot": ["LeftUpperLeg", "LeftLowerLeg", Vector3(0, -1, 0), Vector3(0, 0, 1)],
    "RightFoot": ["RightUpperLeg", "RightLowerLeg", Vector3(0, -1, 0), Vector3(0, 0, 1)],
    "LeftHand": ["LeftUpperArm", "LeftLowerArm", Vector3(1, 0, 0), Vector3(0, 0, -1)],
    "RightHand": ["RightUpperArm", "RightLowerArm", Vector3(-1, 0, 0), Vector3(0, 0, -1)],
}


## A body pose: per bone (in `names` order) its rotation relative to rest and its offset (m).
class Pose:
    var rot: Array = []
    var off: Array = []

    func _init(n := 0) -> void:
        rot.resize(n)
        off.resize(n)
        for i in n:
            rot[i] = Quaternion.IDENTITY
            off[i] = Vector3.ZERO

    func copy() -> Pose:
        var p := Pose.new()
        p.rot = rot.duplicate()
        p.off = off.duplicate()
        return p


## A thing motions act on: a node whose extras carry its `grips` (motion.grip_points: a mug's
## handle, a table's top), or a fixed world transform (a trace's things; a spot to put a thing).
## While a hand holds it, the rig moves it (`xform`); whoever owns the node follows that.
class Thing:
    var name := ""
    var node: Node3D = null
    var xform := Transform3D.IDENTITY    # where it is when it has no node, or while it is held
    var grips := {}                      # attachment -> {"grip": String, "local": Transform3D}
    var affordances := []                # its extras' affordances: [{type, position, yaw}] (a chair's sit)
    var held := false

    func world() -> Transform3D:
        return node.global_transform if node != null and not held and is_instance_valid(node) else xform

    ## The thing `node` is, with the grips its extras give ([tx, ty, tz, qx, qy, qz, qw] each).
    static func of(node_: Node3D, grips_: Dictionary, affordances_ := []) -> Thing:
        var t := Thing.new()
        t.name = String(node_.name)
        t.node = node_
        t.xform = node_.global_transform
        t.set_grips(grips_)
        t.affordances = affordances_
        return t

    ## A thing that stays at `at` (world).
    static func fixed(name_: String, at: Transform3D, grips_ := {}, affordances_ := []) -> Thing:
        var t := Thing.new()
        t.name = name_
        t.xform = at
        t.set_grips(grips_)
        t.affordances = affordances_
        return t

    func set_grips(grips_: Dictionary) -> void:
        grips = {}
        for k in grips_:
            var v: Array = grips_[k].get("at", [0, 0, 0, 0, 0, 0, 1])
            grips[k] = {"grip": str(grips_[k].get("grip", "")),
                "local": Transform3D(Basis(Quaternion(v[3], v[4], v[5], v[6])), Vector3(v[0], v[1], v[2]))}


## A point a beat aims at: `thing`'s world transform times `local` (followed as it moves), or
## `local` as a fixed world transform when there is no thing.
class Target:
    var thing: Thing = null
    var local := Transform3D.IDENTITY
    var name := ""
    var grip = null                      # how a body touches it (the attachment's grip)

    func world() -> Transform3D:
        return local if thing == null else thing.world() * local

    static func make(thing_: Thing, local_: Transform3D, name_: String, grip_ = null) -> Target:
        var t := Target.new()
        t.thing = thing_
        t.local = local_
        t.name = name_
        t.grip = grip_
        return t


## One motion running: its resolved spec, where it is, and the body as its beat found it.
class Run:
    var motion: Dictionary
    var name: String
    var params: Dictionary
    var order: int
    var beat := ""
    var t := 0.0
    var over := 0.0
    var dwell := 0.0
    var pending_exit := false
    var done := false
    var arrived := false
    var loops := {}
    var paused = null                # the beat an overlay took a channel in (its clock stopped)
    var resume_to = null             # asked to yield while paused: its exit, not its resume
    var start_pose: Pose
    var start_ends := {}
    var start_feet := {}
    var start_hands := {}            # hand -> its body-frame rotation as the beat found it
    var start_held := []             # [Thing, Vector3 from it to the hand holding it] as the beat found them
    var start_root := Transform3D.IDENTITY   # where the body was (world) as the beat began
    var start_posture := "stand"
    var start_morphs := {}
    var bindings := {}               # role -> Target
    var origins := {}                # role -> its thing's world transform when the run started

    func spec() -> Dictionary:
        var b = motion["beats"].get(beat)
        return b if b is Dictionary else {}

    func strength(channel: String) -> int:
        var holds: Dictionary = motion.get("holds", {})
        var best := 0
        for kind in STRENGTHS:
            if channel in holds.get(kind, []):
                best = maxi(best, STRENGTHS[kind])
        if holds.get("share", {}).has(channel):
            best = maxi(best, STRENGTHS["claim"])
        return best

    ## Its weight in time-slicing `channel` (holds.share), 0 when it doesn't share it.
    func share(channel: String) -> float:
        var shares = motion.get("holds", {}).get("share", {})
        return float(shares.get(channel, 0.0)) if shares is Dictionary and shares.get(channel) != null else 0.0


var names := PackedStringArray()
var index := {}                  # bone -> its index in names
var parent := PackedInt32Array()  # -1 for the root bone
var rest: Array = []             # Transform3D per bone, in its parent's frame
var rest_world: Array = []       # Transform3D per bone, in the body's frame
var params := {}
var channels := {}               # channel -> Array of bone names
var channel_of := {}             # bone name -> channel
var shared := {}                 # channel -> true for a shared one (the gaze): losing it pauses nothing
var hand_shapes := {}
var poses := {}                  # pose name -> Pose
var fingers := {"Left": [], "Right": []}

var dt := 1.0 / 30.0
var time := 0.0
var posture := "stand"
var out: Pose
var runs: Array = []
var owners := {}                 # channel -> Run (or null)
var trace: Array = []            # events, as geogen/motion.py writes them
var _order := 0
var _changed := {}               # channels whose owner changed this step
var _offsets := {}               # bone index -> [Quaternion, age, half-life]
var _prelook := {}               # bone index -> Quaternion (the head before the look)
var _look_key: Run = null
var _look_t := 0.0
var _look_point := Vector3.ZERO
var _look_weight := 0.0          # how far the current look has turned the head
var _look_fade = null            # [point, weight, t]: a look that ended, easing off
var body_world := Transform3D.IDENTITY   # where the body is in the world (its node's transform)
var root_moved := false          # the last step moved the body (its node goes there)
var floor_y := 0.0               # standing root targets stand on this floor (where the body stood)
var root_poses := {}             # posture -> Transform3D: where it puts the body against what it is at
var transitions := {}            # "from>to" -> a transition clip (add_transitions)
var pose_morphs := {}            # pose -> {"node:target": weight}: the drape correctives it was baked with
var morph_keys := {}             # every "node:target" the body has
var morphs := {}                 # "node:target" -> weight, as the body is now
var _morph_goal := {}
var _morph_offset = null         # [delta {key: w}, age]
var _root_offset = null          # [Transform3D, age, half-life]
var _share_t := {}               # channel -> how long its sharer has had this turn
var _share_looked := {}          # "channel/order" -> that run's time with the channel
var attached_ := {}              # role -> {thing, end, rel: Transform3D, level}: what the hands hold
## (name) -> Thing or null: a scene reference a motion or a binding names ("cup", "table").
var resolver := Callable()


## A rig for the body whose `extras.geogen.rig` is `data`, stepping `fps` times a second.
static func from_extras(data: Dictionary, fps := 30) -> GeogenMotionRig:
    var r := GeogenMotionRig.new()
    r.dt = 1.0 / fps
    for b in data["bones"]:
        var i := r.names.size()
        r.names.append(b["name"])
        r.index[b["name"]] = i
        r.parent.append(r.index.get(b["parent"], -1) if b["parent"] != "" else -1)
        var v: Array = b["rest"]
        var local := Transform3D(Basis(Quaternion(v[3], v[4], v[5], v[6])), Vector3(v[0], v[1], v[2]))
        r.rest.append(local)
        r.rest_world.append(local if r.parent[i] < 0 else r.rest_world[r.parent[i]] * local)
    for k in data.get("params", {}):
        r.params[k] = float(data["params"][k])
    for c in data.get("channels", {}):
        r.channels[c] = Array(data["channels"][c]["bones"])
        r.shared[c] = bool(data["channels"][c].get("shared", false))
        for bone in r.channels[c]:
            r.channel_of[bone] = c
    r.hand_shapes = data.get("hand_shapes", {})
    for pose_name in data.get("poses", {}):
        var spec: Dictionary = data["poses"][pose_name]
        var p := Pose.new(r.names.size())
        for bone in spec.get("rotations", {}):
            if r.index.has(bone):
                var q: Array = spec["rotations"][bone]
                p.rot[r.index[bone]] = Quaternion(q[0], q[1], q[2], q[3])
        for bone in spec.get("offsets", {}):
            if r.index.has(bone):
                var o: Array = spec["offsets"][bone]
                p.off[r.index[bone]] = Vector3(o[0], o[1], o[2])
        r.poses[pose_name] = p
    if not r.poses.has("stand"):
        r.poses["stand"] = Pose.new(r.names.size())
    for pose_name in data.get("root_poses", {}):
        var rp: Dictionary = data["root_poses"][pose_name]
        var o: Array = rp.get("offset", [0, 0, 0])
        var d: Array = rp.get("rotation", [0, 0, 0])
        r.root_poses[pose_name] = Transform3D(_euler(Vector3(d[0], d[1], d[2])), Vector3(o[0], o[1], o[2]))
    for side in ["Left", "Right"]:
        for bone in r.names:
            if bone.begins_with(side) and FINGERS.any(func(f): return bone.contains(f)):
                r.fingers[side].append(bone)
    for c in r.channels:
        r.owners[c] = null
    r.out = r.poses["stand"].copy()
    for b in LOOK_BONES:
        if r.index.has(b):
            r._prelook[r.index[b]] = r.out.rot[r.index[b]]
    return r


## The body's transition clips and pose correctives (motion.transition_data: a trace carries them;
## GeogenMotionPlayer reads the same from the body's animations): `transitions` [{from, to,
## duration, tracks: {bone: {times, rotations?, translations?}}, root: {times, positions,
## rotations}, morphs: {key: {times, values}}}], `pose_morphs` {pose: {key: weight}}.
func add_transitions(data: Dictionary) -> void:
    for tr in data.get("transitions", []):
        var clip := {"duration": float(tr["duration"]), "tracks": {}, "root": null, "morphs": {}}
        for bone in tr.get("tracks", {}):
            if not index.has(bone):
                continue
            var i: int = index[bone]
            var spec: Dictionary = tr["tracks"][bone]
            var track := {"times": PackedFloat64Array(spec["times"]), "rot": [], "pos": []}
            var rest_q: Quaternion = (rest[i] as Transform3D).basis.get_rotation_quaternion()
            for q in spec.get("rotations", []):
                track["rot"].append(rest_q.inverse() * Quaternion(q[0], q[1], q[2], q[3]))
            if parent[i] >= 0:
                for v in spec.get("translations", []):
                    track["pos"].append(Vector3(v[0], v[1], v[2]) - rest[i].origin)
            clip["tracks"][i] = track
        var root = tr.get("root")
        if root is Dictionary:
            var path := {"times": PackedFloat64Array(root["times"]), "pos": [], "rot": []}
            for v in root["positions"]:
                path["pos"].append(Vector3(v[0], v[1], v[2]))
            for q in root["rotations"]:
                path["rot"].append(Quaternion(q[0], q[1], q[2], q[3]))
            clip["root"] = path
        for key in tr.get("morphs", {}):
            var m: Dictionary = tr["morphs"][key]
            clip["morphs"][key] = [PackedFloat64Array(m["times"]), PackedFloat64Array(m["values"])]
            morph_keys[key] = true
        transitions["%s>%s" % [tr["from"], tr["to"]]] = clip
    for pose_name in data.get("pose_morphs", {}):
        pose_morphs[pose_name] = data["pose_morphs"][pose_name]
        for key in pose_morphs[pose_name]:
            morph_keys[key] = true
    morphs = _pose_morph_vector("stand")
    _morph_goal = morphs.duplicate()


## The drape correctives `pose` was baked with, for every morph target of the body.
func _pose_morph_vector(pose: String) -> Dictionary:
    var known: Dictionary = pose_morphs.get(pose, pose_morphs.get("stand", {}))
    if known.is_empty():
        known = pose_morphs.get("stand", {})
    var out_ := {}
    for k in morph_keys:
        out_[k] = float(known.get(k, 0.0))
    return out_


## The transition clip this beat's posture change follows (sit_down), or null.
func _transition(run: Run):
    var to = run.spec().get("pose")
    if to == null or to == "" or run.start_posture == str(to):
        return null
    return transitions.get("%s>%s" % [run.start_posture, to])


## A span of sorted key `times` at `t` (clamped): [i, j, f] (core/skin.py Track._span).
static func _span(times: PackedFloat64Array, t: float) -> Array:
    var n := times.size()
    if n == 1 or t <= times[0]:
        return [0, 0, 0.0]
    if t >= times[n - 1]:
        return [n - 1, n - 1, 0.0]
    var i := times.bsearch(t, false) - 1
    return [i, i + 1, (t - times[i]) / (times[i + 1] - times[i])]


## A transition clip's pose at `t` (motion.clip_pose); `has_off`: bones that have an offset.
func _clip_pose(clip: Dictionary, t: float) -> Array:
    var pose := Pose.new(names.size())
    var has_off := {}
    for i in clip["tracks"]:
        var track: Dictionary = clip["tracks"][i]
        var sp := _span(track["times"], t)
        if not (track["rot"] as Array).is_empty():
            pose.rot[i] = (track["rot"][sp[0]] as Quaternion).slerp(track["rot"][sp[1]], sp[2])
        if not (track["pos"] as Array).is_empty():
            var va: Vector3 = track["pos"][sp[0]]
            var v := va + ((track["pos"][sp[1]] as Vector3) - va) * float(sp[2])
            if maxf(absf(v.x), maxf(absf(v.y), absf(v.z))) > 1e-6:
                pose.off[i] = v
                has_off[i] = true
    return [pose, has_off]


## A transition clip's root path at `t`: the body's transform in the frame of where it ends up.
static func _root_at(path: Dictionary, t: float) -> Transform3D:
    var times: PackedFloat64Array = path["times"]
    t = clampf(t, times[0], times[times.size() - 1])
    var i := clampi(times.bsearch(t, false) - 1, 0, times.size() - 2)
    var f := (t - times[i]) / maxf(times[i + 1] - times[i], 1e-9)
    var q := (path["rot"][i] as Quaternion).slerp(path["rot"][i + 1], f)
    return Transform3D(Basis(q), (path["pos"][i] as Vector3) * (1.0 - f) + (path["pos"][i + 1] as Vector3) * f)


## The morph weights of a transition clip at `t`.
static func _clip_morphs(clip: Dictionary, t: float) -> Dictionary:
    var out_ := {}
    for k in clip["morphs"]:
        var m: Array = clip["morphs"][k]
        out_[k] = _interp(m[0], m[1], t)
    return out_


## numpy.interp: linear, clamped at the ends.
static func _interp(xs: PackedFloat64Array, ys: PackedFloat64Array, t: float) -> float:
    var n := xs.size()
    if t <= xs[0]:
        return ys[0]
    if t >= xs[n - 1]:
        return ys[n - 1]
    var i := xs.bsearch(t, false) - 1
    return ys[i] + (ys[i + 1] - ys[i]) * (t - xs[i]) / (xs[i + 1] - xs[i])


static func _yaw_of(m: Transform3D) -> float:
    var z := m.basis.z
    return atan2(z.x, z.z)


static func _placement(pos: Vector3, yaw: float) -> Transform3D:
    return Transform3D(Basis(Vector3.UP, yaw), pos)


static func _wrap(a: float) -> float:
    return fposmod(a + PI, TAU) - PI


## Where a beat's `root` puts the body (world): `at` a reference with `offset` and `facing`,
## standing on the floor the body stood on, or `at` + `posture` where that posture puts it.
func _root_target(run: Run) -> Transform3D:
    var spec: Dictionary = run.spec()["root"]
    var target = _ref(run, spec.get("at"))
    var m: Transform3D
    if target is Target:
        m = target.world()
    elif target is Vector3:
        m = _placement(body_world * target, _yaw_of(body_world))
    else:
        push_error("motion '%s' beat '%s': root at %s is not bound" % [run.name, run.beat, spec.get("at")])
        return body_world
    m = _placement(m.origin, _yaw_of(m))
    var posture_ = spec.get("posture")
    if posture_ != null:
        return m * root_poses.get(str(posture_), Transform3D.IDENTITY)
    var o: Array = spec.get("offset", [0, 0, 0])
    m = m * Transform3D(Basis.IDENTITY, Vector3(float(o[0]), float(o[1]), float(o[2])))
    var facing = spec.get("facing", "away")
    var turn := 0.0
    if facing is String:
        turn = 0.0 if facing == "away" else PI if facing == "toward" else deg_to_rad(float(facing))
    else:
        turn = deg_to_rad(float(facing))
    return _placement(Vector3(m.origin.x, floor_y, m.origin.z), _yaw_of(m) + turn)


# --- requests ---

## Start `motion` (a spec from motions.json) with `values` for its params (`reps`, ...) and its
## roles (a Thing, a Target, or a name `resolver` finds: "cup", "table.top").
## Returns the run, or null with the reason pushed as an error.
func start(motion: Dictionary, values := {}) -> Run:
    var name := str(motion.get("name", "motion"))
    var roles: Dictionary = motion.get("roles", {})
    var bound := {}
    for role in roles:
        var entry = roles[role] if roles[role] is Dictionary else {}
        var value = values.get(role)
        if value == null:
            if not entry.get("optional", false):
                push_error("motion '%s': role '%s' is not bound" % [name, role])
                return null
            continue
        var target := _bind(value, entry.get("grip"))
        if target == null:
            push_error("motion '%s': role '%s': can't find %s" % [name, role, value])
            return null
        bound[role] = target
    var given := {}
    for k in motion.get("params", {}):
        var d = motion["params"][k]
        given[k] = d.get("default") if d is Dictionary else d
    for k in values:
        if roles.has(k):
            continue
        if not given.has(k) and k != "duration":
            push_error("motion '%s': no param '%s'" % [name, k])
            return null
        given[k] = values[k]
    var holds: Dictionary = motion.get("holds", {})
    for kind in ["require", "claim", "prefer"]:
        for c in holds.get(kind, []):
            if not channels.has(c):
                push_error("motion '%s': this body has no channel '%s'" % [name, c])
                return null
    var env := params.duplicate()
    for k in given:
        if given[k] is float or given[k] is int:
            env[k] = float(given[k])
    for k in given:                      # a param of the body's size ("{0.2 * height}")
        if given[k] is String and given[k].begins_with("{"):
            given[k] = resolve(given[k], env)
            env[k] = given[k]
    var beats = resolve(motion["beats"], env)
    for beat in beats:
        for key in NOT_PORTED:
            if (beats[beat] as Dictionary).has(key):
                push_error("motion '%s' beat '%s': '%s' is not ported to the runtime rig yet" % [name, beat, key])
                return null
    var spec := motion.duplicate()
    spec["beats"] = beats
    if posture == "stand" and owners.get("root") == null:
        floor_y = body_world.origin.y
    _order += 1
    var run := Run.new()
    run.motion = spec
    run.name = name
    run.params = given
    run.order = _order
    run.bindings = bound
    for role in bound:
        if (bound[role] as Target).thing != null:
            run.origins[role] = (bound[role] as Target).thing.world()
    runs.append(run)
    _event("start", run, {})
    _enter(run, str(spec.get("initial", beats.keys()[0])))
    return run


## Ask the running motion `name` (the latest started) to stop: `yield` follows its beat's
## policy, `drop` stops it where it is. Returns what happened.
func interrupt(name: String, how := "yield") -> String:
    var run: Run = null
    for r in runs:
        if not r.done and r.name == name:
            run = r
    if run == null:
        return "not running"
    var policy: String = "drop" if how == "drop" else str(run.spec().get("interrupt", "yield"))
    _event("interrupt", run, {"policy": policy})
    if policy == "hold":
        return "held"
    if policy == "finish":
        run.pending_exit = true
        return "finishing"
    if policy == "drop":
        _end(run)
        return "dropped"
    _mark(_owned(run))
    var exit_beat = run.spec().get("exit")
    if exit_beat == null or exit_beat == "done":
        _end(run)
        return "ended"
    if run.paused != null:
        run.paused = exit_beat           # out through its exit once the overlay gives the channel back
        run.resume_to = exit_beat
        return "exiting"
    _enter(run, exit_beat)
    return "exiting"


## Everything stops at once (a shove, a fall): every motion drops with no exits and the body
## inertializes into `reaction` (a motion: a stagger) when one is given.
func break_(reaction = null, values := {}) -> Run:
    _event("break", null, {"reaction": reaction.get("name") if reaction is Dictionary else null})
    for run in runs:
        if not run.done:
            _end(run)
    for role in attached_.keys():
        var held: Dictionary = attached_[role]
        attached_.erase(role)
        (held["thing"] as Thing).held = false
        _event("release", null, {"role": role, "to": "physics"})
        released.emit(held["thing"], true)
    return start(reaction, values) if reaction is Dictionary else null


## The body standing at `at` (world) at once, nothing easing: a gallery or a script putting it back.
func reset_standing(at: Transform3D) -> void:
    posture = "stand"
    out = (poses["stand"] as Pose).copy()
    for i in _prelook:
        _prelook[i] = out.rot[i]
    _offsets = {}
    _changed = {}
    _root_offset = null
    _morph_offset = null
    _look_key = null
    _look_fade = null
    body_world = at
    floor_y = at.origin.y
    morphs = _pose_morph_vector("stand")
    _morph_goal = morphs.duplicate()


func running() -> Array:
    return runs.filter(func(r): return not r.done).map(func(r): return r.name)


# --- the step ---

func step() -> void:
    var before := body_world
    time += dt
    for run in runs.duplicate():
        if not run.done:
            _advance(run)
    runs = runs.filter(func(r): return not r.done)
    _arbitrate()
    out = _compose()
    root_moved = not body_world.is_equal_approx(before)
    _carry()


## What the hands hold goes with them: each held thing's world transform from the hand's.
func _carry() -> void:
    if attached_.is_empty():
        return
    var fk := _fk(out)
    for role in attached_:
        var h: Dictionary = attached_[role]
        var hand: Transform3D = fk[index[h["end"]]]
        var rel: Transform3D = h["rel"]
        var at := Transform3D(rel.basis, hand.origin + rel.origin) if h["level"] else hand * rel
        (h["thing"] as Thing).xform = body_world * at


## Every bone's local transform (in its parent's frame) as the body is now.
func local(i: int) -> Transform3D:
    return _posed_local(i, out)


## Every bone's transform in the body's frame as the body is now.
func world() -> Array:
    return _fk(out)


func _advance(run: Run) -> void:
    if run.paused != null:
        return
    run.t += dt
    _arrive(run)
    var guard := 0
    while not run.done and run.t >= run.over + run.dwell and guard < 16:
        guard += 1
        var spill := run.t - (run.over + run.dwell)
        var nxt = run.spec().get("exit") if run.pending_exit else _next(run)
        run.pending_exit = false
        if nxt == null or nxt == "done":
            _end(run, true)
            return
        _enter(run, nxt)
        run.t = spill
        _arrive(run)


func _next(run: Run) -> String:
    var loop = run.spec().get("loop")
    if loop is Dictionary:
        var done: int = int(run.loops.get(run.beat, 0)) + 1
        if done < int(loop.get("times", 1)):
            run.loops[run.beat] = done
            _event("loop", run, {"to": loop["to"], "round": done + 1})
            return str(loop["to"])
        run.loops[run.beat] = 0
    return str(run.spec().get("then", "done"))


func _arrive(run: Run) -> void:
    if run.done or run.t < run.over or run.arrived:
        return
    run.arrived = true
    var spec := run.spec()
    var attach: Dictionary = spec.get("attach", {}) if spec.get("attach") is Dictionary else {}
    for role in attach:
        var target: Target = run.bindings.get(role)
        if target == null or target.thing == null:
            push_error("motion '%s': attach: '%s' is not a bound thing" % [run.name, role])
            continue
        var how = attach[role] if attach[role] is Dictionary else {"hand": attach[role]}
        var end: String = HANDS.get(str(how.get("hand", "hand")), str(how.get("hand", "")))
        if not index.has(end):
            push_error("motion '%s': attach: '%s' to '%s', which is not a hand" % [run.name, role, how.get("hand")])
            continue
        var hand: Transform3D = _fk(out)[index[end]]
        var thing := _body(target.thing.world())
        var level := bool(how.get("level", false))
        # Level: kept as it is against the body's heading, carried along by the hand's position.
        var rel := Transform3D(thing.basis, thing.origin - hand.origin) if level else hand.affine_inverse() * thing
        target.thing.xform = target.thing.world()
        target.thing.held = true
        attached_[role] = {"thing": target.thing, "end": end, "rel": rel, "level": level}
        _event("attach", run, {"role": role, "hand": end})
        attached.emit(target.thing, end)
    var release = spec.get("release")
    for role in ([release] if release is String else (release if release is Array else [])):
        if not attached_.has(role):
            continue
        var held: Dictionary = attached_[role]
        attached_.erase(role)
        (held["thing"] as Thing).held = false
        _event("release", run, {"role": role})
        released.emit(held["thing"], false)
        # Whatever else holds it (a carry) has nothing left to do.
        for other in runs:
            if not other.done and other != run and other.bindings.values().any(
                    func(b): return (b as Target).thing == held["thing"]):
                _end(other)
    if spec.get("emit"):
        _event("emit", run, {"emit": spec["emit"]})
        emitted.emit(run.name, str(spec["emit"]))
    if spec.get("pose") and "hips" in _owned(run) and posture != spec["pose"]:
        posture = spec["pose"]
        var free := []
        for c in owners:
            if owners[c] == null:
                free.append(c)
        _mark(free)
        _event("posture", run, {"posture": posture})


func _enter(run: Run, beat: String) -> void:
    run.beat = beat
    run.t = 0.0
    run.arrived = false
    var start := out.copy()
    for i in _prelook:
        start.rot[i] = _prelook[i]
    run.start_pose = start
    var fks := _fk(start)
    run.start_hands = {}
    for e in ["LeftHand", "RightHand"]:
        if index.has(e):
            run.start_hands[e] = Quaternion((fks[index[e]] as Transform3D).basis.orthonormalized())
    var fk := _fk(out)
    for e in ["LeftHand", "RightHand", "LeftFoot", "RightFoot"]:
        if index.has(e):
            run.start_ends[e] = fk[index[e]].origin
    run.start_feet = {}
    for f in ["LeftFoot", "RightFoot"]:
        if index.has(f):
            run.start_feet[f] = fk[index[f]].origin
    run.start_root = body_world
    run.start_posture = posture
    run.start_morphs = morphs.duplicate()
    run.start_held = []
    for role in attached_:
        var h: Dictionary = attached_[role]
        run.start_held.append([h["thing"], run.start_ends[h["end"]] - _body((h["thing"] as Thing).world()).origin])
    var entries := _ik_entries(run)
    for e in entries:
        if entries[e].get("holding") != null:
            _held_gap(run, e, str(entries[e]["holding"]))    # (an error when nothing holds it)
    run.over = _over(run)
    var after = run.spec().get("after", 0.0)
    if after is String and after == "duration":
        run.dwell = float(run.params.get("duration", 3.0))
    else:
        run.dwell = float(after if after != null else 0.0)
    _event("beat", run, {"beat": beat, "over": snappedf(run.over, 0.001), "dwell": snappedf(run.dwell, 0.001)})


func _over(run: Run) -> float:
    var spec := run.spec()
    if spec.has("over"):
        return maxf(float(spec["over"]), dt)
    var travel := 0.0
    var entries := _ik_entries(run)
    for e in entries:
        var target = _ik_target(run, e, entries[e])
        if target != null and run.start_ends.has(e):
            travel = maxf(travel, (target - run.start_ends[e]).length())
    if spec.has("root"):
        travel = maxf(travel, (_root_target(run).origin - body_world.origin).length())
    if travel > TRAVEL_EPS:
        return maxf(MIN_OVER, travel / REACH_SPEED + 0.15)
    return MIN_OVER if spec.get("pose") else 0.0


## Stop `run`. One whose last beat arrived leaves the body where that beat put it.
func _end(run: Run, arrived := false) -> void:
    if not run.done:
        if arrived and run.spec().has("root") and owners.get("root") == run:
            body_world = _root_target(run)
        _mark(_owned(run))
        run.done = true
        _event("end", run, {})


func _arbitrate() -> void:
    for c in channels:
        var best: Run = null
        for run in runs:
            var s: int = run.strength(c)
            if s > 0 and (best == null or _outranks(run, best, c)):
                best = run
        best = _share_turn(c, best)
        var was: Run = owners.get(c)
        if best != was:
            _share_t[c] = 0.0
            if c == "gaze" and _look_key != null:
                _fade_look()             # the head eases off the old look rather than springing
                _changed.erase(c)        # (an ended owner asked for it to be inertialized)
            else:
                _changed[c] = true
            _event("owner", best, {"channel": c, "was": was.name if was != null else null})
            if was != null and not was.done and was.strength(c) == STRENGTHS["require"]:
                interrupt(was.name, "yield")
        owners[c] = best
        if best != null and best.share(c) > 0.0:
            _share_t[c] = float(_share_t.get(c, 0.0)) + dt
            var key := "%s/%d" % [c, best.order]
            _share_looked[key] = float(_share_looked.get(key, 0.0)) + dt
    _pause_and_resume()


## Runs that share a channel (holds.share: {gaze: weight}) and rank equal take turns at it: the
## one that has it keeps it for SHARE_TURN x its weight, then it passes to the one that has had it
## least for its weight. A claim at the same strength outranks every share (motion._share_turn).
func _share_turn(c: String, best: Run) -> Run:
    if best == null or best.share(c) <= 0.0:
        return best
    var sharers := []
    for r in runs:
        if not r.done and r.share(c) > 0.0 and r.strength(c) == best.strength(c) \
                and int(r.motion.get("priority", 0)) == int(best.motion.get("priority", 0)):
            sharers.append(r)
    if sharers.size() < 2:
        return best
    var was: Run = owners.get(c)
    if was in sharers and float(_share_t.get(c, 0.0)) < SHARE_TURN * was.share(c) - 1e-9:
        return was
    var pick: Run = null
    var pick_key := []
    for r in sharers:
        if r == was:
            continue
        var key := [float(_share_looked.get("%s/%d" % [c, r.order], 0.0)) / r.share(c), -r.order]
        if pick == null or key[0] < pick_key[0] or (key[0] == pick_key[0] and key[1] < pick_key[1]):
            pick = r
            pick_key = key
    return pick


## A run that loses a channel it claims (not a shared one) to another waits, its beat's clock
## stopped; with them all back it picks up from the beat's `resume` (else that beat again).
func _pause_and_resume() -> void:
    for run in runs:
        if run.done:
            continue
        var lost := []
        for c in owners:
            var o: Run = owners[c]
            if o != null and o != run and run.strength(c) >= STRENGTHS["claim"] and not shared.get(c, false):
                lost.append(c)
        if not lost.is_empty() and run.paused == null:
            run.paused = run.beat
            lost.sort()
            _event("pause", run, {"channels": lost})
        elif lost.is_empty() and run.paused != null:
            var beat = run.resume_to if run.resume_to != null else run.spec().get("resume", run.paused)
            run.paused = null
            run.resume_to = null
            _event("resume", run, {"to": beat})
            _enter(run, str(beat))


## Strength, then priority, then a claim over a share, then the later started.
func _outranks(a: Run, b: Run, c: String) -> bool:
    var ka := [a.strength(c), int(a.motion.get("priority", 0)), 0 if a.share(c) > 0.0 else 1, a.order]
    var kb := [b.strength(c), int(b.motion.get("priority", 0)), 0 if b.share(c) > 0.0 else 1, b.order]
    for i in 4:
        if ka[i] != kb[i]:
            return ka[i] > kb[i]
    return false


func _owned(run: Run) -> Array:
    var out_ := []
    for c in owners:
        if owners[c] == run:
            out_.append(c)
    return out_


func _mark(cs: Array) -> void:
    for c in cs:
        _changed[c] = true


# --- composing the pose ---

func _compose() -> Pose:
    var pose: Pose = (poses.get(posture, poses["stand"]) as Pose).copy()
    var root := body_world
    # Poses first, every owner over its own channels, eased from where its beat began.
    for run in runs:
        var held := _owned(run)
        if held.is_empty():
            continue
        var spec: Dictionary = run.spec()
        var u := ease_(str(spec.get("ease", "smooth")), run.t / run.over if run.over > 0.0 else 1.0)
        var target := _beat_pose(run)
        # A posture change plays the body's transition clip (sit_down) on the channels it holds,
        # other than a hand it reaches with and the gaze, taken whole (the build cleared its arms
        # of the body and keyed its drape against it); what the body started with fades out.
        var clip = _transition(run)
        var carried := {}
        var at := 0.0
        if clip != null:
            var reaching := {}
            var entries := _ik_entries(run)
            for e in entries:
                if _ik_target(run, e, entries[e]) != null:
                    reaching[channel_of.get(e, "")] = true
            at = minf(run.t / run.over if run.over > 0.0 else 1.0, 1.0) * float(clip["duration"])
            var now_ := _clip_pose(clip, at)
            var first_ := _clip_pose(clip, 0.0)
            var now: Pose = now_[0]
            var first: Pose = first_[0]
            for c in held:
                if reaching.has(c) or c == "gaze" or c == "root":
                    continue
                for bone in channels[c]:
                    var i: int = index[bone]
                    var offset: Quaternion = run.start_pose.rot[i] * first.rot[i].inverse()
                    pose.rot[i] = Quaternion.IDENTITY.slerp(offset, 1.0 - u) * now.rot[i]
                    var oa: Vector3 = run.start_pose.off[i]
                    if oa != Vector3.ZERO or now_[1].has(i):
                        pose.off[i] = now.off[i] + (oa - first.off[i]) * (1.0 - u)
                    carried[i] = true
        for i in names.size():
            if not (channel_of.get(names[i], "") in held) or carried.has(i):
                continue
            pose.rot[i] = run.start_pose.rot[i].slerp(target.rot[i], u)
            pose.off[i] = run.start_pose.off[i] + (target.off[i] - run.start_pose.off[i]) * u
        if "root" in held and spec.has("root"):
            if clip != null and clip["root"] != null:
                # The clip's own root path, ending where the beat puts the body.
                var path: Dictionary = clip["root"]
                var goal := _root_target(run) * _root_at(path, float(clip["duration"])).affine_inverse()
                var now_root := goal * _root_at(path, at)
                var first_root := goal * _root_at(path, 0.0)
                var start: Transform3D = run.start_root
                var turn := Quaternion((start.basis * first_root.basis.transposed()).orthonormalized())
                root = Transform3D(Basis(Quaternion.IDENTITY.slerp(turn, 1.0 - u)) * now_root.basis,
                    now_root.origin + (start.origin - first_root.origin) * (1.0 - u))
            else:
                var goal := _root_target(run)
                var start: Transform3D = run.start_root
                var ur := u
                if spec.get("plant") == "feet":
                    # Over planted feet the body travels before it drops or rises: the root leads.
                    ur = ease_("smooth", (run.t / run.over if run.over > 0.0 else 1.0) / ROOT_LEAD)
                var yaw := _yaw_of(start) + _wrap(_yaw_of(goal) - _yaw_of(start)) * ur
                root = _placement(start.origin + (goal.origin - start.origin) * ur, yaw)
    # Drape correctives follow the body pose the hips owner is laying (a skirt seated or standing).
    var mor := _pose_morph_vector(posture)
    var hips: Run = owners.get("hips")
    if hips != null and not morph_keys.is_empty():
        var hp = hips.spec().get("pose")
        var goal := _pose_morph_vector(str(hp) if hp else posture)
        var uh := ease_(str(hips.spec().get("ease", "smooth")), hips.t / hips.over if hips.over > 0.0 else 1.0)
        var clip = _transition(hips)
        mor = {}
        if clip != null and not (clip["morphs"] as Dictionary).is_empty():
            var at := minf(hips.t / hips.over if hips.over > 0.0 else 1.0, 1.0) * float(clip["duration"])
            var now := _clip_morphs(clip, at)
            var first := _clip_morphs(clip, 0.0)
            for k in goal:
                mor[k] = float(now.get(k, goal[k])) + (float(hips.start_morphs.get(k, 0.0)) - float(first.get(k, 0.0))) * (1.0 - uh)
        else:
            for k in goal:
                var a := float(hips.start_morphs.get(k, 0.0))
                mor[k] = a + (float(goal[k]) - a) * uh
    _morph_goal = mor
    root = _inertialize(pose, root)
    body_world = root
    # Then the solve, outside in: lean, planted feet, hands, gaze.
    for run in runs:
        var held := _owned(run)
        var spec: Dictionary = run.spec()
        var u_raw: float = run.t / run.over if run.over > 0.0 else 1.0
        var u := ease_(str(spec.get("ease", "smooth")), u_raw)
        var lean = spec.get("lean")
        if lean is Dictionary and "spine" in held and index.has("Spine") and _transition(run) == null:   # (a clip leans itself)
            var by := float(lean.get("by", 0.0)) * _bump(u_raw, float(lean.get("at", 0.5)))
            var i: int = index["Spine"]
            pose.rot[i] = pose.rot[i] * _body_quat(i, _euler(Vector3(by, 0, 0)))
        if spec.get("plant") == "feet" and "legs" in held:
            for foot in run.start_feet:
                _solve_limb(pose, foot, _since(run, run.start_feet[foot]), null, Basis.IDENTITY)   # flat on the floor
        var entries := _ik_entries(run)
        var laid := pose.copy()          # the spine as laid, before any reach bends it
        for e in entries:
            if not (channel_of.get(e, "") in held):
                continue
            var goal = _ik_target(run, e, entries[e])
            if goal == null:
                continue
            var entry: Dictionary = _turn_entry(run, entries[e])
            if entry.get("tip"):
                # The fingertip goes on the target, the hand turned as asked (motion.py).
                var full = _hand_turn(e, entry)
                if full == null:
                    push_error("motion '%s': %s tip: needs fingers / palm to turn the hand" % [run.name, e])
                else:
                    goal -= (full as Basis) * _tip(e, str(entry["tip"]), pose)
            var start: Vector3 = _since(run, run.start_ends[e])
            var point: Vector3 = start + (goal - start) * u
            if entry.get("arc"):
                # Up and over on the way: asked for 6.75 (1 - u)^2 arc, the arm as blended in by
                # `u` rises by about 6.75 u (1 - u)^2 arc, a third of the way along (motion.py).
                point += Vector3(0, float(entry["arc"]) * minf(6.75 * (1.0 - u) * (1.0 - u), 2.0), 0)
            if "spine" in held and e in ["LeftHand", "RightHand"]:
                _bend_to_reach(pose, laid, e, point, run, u, goal if entry.get("arc") else null)
            # A hand turned by `fingers` / `palm` turns in the body's frame from how the beat found
            # it (motion.py: the same arc every frame, whatever the arm does).
            var turn = _hand_turn(e, entry)
            if turn != null:
                var rest_b: Basis = rest_world[index[e]].basis
                var goal_q := Quaternion(((turn as Basis) * rest_b).orthonormalized())
                turn = Basis((run.start_hands[e] as Quaternion).slerp(goal_q, u)) * rest_b.inverse()
            var solved := pose.copy()
            _solve_limb(solved, e, point, entry.get("pole"), turn)
            var limb: Array = LIMBS[e]
            for bone in [limb[0], limb[1], e]:
                var i: int = index[bone]
                pose.rot[i] = run.start_pose.rot[i].slerp(solved.rot[i], u)
            if turn != null:                # under the arm as blended, not as solved
                var j: int = index[e]
                pose.rot[j] = _pose_quat(j, _world_of(index[limb[1]], pose), (turn as Basis) * rest_world[j].basis)
    _hang_free_arms(pose)
    for b in LOOK_BONES:
        if index.has(b):
            _prelook[index[b]] = pose.rot[index[b]]
    var gaze: Run = owners.get("gaze")
    var look = gaze.spec().get("look") if gaze != null else null
    var point = null
    if look != null:
        point = _point(gaze, look.get("at") if look is Dictionary else look)
        if point != null:
            point = body_world * (point as Vector3)      # followed in the world: the body may move
    # A look that ended eases off over LOOK_OUT, under whatever look comes next.
    if _look_key != null and (point == null or _look_key != gaze):
        _fade_look()
    if _look_fade != null:
        _look_fade[2] += dt
        if _look_fade[2] >= LOOK_OUT:
            _look_fade = null
        else:
            _turn_head(pose, _look_fade[0], _look_fade[1] * (1.0 - ease_("smooth", _look_fade[2] / LOOK_OUT)))
    if point == null:
        return pose
    if _look_key == null:
        _look_key = gaze
        _look_t = 0.0
        _look_point = point
    else:
        _look_t += dt
        var k := 1.0 - pow(0.5, dt / LOOK_FOLLOW)
        _look_point = _look_point + (point - _look_point) * k
    var weight := float(look.get("weight", 1.0)) if look is Dictionary else 1.0
    _look_weight = weight * ease_("smooth", _look_t / LOOK_IN)
    _turn_head(pose, _look_point, _look_weight)
    return pose


## An arm nothing holds hangs as from an upright spine when the spine bends or leans: the upper
## arm swings so the shoulder-to-hand line is what it is with the posture's spine.
func _hang_free_arms(pose: Pose) -> void:
    if posture != "stand":
        return
    var base: Pose = poses.get(posture, poses["stand"])
    var spine: Array = channels.get("spine", [])
    var bent := false
    for b in spine:
        var i: int = index[b]
        if absf(pose.rot[i].dot(base.rot[i])) < cos(5e-5):
            bent = true
    if not bent:
        return
    var free := []
    for side in ["Left", "Right"]:
        if owners.get(side.to_lower() + "_hand") == null:
            free.append(side)
    if free.is_empty():
        return
    var upright := pose.copy()
    for b in spine:
        upright.rot[index[b]] = base.rot[index[b]]
    var was := _fk(upright)
    var now := _fk(pose)
    for side in free:
        var upper: int = index[side + "UpperArm"]
        var end: int = index[side + "Hand"]
        var a: Vector3 = (now[end].origin - now[upper].origin).normalized()
        var b: Vector3 = (was[end].origin - was[upper].origin).normalized()
        var swing := Basis(Quaternion.IDENTITY.slerp(Quaternion(a, b), HANG_SHARE))
        pose.rot[upper] = _pose_quat(upper, _world_of(parent[upper], pose), swing * now[upper].basis)


## The current look ends: its turn eases off from where it is (an inertialized gaze snaps back).
func _fade_look() -> void:
    if _look_key != null:
        _look_fade = [_look_point, _look_weight, 0.0]
    _look_key = null


func _beat_pose(run: Run) -> Pose:
    var spec := run.spec()
    var base: Pose = poses.get(str(spec.get("pose", posture)), poses.get(posture, poses["stand"]))
    if spec.get("pose") and not poses.has(spec["pose"]):
        push_error("motion '%s': this body has no pose '%s'" % [run.name, spec["pose"]])
    var pose := base.copy()
    if spec.get("hips") != null and index.has("Hips"):
        var h: Array = spec["hips"]
        pose.off[index["Hips"]] += Vector3(h[0], h[1], h[2])
    var bones: Dictionary = spec.get("bones", {})
    for bone in bones:
        if index.has(bone):
            var d: Array = bones[bone]
            var i: int = index[bone]
            pose.rot[i] = pose.rot[i] * _body_quat(i, _euler(Vector3(d[0], d[1], d[2])))
    var entries := _ik_entries(run)
    for e in entries:
        var shape = _hand_shape(run, e, entries[e])
        if shape == null:
            continue
        var side: String = e.substr(0, e.length() - 4)
        for bone in fingers[side]:
            # A shape poses every finger (both hands' bones are in it): one it doesn't name is straight.
            var d: Array = shape.get(bone, [0.0, 0.0, 0.0])
            pose.rot[index[bone]] = _body_quat(index[bone], _euler(Vector3(d[0], d[1], d[2])))
    return pose


## The hand shape a hand's `ik` entry takes (`grip:`, a shape or a role whose contact's grip names
## one; else its target's grip when that is a shape), or null.
func _hand_shape(run: Run, e: String, entry: Dictionary):
    if not e in ["LeftHand", "RightHand"]:
        return null
    var grip = entry.get("grip")
    if grip == null:
        var target = _ref(run, entry.get("target"))
        grip = target.grip if target is Target else null
        return hand_shapes.get(grip) if grip != null else null
    if hand_shapes.has(grip):
        return hand_shapes[grip]
    if run.bindings.has(grip):
        return hand_shapes.get((run.bindings[grip] as Target).grip)
    push_error("motion '%s' beat '%s': %s grip '%s' is neither a hand shape nor one of its roles" % [run.name, run.beat, e, grip])
    return null


func _ik_entries(run: Run) -> Dictionary:
    var spec: Dictionary = run.spec().get("ik", {})
    var entries := {}
    for k in spec:
        if k != "mirror":
            entries[k] = spec[k].duplicate()
    if spec.get("mirror", false):
        for k in entries.keys():
            var twin := _mirror_name(k)
            if twin != "" and not entries.has(twin):
                var e: Dictionary = entries[k].duplicate()
                var t = e.get("target")
                if t is String and t.ends_with(".left"):
                    e["target"] = t.substr(0, t.length() - 5) + ".right"
                elif t is Array:
                    e["target"] = [-float(t[0]), t[1], t[2]]
                for key in ["pole", "fingers", "palm"]:
                    if e.has(key):
                        var p: Array = e[key]
                        e[key] = [-float(p[0]), p[1], p[2]]
                entries[twin] = e
    return entries


## The body-frame turn from a hand's T-pose orientation to the one its `ik` entry asks for
## (`fingers`, `palm`: motion._hand_turn), or null.
static func _hand_turn(e: String, entry: Dictionary):
    if not (e in ["LeftHand", "RightHand"]) or not (entry.has("fingers") or entry.has("palm")):
        return null
    var side := 1.0 if e == "LeftHand" else -1.0
    var rest_f := Vector3(side, 0, 0)
    var rest_p := Vector3(0, -1, 0)
    var f := rest_f
    if entry.has("fingers"):
        var a: Array = entry["fingers"]
        f = Vector3(a[0], a[1], a[2]).normalized()
    var p := rest_p
    if entry.has("palm"):
        var a: Array = entry["palm"]
        p = Vector3(a[0], a[1], a[2])
    p = (p - p.dot(f) * f).normalized()
    var target := Basis(f, p, f.cross(p))
    var rest := Basis(rest_f, rest_p, rest_f.cross(rest_p))
    return target * rest.transposed()


## `entry` with its `fingers` / `palm` in the body's frame: `frame: target` gives them in the
## target's frame (motion._turn_entry).
func _turn_entry(run: Run, entry: Dictionary) -> Dictionary:
    if str(entry.get("frame", "body")) == "body" or not (entry.has("fingers") or entry.has("palm")):
        return entry
    var target = _ref(run, entry.get("target"))
    if not target is Target:
        return entry
    var turn: Basis = _body(target.world()).basis
    var out_ := entry.duplicate()
    for k in ["fingers", "palm"]:
        if entry.has(k):
            var a: Array = entry[k]
            var v: Vector3 = turn * Vector3(float(a[0]), float(a[1]), float(a[2]))
            out_[k] = [v.x, v.y, v.z]
    return out_


## From the wrist to `finger`'s tip (index, ...) in the body's frame, the hand at its rest
## orientation with the fingers as `pose` shapes them (motion._tip).
func _tip(e: String, finger: String, pose: Pose) -> Vector3:
    var side := e.substr(0, e.length() - 4)
    var n := finger.capitalize()
    if not TIP_LENGTH.has(n):
        push_error("%s tip '%s': not a finger" % [e, finger])
        return Vector3.ZERO
    var shaped := Pose.new(names.size())
    for i in names.size():
        if str(names[i]).begins_with(side + n):
            shaped.rot[i] = pose.rot[i]
    var w := _fk(shaped)
    var distal: Vector3 = w[index[side + n + "Distal"]].origin
    var along: Vector3 = (distal - w[index[side + n + "Intermediate"]].origin).normalized()
    var scale := float(params.get("height", 1.7)) / 1.7 * float(params.get("hand_size", 1.0))
    return distal + along * TIP_LENGTH[n] * scale - w[index[e]].origin


static func _mirror_name(n: String) -> String:
    if n.begins_with("Left"):
        return "Right" + n.substr(4)
    if n.begins_with("Right"):
        return "Left" + n.substr(5)
    return ""


## Where an `ik` entry puts its end (body frame), or null: its target with `offset` in the target's
## frame; `holding: <role>` puts that thing there (the hand by the hand-to-thing gap as the beat
## found it).
func _ik_target(run: Run, e: String, entry: Dictionary):
    var target = _ref(run, entry.get("target"))
    if target == null:
        return null
    var point := Vector3.ZERO
    if target is Target:
        var o: Array = entry.get("offset", [0, 0, 0])
        point = _body(target.world()) * Vector3(float(o[0]), float(o[1]), float(o[2]))
    else:
        point = target
    if entry.get("holding") != null:
        point += _held_gap(run, e, str(entry["holding"]))
    return point


## From the thing bound to `role` to the hand holding it, as the beat found them.
func _held_gap(run: Run, e: String, role: String) -> Vector3:
    var bound: Target = run.bindings.get(role)
    for pair in run.start_held:
        if bound != null and pair[0] == bound.thing:
            return body_world.basis.inverse() * (run.start_root.basis * (pair[1] as Vector3))
    push_error("motion '%s' beat '%s': %s holding '%s', which no hand holds" % [run.name, run.beat, e, role])
    return Vector3.ZERO


## A beat's point (body frame) for a look: [x, y, z] in the body's frame, or a reference's origin.
func _point(run: Run, ref):
    var target = _ref(run, ref)
    if target is Target:
        return _body(target.world()).origin
    return target


## A beat's reference: [x, y, z] in the body frame (a Vector3), a role, `role.attachment`,
## `role@start` (where the role's thing stood when the run started), `self.<Bone>` (that bone as
## the body is now, turned with the body), or a thing `resolver` finds; a Target, or null for an
## optional role left unbound.
func _ref(run: Run, ref):
    if ref is Array and ref.size() == 3:
        return Vector3(float(ref[0]), float(ref[1]), float(ref[2]))
    if not ref is String:
        return null
    if ref.ends_with("@start"):
        var role: String = ref.substr(0, ref.length() - 6)
        if not run.origins.has(role):
            push_error("motion '%s': '%s': '%s' is not a bound thing" % [run.name, ref, role])
            return null
        return Target.make(null, run.origins[role], ref)
    if ref.begins_with("self."):
        var bone: String = ref.substr(5)
        if not index.has(bone):
            push_error("motion '%s': this body has no bone '%s'" % [run.name, bone])
            return null
        return Target.make(null, body_world * Transform3D(Basis.IDENTITY, _fk(out)[index[bone]].origin), ref)
    if run.bindings.has(ref):
        return run.bindings[ref]
    var dot: int = ref.find(".")
    var head: String = ref if dot < 0 else ref.substr(0, dot)
    if run.bindings.has(head):
        var bound: Target = run.bindings[head]
        return contact(bound.thing, null, ref.substr(dot + 1)) if bound.thing != null else null
    if (run.motion.get("roles", {}) as Dictionary).has(head):
        return null                                    # an optional role left unbound
    return _bind(ref, null)


## A role's binding: a Thing (its contact for `grip`), a Target, or a name `resolver` finds
## ("cup"; "table.top": the attachment `top` of `table`).
func _bind(value, grip) -> Target:
    if value is Target:
        return value
    if value is Thing:
        return contact(value, grip)
    if not value is String or not resolver.is_valid():
        return null
    var thing = resolver.call(value)
    if thing is Thing:
        return contact(thing, grip)
    var dot: int = value.rfind(".")
    if dot < 0:
        return null
    thing = resolver.call(value.substr(0, dot))
    return contact(thing, null, value.substr(dot + 1)) if thing is Thing else null


## Where a body touches `thing` (motion.contact): its grip point `name`, else the first with
## `grip` (`any`: the first a hand holds it by), else its origin.
static func contact(thing: Thing, grip, name = null) -> Target:
    if thing == null:
        return null
    if name != null:
        if not thing.grips.has(name):
            push_error("'%s' has no grip point '%s' (it has %s)" % [thing.name, name, thing.grips.keys()])
            return null
        return Target.make(thing, thing.grips[name]["local"], "%s.%s" % [thing.name, name], thing.grips[name]["grip"])
    if grip != null:
        for k in thing.grips:
            var g: String = thing.grips[k]["grip"]
            if g == grip or (grip == "any" and g != "" and not g in NOT_HELD):
                return Target.make(thing, thing.grips[k]["local"], "%s.%s" % [thing.name, k], g)
        for a in thing.affordances:          # a seat's sit point: its first affordance of that type
            if a is Dictionary and a.get("type") == grip:
                var p: Array = a["position"]
                return Target.make(thing, Transform3D(Basis(Vector3.UP, deg_to_rad(float(a["yaw"]))),
                    Vector3(p[0], p[1], p[2])), "%s[%s]" % [thing.name, grip], grip)
    return Target.make(thing, Transform3D.IDENTITY, thing.name)


## A world transform in the body's frame.
func _body(t: Transform3D) -> Transform3D:
    return body_world.affine_inverse() * t


func _bend_to_reach(pose: Pose, beat: Pose, e: String, point: Vector3, run: Run, u: float, lead = null) -> void:
    var shoulder := e.replace("Hand", "UpperArm")
    var elbow := e.replace("Hand", "LowerArm")
    var reach: float = REACH_SLACK * ((rest_world[index[shoulder]].origin - rest_world[index[elbow]].origin).length()
        + (rest_world[index[elbow]].origin - rest_world[index[e]].origin).length())
    var full: float = reach / REACH_SLACK
    var spanned: float = (run.start_ends[e] - _fk(run.start_pose)[index[shoulder]].origin).length()
    reach = maxf(reach, reach + (minf(spanned, full) - reach) * (1.0 - u))
    var trial := pose.copy()
    var bend := func(degrees: float, at: Vector3, within: float) -> bool:
        for b in BEND_BONES:
            var i: int = index[b]
            trial.rot[i] = beat.rot[i] * _body_quat(i, _euler(Vector3(degrees / BEND_BONES.size(), 0, 0)))
        return at.distance_to(_world_of(index[shoulder], trial).origin) <= within
    var least := func(at: Vector3, within: float) -> float:
        if bend.call(0.0, at, within):
            return 0.0
        var lo := 0.0
        var hi := MAX_BEND
        if bend.call(hi, at, within):
            for _k in 14:
                var mid := 0.5 * (lo + hi)
                if bend.call(mid, at, within):
                    hi = mid
                else:
                    lo = mid
        return hi
    var degrees: float = least.call(point, reach)
    if lead != null:     # an arcing reach: the goal's bend eases in with the beat (motion.py)
        degrees = maxf(degrees, float(least.call(lead, full * REACH_SLACK)) * u)
    bend.call(degrees, point, reach)
    for b in BEND_BONES:
        pose.rot[index[b]] = trial.rot[index[b]]


## Turn the head toward `point` (world).
func _turn_head(pose: Pose, point: Vector3, weight: float) -> void:
    var head: Vector3 = _fk(pose)[index["Head"]].origin
    var local := body_world.affine_inverse() * point - head
    var yaw := YAW_LIMIT * tanh(atan2(local.x, local.z) / YAW_LIMIT)
    var pitch := PITCH_LIMIT * tanh(-atan2(local.y, Vector2(local.x, local.z).length()) / PITCH_LIMIT)
    for pair in [["Neck", NECK_SHARE], ["Head", 1.0 - NECK_SHARE]]:
        var i: int = index[pair[0]]
        var r := _euler_rad(Vector3(pitch * pair[1] * weight, yaw * pair[1] * weight, 0.0))
        pose.rot[i] = pose.rot[i] * _body_quat(i, r)


## Channels whose owner changed start from where the body was: the difference decays over the
## channel's half-life (the root and the drape correctives too). Returns the root as shown.
func _inertialize(pose: Pose, root: Transform3D) -> Transform3D:
    for c in _changed:
        var life: float = HALF_LIFE.get(c, DEFAULT_HALF_LIFE)
        for bone in channels.get(c, []):
            var i: int = index[bone]
            _offsets[i] = [out.rot[i] * pose.rot[i].inverse(), 0.0, life]
        if c == "root":
            _root_offset = [body_world * root.affine_inverse(), 0.0, life]
        if c == "hips":
            var delta := {}
            for k in _morph_goal:
                delta[k] = float(morphs.get(k, 0.0)) - float(_morph_goal[k])
            _morph_offset = [delta, 0.0]
    _changed = {}
    var mor := _morph_goal.duplicate()
    if _morph_offset != null:
        var wm := pow(0.5, _morph_offset[1] / MORPH_HALF_LIFE)
        if wm < 0.01:
            _morph_offset = null
        else:
            for k in mor:
                mor[k] = float(mor[k]) + float(_morph_offset[0].get(k, 0.0)) * wm
            _morph_offset[1] += dt
    morphs = mor
    for i in _offsets.keys():
        var o: Array = _offsets[i]
        var w := pow(0.5, o[1] / o[2])
        if w < 0.01:
            _offsets.erase(i)
            continue
        pose.rot[i] = Quaternion.IDENTITY.slerp(o[0], w) * pose.rot[i]
        o[1] += dt
    if _root_offset != null:
        var wr := pow(0.5, _root_offset[1] / _root_offset[2])
        if wr < 0.01:
            _root_offset = null
        else:
            var was: Transform3D = _root_offset[0] * root
            var yaw := _yaw_of(root) + _wrap(_yaw_of(was) - _yaw_of(root)) * wr
            root = _placement(root.origin + (was.origin - root.origin) * wr, yaw)
            _root_offset[1] += dt
    return root


## A body-frame point as a beat found it, in the body's frame now (the body may have moved since).
func _since(run: Run, p: Vector3) -> Vector3:
    return body_world.affine_inverse() * (run.start_root * p)


# --- the skeleton's maths (core/skeleton.py, core/ik.py) ---

func _posed_local(i: int, pose: Pose) -> Transform3D:
    var r: Transform3D = rest[i]
    return Transform3D(r.basis * Basis(pose.rot[i]), r.origin + pose.off[i])


func _fk(pose: Pose) -> Array:
    var w := []
    w.resize(names.size())
    for i in names.size():
        var local_ := _posed_local(i, pose)
        w[i] = local_ if parent[i] < 0 else w[parent[i]] * local_
    return w


func _world_of(i: int, pose: Pose) -> Transform3D:
    var chain := []
    var j := i
    while j >= 0:
        chain.push_front(j)
        j = parent[j]
    var w := Transform3D.IDENTITY
    for k in chain:
        w = w * _posed_local(k, pose)
    return w


## A body-frame rotation as a pose rotation for bone `i` (Skeleton.body_quat).
func _body_quat(i: int, r: Basis) -> Quaternion:
    var rw: Basis = rest_world[i].basis
    return Quaternion((rw.transposed() * r * rw).orthonormalized())


## The pose rotation that gives bone `i` world rotation `world_rot` under `parent_world`.
func _pose_quat(i: int, parent_world: Transform3D, world_rot: Basis) -> Quaternion:
    var local_ := parent_world.basis.transposed() * world_rot
    return Quaternion((rest[i].basis.transposed() * local_).orthonormalized())


func _solve_limb(pose: Pose, e: String, target: Vector3, pole_, end_rotation) -> bool:
    var limb: Array = LIMBS[e]
    var upper: int = index[limb[0]]
    var lower: int = index[limb[1]]
    var end: int = index[e]
    var rest_dir: Vector3 = limb[2]
    var rest_pole: Vector3 = limb[3]
    var pole: Vector3 = rest_pole if pole_ == null else Vector3(pole_[0], pole_[1], pole_[2])
    var rest_normal := rest_pole.cross(rest_dir)
    var a: float = (rest_world[lower].origin - rest_world[upper].origin).length()
    var b: float = (rest_world[end].origin - rest_world[lower].origin).length()
    var parent_world := _world_of(parent[upper], pose)
    var root := (parent_world * _posed_local(upper, pose)).origin
    var placed := _two_bone(root, a, b, target, pole)
    var middle: Vector3 = placed[0]
    var tip: Vector3 = placed[1]
    var normal := pole.cross(tip - root)
    var up := _aim(rest_world[upper].basis, rest_dir, rest_normal, middle - root, normal)
    pose.rot[upper] = _pose_quat(upper, parent_world, up)
    var upper_world := parent_world * _posed_local(upper, pose)
    var low := _aim(rest_world[lower].basis, rest_dir, rest_normal, tip - middle, normal)
    pose.rot[lower] = _pose_quat(lower, upper_world, low)
    if end_rotation != null:
        var lower_world := upper_world * _posed_local(lower, pose)
        pose.rot[end] = _pose_quat(end, lower_world, end_rotation * rest_world[end].basis)
    return placed[2]


static func _two_bone(root: Vector3, upper: float, lower: float, target: Vector3, pole: Vector3) -> Array:
    var delta := target - root
    var dist := delta.length()
    var u := delta / dist if dist > 1e-9 else Vector3(0, -1, 0)
    var lo := absf(upper - lower) + 1e-6
    var hi := upper + lower - 1e-6
    var reach := clampf(dist, lo, hi)
    var cos_a := clampf((upper * upper + reach * reach - lower * lower) / (2.0 * upper * reach), -1.0, 1.0)
    var side := pole - pole.dot(u) * u
    if side.length() < 1e-9:
        side = u.cross(Vector3(1, 0, 0)) if absf(u.x) < 0.9 else u.cross(Vector3(0, 0, 1))
    side = side.normalized()
    var middle := root + upper * (cos_a * u + sqrt(1.0 - cos_a * cos_a) * side)
    return [middle, root + reach * u, absf(reach - dist) < 1e-4]


static func _aim(rest_rotation: Basis, rest_dir: Vector3, rest_normal: Vector3, direction: Vector3, normal: Vector3) -> Basis:
    return _frame(direction, normal) * _frame(rest_dir, rest_normal).transposed() * rest_rotation


static func _frame(d: Vector3, n: Vector3) -> Basis:
    d = d.normalized()
    n = (n - n.dot(d) * d).normalized()
    return Basis(d, n, d.cross(n))


## XYZ Euler degrees as Transform does it: R = Rz * Ry * Rx.
static func _euler(degrees: Vector3) -> Basis:
    return _euler_rad(degrees * (PI / 180.0))


static func _euler_rad(r: Vector3) -> Basis:
    return Basis(Vector3(0, 0, 1), r.z) * Basis(Vector3(0, 1, 0), r.y) * Basis(Vector3(1, 0, 0), r.x)


static func ease_(kind: String, u: float) -> float:
    u = clampf(u, 0.0, 1.0)
    if kind == "linear":
        return u
    if kind == "settle":
        return 1.0 - pow(1.0 - u, 4)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


static func _bump(u: float, at: float) -> float:
    u = clampf(u, 0.0, 1.0)
    var x := u / at if u < at else (1.0 - u) / maxf(1.0 - at, 1e-6)
    return ease_("smooth", x)


# --- {expr} ---

## `value` with every "{expr}" string evaluated against `env` (geogen/layout/expressions.py does
## this at build time; a motion's are left for its start, from its params and the body's).
static func resolve(value, env: Dictionary):
    if value is Dictionary:
        var out_ := {}
        for k in value:
            out_[k] = resolve(value[k], env)
        return out_
    if value is Array:
        return value.map(func(v): return resolve(v, env))
    if value is String and value.begins_with("{") and value.ends_with("}"):
        var e := Expression.new()
        var keys := env.keys()
        if e.parse(value.substr(1, value.length() - 2), PackedStringArray(keys)) != OK:
            push_error("motion expression %s: %s" % [value, e.get_error_text()])
            return 0.0
        var got = e.execute(keys.map(func(k): return env[k]))
        if e.has_execute_failed():
            push_error("motion expression %s: %s" % [value, e.get_error_text()])
            return 0.0
        return float(got)
    return value


func _event(event: String, run: Run, data: Dictionary) -> void:
    var entry := {"t": snappedf(time, 0.0001), "event": event, "motion": run.name if run != null else null}
    if run != null:
        entry["beat"] = run.beat if run.beat != "" else null
    entry.merge(data)
    trace.append(entry)


# --- conformance (tests/data/motion, geogen/motion_review.py motion_trace) ---

## Replay a motion trace; the mismatches ([] when this rig conforms to the reference). Its
## `things` are fixed Things in the body's frame (the trace's body stands at the origin); a
## script's `<thing>@start` is a fixed Thing where that one stood.
static func run_trace(trace: Dictionary) -> Array[String]:
    var problems: Array[String] = []
    var rig := GeogenMotionRig.from_extras(trace["rig"], int(trace.get("fps", 30)))
    rig.add_transitions(trace)
    var things := {}
    for k in trace.get("things", {}):
        var spec: Dictionary = trace["things"][k]
        things[k] = Thing.fixed(k, _xform12(spec["world"]), spec.get("grips", {}), spec.get("affordances", []))
    rig.resolver = func(name: String): return things.get(name)
    var motions: Dictionary = trace.get("motions", {})
    var pending: Array = (trace.get("script", []) as Array).duplicate()
    var expected := {}
    var expected_things := {}
    var expected_root := {}
    var expected_morphs := {}
    for f in trace["expect"]["positions"]:
        expected[int(f["frame"])] = f["positions"]
        if f.has("things"):
            expected_things[int(f["frame"])] = f["things"]
        if f.has("root"):
            expected_root[int(f["frame"])] = f["root"]
        if f.has("morphs"):
            expected_morphs[int(f["frame"])] = f["morphs"]
    var tolerance := float(trace.get("tolerance", 0.002))
    var worst := 0.0
    var where := ""
    var worst_morph := 0.0
    var where_morph := ""
    var frames := int(trace["frames"])
    for frame in range(1, frames + 1):
        while not pending.is_empty() and float(pending[0].get("at", 0.0)) <= rig.time + 1e-9:
            var entry: Dictionary = pending.pop_front()
            var values := {}
            for k in entry:
                var v = entry[k]
                if v is String and v.ends_with("@start") and things.has(v.substr(0, v.length() - 6)):
                    var base: String = v.substr(0, v.length() - 6)
                    v = Thing.fixed(v, _xform12(trace["things"][base]["world"]))
                values[k] = v
            for k in ["at", "start", "break", "interrupt", "how"]:
                values.erase(k)
            if entry.has("start"):
                if rig.start(motions[entry["start"]], values) == null:
                    problems.append("frame %d: %s didn't start" % [frame, entry["start"]])
            elif entry.has("interrupt"):
                rig.interrupt(entry["interrupt"], str(entry.get("how", "yield")))
            elif entry.has("break"):
                rig.break_(motions.get(entry["break"]) if entry["break"] is String else null, values)
        rig.step()
        if expected.has(frame):
            var w := rig.world()
            var want: Dictionary = expected[frame]
            for bone in want:
                var p: Array = want[bone]
                var d: float = w[rig.index[bone]].origin.distance_to(Vector3(p[0], p[1], p[2]))
                if d > worst:
                    worst = d
                    where = "%s at frame %d" % [bone, frame]
        if expected_root.has(frame):
            var want: Transform3D = _xform12(expected_root[frame])
            for probe in [Vector3.ZERO, Vector3(0.1, 0, 0), Vector3(0, 0.1, 0), Vector3(0, 0, 0.1)]:
                var d: float = (rig.body_world * probe).distance_to(want * probe)
                if d > worst:
                    worst = d
                    where = "the body's root at frame %d" % frame
        if expected_morphs.has(frame):
            for k in expected_morphs[frame]:
                var d := absf(float(rig.morphs.get(k, 0.0)) - float(expected_morphs[frame][k]))
                if d > worst_morph:
                    worst_morph = d
                    where_morph = "%s at frame %d" % [k, frame]
        if expected_things.has(frame):
            for k in expected_things[frame]:
                var want: Transform3D = _xform12(expected_things[frame][k])
                var have: Transform3D = (things[k] as Thing).world()
                # Its origin, and points 0.1 m along each of its axes.
                for probe in [Vector3.ZERO, Vector3(0.1, 0, 0), Vector3(0, 0.1, 0), Vector3(0, 0, 0.1)]:
                    var d: float = (have * probe).distance_to(want * probe)
                    if d > worst:
                        worst = d
                        where = "%s at frame %d" % [k, frame]
    if worst > tolerance:
        problems.append("bones off the reference by up to %.1f mm (%s; tolerance %.1f mm)" % [worst * 1000.0, where, tolerance * 1000.0])
    if worst_morph > MORPH_TOLERANCE:
        problems.append("drape correctives off the reference by up to %.4f (%s)" % [worst_morph, where_morph])
    var events := rig.trace.filter(func(e): return e["event"] in TRACE_EVENTS) \
        .map(func(e): return [e["event"], e["motion"], e.get("beat")])
    var want_events: Array = trace["expect"]["events"]
    if JSON.stringify(events) != JSON.stringify(want_events):
        problems.append("events %s, expected %s" % [JSON.stringify(events), JSON.stringify(want_events)])
    if not rig.running().is_empty():
        problems.append("still running after %d frames: %s" % [frames, rig.running()])
    return problems


const MORPH_TOLERANCE := 0.002   # of a morph weight

## The events a trace holds the rig to (motion_review.TRACE_EVENTS).
const TRACE_EVENTS := ["start", "beat", "loop", "emit", "interrupt", "end", "break", "pause", "resume", "attach", "release"]


## A transform from 12 numbers: a 3x4 matrix, row by row (motion_review.motion_trace).
static func _xform12(m: Array) -> Transform3D:
    return Transform3D(Basis(Vector3(m[0], m[4], m[8]), Vector3(m[1], m[5], m[9]), Vector3(m[2], m[6], m[10])),
        Vector3(m[3], m[7], m[11]))
