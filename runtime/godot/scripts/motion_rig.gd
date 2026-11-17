class_name GeogenMotionRig
extends RefCounted
## Runs motions (`kind: motion`, motions.json) on one body: the runtime port of the reference rig,
## geogen/motion.py MotionRig (docs/dev/characters.md "Motions"). Pure data: built from the body's
## `extras.geogen.rig` (bones and their rests, params, channels, hand shapes, pose library), it
## steps at a fixed rate and gives every bone's local transform (`local()`), which
## GeogenMotionPlayer writes to the Skeleton3D. Conformance traces (tests/data/motion/) hold the
## two rigs to the same bone positions.
##
## Ported so far: beats (pose, bones, hips, ik onto body-frame points with hand shapes and pole,
## look at a body-frame point, lean, plant: feet, over, ease, after, emit, then, loop, interrupt
## and exit), channel arbitration, pausing under an overlay and `resume`, a break into a reaction,
## inertialization, the reach bend, free arms hanging under a bent spine and the trace. A motion that
## names things in the world (roles, root targets, attach/release) or a posture change with a
## transition clip is not ported yet: `start` refuses it.

const REACH_SPEED := 0.9
const MIN_OVER := 0.35
const HALF_LIFE := {"gaze": 0.08, "left_hand": 0.12, "right_hand": 0.12}
const DEFAULT_HALF_LIFE := 0.18
const REACH_SLACK := 0.96
const BEND_BONES := ["Spine", "Chest"]
const MAX_BEND := 75.0
const LOOK_IN := 0.3
const LOOK_OUT := 0.4            # s: a look that ends eases off in this long
const LOOK_FOLLOW := 0.1
const YAW_LIMIT := deg_to_rad(70.0)
const PITCH_LIMIT := deg_to_rad(35.0)
const NECK_SHARE := 0.4
const STRENGTHS := {"prefer": 1, "claim": 2, "require": 3}
const FINGERS := ["Thumb", "Index", "Middle", "Ring", "Little"]
const LOOK_BONES := ["Neck", "Head"]
const NOT_PORTED := ["root", "attach", "release", "when", "until", "do", "bend"]
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


# --- requests ---

## Start `motion` (a spec from motions.json) with `values` for its params (`reps`, ...).
## Returns the run, or null with the reason pushed as an error.
func start(motion: Dictionary, values := {}) -> Run:
    var name := str(motion.get("name", "motion"))
    if not (motion.get("roles", {}) as Dictionary).is_empty():
        push_error("motion '%s': roles (things in the world) are not ported to the runtime rig yet" % name)
        return null
    var given := {}
    for k in motion.get("params", {}):
        var d = motion["params"][k]
        given[k] = d.get("default") if d is Dictionary else d
    for k in values:
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
    var beats = resolve(motion["beats"], env)
    for beat in beats:
        for key in NOT_PORTED:
            if (beats[beat] as Dictionary).has(key):
                push_error("motion '%s' beat '%s': '%s' is not ported to the runtime rig yet" % [name, beat, key])
                return null
    var spec := motion.duplicate()
    spec["beats"] = beats
    _order += 1
    var run := Run.new()
    run.motion = spec
    run.name = name
    run.params = given
    run.order = _order
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
    return start(reaction, values) if reaction is Dictionary else null


func running() -> Array:
    return runs.filter(func(r): return not r.done).map(func(r): return r.name)


# --- the step ---

func step() -> void:
    time += dt
    for run in runs.duplicate():
        if not run.done:
            _advance(run)
    runs = runs.filter(func(r): return not r.done)
    _arbitrate()
    out = _compose()


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
            _end(run)
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
    if spec.get("emit"):
        _event("emit", run, {"emit": spec["emit"]})
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
    var fk := _fk(out)
    for e in ["LeftHand", "RightHand", "LeftFoot", "RightFoot"]:
        if index.has(e):
            run.start_ends[e] = fk[index[e]].origin
    run.start_feet = {}
    for f in ["LeftFoot", "RightFoot"]:
        if index.has(f):
            run.start_feet[f] = fk[index[f]].origin
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
        var target = _ik_target(entries[e])
        if target != null and run.start_ends.has(e):
            travel = maxf(travel, (target - run.start_ends[e]).length())
    if travel > 0.0:
        return maxf(MIN_OVER, travel / REACH_SPEED + 0.15)
    return MIN_OVER if spec.get("pose") else 0.0


func _end(run: Run) -> void:
    if not run.done:
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
        var was: Run = owners.get(c)
        if best != was:
            if c == "gaze" and _look_key != null:
                _fade_look()             # the head eases off the old look rather than springing
                _changed.erase(c)        # (an ended owner asked for it to be inertialized)
            else:
                _changed[c] = true
            _event("owner", best, {"channel": c, "was": was.name if was != null else null})
            if was != null and not was.done and was.strength(c) == STRENGTHS["require"]:
                interrupt(was.name, "yield")
        owners[c] = best
    _pause_and_resume()


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


## Strength, then priority, then the later started.
func _outranks(a: Run, b: Run, c: String) -> bool:
    var ka := [a.strength(c), int(a.motion.get("priority", 0)), a.order]
    var kb := [b.strength(c), int(b.motion.get("priority", 0)), b.order]
    for i in 3:
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
    # Poses first, every owner over its own channels, eased from where its beat began.
    for run in runs:
        var held := _owned(run)
        if held.is_empty():
            continue
        var spec: Dictionary = run.spec()
        var u := ease_(str(spec.get("ease", "smooth")), run.t / run.over if run.over > 0.0 else 1.0)
        var target := _beat_pose(run)
        for i in names.size():
            if not (channel_of.get(names[i], "") in held):
                continue
            pose.rot[i] = run.start_pose.rot[i].slerp(target.rot[i], u)
            pose.off[i] = run.start_pose.off[i] + (target.off[i] - run.start_pose.off[i]) * u
    _inertialize(pose)
    # Then the solve, outside in: lean, planted feet, hands, gaze.
    for run in runs:
        var held := _owned(run)
        var spec: Dictionary = run.spec()
        var u_raw: float = run.t / run.over if run.over > 0.0 else 1.0
        var u := ease_(str(spec.get("ease", "smooth")), u_raw)
        var lean = spec.get("lean")
        if lean is Dictionary and "spine" in held and index.has("Spine"):
            var by := float(lean.get("by", 0.0)) * _bump(u_raw, float(lean.get("at", 0.5)))
            var i: int = index["Spine"]
            pose.rot[i] = pose.rot[i] * _body_quat(i, _euler(Vector3(by, 0, 0)))
        if spec.get("plant") == "feet" and "legs" in held:
            for foot in run.start_feet:
                _solve_limb(pose, foot, run.start_feet[foot], null, Basis.IDENTITY)   # flat on the floor
        var entries := _ik_entries(run)
        var laid := pose.copy()          # the spine as laid, before any reach bends it
        for e in entries:
            if not (channel_of.get(e, "") in held):
                continue
            var goal = _ik_target(entries[e])
            if goal == null:
                continue
            var start: Vector3 = run.start_ends[e]
            var point: Vector3 = start + (goal - start) * u
            if "spine" in held and e in ["LeftHand", "RightHand"]:
                _bend_to_reach(pose, laid, e, point, run, u)
            var solved := pose.copy()
            _solve_limb(solved, e, point, entries[e].get("pole"), null)
            var limb: Array = LIMBS[e]
            for bone in [limb[0], limb[1], e]:
                var i: int = index[bone]
                pose.rot[i] = run.start_pose.rot[i].slerp(solved.rot[i], u)
    _hang_free_arms(pose)
    for b in LOOK_BONES:
        if index.has(b):
            _prelook[index[b]] = pose.rot[index[b]]
    var gaze: Run = owners.get("gaze")
    var look = gaze.spec().get("look") if gaze != null else null
    var point = null
    if look != null:
        point = _point(look.get("at") if look is Dictionary else look)
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
        var swing := Basis(Quaternion(a, b))
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
        var shape = _hand_shape(e, entries[e])
        if shape == null:
            continue
        var side: String = e.substr(0, e.length() - 4)
        for bone in fingers[side]:
            # A shape poses every finger (both hands' bones are in it): one it doesn't name is straight.
            var d: Array = shape.get(bone, [0.0, 0.0, 0.0])
            pose.rot[index[bone]] = _body_quat(index[bone], _euler(Vector3(d[0], d[1], d[2])))
    return pose


func _hand_shape(e: String, entry: Dictionary):
    if not e in ["LeftHand", "RightHand"]:
        return null
    var grip = entry.get("grip")
    if grip == null:
        return null
    if hand_shapes.has(grip):
        return hand_shapes[grip]
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
                if t is Array:
                    e["target"] = [-float(t[0]), t[1], t[2]]
                if e.has("pole"):
                    var p: Array = e["pole"]
                    e["pole"] = [-float(p[0]), p[1], p[2]]
                entries[twin] = e
    return entries


static func _mirror_name(n: String) -> String:
    if n.begins_with("Left"):
        return "Right" + n.substr(4)
    if n.begins_with("Right"):
        return "Left" + n.substr(5)
    return ""


func _ik_target(entry: Dictionary):
    return _point(entry.get("target"))


## A beat's point: [x, y, z] in the body's frame (what the runtime rig supports so far).
func _point(ref):
    if ref is Array and ref.size() == 3:
        return Vector3(float(ref[0]), float(ref[1]), float(ref[2]))
    return null


func _bend_to_reach(pose: Pose, beat: Pose, e: String, point: Vector3, run: Run, u: float) -> void:
    var shoulder := e.replace("Hand", "UpperArm")
    var elbow := e.replace("Hand", "LowerArm")
    var reach: float = REACH_SLACK * ((rest_world[index[shoulder]].origin - rest_world[index[elbow]].origin).length()
        + (rest_world[index[elbow]].origin - rest_world[index[e]].origin).length())
    var full: float = reach / REACH_SLACK
    var spanned: float = (run.start_ends[e] - _fk(run.start_pose)[index[shoulder]].origin).length()
    reach = maxf(reach, reach + (minf(spanned, full) - reach) * (1.0 - u))
    var trial := pose.copy()
    var bend := func(degrees: float) -> bool:
        for b in BEND_BONES:
            var i: int = index[b]
            trial.rot[i] = beat.rot[i] * _body_quat(i, _euler(Vector3(degrees / BEND_BONES.size(), 0, 0)))
        return point.distance_to(_world_of(index[shoulder], trial).origin) <= reach
    if not bend.call(0.0):
        var lo := 0.0
        var hi := MAX_BEND
        if bend.call(hi):
            for _k in 14:
                var mid := 0.5 * (lo + hi)
                if bend.call(mid):
                    hi = mid
                else:
                    lo = mid
        bend.call(hi)
    for b in BEND_BONES:
        pose.rot[index[b]] = trial.rot[index[b]]


func _turn_head(pose: Pose, point: Vector3, weight: float) -> void:
    var head: Vector3 = _fk(pose)[index["Head"]].origin
    var local := point - head
    var yaw := YAW_LIMIT * tanh(atan2(local.x, local.z) / YAW_LIMIT)
    var pitch := PITCH_LIMIT * tanh(-atan2(local.y, Vector2(local.x, local.z).length()) / PITCH_LIMIT)
    for pair in [["Neck", NECK_SHARE], ["Head", 1.0 - NECK_SHARE]]:
        var i: int = index[pair[0]]
        var r := _euler_rad(Vector3(pitch * pair[1] * weight, yaw * pair[1] * weight, 0.0))
        pose.rot[i] = pose.rot[i] * _body_quat(i, r)


func _inertialize(pose: Pose) -> void:
    for c in _changed:
        var life: float = HALF_LIFE.get(c, DEFAULT_HALF_LIFE)
        for bone in channels.get(c, []):
            var i: int = index[bone]
            _offsets[i] = [out.rot[i] * pose.rot[i].inverse(), 0.0, life]
    _changed = {}
    for i in _offsets.keys():
        var o: Array = _offsets[i]
        var w := pow(0.5, o[1] / o[2])
        if w < 0.01:
            _offsets.erase(i)
            continue
        pose.rot[i] = Quaternion.IDENTITY.slerp(o[0], w) * pose.rot[i]
        o[1] += dt


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

## Replay a motion trace; the mismatches ([] when this rig conforms to the reference).
static func run_trace(trace: Dictionary) -> Array[String]:
    var problems: Array[String] = []
    var rig := GeogenMotionRig.from_extras(trace["rig"], int(trace.get("fps", 30)))
    var motions: Dictionary = trace.get("motions", {})
    var pending: Array = (trace.get("script", []) as Array).duplicate()
    var expected := {}
    for f in trace["expect"]["positions"]:
        expected[int(f["frame"])] = f["positions"]
    var tolerance := float(trace.get("tolerance", 0.002))
    var worst := 0.0
    var where := ""
    var frames := int(trace["frames"])
    for frame in range(1, frames + 1):
        while not pending.is_empty() and float(pending[0].get("at", 0.0)) <= rig.time + 1e-9:
            var entry: Dictionary = pending.pop_front()
            if entry.has("start"):
                var values := entry.duplicate()
                for k in ["at", "start"]:
                    values.erase(k)
                if rig.start(motions[entry["start"]], values) == null:
                    problems.append("frame %d: %s didn't start" % [frame, entry["start"]])
            elif entry.has("interrupt"):
                rig.interrupt(entry["interrupt"], str(entry.get("how", "yield")))
            elif entry.has("break"):
                var values := entry.duplicate()
                for k in ["at", "break"]:
                    values.erase(k)
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
    if worst > tolerance:
        problems.append("bones off the reference by up to %.1f mm (%s; tolerance %.1f mm)" % [worst * 1000.0, where, tolerance * 1000.0])
    var events := rig.trace.filter(func(e): return e["event"] in ["start", "beat", "loop", "emit", "interrupt", "end",
                                               "break", "pause", "resume"]) \
        .map(func(e): return [e["event"], e["motion"], e.get("beat")])
    var want_events: Array = trace["expect"]["events"]
    if JSON.stringify(events) != JSON.stringify(want_events):
        problems.append("events %s, expected %s" % [JSON.stringify(events), JSON.stringify(want_events)])
    if not rig.running().is_empty():
        problems.append("still running after %d frames: %s" % [frames, rig.running()])
    return problems
