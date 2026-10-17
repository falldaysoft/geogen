class_name GeogenNpcAnimator
extends RefCounted
## An NPC body's skeletal animation (npc.gd): its clips (extras.geogen.clips) on an AnimationPlayer
## under the body, the keyframed moves between poses (sit_down, stand_up, ...) with their root paths,
## pose clips (pose_<name>; standing: idle) and walk/idle locomotion. The NPC decides the pose and
## places itself; this poses the skeleton and moves the body under it.

const TRANSITION_BLEND := 0.12    # s to cross-fade into and out of a transition clip (sit_down, ...)
const POSE_BLEND := 0.4           # s to cross-fade skeletal pose clips (and ease the body into place)
const LOCOMOTION_BLEND := 0.25    # s to cross-fade walk <-> idle
const WALK_THRESHOLD := 0.15      # m/s: slower than this counts as standing still

var body: Node3D
var player: AnimationPlayer = null  # null: a body without clips (a capsule)
var clip_speed := {}                # clip name -> ground speed it was authored for (m/s)
var transitions := {}               # "<from>><to>" -> {clip, times, positions, rotations}
var transition := {}                # the one playing: {move, t, start, target, pose}
var _clips := {}                    # clip name -> Animation, until attach()
var _tween: Tween = null


## Read the body's clips while it is still where the export put it (beside its AnimationPlayer).
static func load(body_: Node3D) -> GeogenNpcAnimator:
    var out := GeogenNpcAnimator.new()
    out.body = body_
    out._clips = _body_clips(body_)
    for clip in WorldLoader.geogen_extras(body_).get("clips", []):
        if clip.has("speed"):
            out.clip_speed[clip["name"]] = float(clip["speed"])
        if clip.has("root") and out._clips.has(clip["name"]):
            var root: Dictionary = clip["root"]
            var move := {"clip": clip["name"], "times": PackedFloat32Array(root["times"]),
                "positions": PackedVector3Array(), "rotations": []}
            for q in root["positions"]:
                move["positions"].append(Vector3(q[0], q[1], q[2]))
            for q in root["rotations"]:
                move["rotations"].append(Quaternion(q[0], q[1], q[2], q[3]))
            out.transitions["%s>%s" % [clip["from"], clip["to"]]] = move
    return out


## Once the body is under the NPC: an AnimationPlayer for its clips, standing.
func attach() -> void:
    if _clips.is_empty():
        return
    player = AnimationPlayer.new()
    player.name = "Clips"
    var library := AnimationLibrary.new()
    for clip_name in _clips:
        library.add_animation(clip_name, _clips[clip_name])
    library.add_animation("RESET", _rest_animation(body, _clips.values()))
    player.add_animation_library("", library)
    body.add_child(player)   # root_node ".." = the body
    _clips = {}
    play_pose("stand", 0.0)


func has(clip: String) -> bool:
    return player != null and player.has_animation(clip)


func clip(clip_name: String) -> Animation:
    return player.get_animation(clip_name)


## The clip playing ("" without clips).
func current() -> String:
    return String(player.current_animation) if player != null else ""


## A transition clip is playing (its length sets the pace of the pose step).
func moving() -> bool:
    return not transition.is_empty()


## Play ``clip_name`` once from standing; its length.
func play_once(clip_name: String) -> float:
    player.speed_scale = 1.0
    player.play(clip_name, LOCOMOTION_BLEND)
    return player.get_animation(clip_name).length


## The NPC changed pose from ``was``: the body goes from ``start`` (its old transform, in the NPC's
## new frame) to ``target``, by the change's keyframed move if it has one (true: it is playing),
## else easing there while the pose clips cross-fade (``owner`` runs the tween).
func change_pose(was: String, pose_name: String, start: Transform3D, target: Transform3D, owner: Node) -> bool:
    if _tween != null:
        _tween.kill()
    transition = {}
    var move: Dictionary = transitions.get("%s>%s" % [was, pose_name], {})
    if player != null and not move.is_empty():
        # A keyframed move (sit_down, stand_up, ...): the clip poses the skeleton while the
        # body follows the clip's root path from where it was to where the pose puts it.
        transition = {"move": move, "t": 0.0, "start": start, "target": target, "pose": pose_name}
        body.transform = start
        player.speed_scale = 1.0
        player.play(move["clip"], TRANSITION_BLEND)
        return true
    if player != null:
        # No clip for this change: ease into place while the pose clips cross-fade.
        _tween = owner.create_tween()
        _tween.tween_property(body, "transform", target, POSE_BLEND)
    else:
        body.transform = target
    play_pose(pose_name, POSE_BLEND)
    return false


## Follow the playing transition's root path. The path was made for the action's default
## approach and seat height; what differs here (a deeper or lower seat) is spread over the move.
func advance(delta: float) -> void:
    if transition.is_empty():
        return
    var move: Dictionary = transition["move"]
    var times: PackedFloat32Array = move["times"]
    var positions: PackedVector3Array = move["positions"]
    var last := times.size() - 1
    var target: Transform3D = transition["target"]
    transition["t"] += delta
    var t := minf(transition["t"], times[last])
    if t >= times[last]:
        body.transform = target
        var pose_name: String = transition["pose"]
        transition = {}
        play_pose(pose_name, TRANSITION_BLEND)
        return
    var i := 0
    while i < last - 1 and times[i + 1] <= t:
        i += 1
    var f := clampf((t - times[i]) / maxf(times[i + 1] - times[i], 1e-6), 0.0, 1.0)
    var g := smoothstep(0.0, 1.0, t / times[last])
    var start: Transform3D = transition["start"]
    var origin := positions[i].lerp(positions[i + 1], f) + (start.origin - positions[0]) * (1.0 - g) \
        + (target.origin - positions[last]) * g
    var turn: Quaternion = (move["rotations"][i] as Quaternion).slerp(move["rotations"][i + 1], f)
    body.transform = Transform3D(Basis(turn).scaled_local(target.basis.get_scale()), origin)


## Skeletal bodies: hold the pose's clip (pose_<name>; standing: walk or idle); capsules have none.
func play_pose(pose_name: String, blend: float) -> void:
    if player == null:
        return
    var clip := "pose_" + pose_name
    if pose_name == "stand" and player.has_animation("idle"):
        clip = "idle"
    if player.has_animation(clip):
        player.speed_scale = 1.0
        player.play(clip, blend)


## Standing skeletal bodies walk while moving at ``speed`` (playback scaled to the ground speed so
## the feet don't slide) and idle when still; not while a clip is held (``holding``).
func locomote(speed: float, standing: bool, holding: bool) -> void:
    if player == null or not standing or holding or not transition.is_empty() \
            or not player.has_animation("walk"):
        return
    var clip := "walk" if speed > WALK_THRESHOLD else ("idle" if player.has_animation("idle") else "pose_stand")
    if clip == "walk":
        player.speed_scale = speed / float(clip_speed.get("walk", 1.2))
    else:
        player.speed_scale = 1.0
    if player.current_animation != clip:
        player.play(clip, LOCOMOTION_BLEND)


## RESET: every track the clips animate, keyed at the skeleton's rest (blend shapes at 0). The importer drops
## tracks equal to the rest (the exported stand pose), so blending from a clip that moves a
## bone into one that doesn't must fall back to the rest, not keep the old pose.
static func _rest_animation(body: Node, anims: Array) -> Animation:
    var reset := Animation.new()
    reset.length = 0.001
    var seen := {}
    for anim: Animation in anims:
        for t in anim.get_track_count():
            var path := anim.track_get_path(t)
            var kind := anim.track_get_type(t)
            var key := "%s|%d" % [path, kind]
            if seen.has(key) or not kind in [Animation.TYPE_ROTATION_3D, Animation.TYPE_POSITION_3D,
                    Animation.TYPE_BLEND_SHAPE]:
                continue
            seen[key] = true
            if kind == Animation.TYPE_BLEND_SHAPE:     # pose correctives (a seated skirt) rest at 0
                var shape := reset.add_track(kind)
                reset.track_set_path(shape, path)
                reset.blend_shape_track_insert_key(shape, 0.0, 0.0)
                continue
            var sk := body.get_node_or_null(NodePath(String(path).get_slice(":", 0))) as Skeleton3D
            var bone := sk.find_bone(String(path).get_slice(":", 1)) if sk != null else -1
            if bone < 0:
                continue
            var rest := sk.get_bone_rest(bone)
            var track := reset.add_track(kind)
            reset.track_set_path(track, path)
            if kind == Animation.TYPE_ROTATION_3D:
                reset.rotation_track_insert_key(track, 0.0, rest.basis.get_rotation_quaternion())
            else:
                reset.position_track_insert_key(track, 0.0, rest.origin)
    return reset


## The body's skeletal clips (extras.geogen.clips) copied out of the export's AnimationPlayer
## with their tracks re-pathed relative to the body, so they still play once the body
## moves under the NPC: {clip name: Animation}.
static func _body_clips(body: Node) -> Dictionary:
    var out := {}
    var listed: Array = WorldLoader.geogen_extras(body).get("clips", [])
    if listed.is_empty():
        return out
    var root := body
    while root.get_parent() != null and not root.get_parent() is WorldLoader:
        root = root.get_parent()
    for ap: AnimationPlayer in root.find_children("*", "AnimationPlayer", true, false):
        var base := ap.get_node(ap.root_node)
        var prefix := String(base.get_path_to(body)) + "/"
        for clip in listed:
            var anim_name := String(clip.get("animation", ""))
            if out.has(clip["name"]) or not ap.has_animation(anim_name):
                continue
            var anim: Animation = ap.get_animation(anim_name).duplicate(true)
            for t in anim.get_track_count():
                var path := String(anim.track_get_path(t))
                if path.begins_with(prefix):
                    anim.track_set_path(t, NodePath(path.substr(prefix.length())))
            # Static pose clips loop so the pose (and current_animation) holds.
            var hold: bool = clip.get("loop", false) or String(clip["name"]).begins_with("pose_")
            anim.loop_mode = Animation.LOOP_LINEAR if hold else Animation.LOOP_NONE
            out[clip["name"]] = anim
    return out
