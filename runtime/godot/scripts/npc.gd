class_name GeogenNpc
extends CharacterBody3D
## An NPC exported in a node's extras.geogen.npc (see geogen/npc.py and
## assets/npcs/). This script is a generic interpreter: everything specific
## comes from the definition and from what the world advertises.
##
## - Needs fall at their declared rates; doing something adds its
##   `advertises` to them.
## - Deciding: every option (affordances in the home region with a free
##   slot, plus the NPC's own activities) scores
##       sum((1 - need) x advertised) x preference(tags)
##       - distance x metres - recency x exp(-age / memory) + noise x random
##   with weights from the definition's `scoring`; the best one runs.
## - Running: an option's action is a list of steps (actions.yaml) from a
##   fixed vocabulary: go_to, face, pose, wait, use. A path through a closed
##   door (a `portal`) pushes that door's `pass` action first.
##
## The body asset (a child of the exported node) is reparented under this
## CharacterBody3D; poses are root transforms the body asset declares.

signal traced(event: Dictionary)
## An actor greeted this NPC: `signal_name` is its definition's `attention.greet.signal`,
## which WorldLoader posts on the event bus as a signal from the NPC (docs/rules.md).
signal greeted(signal_name: String, by: String)

const LAYER := 2                  # NPC bodies; the navmesh bakes layer 1 only
const WAYPOINT_RADIUS := 0.15
const STALL_SECONDS := 0.6        # no progress this long: sidestep
const SIDESTEP_SECONDS := 0.4
const FAIL_SECONDS := 4.0         # no progress this long: give up on the step
const REPLAN_SECONDS := 1.5       # no progress this long on a detour round a character: plan again
const MAX_REPLANS := 2            # ... this often per go_to
const STAIR_HOLD := 2.0           # m before a stair to wait while someone is coming along it
const STAIR_PATIENCE := 45.0      # s waiting for a stair before giving up on the step
const LEVEL_TOLERANCE := NpcGeometry.LEVEL_TOLERANCE
const USE_TIMEOUT := 6.0
const POSE_SECONDS := 0.5         # hold after changing pose (bodies without a transition clip)
const SEAT_CLEAR := 0.03          # m the calves keep in front of a seat's front edge
const PORTAL_LOOKAHEAD := 2.5     # react to closed doors this far ahead
const AVOID_AHEAD := 1.6          # steer around characters (player, NPCs) this far ahead
const AVOID_MARGIN := 0.2
const DETOUR_HORIZON := 8.0       # m of path ahead whose waypoints are moved off characters (farther ones as they come up)
const CROWD_CELL := 2.0           # m, cells of the shared character grid
const THROTTLE_NEAR := 10.0       # m: off-screen NPCs nearer the player than this still step every frame
const THROTTLE_FAR := 60.0        # m: beyond this NPCs step every other frame even on screen
## Moving bodies NPCs steer around (the player joins it too).
const CHARACTER_GROUP := "geogen_character"
## Traffic (traffic.gd): wait for moving vehicles, walk around stopped ones.
const VEHICLE_GROUP := "geogen_vehicle"
const VEHICLE_MARGIN := 0.35
const VEHICLE_LOOKAHEAD := 3.0    # s: a vehicle that would cross our next steps this soon has priority
const VEHICLE_PROBE := 2.5        # m of our path checked against vehicles
const VEHICLE_PATIENCE := 3.0     # s waiting at the kerb before stepping out anyway (traffic yields)

var world: WorldLoader
var definition := {}
var needs := {}                   # need -> 0..1
var rng := RandomNumberGenerator.new()
var body: Node3D
var home := PackedVector2Array()  # world XZ polygon
var home_y := 0.0
## Affordances more than this above or below the home's floor belong to another storey.
const FLOOR_TOLERANCE := 1.5
var step_height := 0.3
var clock := 0.0                  # seconds since spawn
var trace_enabled := false
var label: Label3D
## Per-NPC counters for tests: used {id: count}, decisions, failures,
## max_stall (s), outside (s outside home + margin), passes (doors).
var stats := {"used": {}, "decisions": 0, "failures": [], "max_stall": 0.0, "outside": 0.0, "passes": 0}
var _stuck_at = null     # where the last wedged failure happened (see _unstick)
var _wedged := false     # the step failing now made no progress at all (_walk), not e.g. a busy stair

var _stack: Array[Dictionary] = []   # frames: {action, steps, index, started, ctx}
var _option := {}
var _recent := {}                    # option id -> clock when last finished
var _retry_at := {}                  # failed option id -> clock when it may be tried again
var _forced := ""                    # option id the next decision takes (force_option)
var _shape: CollisionShape3D
var _gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity")
var _pose := "stand"
var _pose_yaw := 0.0                 # world heading the current pose was set to (rad)
var animator: GeogenNpcAnimator = null   # the body's skeletal clips, poses and locomotion (null: no body)
## "from>to" (poses) -> {clip, times, positions, rotations}: the clip that moves the body between
## them and its root path, in the frame of where the NPC ends up (the clip's extras `root`).
var _cost_usec := 0                 # script time spent in _physics_process (report: cost_us per frame)
var _cost_frames := 0
# Every character (player and NPCs) as {node, pos, radius, npc}, gathered once a physics frame
# and shared by all NPCs (geogen.11.18): scanning the group and each body's shape per NPC per
# call grew with the square of the crowd.
var _frame_delta := 1.0 / 60.0       # this physics frame's delta (move_and_slide moves one of these)
var _skipped := 0.0                  # s of frames skipped while throttled, made up by the next step
var _phase := 0                      # which frames a throttled NPC steps on (alternates between NPCs)
static var _phases := 0
static var _crowd: Array[Dictionary] = []
static var _crowd_frame := -1
static var _crowd_grid := {}          # Vector2i cell -> Array of _crowd entries
static var _crowd_radius := 0.0       # the largest character radius
# Step state
var _timer := 0.0
var _path := PackedVector3Array()
var _waypoint := 0
var _best := INF
var _stall := 0.0
var _sidestep := 0.0
var _sidestep_dir := 1.0
var _yaw_goal := 0.0
var _detour := -1                 # index in _path of the last detour waypoint
var _vehicle_wait := 0.0          # s spent waiting for traffic at this waypoint
var _assertive := -1              # waypoint we're walking to regardless of moving traffic
var _vehicle_detours := 0         # detours round stopped vehicles on this path
var _signal_wait := 0.0           # s spent on the kerb waiting for the walk signal this time
var _streets := GeogenNpcStreets.new()   # the crossings and pedestrian signals (world set in _ready)
var _was_at := Vector2.ZERO       # where we were a frame ago (plan)
## Off the map during a routine's `away` block (vanished into a building), back when it ends.
var away := false
var _look: GeogenHeadLook = null   # turns the head toward the player (attention.look)
var _playing := false             # a `play:` step holds the body's animation
var _greeted_at := -INF           # clock of the last greeting taken (attention.greet debounce)
var _waved_at := -INF             # clock of the last wave back (attention.greet cooldown)
var _attend_until := -INF         # keep looking at whoever greeted us until then (acknowledge)
const GESTURE_BONES := ["RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand"]
var _yielded_at := -INF
const YIELD_COOLDOWN := 3.0       # s between stepping aside
const YIELD_FAR := 6.0            # m: beyond the yield distance plus this, nobody is walking at us (0.6 s at 10 m/s)
const GOAL_STANDOFF := 0.6        # m in front of a thing to stand when using it for a quest
const KERB_STEP := 0.7            # wait for the walk signal when the road of a crossing is this close ahead (m)
const VANISH_DISTANCE := 12.0     # don't vanish in front of the player: wait until they're this far


## Build an NPC from an exported npc node (its body child moves under the NPC).
static func spawn(world_: WorldLoader, node: Node3D, data: Dictionary) -> GeogenNpc:
    var npc := GeogenNpc.new()
    npc.world = world_
    npc.definition = data
    npc.name = "%s_npc" % node.name
    var xform := node.global_transform
    var yaw := atan2(xform.basis.z.x, xform.basis.z.z)
    node.get_parent().add_child(npc)
    npc.global_transform = Transform3D(Basis(Vector3.UP, yaw), xform.origin)
    npc.body = node.get_node_or_null(str(data.get("body", {}).get("node", "body")))
    if npc.body != null:
        npc.animator = GeogenNpcAnimator.load(npc.body)
        npc.body.reparent(npc, false)
        npc.body.transform = Transform3D.IDENTITY
        npc.animator.attach()
        var look: Dictionary = data.get("attention", {}).get("look", {})
        var skeletons := npc.body.find_children("*", "Skeleton3D", true, false)
        if not look.is_empty() and not skeletons.is_empty():
            npc._look = GeogenHeadLook.new()
            npc._look.name = "HeadLook"
            npc._look.yaw_limit = deg_to_rad(float(look["yaw"]))
            npc._look.pitch_limit = deg_to_rad(float(look["pitch"]))
            npc._look.speed = float(look["speed"])
            skeletons[0].add_child(npc._look)
    var h: Dictionary = data.get("home", {})
    for p in h.get("polygon", []):
        var w := xform * Vector3(p[0], 0.0, p[1])
        npc.home.append(Vector2(w.x, w.z))
    npc.home_y = (xform * Vector3(0, float(h.get("y", 0.0)), 0)).y
    node.set_meta("geogen_npc", npc)
    return npc


func _ready() -> void:
    _phase = _phases
    _phases = (_phases + 1) % 2
    _streets.world = world
    collision_layer = LAYER
    collision_mask = 1 | LAYER
    add_to_group(CHARACTER_GROUP)
    floor_snap_length = step_height
    floor_block_on_wall = true
    _shape = CollisionShape3D.new()
    var cyl := CylinderShape3D.new()   # flat bottom: climbs steps (see Player)
    cyl.radius = float(definition.get("radius", 0.25))
    cyl.height = float(definition.get("height", 1.75))
    cyl.margin = Player.SHAPE_MARGIN
    _shape.shape = cyl
    _shape.position.y = cyl.height / 2.0
    add_child(_shape)
    rng.seed = hash("%s/%d" % [name, int(definition.get("seed", 0))])
    for need in definition.get("needs", {}):
        needs[need] = float(definition["needs"][need].get("initial", 0.5))
    label = Label3D.new()
    label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
    label.position.y = cyl.height + 0.35
    label.font_size = 28
    label.pixel_size = 0.004
    label.no_depth_test = true
    label.visible = false
    add_child(label)


# --- deciding -----------------------------------------------------------

## Every option this NPC could take now, scored (best first).
## The routine blocks active at the world clock's hour.
func routine_blocks() -> Array:
    var out := []
    if world == null or world.clock == null:
        return out
    var h: float = world.clock.hours
    for b in definition.get("routine", []):
        var a := float(b["from"])
        var z := float(b["to"])
        if (a <= z and h >= a and h < z) or (a > z and (h >= a or h < z)):
            out.append(b)
    return out


func _away_block() -> bool:
    return routine_blocks().any(func(b): return b.get("away", false))


## Back from an `away` block: at an entrance if there is one, else somewhere in the NPC's haunts.
func _reappear() -> void:
    var at = null
    var exits := world.exit_points()
    if not exits.is_empty():
        at = exits[rng.randi() % exits.size()]
    if at == null:
        at = _random_home_point()
    if at == null:
        return      # try again next frame (navmesh not ready)
    global_position = at
    away = false
    visible = true
    _shape.set_deferred("disabled", false)
    add_to_group(CHARACTER_GROUP)
    stats["returns"] = int(stats.get("returns", 0)) + 1
    _trace({"event": "return", "pos": _v(global_position)})


func options() -> Array[Dictionary]:
    var result: Array[Dictionary] = []
    var actions: Dictionary = definition.get("actions", {})
    var margin := float(definition.get("home_margin", 1.0))
    var only: Array = definition.get("affordance_tags", [])
    for a in world.affordances:
        if not actions.has(a.get("action", "")) or not is_instance_valid(a.get("node")):
            continue
        if not only.is_empty() and not only.any(func(t): return t in a.get("tags", [])):
            continue
        if clock < float(_retry_at.get(a["id"], -1.0)):
            continue
        if home_distance(a["approach"]) > margin:
            continue
        if absf((a["approach"] as Vector3).y - home_y) > FLOOR_TOLERANCE:
            continue    # another storey above or below the home region
        if world.npc_reserved(a["id"], self) >= int(a.get("slots", 1)):
            continue
        result.append({"id": a["id"], "action": a["action"], "affordance": a,
            "advertises": a.get("advertises", {}), "tags": a.get("tags", []), "at": a["approach"]})
    var activities: Dictionary = definition.get("activities", {})
    for n in activities:
        var act: Dictionary = activities[n]
        result.append({"id": "self/%s" % n, "action": act["action"], "activity": act,
            "advertises": act.get("advertises", {}), "tags": [], "at": global_position})
    var w: Dictionary = definition.get("scoring", {})
    var prefs: Dictionary = definition.get("preferences", {}).duplicate()
    var act_mult := {}
    for b in routine_blocks():      # routines bias the choice by time of day
        for t in b.get("preferences", {}):
            prefs[t] = float(prefs.get(t, 1.0)) * float(b["preferences"][t])
        for n in b.get("activities", {}):
            act_mult["self/%s" % n] = float(act_mult.get("self/%s" % n, 1.0)) * float(b["activities"][n])
    for o in result:
        var utility := 0.0
        for need in o["advertises"]:
            utility += (1.0 - float(needs.get(need, 1.0))) * float(o["advertises"][need])
        for tag in o["tags"]:
            utility *= float(prefs.get(tag, 1.0))
        utility *= float(act_mult.get(o["id"], 1.0))
        var distance := global_position.distance_to(o["at"])
        var recency := 0.0
        if _recent.has(o["id"]):
            recency = exp(-(clock - float(_recent[o["id"]])) / maxf(float(w.get("memory", 120.0)), 1.0))
        o["score"] = utility - float(w.get("distance", 0.02)) * distance \
            - float(w.get("recency", 0.4)) * recency + float(w.get("noise", 0.05)) * rng.randf()
    result.sort_custom(func(x, y): return x["score"] > y["score"])
    return result


## Make the next decision take option ``id`` (an affordance id or "self/<activity>") if it is
## still available then. False if it isn't one of the options now.
func force_option(id: String) -> bool:
    if not options().any(func(o): return o["id"] == id):
        return false
    _forced = id
    return true


## A quest this NPC holds (geogen.15.8): the nearest goal of an open objective of an active quest
## (the manifest's objective `goals`: enter a place, use an interaction, use one until it shows a
## state) that isn't met yet and hasn't just failed, as an option with its action context; {}
## when there's none.
func quest_option() -> Dictionary:
    # (Not before the navmesh has synced: there'd be no path yet, and the goal would wait a retry.)
    if world.rules == null or NavigationServer3D.map_get_iteration_id(get_world_3d().navigation_map) == 0:
        return {}
    var best := {}
    var nearest := INF
    for quest in world.rules.report(WorldLoader.actor_id(self)):
        if quest["status"] != "active":
            continue
        for objective in quest["objectives"]:
            if objective["state"] != "open":
                continue
            for goal in objective.get("goals", []):
                var id := "quest/%s/%s/%s" % [quest["id"], objective["id"], goal["subject"]]
                if clock < float(_retry_at.get(id, -1.0)):
                    continue
                var ctx := _goal_context(goal)
                if ctx.is_empty():
                    continue
                var d := global_position.distance_to(ctx["approach"])
                if d < nearest:
                    nearest = d
                    best = {"id": id, "action": "quest_go" if goal.get("enter", false) else "quest_use",
                        "advertises": {}, "tags": [], "at": ctx["approach"], "quest": quest["id"], "ctx": ctx}
    return best


## Where to go and what to work for one quest goal; {} if the goal is met or its subject isn't here.
func _goal_context(goal: Dictionary) -> Dictionary:
    var subject := str(goal["subject"])
    if goal.get("enter", false):
        var place = world.node_by_uid(subject)
        if not place is Node3D:
            return {}
        var at := NavigationServer3D.map_get_closest_point(get_world_3d().navigation_map, (place as Node3D).global_position)
        return {"anchor": at, "approach": at, "yaw_deg": rad_to_deg(_yaw()), "duration": [0.0, 0.0]}
    var it = world.interactions_by_key.get("%s#%s" % [subject, goal["interaction"]])
    if not it is GeogenInteraction or not is_instance_valid(it) or it.asset == null:
        return {}
    var state := str(goal["state"]) if goal.has("state") else str(it.states.get(it.target, {}).get("next", ""))
    if goal.has("state") and it.state == state and it.target == state:
        return {}                       # done already
    var thing: Node3D = it.targets[0] if not it.targets.is_empty() else it.asset
    var anchor := thing.global_position
    # Things face +Z (a wall switch out of its wall): stand in front of it, on the navmesh.
    var front := NpcGeometry.flat(it.asset.global_basis.z).normalized()
    var approach := NavigationServer3D.map_get_closest_point(get_world_3d().navigation_map, anchor + front * GOAL_STANDOFF)
    var look := NpcGeometry.flat(anchor - approach)
    var yaw := atan2(look.x, look.z) if look.length() > 1e-3 else _yaw()
    return {"anchor": anchor, "approach": approach, "interaction": it, "state": state,
        "yaw_deg": rad_to_deg(yaw), "duration": [0.4, 0.8]}


func _decide() -> void:
    var goal := quest_option() if _forced == "" else {}
    if not goal.is_empty():
        stats["decisions"] += 1
        stats["quest_goals"] = int(stats.get("quest_goals", 0)) + 1
        _option = goal
        _trace({"event": "decide", "chosen": goal["id"], "quest": goal["quest"]})
        _push(goal["action"], goal["ctx"])
        return
    var ranked := options()
    stats["decisions"] += 1
    if ranked.is_empty():
        return
    _option = ranked[0]
    if _forced != "":
        for o in ranked:
            if o["id"] == _forced:
                _option = o
        _forced = ""
    if _option.has("affordance"):
        world.npc_reserve(_option["id"], self)
    var ctx := {}
    var a: Dictionary = _option.get("affordance", {})
    if not a.is_empty():
        ctx = {"anchor": a["position"], "yaw_deg": a["npc_yaw_deg"], "approach": a["approach"],
            "interaction": a.get("interaction"), "duration": a.get("duration"), "depth": a.get("depth")}
    else:
        ctx = {"anchor": global_position, "yaw_deg": rad_to_deg(_yaw()), "approach": global_position,
            "duration": _option["activity"].get("duration")}
    _trace({"event": "decide", "chosen": _option["id"], "top": ranked.slice(0, 3).map(
        func(o): return {"id": o["id"], "score": snappedf(o["score"], 0.001)}),
        "needs": _rounded_needs()})
    _push(_option["action"], ctx)


func _push(action_name: String, ctx: Dictionary) -> void:
    var action: Dictionary = definition["actions"][action_name]
    if ctx.get("duration") == null:
        ctx["duration"] = action.get("duration", [3.0, 6.0])
    _stack.append({"action": action_name, "steps": action["steps"], "index": 0, "started": false, "ctx": ctx})


func _finish_option(ok: bool, reason := "") -> void:
    if _option.is_empty():
        return
    var id: String = _option["id"]
    _recent[id] = clock
    if ok:
        for need in _option["advertises"]:
            if needs.has(need):
                needs[need] = clampf(needs[need] + float(_option["advertises"][need]), 0.0, 1.0)
        stats["used"][id] = int(stats["used"].get(id, 0)) + 1
        _trace({"event": "done", "option": id, "needs": _rounded_needs()})
    else:
        stats["failures"].append("%s: %s" % [id, reason])
        _retry_at[id] = clock + float(definition.get("scoring", {}).get("retry", 30.0))
        _trace({"event": "fail", "option": id, "reason": reason})
    world.npc_release(id, self)
    _option = {}


# --- running --------------------------------------------------------------

func _physics_process(delta: float) -> void:
    var started := Time.get_ticks_usec()
    _frame_delta = delta
    # Out of sight (or far off), step every other frame over both frames' time: half the
    # cost, and nobody sees the coarser motion. NPCs alternate, so each frame takes half.
    if _throttled() and (Engine.get_physics_frames() + _phase) % 2 == 0:
        _skipped += delta
        stats["throttled"] = int(stats.get("throttled", 0)) + 1
    else:
        _step(delta + _skipped)
        _skipped = 0.0
    _cost_usec += Time.get_ticks_usec() - started
    _cost_frames += 1


func _step(delta: float) -> void:
    clock += delta
    if away:
        if not _away_block():
            _reappear()
        return
    if animator != null:
        animator.advance(delta)
        animator.locomote(Vector2(velocity.x, velocity.z).length(), _pose == "stand", _playing)
    _attend(delta)
    for need in needs:
        needs[need] = maxf(0.0, needs[need] - float(definition["needs"][need].get("decay", 0.0)) * delta)
    if home_distance(global_position) > float(definition.get("home_margin", 1.0)) + 0.5:
        stats["outside"] += delta
    if label.visible:
        label.text = "%s\n%s" % [_status(), _needs_text()]
    if _stack.is_empty():
        var map := get_world_3d().navigation_map
        if NavigationServer3D.map_get_iteration_id(map) > 0 \
                and NavigationServer3D.map_get_closest_point(map, global_position).distance_to(global_position) < 1.0:
            if _away_block() and definition.get("actions", {}).has("leave"):
                _option = {"id": "self/leave", "action": "leave", "advertises": {}}
                _trace({"event": "decide", "chosen": "self/leave", "reason": "routine"})
                _push("leave", {"anchor": global_position, "yaw_deg": rad_to_deg(_yaw()),
                    "approach": global_position, "duration": [0.0, 0.0]})
            else:
                _decide()
        return
    var frame: Dictionary = _stack.back()
    if frame["index"] >= frame["steps"].size():
        _stack.pop_back()
        if _stack.is_empty():
            _finish_option(true)
        else:
            _stack.back()["started"] = false   # re-run the step a door interrupted
        return
    var step: Dictionary = frame["steps"][frame["index"]]
    if step.has("if") and not definition.get("flags", {}).get(step["if"], false):
        frame["index"] += 1
        return
    if not frame["started"]:
        frame["started"] = true
        var err := _begin(step, frame["ctx"])
        _trace({"event": "step", "action": frame["action"], "step": step, "pos": _v(global_position)})
        if err != "":
            _fail("%s %s: %s" % [frame["action"], JSON.stringify(step), err])
            return
    var result := _tick(step, frame["ctx"], delta)
    if result == "done":
        frame["index"] += 1
        frame["started"] = false
    elif result != "running":
        _fail("%s %s: %s" % [frame["action"], JSON.stringify(step), result])


func _fail(reason: String) -> void:
    _stack.clear()
    _stair = null
    _playing = false
    if _pose != "stand":
        _set_pose("stand", {"anchor": global_position, "yaw_deg": rad_to_deg(_yaw())}, "approach")
    velocity = Vector3.ZERO
    _unstick(_wedged)
    _wedged = false
    _finish_option(false, reason)


## Wedged (stuck twice running at the same spot, e.g. against a post at the top of some steps):
## step back onto the nearest point of the navmesh instead of failing every plan from there.
func _unstick(wedged: bool) -> void:
    if not wedged:
        _stuck_at = null
        return
    if _stuck_at != null and (_stuck_at as Vector3).distance_to(global_position) < 0.3:
        var map := get_world_3d().navigation_map
        var q := NavigationServer3D.map_get_closest_point(map, global_position)
        if q.distance_to(global_position) > 0.05:
            global_position = q + Vector3.UP * 0.02
        else:
            global_position += -global_basis.z * 0.3   # on the navmesh but snagged: back off
        stats["unstuck"] = int(stats.get("unstuck", 0)) + 1
        _trace({"event": "unstuck", "pos": _v(global_position)})
        _stuck_at = null
        return
    _stuck_at = global_position


## Start a step; returns "" or why it can't run.
func _begin(step: Dictionary, ctx: Dictionary) -> String:
    _timer = 0.0
    if step.has("go_to"):
        var target = _target(step["go_to"], ctx)
        if target == null:
            return "no %s point" % step["go_to"]
        _replans = 0
        return _plan(target)
    if step.has("face"):
        var dir: Vector3
        if step["face"] == "portal" and ctx.has("portal"):
            dir = -ctx["portal"]["normal"] * float(ctx["side"])
        elif step["face"] == "player":
            var who := _player()
            if who == null:
                return ""
            dir = NpcGeometry.flat(who.global_position - global_position)
            if dir.length() < 1e-3:
                return ""
        else:
            dir = Basis(Vector3.UP, deg_to_rad(float(ctx["yaw_deg"]))) * Vector3.BACK
        _yaw_goal = atan2(dir.x, dir.z)
        return ""
    if step.has("pose"):
        if not definition.get("body", {}).get("poses", {}).has(step["pose"]):
            return "body has no pose '%s'" % step["pose"]
        _set_pose(step["pose"], ctx, str(step.get("at", "anchor")))
        return ""
    if step.has("wait"):
        var r = ctx["duration"] if str(step["wait"]) == "duration" else step["wait"]
        _timer = rng.randf_range(float(r[0]), float(r[1]))
        return ""
    if step.has("vanish"):
        return ""
    if step.has("play"):
        # A skeletal body plays the clip once (then back to standing); others just pause.
        var clip := str(step["play"])
        velocity = Vector3.ZERO
        _timer = 1.0
        if animator != null and animator.has(clip) and _pose == "stand":
            _timer = animator.play_once(clip)
            _playing = true
        return ""
    if step.has("use"):
        var it: GeogenInteraction = ctx.get("interaction")
        if it == null:
            return ""   # nothing to operate: the step is a no-op
        var state := _state_name(str(step["use"]), ctx)
        if not it.states.has(state):
            return ""
        if it.target != state:
            if not it.use(WorldLoader.actor_id(self)):
                return "can't use %s (locked?)" % it.name
        return ""
    return "unknown step"


func _tick(step: Dictionary, ctx: Dictionary, delta: float) -> String:
    if step.has("go_to"):
        return _walk(delta)
    if step.has("face"):
        var diff := wrapf(_yaw_goal - _yaw(), -PI, PI)
        var turn := deg_to_rad(float(definition.get("turn_speed", 360.0))) * delta
        rotation.y += clampf(diff, -turn, turn)
        return "done" if absf(diff) <= turn else "running"
    if step.has("pose"):
        if _moving_pose():
            return "running"              # the transition clip sets the pace
        _timer += delta
        return "done" if _timer >= POSE_SECONDS else "running"
    if step.has("wait"):
        _timer -= delta
        return "done" if _timer <= 0.0 else "running"
    if step.has("play"):
        _timer -= delta
        if _timer > 0.0:
            return "running"
        _playing = false
        return "done"
    if step.has("vanish"):
        if not _away_block():
            return "done"                 # the time to be away is over: stay
        for other in get_tree().get_nodes_in_group(CHARACTER_GROUP):
            if other is Node3D and not other is GeogenNpc \
                    and (other as Node3D).global_position.distance_to(global_position) < VANISH_DISTANCE:
                return "running"          # not in front of the player
        away = true
        visible = false
        velocity = Vector3.ZERO
        _shape.set_deferred("disabled", true)
        remove_from_group(CHARACTER_GROUP)
        stats["aways"] = int(stats.get("aways", 0)) + 1
        _trace({"event": "away", "pos": _v(global_position)})
        return "done"
    if step.has("use"):
        var it: GeogenInteraction = ctx.get("interaction")
        var state := _state_name(str(step["use"]), ctx)
        if it == null or not it.states.has(state) or (it.state == state and it.target == state):
            return "done"
        _timer += delta
        if it.target != state and it.can_use():
            it.use(WorldLoader.actor_id(self))   # someone turned it back: press again
        return "running" if _timer < USE_TIMEOUT else "%s never reached '%s'" % [it.name, state]
    return "unknown step"


## A portal's generic open/closed names map to its interaction's states; `goal` is the state a
## quest goal wants (ctx.state).
func _state_name(state: String, ctx: Dictionary) -> String:
    if state == "goal":
        return str(ctx.get("state", ""))
    if ctx.has("portal"):
        return str(ctx["portal"].get(state, state))
    return state


func _target(kind: String, ctx: Dictionary):
    match kind:
        "aside":
            return ctx.get("aside")
        "approach":
            return ctx["approach"]
        "anchor":
            return ctx["anchor"]
        "near", "far":
            if not ctx.has("portal"):
                return null
            var p: Dictionary = ctx["portal"]
            var side := float(ctx["side"]) * (1.0 if kind == "near" else -1.0)
            return p["center"] + p["normal"] * side * (p["depth"] / 2.0 + p["clearance"])
        "random":
            return _random_home_point()
        "exit":
            # The nearest way out (a building entrance); none: stay put and vanish from here.
            var best = null
            for p in world.exit_points():
                if best == null or global_position.distance_to(p) < global_position.distance_to(best):
                    best = p
            return best if best != null else global_position
    return null


func _random_home_point():
    var tags: Array = definition.get("wander", {}).get("tags", [])
    if not tags.is_empty():
        return _random_surface_point(tags)
    if home.size() < 3:
        return null
    var lo := home[0]
    var hi := home[0]
    for p in home:
        lo = Vector2(minf(lo.x, p.x), minf(lo.y, p.y))
        hi = Vector2(maxf(hi.x, p.x), maxf(hi.y, p.y))
    for _i in 20:
        var p := Vector2(rng.randf_range(lo.x, hi.x), rng.randf_range(lo.y, hi.y))
        if Geometry2D.is_point_in_polygon(p, home):
            var map := get_world_3d().navigation_map
            var q := NavigationServer3D.map_get_closest_point(map, Vector3(p.x, home_y, p.y))
            if Vector2(q.x - p.x, q.z - p.y).length() < 0.5 and absf(q.y - home_y) < 0.5:
                return q
    return null


var _wander_tris := PackedVector3Array()   # up-facing triangles (world) of the wander surfaces
var _wander_area := PackedFloat32Array()   # cumulative areas


## A random point (area-weighted) on the up-facing faces of nodes tagged ``tags``, inside the
## home region, snapped onto the navmesh.
func _random_surface_point(tags: Array):
    if _wander_tris.is_empty():
        var total := 0.0
        for tag in tags:
            for node in get_tree().get_nodes_in_group(tag):
                for mi: MeshInstance3D in ([node] if node is MeshInstance3D else []) + node.find_children("*", "MeshInstance3D", true, false):
                    if mi.mesh == null:
                        continue
                    var faces := mi.mesh.get_faces()
                    var xf := mi.global_transform
                    for k in range(0, faces.size(), 3):
                        var a := xf * faces[k]
                        var b := xf * faces[k + 1]
                        var c := xf * faces[k + 2]
                        var n := (b - a).cross(c - a)
                        if absf(n.y) < 0.9 * n.length() or n.length() < 1e-4:
                            continue
                        if home.size() >= 3 and not Geometry2D.is_point_in_polygon(Vector2((a.x + b.x + c.x) / 3.0, (a.z + b.z + c.z) / 3.0), home):
                            continue
                        total += n.length() / 2.0
                        _wander_tris.append_array([a, b, c])
                        _wander_area.append(total)
    if _wander_area.is_empty():
        return null
    var map := get_world_3d().navigation_map
    for _i in 10:
        var pick := rng.randf() * _wander_area[-1]
        var k := _wander_area.bsearch(pick)
        var u := rng.randf()
        var v := rng.randf()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        var a := _wander_tris[3 * k]
        var p := a + (_wander_tris[3 * k + 1] - a) * u + (_wander_tris[3 * k + 2] - a) * v
        var q := NavigationServer3D.map_get_closest_point(map, p)
        if Vector2(q.x - p.x, q.z - p.z).length() < 0.6 and absf(q.y - p.y) < 0.5:
            return q
    return null


func _plan(target: Vector3) -> String:
    var map := get_world_3d().navigation_map
    var from := NavigationServer3D.map_get_closest_point(map, global_position)
    var to := NavigationServer3D.map_get_closest_point(map, target)
    _path = GeogenWalker.find_path(map, from, to)
    _stairs = []
    _stair = null
    _stair_wait = 0.0
    _waypoint = 1
    _detour = -1
    _assertive = -1
    _vehicle_wait = 0.0
    _vehicle_detours = 0
    _signal_wait = 0.0
    _best = INF
    _stall = 0.0
    _sidestep = 0.0
    if _path.is_empty():
        return "no path"
    if Vector2(to.x - target.x, to.z - target.z).length() > 0.8:
        return "target off the navmesh (%.1f m)" % Vector2(to.x - target.x, to.z - target.z).length()
    var over := _streets.crosswalk_route(map, _path, from, to)
    if not over.is_empty():
        _path = over
        stats["crosswalks"] = int(stats.get("crosswalks", 0)) + 1
    # A navigation link on the way (a spiral stair) is two points, a straight line through the
    # stair: walk its way round instead. Last, so the reroute above sees the navmesh's own path.
    _path = GeogenWalker.along_links(_path, get_tree(), _stairs)
    return ""


# Stairs are walked in single file. A spiral stair's navigation link is one body wide, and two
# NPCs meeting on it push each other off the treads: whoever gets to an end while someone is
# coming the other way waits there, a little back from the end so the other can step off.
var _stairs := []                  # [{link, first, last, down}]: the path's stretches along links
var _stair: NavigationLink3D       # the link we're on (others read it: see _stair_free)
var _stair_down := false
var _stair_wait := 0.0
var _replans := 0


## May we walk on? False while the path is about to take a stair someone is coming along the
## other way. Taking one marks us as on it, until the path leaves it or is dropped.
func _stair_free() -> bool:
    var ahead := 0.0
    var at := global_position
    for i in range(_waypoint, _path.size()):
        ahead += at.distance_to(_path[i])
        at = _path[i]
        if ahead > STAIR_HOLD:
            break
        for span in _stairs:
            if i < span["first"] or i > span["last"] or not is_instance_valid(span["link"]):
                continue
            if _stair == span["link"]:
                return true
            for other in get_tree().get_nodes_in_group(CHARACTER_GROUP):
                if other is GeogenNpc and other != self and other._stair == span["link"] \
                        and other._stair_down != span["down"]:
                    return false
            _stair = span["link"]
            _stair_down = span["down"]
            return true
    _stair = null      # no stair in the next few steps: off the one we were on, if any
    return true


## Is waypoint ``index`` on a stair's way round (``after_first``: and not where it starts)?
## There the line is the only way: nothing is moved aside or walked around.
func _on_stair(index: int, after_first := false) -> bool:
    for span in _stairs:
        if index >= span["first"] + (1 if after_first else 0) and index <= span["last"]:
            return true
    return false


## A waypoint added in front of ``index`` moves the stair stretches after it along.
func _insert_waypoint(index: int, point: Vector3) -> void:
    _path.insert(index, point)
    for span in _stairs:
        if span["first"] >= index:
            span["first"] += 1
        if span["last"] >= index:
            span["last"] += 1


## On the kerb with a signalled crossing's road in the next step: wait for walk. Not once we've
## stepped off (someone caught by the flashing don't-walk finishes crossing), and whatever the
## path: the rule is where we stand and where we're headed. True while waiting.
func _wait_for_walk(delta: float) -> bool:
    var here := Vector2(global_position.x, global_position.z)
    var to := NpcGeometry.flat(_path[_waypoint] - global_position)
    var step := to.normalized() * minf(to.length(), KERB_STEP)
    var id := _streets.crossing_from_kerb(here, here + Vector2(step.x, step.z))
    var state := "" if id == "" else _streets.pedestrian_state(id)
    if state == "walk" or state == "":
        _signal_wait = 0.0
        return false
    if _signal_wait == 0.0:
        stats["signal_waits"] = int(stats.get("signal_waits", 0)) + 1
        _trace({"event": "signal_wait", "crossing": id, "state": state})
    _signal_wait += delta
    stats["signal_wait"] = maxf(float(stats.get("signal_wait", 0.0)), _signal_wait)
    # Stand facing the crossing.
    if to.length() > 0.05:
        var turn := deg_to_rad(float(definition.get("turn_speed", 360.0))) * delta
        rotation.y += clampf(wrapf(atan2(to.x, to.z) - _yaw(), -PI, PI), -turn, turn)
    velocity.x = 0.0
    velocity.z = 0.0
    _move(delta)
    return true


## Count our steps off a kerb onto a signalled crossing: `signal_crossings`, and
## `signal_violations` for the ones made while it showed a steady don't-walk.
func _watch_crossing() -> void:
    var here := Vector2(global_position.x, global_position.z)
    # (A frame's step: not where we were before sitting down, or before being put back on the navmesh.)
    var id := _streets.crossing_from_kerb(_was_at, here) if _was_at.distance_to(here) < 0.5 else ""
    _was_at = here
    if id == "":
        return
    var state := _streets.pedestrian_state(id)
    var key := "signal_violations" if state == "dont" else "signal_crossings"
    stats[key] = int(stats.get(key, 0)) + 1
    _trace({"event": "signal_step", "crossing": id, "state": state, "position": _v(global_position)})


func _walk(delta: float) -> String:
    if _waypoint >= _path.size():
        velocity.x = 0.0
        velocity.z = 0.0
        _stair = null
        _move(delta)
        return "done"
    # A closed door ahead: go through it first.
    var door := _portal_ahead()
    if not door.is_empty():
        stats["passes"] += 1
        _push("pass", door)
        return "running"
    _watch_crossing()
    if _wait_for_walk(delta):
        return "running"
    if _waypoint != _assertive and not _clear_of_vehicles():
        # Traffic coming: wait at the kerb (not stalling), but not forever: after a while step
        # out anyway until the next waypoint; vehicles yield to people in their way.
        _vehicle_wait += delta
        if _vehicle_wait > VEHICLE_PATIENCE:
            _assertive = _waypoint
        velocity.x = 0.0
        velocity.z = 0.0
        _move(delta)
        return "running"
    _vehicle_wait = 0.0
    if not _stair_free():
        # Someone is coming along the stair: wait for them (not stalling), but not all day.
        _stair_wait += delta
        velocity.x = 0.0
        velocity.z = 0.0
        _move(delta)
        return "stuck: stair in use" if _stair_wait > STAIR_PATIENCE else "running"
    _stair_wait = 0.0
    if not _detour_around_characters():
        # Someone is standing on our destination: wait (the stall timer gives up eventually).
        velocity.x = 0.0
        velocity.z = 0.0
        _move(delta)
        _stall += delta
        return "stuck: destination occupied" if _stall > FAIL_SECONDS else "running"
    var target := _path[_waypoint]
    var to := target - global_position
    var flat := Vector2(to.x, to.z)
    var speed := float(definition.get("speed", 1.2))
    # (On its own level: a waypoint a turn of a spiral stair above isn't reached from below.)
    if flat.length() < maxf(WAYPOINT_RADIUS, speed * delta) and absf(to.y) < LEVEL_TOLERANCE:
        _waypoint += 1
        _best = INF
        _stall = 0.0
        return "running"
    var dir := Vector3(to.x, 0, to.z).normalized()
    if not _on_stair(_waypoint, true):
        dir = _avoid(dir)
    var heading := atan2(dir.x, dir.z)
    var turn := deg_to_rad(float(definition.get("turn_speed", 360.0))) * delta
    rotation.y += clampf(wrapf(heading - _yaw(), -PI, PI), -turn, turn)
    if _sidestep > 0.0:
        _sidestep -= delta
        dir = (dir + dir.cross(Vector3.UP) * _sidestep_dir * 1.5).normalized()
    velocity.x = dir.x * speed
    velocity.z = dir.z * speed
    _move(delta)
    if flat.length() < _best - 0.03:
        _best = flat.length()
        _stall = 0.0
    else:
        _stall += delta
        stats["max_stall"] = maxf(stats["max_stall"], _stall)
        if _stall > FAIL_SECONDS:
            _wedged = true
            return "stuck at %s" % str(_v(global_position))
        if _stall > REPLAN_SECONDS and _waypoint <= _detour and _replans < MAX_REPLANS:
            # Held up on a way round someone (who may have gone since): forget it, plan afresh.
            _replans += 1
            var err := _plan(_path[-1])
            return "running" if err == "" else err
        if _stall > STALL_SECONDS and _sidestep <= 0.0 and fmod(_stall, STALL_SECONDS * 2.0) < delta * 1.5:
            _sidestep = SIDESTEP_SECONDS
            _sidestep_dir = -_sidestep_dir
    return "running"


## Characters aren't in the navmesh, so walk around them: path waypoints
## inside a character's reach move out to its edge, and a segment that
## crosses one gets an arc of waypoints around it, on the shorter side whose
## points are all on the navmesh (which is already shrunk by the agent
## radius, so a point on it has room for us). Returns false while a
## character is standing on the destination itself (wait for them).
func _detour_around_characters() -> bool:
    var radius := float(definition.get("radius", 0.25))
    var map := get_world_3d().navigation_map
    var last := _path.size() - 1
    _characters()
    var widest := radius + _crowd_radius + AVOID_MARGIN
    var goal := Vector2(_path[last].x, _path[last].z)
    for entry in _characters_in(goal - Vector2(widest, widest), goal + Vector2(widest, widest)):
        var c: Vector3 = entry["pos"]
        if entry["node"] != self and NpcGeometry.flat(_path[last] - c).length() < radius + float(entry["radius"]) + AVOID_MARGIN \
                and NpcGeometry.same_level(_path[last], c):
            return false   # someone is standing where we're going
    if _waypoint <= _detour:
        return true
    # The waypoints of the next DETOUR_HORIZON m (not the destination); farther ones are
    # seen to as they come up. Only characters within reach of these, or of the way to the
    # next one, matter.
    var end := _waypoint
    var ahead := 0.0
    var at := global_position
    var lo := Vector2(at.x, at.z)
    var hi := lo
    while end < last and ahead < DETOUR_HORIZON:
        ahead += at.distance_to(_path[end])
        at = _path[end]
        lo = lo.min(Vector2(at.x, at.z))
        hi = hi.max(Vector2(at.x, at.z))
        end += 1
    for entry in _characters_in(lo - Vector2(widest, widest), hi + Vector2(widest, widest)):
        if entry["node"] == self:
            continue
        var c: Vector3 = entry["pos"]
        var reach := radius + float(entry["radius"]) + AVOID_MARGIN
        # Waypoints inside its reach move out to the edge.
        for i in range(_waypoint, end):
            if NpcGeometry.flat(_path[i] - c).length() < reach and NpcGeometry.same_level(_path[i], c) and not _on_stair(i):
                var out = NpcGeometry.edge_point(c, _path[i], reach + 0.05, map)
                if out != null:
                    _path[i] = out
        var a := global_position
        var b := _path[_waypoint]
        var seg := NpcGeometry.flat(b - a)
        if seg.length() < 0.05 or not NpcGeometry.same_level(a, c) or _on_stair(_waypoint, true):
            continue
        var dir := seg.normalized()
        var rel := NpcGeometry.flat(c - a)
        var along := rel.dot(dir)
        if along <= 0.0 or along > minf(seg.length() + reach, AVOID_AHEAD + reach):
            continue
        if (rel - dir * along).length() >= reach:
            continue
        var arc := NpcGeometry.arc_around(c, a, b, reach + 0.05, map)
        if arc.is_empty():
            continue   # boxed in: press on and let the stall timer decide
        for k in arc.size():
            _insert_waypoint(_waypoint + k, arc[k])
        _detour = _waypoint + arc.size() - 1
        _best = INF
        return true    # the rest of the path has moved along: the others next frame
    return true


## Vehicles aren't in the navmesh either. A moving one whose next few seconds of travel crosses
## the next steps of our path makes us wait (unless we're already in its way: then press on and
## it yields); a stopped one in the way gets a detour round its nearer end.
func _clear_of_vehicles() -> bool:
    var radius := float(definition.get("radius", 0.25))
    var a := global_position
    var seg := NpcGeometry.flat(_path[_waypoint] - a)
    if seg.length() < 0.05:
        return true
    var probe := a + seg.normalized() * minf(seg.length(), VEHICLE_PROBE)
    var map := get_world_3d().navigation_map
    for car in get_tree().get_nodes_in_group(VEHICLE_GROUP):
        var box: Dictionary = car.get_meta("geogen_vehicle_box", {})
        if box.is_empty():
            continue
        var xf: Transform3D = car.global_transform
        if absf(xf.origin.y - a.y) > 2.0:
            continue
        # Too far off for its box, swept ahead, to reach the probed stretch of our path.
        var speed_now := float(car.get_meta("geogen_speed", 0.0))
        if NpcGeometry.flat(xf.origin - a).length() > absf(float(box["offset"])) + float(box["half"]) + float(box["half_w"]) \
                + speed_now * VEHICLE_LOOKAHEAD + VEHICLE_PROBE + radius + VEHICLE_MARGIN:
            continue
        var fwd := NpcGeometry.flat(xf.basis.z).normalized()
        var right := Vector3(fwd.z, 0.0, -fwd.x)
        var center: Vector3 = xf.origin + fwd * float(box["offset"])
        var hl := float(box["half"]) + radius + VEHICLE_MARGIN
        var hw := float(box["half_w"]) + radius + VEHICLE_MARGIN
        var speed := float(car.get_meta("geogen_speed", 0.0))
        if speed > 0.5:
            var reach := speed * VEHICLE_LOOKAHEAD
            var swept_c := center + fwd * reach / 2.0
            if NpcGeometry.in_box(a, swept_c, fwd, right, hl + reach / 2.0, hw):
                continue          # already in its way: keep going, it yields to us
            if NpcGeometry.segment_hits_box(a, probe, swept_c, fwd, right, hl + reach / 2.0, hw):
                return false
        elif _waypoint > _detour and _vehicle_detours < 2 and NpcGeometry.segment_hits_box(a, probe, center, fwd, right, hl, hw):
            # (Also when pressed up against it: the corners on our side lead round it.)
            _vehicle_detours += 1
            var detour := NpcGeometry.around_box(a, _path[_waypoint], center, fwd, right, hl + 0.05, hw + 0.05, map)
            if not detour.is_empty():
                for k in detour.size():
                    _insert_waypoint(_waypoint + k, detour[k])
                _detour = _waypoint + detour.size() - 1
                _best = INF
    return true


## Bend ``dir`` around characters standing in the way (static geometry is the navmesh's job).
func _avoid(dir: Vector3) -> Vector3:
    var radius := float(definition.get("radius", 0.25))
    var steer := Vector3.ZERO
    var p := Vector2(global_position.x, global_position.z)
    _characters()
    var near := AVOID_AHEAD + radius + _crowd_radius + AVOID_MARGIN
    for entry in _characters_in(p - Vector2(near, near), p + Vector2(near, near)):
        if entry["node"] == self:
            continue
        var c: Vector3 = entry["pos"]
        if not NpcGeometry.same_level(c, global_position):
            continue
        var rel := NpcGeometry.flat(c - global_position)
        var along := rel.dot(dir)
        if along <= 0.0 or along > AVOID_AHEAD:
            continue
        var lateral := rel - dir * along
        var reach := radius + float(entry["radius"]) + AVOID_MARGIN
        if lateral.length() >= reach:
            continue
        # Pass on the side it's already off to (the right if dead ahead).
        var away := -lateral.normalized() if lateral.length() > 0.02 else dir.cross(Vector3.UP)
        steer += away * (1.0 - lateral.length() / reach) * (1.0 - along / AVOID_AHEAD) * 3.0
    return (dir + steer).normalized() if steer != Vector3.ZERO else dir


func _move(delta: float) -> void:
    if is_on_floor():
        velocity.y = 0.0
        if Player.step_up(self, Vector3(velocity.x, 0, velocity.z) * delta, step_height):
            return
    else:
        velocity.y -= _gravity * delta
    var scale := delta / _frame_delta
    if scale > 1.01:      # a throttled step covers skipped frames too
        velocity *= scale
        move_and_slide()
        velocity /= scale
    else:
        move_and_slide()


## Whether to step every other frame: off-screen and not near the player, or far off. Not
## while greeting, in a transition or on a stair, where timing and single file matter.
func _throttled() -> bool:
    if away or _moving_pose() or _stair != null or clock < _attend_until:
        return false
    var camera := get_viewport().get_camera_3d()
    var who := _player()
    if camera == null or who == null:
        return false
    var d := NpcGeometry.flat(who.global_position - global_position).length()
    if d > THROTTLE_FAR:
        return true
    return d > THROTTLE_NEAR and not camera.is_position_in_frustum(global_position + Vector3.UP * 0.9)


## The first closed portal the remaining path crosses within PORTAL_LOOKAHEAD, as a pass context.
func _portal_ahead() -> Dictionary:
    if _stack.back()["action"] == "pass":
        return {}
    var walked := 0.0
    var a := global_position
    for i in range(_waypoint, _path.size()):
        var b := _path[i]
        for p in world.portals:
            var it: GeogenInteraction = p["interaction"]
            if not is_instance_valid(it) or (it.state == p["open"] and it.target == p["open"]):
                continue
            var da: float = (a - p["center"]).dot(p["normal"])
            var db: float = (b - p["center"]).dot(p["normal"])
            if signf(da) == signf(db):
                continue
            var hit := a.lerp(b, da / (da - db))
            var lateral: Vector3 = hit - p["center"]
            lateral -= p["normal"] * lateral.dot(p["normal"])
            lateral.y = 0.0
            if lateral.length() > p["width"] / 2.0 + 0.4 or absf(hit.y - (p["center"].y - p["height"] / 2.0)) > 1.0:
                continue
            if walked + a.distance_to(hit) > PORTAL_LOOKAHEAD:
                return {}
            var side := signf((global_position - p["center"]).dot(p["normal"]))
            return {"portal": p, "side": side if side != 0.0 else 1.0, "interaction": it,
                "anchor": p["center"], "yaw_deg": 0.0, "duration": [0.0, 0.0]}
        walked += a.distance_to(b)
        if walked > PORTAL_LOOKAHEAD:
            break
        a = b
    return {}


# --- body -----------------------------------------------------------------

## A keyframed move between poses is playing (sit_down, stand_up, ...).
func _moving_pose() -> bool:
    return animator != null and animator.moving()


## World heading (rad). The NPC sits under its placement's parent, which may be turned
## (a cottage facing south), while affordance and path headings are in world space.
func _yaw() -> float:
    return rotation.y + _parent_yaw()


func _parent_yaw() -> float:
    var parent := get_parent_node_3d()
    if parent == null:
        return 0.0
    var z := parent.global_basis.z
    return atan2(z.x, z.z)


## Put the body in a pose: sit/lie at the anchor (collider off), stand where it stands.
func _set_pose(pose_name: String, ctx: Dictionary, at: String) -> void:
    var pose: Dictionary = definition["body"]["poses"][pose_name]
    var yaw := deg_to_rad(float(ctx["yaw_deg"]))
    var place: Vector3 = ctx["anchor"] if at == "anchor" else ctx["approach"]
    var was := _pose
    var before := body.global_transform if body != null else Transform3D.IDENTITY
    if pose_name == "stand":
        var map := get_world_3d().navigation_map
        place = NavigationServer3D.map_get_closest_point(map, place)
    global_position = place
    rotation = Vector3(0, yaw - _parent_yaw(), 0)
    _pose_yaw = yaw
    velocity = Vector3.ZERO
    _shape.set_deferred("disabled", pose_name != "stand")
    _pose = pose_name
    if at == "anchor" and pose.has("reach") and ctx.get("depth") != null:
        # Deep seats: sit forward until the shins clear the front edge.
        var forward := Basis(Vector3.UP, yaw) * Vector3.BACK
        global_position += forward * maxf(0.0, float(ctx["depth"]) + SEAT_CLEAR - float(pose["reach"]))
    if body != null:
        var o: Array = pose["offset"]
        var r: Array = pose["rotation"]
        var s: Array = pose["scale"]
        var basis := Basis.from_euler(Vector3(deg_to_rad(r[0]), deg_to_rad(r[1]), deg_to_rad(r[2])))
        var target := Transform3D(basis.scaled_local(Vector3(s[0], s[1], s[2])), Vector3(o[0], o[1], o[2]))
        if animator.change_pose(was, pose_name, global_transform.affine_inverse() * before, target, self):
            _timer = POSE_SECONDS             # no extra hold once it has played
            stats["transitions"] = int(stats.get("transitions", 0)) + 1


# --- attention (the player) ---------------------------------------------------

## The player (the one character that isn't an NPC), or null.
func _player() -> Node3D:
    for entry in _characters():
        if not entry["npc"] and is_instance_valid(entry["node"]):
            return entry["node"]
    return null


## The characters this physics frame (see ``_crowd``): positions as the frame began.
func _characters() -> Array[Dictionary]:
    var frame := Engine.get_physics_frames()
    if frame != _crowd_frame:
        _crowd_frame = frame
        _crowd = []
        _crowd_grid = {}
        _crowd_radius = 0.0
        for other in get_tree().get_nodes_in_group(CHARACTER_GROUP):
            if other is CharacterBody3D:
                var entry := {"node": other, "pos": other.global_position,
                    "radius": NpcGeometry.radius_of(other), "npc": other is GeogenNpc}
                _crowd.append(entry)
                _crowd_radius = maxf(_crowd_radius, entry["radius"])
                var cell := Vector2i(floori(entry["pos"].x / CROWD_CELL), floori(entry["pos"].z / CROWD_CELL))
                if not _crowd_grid.has(cell):
                    _crowd_grid[cell] = []
                _crowd_grid[cell].append(entry)
    return _crowd


## The characters this physics frame whose cells overlap the plan rectangle ``lo``..``hi`` (x, z).
func _characters_in(lo: Vector2, hi: Vector2) -> Array:
    _characters()
    var out := []
    for cx in range(floori(lo.x / CROWD_CELL), floori(hi.x / CROWD_CELL) + 1):
        for cz in range(floori(lo.y / CROWD_CELL), floori(hi.y / CROWD_CELL) + 1):
            var cell = _crowd_grid.get(Vector2i(cx, cz))
            if cell != null:
                out.append_array(cell)
    return out


## Look at the player when they're near and ahead; step out of their way when standing in it.
func _attend(delta: float) -> void:
    var attention: Dictionary = definition.get("attention", {})
    var who := _player()
    if _look != null:
        var look: Dictionary = attention.get("look", {})
        var want := 0.0
        # (Out of range and not just greeted: nothing to look at; skip the camera lookups.)
        if who != null and _pose in ["stand", "sit"] and (clock < _attend_until
                or NpcGeometry.flat(who.global_position - global_position).length() < float(look["range"]) + 1.0):
            var eye: Vector3 = who.global_position + Vector3(0, 1.6, 0)
            var cam = who.get("camera")
            if cam is Camera3D:
                eye = (cam as Camera3D).global_position
            var to := NpcGeometry.flat(eye - global_position)
            var forward := Basis(Vector3.UP, _yaw()) * Vector3.BACK
            # Just greeted: look their way wherever they stand (the head's own limits clamp it).
            if clock < _attend_until or (to.length() < float(look["range"])
                    and rad_to_deg(forward.angle_to(to)) < float(look["cone"]) / 2.0):
                want = 1.0
                _look.target = eye
        if _look.gesturing() and (_pose != "sit" or _moving_pose()):
            _look.stop_gesture()               # they got up: the arm goes with the body's clip
        _look.want = want
        if want > 0.0:
            stats["looking"] = float(stats.get("looking", 0.0)) + delta
    var yield_spec: Dictionary = attention.get("yield", {})
    if who == null or yield_spec.is_empty() or _pose != "stand" or _moving_pose() \
            or clock - _yielded_at < YIELD_COOLDOWN:
        return
    if NpcGeometry.flat(global_position - who.global_position).length() > float(yield_spec["distance"]) + YIELD_FAR:
        return                             # too far off to be walking at us, even sprinting
    if not _stack.is_empty() and (_stack.back()["action"] in ["greet", yield_spec["action"]]
            or _stack.back()["steps"][mini(_stack.back()["index"], _stack.back()["steps"].size() - 1)].has("go_to")):
        return                             # walking already avoids the player; reactions run out
    var body_velocity = who.get("velocity")
    if not body_velocity is Vector3:
        return
    var v := NpcGeometry.flat(body_velocity)
    var offset := NpcGeometry.flat(global_position - who.global_position)
    if v.length() < 0.3 or offset.length() > float(yield_spec["distance"]) + v.length() * 0.6 \
            or v.normalized().dot(offset.normalized()) < 0.7:
        return                             # not walking at us
    var side := signf(v.cross(offset).y)
    if side == 0.0:
        side = 1.0 if rng.randf() < 0.5 else -1.0
    var perp := Vector3(v.z, 0, -v.x).normalized() * -side
    var aside := global_position + perp * float(yield_spec["step"])
    var map := get_world_3d().navigation_map
    aside = NavigationServer3D.map_get_closest_point(map, aside)
    _yielded_at = clock
    stats["yields"] = int(stats.get("yields", 0)) + 1
    _trace({"event": "yield", "to": _v(aside)})
    _react(str(yield_spec["action"]), {"aside": aside})


## Whether the player can greet this NPC now (their E prompt): it has a greeting and is here,
## whatever it is doing (standing, walking, seated, lying down).
func can_greet() -> bool:
    return not definition.get("attention", {}).get("greet", {}).is_empty() and not away


## Seated or lying: out of its standing collider.
func posed() -> bool:
    return _pose != "stand"


## How far along an aim line (from `from`, unit `forward`) it passes this NPC's posed body, or
## INF if it misses. A seated or lying NPC has no collider for the player's aim ray to hit, so
## the E prompt asks this instead: the body is the line from its hips to its head (the root
## to head height without a skeleton), `radius` thick.
func aim_distance(from: Vector3, forward: Vector3) -> float:
    var radius := float(definition.get("radius", 0.25))
    var line := _body_line()
    var best := INF
    for i in 5:
        var p: Vector3 = line[0].lerp(line[1], i / 4.0)
        var along := (p - from).dot(forward)
        if along > 0.0 and (p - from - forward * along).length() <= radius:
            best = minf(best, along)
    return best


## The posed body's head: what the player must have a clear line to, to greet it.
func head_point() -> Vector3:
    return _body_line()[1]


## The body as a line, hips to head (root to head height without a skeleton).
func _body_line() -> Array:
    var radius := float(definition.get("radius", 0.25))
    var a := global_position + Vector3.UP * 0.3
    var b := global_position + Vector3.UP * (float(definition.get("height", 1.75)) - radius)
    var skeletons := body.find_children("*", "Skeleton3D", true, false) if body != null else []
    if not skeletons.is_empty():
        var skeleton := skeletons[0] as Skeleton3D
        var hips := skeleton.find_bone("Hips")
        var head := skeleton.find_bone("Head")
        if hips >= 0 and head >= 0:
            a = skeleton.global_transform * skeleton.get_bone_global_pose(hips).origin
            b = skeleton.global_transform * skeleton.get_bone_global_pose(head).origin
    return [a, b]


func greet_prompt() -> String:
    return str(definition.get("attention", {}).get("greet", {}).get("prompt", "Say hello"))


## An actor (the player by default) greets this NPC. Rules hear it as the NPC's `greeted`
## signal, with `by` as the actor, in any pose and in the middle of any action: only a second
## press within `debounce` s is ignored. What the NPC does about it fits what it is doing:
## standing, it runs its greet action (turns to them and waves) and then carries on; seated, it
## stays put, looks at them for `acknowledge` s and waves from the chair; lying down or in
## the middle of sitting down, it only takes the greeting. It waves once per `cooldown` s.
func greet(by := "player") -> bool:
    var spec: Dictionary = definition.get("attention", {}).get("greet", {})
    if not can_greet() or clock - _greeted_at < float(spec.get("debounce", 0.5)):
        return false
    _greeted_at = clock
    stats["greets"] = int(stats.get("greets", 0)) + 1
    _attend_until = clock + float(spec.get("acknowledge", 3.0))
    var response := "none"
    var action := str(spec["action"])
    var reacting: bool = not _stack.is_empty() and _stack.back()["action"] == action
    if clock - _waved_at >= float(spec["cooldown"]) and not _moving_pose() and not reacting:
        if _pose == "stand":
            response = "action"
            _react(action, {})
        elif _pose == "sit" and _look != null and animator != null and animator.has("wave") \
                and _look.gesture(animator.clip("wave"), GESTURE_BONES):
            response = "gesture"
    if response != "none":
        _waved_at = clock
        stats["waves"] = int(stats.get("waves", 0)) + 1
    _trace({"event": "greet", "pos": _v(global_position), "pose": _pose, "response": response})
    var signal_name := str(spec.get("signal", ""))
    if signal_name != "":
        greeted.emit(signal_name, by)
    return true


## Run a reaction on top of whatever the NPC is doing; the interrupted step starts again after.
func _react(action_name: String, extra: Dictionary) -> void:
    if not _stack.is_empty():
        _stack.back()["started"] = false
    velocity = Vector3.ZERO
    _path = PackedVector3Array()
    _stairs = []
    _stair = null
    var ctx := {"anchor": global_position, "yaw_deg": rad_to_deg(_yaw()), "approach": global_position}
    ctx.merge(extra)
    _push(action_name, ctx)


# --- helpers ------------------------------------------------------------------

## 0 inside the home polygon, else metres to its edge (XZ).
func home_distance(p: Vector3) -> float:
    if home.size() < 3:
        return 0.0
    var q := Vector2(p.x, p.z)
    if Geometry2D.is_point_in_polygon(q, home):
        return 0.0
    var best := INF
    for i in home.size():
        var c := Geometry2D.get_closest_point_to_segment(q, home[i], home[(i + 1) % home.size()])
        best = minf(best, c.distance_to(q))
    return best


## Current action and step, e.g. "sit (armchair): wait".
func _status() -> String:
    if _stack.is_empty():
        return "deciding"
    var frame: Dictionary = _stack.back()
    var step: Dictionary = frame["steps"][mini(frame["index"], frame["steps"].size() - 1)]
    var verb := ""
    for k in step:
        if k != "if" and k != "at":
            verb = "%s %s" % [k, step[k]]
    return "%s: %s" % [_option.get("id", frame["action"]), verb]


func _needs_text() -> String:
    var parts := []
    for need in needs:
        parts.append("%s %d%%" % [need, roundi(needs[need] * 100)])
    return "  ".join(parts)


func _rounded_needs() -> Dictionary:
    var out := {}
    for need in needs:
        out[need] = snappedf(needs[need], 0.01)
    return out


func _trace(event: Dictionary) -> void:
    event["t"] = snappedf(clock, 0.01)
    event["npc"] = String(name)
    traced.emit(event)
    if trace_enabled:
        print("npc: %s" % JSON.stringify(event))


static func _v(p: Vector3) -> Array:
    return [snappedf(p.x, 0.01), snappedf(p.y, 0.01), snappedf(p.z, 0.01)]


## Summary for tests and --simulate.
func _quests() -> Dictionary:
    var out := {}
    if world != null and world.rules != null:
        for quest in world.rules.report(WorldLoader.actor_id(self)):
            if quest["status"] in ["active", "complete", "failed"]:
                out[quest["id"]] = quest["status"]
    return out


func report() -> Dictionary:
    var p := global_position
    return {"npc": String(name), "used": stats["used"], "distinct": stats["used"].size(),
        "decisions": stats["decisions"], "failures": stats["failures"], "max_stall": snappedf(stats["max_stall"], 0.01),
        "unstuck": int(stats.get("unstuck", 0)), "throttled": int(stats.get("throttled", 0)),
        "outside": snappedf(stats["outside"], 0.01), "passes": stats["passes"], "pose": _pose, "clip": animator.current() if animator != null else "",
        "position": _v(p), "needs": _rounded_needs(), "doing": _status(), "away": away,
        "aways": int(stats.get("aways", 0)), "returns": int(stats.get("returns", 0)),
        "crosswalks": int(stats.get("crosswalks", 0)), "greets": int(stats.get("greets", 0)),
        # Signalled crossings: waits on the kerb for walk (and the longest, s), steps onto one, and
        # steps onto one showing a steady don't-walk.
        "signal_waits": int(stats.get("signal_waits", 0)), "signal_wait": snappedf(float(stats.get("signal_wait", 0.0)), 0.1),
        "signal_crossings": int(stats.get("signal_crossings", 0)),
        "signal_violations": int(stats.get("signal_violations", 0)),
        "waves": int(stats.get("waves", 0)),
        # Quests it holds ({id: status}) and the quest goals it has set off to work on.
        "quests": _quests(), "quest_goals": int(stats.get("quest_goals", 0)),
        "yields": int(stats.get("yields", 0)), "transitions": int(stats.get("transitions", 0)),
        "cost_us": snappedf(float(_cost_usec) / maxi(_cost_frames, 1), 0.1), "looking": snappedf(float(stats.get("looking", 0.0)), 0.1),
        # World heading (deg) and how far it is from the pose's (sitting: the seat's facing).
        "yaw": snappedf(rad_to_deg(wrapf(_yaw(), -PI, PI)), 0.1),
        "pose_yaw_error": snappedf(rad_to_deg(absf(wrapf(_yaw() - _pose_yaw, -PI, PI))), 0.1)}
