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

const LAYER := 2                  # NPC bodies; the navmesh bakes layer 1 only
const WAYPOINT_RADIUS := 0.15
const STALL_SECONDS := 0.6        # no progress this long: sidestep
const SIDESTEP_SECONDS := 0.4
const FAIL_SECONDS := 4.0         # no progress this long: give up on the step
const USE_TIMEOUT := 6.0
const POSE_SECONDS := 0.5         # hold after changing pose (until there are clips)
const SEAT_CLEAR := 0.03          # m the calves keep in front of a seat's front edge
const POSE_BLEND := 0.4           # s to cross-fade skeletal pose clips (and ease the body into place)
const LOCOMOTION_BLEND := 0.25    # s to cross-fade walk <-> idle
const WALK_THRESHOLD := 0.15      # m/s: slower than this counts as standing still
const PORTAL_LOOKAHEAD := 2.5     # react to closed doors this far ahead
const AVOID_AHEAD := 1.6          # steer around characters (player, NPCs) this far ahead
const AVOID_MARGIN := 0.2
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
var step_height := 0.3
var clock := 0.0                  # seconds since spawn
var trace_enabled := false
var label: Label3D
## Per-NPC counters for tests: used {id: count}, decisions, failures,
## max_stall (s), outside (s outside home + margin), passes (doors).
var stats := {"used": {}, "decisions": 0, "failures": [], "max_stall": 0.0, "outside": 0.0, "passes": 0}

var _stack: Array[Dictionary] = []   # frames: {action, steps, index, started, ctx}
var _option := {}
var _recent := {}                    # option id -> clock when last finished
var _retry_at := {}                  # failed option id -> clock when it may be tried again
var _shape: CollisionShape3D
var _gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity")
var _pose := "stand"
var _anim: AnimationPlayer = null   # the body's skeletal clips (pose_<name>, walk, idle, ...), if any
var _clip_speed := {}               # clip name -> ground speed it was authored for (m/s)
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
## Off the map during a routine's `away` block (vanished into a building), back when it ends.
var away := false
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
		var clips := _body_clips(npc.body)
		for clip in WorldLoader.geogen_extras(npc.body).get("clips", []):
			if clip.has("speed"):
				npc._clip_speed[clip["name"]] = float(clip["speed"])
		npc.body.reparent(npc, false)
		npc.body.transform = Transform3D.IDENTITY
		if not clips.is_empty():
			npc._anim = AnimationPlayer.new()
			npc._anim.name = "Clips"
			var library := AnimationLibrary.new()
			for clip_name in clips:
				library.add_animation(clip_name, clips[clip_name])
			library.add_animation("RESET", _rest_animation(npc.body, clips.values()))
			npc._anim.add_animation_library("", library)
			npc.body.add_child(npc._anim)   # root_node ".." = the body
			npc._play_pose_clip("stand", 0.0)
	var h: Dictionary = data.get("home", {})
	for p in h.get("polygon", []):
		var w := xform * Vector3(p[0], 0.0, p[1])
		npc.home.append(Vector2(w.x, w.z))
	npc.home_y = (xform * Vector3(0, float(h.get("y", 0.0)), 0)).y
	node.set_meta("geogen_npc", npc)
	return npc


func _ready() -> void:
	collision_layer = LAYER
	collision_mask = 1 | LAYER
	add_to_group(CHARACTER_GROUP)
	floor_snap_length = step_height
	floor_block_on_wall = true
	_shape = CollisionShape3D.new()
	var cyl := CylinderShape3D.new()   # flat bottom: climbs steps (see Player)
	cyl.radius = float(definition.get("radius", 0.25))
	cyl.height = float(definition.get("height", 1.75))
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


func _decide() -> void:
	var ranked := options()
	stats["decisions"] += 1
	if ranked.is_empty():
		return
	_option = ranked[0]
	if _option.has("affordance"):
		world.npc_reserve(_option["id"], self)
	var ctx := {}
	var a: Dictionary = _option.get("affordance", {})
	if not a.is_empty():
		ctx = {"anchor": a["position"], "yaw_deg": a["npc_yaw_deg"], "approach": a["approach"],
			"interaction": a.get("interaction"), "duration": a.get("duration"), "depth": a.get("depth")}
	else:
		ctx = {"anchor": global_position, "yaw_deg": rad_to_deg(rotation.y), "approach": global_position,
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
	clock += delta
	if away:
		if not _away_block():
			_reappear()
		return
	_animate_locomotion()
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
				_push("leave", {"anchor": global_position, "yaw_deg": rad_to_deg(rotation.y),
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
	if _pose != "stand":
		_set_pose("stand", {"anchor": global_position, "yaw_deg": rad_to_deg(rotation.y)}, "approach")
	velocity = Vector3.ZERO
	_finish_option(false, reason)


## Start a step; returns "" or why it can't run.
func _begin(step: Dictionary, ctx: Dictionary) -> String:
	_timer = 0.0
	if step.has("go_to"):
		var target = _target(step["go_to"], ctx)
		if target == null:
			return "no %s point" % step["go_to"]
		return _plan(target)
	if step.has("face"):
		var dir: Vector3
		if step["face"] == "portal" and ctx.has("portal"):
			dir = -ctx["portal"]["normal"] * float(ctx["side"])
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
	if step.has("use"):
		var it: GeogenInteraction = ctx.get("interaction")
		if it == null:
			return ""   # nothing to operate: the step is a no-op
		var state := _state_name(str(step["use"]), ctx)
		if not it.states.has(state):
			return ""
		if it.target != state:
			if not it.use():
				return "can't use %s (locked?)" % it.name
		return ""
	return "unknown step"


func _tick(step: Dictionary, ctx: Dictionary, delta: float) -> String:
	if step.has("go_to"):
		return _walk(delta)
	if step.has("face"):
		var diff := wrapf(_yaw_goal - rotation.y, -PI, PI)
		var turn := deg_to_rad(float(definition.get("turn_speed", 360.0))) * delta
		rotation.y += clampf(diff, -turn, turn)
		return "done" if absf(diff) <= turn else "running"
	if step.has("pose"):
		_timer += delta
		return "done" if _timer >= POSE_SECONDS else "running"
	if step.has("wait"):
		_timer -= delta
		return "done" if _timer <= 0.0 else "running"
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
			it.use()   # someone turned it back: press again
		return "running" if _timer < USE_TIMEOUT else "%s never reached '%s'" % [it.name, state]
	return "unknown step"


## A portal's generic open/closed names map to its interaction's states.
func _state_name(state: String, ctx: Dictionary) -> String:
	if ctx.has("portal"):
		return str(ctx["portal"].get(state, state))
	return state


func _target(kind: String, ctx: Dictionary):
	match kind:
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
	_path = NavigationServer3D.map_get_path(map, from, to, true)
	_waypoint = 1
	_detour = -1
	_assertive = -1
	_vehicle_wait = 0.0
	_vehicle_detours = 0
	_best = INF
	_stall = 0.0
	_sidestep = 0.0
	if _path.is_empty():
		return "no path"
	if Vector2(to.x - target.x, to.z - target.z).length() > 0.8:
		return "target off the navmesh (%.1f m)" % Vector2(to.x - target.x, to.z - target.z).length()
	_use_crosswalk(map, from, to)
	return ""


var _crossing_ends := []   # [[end a, end b]] of the painted crossings, on the pavement (world)
var _crossing_bounds := AABB()


## A path that walks along the road (below both its ends, inside the streets) is rerouted over
## the nearest painted crossing when that isn't much longer: to one end, straight across, on.
func _use_crosswalk(map: RID, from: Vector3, to: Vector3) -> void:
	if _crossing_ends.is_empty():
		for r in world.traffic_graph.get("crosswalks", []):
			var lo := Vector2(float(r[0]), float(r[1]))
			var hi := Vector2(float(r[2]), float(r[3]))
			var c := (lo + hi) / 2.0
			var along_x := (hi.x - lo.x) > (hi.y - lo.y)      # the long side runs across the street
			var a := Vector2(lo.x - 0.8, c.y) if along_x else Vector2(c.x, lo.y - 0.8)
			var b := Vector2(hi.x + 0.8, c.y) if along_x else Vector2(c.x, hi.y + 0.8)
			var pa := NavigationServer3D.map_get_closest_point(map, Vector3(a.x, 0.2, a.y))
			var pb := NavigationServer3D.map_get_closest_point(map, Vector3(b.x, 0.2, b.y))
			if Vector2(pa.x - a.x, pa.z - a.y).length() < 1.0 and Vector2(pb.x - b.x, pb.z - b.y).length() < 1.0:
				_crossing_ends.append([pa, pb])
				var box := AABB(Vector3(lo.x, -1.0, lo.y), Vector3(hi.x - lo.x, 2.0, hi.y - lo.y))
				_crossing_bounds = box if _crossing_bounds.size == Vector3.ZERO else _crossing_bounds.merge(box)
		if _crossing_ends.is_empty():
			_crossing_ends = [null]            # none: don't look again
	if _crossing_ends[0] == null or not _on_road(_path):
		return
	var direct := _length(_path)
	var best := PackedVector3Array()
	var best_len := direct * 2.0 + 25.0
	for ends in _crossing_ends:
		for flip in [false, true]:
			var e1: Vector3 = ends[1] if flip else ends[0]
			var e2: Vector3 = ends[0] if flip else ends[1]
			if from.distance_to(e1) > 60.0 and to.distance_to(e2) > 60.0:
				continue
			var first := NavigationServer3D.map_get_path(map, from, e1, true)
			var last := NavigationServer3D.map_get_path(map, e2, to, true)
			if first.is_empty() or last.is_empty() or _on_road(first) or _on_road(last):
				continue
			var total := _length(first) + e1.distance_to(e2) + _length(last)
			if total < best_len:
				best_len = total
				best = first + last
	if not best.is_empty():
		_path = best
		stats["crosswalks"] = int(stats.get("crosswalks", 0)) + 1


## Does the path dip to road level (below both its ends) inside the streets?
func _on_road(path: PackedVector3Array) -> bool:
	if path.size() < 2:
		return false
	var level := minf(path[0].y, path[-1].y) - 0.08
	for p in path:
		if p.y < level and _crossing_bounds.grow(2.0).has_point(Vector3(p.x, 0.0, p.z)):
			return true
	return false


static func _length(path: PackedVector3Array) -> float:
	var total := 0.0
	for k in range(1, path.size()):
		total += path[k].distance_to(path[k - 1])
	return total


func _walk(delta: float) -> String:
	if _waypoint >= _path.size():
		velocity.x = 0.0
		velocity.z = 0.0
		_move(delta)
		return "done"
	# A closed door ahead: go through it first.
	var door := _portal_ahead()
	if not door.is_empty():
		stats["passes"] += 1
		_push("pass", door)
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
	if flat.length() < maxf(WAYPOINT_RADIUS, speed * delta):
		_waypoint += 1
		_best = INF
		_stall = 0.0
		return "running"
	var dir := _avoid(Vector3(to.x, 0, to.z).normalized())
	var heading := atan2(dir.x, dir.z)
	var turn := deg_to_rad(float(definition.get("turn_speed", 360.0))) * delta
	rotation.y += clampf(wrapf(heading - rotation.y, -PI, PI), -turn, turn)
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
			return "stuck at %s" % str(_v(global_position))
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
	for other in get_tree().get_nodes_in_group(CHARACTER_GROUP):
		if other == self or not other is CharacterBody3D:
			continue
		var c: Vector3 = other.global_position
		var reach := radius + _radius_of(other) + AVOID_MARGIN
		if _flat(_path[last] - c).length() < reach:
			return false   # someone is standing where we're going
		if _waypoint <= _detour:
			continue
		# Waypoints inside its reach move out to the edge.
		for i in range(_waypoint, last):
			if _flat(_path[i] - c).length() < reach:
				var out = _edge_point(c, _path[i], reach + 0.05, map)
				if out != null:
					_path[i] = out
		var a := global_position
		var b := _path[_waypoint]
		var seg := _flat(b - a)
		if seg.length() < 0.05:
			continue
		var dir := seg.normalized()
		var rel := _flat(c - a)
		var along := rel.dot(dir)
		if along <= 0.0 or along > minf(seg.length() + reach, AVOID_AHEAD + reach):
			continue
		if (rel - dir * along).length() >= reach:
			continue
		var arc := _arc_around(c, a, b, reach + 0.05, map)
		if arc.is_empty():
			continue   # boxed in: press on and let the stall timer decide
		for k in arc.size():
			_path.insert(_waypoint + k, arc[k])
		_detour = _waypoint + arc.size() - 1
		_best = INF
	return true


## Vehicles aren't in the navmesh either. A moving one whose next few seconds of travel crosses
## the next steps of our path makes us wait (unless we're already in its way: then press on and
## it yields); a stopped one in the way gets a detour round its nearer end.
func _clear_of_vehicles() -> bool:
	var radius := float(definition.get("radius", 0.25))
	var a := global_position
	var seg := _flat(_path[_waypoint] - a)
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
		var fwd := _flat(xf.basis.z).normalized()
		var right := Vector3(fwd.z, 0.0, -fwd.x)
		var center: Vector3 = xf.origin + fwd * float(box["offset"])
		var hl := float(box["half"]) + radius + VEHICLE_MARGIN
		var hw := float(box["half_w"]) + radius + VEHICLE_MARGIN
		var speed := float(car.get_meta("geogen_speed", 0.0))
		if speed > 0.5:
			var reach := speed * VEHICLE_LOOKAHEAD
			var swept_c := center + fwd * reach / 2.0
			if _in_box(a, swept_c, fwd, right, hl + reach / 2.0, hw):
				continue          # already in its way: keep going, it yields to us
			if _segment_hits_box(a, probe, swept_c, fwd, right, hl + reach / 2.0, hw):
				return false
		elif _waypoint > _detour and _vehicle_detours < 2 and _segment_hits_box(a, probe, center, fwd, right, hl, hw):
			# (Also when pressed up against it: the corners on our side lead round it.)
			_vehicle_detours += 1
			var detour := _around_box(a, _path[_waypoint], center, fwd, right, hl + 0.05, hw + 0.05, map)
			if not detour.is_empty():
				for k in detour.size():
					_path.insert(_waypoint + k, detour[k])
				_detour = _waypoint + detour.size() - 1
				_best = INF
	return true


static func _in_box(p: Vector3, c: Vector3, fwd: Vector3, right: Vector3, hl: float, hw: float) -> bool:
	var d := _flat(p - c)
	return absf(d.dot(fwd)) < hl and absf(d.dot(right)) < hw


## Does segment a-b (plan) cross the box centred on c (half extents hl along fwd, hw along right)?
static func _segment_hits_box(a: Vector3, b: Vector3, c: Vector3, fwd: Vector3, right: Vector3,
		hl: float, hw: float) -> bool:
	var p0 := Vector2(_flat(a - c).dot(fwd), _flat(a - c).dot(right))
	var p1 := Vector2(_flat(b - c).dot(fwd), _flat(b - c).dot(right))
	var d := p1 - p0
	var t0 := 0.0
	var t1 := 1.0
	for axis in 2:
		var lo := -hl if axis == 0 else -hw
		var hi := hl if axis == 0 else hw
		var start := p0[axis]
		var delta := d[axis]
		if absf(delta) < 1e-9:
			if start < lo or start > hi:
				return false
			continue
		var ta := (lo - start) / delta
		var tb := (hi - start) / delta
		t0 = maxf(t0, minf(ta, tb))
		t1 = minf(t1, maxf(ta, tb))
		if t0 > t1:
			return false
	return true


## Two waypoints round the end of the box we're nearer to (in front of a car that has stopped
## for us, behind one queuing), falling back to the other end; [] if neither is on the navmesh.
func _around_box(a: Vector3, _b: Vector3, c: Vector3, fwd: Vector3, right: Vector3, hl: float, hw: float,
		map: RID) -> Array:
	var side := signf(_flat(a - c).dot(right))
	if side == 0.0:
		side = 1.0
	var near := signf(_flat(a - c).dot(fwd))
	if near == 0.0:
		near = 1.0
	for end in [near, -near]:
		var p1 = _on_nav_near(c + fwd * end * hl + right * side * hw, map)
		var p2 = _on_nav_near(c + fwd * end * hl - right * side * hw, map)
		if p1 != null and p2 != null:
			return [p1, p2]
	return []


## The navmesh point nearest ``p`` if it's within 0.4 m (a box corner may sit just off it).
static func _on_nav_near(p: Vector3, map: RID):
	var q := NavigationServer3D.map_get_closest_point(map, p)
	if Vector2(q.x - p.x, q.z - p.z).length() < 0.4 and absf(q.y - p.y) < 0.5:
		return q
	return null


static func _flat(v: Vector3) -> Vector3:
	return Vector3(v.x, 0.0, v.z)


## ``p`` pushed out from ``c`` to distance ``r`` (turning up to 90 deg either way to stay on the navmesh), or null.
func _edge_point(c: Vector3, p: Vector3, r: float, map: RID):
	var base := atan2(p.x - c.x, p.z - c.z)
	for turn in [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5]:
		var q = _on_nav(Vector3(c.x + sin(base + turn) * r, p.y, c.z + cos(base + turn) * r), map)
		if q != null:
			return q
	return null


## Waypoints on a circle of radius ``r`` about ``c`` from ``a``'s bearing to ``b``'s (shorter valid side), or [].
func _arc_around(c: Vector3, a: Vector3, b: Vector3, r: float, map: RID) -> Array:
	var ta := atan2(a.x - c.x, a.z - c.z)
	var tb := atan2(b.x - c.x, b.z - c.z)
	var short := wrapf(tb - ta, -PI, PI)
	for sweep in [short, short - signf(short) * TAU]:
		var steps := maxi(2, ceili(absf(sweep) / deg_to_rad(40.0)))
		var points := []
		for k in range(1, steps + 1):
			var t: float = ta + sweep * k / steps
			var q = _on_nav(Vector3(c.x + sin(t) * r, a.y, c.z + cos(t) * r), map)
			if q == null:
				points = []
				break
			points.append(q)
		if not points.is_empty():
			return points
	return []


## The navmesh point at ``p`` (same floor), or null when ``p`` isn't on it.
static func _on_nav(p: Vector3, map: RID):
	var q := NavigationServer3D.map_get_closest_point(map, p)
	if Vector2(q.x - p.x, q.z - p.z).length() < 0.08 and absf(q.y - p.y) < 0.35:
		return q
	return null


static func _radius_of(body: Node) -> float:
	for child in body.get_children():
		if child is CollisionShape3D and child.shape is CylinderShape3D:
			return child.shape.radius
	return 0.3


## Bend ``dir`` around characters standing in the way (static geometry is the navmesh's job).
func _avoid(dir: Vector3) -> Vector3:
	var radius := float(definition.get("radius", 0.25))
	var steer := Vector3.ZERO
	for other in get_tree().get_nodes_in_group(CHARACTER_GROUP):
		if other == self or not other is CharacterBody3D:
			continue
		var rel: Vector3 = other.global_position - global_position
		rel.y = 0.0
		var along := rel.dot(dir)
		if along <= 0.0 or along > AVOID_AHEAD or absf(rel.y) > 1.0:
			continue
		var lateral := rel - dir * along
		var reach := radius + _radius_of(other) + AVOID_MARGIN
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
	move_and_slide()


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

## RESET: every track the clips animate, keyed at the skeleton's rest. The importer drops
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
			if seen.has(key) or not kind in [Animation.TYPE_ROTATION_3D, Animation.TYPE_POSITION_3D]:
				continue
			seen[key] = true
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


## Put the body in a pose: sit/lie at the anchor (collider off), stand where it stands.
func _set_pose(pose_name: String, ctx: Dictionary, at: String) -> void:
	var pose: Dictionary = definition["body"]["poses"][pose_name]
	var yaw := deg_to_rad(float(ctx["yaw_deg"]))
	var place: Vector3 = ctx["anchor"] if at == "anchor" else ctx["approach"]
	if pose_name == "stand":
		var map := get_world_3d().navigation_map
		place = NavigationServer3D.map_get_closest_point(map, place)
	global_position = place
	rotation = Vector3(0, yaw, 0)
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
		if _anim != null:
			# Skeletal bodies ease into place while the clips cross-fade.
			create_tween().tween_property(body, "transform", target, POSE_BLEND)
		else:
			body.transform = target
	_play_pose_clip(pose_name, POSE_BLEND)


## Skeletal bodies: hold the pose's clip (pose_<name>; standing: walk or idle); capsules have none.
func _play_pose_clip(pose_name: String, blend: float) -> void:
	if _anim == null:
		return
	var clip := "pose_" + pose_name
	if pose_name == "stand" and _anim.has_animation("idle"):
		clip = "idle"
	if _anim.has_animation(clip):
		_anim.speed_scale = 1.0
		_anim.play(clip, blend)


## Standing skeletal bodies walk while moving (playback scaled to the ground speed so the
## feet don't slide) and idle when still.
func _animate_locomotion() -> void:
	if _anim == null or _pose != "stand" or not _anim.has_animation("walk"):
		return
	var speed := Vector2(velocity.x, velocity.z).length()
	var clip := "walk" if speed > WALK_THRESHOLD else ("idle" if _anim.has_animation("idle") else "pose_stand")
	if clip == "walk":
		_anim.speed_scale = speed / float(_clip_speed.get("walk", 1.2))
	else:
		_anim.speed_scale = 1.0
	if _anim.current_animation != clip:
		_anim.play(clip, LOCOMOTION_BLEND)


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
func report() -> Dictionary:
	var p := global_position
	return {"npc": String(name), "used": stats["used"], "distinct": stats["used"].size(),
		"decisions": stats["decisions"], "failures": stats["failures"], "max_stall": snappedf(stats["max_stall"], 0.01),
		"outside": snappedf(stats["outside"], 0.01), "passes": stats["passes"], "pose": _pose, "clip": String(_anim.current_animation) if _anim != null else "",
		"position": _v(p), "needs": _rounded_needs(), "doing": _status(), "away": away,
		"aways": int(stats.get("aways", 0)), "returns": int(stats.get("returns", 0)),
		"crosswalks": int(stats.get("crosswalks", 0))}
