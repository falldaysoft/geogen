class_name GeogenPlaytest
extends Node
## Headless enterability check (``--playtest``): from the spawn point, every
## room volume and every interaction target must be reachable on the navmesh,
## and a scripted bot must be able to walk a sample of those routes with the
## real player body (catching steps, slopes and door widths the 2D checks
## miss). Doors are opened before the navmesh bake (main sets
## WorldLoader.open_before_bake); opening them is an agent's job.
## Prints ``playtest: {...}`` (JSON) and quits with exit code 1 on failure.

const WAYPOINT_RADIUS := 0.35
const STALL_SECONDS := 0.6      # no progress this long: sidestep
const SIDESTEP_SECONDS := 0.35
const MAX_SIDESTEPS := 3        # then the route counts as stuck

var world: WorldLoader
var player: Player
var start := Vector3.ZERO
var start_yaw := 0.0
## How many routes the bot walks (spread evenly over the reachable rooms).
var walk_count := 6

var _report := {"rooms": 0, "reachable": [], "unreachable": [], "targets": 0, "unreachable_targets": [],
	"walked": [], "stuck": []}
var _routes: Array[Dictionary] = []   # {name, path: PackedVector3Array}
var _route := -1
var _waypoint := 0
var _stuck_time := 0.0
var _sidesteps := 0
var _sidestep_left := 0.0
var _sidestep_dir := 1.0
var _best_distance := INF
var _phase := "sync"
var _frames := 0


func _physics_process(delta: float) -> void:
	match _phase:
		"sync":
			_frames += 1
			var map := get_viewport().get_world_3d().navigation_map
			# Synced once the map answers queries near the spawn.
			var synced := false
			if NavigationServer3D.map_get_iteration_id(map) > 0:
				var near := NavigationServer3D.map_get_closest_point(map, start)
				synced = Vector2(near.x - start.x, near.z - start.z).length() < 1.0
			if synced or _frames > 240:
				_check_paths()
				_phase = "walk"
				_next_route()
		"walk":
			_walk(delta)


func _check_paths() -> void:
	var map := get_viewport().get_world_3d().navigation_map
	var from := NavigationServer3D.map_get_closest_point(map, start)
	var reachable_rooms: Array[Dictionary] = []
	for room in world.rooms:
		_report["rooms"] += 1
		var xform: Transform3D = room["xform"]
		var size: Vector3 = room["size"]
		# A room counts as reachable if a path reaches any of a 3 x 3 grid of
		# floor points in it (its centre may be under a bed).
		var ok := false
		var path := PackedVector3Array()
		for gx in [-0.3, 0.0, 0.3]:
			for gz in [-0.3, 0.0, 0.3]:
				if ok:
					continue
				var floor_point := xform * Vector3(size.x * gx, -size.y / 2.0, size.z * gz)
				var target := NavigationServer3D.map_get_closest_point(map, floor_point)
				var local := xform.affine_inverse() * target
				var inside := absf(local.x) <= size.x / 2.0 and absf(local.z) <= size.z / 2.0 \
					and absf(local.y + size.y / 2.0) <= 0.35
				if not inside:
					continue
				path = NavigationServer3D.map_get_path(map, from, target, true)
				ok = path.size() > 0 and path[-1].distance_to(target) < 0.1
		var label := "%s (%s)" % [room["id"], room["type"]]
		if ok:
			_report["reachable"].append(label)
			reachable_rooms.append({"name": room["id"], "path": path})
		else:
			_report["unreachable"].append(label)
	for it in world.interactions:
		for target_node in it.targets:
			_report["targets"] += 1
			var p := target_node.global_position
			var nearest := NavigationServer3D.map_get_closest_point(map, Vector3(p.x, p.y - 1.0, p.z))
			var path := NavigationServer3D.map_get_path(map, from, nearest, true)
			if path.size() == 0 or Vector2(nearest.x - p.x, nearest.z - p.z).length() > player.spec.reach \
					or absf(nearest.y - p.y) > 2.0:
				_report["unreachable_targets"].append("%s/%s" % [it.name, target_node.name])
	# Walk an even spread of the reachable rooms (farthest-first ordering is overkill here).
	var count := mini(walk_count, reachable_rooms.size())
	for i in count:
		_routes.append(reachable_rooms[int(float(i) * reachable_rooms.size() / count)])


func _next_route() -> void:
	_route += 1
	if _route >= _routes.size():
		_finish()
		return
	player.spawn(start, start_yaw)
	_waypoint = 1
	_stuck_time = 0.0
	_sidesteps = 0
	_sidestep_left = 0.0
	_best_distance = INF


func _walk(delta: float) -> void:
	if _route >= _routes.size():
		return
	var route: Dictionary = _routes[_route]
	var path: PackedVector3Array = route["path"]
	if _waypoint >= path.size():
		var room := world.room_at(player.global_position + Vector3(0, 0.5, 0))
		if room == route["name"]:
			_report["walked"].append(route["name"])
		else:
			_report["stuck"].append("%s (ended in '%s')" % [route["name"], room])
		player.scripted_move = null
		_next_route()
		return
	var target := path[_waypoint]
	var to := target - player.global_position
	var flat := Vector2(to.x, to.z)
	if flat.length() < WAYPOINT_RADIUS:
		_waypoint += 1
		_best_distance = INF
		return
	player.rotation.y = atan2(-to.x, -to.z)
	if _sidestep_left > 0.0:
		# Slide off whatever we're caught on (a jamb, a leaf edge), then retry.
		_sidestep_left -= delta
		player.scripted_move = Vector2(_sidestep_dir, 0.3)
		return
	player.scripted_move = Vector2(0, 1)
	if flat.length() < _best_distance - 0.05:
		_best_distance = flat.length()
		_stuck_time = 0.0
	else:
		_stuck_time += delta
		if _stuck_time > STALL_SECONDS:
			_stuck_time = 0.0
			_sidesteps += 1
			if _sidesteps > MAX_SIDESTEPS:
				var at := player.global_position
				_report["stuck"].append("%s (at %.1f, %.1f, %.1f)" % [route["name"], at.x, at.y, at.z])
				player.scripted_move = null
				_next_route()
				return
			# Step toward the side the path continues on (alternate if unsure).
			var after := path[mini(_waypoint + 1, path.size() - 1)] - player.global_position
			var right := Vector2(-cos(player.rotation.y), sin(player.rotation.y))
			var lateral := Vector2(after.x, after.z).dot(right)
			_sidestep_dir = signf(lateral) if absf(lateral) > 0.05 else (1.0 if _sidesteps % 2 else -1.0)
			_sidestep_left = SIDESTEP_SECONDS


func _finish() -> void:
	_phase = "done"
	var ok: bool = _report["unreachable"].is_empty() and _report["unreachable_targets"].is_empty() \
		and _report["stuck"].is_empty()
	_report["ok"] = ok
	print("playtest: %s" % JSON.stringify(_report))
	get_tree().quit(0 if ok else 1)
