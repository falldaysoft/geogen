class_name GeogenTraffic
extends Node3D
## Vehicles driving the lane graph (manifest "traffic", see geogen/traffic.py).
##
## A generic interpreter, like npc.gd: everything comes from the export.
## - The graph: directed lanes resampled every `step` metres, their successors
##   (connectors through intersections), conflict zones and crosswalk ranges.
## - A traffic node (extras.geogen.fleet: driving parameters, turn weights,
##   who to yield to) whose children are the starting vehicles, each with
##   extras.geogen.driving {lane, s, factor} and extras.geogen.vehicle.
##
## A vehicle is (lane, s, v). Its axles sit on the curve half a wheelbase
## either side of s, so it turns like a car. Each physics step:
## - Car following: the Intelligent Driver Model against the vehicle ahead on
##   its lane or the lane it will take next, with the lane's (curvature-capped)
##   speed limit, slowing in time for a slower connector.
## - All-way stops: every vehicle stops at the stop line (its lane's end), waits
##   `stop_wait`, then claims its connector once no vehicle is on or has claimed
##   a conflicting connector and nobody who arrived earlier is waiting for one
##   (first come, first served: nobody waits forever).
## - Yielding: people (the player, NPCs) in the corridor ahead, or on a crosswalk
##   the vehicle is about to cross, are obstacles it stops for.
## Vehicles are kinematic: AnimatableBody3D boxes that block the player and are
## left out of the navmesh; they never push anything.

const CHARACTER_GROUP := "geogen_character"
## Moving vehicles, for NPCs to wait for or walk around (meta geogen_vehicle_box, geogen_speed).
const VEHICLE_GROUP := "geogen_vehicle"
const OVERLAP_CHECK := 0.25       # s between vehicle overlap checks
const STOP_SPEED := 0.3           # m/s: counts as stopped at the line
const PERSON_RADIUS := 0.4
const PERSON_MARGIN := 0.9        # extra space kept in front of a person (m)

var world: WorldLoader
var fleet := {}
var driving := {}
var lanes: Array[Dictionary] = []     # see _load_graph
var lane_index := {}                  # id -> index
var vehicles: Array[Dictionary] = []
var rng := RandomNumberGenerator.new()
var clock := 0.0
var trace_enabled := false
## Counters for tests and --simulate reports.
var stats := {"distance": 0.0, "claims": 0, "yields": 0, "overlaps": 0, "max_wait": 0.0, "turns": {},
	"min_person_gap": INF, "min_player_gap": INF, "exits": 0}

var _templates: Array[Node3D] = []   # detached copies of the starting vehicles (for respawns)
var _overlap_timer := 0.0


## Build the controller for a traffic node; its vehicle children move under the controller.
static func spawn(world_: WorldLoader, node: Node3D, data: Dictionary, graph: Dictionary,
		offset := Vector3.ZERO) -> GeogenTraffic:
	var t := GeogenTraffic.new()
	t.world = world_
	t.fleet = data
	t.driving = data.get("driving", {})
	t.name = "%s_traffic" % node.name
	t.rng.seed = int(data.get("seed", 0))
	world_.add_child(t)
	t._load_graph(graph, offset)
	for child in node.get_children():
		if child is Node3D and WorldLoader.geogen_extras(child).get("driving") is Dictionary:
			var template := child.duplicate() as Node3D
			_strip_bodies(template)
			t._templates.append(template)
			t._adopt(child as Node3D)
	return t


func _exit_tree() -> void:
	for template in _templates:
		template.free()
	_templates.clear()


func trace(event: Dictionary) -> void:
	if trace_enabled:
		print("traffic: %s" % JSON.stringify(event))


# --- graph ---------------------------------------------------------------------------------------

func _load_graph(graph: Dictionary, offset: Vector3) -> void:
	for ln in graph.get("lanes", []):
		var pts := PackedVector3Array()
		for p in ln["points"]:
			pts.append(Vector3(p[0], p[1], p[2]) + offset)
		var length := 0.0
		for k in range(1, pts.size()):
			length += pts[k].distance_to(pts[k - 1])
		var crosswalks: Array = ln.get("crosswalks", [])
		lane_index[ln["id"]] = lanes.size()
		lanes.append({"id": ln["id"], "pts": pts, "len": length, "ds": length / maxf(pts.size() - 1, 1),
			"speed": float(ln["speed"]), "turn": ln.get("turn", ""), "x": ln.get("intersection", ""),
			"exit": ln.get("end", "") == "exit", "succ_ids": ln.get("successors", []), "succ": [],
			"crosswalks": crosswalks, "conflicts": []})
	for lane in lanes:
		for sid in lane["succ_ids"]:
			if lane_index.has(sid):
				lane["succ"].append(lane_index[sid])
	for c in graph.get("conflicts", []):
		if lane_index.has(c["a"]) and lane_index.has(c["b"]):
			var a: int = lane_index[c["a"]]
			var b: int = lane_index[c["b"]]
			lanes[a]["conflicts"].append(b)
			lanes[b]["conflicts"].append(a)


## Point at distance s along lane i (clamped).
func _lane_point(i: int, s: float) -> Vector3:
	var lane: Dictionary = lanes[i]
	var pts: PackedVector3Array = lane["pts"]
	var t := clampf(s / float(lane["ds"]), 0.0, float(pts.size() - 1))
	var k := mini(int(t), pts.size() - 2)
	return pts[k].lerp(pts[k + 1], t - k)


## Point s metres from the start of a vehicle's lane, following its previous or next lane beyond the ends.
func _path_point(v: Dictionary, s: float) -> Vector3:
	var lane: int = v["lane"]
	if s < 0.0 and v["prev"] >= 0:
		return _lane_point(v["prev"], lanes[v["prev"]]["len"] + s)
	if s > lanes[lane]["len"] and v["next"] >= 0:
		return _lane_point(v["next"], s - lanes[lane]["len"])
	return _lane_point(lane, s)


func _choose_next(lane: int) -> int:
	var succ: Array = lanes[lane]["succ"]
	if succ.is_empty():
		return -1
	var turns: Dictionary = fleet.get("turns", {})
	var weights := []
	var total := 0.0
	for s in succ:
		var w := float(turns.get(lanes[s]["turn"], 1.0)) if lanes[s]["turn"] != "" else 1.0
		weights.append(w)
		total += w
	var pick := rng.randf() * total
	for k in succ.size():
		pick -= weights[k]
		if pick <= 0.0:
			return succ[k]
	return succ[-1]


# --- vehicles ------------------------------------------------------------------------------------

static func _strip_bodies(node: Node) -> void:
	for body in node.find_children("*", "CollisionObject3D", true, false):
		body.get_parent().remove_child(body)
		body.free()


func _adopt(node: Node3D) -> Dictionary:
	var g := WorldLoader.geogen_extras(node)
	var d: Dictionary = g.get("driving", {})
	var info: Dictionary = g.get("vehicle", {})
	var xform := node.global_transform
	if node.get_parent() != self:
		if node.get_parent() != null:
			node.reparent(self)
		else:
			add_child(node)
	node.global_transform = xform
	_strip_bodies(node)
	var clearance: Array = info.get("clearance", [4.5, 1.8, 1.5])
	var body := AnimatableBody3D.new()
	body.name = "VehicleBody"
	body.sync_to_physics = false       # moved by its parent vehicle node
	var shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = Vector3(float(clearance[1]), float(clearance[2]) - 0.25, float(clearance[0]))
	shape.shape = box
	shape.position = Vector3(0, float(clearance[2]) / 2.0 + 0.125,
		(float(info.get("front", clearance[0] / 2.0)) + float(info.get("rear", -clearance[0] / 2.0))) / 2.0)
	body.add_child(shape)
	node.add_child(body)
	node.add_to_group(VEHICLE_GROUP)
	node.set_meta("geogen_vehicle_box", {"half": float(clearance[0]) / 2.0, "half_w": float(clearance[1]) / 2.0,
		"offset": shape.position.z})
	node.set_meta("geogen_speed", 0.0)
	var wheels := []
	for w in info.get("wheels", []):
		var wheel := node.find_child(str(w["part"]), true, false) as Node3D
		if wheel != null:
			wheels.append({"node": wheel, "r": maxf(float(w["radius"]), 0.1)})
	var lane: int = lane_index.get(d.get("lane", ""), -1)
	var v := {"node": node, "lane": lane, "prev": -1, "next": -1, "s": float(d.get("s", 0.0)),
		"v": 0.0, "factor": float(d.get("factor", 1.0)), "half": float(clearance[0]) / 2.0,
		"half_w": float(clearance[1]) / 2.0, "base": float(info.get("wheelbase", 2.6)) / 2.0,
		"accel": minf(float(info.get("accel", 2.0)), float(driving.get("accel", 1.8))),
		"decel": minf(float(info.get("decel", 4.0)), float(driving.get("decel", 3.0))),
		"wheels": wheels, "waiting": -1.0, "claimed": false, "moved": 0.0, "idle": 0.0,
		"blocked_by": "", "yielding": false}
	if lane >= 0:
		v["next"] = _choose_next(lane)
		vehicles.append(v)
		_place(v)
	return v


func _place(v: Dictionary) -> void:
	var rear := _path_point(v, v["s"] - v["base"])
	var front := _path_point(v, v["s"] + v["base"])
	var node: Node3D = v["node"]
	var dir := front - rear
	if dir.length_squared() < 1e-8:
		return
	var yaw := atan2(dir.x, dir.z)
	var pitch := -atan2(dir.y, Vector2(dir.x, dir.z).length())
	node.global_transform = Transform3D(Basis.from_euler(Vector3(pitch, yaw, 0.0)), (rear + front) / 2.0)


# --- simulation ----------------------------------------------------------------------------------

func _physics_process(delta: float) -> void:
	clock += delta
	var people := _people()
	for v in vehicles:
		_drive(v, delta, people)
	_overlap_timer += delta
	if _overlap_timer >= OVERLAP_CHECK:
		_overlap_timer = 0.0
		_check_overlaps()


## People to yield to: [position, is the player] pairs.
func _people() -> Array:
	var out := []
	var yields: Array = fleet.get("yield", ["player", "npc"])
	for node in get_tree().get_nodes_in_group(CHARACTER_GROUP):
		if not node is Node3D:
			continue
		var is_npc := node is GeogenNpc
		if (is_npc and "npc" in yields) or (not is_npc and "player" in yields):
			out.append([(node as Node3D).global_position, not is_npc])
	return out


func _drive(v: Dictionary, dt: float, people: Array) -> void:
	var lane: Dictionary = lanes[v["lane"]]
	var remaining: float = lane["len"] - v["s"]
	var gap := INF
	var lead_v := 0.0
	v["blocked_by"] = ""
	# Leaders: the nearest vehicle ahead on this lane, on the lane it takes next, or just
	# onto any other successor (a car that turned off may still be in the way).
	var succ: Array = lane["succ"]
	for o in vehicles:
		if o == v:
			continue
		var d := INF
		if o["lane"] == v["lane"] and o["s"] > v["s"]:
			d = o["s"] - v["s"]
		elif o["lane"] == v["next"] or (o["lane"] in succ and o["s"] < o["half"] + v["half"] + 2.0):
			d = remaining + o["s"]
		if d < INF:
			var g: float = d - o["half"] - v["half"]
			if g < gap:
				gap = g
				lead_v = o["v"]
				v["blocked_by"] = "vehicle"
	# The stop line: lane end before an intersection, until the connector is claimed.
	var nxt: int = v["next"]
	if nxt >= 0 and lanes[nxt]["x"] != "" and not v["claimed"]:
		var to_line: float = remaining - v["half"] - 0.3
		# As a standing "leader" the IDM would stop `gap` short of it: fold that in so the
		# bumper comes to rest at the line.
		var line_gap := to_line + float(driving.get("gap", 2.5)) - 0.2
		if line_gap < gap:
			gap = line_gap
			lead_v = 0.0
			v["blocked_by"] = "stop"
		if to_line < 1.0 and v["v"] < STOP_SPEED:
			if v["waiting"] < 0.0:
				v["waiting"] = clock
			var waited: float = clock - v["waiting"]
			stats["max_wait"] = maxf(stats["max_wait"], waited)
			if waited > float(driving.get("give_up", 25.0)) and lanes[v["lane"]]["succ"].size() > 1:
				# Blocked this long: go another way (keeps its place in the queue).
				var others: Array = lanes[v["lane"]]["succ"].filter(func(s): return s != nxt)
				v["next"] = others[rng.randi() % others.size()]
				v["waiting"] = clock - float(driving.get("stop_wait", 0.8))
				trace({"t": snappedf(clock, 0.01), "replan": String(v["node"].name), "to": lanes[v["next"]]["id"]})
				nxt = v["next"]
			if waited >= float(driving.get("stop_wait", 0.8)) and _can_claim(v):
				v["claimed"] = true
				v["waiting"] = -1.0
				stats["claims"] += 1
				trace({"t": snappedf(clock, 0.01), "claim": lanes[nxt]["id"], "vehicle": String(v["node"].name)})
	elif nxt < 0 and not lane["exit"]:
		var to_end: float = remaining - v["half"]
		if to_end < gap:
			gap = to_end
			lead_v = 0.0
	# People in the way (or on a crosswalk just ahead).
	var person_gap := _person_gap(v, people)
	if person_gap < gap:
		gap = person_gap
		lead_v = 0.0
		v["blocked_by"] = "person"
		if not v["yielding"]:
			v["yielding"] = true
			stats["yields"] += 1
	else:
		v["yielding"] = false
	# Intelligent Driver Model.
	var v0: float = lane["speed"] * v["factor"]
	if nxt >= 0 and lanes[nxt]["speed"] < v0:
		# Slow in time for a tighter connector.
		v0 = minf(v0, sqrt(pow(lanes[nxt]["speed"] * v["factor"], 2) + 2.0 * v["decel"] * maxf(remaining - v["half"], 0.0)))
	var a_max: float = v["accel"]
	var b: float = v["decel"]
	var speed: float = v["v"]
	var s_star := float(driving.get("gap", 2.5)) + speed * float(driving.get("headway", 1.3)) \
		+ speed * (speed - lead_v) / (2.0 * sqrt(a_max * b))
	var accel := a_max * (1.0 - pow(speed / maxf(v0, 0.1), 4))
	if gap < INF:
		accel -= a_max * pow(maxf(s_star, 0.0) / maxf(gap, 0.05), 2)
	accel = clampf(accel, -9.0, a_max)
	speed = maxf(speed + accel * dt, 0.0)
	if gap < 0.05:
		speed = 0.0
	var step := speed * dt
	v["v"] = speed
	v["s"] += step
	v["moved"] += step
	stats["distance"] += step
	v["idle"] = v["idle"] + dt if speed < 0.1 else 0.0
	# Onto the next lane.
	while v["s"] > lanes[v["lane"]]["len"]:
		var old: int = v["lane"]
		if v["next"] < 0:
			if lanes[old]["exit"]:
				_respawn(v)
				return
			v["s"] = lanes[old]["len"]
			v["v"] = 0.0
			break
		v["s"] -= lanes[old]["len"]
		v["prev"] = old
		v["lane"] = v["next"]
		if lanes[old]["x"] != "":
			v["claimed"] = false        # left the intersection
		if lanes[v["lane"]]["turn"] != "":
			var turn: String = lanes[v["lane"]]["turn"]
			stats["turns"][turn] = int(stats["turns"].get(turn, 0)) + 1
		v["next"] = _choose_next(v["lane"])
	_place(v)
	(v["node"] as Node3D).set_meta("geogen_speed", speed)
	for w in v["wheels"]:
		(w["node"] as Node3D).rotate(Vector3.RIGHT, step / float(w["r"]))


func _can_claim(v: Dictionary) -> bool:
	var target: int = v["next"]
	var conflicts: Array = lanes[target]["conflicts"]
	# Don't block the box: the lane after the connector needs room for the whole vehicle.
	var exits: Array = lanes[target]["succ"]
	var need: float = 2.0 * v["half"] + float(driving.get("gap", 2.5)) + 1.0
	for o in vehicles:
		if o == v:
			continue
		if o["lane"] in exits and o["s"] - o["half"] < need:
			return false
		if (o["lane"] == target or (o["claimed"] and o["next"] == target)):
			return false
		# Someone on (or committed to) a conflicting movement.
		if o["lane"] in conflicts:
			return false
		if o["claimed"] and o["next"] in conflicts:
			return false
		# Someone who got to their stop line first, waiting for a conflicting movement.
		if o["waiting"] >= 0.0 and o["waiting"] < v["waiting"] and o["next"] in conflicts:
			return false
	return true


func _person_gap(v: Dictionary, people: Array) -> float:
	if people.is_empty():
		return INF
	var node: Node3D = v["node"]
	var look := float(driving.get("look_ahead", 14.0))
	var best := INF
	for person in people:
		var p: Vector3 = person[0]
		if p.distance_squared_to(node.global_position) > pow(look + 6.0, 2):
			continue
		# Walk the path ahead of the front bumper in half-metre steps.
		var along := 0.0
		while along <= look:
			var s: float = v["s"] + v["half"] + along
			var q := _path_point(v, s)
			var width: float = v["half_w"] + PERSON_RADIUS + 0.3
			if _on_crosswalk(v, s):
				width += 1.0          # people about to step onto it (not the whole sidewalk)
			if Vector2(q.x - p.x, q.z - p.z).length() < width and absf(q.y - p.y) < 2.0:
				var gap := along - PERSON_MARGIN
				stats["min_person_gap"] = minf(stats["min_person_gap"], maxf(along, 0.0))
				if person[1]:
					stats["min_player_gap"] = minf(stats["min_player_gap"], maxf(along, 0.0))
				best = minf(best, gap)
				break
			along += 0.5
	return best


func _on_crosswalk(v: Dictionary, s: float) -> bool:
	var lane: int = v["lane"]
	if s > lanes[lane]["len"] and v["next"] >= 0:
		s -= lanes[lane]["len"]
		lane = v["next"]
	for r in lanes[lane]["crosswalks"]:
		if s >= float(r[0]) - 1.0 and s <= float(r[1]) + 1.0:
			return true
	return false


## A vehicle that drove off an open route comes back at a random route start.
func _respawn(v: Dictionary) -> void:
	stats["exits"] += 1
	var starts := []
	for i in lanes.size():
		if lanes[i]["turn"] == "" and lanes[i]["x"] == "" and not _is_successor(i):
			starts.append(i)
	if starts.is_empty():
		starts = range(lanes.size())
	for attempt in 8:
		var lane: int = starts[rng.randi() % starts.size()]
		if _clear_at(lane, v["half"] + 6.0, v):
			v["lane"] = lane
			v["prev"] = -1
			v["s"] = v["half"] + 0.5
			v["v"] = lanes[lane]["speed"] * 0.5
			v["claimed"] = false
			v["waiting"] = -1.0
			v["next"] = _choose_next(lane)
			_place(v)
			trace({"t": snappedf(clock, 0.01), "respawn": String(v["node"].name), "lane": lanes[lane]["id"]})
			return
	v["s"] = lanes[v["lane"]]["len"]      # nowhere free: wait at the end


func _is_successor(i: int) -> bool:
	for lane in lanes:
		if i in lane["succ"]:
			return true
	return false


func _clear_at(lane: int, room: float, me: Dictionary) -> bool:
	for o in vehicles:
		if o != me and o["lane"] == lane and o["s"] < room + o["half"]:
			return false
	return true


func _check_overlaps() -> void:
	for i in vehicles.size():
		for j in range(i + 1, vehicles.size()):
			if _boxes_overlap(vehicles[i], vehicles[j]):
				stats["overlaps"] += 1
				trace({"t": snappedf(clock, 0.01), "overlap": [_where(vehicles[i]), _where(vehicles[j])]})


func _where(v: Dictionary) -> Array:
	return [String(v["node"].name), lanes[v["lane"]]["id"], snappedf(v["s"], 0.1), snappedf(v["v"], 0.1),
		lanes[v["next"]]["id"] if v["next"] >= 0 else "",
		[snappedf(v["node"].global_position.x, 0.1), snappedf(v["node"].global_position.z, 0.1)],
		snappedf(rad_to_deg(v["node"].rotation.y), 1), snappedf(v["half"] * 2, 0.1)]


static func _corners(v: Dictionary) -> PackedVector2Array:
	var node: Node3D = v["node"]
	var b := node.global_transform.basis
	var f := Vector2(b.z.x, b.z.z).normalized() * (float(v["half"]) - 0.05)
	var r := Vector2(b.x.x, b.x.z).normalized() * (float(v["half_w"]) - 0.05)
	var c := Vector2(node.global_position.x, node.global_position.z)
	return PackedVector2Array([c + f + r, c + f - r, c - f - r, c - f + r])


static func _boxes_overlap(a: Dictionary, b: Dictionary) -> bool:
	var pa := _corners(a)
	var pb := _corners(b)
	for poly in [pa, pb]:
		for k in 4:
			var edge: Vector2 = poly[(k + 1) % 4] - poly[k]
			var axis := Vector2(-edge.y, edge.x)
			var a0 := INF
			var a1 := -INF
			var b0 := INF
			var b1 := -INF
			for p in pa:
				var d := axis.dot(p)
				a0 = minf(a0, d)
				a1 = maxf(a1, d)
			for p in pb:
				var d := axis.dot(p)
				b0 = minf(b0, d)
				b1 = maxf(b1, d)
			if a1 < b0 or b1 < a0:
				return false
	return true


## Summary for tests: counts, distances, the longest any vehicle has stood still.
func report() -> Dictionary:
	var min_moved := INF
	var max_idle := 0.0
	for v in vehicles:
		min_moved = minf(min_moved, v["moved"])
		max_idle = maxf(max_idle, v["idle"])
	var gap = stats["min_person_gap"]
	var player_gap = stats["min_player_gap"]
	return {"vehicles": vehicles.size(), "distance": snappedf(stats["distance"], 0.1),
		"min_moved": snappedf(min_moved, 0.1) if min_moved < INF else 0.0, "max_idle": snappedf(max_idle, 0.1),
		"claims": stats["claims"], "yields": stats["yields"], "overlaps": stats["overlaps"],
		"max_wait": snappedf(stats["max_wait"], 0.1), "turns": stats["turns"], "exits": stats["exits"],
		"min_person_gap": snappedf(gap, 0.01) if gap < INF else -1.0,
		"min_player_gap": snappedf(player_gap, 0.01) if player_gap < INF else -1.0, "clock": snappedf(clock, 0.1)}
