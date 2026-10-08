class_name GeogenTrain
extends Node3D
## Trains on a railway (manifest traffic.railways, see geogen/railway.py and geogen/trains.py).
##
## Everything is precomputed: the train node's extras.geogen.train holds the run (the head's
## distance along the line every `dt` seconds after departure, station dwells included), the
## timetable, each car's offset from the front and bogie positions, and each level crossing's
## closed windows. Departure k leaves at offset + k * headway seconds of world time, so a train
## is where it should be whenever the world is loaded. Per frame this is a lookup:
## - every departure on the line gets a copy of the consist; each car's bogies sit on the track
##   at their distance behind the head, the body between them, the bogies turned to the rails;
## - crossings are closed during their windows: barriers go down (their `barrier`
##   interaction) and road traffic (traffic.gd) waits at the crossing's stop points.

## Every trace event (the runtime API streams them), printed too with trace_enabled.
signal traced(event: Dictionary)

const VEHICLE_GROUP := "geogen_vehicle"

var world: WorldLoader
var data := {}
var railway := {}
var trace_enabled := false
var clock := 0.0                 # physics time since load (the fallback world time)
var stats := {"departures": {}, "closed_time": 0.0}

var _pts := PackedVector3Array()
var _cum := PackedFloat32Array()
var _length := 0.0
var _loop := false
var _template: Array[Node3D] = []        # the exported consist (set 0)
var _sets: Array = []                    # [[car nodes]] consist copies
var _set_of := {}                        # departure k -> set index
var _crossings := {}                     # id -> {closed, barriers: [names]}
var _lamp_materials := {}


static func spawn(world_: WorldLoader, node: Node3D, data_: Dictionary, railway_: Dictionary,
        offset := Vector3.ZERO) -> GeogenTrain:
    var t := GeogenTrain.new()
    t.world = world_
    t.data = data_
    t.railway = railway_
    t.name = "%s_train" % node.name
    world_.add_child(t)
    for p in railway_.get("points", []):
        t._pts.append(Vector3(p[0], p[1], p[2]) + offset)
    var total := 0.0
    t._cum.append(0.0)
    for k in range(1, t._pts.size()):
        total += t._pts[k].distance_to(t._pts[k - 1])
        t._cum.append(total)
    t._length = total
    t._loop = bool(railway_.get("loop", false))
    var cars: Array[Node3D] = []
    for car in data_.get("cars", []):
        var n := node.find_child(str(car["node"]), true, false) as Node3D
        if n != null:
            cars.append(n)
    for n in cars:
        var xf := n.global_transform
        n.reparent(t)
        n.global_transform = xf
        t._prepare(n)
    t._template = cars
    t._sets.append(cars)
    for c in railway_.get("crossings", []):
        t._crossings[c["id"]] = {"closed": false, "barriers": c.get("barriers", [])}
    return t


## A car: kinematic box body (blocks the player, no pushing), in the vehicle group for NPCs.
func _prepare(car: Node3D) -> void:
    GeogenTraffic._strip_bodies(car)
    var info: Dictionary = WorldLoader.geogen_extras(car).get("vehicle", {})
    var clearance: Array = info.get("clearance", [20, 2.8, 3.8])
    var body := AnimatableBody3D.new()
    body.sync_to_physics = false
    var shape := CollisionShape3D.new()
    var box := BoxShape3D.new()
    box.size = Vector3(float(clearance[1]), float(clearance[2]) - 0.6, float(clearance[0]))
    shape.shape = box
    shape.position = Vector3(0, float(clearance[2]) / 2.0 + 0.3, 0)
    body.add_child(shape)
    car.add_child(body)
    car.add_to_group(VEHICLE_GROUP)
    car.set_meta("geogen_vehicle_box", {"half": float(clearance[0]) / 2.0, "half_w": float(clearance[1]) / 2.0,
        "offset": 0.0})
    car.set_meta("geogen_speed", 0.0)
    GeogenSound.add_engine(car, info)
    for kind in info.get("lamps", {}):
        for part_name in info["lamps"][kind]:
            var part := car.find_child(str(part_name), true, false)
            if part == null:
                continue
            for mi: MeshInstance3D in ([part] if part is MeshInstance3D else []) + part.find_children("*", "MeshInstance3D", true, false):
                for i in mi.mesh.get_surface_count() if mi.mesh else 0:
                    var mat := mi.get_active_material(i) as BaseMaterial3D
                    if mat != null and mat.emission_enabled and not _lamp_materials.has(mat):
                        _lamp_materials[mat] = [mat.emission_energy_multiplier, kind]


func set_night(on: bool) -> void:
    for mat in _lamp_materials:
        var boost := 5.0 if _lamp_materials[mat][1] == "head" else 3.0
        (mat as BaseMaterial3D).emission_energy_multiplier = _lamp_materials[mat][0] * (boost if on else 1.0)


func trace(event: Dictionary) -> void:
    traced.emit(event)
    if trace_enabled:
        print("train: %s" % JSON.stringify(event))


## Seconds of world time: the clock's position in its (compressed) day, else time since load.
func world_time() -> float:
    if world != null and world.clock != null and world.clock.day_length > 0.0:
        return world.clock.hours / 24.0 * world.clock.day_length
    return clock


func _point(d: float) -> Vector3:
    d = fposmod(d, _length) if _loop else clampf(d, 0.0, _length)
    var k := _cum.bsearch(d)
    k = clampi(k, 1, _pts.size() - 1)
    var a := _cum[k - 1]
    var span := maxf(_cum[k] - a, 1e-6)
    return _pts[k - 1].lerp(_pts[k], (d - a) / span)


func head_at(tau: float) -> float:
    var run: Array = data.get("run", [])
    var dt := float(data.get("dt", 0.5))
    var f := clampf(tau / dt, 0.0, run.size() - 1.0)
    var k := mini(int(f), run.size() - 2)
    return lerpf(float(run[k]), float(run[k + 1]), f - k)


func _physics_process(delta: float) -> void:
    clock += delta
    var t := world_time()
    var tt: Dictionary = data.get("timetable", {})
    var headway := maxf(float(tt.get("headway", 300.0)), 1.0)
    var offset := float(tt.get("offset", 0.0))
    var duration := float(data.get("duration", 0.0))
    var active := {}
    var k0 := ceili((t - offset - duration) / headway)
    var k1 := floori((t - offset) / headway)
    for k in range(k0, k1 + 1):
        var tau := t - (offset + k * headway)
        if tau >= 0.0 and tau <= duration:
            active[k] = tau
    # Consist copies: keep each departure's set, free the finished ones, clone when short.
    for k in _set_of.keys():
        if not active.has(k):
            _set_of.erase(k)
    var used := {}
    for k in _set_of:
        used[_set_of[k]] = true
    for k in active:
        if _set_of.has(k):
            continue
        var free := -1
        for i in _sets.size():
            if not used.has(i):
                free = i
                break
        if free < 0:
            free = _sets.size()
            _sets.append(_clone_set())
        _set_of[k] = free
        used[free] = true
        stats["departures"][str(k)] = true
        trace({"t": snappedf(t, 0.1), "depart": k})
    for i in _sets.size():
        for car in _sets[i]:
            car.visible = used.has(i)
            for shape: CollisionShape3D in car.find_children("*", "CollisionShape3D", true, false):
                shape.disabled = not used.has(i)
    for k in active:
        _place(_sets[_set_of[k]], active[k], delta)
    # Level crossings: closed during their windows of any train on the line.
    var closures: Dictionary = data.get("closures", {})
    for cid in _crossings:
        var closed := false
        for k in active:
            for w in closures.get(cid, []):
                if active[k] >= float(w[0]) and active[k] <= float(w[1]):
                    closed = true
        var c: Dictionary = _crossings[cid]
        if closed:
            stats["closed_time"] += delta
        if closed != c["closed"]:
            c["closed"] = closed
            world.set_crossing(cid, closed)
            if closed:
                _sound_horns()
            for b in c["barriers"]:
                for it in world.interactions_of(str(b)):
                    if it.states.has("down"):
                        it.set_state("down" if closed else "up")
            trace({"t": snappedf(t, 0.1), "crossing": cid, "closed": closed})


## The lead cars of the trains on the line sound their horns (approaching a crossing).
func _sound_horns() -> void:
    for k in _set_of:
        var lead: Node3D = _sets[_set_of[k]][0]
        if GeogenSound.horn(lead, WorldLoader.geogen_extras(lead).get("vehicle", {})):
            stats["horns"] = int(stats.get("horns", 0)) + 1


func _clone_set() -> Array:
    var out: Array[Node3D] = []
    for car in _template:
        var copy := car.duplicate() as Node3D
        add_child(copy)
        out.append(copy)
    return out


## Cars of one departure at time tau after it left.
func _place(cars: Array, tau: float, delta: float) -> void:
    var head := head_at(tau)
    var speed := (head - head_at(tau - delta)) / maxf(delta, 1e-6)
    var specs: Array = data.get("cars", [])
    for i in cars.size():
        var car: Node3D = cars[i]
        var spec: Dictionary = specs[i]
        var centre: float = head - float(spec["offset"])
        var bogies: Array = spec.get("bogies", [5.0, -5.0])
        var front := _point(centre + float(bogies[0]))
        var rear := _point(centre + float(bogies[-1]))
        var dir := front - rear
        if dir.length_squared() < 1e-6:
            continue
        var yaw := atan2(dir.x, dir.z)
        var pitch := -atan2(dir.y, Vector2(dir.x, dir.z).length())
        car.global_transform = Transform3D(Basis.from_euler(Vector3(pitch, yaw, 0.0)), (front + rear) / 2.0)
        car.set_meta("geogen_speed", speed)
        GeogenSound.set_speed(car.get_node_or_null("Engine") as AudioStreamPlayer3D, speed)
        # Bogies turn to the rails under them; wheels spin.
        var info: Dictionary = WorldLoader.geogen_extras(car).get("vehicle", {})
        var names: Array = info.get("bogies", [])
        for b in names.size():
            var bogie := car.find_child(str(names[b]), true, false) as Node3D
            if bogie == null:
                continue
            var at := centre + float(bogies[mini(b, bogies.size() - 1)])
            var tangent := _point(at + 1.0) - _point(at - 1.0)
            bogie.rotation.y = wrapf(atan2(tangent.x, tangent.z) - yaw, -PI, PI)
        var step := speed * delta
        for w in info.get("wheels", []):
            var wheel := car.find_child(str(w["part"]), true, false) as Node3D
            if wheel != null:
                wheel.rotate(Vector3.RIGHT, step / maxf(float(w["radius"]), 0.1))


func report() -> Dictionary:
    var t := world_time()
    var trains := []
    var tt: Dictionary = data.get("timetable", {})
    for k in _set_of:
        var tau := t - (float(tt.get("offset", 0.0)) + int(k) * float(tt.get("headway", 300.0)))
        var car: Node3D = _sets[_set_of[k]][0]
        trains.append({"departure": k, "tau": snappedf(tau, 0.1), "head": snappedf(head_at(tau), 0.1),
            "position": [snappedf(car.global_position.x, 0.1), snappedf(car.global_position.z, 0.1)]})
    var crossings := {}
    for cid in _crossings:
        crossings[cid] = _crossings[cid]["closed"]
    return {"train": String(name), "time": snappedf(t, 0.1), "active": trains, "departures": stats["departures"].size(),
        "sets": _sets.size(), "crossings": crossings, "horns": int(stats.get("horns", 0)), "closed_time": snappedf(stats["closed_time"], 0.1)}
