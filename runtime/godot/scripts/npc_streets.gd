class_name GeogenNpcStreets
extends RefCounted
## What an NPC on foot knows of the streets (npc.gd): the painted crossings (from the traffic
## graph), routes over them, and the pedestrian signals. Worked out on first use.

var world: WorldLoader
var _crossing_ends := []   # [[end a, end b]] of the painted crossings, on the pavement (world)
var _signal_rects = null   # [[Rect2 (plan), signal id, walked along x?]] of the signalled crossings
var _crossing_bounds := AABB()


## A path that walks along the road (below both its ends, inside the streets) rerouted over the
## nearest painted crossing when that isn't much longer: to one end, straight across, on. Empty
## when ``path`` stays off the road or no crossing helps.
func crosswalk_route(map: RID, path: PackedVector3Array, from: Vector3, to: Vector3) -> PackedVector3Array:
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
    if _crossing_ends[0] == null or not on_road(path):
        return PackedVector3Array()
    var direct := NpcGeometry.length(path)
    var best := PackedVector3Array()
    var best_len := direct * 2.0 + 25.0
    for ends in _crossing_ends:
        for flip in [false, true]:
            var e1: Vector3 = ends[1] if flip else ends[0]
            var e2: Vector3 = ends[0] if flip else ends[1]
            if from.distance_to(e1) > 60.0 and to.distance_to(e2) > 60.0:
                continue
            var first := GeogenWalker.find_path(map, from, e1)
            var last := GeogenWalker.find_path(map, e2, to)
            if first.is_empty() or last.is_empty() or on_road(first) or on_road(last):
                continue
            var total := NpcGeometry.length(first) + e1.distance_to(e2) + NpcGeometry.length(last)
            if total < best_len:
                best_len = total
                best = first + last
    return best


## The walk signal of a pedestrian crossing ("" where nothing runs the signals: no traffic).
func pedestrian_state(id: String) -> String:
    for t in world.traffic:
        if is_instance_valid(t):
            return t.pedestrian_state(id)
    return ""


## The signalled crossing whose road `to` is on, reached from `from` on one of its kerbs (beyond
## an end of its long side; the other two sides are road); "" if none.
func crossing_from_kerb(from: Vector2, to: Vector2) -> String:
    if _signal_rects == null:
        _signal_rects = []
        for sig in world.traffic_graph.get("pedestrian_signals", []):
            var r: Array = sig["crosswalk"]
            var rect := Rect2(float(r[0]), float(r[1]), float(r[2]) - float(r[0]), float(r[3]) - float(r[1]))
            _signal_rects.append([rect, str(sig["id"]), rect.size.x > rect.size.y])
    for entry in _signal_rects:
        var rect: Rect2 = entry[0]
        if not rect.has_point(to) or rect.has_point(from):
            continue
        var along: float = from.x if entry[2] else from.y
        var lo: float = rect.position.x if entry[2] else rect.position.y
        var hi: float = rect.end.x if entry[2] else rect.end.y
        if along <= lo or along >= hi:
            return entry[1]
    return ""


## Does the path dip to road level (below both its ends) inside the streets?
func on_road(path: PackedVector3Array) -> bool:
    if path.size() < 2:
        return false
    var level := minf(path[0].y, path[-1].y) - 0.08
    for p in path:
        if p.y < level and _crossing_bounds.grow(2.0).has_point(Vector3(p.x, 0.0, p.z)):
            return true
    return false
