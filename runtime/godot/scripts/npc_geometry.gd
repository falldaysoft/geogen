class_name NpcGeometry
extends RefCounted
## Plan geometry for NPC paths (npc.gd): boxes round vehicles, arcs round characters, snapping to
## the navmesh. Points are world space; "plan" is XZ.

## Characters further apart in height than this aren't in each other's way (a storey up, a turn
## of a spiral stair above).
const LEVEL_TOLERANCE := 1.2


static func in_box(p: Vector3, c: Vector3, fwd: Vector3, right: Vector3, hl: float, hw: float) -> bool:
    var d := flat(p - c)
    return absf(d.dot(fwd)) < hl and absf(d.dot(right)) < hw


## Does segment a-b (plan) cross the box centred on c (half extents hl along fwd, hw along right)?
static func segment_hits_box(a: Vector3, b: Vector3, c: Vector3, fwd: Vector3, right: Vector3,
        hl: float, hw: float) -> bool:
    var p0 := Vector2(flat(a - c).dot(fwd), flat(a - c).dot(right))
    var p1 := Vector2(flat(b - c).dot(fwd), flat(b - c).dot(right))
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
static func around_box(a: Vector3, _b: Vector3, c: Vector3, fwd: Vector3, right: Vector3, hl: float, hw: float,
        map: RID) -> Array:
    var side := signf(flat(a - c).dot(right))
    if side == 0.0:
        side = 1.0
    var near := signf(flat(a - c).dot(fwd))
    if near == 0.0:
        near = 1.0
    for end in [near, -near]:
        var p1 = on_nav_near(c + fwd * end * hl + right * side * hw, map)
        var p2 = on_nav_near(c + fwd * end * hl - right * side * hw, map)
        if p1 != null and p2 != null:
            return [p1, p2]
    return []


## The navmesh point nearest ``p`` if it's within 0.4 m (a box corner may sit just off it).
static func on_nav_near(p: Vector3, map: RID):
    var q := NavigationServer3D.map_get_closest_point(map, p)
    if Vector2(q.x - p.x, q.z - p.z).length() < 0.4 and absf(q.y - p.y) < 0.5:
        return q
    return null


static func flat(v: Vector3) -> Vector3:
    return Vector3(v.x, 0.0, v.z)


## Are two points on one level? Someone a storey up, or a turn of a spiral stair above, is
## not in the way, however near they are in plan.
static func same_level(a: Vector3, b: Vector3) -> bool:
    return absf(a.y - b.y) < LEVEL_TOLERANCE


## ``p`` pushed out from ``c`` to distance ``r`` (turning up to 90 deg either way to stay on the navmesh), or null.
static func edge_point(c: Vector3, p: Vector3, r: float, map: RID):
    var base := atan2(p.x - c.x, p.z - c.z)
    for turn in [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5]:
        var q = on_nav(Vector3(c.x + sin(base + turn) * r, p.y, c.z + cos(base + turn) * r), map)
        if q != null:
            return q
    return null


## Waypoints on a circle of radius ``r`` about ``c`` from ``a``'s bearing to ``b``'s (shorter valid side), or [].
static func arc_around(c: Vector3, a: Vector3, b: Vector3, r: float, map: RID) -> Array:
    var ta := atan2(a.x - c.x, a.z - c.z)
    var tb := atan2(b.x - c.x, b.z - c.z)
    var short := wrapf(tb - ta, -PI, PI)
    for sweep in [short, short - signf(short) * TAU]:
        var steps := maxi(2, ceili(absf(sweep) / deg_to_rad(40.0)))
        var points := []
        for k in range(1, steps + 1):
            var t: float = ta + sweep * k / steps
            var q = on_nav(Vector3(c.x + sin(t) * r, a.y, c.z + cos(t) * r), map)
            if q == null:
                points = []
                break
            points.append(q)
        # Points we're already standing on are no way round (a stair too narrow for the
        # circle: they'd count as reached at once, and the detour would be made again).
        while not points.is_empty() and flat(points[0] - a).length() < GeogenNpc.WAYPOINT_RADIUS * 2.0:
            points.pop_front()
        if not points.is_empty():
            return points
    return []


## The navmesh point at ``p`` (same floor), or null when ``p`` isn't on it.
static func on_nav(p: Vector3, map: RID):
    var q := NavigationServer3D.map_get_closest_point(map, p)
    if Vector2(q.x - p.x, q.z - p.z).length() < 0.08 and absf(q.y - p.y) < 0.35:
        return q
    return null


static func radius_of(body: Node) -> float:
    for child in body.get_children():
        if child is CollisionShape3D and child.shape is CylinderShape3D:
            return child.shape.radius
    return 0.3


static func length(path: PackedVector3Array) -> float:
    var total := 0.0
    for k in range(1, path.size()):
        total += path[k].distance_to(path[k - 1])
    return total
