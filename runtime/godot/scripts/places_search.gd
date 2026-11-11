class_name GeogenPlaceSearch
extends RefCounted
## Where things can be put (geogen.97.2): spots found against the world's colliders, not
## authored surfaces, so anything that physically holds a thing is a candidate: a table, a
## chair seat, a shelf, a window sill, the top of the fridge. Wants (npc.gd) ask for the
## spots that hold a prop within an actor's reach and pick one by their own weights.
##
## A spot is found by casting down through a grid of columns over the search area. Each level
## a column meets (a table top, then the floor under it) is kept when it is level, isn't
## somewhere actors walk, has the room the thing's box needs, and holds all four corners of
## the box.

const WORLD_MASK := 1            ## static world colliders and props (GeogenProps.LAYER)
const GRID := 0.2                ## m between columns
const LEVELS := 4                ## levels looked at per column, from the top down
const LEVEL_NORMAL := 0.97       ## a spot's normal is at least this upright
const CORNER_INSET := 0.7        ## the box's corners, this far out from its centre, must be held
const CORNER_DROP := 0.02        ## m a corner's support may lie below the centre's
const CLEAR_GAP := 0.02          ## m the box is lifted off the spot for the free-space test
const FLOOR_NEAR := 1.0          ## m (flat) to the navmesh of the floor a point is part of (it stops short of walls)
const FLOOR_STEP := 0.08         ## m above or below that floor that is still it
const QUALITIES := ["high", "enclosed", "hidden", "warm", "soft"]   ## the ones with a meaning here (npc.py PLACE_QUALITIES)
const QUALITY_TAG := "quality."  ## a thing's tag `quality.<name>` gives its spots that quality
const PROBE_LIFT := 0.1          ## m above a spot its surroundings are looked at from
const RING := 8                  ## directions looked in round a spot
const HIGH_FULL := 2.0           ## m above the floor that is as high as it gets
const ENCLOSED_REACH := 0.6      ## m within which a wall or a top shuts a spot in
const HIDDEN_RING := 2.0         ## m from a spot the places to stand and look at it from
const EYE := 1.5                 ## m above the floor one looks from
const WARM_RADIUS := 1.5         ## m from a warm thing its warmth is gone
const SUN_WARM := 0.7            ## how warm a spot in the sun is
const SUN_UP := 0.1              ## clock.sun_height() above which the sun warms
const SUN_RAY := 60.0            ## m toward the sun that must be clear
const TAGGED_MSEC := 5000        ## how long a tag lookup is kept
const NAV_LIFT := 0.3            ## m the navmesh may lie above the floor it was baked from


## How much a spot has of quality ``name``, 0..1 (npc.py PLACE_QUALITIES). A thing tagged
## `quality.<name>` (or part of one) gives its spots 1; otherwise high, enclosed, hidden and
## warm are worked out here, and any other name is 0. ``floor_y``: the floor the asker stands on.
static func quality(world: WorldLoader, space: PhysicsDirectSpaceState3D, map: RID, spot: Dictionary, name: String,
        floor_y: float, exclude: Array[RID]) -> float:
    if (spot["tags"] as Array).has(QUALITY_TAG + name):
        return 1.0
    var at: Vector3 = (spot["position"] as Vector3) + Vector3.UP * PROBE_LIFT
    match name:
        "high":
            return clampf(((spot["position"] as Vector3).y - floor_y) / HIGH_FULL, 0.0, 1.0)
        "enclosed":
            var shut := 0
            for i in RING:
                if _blocked(space, at, at + Vector3.RIGHT.rotated(Vector3.UP, TAU * i / RING) * ENCLOSED_REACH, exclude):
                    shut += 1
            if _blocked(space, at, at + Vector3.UP * ENCLOSED_REACH, exclude):
                shut += 1
            return float(shut) / (RING + 1)
        "hidden":
            # From where one could stand around it, at eye height: how many can't see it.
            var stands := 0
            var blind := 0
            for i in RING:
                var p := at + Vector3.RIGHT.rotated(Vector3.UP, TAU * i / RING) * HIDDEN_RING
                var near := NavigationServer3D.map_get_closest_point(map, Vector3(p.x, floor_y, p.z))
                if Vector2(near.x - p.x, near.z - p.z).length() > 0.4 or absf(near.y - floor_y) > 0.5:
                    continue
                stands += 1
                if _blocked(space, near + Vector3.UP * EYE, at, exclude):
                    blind += 1
            return float(blind) / stands if stands > 0 else 1.0
        "warm":
            var warm := 0.0
            for thing in _tagged(world, QUALITY_TAG + "warm"):
                warm = maxf(warm, 1.0 - (thing as Node3D).global_position.distance_to(at) / WARM_RADIUS)
            if warm < SUN_WARM and world.clock != null and world.clock.sun != null and world.clock.sun_height() > SUN_UP \
                    and not _blocked(space, at, at + world.clock.sun.global_basis.z * SUN_RAY, exclude):
                warm = SUN_WARM                # the sun reaches it now
            return warm
    return 0.0


## Whether something stands between ``from`` and ``to`` (short of ``to`` itself).
static func _blocked(space: PhysicsDirectSpaceState3D, from: Vector3, to: Vector3, exclude: Array[RID]) -> bool:
    var hit := space.intersect_ray(PhysicsRayQueryParameters3D.create(from, to, WORLD_MASK, exclude))
    return not hit.is_empty() and (hit["position"] as Vector3).distance_to(to) > 0.05


## The loaded things carrying ``tag`` (looked up again every few seconds: worlds stream).
static func _tagged(world: WorldLoader, tag: String) -> Array:
    var cache: Dictionary = world.get_meta("place_tagged", {})
    var entry: Dictionary = cache.get(tag, {})
    if entry.is_empty() or Time.get_ticks_msec() - int(entry["at"]) > TAGGED_MSEC:
        var found := world.find_children("*", "Node3D", true, false).filter(
            func(n): return tag in WorldLoader.geogen_extras(n).get("tags", []))
        entry = {"at": Time.get_ticks_msec(), "nodes": found}
        cache[tag] = entry
        world.set_meta("place_tagged", cache)
    return (entry["nodes"] as Array).filter(func(n): return is_instance_valid(n))


## What prop ``uid`` rests on: {kind: "floor" | "thing" | "none", node, uid}. "floor" is anything
## actors walk on (the navmesh is there), whatever its height; "none" while it is held or falling.
static func resting_on(world: WorldLoader, uid: String) -> Dictionary:
    var entry: Dictionary = world.props.props.get(uid, {})
    if entry.is_empty() or world.props.holder_of(uid) != null:
        return {"kind": "none"}
    var body: RigidBody3D = entry["body"]
    if body.linear_velocity.length() > 0.1:
        return {"kind": "none"}
    var box: Dictionary = world.props.box_of(uid)
    var centre: Vector3 = body.global_transform * (box["center"] as Vector3)
    var reach: float = (box["size"] as Vector3).length() / 2.0 + 0.1
    var space := body.get_world_3d().direct_space_state
    var query := PhysicsRayQueryParameters3D.create(centre, centre + Vector3.DOWN * reach, WORLD_MASK, [body.get_rid()])
    var hit := space.intersect_ray(query)
    if hit.is_empty():
        return {"kind": "none"}
    var owner := _owner(hit["collider"])
    var kind := "floor" if walkable(space, body.get_world_3d().navigation_map, hit["position"]) else "thing"
    return {"kind": kind, "node": owner, "uid": WorldLoader.uid_of(owner) if owner != null else ""}


## Whether ``point`` (on a collider) is floor: level with what actors walk on nearby. The
## navmesh keeps a body's width off walls and furniture, so the floor by a wall has none over
## it; the floor under the nearest navmesh is found and compared by height.
static func walkable(space: PhysicsDirectSpaceState3D, map: RID, point: Vector3) -> bool:
    var near := NavigationServer3D.map_get_closest_point(map, point)
    if Vector2(near.x - point.x, near.z - point.z).length() > FLOOR_NEAR or absf(near.y - point.y) > NAV_LIFT + FLOOR_STEP:
        return false
    var under := _down(space, near + Vector3.UP * 0.02, near.y - NAV_LIFT, [])
    var floor_y: float = (under["position"] as Vector3).y if not under.is_empty() else near.y
    return absf(floor_y - point.y) < FLOOR_STEP


## The spots that hold a box of ``size`` (its footprint x, z and height y) within ``area`` (world
## XZ polygon; empty: ``radius`` m round ``origin``), between ``low`` and ``high`` (world y):
## [{position (where its underside rests), node, uid, tags, height}]. ``exclude``: bodies that
## aren't in the way (the thing itself). Floors (where actors walk) are left out unless ``floors``.
static func spots(world: WorldLoader, space: PhysicsDirectSpaceState3D, map: RID, size: Vector3, origin: Vector3,
        area: PackedVector2Array, radius: float, low: float, high: float, exclude: Array[RID], floors := false) -> Array:
    var lo := Vector2(origin.x - radius, origin.z - radius)
    var hi := Vector2(origin.x + radius, origin.z + radius)
    if area.size() >= 3:
        lo = area[0]
        hi = area[0]
        for p in area:
            lo = lo.min(p)
            hi = hi.max(p)
    var shape := BoxShape3D.new()
    shape.size = Vector3(size.x, maxf(size.y - CLEAR_GAP, 0.01), size.z)
    var free := PhysicsShapeQueryParameters3D.new()
    free.shape = shape
    free.collision_mask = WORLD_MASK | 2 | Player.LAYER
    free.exclude = exclude
    var out := []
    var x := lo.x + GRID / 2.0
    while x < hi.x:
        var z := lo.y + GRID / 2.0
        while z < hi.y:
            if area.size() < 3 or Geometry2D.is_point_in_polygon(Vector2(x, z), area):
                var top := high + size.y
                for _level in LEVELS:
                    var hit := _down(space, Vector3(x, top, z), low - 0.05, exclude)
                    if hit.is_empty():
                        break
                    top = (hit["position"] as Vector3).y - 0.03       # the next level is under this one
                    var spot := _spot(world, space, map, free, size, hit, low, high, exclude, floors)
                    if not spot.is_empty():
                        out.append(spot)
            z += GRID
        x += GRID
    return out


static func _spot(world: WorldLoader, space: PhysicsDirectSpaceState3D, map: RID, free: PhysicsShapeQueryParameters3D,
        size: Vector3, hit: Dictionary, low: float, high: float, exclude: Array[RID], floors: bool) -> Dictionary:
    var at: Vector3 = hit["position"]
    if (hit["normal"] as Vector3).y < LEVEL_NORMAL or at.y < low or at.y > high:
        return {}
    var collider: Object = hit["collider"]
    if collider is RigidBody3D or collider is CharacterBody3D or collider is AnimatableBody3D:
        return {}                              # not on another loose thing, on someone, or on what moves
    if not floors and walkable(space, map, at):
        return {}
    for corner in [Vector2(-1, -1), Vector2(-1, 1), Vector2(1, -1), Vector2(1, 1)]:
        var c := at + Vector3(corner.x * size.x / 2.0, 0, corner.y * size.z / 2.0) * CORNER_INSET
        var under := _down(space, c + Vector3.UP * 0.05, at.y - CORNER_DROP - 0.01, exclude)
        if under.is_empty() or absf((under["position"] as Vector3).y - at.y) > CORNER_DROP:
            return {}                          # it would overhang an edge, or straddle a step
    free.transform = Transform3D(Basis.IDENTITY, at + Vector3.UP * (size.y / 2.0 + CLEAR_GAP))
    if not space.intersect_shape(free, 1).is_empty():
        return {}                              # something is there already, or there's no headroom
    var owner := _owner(collider)
    return {"position": at, "node": owner, "uid": WorldLoader.uid_of(owner) if owner != null else "",
        "tags": _tags(owner), "height": at.y}


static func _down(space: PhysicsDirectSpaceState3D, from: Vector3, to_y: float, exclude: Array[RID]) -> Dictionary:
    if from.y <= to_y:
        return {}
    var query := PhysicsRayQueryParameters3D.create(from, Vector3(from.x, to_y, from.z), WORLD_MASK, exclude)
    return space.intersect_ray(query)


## The thing a collider belongs to: the nearest node above it with a stable ID.
static func _owner(collider: Object) -> Node:
    var node := collider as Node
    while node != null and WorldLoader.uid_of(node) == "":
        node = node.get_parent()
    return node


## The tags of a thing and of what it is part of (extras `tags`), nearest first.
static func _tags(node: Node) -> Array:
    var out := []
    while node != null:
        for tag in WorldLoader.geogen_extras(node).get("tags", []):
            if not tag in out:
                out.append(tag)
        node = node.get_parent()
    return out
