@tool
class_name GeogenSceneBuilder
extends RefCounted
## Builds Godot gameplay nodes from geogen glTF node extras (extras.geogen,
## schema: docs/schema/geogen-extras.v1.schema.json):
##
## - type: room_volume -> Area3D "<id>_area" (group "geogen_room", meta
##   geogen_room = {id, type}) with a BoxShape3D of the volume's size
## - type: place       -> Area3D "PlaceArea" (group "geogen_place", meta geogen_place
##   = {id, uid}) with a box or an extruded polygon; WorldLoader reports actors
##   entering and leaving it and the room areas
## - type: spawn       -> Marker3D child (group "geogen_spawn")
## - tags              -> node groups, e.g. "furniture.bed" and "furniture"
## - navigation        -> NavigationRegion3D baked from the static colliders
##                        (build_navigation, sized from a PlayerSpec)
##
## Unknown extras versions are reported once per scene.

const SUPPORTED_VERSION := 1
const NAV_MARGIN := 0.05  # extra agent radius (keep radius + margin a multiple of the 5 cm cell)
## Every key of extras.geogen: the properties of docs/schema/geogen-extras.v1.schema.json, in its order
## (tests/test_extras_contract.py fails when the two differ). The exporter writes no others.
const KNOWN_KEYS := ["version", "uid", "tags", "type", "shape", "collider", "level", "ratio", "walkable", "room", "door",
    "joint", "clips", "interactions", "footprint", "clearance", "light",
    "affordances", "portal", "place", "physics", "flows", "item", "travel", "poses", "npc", "container", "nav", "size",
    "city", "block", "lot", "building", "street_furniture", "lod", "interior",
    "signal", "character", "stairs", "water", "driving", "fleet", "train", "vehicle"]


static func extras(node: Node) -> Dictionary:
    if not node.has_meta("extras"):
        return {}
    var e = node.get_meta("extras")
    if e is Dictionary and e.get("geogen") is Dictionary:
        return e["geogen"]
    return {}


static func is_geogen_scene(root: Node) -> bool:
    if not extras(root).is_empty():
        return true
    for node in root.find_children("*", "", true, false):
        if not extras(node).is_empty():
            return true
    return false


## Apply every conversion to the tree under ``root``. Returns a summary.
static func build(root: Node, options := {}) -> Dictionary:
    var summary := {"rooms": 0, "places": 0, "spawns": 0, "tagged": 0, "warnings": []}
    var warned := {}
    for node in [root] + root.find_children("*", "", true, false):
        var g := extras(node)
        if g.is_empty():
            continue
        var version := int(g.get("version", 0))
        if version != SUPPORTED_VERSION and not warned.has("version"):
            warned["version"] = true
            summary["warnings"].append("extras.geogen version %d (supported: %d)" % [version, SUPPORTED_VERSION])
        for key in g:
            if not key in KNOWN_KEYS and not warned.has(key):
                warned[key] = true
                summary["warnings"].append("unknown extras.geogen key '%s'" % key)
        for tag in g.get("tags", []):
            _add_tag_groups(node, str(tag))
            summary["tagged"] += 1
        match g.get("type", ""):
            "room_volume":
                _room_area(node as Node3D, g)
                summary["rooms"] += 1
            "place":
                _place_area(node as Node3D, g)
                summary["places"] += 1
            "spawn":
                _spawn_marker(node as Node3D)
                summary["spawns"] += 1
        var stairs = g.get("stairs", {})
        if stairs is Dictionary and stairs.has("ends"):
            _stairs_link(node as Node3D, stairs["ends"], stairs.get("path", []))
    for w in summary["warnings"]:
        push_warning("geogen: %s" % w)
    return summary


## "furniture.bed" -> groups "furniture.bed" and "furniture" (persistent in saved scenes).
static func _add_tag_groups(node: Node, tag: String) -> void:
    var parts := tag.split(".")
    for i in parts.size():
        var group := ".".join(parts.slice(0, i + 1))
        if not node.is_in_group(group):
            node.add_to_group(group, true)


static func _room_area(node: Node3D, g: Dictionary) -> void:
    if node == null or node.has_node("RoomArea"):
        return
    var size: Array = g.get("size", [1, 1, 1])
    var area := Area3D.new()
    area.name = "RoomArea"
    area.monitorable = false
    area.collision_mask = 1 | 2   # the player, and NPC bodies (GeogenNpc.LAYER)
    area.set_meta("geogen_room", g.get("room", {}))
    area.add_to_group("geogen_room", true)
    var shape := CollisionShape3D.new()
    var box := BoxShape3D.new()
    box.size = Vector3(size[0], size[1], size[2])
    shape.shape = box
    area.add_child(shape)
    node.add_child(area)
    _own(area, node)


## A named place: the node is the middle of the volume's bottom, and the volume
## (a box, or a footprint polygon extruded upward) rises along its +Y.
static func _place_area(node: Node3D, g: Dictionary) -> void:
    var place = g.get("place")
    if node == null or not place is Dictionary or node.has_node("PlaceArea"):
        return
    var area := Area3D.new()
    area.name = "PlaceArea"
    area.monitorable = false
    area.collision_layer = 0
    area.collision_mask = 1 | 2   # the player, and NPC bodies (GeogenNpc.LAYER)
    area.set_meta("geogen_place", {"id": str(place.get("id", node.name)), "uid": str(g.get("uid", "")),
        "detect": place.get("detect", ["actors"]), "sink": place.get("sink")})
    area.add_to_group("geogen_place", true)
    if place.has("polygon"):
        # CollisionPolygon3D extrudes along its Z: turned a quarter about X, its
        # (x, y) points are (x, z) on the ground and the depth is the height.
        var height := float(place.get("height", 2.5))
        var shape := CollisionPolygon3D.new()
        var points := PackedVector2Array()
        for p in place["polygon"]:
            points.append(Vector2(p[0], p[1]))
        shape.polygon = points
        shape.depth = height
        shape.rotation = Vector3(PI / 2, 0, 0)
        shape.position = Vector3(0, height / 2.0, 0)
        area.add_child(shape)
    else:
        var size: Array = place.get("size", [1, 1, 1])
        var shape := CollisionShape3D.new()
        var box := BoxShape3D.new()
        box.size = Vector3(size[0], size[1], size[2])
        shape.shape = box
        shape.position = Vector3(0, size[1] / 2.0, 0)
        area.add_child(shape)
    node.add_child(area)
    _own(area, node)


static func _spawn_marker(node: Node3D) -> void:
    if node == null or node.has_node("SpawnMarker"):
        return
    var marker := Marker3D.new()
    marker.name = "SpawnMarker"
    marker.add_to_group("geogen_spawn", true)
    node.add_child(marker)
    _own(marker, node)


## Spiral stairs: link the bottom and the top on the navigation map. A tall spiral's
## navmesh can come out in pieces (Recast splits self-overlapping regions and can't
## join them across a riser), so paths use the link where the climb is cut.
## `walk` is the way round between the ends (the node's frame), kept on the link for
## bodies that walk a path (GeogenWalker.along_links).
static func _stairs_link(node: Node3D, ends: Array, walk: Array = []) -> void:
    if node == null or node.has_node("StairsLink") or ends.size() != 2:
        return
    var link := NavigationLink3D.new()
    link.name = "StairsLink"
    link.bidirectional = true
    link.start_position = Vector3(ends[0][0], ends[0][1], ends[0][2])
    link.end_position = Vector3(ends[1][0], ends[1][1], ends[1][2])
    var line := PackedVector3Array()
    for p in walk:
        line.append(Vector3(p[0], p[1], p[2]))
    link.set_meta("walk", line)
    link.add_to_group("geogen_nav_link", true)
    node.add_child(link)
    _own(link, node)


## Keep generated nodes when the importer saves the scene.
static func _own(child: Node, parent: Node) -> void:
    var owner := parent.owner if parent.owner != null else parent
    child.owner = owner
    for c in child.get_children():
        c.owner = owner


## Bake a NavigationRegion3D over ``root`` from its static colliders, with the
## agent sized from ``spec`` (a PlayerSpec). Call after colliders exist.
## Moving parts (door leaves, drawers) are AnimatableBody3Ds, i.e. static
## colliders; they're left out unless ``include_moving`` (so closed doors
## don't block the navmesh; agents open them).
## ``ground_margin`` > 0 adds the runtime's ground plane (y = 0, which isn't
## part of the export) under the model and that far around it.
const NAV_SINGLE_BAKE := 160.0   # m: models larger than this bake in tiles
const NAV_TILE := 48.0
const NAV_BORDER := 1.0


static func build_navigation(root: Node3D, spec, include_moving := false, ground_margin := 0.0) -> Node3D:
    var box := WorldLoader._aabb(root).grow(ground_margin)
    if maxf(box.size.x, box.size.z) <= NAV_SINGLE_BAKE:
        var region := NavigationRegion3D.new()
        region.name = "Navigation"
        region.navigation_mesh = bake_navigation_mesh(root, spec, include_moving, ground_margin)
        region.add_to_group("geogen_navigation")
        root.add_child(region)
        return region
    # Large models (a town with a railway): one bake over the whole area overflows the voxel
    # grid, so parse once and bake NAV_TILE-metre tiles clipped edge to edge.
    var parsed := _parse_source(root, spec, include_moving, ground_margin)
    var source: NavigationMeshSourceGeometryData3D = parsed[1]
    var group := Node3D.new()
    group.name = "Navigation"
    root.add_child(group)
    var nx := ceili(box.size.x / NAV_TILE)
    var nz := ceili(box.size.z / NAV_TILE)
    for i in nx:
        for j in nz:
            var tile := AABB(Vector3(box.position.x + i * NAV_TILE, box.position.y - 1.0, box.position.z + j * NAV_TILE),
                Vector3(NAV_TILE, box.size.y + 2.0, NAV_TILE))
            var mesh := nav_mesh_for(spec)
            mesh.filter_baking_aabb = tile.grow(NAV_BORDER)
            mesh.border_size = NAV_BORDER
            NavigationServer3D.bake_from_source_geometry_data(mesh, source)
            if mesh.get_polygon_count() == 0:
                continue
            var region := NavigationRegion3D.new()
            region.name = "NavTile_%d_%d" % [i, j]
            region.navigation_mesh = mesh
            region.add_to_group("geogen_navigation")
            group.add_child(region)
    return group


## Navigation mesh settings for the player (cell size matches the project's map).
static func nav_mesh_for(spec) -> NavigationMesh:
    var mesh := NavigationMesh.new()
    mesh.geometry_parsed_geometry_type = NavigationMesh.PARSED_GEOMETRY_STATIC_COLLIDERS
    mesh.geometry_collision_mask = 1
    mesh.cell_size = 0.05
    mesh.cell_height = 0.05
    mesh.agent_radius = (spec.radius if spec else 0.3) + NAV_MARGIN
    mesh.agent_height = spec.height if spec else 1.8
    mesh.agent_max_climb = spec.step_height if spec else 0.3
    mesh.agent_max_slope = spec.max_slope_deg if spec else 40.0
    return mesh


## Parse static collider ``bodies`` for one navigation tile (the caller picks the ones
## overlapping it: parsing a whole streamed piece, like a city's streets, costs a long frame):
## the source geometry reaches ``border`` beyond ``tile`` and the bake is clipped back to it,
## so neighbouring tiles meet edge to edge.
## Bake the result with NavigationServer3D.bake_from_source_geometry_data(_async).
static func parse_tile(bodies: Array, tile: AABB, spec, border: float) -> Array:
    var mesh := nav_mesh_for(spec)
    mesh.geometry_source_geometry_mode = NavigationMesh.SOURCE_GEOMETRY_GROUPS_EXPLICIT
    mesh.geometry_source_group_name = &"geogen_nav_parse"
    mesh.filter_baking_aabb = tile.grow(border)
    mesh.border_size = border
    var source := NavigationMeshSourceGeometryData3D.new()
    if bodies.is_empty():
        return [mesh, source]
    for body in bodies:
        body.add_to_group("geogen_nav_parse")
    NavigationServer3D.parse_source_geometry_data(mesh, source, bodies[0])
    for body in bodies:
        body.remove_from_group("geogen_nav_parse")
    return [mesh, source]


static func bake_navigation_mesh(root: Node3D, spec, include_moving := false, ground_margin := 0.0) -> NavigationMesh:
    var parsed := _parse_source(root, spec, include_moving, ground_margin)
    NavigationServer3D.bake_from_source_geometry_data(parsed[0], parsed[1])
    return parsed[0]


## The navmesh settings and the source geometry (static colliders, moving parts left out or
## as obstructions, the runtime ground around the model) for a model: [mesh, source].
static func _parse_source(root: Node3D, spec, include_moving := false, ground_margin := 0.0) -> Array:
    var mesh := NavigationMesh.new()
    mesh.geometry_parsed_geometry_type = NavigationMesh.PARSED_GEOMETRY_STATIC_COLLIDERS
    mesh.geometry_source_geometry_mode = NavigationMesh.SOURCE_GEOMETRY_ROOT_NODE_CHILDREN
    mesh.geometry_collision_mask = 1
    mesh.cell_size = 0.05
    mesh.cell_height = 0.05
    mesh.agent_radius = (spec.radius if spec else 0.3) + NAV_MARGIN
    mesh.agent_height = spec.height if spec else 1.8
    mesh.agent_max_climb = spec.step_height if spec else 0.3
    mesh.agent_max_slope = spec.max_slope_deg if spec else 40.0
    var moving: Array[CollisionObject3D] = []
    if not include_moving:
        for body in root.find_children("*", "AnimatableBody3D", true, false):
            if body.collision_layer & 1:
                moving.append(body)
                body.collision_layer &= ~1
    var source := NavigationMeshSourceGeometryData3D.new()
    NavigationServer3D.parse_source_geometry_data(mesh, source, root)
    for body in moving:
        body.collision_layer |= 1
    if include_moving:
        # Door leaves are thinner than a voxel and vanish in the bake; add each
        # moving part's footprint as an obstruction instead.
        for body in root.find_children("*", "AnimatableBody3D", true, false):
            if not body.collision_layer & 1:
                continue  # left out of this bake (a shut door when only some were opened)
            var part := body.get_parent() as MeshInstance3D
            if part == null or part.mesh == null:
                continue
            var box: AABB = part.global_transform * part.get_aabb()
            if minf(box.size.x, box.size.z) > 0.15:
                continue  # thick enough to rasterise (a lift car floor is walkable)
            box = box.grow(0.005)
            var corners := PackedVector3Array([
                Vector3(box.position.x, 0, box.position.z), Vector3(box.end.x, 0, box.position.z),
                Vector3(box.end.x, 0, box.end.z), Vector3(box.position.x, 0, box.end.z)])
            source.add_projected_obstruction(corners, box.position.y, box.size.y, false)
    if ground_margin > 0.0:
        var box := WorldLoader._aabb(root).grow(ground_margin)
        var a := Vector3(box.position.x, 0, box.position.z)
        var b := Vector3(box.end.x, 0, box.position.z)
        var c := Vector3(box.end.x, 0, box.end.z)
        var d := Vector3(box.position.x, 0, box.end.z)
        source.add_faces(PackedVector3Array([a, b, c, a, c, d]), Transform3D.IDENTITY)
    return [mesh, source]
