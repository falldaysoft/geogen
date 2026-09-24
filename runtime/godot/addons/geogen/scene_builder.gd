@tool
class_name GeogenSceneBuilder
extends RefCounted
## Builds Godot gameplay nodes from geogen glTF node extras (extras.geogen,
## schema: docs/schema/geogen-extras.v1.schema.json):
##
## - type: room_volume -> Area3D "<id>_area" (group "geogen_room", meta
##   geogen_room = {id, type}) with a BoxShape3D of the volume's size
## - type: spawn       -> Marker3D child (group "geogen_spawn")
## - tags              -> node groups, e.g. "furniture.bed" and "furniture"
## - navigation        -> NavigationRegion3D baked from the static colliders
##                        (build_navigation, sized from a PlayerSpec)
##
## Unknown extras versions are reported once per scene.

const SUPPORTED_VERSION := 1
const NAV_MARGIN := 0.05  # extra agent radius (keep radius + margin a multiple of the 5 cm cell)
const KNOWN_KEYS := ["version", "tags", "type", "shape", "collider", "walkable", "room", "size", "door",
	"joint", "door_swings", "footprint", "clearance", "interactions", "light", "switch", "openings",
	"clear_height", "wall_inset", "placed_by", "furnish_report", "meta", "floorplan", "gate", "nav", "stairs",
	"storey", "building", "facade", "walkable"]


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
	var summary := {"rooms": 0, "spawns": 0, "tagged": 0, "warnings": []}
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
			"spawn":
				_spawn_marker(node as Node3D)
				summary["spawns"] += 1
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
	area.set_meta("geogen_room", g.get("room", {}))
	area.add_to_group("geogen_room", true)
	var shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = Vector3(size[0], size[1], size[2])
	shape.shape = box
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
static func build_navigation(root: Node3D, spec, include_moving := false) -> NavigationRegion3D:
	var region := NavigationRegion3D.new()
	region.name = "Navigation"
	region.navigation_mesh = bake_navigation_mesh(root, spec, include_moving)
	region.add_to_group("geogen_navigation")
	root.add_child(region)
	return region


static func bake_navigation_mesh(root: Node3D, spec, include_moving := false) -> NavigationMesh:
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
	NavigationServer3D.bake_from_source_geometry_data(mesh, source)
	return mesh
