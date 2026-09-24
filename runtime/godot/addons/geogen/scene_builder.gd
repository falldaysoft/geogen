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
const KNOWN_KEYS := ["version", "tags", "type", "shape", "collider", "walkable", "room", "size", "door",
	"joint", "door_swings", "footprint", "clearance", "interactions", "light", "switch", "openings",
	"clear_height", "wall_inset", "placed_by", "furnish_report", "meta", "floorplan"]


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
static func build_navigation(root: Node3D, spec) -> NavigationRegion3D:
	var region := NavigationRegion3D.new()
	region.name = "Navigation"
	var mesh := NavigationMesh.new()
	mesh.geometry_parsed_geometry_type = NavigationMesh.PARSED_GEOMETRY_STATIC_COLLIDERS
	mesh.geometry_source_geometry_mode = NavigationMesh.SOURCE_GEOMETRY_ROOT_NODE_CHILDREN
	mesh.cell_size = 0.05
	mesh.cell_height = 0.05
	mesh.agent_radius = spec.radius if spec else 0.3
	mesh.agent_height = spec.height if spec else 1.8
	mesh.agent_max_climb = spec.step_height if spec else 0.3
	mesh.agent_max_slope = spec.max_slope_deg if spec else 40.0
	# Moving parts (door leaves, drawers) are AnimatableBody3Ds, i.e. static
	# colliders; leave them out so doorways stay navigable (agents open doors).
	var moving: Array[CollisionObject3D] = []
	for body in root.find_children("*", "AnimatableBody3D", true, false):
		if body.collision_layer & 1:
			moving.append(body)
			body.collision_layer &= ~1
	var source := NavigationMeshSourceGeometryData3D.new()
	mesh.geometry_collision_mask = 1
	NavigationServer3D.parse_source_geometry_data(mesh, source, root)
	for body in moving:
		body.collision_layer |= 1
	NavigationServer3D.bake_from_source_geometry_data(mesh, source)
	region.navigation_mesh = mesh
	region.add_to_group("geogen_navigation")
	root.add_child(region)
	return region
