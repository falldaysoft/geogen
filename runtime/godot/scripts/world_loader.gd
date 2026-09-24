class_name WorldLoader
extends Node3D
## Loads geogen exports (.glb + .manifest.json) from generated/ at runtime.
## Collision comes from the exporter's collider nodes (Godot import suffixes
## -colonly = trimesh, -convcolonly = convex), which runtime glTF loading
## doesn't process, so they are turned into static bodies here. Room volumes
## and spawns come from node extras / the manifest. Watches the manifests
## (written last by the exporter) and reloads a model when it is re-exported.

signal world_loaded(aabb: AABB)

const DEFAULT_GENERATED_DIR := "res://generated"
const POLL_SECONDS := 0.5

## Scene name to load (e.g. "cottage"); empty loads every export in generated/.
var scene_name := ""
## Directory holding exports (res:// or absolute path).
var generated_dir := DEFAULT_GENERATED_DIR
## Show collision shapes as wireframe overlays.
var show_colliders := false:
	set(value):
		show_colliders = value
		for node in get_tree().get_nodes_in_group("geogen_collider_debug"):
			node.visible = value

var _mtimes := {}  # manifest path -> modified time at last load
## Room volumes: [{id, type, xform: Transform3D (global), size: Vector3}]
var rooms: Array[Dictionary] = []
## Spawn points from the manifests: [{name, position: Vector3, yaw_deg: float}]
var spawns: Array[Dictionary] = []
var _poll := 0.0
var _collider_material: StandardMaterial3D


func _ready() -> void:
	_collider_material = StandardMaterial3D.new()
	_collider_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_collider_material.albedo_color = Color(0.2, 1.0, 0.4)
	_collider_material.albedo_color = Color(0.2, 1.0, 0.4, 0.6)
	_collider_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA


## Load (or reload) everything requested. Returns the combined bounds.
func load_all() -> AABB:
	for child in get_children():
		child.free()
	_mtimes.clear()
	rooms.clear()
	spawns.clear()
	for manifest in _manifests():
		_load_model(manifest)
	var aabb := world_aabb()
	world_loaded.emit(aabb)
	return aabb


## Player spec from the first loaded manifest, or null if none.
func player_spec() -> PlayerSpec:
	for manifest in _manifests():
		return PlayerSpec.from_manifest(manifest)
	return null


func world_aabb() -> AABB:
	var aabb := AABB()
	var first := true
	for mi in find_children("*", "MeshInstance3D", true, false):
		if mi.is_in_group("geogen_collider_debug"):
			continue
		var box: AABB = mi.global_transform * mi.get_aabb()
		aabb = box if first else aabb.merge(box)
		first = false
	return aabb


func _process(delta: float) -> void:
	_poll += delta
	if _poll < POLL_SECONDS:
		return
	_poll = 0.0
	for manifest in _manifests():
		if FileAccess.get_modified_time(manifest) != _mtimes.get(manifest, -1):
			print("geogen: %s changed, reloading" % manifest.get_file())
			load_all()
			return


func _manifests() -> Array[String]:
	var result: Array[String] = []
	if scene_name != "":
		var path := "%s/%s.manifest.json" % [generated_dir, scene_name]
		if FileAccess.file_exists(path):
			result.append(path)
		return result
	var dir := DirAccess.open(generated_dir)
	if dir == null:
		return result
	for file in dir.get_files():
		if file.ends_with(".manifest.json"):
			result.append("%s/%s" % [generated_dir, file])
	result.sort()
	return result


func _load_model(manifest_path: String) -> void:
	_mtimes[manifest_path] = FileAccess.get_modified_time(manifest_path)
	var manifest = JSON.parse_string(FileAccess.get_file_as_string(manifest_path))
	if not manifest is Dictionary:
		push_error("geogen: bad manifest %s" % manifest_path)
		return
	var model_path: String = manifest_path.get_base_dir().path_join(manifest.get("model", ""))
	var doc := GLTFDocument.new()
	var state := GLTFState.new()
	var err := doc.append_from_file(ProjectSettings.globalize_path(model_path), state)
	if err != OK:
		push_error("geogen: cannot load %s (%s)" % [model_path, error_string(err)])
		return
	var root := doc.generate_scene(state)
	root.name = manifest.get("name", model_path.get_file().get_basename())
	add_child(root)
	_prepare_materials(root)
	var count := _add_collision(root)
	_collect_rooms(root)
	for s in manifest.get("spawns", []):
		var f: Array = s.get("forward", [0, 0, -1])
		var p: Array = s.get("position", [0, 0, 0])
		spawns.append({"name": s.get("name", ""), "position": Vector3(p[0], p[1], p[2]),
			"yaw_deg": rad_to_deg(atan2(-float(f[0]), -float(f[2])))})
	print("geogen: loaded %s (%d meshes, %d rooms)" % [model_path.get_file(), count, rooms.size()])


## extras.geogen of a node imported from glTF, or {}.
static func geogen_extras(node: Node) -> Dictionary:
	if not node.has_meta("extras"):
		return {}
	var extras = node.get_meta("extras")
	if extras is Dictionary and extras.get("geogen") is Dictionary:
		return extras["geogen"]
	return {}


func _collect_rooms(root: Node) -> void:
	for node in root.find_children("*", "Node3D", true, false):
		var g := geogen_extras(node)
		if g.get("type") == "room_volume":
			var size: Array = g.get("size", [0, 0, 0])
			var room: Dictionary = g.get("room", {})
			rooms.append({"id": room.get("id", node.name), "type": room.get("type", ""),
				"xform": (node as Node3D).global_transform, "size": Vector3(size[0], size[1], size[2])})


## Id of the room volume containing ``pos`` (world space), or "".
func room_at(pos: Vector3) -> String:
	for room in rooms:
		var local: Vector3 = room["xform"].affine_inverse() * pos
		var half: Vector3 = room["size"] / 2.0
		if absf(local.x) <= half.x and absf(local.y) <= half.y and absf(local.z) <= half.z:
			return room["id"]
	return ""


## Runtime-loaded glTF textures have no mipmaps, so fine patterns (brick,
## shingles) alias badly at a distance. Build mipmaps and filter anisotropically.
func _prepare_materials(root: Node) -> void:
	var done := {}
	for mi: MeshInstance3D in root.find_children("*", "MeshInstance3D", true, false):
		if mi.mesh == null:
			continue
		for surface in mi.mesh.get_surface_count():
			var mat := mi.mesh.surface_get_material(surface) as BaseMaterial3D
			if mat == null or done.has(mat):
				continue
			done[mat] = true
			mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
			for param in [BaseMaterial3D.TEXTURE_ALBEDO, BaseMaterial3D.TEXTURE_METALLIC,
					BaseMaterial3D.TEXTURE_ROUGHNESS, BaseMaterial3D.TEXTURE_NORMAL,
					BaseMaterial3D.TEXTURE_AMBIENT_OCCLUSION]:
				var tex := mat.get_texture(param)
				if tex == null or done.has(tex):
					continue
				var image := tex.get_image()
				if image == null or image.has_mipmaps():
					continue
				if image.is_compressed():
					image.decompress()
				image.generate_mipmaps(param == BaseMaterial3D.TEXTURE_NORMAL)
				var mipped := ImageTexture.create_from_image(image)
				done[mipped] = true
				mat.set_texture(param, mipped)


## Turn exported collider nodes (<name>-colonly / -convcolonly) into static
## bodies and drop their meshes. Exports without collider nodes (older
## geogen) fall back to a trimesh collider on every mesh.
func _add_collision(root: Node) -> int:
	var meshes := root.find_children("*", "MeshInstance3D", true, false)
	var colliders: Array[MeshInstance3D] = []
	for mi: MeshInstance3D in meshes:
		if _is_collider(mi):
			colliders.append(mi)
	if colliders.is_empty():
		for mi: MeshInstance3D in meshes:
			if mi.mesh != null:
				_add_body(mi, mi.mesh.create_trimesh_shape(), Transform3D.IDENTITY)
		return meshes.size()
	for mi in colliders:
		if mi.mesh != null:
			var convex: bool = String(mi.name).ends_with("-convcolonly") or geogen_extras(mi).get("shape") in ["box", "hull"]
			var shape: Shape3D = mi.mesh.create_convex_shape(true, false) if convex else mi.mesh.create_trimesh_shape()
			_add_body(mi.get_parent(), shape, mi.transform)
		mi.get_parent().remove_child(mi)
		mi.free()
	return meshes.size() - colliders.size()


static func _is_collider(mi: MeshInstance3D) -> bool:
	var n := String(mi.name)
	return n.ends_with("-colonly") or n.ends_with("-convcolonly") or geogen_extras(mi).get("type") == "collider"


func _add_body(parent: Node, shape: Shape3D, xform: Transform3D) -> void:
	var body := StaticBody3D.new()
	body.name = "Collider"
	body.transform = xform
	var col := CollisionShape3D.new()
	col.shape = shape
	body.add_child(col)
	parent.add_child(body)
	var debug := MeshInstance3D.new()
	debug.mesh = shape.get_debug_mesh()
	debug.material_override = _collider_material
	debug.visible = show_colliders
	debug.add_to_group("geogen_collider_debug")
	body.add_child(debug)
