class_name WorldLoader
extends Node3D
## Loads geogen exports (.glb + .manifest.json) from generated/ at runtime and
## gives every mesh trimesh collision. Watches the manifests (written last by
## the exporter) and reloads a model when it is re-exported.

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
	print("geogen: loaded %s (%d meshes)" % [model_path.get_file(), count])


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


## Give every mesh a static trimesh collider (until geogen exports explicit
## -col/-colonly colliders, geogen-3cc.17).
func _add_collision(root: Node) -> int:
	var meshes := root.find_children("*", "MeshInstance3D", true, false)
	for mi: MeshInstance3D in meshes:
		if mi.mesh == null:
			continue
		var shape := mi.mesh.create_trimesh_shape()
		var body := StaticBody3D.new()
		body.name = "Collider"
		var col := CollisionShape3D.new()
		col.shape = shape
		body.add_child(col)
		mi.add_child(body)
		var debug := MeshInstance3D.new()
		debug.mesh = _wireframe(shape.get_faces())
		debug.material_override = _collider_material
		debug.visible = show_colliders
		debug.add_to_group("geogen_collider_debug")
		body.add_child(debug)
	return meshes.size()


## Line mesh of every triangle edge, for the collider overlay.
static func _wireframe(faces: PackedVector3Array) -> ArrayMesh:
	var lines := PackedVector3Array()
	lines.resize(faces.size() * 2)
	for t in range(0, faces.size(), 3):
		for e in 3:
			lines[(t + e) * 2] = faces[t + e]
			lines[(t + e) * 2 + 1] = faces[t + (e + 1) % 3]
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = lines
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_LINES, arrays)
	return mesh
