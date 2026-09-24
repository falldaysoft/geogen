class_name WorldLoader
extends Node3D
## Loads geogen exports (.glb + .manifest.json) from generated/ at runtime.
## Collision comes from the exporter's collider nodes (Godot import suffixes
## -colonly = trimesh, -convcolonly = convex), which runtime glTF loading
## doesn't process, so they are turned into static bodies here. Room volumes
## and spawns come from node extras / the manifest. Watches the manifests
## (written last by the exporter) and reloads a model when it is re-exported.

signal world_loaded(aabb: AABB)
## An interaction reached a state: (asset name, interaction name, state, emitted event or "").
signal interaction_event(asset: String, interaction: String, state: String, event: String)

const DEFAULT_GENERATED_DIR := "res://generated"
const POLL_SECONDS := 0.5

## Scene name to load (e.g. "cottage"); empty loads every export in generated/.
var scene_name := ""
## Directory holding exports (res:// or absolute path).
var generated_dir := DEFAULT_GENERATED_DIR
## Bake a navigation mesh per loaded model (from its colliders).
var bake_navigation := true
## Open every openable interaction before baking, with the open parts as
## obstacles (headless playtests: doors must not block routes, leaves must).
var open_before_bake := false
## Show collision shapes as wireframe overlays.
var show_colliders := false:
	set(value):
		show_colliders = value
		for node in get_tree().get_nodes_in_group("geogen_collider_debug"):
			node.visible = value

var _mtimes := {}  # manifest path -> modified time at last load
## Gap between models when several exports load at once.
const MODEL_SPACING := 6.0
var _next_x := 0.0    # where the next model's -X edge goes when loading several
## Bounds of each loaded model, in load order.
var model_aabbs: Array[AABB] = []
## Room volumes: [{id, type, xform: Transform3D (global), size: Vector3}]
var rooms: Array[Dictionary] = []
## Spawn points from the manifests: [{name, position: Vector3, yaw_deg: float}]
var spawns: Array[Dictionary] = []
## Fixture lights by their fixture node's name (light switches refer to them).
var lights_by_name := {}
## Actor poses from extras.geogen.affordances, world space:
## [{type, position: Vector3, yaw_deg: float, height: float, node: Node3D}]
var affordances: Array[Dictionary] = []
## Interactions from asset extras (GeogenInteraction nodes, children of this loader).
var interactions: Array[GeogenInteraction] = []
var _moving := {}     # Node3D driven by an interaction -> true
## Gates: [{node: Node3D, interaction: GeogenInteraction, open_in: String, open: bool}]
var _gates: Array[Dictionary] = []
var _target_of := {}  # Node3D aimed at -> GeogenInteraction
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
	_next_x = 0.0
	model_aabbs.clear()
	rooms.clear()
	spawns.clear()
	interactions.clear()
	lights_by_name.clear()
	affordances.clear()
	_gates.clear()
	_moving.clear()
	_target_of.clear()
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
	return _aabb(self)


static func _aabb(root: Node) -> AABB:
	var aabb := AABB()
	var first := true
	for mi in root.find_children("*", "MeshInstance3D", true, false):
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
	# Every export is authored around the origin, so when several load at once
	# (no --scene) lay them out in a row along X instead of on top of each other.
	var offset := Vector3.ZERO
	if scene_name == "":
		var box := _aabb(root)
		if box.size != Vector3.ZERO:
			offset.x = _next_x - box.position.x
			_next_x += box.size.x + MODEL_SPACING
		root.position = offset
	model_aabbs.append(_aabb(root))
	_prepare_materials(root)
	_collect_interactions(root)
	_add_lights(root)
	_wire_switches()
	_collect_affordances(root)
	var count := _add_collision(root)
	_collect_gates(root)
	_collect_rooms(root)
	var summary := GeogenSceneBuilder.build(root)
	if open_before_bake:
		for it in interactions:
			if it.asset.is_ancestor_of(root) or root.is_ancestor_of(it.asset):
				if it.states.has("open"):
					it.set_state("open", true)
					it.hold = true
	if bake_navigation:
		GeogenSceneBuilder.build_navigation(root, PlayerSpec.from_manifest(manifest_path), open_before_bake)
	for s in manifest.get("spawns", []):
		var f: Array = s.get("forward", [0, 0, -1])
		var p: Array = s.get("position", [0, 0, 0])
		spawns.append({"name": s.get("name", ""), "position": Vector3(p[0], p[1], p[2]) + offset,
			"yaw_deg": rad_to_deg(atan2(-float(f[0]), -float(f[2])))})
	print("geogen: loaded %s (%d meshes, %d rooms, %d spawns, %d tagged)" % [
		model_path.get_file(), count, summary["rooms"], summary["spawns"], summary["tagged"]])


## extras.geogen of a node imported from glTF, or {}.
static func geogen_extras(node: Node) -> Dictionary:
	if not node.has_meta("extras"):
		return {}
	var extras = node.get_meta("extras")
	if extras is Dictionary and extras.get("geogen") is Dictionary:
		return extras["geogen"]
	return {}


## Ceiling fixtures etc. carry extras.geogen.light: add a light just below them.
func _add_lights(root: Node) -> void:
	for node in root.find_children("*", "Node3D", true, false):
		var spec = geogen_extras(node).get("light")
		if not spec is Dictionary:
			continue
		# Exports carry a KHR_lights_punctual light (imported as a Light3D child);
		# tune it with our energy/range/shadows. Older exports get one added.
		var light: OmniLight3D = null
		for child in node.get_children():
			if child is OmniLight3D:
				light = child
			elif geogen_extras(child).get("type") == "light":
				for grand in child.get_children():
					if grand is OmniLight3D:
						light = grand
				if light == null and child is OmniLight3D:
					light = child
		if light == null:
			light = OmniLight3D.new()
			var offset = spec.get("offset", [0, -0.5, 0])
			light.position = Vector3(offset[0], offset[1], offset[2]) if offset is Array else Vector3(0, offset, 0)
			node.add_child(light)
		light.name = "Light"
		var c: Array = spec.get("color", [1, 1, 1])
		light.light_color = Color(c[0], c[1], c[2])
		light.light_energy = float(spec.get("energy", 1.0))
		light.omni_range = float(spec.get("range", 5.0))
		light.omni_attenuation = 1.0
		light.shadow_enabled = true
		light.add_to_group("geogen_light")
		lights_by_name[String(node.name)] = light
		# The fixture itself mustn't shadow its own light.
		for mi: MeshInstance3D in node.find_children("*", "MeshInstance3D", true, false):
			mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF


## Light switches: their interaction's on/off state shows or hides the named light.
func _wire_switches() -> void:
	for it in interactions:
		var spec = geogen_extras(it.asset).get("switch")
		if not spec is Dictionary:
			continue
		var light: Light3D = lights_by_name.get(str(spec.get("light", "")))
		if light == null:
			continue
		var fixture := light.get_parent()
		while fixture != null and not geogen_extras(fixture).has("light"):
			fixture = fixture.get_parent()
		it.state_entered.connect(func(state: String, _event: String):
			light.visible = state != "off"
			if fixture != null:
				_set_emission(fixture, state != "off"))


## Dim or restore a fixture's glowing materials (lamp shades) with its light.
static func _set_emission(fixture: Node, on: bool) -> void:
	for mi: MeshInstance3D in fixture.find_children("*", "MeshInstance3D", true, false):
		for i in mi.mesh.get_surface_count() if mi.mesh else 0:
			var mat := mi.get_active_material(i) as BaseMaterial3D
			if mat == null or not (mat.emission_enabled or mi.has_meta("geogen_dimmed")):
				continue
			if mi.get_surface_override_material(i) == null:
				mat = mat.duplicate()
				mi.set_surface_override_material(i, mat)
			mat.emission_enabled = on
			mi.set_meta("geogen_dimmed", not on)


func _collect_affordances(root: Node) -> void:
	for node in root.find_children("*", "Node3D", true, false):
		var list = geogen_extras(node).get("affordances")
		if not list is Array:
			continue
		var xform := (node as Node3D).global_transform
		for a in list:
			var p: Array = a.get("position", [0, 0, 0])
			var forward := xform.basis * Basis(Vector3.UP, deg_to_rad(float(a.get("yaw", 0.0)))) * Vector3.BACK
			affordances.append({"type": str(a.get("type", "sit")), "position": xform * Vector3(p[0], p[1], p[2]),
				"yaw_deg": rad_to_deg(atan2(-forward.x, -forward.z)), "node": node,
				"height": float(a.get("height", p[1])), "asset": String(node.name)})


## Nearest affordance within ``radius`` of a world point, or {}.
func affordance_near(point: Vector3, radius := 0.9) -> Dictionary:
	var best := {}
	var best_d := radius
	for a in affordances:
		var d: float = (a["position"] as Vector3).distance_to(point)
		if d < best_d:
			best_d = d
			best = a
	return best


## Interaction state of every interaction, keyed "<asset>/<interaction>".
func save_state() -> Dictionary:
	var data := {}
	for it in interactions:
		data["%s/%s" % [it.asset.name, it.interaction_name]] = it.snapshot()
	return data


func load_state(data: Dictionary) -> void:
	for it in interactions:
		var key := "%s/%s" % [it.asset.name, it.interaction_name]
		if data.has(key):
			it.restore(data[key])


func _collect_rooms(root: Node) -> void:
	for node in root.find_children("*", "Node3D", true, false):
		var g := geogen_extras(node)
		if g.get("type") == "room_volume":
			var size: Array = g.get("size", [0, 0, 0])
			var room: Dictionary = g.get("room", {})
			rooms.append({"id": room.get("id", node.name), "type": room.get("type", ""), "nav": g.get("nav", true),
				"xform": (node as Node3D).global_transform, "size": Vector3(size[0], size[1], size[2])})


func _collect_interactions(root: Node) -> void:
	for node in root.find_children("*", "Node3D", true, false):
		var data = geogen_extras(node).get("interactions")
		if not data is Dictionary:
			continue
		for iname in data:
			var it := GeogenInteraction.from_extras(node, iname, data[iname])
			add_child(it)
			interactions.append(it)
			var asset_name := String(node.name)
			it.state_entered.connect(func(state: String, event: String):
				interaction_event.emit(asset_name, iname, state, event))
			for part in it.moving_nodes():
				_moving[part] = true
			for t in it.targets:
				_target_of[t] = it


## The interaction whose target contains ``node`` (a hit collider), or null.
func interaction_for(node: Node) -> GeogenInteraction:
	while node != null and node != self:
		if node.has_meta("geogen_interaction"):
			return node.get_meta("geogen_interaction")
		if _target_of.has(node):
			return _target_of[node]
		node = node.get_parent()
	return null


## Interactions of the asset node named ``asset_name``.
func interactions_of(asset_name: String) -> Array[GeogenInteraction]:
	var result: Array[GeogenInteraction] = []
	for it in interactions:
		if String(it.asset.name) == asset_name:
			result.append(it)
	return result


## Nodes with extras.geogen.gate: solid unless their interaction has arrived at open_in.
func _collect_gates(root: Node) -> void:
	for node in root.find_children("*", "Node3D", true, false):
		var gate = geogen_extras(node).get("gate")
		if not gate is Dictionary:
			continue
		var it: GeogenInteraction = null
		var ancestor := node.get_parent()
		while ancestor != null and it == null:
			for candidate in interactions:
				if candidate.asset == ancestor and candidate.interaction_name == gate.get("interaction", ""):
					it = candidate
			ancestor = ancestor.get_parent()
		if it == null:
			push_warning("geogen: gate %s: no interaction '%s'" % [node.name, gate.get("interaction", "")])
			continue
		var entry := {"node": node, "interaction": it, "open_in": str(gate.get("open_in", "")), "open": null}
		_gates.append(entry)
		_update_gate(entry)


func _update_gate(gate: Dictionary) -> void:
	var it: GeogenInteraction = gate["interaction"]
	var open: bool = it.state == gate["open_in"] and it.target == it.state
	if gate["open"] == open:
		return
	gate["open"] = open
	var node: Node3D = gate["node"]
	node.visible = not open
	for shape: CollisionShape3D in node.find_children("*", "CollisionShape3D", true, false):
		shape.set_deferred("disabled", open)


func _physics_process(_delta: float) -> void:
	for gate in _gates:
		_update_gate(gate)


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


func _is_moving(node: Node) -> bool:
	while node != null and node != self:
		if _moving.has(node):
			return true
		node = node.get_parent()
	return false


static func _is_collider(mi: MeshInstance3D) -> bool:
	var n := String(mi.name)
	return n.ends_with("-colonly") or n.ends_with("-convcolonly") or geogen_extras(mi).get("type") == "collider"


func _add_body(parent: Node, shape: Shape3D, xform: Transform3D) -> void:
	# Bodies under a part an interaction moves must be animatable so they
	# push the player and carry their new pose into physics.
	var body: PhysicsBody3D = StaticBody3D.new()
	if _is_moving(parent):
		var animatable := AnimatableBody3D.new()
		# Moved by its parent part, not by itself: sync_to_physics would only
		# track the body's own transform and leave the collider behind.
		animatable.sync_to_physics = false
		body = animatable
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
