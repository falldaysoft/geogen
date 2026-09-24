class_name GeogenChunkStreamer
extends Node
## Streams a chunked geogen export (a ``geogen-chunks`` index written by
## ``python -m geogen.main -s X --export-godot --stream``) around a focus
## point, normally the player:
##
## - ``base`` (streets, street furniture) is always loaded;
## - a chunk's full exterior within ``full_radius`` of its bounds, its
##   decimated ``lod`` out to ``lod_radius``, nothing beyond (the LOD stays
##   until the full version is in, so nothing pops out);
## - a building's interior within ``interior_radius`` of its bounds while
##   the full exterior is loaded;
## - pieces unload ``hysteresis`` metres further out than they load.
##
## GLB parsing and scene generation run on the WorkerThreadPool; the main
## thread builds texture mipmaps (reading texture images off the main thread
## crashes), adds the nodes and runs WorldLoader.setup_root.
## ``prime()`` loads what the start position needs synchronously.

const INDEX_FORMAT := "geogen-chunks"

var world: WorldLoader
var full_radius := 60.0
var lod_radius := 400.0
var interior_radius := 14.0
var hysteresis := 10.0
var threaded := true

var chunks: Array[Dictionary] = []   # {name, file, lod, bounds: AABB, interiors: [{building, file, bounds}]}
var bounds := AABB()
var _dir := ""
var _loaded := {}     # file -> Node3D in the tree
var _tasks := {}      # file -> {"id": int, "out": Array}
var _loads := 0
var _unloads := 0


func open(index_path: String, index: Dictionary) -> void:
	_dir = index_path.get_base_dir()
	var first := true
	for c in index.get("chunks", []):
		var chunk := {"name": c.get("name", ""), "file": c.get("file", ""), "lod": c.get("lod", ""),
			"bounds": _to_aabb(c.get("bounds")), "interiors": []}
		for it in c.get("interiors", []):
			chunk["interiors"].append({"building": it.get("building", ""), "file": it.get("file", ""),
				"bounds": _to_aabb(it.get("bounds"))})
		chunks.append(chunk)
		if chunk["bounds"].size != Vector3.ZERO:
			bounds = chunk["bounds"] if first else bounds.merge(chunk["bounds"])
			first = false


## Load everything wanted at ``focus`` right now (no threads): the start of a session.
func prime(focus: Vector3) -> void:
	# Interiors are only wanted once their exterior is in, so repeat until settled.
	for _pass in 3:
		var added := false
		for file in _wanted(focus):
			if not _loaded.has(file):
				_attach(file, _load_glb(_path(file), world))
				added = true
		if not added:
			break
	_unload_unwanted(focus)


func _process(_delta: float) -> void:
	var focus := world.stream_focus
	var wanted := _wanted(focus)
	for file in wanted:
		if _loaded.has(file) or _tasks.has(file):
			continue
		if not threaded:
			_attach(file, _load_glb(_path(file), world))
			continue
		var out := []
		var path := _path(file)
		var w := world
		var id := WorkerThreadPool.add_task(func(): out.append(GeogenChunkStreamer._load_glb(path, w)))
		_tasks[file] = {"id": id, "out": out}
	for file in _tasks.keys():
		var task: Dictionary = _tasks[file]
		if not WorkerThreadPool.is_task_completed(task["id"]):
			continue
		WorkerThreadPool.wait_for_task_completion(task["id"])
		_tasks.erase(file)
		var root: Node3D = task["out"][0] if not task["out"].is_empty() else null
		if file in _wanted(world.stream_focus):
			_attach(file, root)
		elif root != null:
			root.free()
	_unload_unwanted(focus)


## Never leave a worker parsing a chunk behind (quitting or reloading).
func _exit_tree() -> void:
	for file in _tasks.keys():
		WorkerThreadPool.wait_for_task_completion(_tasks[file]["id"])
		for root in _tasks[file]["out"]:
			if root != null:
				root.free()
	_tasks.clear()


## Files that should be loaded for ``focus`` (with hysteresis for loaded ones).
func _wanted(focus: Vector3) -> Array[String]:
	var out: Array[String] = []
	for chunk in chunks:
		var file: String = chunk["file"]
		if chunk["name"] == "base":
			out.append(file)
			continue
		var d := _distance(focus, chunk["bounds"])
		var full_r := full_radius + (hysteresis if _loaded.has(file) else 0.0)
		var want_full := d <= full_r
		if want_full:
			out.append(file)
		var lod: String = chunk["lod"]
		if lod != "":
			var lod_r := lod_radius + (hysteresis if _loaded.has(lod) else 0.0)
			# Keep the LOD up until the full exterior has arrived.
			if (not want_full and d <= lod_r) or (want_full and not _loaded.has(file)):
				out.append(lod)
		if want_full and _loaded.has(file):
			for interior in chunk["interiors"]:
				var ifile: String = interior["file"]
				var r := interior_radius + (hysteresis if _loaded.has(ifile) else 0.0)
				if _distance(focus, interior["bounds"]) <= r:
					out.append(ifile)
	return out


func _unload_unwanted(focus: Vector3) -> void:
	var wanted := _wanted(focus)
	for file in _loaded.keys():
		if file in wanted:
			continue
		var root: Node3D = _loaded[file]
		_loaded.erase(file)
		world.forget(root)
		root.queue_free()
		_unloads += 1


func _attach(file: String, root: Node3D) -> void:
	if root == null:
		push_error("geogen: cannot load chunk %s" % file)
		_loaded[file] = Node3D.new()   # don't retry every frame
		world.add_child(_loaded[file])
		return
	root.name = file.get_basename()
	world._prepare_materials(root)
	world.add_child(root)
	world.setup_root(root)
	_loaded[file] = root
	_loads += 1


## Parse a GLB into a scene (safe on a worker thread: nothing is in the tree yet).
static func _load_glb(path: String, _w: WorldLoader = null) -> Node3D:
	var doc := GLTFDocument.new()
	var state := GLTFState.new()
	if doc.append_from_file(ProjectSettings.globalize_path(path), state) != OK:
		return null
	return doc.generate_scene(state) as Node3D


func _path(file: String) -> String:
	return _dir.path_join(file)


static func _to_aabb(b) -> AABB:
	if not b is Array or b.size() != 2:
		return AABB()
	var lo := Vector3(b[0][0], b[0][1], b[0][2])
	var hi := Vector3(b[1][0], b[1][1], b[1][2])
	return AABB(lo, hi - lo)


## Horizontal distance from ``p`` to a box (0 inside it).
static func _distance(p: Vector3, box: AABB) -> float:
	if box.size == Vector3.ZERO:
		return 0.0
	var dx := maxf(maxf(box.position.x - p.x, 0.0), p.x - box.end.x)
	var dz := maxf(maxf(box.position.z - p.z, 0.0), p.z - box.end.z)
	return Vector2(dx, dz).length()


## What is loaded now: chunk names at full / LOD detail and loaded interiors.
func report() -> Dictionary:
	var full := []
	var lod := []
	var interiors := []
	for chunk in chunks:
		if _loaded.has(chunk["file"]):
			full.append(chunk["name"])
		if chunk["lod"] != "" and _loaded.has(chunk["lod"]):
			lod.append(chunk["name"])
		for interior in chunk["interiors"]:
			if _loaded.has(interior["file"]):
				interiors.append("%s/%s" % [chunk["name"], interior["building"]])
	return {"full": full, "lod": lod, "interiors": interiors, "pending": _tasks.size(),
		"loads": _loads, "unloads": _unloads}
