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
## The player used or walked into a travel point: its extras.geogen.travel ({scene?, spawn?, prompt, on}).
signal travel_requested(travel: Dictionary)

const DEFAULT_GENERATED_DIR := "res://generated"
const POLL_SECONDS := 0.5

## Scene name to load (e.g. "cottage"); empty loads every export in generated/.
var scene_name := ""
## generated/catalogue.json (format geogen-catalogue, written by
## `python -m geogen.main --export-catalogue`), or {} when there is none.
var catalogue := {}
## Directory holding exports (res:// or absolute path).
var generated_dir := DEFAULT_GENERATED_DIR
## Bake a navigation mesh per loaded model (from its colliders).
var bake_navigation := true
## The runtime's ground plane (main.tscn, y = 0) is walkable this far around each model.
var ground_margin := 6.0
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
## NPCs spawned from extras.geogen.npc nodes (see npc.gd).
var npcs: Array[GeogenNpc] = []
## Print every NPC decision and step (--npc-trace).
var npc_trace := false
## The lane graph of the loaded export (manifest / chunk index "traffic"), or {}.
var traffic_graph := {}
var _traffic_offset := Vector3.ZERO
## Traffic controllers (traffic.gd), one per exported traffic node.
var traffic: Array[GeogenTraffic] = []
## Trains (train.gd), one per exported train node.
var trains: Array[GeogenTrain] = []
## Level crossings: id -> true while the barriers are down (trains set it, traffic reads it).
var crossing_closed := {}
## Print traffic claims, overlaps and respawns (--traffic-trace).
var traffic_trace := false
## The world clock (clock.gd), set by main; NPC routines and traffic schedules read it.
var clock: GeogenClock = null
## After dark: `auto: night` lights and vehicle lamps are on (clock.gd sets it).
var night := false
var _auto_lights: Array[Dictionary] = []   # [{light, fixture}] for `auto: night` fixtures
## Doorways NPCs path through: [{interaction, open, closed, center, normal (world),
## width, height, depth, clearance, node}]
var portals: Array[Dictionary] = []
var _reservations := {}   # affordance id -> [GeogenNpc]
## Interactions from asset extras (GeogenInteraction nodes, children of this loader).
var interactions: Array[GeogenInteraction] = []
var _moving := {}     # Node3D driven by an interaction -> true
## Gates: [{node: Node3D, interaction: GeogenInteraction, open_in: String, open: bool}]
var _gates: Array[Dictionary] = []
var _target_of := {}  # Node3D aimed at -> GeogenInteraction
var _poll := 0.0
var _collider_material: StandardMaterial3D
## Prefer a chunked export (<scene>_chunks/<scene>.chunks.json) over a single .glb.
var prefer_chunks := false
## --stream: prefer chunked exports for every scene, not just catalogue entries marked stream.
var force_chunks := false
## Where streamed exports load around (main.gd keeps it on the player).
var stream_focus := Vector3.ZERO
## The streamer of a chunked export, or null.
var streamer: GeogenChunkStreamer = null
## Where to prime a streamed export (the player's start when overridden), or null for its first spawn.
var prime_focus = null
## Or the name of the spawn to prime a streamed export around (scene switches and travel).
var prime_spawn := ""
## Optional [full, lod, interior] radii for the streamer (metres).
var stream_radii := PackedFloat64Array()
## Reuse baked navmeshes from <generated>/.navcache when the export hasn't changed.
var nav_cache := true
const NAV_CACHE_VERSION := 1
## Load phases of the last load_all (ms; streamed chunks' setup adds up too), and whether to
## print them (--timings).
var load_timings := {}
var print_timings := false
## True while load_all_async is parsing; hot reload waits.
var loading := false
var _parse_task := -1
var _load_started := 0
## Travel points (extras.geogen.travel): [{node, travel: Dictionary}]
var travel_points: Array[Dictionary] = []
## Walk-in travel volumes stay quiet until this physics frame, so arriving inside one doesn't
## bounce you back (you have to step out and in again).
var travel_armed_at := 0
const TRAVEL_ARM_FRAMES := 5


func _ready() -> void:
    _collider_material = StandardMaterial3D.new()
    _collider_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
    _collider_material.albedo_color = Color(0.2, 1.0, 0.4)
    _collider_material.albedo_color = Color(0.2, 1.0, 0.4, 0.6)
    _collider_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA


## Read generated/catalogue.json into ``catalogue`` ({} if missing or malformed).
func read_catalogue() -> Dictionary:
    catalogue = {}
    var path := "%s/catalogue.json" % generated_dir
    if not FileAccess.file_exists(path):
        return catalogue
    var data = JSON.parse_string(FileAccess.get_file_as_string(path))
    if data is Dictionary and data.get("format") == "geogen-catalogue":
        catalogue = data
    else:
        push_error("geogen: bad catalogue %s" % path)
    return catalogue


## The catalogue entry named ``name``, or {}.
func catalogue_entry(name: String) -> Dictionary:
    if catalogue.is_empty():
        read_catalogue()
    for entry in catalogue.get("scenes", []):
        if entry.get("name") == name:
            return entry
    return {}


## With no scene chosen, pick ``preferred`` (the last scene picked, if it's still
## exported) or else the catalogue's default (streamed if its entry says so).
## Returns the chosen name, or "" to load every export as before.
func use_catalogue_default(preferred := "") -> String:
    if scene_name != "":
        if is_exported(scene_name):
            select_scene(scene_name)    # --scene NAME: still streamed if its catalogue entry says so
        return scene_name
    if read_catalogue().is_empty():
        return scene_name
    if preferred != "" and is_exported(preferred):
        select_scene(preferred)
        print("geogen: last scene %s (from user://settings.cfg)" % preferred)
        return preferred
    var name: String = catalogue.get("default", "")
    if not is_exported(name):
        push_warning("geogen: catalogue default '%s' isn't exported; loading every export" % name)
        return ""
    select_scene(name)
    print("geogen: default scene %s (from catalogue.json)" % name)
    return name


## Whether scene ``name`` has an export in the generated directory (single file or chunked).
func is_exported(name: String) -> bool:
    if name == "":
        return false
    var entry := catalogue_entry(name)
    if not entry.is_empty() and FileAccess.file_exists("%s/%s" % [generated_dir, entry.get("manifest", "")]):
        return true
    return FileAccess.file_exists("%s/%s.manifest.json" % [generated_dir, name]) \
        or FileAccess.file_exists("%s/%s_chunks/%s.chunks.json" % [generated_dir, name, name])


## Make ``name`` the scene the next load_all() loads, streamed if its catalogue
## entry says so (or --stream). False if it isn't exported.
func select_scene(name: String) -> bool:
    if not is_exported(name):
        return false
    scene_name = name
    prefer_chunks = force_chunks or bool(catalogue_entry(name).get("stream", false))
    return true


## Scenes the runtime can switch between: the catalogue's entries in order
## ([{name, group, description, exported}]), or, without a catalogue, every
## export in the generated directory (group "test").
func scene_list() -> Array[Dictionary]:
    read_catalogue()
    var out: Array[Dictionary] = []
    if not catalogue.is_empty():
        for entry in catalogue.get("scenes", []):
            var name := str(entry.get("name", ""))
            out.append({"name": name, "group": str(entry.get("group", "test")),
                "description": str(entry.get("description", "")), "exported": is_exported(name)})
        return out
    var names := {}
    var dir := DirAccess.open(generated_dir)
    if dir != null:
        for file in dir.get_files():
            if file.ends_with(".manifest.json"):
                names[file.trim_suffix(".manifest.json")] = true
        for sub in dir.get_directories():
            if sub.ends_with("_chunks") and is_exported(sub.trim_suffix("_chunks")):
                names[sub.trim_suffix("_chunks")] = true
    var sorted := names.keys()
    sorted.sort()
    for name in sorted:
        out.append({"name": name, "group": "test", "description": "", "exported": true})
    return out


## The spawn named ``name`` (the first one when ``name`` is "" or unknown), or {} with no spawns.
func spawn_named(name := "") -> Dictionary:
    for s in spawns:
        if name != "" and s.get("name", "") == name:
            return s
    return spawns[0] if not spawns.is_empty() else {}


## Load (or reload) everything requested. Returns the combined bounds.
func load_all() -> AABB:
    unload()
    for manifest in _manifests():
        _load_model(manifest)
    var aabb := world_aabb()
    world_loaded.emit(aabb)
    _report_timings()
    return aabb


## Free everything loaded (models, NPCs, traffic, trains, streamer, navmesh,
## interactions) and forget it: the world is empty afterwards.
func unload() -> void:
    load_timings = {}
    var t := Time.get_ticks_usec()
    _load_started = t
    for child in get_children():
        remove_child(child)
        child.free()
    streamer = null
    _lap("unload", t)
    _mtimes.clear()
    _next_x = 0.0
    model_aabbs.clear()
    rooms.clear()
    spawns.clear()
    interactions.clear()
    lights_by_name.clear()
    affordances.clear()
    npcs.clear()
    _auto_lights.clear()
    traffic.clear()
    trains.clear()
    crossing_closed.clear()
    traffic_graph = {}
    portals.clear()
    _reservations.clear()
    _gates.clear()
    _moving.clear()
    _target_of.clear()
    travel_points.clear()
    arm_travel()


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
    if _poll < POLL_SECONDS or loading:
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
        var chunks := "%s/%s_chunks/%s.chunks.json" % [generated_dir, scene_name, scene_name]
        if FileAccess.file_exists(chunks) and (prefer_chunks or not FileAccess.file_exists(path)):
            result.append(chunks)
        elif FileAccess.file_exists(path):
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
    var manifest := _read_manifest(manifest_path)
    if manifest.is_empty():
        return
    if manifest.get("format") == GeogenChunkStreamer.INDEX_FORMAT:
        var t := Time.get_ticks_usec()
        _load_chunked(manifest_path, manifest)
        _lap("stream_prime", t)
        return
    var t := Time.get_ticks_usec()
    var root := GeogenChunkStreamer._load_glb(_model_path(manifest_path, manifest))
    _lap("gltf_parse", t)
    _attach_model(manifest_path, manifest, root)


## Like load_all(), but single-file exports parse on a worker thread while frames keep
## drawing (a loading screen stays responsive). Streamed exports prime synchronously.
func load_all_async() -> AABB:
    loading = true
    unload()
    for manifest_path in _manifests():
        var manifest := _read_manifest(manifest_path)
        if manifest.is_empty():
            continue
        if manifest.get("format") == GeogenChunkStreamer.INDEX_FORMAT:
            await get_tree().process_frame    # show the loading screen first
            var t := Time.get_ticks_usec()
            _load_chunked(manifest_path, manifest)
            _lap("stream_prime", t)
            continue
        var t := Time.get_ticks_usec()
        var out := []
        var path := _model_path(manifest_path, manifest)
        _parse_task = WorkerThreadPool.add_task(func(): out.append(GeogenChunkStreamer._load_glb(path)))
        while not WorkerThreadPool.is_task_completed(_parse_task):
            await get_tree().process_frame
        WorkerThreadPool.wait_for_task_completion(_parse_task)
        _parse_task = -1
        _lap("gltf_parse", t)
        _attach_model(manifest_path, manifest, out[0] if not out.is_empty() else null)
    loading = false
    var aabb := world_aabb()
    world_loaded.emit(aabb)
    _report_timings()
    return aabb


func _exit_tree() -> void:
    if _parse_task >= 0:
        WorkerThreadPool.wait_for_task_completion(_parse_task)


func _read_manifest(manifest_path: String) -> Dictionary:
    _mtimes[manifest_path] = FileAccess.get_modified_time(manifest_path)
    var manifest = JSON.parse_string(FileAccess.get_file_as_string(manifest_path))
    if not manifest is Dictionary:
        push_error("geogen: bad manifest %s" % manifest_path)
        return {}
    return manifest


static func _model_path(manifest_path: String, manifest: Dictionary) -> String:
    return manifest_path.get_base_dir().path_join(manifest.get("model", ""))


## Add a parsed export (``root``, null if it failed to parse) to the world and set it up.
func _attach_model(manifest_path: String, manifest: Dictionary, root: Node3D) -> void:
    var model_path := _model_path(manifest_path, manifest)
    if root == null:
        push_error("geogen: cannot load %s" % model_path)
        return
    root.name = manifest.get("name", model_path.get_file().get_basename())
    var t := Time.get_ticks_usec()
    add_child(root)
    t = _lap("add_to_tree", t)
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
    t = Time.get_ticks_usec()
    _prepare_materials(root)
    _lap("materials", t)
    if manifest.get("traffic") is Dictionary:
        traffic_graph = manifest["traffic"]
        _traffic_offset = offset
    var summary := setup_root(root)
    var count: int = summary["meshes"]
    if open_before_bake:
        for it in interactions:
            if it.asset.is_ancestor_of(root) or root.is_ancestor_of(it.asset):
                if it.states.has("open"):
                    it.set_state("open", true)
                    it.hold = true
    if bake_navigation:
        t = Time.get_ticks_usec()
        _navigation(root, manifest_path, model_path)
        _lap("navigation", t)
    for s in manifest.get("spawns", []):
        var f: Array = s.get("forward", [0, 0, -1])
        var p: Array = s.get("position", [0, 0, 0])
        spawns.append({"name": s.get("name", ""), "position": Vector3(p[0], p[1], p[2]) + offset,
            "yaw_deg": rad_to_deg(atan2(-float(f[0]), -float(f[2]))), "model": model_aabbs.size() - 1})
    print("geogen: loaded %s (%d meshes, %d rooms, %d spawns, %d tagged)" % [
        model_path.get_file(), count, summary["rooms"], summary["spawns"], summary["tagged"]])


## Gameplay setup for a freshly loaded export (or streamed chunk) already in
## the tree: LOD levels dropped, interactions, lights, switches, seats,
## colliders, gates, rooms, then the scene builder (Areas, spawns, groups).
func setup_root(root: Node) -> Dictionary:
    var t := Time.get_ticks_usec()
    # MSFT_lod levels are detached from the hierarchy; Godot makes its own LODs.
    for node in root.find_children("*", "Node3D", true, false):
        if geogen_extras(node).get("type") == "lod":
            node.get_parent().remove_child(node)
            node.queue_free()
    _collect_interactions(root)
    _add_lights(root)
    _wire_switches(root)
    _collect_affordances(root)
    _collect_portals(root)
    _collect_travel(root)
    t = _lap("setup", t)
    var count := _add_collision(root)
    t = _lap("colliders", t)
    _spawn_npcs(root)
    t = _lap("npcs", t)
    _spawn_traffic(root)
    t = _lap("traffic", t)
    _collect_gates(root)
    _collect_rooms(root)
    var summary := GeogenSceneBuilder.build(root)
    _lap("scene_builder", t)
    summary["meshes"] = count
    return summary


## Add the time since ``since`` (Time usec) to phase ``phase`` of load_timings; returns now.
func _lap(phase: String, since: int) -> int:
    var now := Time.get_ticks_usec()
    load_timings[phase] = snappedf(float(load_timings.get(phase, 0.0)) + (now - since) / 1000.0, 0.1)
    return now


## --timings: print the phases of the last load (ms).
func _report_timings() -> void:
    if not print_timings:
        return
    # Wall clock: a streamed export's chunk setup phases also count inside stream_prime.
    var total := (Time.get_ticks_usec() - _load_started) / 1000.0
    print("load timings: %s" % JSON.stringify({"scene": scene_name, "total_ms": snappedf(total, 0.1),
        "phases": load_timings}))


# --- navigation cache ---------------------------------------------------------------------------

## Bake the model's navmesh, or reuse the one baked last time the same export loaded with the
## same settings (<generated>/.navcache/<name>-<key>.scn, keyed by the export's files, the
## player spec, the bake options and the baking scripts' source).
func _navigation(root: Node3D, manifest_path: String, model_path: String) -> void:
    var spec := PlayerSpec.from_manifest(manifest_path)
    var path := _nav_cache_path(root, manifest_path, model_path, spec)
    if path != "" and FileAccess.file_exists(path):
        var packed := ResourceLoader.load(path, "PackedScene", ResourceLoader.CACHE_MODE_IGNORE) as PackedScene
        if packed != null:
            var nav := packed.instantiate()
            root.add_child(nav)
            for region in [nav] + nav.find_children("*", "NavigationRegion3D", true, false):
                if region is NavigationRegion3D:
                    region.add_to_group("geogen_navigation")
            load_timings["nav_cache"] = "hit"
            return
    var nav := GeogenSceneBuilder.build_navigation(root, spec, open_before_bake, ground_margin)
    load_timings["nav_cache"] = "miss"
    if path == "":
        return
    for child in nav.find_children("*", "", true, false):
        child.owner = nav
    var packed := PackedScene.new()
    if packed.pack(nav) != OK:
        return
    var dir := path.get_base_dir()
    DirAccess.make_dir_recursive_absolute(dir)
    var prefix := path.get_file().get_slice("-", 0) + "-"
    for old in DirAccess.get_files_at(dir):
        if old.begins_with(prefix):
            DirAccess.remove_absolute(dir.path_join(old))    # older bakes of this export
    var err := ResourceSaver.save(packed, path, ResourceSaver.FLAG_COMPRESS)
    if err != OK:
        push_warning("geogen: can't cache navmesh %s (%s)" % [path, error_string(err)])


func _nav_cache_path(root: Node3D, manifest_path: String, model_path: String, spec: PlayerSpec) -> String:
    if not nav_cache:
        return ""
    var model_md5 := FileAccess.get_md5(model_path)
    if model_md5 == "":
        return ""
    var key := JSON.stringify([NAV_CACHE_VERSION, ProjectSettings.globalize_path(model_path),
        model_md5, FileAccess.get_md5(manifest_path),
        spec.to_dict() if spec else {}, open_before_bake, ground_margin, var_to_str(root.global_transform),
        FileAccess.get_md5("res://addons/geogen/scene_builder.gd"), FileAccess.get_md5("res://scripts/world_loader.gd")])
    return generated_dir.path_join(".navcache").path_join("%s-%s.scn" % [String(root.name).replace("-", "_"),
        key.md5_text()])


## Unregister everything ``root`` contributed (before freeing a streamed chunk).
func forget(root: Node) -> void:
    var inside := func(node) -> bool:
        return node is Node and is_instance_valid(node) and (node == root or root.is_ancestor_of(node))
    for it in interactions.duplicate():
        if inside.call(it.asset):
            interactions.erase(it)
            it.queue_free()
    rooms = rooms.filter(func(r): return not inside.call(r.get("node")))
    affordances = affordances.filter(func(a): return not inside.call(a.get("node")))
    portals = portals.filter(func(p): return not inside.call(p.get("node")))
    travel_points = travel_points.filter(func(t): return not inside.call(t.get("node")))
    npcs = npcs.filter(func(n): return not inside.call(n))
    _gates = _gates.filter(func(g): return not inside.call(g.get("node")))
    _auto_lights = _auto_lights.filter(func(a): return not inside.call(a.get("fixture")))
    for key in lights_by_name.keys():
        if inside.call(lights_by_name[key]):
            lights_by_name.erase(key)
    for key in _moving.keys():
        if inside.call(key):
            _moving.erase(key)
    for key in _target_of.keys():
        if inside.call(key):
            _target_of.erase(key)


func _load_chunked(index_path: String, index: Dictionary) -> void:
    streamer = GeogenChunkStreamer.new()
    streamer.name = "ChunkStreamer"
    streamer.world = self
    if stream_radii.size() >= 3:
        streamer.full_radius = stream_radii[0]
        streamer.lod_radius = stream_radii[1]
        streamer.interior_radius = stream_radii[2]
    add_child(streamer)
    if index.get("traffic") is Dictionary:
        traffic_graph = index["traffic"]
    streamer.open(index_path, index)
    if prime_focus != null:
        stream_focus = prime_focus
    elif not index.get("spawns", []).is_empty():
        var first: Dictionary = index["spawns"][0]
        for s in index["spawns"]:
            if prime_spawn != "" and s.get("name", "") == prime_spawn:
                first = s
        var p: Array = first.get("position", [0, 0, 0])
        stream_focus = Vector3(p[0], p[1], p[2])
    streamer.prime(stream_focus)
    model_aabbs.append(streamer.bounds)
    for s in index.get("spawns", []):
        var f: Array = s.get("forward", [0, 0, -1])
        var p: Array = s.get("position", [0, 0, 0])
        spawns.append({"name": s.get("name", ""), "position": Vector3(p[0], p[1], p[2]),
            "yaw_deg": rad_to_deg(atan2(-float(f[0]), -float(f[2])))})
    print("geogen: streaming %s (%d chunks)" % [index_path.get_file(), streamer.chunks.size()])


## Report of what the streamer has loaded, or {} for single-file exports.
func stream_report() -> Dictionary:
    return streamer.report() if streamer != null else {}


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
        if spec.get("auto") == "night":
            # Street lamps: many of them, so no shadows; on when the clock says it's dark.
            light.shadow_enabled = false
            _auto_lights.append({"light": light, "fixture": node})
            light.visible = night
            _set_emission(node, night)
        lights_by_name[String(node.name)] = light
        # The fixture itself mustn't shadow its own light.
        for mi: MeshInstance3D in node.find_children("*", "MeshInstance3D", true, false):
            mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF


## Light switches: their interaction's on/off state shows or hides the named light.
func _wire_switches(root: Node) -> void:
    for it in interactions:
        if not root.is_ancestor_of(it.asset):
            continue   # wired when its own export loaded
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


## Ways out of the scene for NPCs going `away` (routines): building entrances (their spawns).
func exit_points() -> Array[Vector3]:
    var out: Array[Vector3] = []
    for s in spawns:
        if String(s.get("name", "")).begins_with("entrance"):
            out.append(s["position"])
    return out


## Night on/off: `auto: night` lights and their glowing glass, and vehicle lamps.
func set_night(on: bool) -> void:
    night = on
    for a in _auto_lights:
        if is_instance_valid(a["light"]):
            a["light"].visible = on
            _set_emission(a["fixture"], on)
    for t in traffic:
        if is_instance_valid(t):
            t.set_night(on)
    for t in trains:
        if is_instance_valid(t):
            t.set_night(on)


## A train closed or opened a level crossing (road traffic stops for closed ones).
func set_crossing(id: String, closed: bool) -> void:
    crossing_closed[id] = closed


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
        for i in list.size():
            var a: Dictionary = list[i]
            var p: Array = a.get("position", [0, 0, 0])
            var forward := xform.basis * Basis(Vector3.UP, deg_to_rad(float(a.get("yaw", 0.0)))) * Vector3.BACK
            var ap: Array = a.get("approach", p)
            var entry := {"type": str(a.get("type", "sit")), "position": xform * Vector3(p[0], p[1], p[2]),
                "yaw_deg": rad_to_deg(atan2(-forward.x, -forward.z)), "node": node,
                "height": float(a.get("height", p[1])), "asset": String(node.name),
                # NPCs (npc.gd): heading of the actor (+Z forward), approach point and the rest.
                "id": "%s#%d" % [node.name, i], "npc_yaw_deg": rad_to_deg(atan2(forward.x, forward.z)),
                "action": str(a.get("action", a.get("type", "sit"))), "approach": xform * Vector3(ap[0], ap[1], ap[2]),
                "advertises": a.get("advertises", {}), "tags": a.get("tags", []), "slots": int(a.get("slots", 1))}
            if a.has("duration"):
                entry["duration"] = a["duration"]
            if a.has("depth"):
                entry["depth"] = float(a["depth"])     # seat front edge ahead of the anchor
            if a.has("interaction"):
                for it in interactions_of(String(node.name)):
                    if it.interaction_name == a["interaction"] and it.asset == node:
                        entry["interaction"] = it
            affordances.append(entry)


## Doors' extras.geogen.portal, in world space, with their interaction.
func _collect_portals(root: Node) -> void:
    for node in root.find_children("*", "Node3D", true, false):
        var p = geogen_extras(node).get("portal")
        if not p is Dictionary:
            continue
        var it: GeogenInteraction = null
        for candidate in interactions:
            if candidate.asset == node and candidate.interaction_name == p.get("interaction", ""):
                it = candidate
        if it == null:
            push_warning("geogen: portal %s: no interaction '%s'" % [node.name, p.get("interaction", "")])
            continue
        var xform := (node as Node3D).global_transform
        var c: Array = p.get("center", [0, 1, 0])
        var n: Array = p.get("normal", [0, 0, 1])
        portals.append({"node": node, "interaction": it, "open": str(p.get("open", "open")),
            "closed": str(p.get("closed", "closed")), "center": xform * Vector3(c[0], c[1], c[2]),
            "normal": (xform.basis * Vector3(n[0], n[1], n[2])).normalized(),
            "width": float(p.get("width", 0.9)), "height": float(p.get("height", 2.0)),
            "depth": float(p.get("depth", 0.3)), "clearance": float(p.get("clearance", 0.9))})


## Travel points: on: use listens for its "travel" interaction's event, on: enter
## gets an Area3D over its volume that fires when the player walks in.
func _collect_travel(root: Node) -> void:
    for node in root.find_children("*", "Node3D", true, false):
        var travel = geogen_extras(node).get("travel")
        if not travel is Dictionary or node.has_meta("geogen_travel"):
            continue
        node.set_meta("geogen_travel", true)
        travel_points.append({"node": node, "travel": travel})
        if travel.get("on", "use") == "use":
            for it in interactions:
                if it.asset == node and it.interaction_name == "travel":
                    it.state_entered.connect(func(_state: String, event: String):
                        if event == "travel":
                            travel_requested.emit(travel))
            continue
        var volume: Dictionary = travel.get("volume", {})
        var c: Array = volume.get("center", [0, 1, 0])
        var sz: Array = volume.get("size", [1, 2, 1])
        var area := Area3D.new()
        area.name = "TravelVolume"
        area.collision_layer = 0
        area.collision_mask = 1
        area.monitorable = false
        var shape := CollisionShape3D.new()
        var box := BoxShape3D.new()
        box.size = Vector3(sz[0], sz[1], sz[2])
        shape.shape = box
        shape.position = Vector3(c[0], c[1], c[2])
        area.add_child(shape)
        node.add_child(area)
        area.body_entered.connect(func(body: Node3D):
            if body is Player and Engine.get_physics_frames() >= travel_armed_at:
                travel_requested.emit(travel))


## Keep walk-in travel volumes quiet for a moment (after loading, or moving the player).
func arm_travel() -> void:
    travel_armed_at = Engine.get_physics_frames() + TRAVEL_ARM_FRAMES


## Nodes with extras.geogen type npc become GeogenNpc bodies.
func _spawn_npcs(root: Node) -> void:
    var spec := player_spec()
    for node in root.find_children("*", "Node3D", true, false):
        var g := geogen_extras(node)
        if g.get("type") != "npc" or not g.get("npc") is Dictionary or node.has_meta("geogen_npc"):
            continue
        var npc := GeogenNpc.spawn(self, node, g["npc"])
        npc.trace_enabled = npc_trace
        if spec != null:
            npc.step_height = spec.step_height
        npcs.append(npc)


## Nodes with extras.geogen type traffic drive their vehicles over the lane graph (traffic.gd).
func _spawn_traffic(root: Node) -> void:
    for node in root.find_children("signal_*", "Node3D", true, false):
        if geogen_extras(node).has("signal"):
            node.add_to_group("geogen_signal")
    for node in root.find_children("*", "Node3D", true, false):
        if not is_instance_valid(node):      # a vehicle's collider, freed as its traffic was set up
            continue
        var g := geogen_extras(node)
        if g.get("type") != "traffic" or not g.get("fleet") is Dictionary or node.has_meta("geogen_traffic"):
            continue
        node.set_meta("geogen_traffic", true)
        if traffic_graph.is_empty():
            push_warning("geogen: traffic node %s but the export has no lane graph" % node.name)
            continue
        var t := GeogenTraffic.spawn(self, node, g["fleet"], traffic_graph, _traffic_offset)
        t.trace_enabled = traffic_trace
        t.set_night(night)
        traffic.append(t)
    for node in root.find_children("*", "Node3D", true, false):
        if not is_instance_valid(node):
            continue
        var g := geogen_extras(node)
        if g.get("type") != "train" or not g.get("train") is Dictionary or node.has_meta("geogen_train"):
            continue
        node.set_meta("geogen_train", true)
        var rail_id := str(g["train"].get("railway", ""))
        var railway := {}
        for r in traffic_graph.get("railways", []):
            if str(r.get("id", "")) == rail_id:
                railway = r
        if railway.is_empty():
            push_warning("geogen: train %s: no railway '%s' in the export" % [node.name, rail_id])
            continue
        var train := GeogenTrain.spawn(self, node, g["train"], railway, _traffic_offset)
        train.trace_enabled = traffic_trace
        train.set_night(night)
        trains.append(train)


## How many NPCs other than ``npc`` hold affordance ``id``.
func npc_reserved(id: String, npc: Node) -> int:
    var holders: Array = _reservations.get(id, []).filter(func(n): return is_instance_valid(n) and n != npc)
    return holders.size()


func npc_reserve(id: String, npc: Node) -> void:
    var holders: Array = _reservations.get(id, [])
    if not npc in holders:
        holders.append(npc)
    _reservations[id] = holders


func npc_release(id: String, npc: Node) -> void:
    if _reservations.has(id):
        _reservations[id].erase(npc)


## Nearest affordance within ``radius`` of a world point, or {}.
## Only poses the player can take (sit, lie); NPC-only actions (look, stand) are skipped.
func affordance_near(point: Vector3, radius := 0.9) -> Dictionary:
    var best := {}
    var best_d := radius
    for a in affordances:
        if not a["type"] in ["sit", "lie"]:
            continue
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
                "xform": (node as Node3D).global_transform, "size": Vector3(size[0], size[1], size[2]),
                "node": node})


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
## Surfaces with vertex colours get them as an albedo tint.
func _prepare_materials(root: Node) -> void:
    var done := {}
    for mi: MeshInstance3D in root.find_children("*", "MeshInstance3D", true, false):
        if mi.mesh == null:
            continue
        for surface in mi.mesh.get_surface_count():
            var mat := mi.mesh.surface_get_material(surface) as BaseMaterial3D
            if mat == null:
                continue
            # COLOR_0 tints (character skin tone, hair colour) multiply the albedo; meshes
            # without colours read white, so a shared material can always enable it.
            if mi.mesh.surface_get_format(surface) & Mesh.ARRAY_FORMAT_COLOR:
                mat.vertex_color_use_as_albedo = true
            if done.has(mat):
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
    var shapes := {}   # [mesh, convex] -> Shape3D: instances share their collider mesh, so share the shape
    for mi in colliders:
        if mi.mesh != null:
            var convex: bool = String(mi.name).ends_with("-convcolonly") or geogen_extras(mi).get("shape") in ["box", "hull"]
            var key := [mi.mesh, convex]
            var shape: Shape3D = shapes.get(key)
            if shape == null:
                shape = mi.mesh.create_convex_shape(true, false) if convex else mi.mesh.create_trimesh_shape()
                if shape == null:   # a flat or degenerate hull: fall back to the exact triangles
                    shape = mi.mesh.create_trimesh_shape()
                shapes[key] = shape
            if shape != null:
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
    var body: PhysicsBody3D
    if _is_moving(parent):
        var animatable := AnimatableBody3D.new()
        # Moved by its parent part, not by itself: sync_to_physics would only
        # track the body's own transform and leave the collider behind.
        animatable.sync_to_physics = false
        body = animatable
    else:
        body = StaticBody3D.new()
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
