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
## An actor entered or left a named place or a room: {type: enter | leave, subject
## (the place's or room's stable ID), place (its id), kind: place | room, actor
## ("player" or the NPC's stable ID)}.
signal place_event(event: Dictionary)

## Everything that happens, as one stream (events.gd): the signals above are views of it.
var events := GeogenEvents.new()
## The loaded package's rules program, running (rules.gd): state, rules, quests.
var rules := GeogenRules.new()
## "<uid>#<interaction>" -> GeogenInteraction, for the ones that are loaded.
var interactions_by_key := {}

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
## Named places and rooms with an area: {id, uid, kind: place | room, node, area, actors: {actor: true}}.
var places: Array[Dictionary] = []
var _held: Array[Dictionary] = []   # {entry, body, actor}: occupants whose body stopped colliding (seated, lying)
## Spawn points from the manifests: [{name, position: Vector3, yaw_deg: float}]
var spawns: Array[Dictionary] = []
## Loaded nodes by stable ID (extras.geogen.uid: "<package>:<authored path>", geogen/refs.py).
## Content and saved state name nodes by these, never by exported node names.
var nodes_by_uid: Dictionary = {}
## Saved interaction state for things that aren't loaded right now (streamed chunks), by save key.
var _pending_state: Dictionary = {}
## Fixture lights by their fixture node's name (light switches refer to them).
var lights_by_name := {}
## Switchable fixtures by stable ID: {uid: {light: OmniLight3D, fixture: Node}}. Their `<uid>#lit`
## state (rules: a switch's binding, or any rule) shows and hides the light and its glow.
var _fixtures := {}
## Actor poses from extras.geogen.affordances, world space:
## [{type, position: Vector3, yaw_deg: float, height: float, node: Node3D}]
var affordances: Array[Dictionary] = []
## NPCs spawned from extras.geogen.npc nodes (see npc.gd).
var npcs: Array[GeogenNpc] = []
## Print every NPC decision and step (--npc-trace).
var npc_trace := false
var npc_limit := -1                 # --npc-limit: spawn at most this many NPCs (crowd measurements); -1 all
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
## Physics props (extras.geogen.physics): rigid bodies, carrying, spawn and stow.
var props: GeogenProps
var _moving := {}     # Node3D driven by an interaction -> true
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
    events.name = "Events"
    add_child(events)
    rules.name = "Rules"
    rules.reset.connect(_sync_trains)
    rules.reset.connect(_sync_fixtures)
    events.posted.connect(_on_lit)
    rules.world = self
    add_child(rules)
    props = GeogenProps.new(self)
    add_child(props)
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


## The spawn ``name`` (the first one when ``name`` is "" or unknown), or {} with no spawns.
## ``name`` is a spawn's stable ID, its authored path in this scene (street/hotel/entrance_spawn)
## or just its name (the shallowest spawn called that).
func spawn_named(name := "") -> Dictionary:
    if name != "":
        for s in spawns:
            var uid := str(s.get("uid", ""))
            if uid != "" and (uid == name or uid == "%s:%s" % [scene_name, name]):
                return s
        for s in spawns:
            if s.get("name", "") == name:
                return s
    return spawns[0] if not spawns.is_empty() else {}


## The loaded node with stable ID ``uid`` ("cottage:door", or "door" in the current scene), or null.
func node_by_uid(uid: String) -> Node:
    var node = nodes_by_uid.get(uid)
    if node == null and not ":" in uid:
        node = nodes_by_uid.get("%s:%s" % [scene_name, uid])
    return node if node != null and is_instance_valid(node) else null


## The stable ID of ``node`` ("" if it has none).
static func uid_of(node: Node) -> String:
    return str(geogen_extras(node).get("uid", ""))


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
    rules.remember_reports()      # for the quest log, while the interactions it counts are still here
    for child in get_children():
        if child == events or child == rules or child == props:
            continue
        remove_child(child)
        child.free()
    props.clear()
    rules.unload()
    interactions_by_key.clear()
    streamer = null
    _lap("unload", t)
    _mtimes.clear()
    _next_x = 0.0
    model_aabbs.clear()
    rooms.clear()
    places.clear()
    _held.clear()
    spawns.clear()
    nodes_by_uid.clear()
    _pending_state.clear()
    interactions.clear()
    lights_by_name.clear()
    _fixtures.clear()
    affordances.clear()
    npcs.clear()
    _auto_lights.clear()
    traffic.clear()
    trains.clear()
    crossing_closed.clear()
    traffic_graph = {}
    portals.clear()
    _reservations.clear()
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
    events.package = scene_name if scene_name != "" else String(root.name)
    rules.load_package(events.package, manifest)
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
        spawns.append({"name": s.get("name", ""), "uid": s.get("uid", ""),
            "position": Vector3(p[0], p[1], p[2]) + offset,
            "yaw_deg": rad_to_deg(atan2(-float(f[0]), -float(f[2]))), "model": model_aabbs.size() - 1})
    print("geogen: loaded %s (%d meshes, %d rooms, %d spawns, %d tagged)" % [
        model_path.get_file(), count, summary["rooms"], summary["spawns"], summary["tagged"]])


## Gameplay setup for a freshly loaded export (or streamed chunk) already in
## the tree: LOD levels dropped, interactions, lights, switches, seats,
## colliders, rooms, then the scene builder (Areas, spawns, groups).
func setup_root(root: Node) -> Dictionary:
    var t := Time.get_ticks_usec()
    # MSFT_lod levels are detached from the hierarchy; Godot makes its own LODs.
    for node in root.find_children("*", "Node3D", true, false):
        if geogen_extras(node).get("type") == "lod":
            node.get_parent().remove_child(node)
            node.queue_free()
    for node in [root] + root.find_children("*", "Node3D", true, false):
        var uid := uid_of(node)
        if uid != "":
            nodes_by_uid[uid] = node
    _collect_interactions(root)
    props.setup(root)        # before colliders: a prop's parts' colliders give way to its body
    _add_lights(root)
    _sync_fixtures()
    _collect_affordances(root)
    _collect_portals(root)
    _collect_travel(root)
    _apply_water(root)
    t = _lap("setup", t)
    var count := _add_collision(root)
    t = _lap("colliders", t)
    _spawn_npcs(root)
    t = _lap("npcs", t)
    _spawn_traffic(root)
    t = _lap("traffic", t)
    _collect_rooms(root)
    var summary := GeogenSceneBuilder.build(root)
    _collect_places(root)
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
            # Remembered until its chunk loads again: by the rules program, which reads and
            # drives it meanwhile, else here.
            if uid_of(it.asset) != "" and not rules.release(_state_key(it), it.snapshot(), it.progress()):
                _pending_state[_state_key(it)] = it.snapshot()
            interactions.erase(it)
            it.queue_free()
    props.forget(root)
    for uid in nodes_by_uid.keys():
        if inside.call(nodes_by_uid[uid]):
            nodes_by_uid.erase(uid)
    rooms = rooms.filter(func(r): return not inside.call(r.get("node")))
    places = places.filter(func(p): return not inside.call(p.get("node")))
    _held = _held.filter(func(h): return not inside.call(h["entry"].get("node")))
    affordances = affordances.filter(func(a): return not inside.call(a.get("node")))
    portals = portals.filter(func(p): return not inside.call(p.get("node")))
    travel_points = travel_points.filter(func(t): return not inside.call(t.get("node")))
    npcs = npcs.filter(func(n): return not inside.call(n))
    _auto_lights = _auto_lights.filter(func(a): return not inside.call(a.get("fixture")))
    for key in lights_by_name.keys():
        if inside.call(lights_by_name[key]):
            lights_by_name.erase(key)
    for key in _fixtures.keys():
        if inside.call(_fixtures[key]["fixture"]):
            _fixtures.erase(key)
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
    events.package = scene_name
    rules.load_package(scene_name, index)
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
        spawns.append({"name": s.get("name", ""), "uid": s.get("uid", ""), "position": Vector3(p[0], p[1], p[2]),
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
        else:
            var uid := uid_of(node)
            if uid != "":
                _fixtures[uid] = {"light": light, "fixture": node}
        lights_by_name[String(node.name)] = light
        # The fixture itself mustn't shadow its own light.
        for mi: MeshInstance3D in node.find_children("*", "MeshInstance3D", true, false):
            mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF


## A fixture's `lit` state (state.light_state) shows or hides its light and glow: a switch's binding
## sets it, and so may any rule. Synced when a package (or a save) loads and when a chunk adds fixtures.
func _sync_fixtures() -> void:
    for uid in _fixtures:
        var lit = rules.get_value(uid + "#lit")
        if lit is bool:
            _light_fixture(uid, lit)


func _on_lit(event: Dictionary) -> void:
    if event["type"] == "changed" and event["name"] == "lit" and _fixtures.has(event["subject"]):
        _light_fixture(event["subject"], bool(event["data"].get("to", true)))


func _light_fixture(uid: String, lit: bool) -> void:
    var entry: Dictionary = _fixtures[uid]
    if not is_instance_valid(entry["fixture"]):
        return
    entry["light"].visible = lit
    _set_emission(entry["fixture"], lit)


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
                            _request_travel(node, travel, it.actor))
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
                _request_travel(node, travel, "player"))


func _request_travel(node: Node, travel: Dictionary, actor: String) -> void:
    events.post("travel", uid_of(node), str(travel.get("scene", scene_name)), actor if actor != "" else "player",
        {"spawn": travel.get("spawn", ""), "on": travel.get("on", "use")})
    travel_requested.emit(travel)


## Nodes with extras.geogen.water get the animated water shader (scripts/water.gdshader),
## tuned by their extras (colours, depth_scale, wave_scale, wave_speed, foam).
const WATER_SHADER := preload("res://scripts/water.gdshader")
func _apply_water(root: Node) -> void:
    for node in root.find_children("*", "Node3D", true, false):
        var spec = geogen_extras(node).get("water")
        if not spec is Dictionary:
            continue
        var mat := ShaderMaterial.new()
        mat.shader = WATER_SHADER
        for key in ["shallow_color", "deep_color"]:
            if spec.has(key):
                var c: Array = spec[key]
                mat.set_shader_parameter(key, Color(c[0], c[1], c[2]))
        for key in ["depth_scale", "wave_scale", "wave_speed", "foam"]:
            if spec.has(key):
                mat.set_shader_parameter(key, float(spec[key]))
        var meshes: Array = [node] if node is MeshInstance3D else []
        meshes.append_array(node.get_children().filter(func(c): return c is MeshInstance3D and not _is_collider(c)))
        for mi: MeshInstance3D in meshes:
            mi.material_override = mat
            mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF


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
        if npc_limit >= 0 and npcs.size() >= npc_limit:
            node.get_parent().remove_child(node)
            node.queue_free()
            continue
        var npc := GeogenNpc.spawn(self, node, g["npc"])
        if uid_of(node) != "":
            npc.set_meta("geogen_uid", uid_of(node))
        npc.trace_enabled = npc_trace
        var actor := actor_id(npc)
        npc.traced.connect(func(event: Dictionary): events.post("npc", actor, str(event.get("event", "")), actor, event))
        npc.greeted.connect(func(signal_name: String, by: String): events.post("signal", actor, signal_name, by))
        rules.add_actor(actor, "npc")
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
        train.uid = uid_of(node)
        train.standing.connect(func(station: String, here: bool): _train_standing(train, station, here))
        trains.append(train)


## A train stopped at a station or left it: its `at_<station>` state (declared with
## `driven: train`), which rules and quests read. Stations it doesn't declare are skipped.
func _train_standing(train: GeogenTrain, station: String, here: bool) -> void:
    var key := "%s#at_%s" % [train.uid, station]
    if rules.machine != null and rules.machine.variables.has(key):
        rules.set_value(key, here)


## After the rules program is loaded or a save is read: the trains say where they stand.
func _sync_trains() -> void:
    for train in trains:
        if not is_instance_valid(train):
            continue
        for stop in train.data.get("stops", []):
            _train_standing(train, str(stop["station"]), train.standing_at.has(str(stop["station"])))


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


## The key an interaction's state is saved under: "<stable ID>#<interaction>", so a save
## survives a re-export that renames nodes (exports without IDs: "<node name>/<interaction>").
const SAVE_FORMAT := "geogen-save"       # docs/schema/geogen-save.v1.schema.json
const RULES_SAVE := "geogen:rules"       # where saves from before that format kept the rules program's state


static func _state_key(it: GeogenInteraction) -> String:
    var uid := uid_of(it.asset)
    if uid == "":
        return "%s/%s" % [it.asset.name, it.interaction_name]
    return "%s#%s" % [uid, it.interaction_name]


## {state, locked} of every interaction, by ``_state_key``: the loaded ones, plus what was
## last known of those in chunks that aren't loaded.
func interaction_states() -> Dictionary:
    var data := _pending_state.duplicate(true)
    for it in interactions:
        data[_state_key(it)] = it.snapshot()
    return data


## The game as a save (docs/rules.md "Saving"): the rules program's world and actor sections,
## clock and timers, with every interaction in ``world`` as the instance state it is ("<key>":
## its state, "<key>_locked": its lock). Only what differs from how things start is there.
func save_state() -> Dictionary:
    var data := rules.save()
    if data.is_empty():                  # no rules program: only interactions
        data = {"format": SAVE_FORMAT, "version": 1, "now": 0.0, "clock": _clock_now(), "world": {},
            "actors": {}, "actor_info": {}, "timers": []}
    data["package"] = rules.package if rules.machine != null else scene_name
    data.merge(rules.save_others())      # the other packages visited, and content packages' values
    var stored: Dictionary = data["world"]
    var live := _live_interactions()
    var states := interaction_states()
    for key in states:
        var snap: Dictionary = states[key]
        var it: GeogenInteraction = live.get(key)
        _store(stored, key, snap.get("state"), it.initial if it != null else _default_of(key))
        _store(stored, key + "_locked", snap.get("locked"),
            it.initial_locked if it != null else _default_of(key + "_locked"))
    # How far loaded interactions had got (a door on its way, one waiting to close itself), as
    # the timers the rules program keeps for the ones it holds.
    var timers: Array = data.get("timers", [])
    var now := float(data.get("now", 0.0))
    var seq := 0
    for timer in timers:
        seq = maxi(seq, int(timer["seq"]))
    for key in live:
        var progress: Dictionary = live[key].progress()
        if progress.is_empty():
            continue
        seq += 1
        var timer := {"at": snappedf(now + float(progress["in"]), 0.000001), "seq": seq,
            "kind": progress["kind"], "key": key, "state": progress["state"], "actor": progress["actor"]}
        if progress.has("from"):
            timer["from"] = progress["from"]
        timers.append(timer)
    timers.sort_custom(GeogenMachine._sooner)
    data["timers"] = timers
    var bodies := props.save()
    if not bodies.is_empty():
        data["bodies"] = bodies
    return data


func _live_interactions() -> Dictionary:
    var live := {}
    for it in interactions:
        live[_state_key(it)] = it
    return live


static func _store(stored: Dictionary, key: String, value: Variant, default: Variant) -> void:
    if value == null or (default != null and value == default):
        stored.erase(key)
    else:
        stored[key] = value


## What the package says a variable starts as (null: it doesn't say).
func _default_of(key: String) -> Variant:
    if rules.machine == null:
        return null
    return rules.machine.variables.get(key, {}).get("default")


func _clock_now() -> Dictionary:
    if clock == null:
        return {"time": "12:00", "hour": 12.0, "night": false, "day": 0}
    var total := int(clock.hours * 60.0) % 1440
    return {"time": "%02d:%02d" % [total / 60, total % 60], "hour": total / 60.0, "night": clock.night, "day": 0}


## Restore a save: whatever it doesn't mention goes back to how it started. Entries for
## things that aren't loaded yet apply when they load. A save from before the format (a flat
## {key: {state, locked}}, the rules state under "geogen:rules") still loads; it only moves
## what it names.
func load_state(data: Dictionary) -> void:
    if data.get("format") != SAVE_FORMAT:
        _pending_state = {}
        for key in data:
            if key != RULES_SAVE and data[key] is Dictionary:
                _pending_state[key] = data[key].duplicate(true)
        for it in interactions:
            _restore_pending(it)
        if data.get(RULES_SAVE) is Dictionary:
            rules.load_save(data[RULES_SAVE])
        return
    # Every package it holds: the loaded one's part is restored here, the rest wait for their
    # scene. A save made in another scene loads this one as that save left it.
    data = rules.take_save(data, rules.package if rules.machine != null else scene_name)
    if clock != null and data.get("clock") is Dictionary and data["clock"].has("hour"):
        clock.hours = fposmod(float(data["clock"]["hour"]), 24.0)
        clock.apply()
    # The interactions in the world section. A loaded one is its node's to hold; the others
    # wait for their chunk, and the rules program simulates them meanwhile.
    var live := _live_interactions()
    var for_rules: Dictionary = data.duplicate(true)
    var stored: Dictionary = for_rules.get("world", {})
    _pending_state = {}
    for key in stored.keys():
        var owner_key := _saved_interaction(key, stored[key])
        if owner_key == "":
            continue
        if not _pending_state.has(owner_key):
            _pending_state[owner_key] = {}
        _pending_state[owner_key]["state" if owner_key == key else "locked"] = stored[key]
        if live.has(owner_key) or rules.machine == null or not rules.machine.variables.has(key):
            stored.erase(key)
    for key in live:
        var it: GeogenInteraction = live[key]
        var snap: Dictionary = _pending_state.get(key, {})
        _pending_state.erase(key)
        it.restore({"state": snap.get("state", it.initial), "locked": snap.get("locked", it.initial_locked)})
    props.restore(data.get("bodies", {}))     # after their spawn interactions: where they had got to
    # Loaded interactions carry on from their own timers; the program keeps the rest (rule
    # timers, and those of interactions it holds: ones it doesn't know are dropped).
    var kept := []
    var now := float(for_rules.get("now", 0.0))
    for timer in for_rules.get("timers", []):
        var key := str(timer.get("key", ""))
        if timer.get("kind") in ["arrive", "dwell"] and live.has(key):
            live[key].resume(timer, float(timer["at"]) - now)
        elif timer.get("kind") == "rule" or (rules.machine != null and rules.machine.interactions.has(key)):
            kept.append(timer)
    for_rules["timers"] = kept
    rules.load_save(for_rules)           # after the nodes: bindings are recomputed from them
    rules.hold_absent()                  # the program has the rest until their chunks load
    for key in _pending_state.keys():
        if rules.holds(key):
            _pending_state.erase(key)


## Write a save as one JSON file; "" when done, else why not.
static func write_save(path: String, data: Dictionary) -> String:
    var f := FileAccess.open(path, FileAccess.WRITE)
    if f == null:
        return "can't write %s: %s" % [path, error_string(FileAccess.get_open_error())]
    f.store_string(JSON.stringify(data, "  "))
    f.close()
    return ""


## The save in a file (a Dictionary), or null.
static func read_save(path: String) -> Variant:
    if not FileAccess.file_exists(path):
        return null
    return JSON.parse_string(FileAccess.get_file_as_string(path))


## The interaction a key of a save's world section belongs to ("" for a variable or a
## latch): the key itself for its state, the key less "_locked" for its lock.
func _saved_interaction(key: String, value: Variant) -> String:
    var base := key.trim_suffix("_locked")
    if rules.machine != null:
        if rules.machine.interactions.has(key):
            return key
        if base != key and rules.machine.interactions.has(base):
            return base
        if rules.machine.variables.has(key):
            return ""
    if key.begins_with("world:") or key.begins_with("once:") or key.begins_with("watch:"):
        return ""
    for it in interactions:              # an export without stable IDs or a state section
        var own := _state_key(it)
        if own == key or own == base:
            return own
    if rules.machine != null:
        return ""
    return base if value is bool and base != key else key


func _restore_pending(it: GeogenInteraction) -> void:
    for key in [_state_key(it), "%s/%s" % [it.asset.name, it.interaction_name]]:   # the second: older saves
        if _pending_state.has(key):
            it.restore(_pending_state[key])
            _pending_state.erase(key)
            return


func _collect_rooms(root: Node) -> void:
    for node in root.find_children("*", "Node3D", true, false):
        var g := geogen_extras(node)
        if g.get("type") == "room_volume":
            var size: Array = g.get("size", [0, 0, 0])
            var room: Dictionary = g.get("room", {})
            rooms.append({"id": room.get("id", node.name), "type": room.get("type", ""), "nav": g.get("nav", true),
                "xform": (node as Node3D).global_transform, "size": Vector3(size[0], size[1], size[2]),
                "node": node})


## Place and room areas (made by the scene builder) report the actors that walk
## in and out: the player and NPCs.
func _collect_places(root: Node) -> void:
    for area: Area3D in root.find_children("*", "Area3D", true, false):
        var is_room := area.is_in_group("geogen_room")
        if not (is_room or area.is_in_group("geogen_place")) or area.has_meta("geogen_wired"):
            continue
        area.set_meta("geogen_wired", true)
        var node := area.get_parent()
        var entry := {"kind": "room" if is_room else "place", "node": node, "area": area, "actors": {}}
        if is_room:      # the room is the volume's parent; rules name it, not its volume
            var room: Dictionary = area.get_meta("geogen_room", {})
            entry["id"] = str(room.get("id", node.name))
            entry["uid"] = uid_of(node.get_parent()) if node.get_parent() != null else ""
            if entry["uid"] == "":
                entry["uid"] = uid_of(node)
        else:
            var place: Dictionary = area.get_meta("geogen_place", {})
            entry["id"] = str(place.get("id", node.name))
            entry["uid"] = str(place.get("uid", ""))
            entry["detect"] = place.get("detect", ["actors"])
            entry["things"] = {}
        places.append(entry)
        area.body_entered.connect(func(body: Node3D): _place_crossed(entry, body, true))
        area.body_exited.connect(func(body: Node3D): _place_crossed(entry, body, false))


func _place_crossed(entry: Dictionary, body: Node3D, entered: bool) -> void:
    var actor := actor_id(body)
    if actor == "":
        var prop := props.prop_of(body)
        if prop != "" and "props" in entry.get("detect", []):
            _occupy(entry, prop, entered)
        return
    if entered:
        # Bodies sit at the origin for a frame while a world loads or they spawn:
        # an arrival counts once the body is still inside two frames on.
        for i in 2:
            await get_tree().physics_frame
        var area = entry["area"]
        if not is_instance_valid(body) or not is_instance_valid(area) or not area.overlaps_body(body) \
                or entry["actors"].has(actor):
            return
        entry["actors"][actor] = true
    elif not entry["actors"].has(actor):
        return
    elif _parked(body):
        _held.append({"entry": entry, "body": body, "actor": actor})
        return
    else:
        entry["actors"].erase(actor)
    _post_place("enter" if entered else "leave", entry, actor)


## Seated and lying actors turn their collision shape off, which an area reads as
## walking out: they stay in the place until the shape is back (or they vanish).
func _parked(body: Node3D) -> bool:
    if not is_instance_valid(body) or not body.is_inside_tree() or not body.visible:
        return false
    for child in body.get_children():
        if child is CollisionShape3D and not child.disabled:
            return false
    return true


func _release_held() -> void:
    for held in _held.duplicate():
        if _parked(held["body"]):
            continue
        _held.erase(held)
        _leave_unless_inside(held)


func _leave_unless_inside(held: Dictionary) -> void:
    for i in 2:      # the area sees a shape that came back on the next physics frame
        await get_tree().physics_frame
    var entry: Dictionary = held["entry"]
    var body = held["body"]
    var area = entry["area"]
    if is_instance_valid(body) and is_instance_valid(area) and body.visible and area.overlaps_body(body):
        return
    if entry["actors"].erase(held["actor"]):
        _post_place("leave", entry, held["actor"])


func _post_place(type: String, entry: Dictionary, actor: String) -> void:
    if "props" in entry.get("detect", []):
        _occupy(entry, actor, type == "enter")
    events.post(type, entry["uid"], "", actor, {"place": entry["id"], "kind": entry["kind"]})
    place_event.emit({"type": type, "subject": entry["uid"], "place": entry["id"], "kind": entry["kind"],
        "actor": actor})


## A place that detects props counts what is in it (actors and props) as its
## `occupants` (declared with `driven: place`), which rules read.
func _occupy(entry: Dictionary, thing: String, inside: bool) -> void:
    var things: Dictionary = entry["things"]
    if inside == things.has(thing):
        return
    if inside:
        things[thing] = true
    else:
        things.erase(thing)
    var key := "%s#occupants" % entry["uid"]
    if rules.machine != null and rules.machine.variables.has(key):
        rules.set_value(key, float(things.size()))


## What events call ``body``: "player", an NPC's stable ID (its name without one), or "" for anything else.
static func actor_id(body: Node) -> String:
    if body is Player:
        return "player"
    if body is GeogenNpc:
        return str(body.get_meta("geogen_uid", body.name))
    return ""


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
            var uid := uid_of(node)
            var key := "%s#%s" % [uid, iname]
            interactions_by_key[key] = it
            it.guard = func(actor: String) -> bool: return rules.allows(key, actor)
            it.used.connect(func(actor: String): events.post("use", uid, iname, actor))
            it.target_changed.connect(func(from: String, to: String, actor: String):
                events.post("changed", uid, iname, actor, {"key": key, "from": from, "to": to}))
            it.lock_changed.connect(func(now_locked: bool, actor: String):
                events.post("changed", uid, iname + "_locked", actor,
                    {"key": key + "_locked", "from": not now_locked, "to": now_locked}))
            it.reached.connect(func(state: String, emit: String):
                events.post("arrived", uid, iname, it.actor, {"state": state, "emit": emit, "asset": asset_name}))
            it.raised.connect(func(event: String, actor: String): events.post("signal", uid, event, actor))
            it.refused_use.connect(func(text: String, actor: String):      # as GeogenMachine.use says it
                rules.tell("notify", text, actor)
                rules.tell("play", "refused", actor))
            for part in it.moving_nodes():
                _moving[part] = true
            for t in it.targets:
                _target_of[t] = it
            _restore_pending(it)
            rules.adopt(key, it)        # what rules made of it before its node existed


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


func _physics_process(_delta: float) -> void:
    if not _held.is_empty():
        _release_held()


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
