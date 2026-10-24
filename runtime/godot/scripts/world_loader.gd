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
## The loaded scene said ``ground: false`` (its manifest): the ground plane is no floor while
## it is loaded, so the scene's own floors are all there is and a pit goes below them.
var ground_off := false
## Open every openable interaction before baking, with the open parts as
## obstacles (headless playtests: doors must not block routes, leaves must).
var open_before_bake := false
var open_only := ""              # with open_before_bake: only assets whose name matches this glob ("" = all)
## Show collision shapes as wireframe overlays.
var show_colliders := false:
    set(value):
        show_colliders = value
        if value:                        # built on first show: most runs never look at them
            for body in get_tree().get_nodes_in_group("geogen_collider_body"):
                _add_debug_mesh(body)
        for node in get_tree().get_nodes_in_group("geogen_collider_debug"):
            node.visible = value

var _mtimes := {}  # manifest path -> modified time at last load
var _player_spec: PlayerSpec       # player_spec(), parsed from _player_spec_key (manifest@mtime)
var _player_spec_key := ""
## Gap between models when several exports load at once.
const MODEL_SPACING := 6.0
var _next_x := 0.0    # where the next model's -X edge goes when loading several
## Bounds of each loaded model, in load order.
var model_aabbs: Array[AABB] = []
## Room volumes: [{id, type, xform: Transform3D (global), size: Vector3}]
var rooms: Array[Dictionary] = []
## Named places and rooms with an area: {id, uid, kind: place | room, node, area, actors: {actor: true}}.
var places: Array[Dictionary] = []
## Tracks who is in them (world_places.gd).
var place_tracking := GeogenWorldPlaces.new()
## Spawn points from the manifests: [{name, position: Vector3, yaw_deg: float}]
var spawns: Array[Dictionary] = []
## Loaded nodes by stable ID (extras.geogen.uid: "<package>:<authored path>", geogen/refs.py).
## Content and saved state name nodes by these, never by exported node names.
var nodes_by_uid: Dictionary = {}
## Saving and restoring: the rules sections, interactions, props and items (world_saves.gd).
var saves := GeogenWorldSaves.new()
## Light fixtures: by name, switchable ones by stable ID, `auto: night` lamps (world_lights.gd).
var lights := GeogenWorldLights.new()
## Navmesh baking and its cache (world_navigation.gd).
var navigation := GeogenWorldNavigation.new()
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
## Doorways NPCs path through: [{interaction, open, closed, center, normal (world),
## width, height, depth, clearance, node}]
var portals: Array[Dictionary] = []
var _npcs_scene := ""            # the scene the NPCs spawned for (their states go in saves.trips on unload)
var _reservations := {}   # affordance id -> [GeogenNpc]
## Interactions from asset extras (GeogenInteraction nodes, children of this loader).
var interactions: Array[GeogenInteraction] = []
## Physics props (extras.geogen.physics): rigid bodies, carrying, spawn and stow.
var props: GeogenProps
## Items' icon bodies (extras.geogen.item_icon, the export's hidden ``item_icons`` node): item id -> Node3D,
## out of the tree; ``item_icons()`` renders them for the HUD's hotbar.
var item_bodies := {}
var _item_icons := {}
var _icons_rendering := {}
const ICON_PX := 96
## More item icons were rendered (``item_icons()``).
signal item_icons_ready
## Spawned items (extras.geogen.flows / item): sources, templates and the items in the world.
var flows: GeogenFlows
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
## Reuse baked navmeshes from <generated>/navcache when the export hasn't changed.
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


## Physics steps a second at time scale 1: the project's, or more for a scene that asks (physics: {ticks}).
var physics_rate := 60


func _ready() -> void:
    saves.world = self
    lights.world = self
    navigation.world = self
    place_tracking.world = self
    events.name = "Events"
    add_child(events)
    rules.name = "Rules"
    rules.reset.connect(_sync_trains)
    rules.reset.connect(lights.sync_fixtures)
    events.posted.connect(lights.on_lit)
    events.posted.connect(_on_present)
    rules.reset.connect(func():
        for npc in npcs:
            if is_instance_valid(npc):
                _sync_present(npc))
    rules.world = self
    add_child(rules)
    props = GeogenProps.new(self)
    add_child(props)
    flows = GeogenFlows.new(self)
    add_child(flows)
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
    for manifest_path in _manifests():
        var manifest := _read_manifest(manifest_path)
        if manifest.is_empty() or _load_streamed(manifest_path, manifest):
            continue
        var t := Time.get_ticks_usec()
        var root := GeogenChunkStreamer._load_glb(_model_path(manifest_path, manifest))
        _lap("gltf_parse", t)
        _attach_model(manifest_path, manifest, root)
    return _loaded()


## A streamed export (a chunk index): prime the streamer around the start; false for a single file.
func _load_streamed(manifest_path: String, manifest: Dictionary) -> bool:
    if manifest.get("format") != GeogenChunkStreamer.INDEX_FORMAT:
        return false
    var t := Time.get_ticks_usec()
    _load_chunked(manifest_path, manifest)
    _lap("stream_prime", t)
    return true


## Everything is loaded: say so (world_loaded) with the combined bounds.
func _loaded() -> AABB:
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
    rules.take_interactions()     # their state, kept for when the package comes back
    var people := {}
    for npc in npcs:
        if is_instance_valid(npc):
            people[npc_uid(npc)] = npc.save_state()
    saves.note_characters()
    if not people.is_empty() and _npcs_scene != "":
        saves.trips[_npcs_scene] = people   # and its NPCs', for when the scene loads again
    saves.pending_npcs = {}
    for child in get_children():
        if child == events or child == rules or child == props or child == flows:
            continue
        remove_child(child)
        child.free()
    flows.clear()
    props.clear()
    _clear_item_icons()
    rules.unload()
    physics_rate = int(ProjectSettings.get_setting("physics/common/physics_ticks_per_second", 60))
    apply_physics_rate()
    _use_ground({})
    _use_atmosphere({})
    interactions_by_key.clear()
    streamer = null
    _lap("unload", t)
    _mtimes.clear()
    _next_x = 0.0
    model_aabbs.clear()
    rooms.clear()
    places.clear()
    place_tracking.clear()
    spawns.clear()
    nodes_by_uid.clear()
    saves.pending.clear()
    interactions.clear()
    lights.clear()
    affordances.clear()
    npcs.clear()
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
        # Parsing a manifest (a city's chunk index is megabytes) costs more than a frame:
        # once per file and version, not per chunk or navigation tile.
        var key := "%s@%d" % [manifest, FileAccess.get_modified_time(manifest)]
        if key != _player_spec_key:
            _player_spec = PlayerSpec.from_manifest(manifest)
            _player_spec_key = key
        return _player_spec
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
            _load_streamed(manifest_path, manifest)
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
    return _loaded()


func _exit_tree() -> void:
    if _parse_task >= 0:
        WorkerThreadPool.wait_for_task_completion(_parse_task)


## The ``item_icons`` node of an export (geogen/inventory.py attach_icons): each child is an
## item's body with extras ``item_icon: {item}``. They leave the world here and are kept by
## item id for ``item_icons()`` to draw.
func _take_item_icons(root: Node) -> void:
    for node: Node3D in root.find_children("*", "Node3D", true, false):
        var icon: Variant = geogen_extras(node).get("item_icon")
        if icon is Dictionary and node.get_parent() != null:
            node.get_parent().remove_child(node)
            node.visible = true
            var old: Node = item_bodies.get(str(icon["item"]))
            if old != null and is_instance_valid(old):
                old.free()
            item_bodies[str(icon["item"])] = node
    for node in root.find_children("item_icons", "Node3D", true, false):
        node.get_parent().remove_child(node)
        node.free()


func _clear_item_icons() -> void:
    for body in item_bodies.values():
        if is_instance_valid(body):
            body.free()
    item_bodies.clear()
    _item_icons.clear()
    _icons_rendering.clear()


## The icons of the items this export can draw: item id -> Texture2D, the ones rendered so
## far. A body with no icon yet is rendered now (one frame, off screen, in its own world:
## an orthographic three-quarter view over a transparent background) and `item_icons_ready`
## says when more are in. Headless runs render none: the hotbar shows initials instead.
func item_icons() -> Dictionary:
    if DisplayServer.get_name() != "headless":
        for id in item_bodies:
            if not _item_icons.has(id) and not _icons_rendering.has(id):
                _icons_rendering[id] = true
                _render_item_icon(id)
    return _item_icons


func _render_item_icon(id: String) -> void:
    var body: Node3D = item_bodies[id]
    var viewport := SubViewport.new()
    viewport.size = Vector2i(ICON_PX, ICON_PX)
    viewport.transparent_bg = true
    viewport.own_world_3d = true
    viewport.msaa_3d = Viewport.MSAA_4X
    viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
    add_child(viewport)
    var copy: Node3D = body.duplicate()
    copy.transform = Transform3D.IDENTITY
    viewport.add_child(copy)
    for mi: MeshInstance3D in copy.find_children("*", "MeshInstance3D", true, false):
        if _is_collider(mi):
            mi.visible = false
    var box := _subtree_aabb(copy)
    var centre := box.get_center()
    var radius := maxf(box.size.length() / 2.0, 0.01)
    var camera := Camera3D.new()
    camera.projection = Camera3D.PROJECTION_ORTHOGONAL
    camera.near = 0.01
    camera.far = radius * 12.0 + 1.0
    camera.position = centre + Vector3(1.0, 0.8, 1.2).normalized() * (radius * 5.0 + 0.5)
    viewport.add_child(camera)
    camera.look_at(centre, Vector3.UP)
    # Frame what the camera actually sees: the box's corners in its view, not its diagonal (a
    # lying stick would be a hair otherwise), and centre on them.
    var into_view := camera.global_transform.affine_inverse()
    var lo := Vector2(INF, INF)
    var hi := Vector2(-INF, -INF)
    for i in 8:
        var corner := into_view * box.get_endpoint(i)
        lo = Vector2(minf(lo.x, corner.x), minf(lo.y, corner.y))
        hi = Vector2(maxf(hi.x, corner.x), maxf(hi.y, corner.y))
    var seen := hi - lo
    camera.size = maxf(seen.x, seen.y) * 1.12
    camera.global_translate(camera.global_transform.basis * Vector3((lo.x + hi.x) / 2.0, (lo.y + hi.y) / 2.0, 0.0))
    var key := DirectionalLight3D.new()
    key.rotation = Vector3(deg_to_rad(-50.0), deg_to_rad(30.0), 0.0)
    key.light_energy = 1.1
    viewport.add_child(key)
    var env := WorldEnvironment.new()
    env.environment = Environment.new()
    env.environment.background_mode = Environment.BG_CLEAR_COLOR
    env.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
    env.environment.ambient_light_color = Color(0.75, 0.78, 0.85)
    env.environment.ambient_light_energy = 0.7
    viewport.add_child(env)
    await RenderingServer.frame_post_draw
    await RenderingServer.frame_post_draw
    if is_instance_valid(viewport):
        var image := viewport.get_texture().get_image()
        if image != null and not image.is_empty():
            _item_icons[id] = ImageTexture.create_from_image(image)
        viewport.queue_free()
    _icons_rendering.erase(id)
    item_icons_ready.emit()


## The bounds of every drawn mesh under ``node`` (global space; colliders left out).
static func _subtree_aabb(node: Node3D) -> AABB:
    var out := AABB()
    var first := true
    var meshes: Array[Node] = node.find_children("*", "MeshInstance3D", true, false)
    if node is MeshInstance3D:
        meshes.append(node)
    for mi: MeshInstance3D in meshes:
        if _is_collider(mi) or not mi.visible or mi.mesh == null:
            continue
        var box: AABB = mi.global_transform * mi.get_aabb()
        out = box if first else out.merge(box)
        first = false
    return out


## A scene's ``physics: {ticks}``: step physics at least that often while it is loaded (Jolt
## lets a fast ball sink into a curved trimesh wall at 60; unload puts the project's rate back).
func _use_physics(manifest: Dictionary) -> void:
    var physics = manifest.get("physics")
    if physics is Dictionary and int(physics.get("ticks", 0)) > physics_rate:
        physics_rate = int(physics["ticks"])
        apply_physics_rate()
        print("geogen: physics at %d ticks a second" % physics_rate)


## A scene's ``ground: false`` (the manifest's ``ground``): main.tscn's Ground (a plane at y = 0)
## is hidden and stops colliding while it is loaded, and the navmesh bake leaves it out.
func _use_ground(manifest: Dictionary) -> void:
    ground_off = manifest.get("ground") == false
    var ground := get_parent().get_node_or_null("Ground") if get_parent() != null else null
    if ground == null:
        return
    ground.visible = not ground_off
    for shape in ground.find_children("*", "CollisionShape3D", true, false):
        shape.disabled = ground_off


## A scene's ``atmosphere`` (the manifest's, resolved: fog, grading, sun, sky, moon, night, glow,
## start time): the clock plays the scene in that light while it is loaded (clock.gd).
func _use_atmosphere(manifest: Dictionary) -> void:
    if clock == null:
        return
    var spec = manifest.get("atmosphere")
    clock.set_atmosphere(spec if spec is Dictionary else {}, String(manifest.get("name", scene_name)))
    if spec is Dictionary:
        print("geogen: atmosphere %s" % spec.get("preset", "custom"))


## How far round each model the ground plane is walkable in the navmesh bake: none without one.
func nav_ground_margin() -> float:
    return 0.0 if ground_off else ground_margin


## Physics steps a second for ``physics_rate`` at the world's time scale (sped up, it steps
## more often, so a step covers the same world time).
func apply_physics_rate() -> void:
    var scale := maxf(Engine.time_scale, 1.0)
    Engine.physics_ticks_per_second = int(physics_rate * scale)
    Engine.max_physics_steps_per_frame = int(8 * scale * physics_rate / 60.0)


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
    _use_physics(manifest)
    _use_ground(manifest)
    _use_atmosphere(manifest)
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
                if it.states.has("open") and (open_only == "" or String(it.asset.name).match(open_only)):
                    it.set_state("open", true)
                    it.hold = true
    if bake_navigation:
        t = Time.get_ticks_usec()
        navigation.build(root, manifest_path, model_path)
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
    flows.setup(root)        # item templates leave the world before anything else sees them
    _take_item_icons(root)   # and the items' icon bodies (inventory.attach_icons)
    for node in [root] + root.find_children("*", "Node3D", true, false):
        var uid := uid_of(node)
        if uid != "":
            nodes_by_uid[uid] = node
    _collect_interactions(root)
    props.setup(root)        # before colliders: a prop's parts' colliders give way to its body
    lights.add(root)
    lights.sync_fixtures()
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
    place_tracking.collect(root)
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


## Unregister everything ``root`` contributed (before freeing a streamed chunk).
func forget(root: Node) -> void:
    var inside := func(node) -> bool:
        return node is Node and is_instance_valid(node) and (node == root or root.is_ancestor_of(node))
    for it in interactions.duplicate():
        if inside.call(it.asset):
            # Remembered until its chunk loads again: by the rules program, which reads and
            # drives it meanwhile, else here.
            if uid_of(it.asset) != "" and not rules.release(GeogenWorldSaves.state_key(it), it.snapshot(), it.progress()):
                saves.remember(it)
            interactions.erase(it)
            it.queue_free()
    flows.forget(root)
    props.forget(root)
    for uid in nodes_by_uid.keys():
        if inside.call(nodes_by_uid[uid]):
            nodes_by_uid.erase(uid)
    rooms = rooms.filter(func(r): return not inside.call(r.get("node")))
    place_tracking.forget(inside)
    affordances = affordances.filter(func(a): return not inside.call(a.get("node")))
    portals = portals.filter(func(p): return not inside.call(p.get("node")))
    travel_points = travel_points.filter(func(t): return not inside.call(t.get("node")))
    npcs = npcs.filter(func(n): return not inside.call(n))
    lights.forget(inside)
    for key in _moving.keys():
        if inside.call(key):
            _moving.erase(key)
    for key in _target_of.keys():
        if inside.call(key):
            _target_of.erase(key)


func _load_chunked(index_path: String, index: Dictionary) -> void:
    _use_physics(index)
    _use_ground(index)
    _use_atmosphere(index)
    # Tiles are baked around the player as it goes: sync them off the main thread.
    NavigationServer3D.map_set_use_async_iterations(get_world_3d().navigation_map, true)
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
    lights.set_night(on)
    for t in traffic:
        if is_instance_valid(t):
            t.set_night(on)
    for t in trains:
        if is_instance_valid(t):
            t.set_night(on)


## A train closed or opened a level crossing (road traffic stops for closed ones).
func set_crossing(id: String, closed: bool) -> void:
    crossing_closed[id] = closed


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
        # Handles (front: +normal, back) ride the part of the door nearest them, so a hand on one
        # follows it as the leaf swings: {side: {node, local}}, side 1 or -1. Made for the door as
        # exported (its interaction's initial state).
        var handles := {}
        for side_name in (p.get("handles", {}) as Dictionary):
            var h: Array = p["handles"][side_name]
            var point := xform * Vector3(h[0], h[1], h[2])
            var best: Node3D = null
            for part in it.moving_nodes():
                if best == null or part.global_position.distance_to(point) < best.global_position.distance_to(point):
                    best = part
            if best != null and it.state == it.initial:
                handles[1 if side_name == "front" else -1] = {"node": best, "local": best.global_transform.affine_inverse() * point}
        var swing: int = {"front": 1, "back": -1}.get(str(p.get("swing", "")), 0)
        portals.append({"node": node, "interaction": it, "open": str(p.get("open", "open")),
            "closed": str(p.get("closed", "closed")), "center": xform * Vector3(c[0], c[1], c[2]),
            "normal": (xform.basis * Vector3(n[0], n[1], n[2])).normalized(),
            "width": float(p.get("width", 0.9)), "height": float(p.get("height", 2.0)),
            "depth": float(p.get("depth", 0.3)), "clearance": float(p.get("clearance", 0.9)),
            "handles": handles, "swing": swing})


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
    # Back from a trip to another scene: its NPCs as they were left.
    if saves.trips.has(scene_name):
        saves.pending_npcs.merge(saves.trips[scene_name])
        saves.trips.erase(scene_name)
    _npcs_scene = scene_name
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
        var uid := npc_uid(npc)        # the greeting is the placed NPC's signal (state.npc_signal)
        npc.greeted.connect(func(signal_name: String, by: String): events.post("signal", uid, signal_name, by))
        rules.add_actor(actor, "npc")
        _sync_present.call_deferred(npc)          # (once it is ready)
        if spec != null:
            npc.step_height = spec.step_height
        npcs.append(npc)
        var saved := not saves.pending_npcs.is_empty() or not saves.characters.is_empty()
        if saved and npc.is_node_ready():
            saves.restore_npc(npc)       # a loaded save's, or as it was when its scene was left
        elif saved:
            npc.ready.connect(func(): saves.restore_npc(npc), CONNECT_ONE_SHOT)


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


## ``points`` with every doorway it goes through taken straight: from ``clearance`` + 0.3 m in
## front, through the middle, to as far beyond. A door's leaf (open, or about to be) isn't on
## the navmesh, so a path that cuts the corner past a jamb can snag on it.
func through_portals(points: PackedVector3Array) -> PackedVector3Array:
    var out := PackedVector3Array()
    if points.is_empty():
        return out
    out.append(points[0])
    var i := 1
    while i < points.size():
        var a := out[out.size() - 1]
        var b := points[i]
        var crossed := {}
        for p in portals:
            var da: float = (a - p["center"]).dot(p["normal"])
            var db: float = (b - p["center"]).dot(p["normal"])
            if signf(da) == signf(db) or da == 0.0:
                continue
            var hit := a.lerp(b, da / (da - db))
            var lateral: Vector3 = hit - p["center"]
            lateral -= p["normal"] * lateral.dot(p["normal"])
            lateral.y = 0.0
            if lateral.length() > p["width"] / 2.0 + 0.4 or absf(hit.y - (p["center"].y - p["height"] / 2.0)) > 1.0:
                continue
            crossed = p
            var middle: Vector3 = p["center"]
            middle.y = hit.y
            var reach: float = float(p["clearance"]) + 0.3
            var across: Vector3 = p["normal"] * signf(da)
            if absf(da) > 0.05:
                out.append(middle + across * reach)
            out.append(middle)
            out.append(middle - across * reach)
            # Leave out the path's own points in the doorway's way (before it is beyond).
            while i < points.size() and (points[i] - p["center"]).dot(-across) < reach \
                    and (points[i] - p["center"]).dot(-across) > -reach:
                i += 1
            break
        if crossed.is_empty():
            out.append(b)
            i += 1
    return out


## The first closed portal (door) a path crosses within ``lookahead`` metres of ``from``,
## walking ``points`` from index ``start``: {portal, interaction, hit (where the path crosses
## it), distance (along the path), side (of the portal ``from`` is on: 1 or -1)}, or {}.
## NPCs pass doors this way (npc.gd `pass`), and so does the player's walk_to.
func closed_portal_on(points: PackedVector3Array, from: Vector3, start: int, lookahead: float) -> Dictionary:
    var walked := 0.0
    var a := from
    for i in range(start, points.size()):
        var b := points[i]
        for p in portals:
            var it: GeogenInteraction = p["interaction"]
            if not is_instance_valid(it) or (it.state == p["open"] and it.target == p["open"]):
                continue
            var da: float = (a - p["center"]).dot(p["normal"])
            var db: float = (b - p["center"]).dot(p["normal"])
            if signf(da) == signf(db):
                continue
            var hit := a.lerp(b, da / (da - db))
            var lateral: Vector3 = hit - p["center"]
            lateral -= p["normal"] * lateral.dot(p["normal"])
            lateral.y = 0.0
            if lateral.length() > p["width"] / 2.0 + 0.4 or absf(hit.y - (p["center"].y - p["height"] / 2.0)) > 1.0:
                continue
            if walked + a.distance_to(hit) > lookahead:
                return {}
            var side := signf((from - p["center"]).dot(p["normal"]))
            return {"portal": p, "interaction": it, "hit": hit, "distance": walked + a.distance_to(hit),
                "side": side if side != 0.0 else 1.0}
        walked += a.distance_to(b)
        if walked > lookahead:
            break
        a = b
    return {}


## The key an interaction's state is saved under: "<stable ID>#<interaction>", so a save
## survives a re-export that renames nodes (exports without IDs: "<node name>/<interaction>").
func _collect_rooms(root: Node) -> void:
    for node in root.find_children("*", "Node3D", true, false):
        var g := geogen_extras(node)
        if g.get("type") == "room_volume":
            var size: Array = g.get("size", [0, 0, 0])
            var room: Dictionary = g.get("room", {})
            rooms.append({"id": room.get("id", node.name), "type": room.get("type", ""), "nav": g.get("nav", true),
                "xform": (node as Node3D).global_transform, "size": Vector3(size[0], size[1], size[2]),
                "node": node})


## What events call ``body``: "player", an NPC's stable ID (its name without one), or "" for anything else.
## An NPC placed with `present: <condition>` is in the world only while its `present` state is
## true (a binding the export compiled from the condition).
func _sync_present(npc: GeogenNpc) -> void:
    if is_instance_valid(npc):
        var present = rules.get_value("%s#present" % npc_uid(npc))
        npc.set_present(present != false)


func _on_present(event: Dictionary) -> void:
    if event["type"] != "changed" or event["name"] != "present":
        return
    for npc in npcs:
        if is_instance_valid(npc) and npc_uid(npc) == event["subject"]:
            npc.set_present(bool(event["data"].get("to", true)))


static func actor_id(body: Node) -> String:
    if body is Player:
        return "player"
    if body is GeogenNpc:
        # A named person is one actor wherever it is placed: its quests and the state content
        # packages keep for it follow it from scene to scene.
        var character := str(body.definition.get("character", ""))
        return CHARACTER_ACTOR % character if character != "" else npc_uid(body)
    return ""


const CHARACTER_ACTOR := "character:%s"     # a character's actor ID (geogen.npc.actor_id)


## An NPC's stable ID: its placement, what saves keep where it is by.
static func npc_uid(npc: Node) -> String:
    return str(npc.get_meta("geogen_uid", npc.name))


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
            saves.restore_pending(it)
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
    place_tracking.release_held()


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
    prepare_materials(root)


## Material fixes for a loaded glTF (static: streamed chunks run it on a worker before the
## root enters the tree). ``mipmapped`` maps a texture to its mipmapped stand-in; textures
## not in it get their mipmaps made here, by reading them back from the GPU (slow, and only
## safe on the main thread): with ``read_back`` false they are left, and the root gets meta
## `geogen_unmipped` so the main thread finishes the job.
static func prepare_materials(root: Node, mipmapped := {}, read_back := true) -> void:
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
                if mipmapped.has(tex):
                    mat.set_texture(param, mipmapped[tex])
                    continue
                if not read_back:
                    root.set_meta("geogen_unmipped", true)
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
    var colliders := collider_meshes(meshes)
    var shapes := {}
    if colliders.is_empty():
        for mi: MeshInstance3D in meshes:
            var shape := collider_shape(mi, false, shapes)
            if shape != null:
                _add_body(mi, shape, Transform3D.IDENTITY, mi.get_aabb())
        return meshes.size()
    for mi in colliders:
        var shape := collider_shape(mi, true, shapes)
        if shape != null:
            _add_body(mi.get_parent(), shape, mi.transform, mi.get_aabb())
        mi.get_parent().remove_child(mi)
        mi.free()
    return meshes.size() - colliders.size()


## The exported collider meshes among ``meshes`` (none: older exports, every mesh collides).
static func collider_meshes(meshes: Array) -> Array[MeshInstance3D]:
    var out: Array[MeshInstance3D] = []
    for mi: MeshInstance3D in meshes:
        if _is_collider(mi):
            out.append(mi)
    return out


## The collision shape for a collider mesh (``exported``) or a plain mesh's triangles. Built
## once per mesh and kind (``shapes``: instances share their collider mesh, so share the shape),
## or taken from meta `geogen_shape` when a loader built it on a worker (precompute_shapes).
static func collider_shape(mi: MeshInstance3D, exported: bool, shapes: Dictionary) -> Shape3D:
    if mi.has_meta("geogen_shape"):
        return mi.get_meta("geogen_shape")
    if mi.mesh == null:
        return null
    var convex: bool = exported and (String(mi.name).ends_with("-convcolonly")
        or geogen_extras(mi).get("shape") in ["box", "hull"])
    var key := [mi.mesh, convex]
    var shape: Shape3D = shapes.get(key)
    if shape == null:
        shape = mi.mesh.create_convex_shape(true, false) if convex else mi.mesh.create_trimesh_shape()
        if shape == null:   # a flat or degenerate hull: fall back to the exact triangles
            shape = mi.mesh.create_trimesh_shape()
        shapes[key] = shape
    return shape


## Build a loaded root's collision shapes ahead of _add_collision (safe on a worker thread:
## the root isn't in the tree yet); _add_collision then only adds the bodies.
static func precompute_shapes(root: Node) -> void:
    var meshes := root.find_children("*", "MeshInstance3D", true, false)
    var colliders := collider_meshes(meshes)
    var shapes := {}
    for mi: MeshInstance3D in (meshes if colliders.is_empty() else colliders):
        var shape := collider_shape(mi, not colliders.is_empty(), shapes)
        if shape != null:
            mi.set_meta("geogen_shape", shape)


func _is_moving(node: Node) -> bool:
    while node != null and node != self:
        if _moving.has(node):
            return true
        node = node.get_parent()
    return false


static func _is_collider(mi: MeshInstance3D) -> bool:
    var n := String(mi.name)
    return n.ends_with("-colonly") or n.ends_with("-convcolonly") or geogen_extras(mi).get("type") == "collider"


## ``box``: the shape's bounds in the body's frame (kept as meta `geogen_aabb`: streamed
## navigation tiles parse only the bodies overlapping them).
func _add_body(parent: Node, shape: Shape3D, xform: Transform3D, box: AABB) -> void:
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
    body.set_meta("geogen_aabb", box)
    body.add_to_group("geogen_collider_body")
    var col := CollisionShape3D.new()
    col.shape = shape
    body.add_child(col)
    parent.add_child(body)
    if show_colliders:
        _add_debug_mesh(body)


## The wireframe overlay of a collider body's shape (once per body).
func _add_debug_mesh(body: Node) -> void:
    if body.has_meta("geogen_debug"):
        return
    body.set_meta("geogen_debug", true)
    for col: CollisionShape3D in body.find_children("*", "CollisionShape3D", false, false):
        if col.shape == null:
            continue
        var debug := MeshInstance3D.new()
        debug.mesh = col.shape.get_debug_mesh()
        debug.material_override = _collider_material
        debug.visible = show_colliders
        debug.add_to_group("geogen_collider_debug")
        body.add_child(debug)
