extends Node3D
## Root of the geogen runtime: loads exports from generated/, spawns a
## first-person player sized from the manifest, and shows a debug overlay.
##
## User args (after `--`):
##   --scene NAME | --scene=NAME   load generated/NAME.glb (default: generated/catalogue.json's
##                                 default scene, else every export); a chunked export
##                                 (generated/NAME_chunks/) streams around the player
##   --scene=all                   load every export side by side, ignoring the catalogue
##   --stream                      prefer NAME's chunked export when both exist
##   --stream-radius=F,L,I         streaming radii: full exterior, LOD, interiors (m)
##   --generated=DIR               read exports from DIR instead of res://generated
##   --spawn=X,Y,Z  --yaw=DEG      player start (default: the export's first spawn
##                                 point, else in front of the model)
##   --camera=overview             start from the overview camera (F4 toggles)
##   --colliders                   show collision shapes (F2 toggles)
##   --walk=SECONDS                walk forward, print the final position, quit
##   --greet=NAME[@S]              greet the NPC named NAME... after S s (default 2): it waves back
##   --use=ASSET                   use ASSET's interactions at start (e.g. open "door");
##                                 --use=@aim presses E on whatever the player looks at
##   --wait=SECONDS                wait before the --walk starts (let a door swing)
##   --nav=AX,AZ:BX,BZ             print the navigation path between two floor points, quit
##   --lights                      print the fixture lights (JSON) when quitting
##   --keys=K1,K2                  keys the player holds (lock/unlock doors with L)
##   --lock=@aim                   press L on whatever the player looks at
##   --pitch=DEG                   look up (+) or down (-) at spawn
##   --save=PATH / --load=PATH     write interaction state on quit / restore it at start
##   --status                      print the player's pose/room/focus when quitting
##   --npc-trace                   print every NPC decision and step ("npc: {...}")
##   --timescale=N                 run the world N times faster (physics ticks scale too)
##   --simulate=SECONDS            run for SECONDS of world time, print "npc summary: [...]", quit
##   --npc-labels                  show what each NPC is doing and its needs (F5 toggles)
##   --time=HH:MM                  world clock at start (default 13:00): sun, sky, street lamps,
##                                 routines and traffic follow it
##   --day-length=SECONDS          real seconds per 24 h of world time (default 1440; 0 freezes it)
##   --traffic-trace               print traffic claims, overlaps and respawns ("traffic: {...}")
##                                 (--simulate also prints "traffic summary: [...]")
##   --camera=follow[:NAME]        watch an NPC (the first, or the one whose name starts with NAME)
##
## E uses the focused interaction, sits/lies on the furniture you look at, or greets an NPC
## (E or walking stands you up); L locks/unlocks with a held key.
##   --playtest[=N]                check every room/interaction is reachable, walk N routes
##                                 with a bot (default 6), print "playtest: {...}", quit
##
## In game, look at something interactive within reach and press E.
##   --play=ANIM[@SECONDS]         loop every animation named ANIM (or <node>_ANIM); with @SECONDS,
##                                 freeze it at that time (skeletal clips, interaction animations)
##   --skeletons                   print Skeleton3D bones, skinned mesh bounds and animations on quit
##   --screenshot=PATH             save a frame and quit
##   --quit-after=N                quit after N frames
##   --manifest=PATH               print the player spec read from PATH
##   --menu                        open the scene list at start (screenshots)
##   --timings                     print each load's phases ("load timings: {...}", ms) and whether
##                                 the navmesh came from <generated>/.navcache
##   --list-scenes                 print the scene catalogue (name, group, exported, description), quit
##   --switch=NAME[@S]             switch to scene NAME after S s (default 1; repeatable, in order);
##                                 prints "switched: {...}" and "switch stats: {...}" (node/object
##                                 counts), then quits unless --walk / --status / --screenshot follow
##
## Travel points (extras.geogen.travel: doors you use, portals you walk into) fade out, load
## their target scene (or move within this one) and put the player at the named spawn; prints
## "travelled: {...}".
##
## F6 opens the scene list, F7 / F8 go to the previous / next exported scene. The last scene
## picked is remembered (user://settings.cfg, per generated dir) and loaded when no --scene is
## given (not in headless runs).

var quit_after_frames := 0
var screenshot_path := ""
var walk_seconds := 0.0
var player_spec := PlayerSpec.new()

var _frames := 0
var _spawn_override = null
var _yaw_override = null
var _walk_left := 0.0
var _manifest_arg := false
var _start_overview := false
var _use_assets: Array[String] = []
var _use_aim := false
var _print_lights := false
var keys: Array = []
var _lock_aim := false
var _pitch := 0.0
var _save_path := ""
var _load_path := ""
var _print_status := false
var _focus_affordance := {}
var _focus_npc: GeogenNpc = null
var _greet := {}                  # --greet: {name prefix: world seconds}
var _nav_query := []
var _frames_nav := 0
var _playtest_walks := -1
var _wait_left := 0.0
var _focus: GeogenInteraction = null
var _simulate := 0.0
var _sim_clock := 0.0
var _npc_labels := false
var _follow = null       # NPC name prefix to follow with the overview camera, or null
var _prompt: Label
var _play := ""
var _play_at := -1.0
var _print_skeletons := false
var clock: GeogenClock
var _start_time := 13.0
var _menu: GeogenSceneMenu
var _switches: Array[Dictionary] = []   # --switch: [{name, at: seconds}]
var _switch_clock := 0.0
var _stats_in := -1                     # frames until the post-switch stats print
var _queued := false                    # the last switch came from --switch (not the menu or travel)
var _list_scenes := false
var _travelling := false
var _switching := false
var _loading: Label
var _open_menu_at_start := false
var _fade: ColorRect
const FADE_SECONDS := 0.25

const SETTINGS_PATH := "user://settings.cfg"
var _day_length := 1440.0

@onready var world: WorldLoader = $World
@onready var overview: Camera3D = $OverviewCamera
@onready var overlay: Label = $Overlay/Label
var player: Player


func _ready() -> void:
    var args := OS.get_cmdline_user_args()
    var i := 0
    while i < args.size():
        var arg := args[i]
        var value := arg.get_slice("=", 1) if "=" in arg else ""
        if arg == "--scene" and i + 1 < args.size():
            i += 1
            world.scene_name = args[i]
        elif arg.begins_with("--scene="):
            world.scene_name = value
        elif arg == "--stream":
            world.prefer_chunks = true
            world.force_chunks = true
        elif arg.begins_with("--stream-radius="):
            world.stream_radii = value.split_floats(",")
        elif arg.begins_with("--generated="):
            world.generated_dir = value
        elif arg.begins_with("--spawn="):
            var p := value.split_floats(",")
            _spawn_override = Vector3(p[0], p[1], p[2])
        elif arg.begins_with("--yaw="):
            _yaw_override = float(value)
        elif arg == "--colliders":
            world.show_colliders = true
        elif arg == "--camera=overview":
            _start_overview = true
        elif arg.begins_with("--walk="):
            walk_seconds = float(value)
        elif arg == "--playtest":
            _playtest_walks = 6
        elif arg.begins_with("--playtest="):
            _playtest_walks = int(value)
        elif arg.begins_with("--nav="):
            for point in value.split(":"):
                var xz := point.split_floats(",")
                _nav_query.append(Vector3(xz[0], 0.0, xz[1]))
        elif arg == "--lights":
            _print_lights = true
        elif arg.begins_with("--keys="):
            keys = Array(value.split(",", false))
        elif arg.begins_with("--pitch="):
            _pitch = float(value)
        elif arg == "--lock=@aim":
            _lock_aim = true
        elif arg.begins_with("--save="):
            _save_path = value
        elif arg.begins_with("--load="):
            _load_path = value
        elif arg == "--npc-trace":
            world.npc_trace = true
        elif arg.begins_with("--timescale="):
            var scale := maxf(float(value), 0.01)
            Engine.time_scale = scale
            Engine.physics_ticks_per_second = int(60 * maxf(scale, 1.0))
            Engine.max_physics_steps_per_frame = int(8 * maxf(scale, 1.0))
        elif arg.begins_with("--simulate="):
            _simulate = float(value)
        elif arg.begins_with("--time="):
            _start_time = GeogenClock.parse(value)
        elif arg.begins_with("--day-length="):
            _day_length = float(value)
        elif arg == "--traffic-trace":
            world.traffic_trace = true
        elif arg == "--npc-labels":
            _npc_labels = true
        elif arg == "--camera=follow" or arg.begins_with("--camera=follow:"):
            _follow = arg.get_slice(":", 1) if ":" in arg else ""
            _start_overview = true
        elif arg == "--status":
            _print_status = true
        elif arg == "--use=@aim":
            _use_aim = true
        elif arg.begins_with("--greet="):
            _greet[value.get_slice("@", 0)] = float(value.get_slice("@", 1)) if "@" in value else 2.0
        elif arg.begins_with("--use="):
            _use_assets.append(value)
        elif arg.begins_with("--wait="):
            _wait_left = float(value)
        elif arg.begins_with("--play="):
            _play = value.get_slice("@", 0)
            if "@" in value:
                _play_at = float(value.get_slice("@", 1))
        elif arg == "--skeletons":
            _print_skeletons = true
        elif arg.begins_with("--quit-after="):
            quit_after_frames = int(value)
        elif arg.begins_with("--screenshot="):
            screenshot_path = value
            if quit_after_frames <= 0:
                quit_after_frames = 10  # let shadows/SSAO settle
        elif arg == "--timings":
            world.print_timings = true
        elif arg == "--menu":
            _open_menu_at_start = true
        elif arg == "--list-scenes":
            _list_scenes = true
        elif arg.begins_with("--switch="):
            var at := float(value.get_slice("@", 1)) if "@" in value else 1.0
            var after: float = _switches[-1]["at"] if not _switches.is_empty() else 0.0
            _switches.append({"name": value.get_slice("@", 0), "at": maxf(at, after)})
        elif arg.begins_with("--manifest="):
            var spec := PlayerSpec.from_manifest(value)
            if spec:
                player_spec = spec
                _manifest_arg = true
        i += 1
    print("geogen runtime ready (Godot %s)" % Engine.get_version_info().string)
    if _list_scenes:
        _print_scene_list()
        set_process(false)
        set_physics_process(false)
        get_tree().quit()
        return

    world.open_before_bake = _playtest_walks >= 0
    world.prime_focus = _spawn_override
    world.world_loaded.connect(_on_world_loaded)
    world.travel_requested.connect(_travel)
    world.interaction_event.connect(func(asset: String, interaction: String, state: String, event: String):
        print("interaction event: %s" % JSON.stringify(
            {"asset": asset, "interaction": interaction, "state": state, "event": event})))
    clock = GeogenClock.new()
    clock.name = "Clock"
    clock.hours = _start_time
    clock.day_length = _day_length
    clock.sun = $Sun
    clock.environment = ($WorldEnvironment as WorldEnvironment).environment
    clock.world = world
    world.clock = clock
    if world.scene_name == "all":
        world.scene_name = ""
    else:
        world.use_catalogue_default(_remembered_scene())
    world.load_all()
    add_child(clock)          # applies the time (night lights) once the world is loaded
    _show_npc_labels()
    if _play != "":
        _play_animations()
    if not world.npcs.is_empty():
        print("npcs: %s" % JSON.stringify(world.npcs.map(func(n): return String(n.name))))
    if _load_path != "":
        var saved = JSON.parse_string(FileAccess.get_file_as_string(_load_path))
        if saved is Dictionary:
            world.load_state(saved)
    for asset_name in _use_assets:
        var found := world.interactions_of(asset_name)
        if found.is_empty():
            push_warning("geogen: --use=%s: no interactions on that node" % asset_name)
        for it in found:
            it.use()
    _prompt = Label.new()
    _prompt.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    _prompt.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
    _prompt.position.y += 40
    $Overlay.add_child(_prompt)
    var spec := world.player_spec()
    if spec and not _manifest_arg:
        player_spec = spec
    print("player spec: %s" % JSON.stringify(player_spec.to_dict()))

    player = Player.new()
    player.spec = player_spec
    add_child(player)
    _fade = ColorRect.new()
    _fade.color = Color(0, 0, 0, 0)
    _fade.mouse_filter = Control.MOUSE_FILTER_IGNORE
    _fade.set_anchors_preset(Control.PRESET_FULL_RECT)
    $Overlay.add_child(_fade)
    _loading = Label.new()
    _loading.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    _loading.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
    _loading.add_theme_font_size_override("font_size", 22)
    _loading.visible = false
    $Overlay.add_child(_loading)
    _menu = GeogenSceneMenu.new()
    _menu.picked.connect(func(scene: String): switch_scene(scene))
    _menu.closed.connect(_menu_closed)
    add_child(_menu)
    _place_player(world.world_aabb())
    _frame_overview(world.world_aabb())
    # The viewport auto-selects the first camera to enter the tree (the
    # overview), so always pick one explicitly.
    (overview if _start_overview else player.camera).make_current()
    if _playtest_walks >= 0:
        var playtest := GeogenPlaytest.new()
        playtest.world = world
        playtest.player = player
        playtest.start = player.global_position
        playtest.start_yaw = rad_to_deg(player.rotation.y)
        playtest.walk_count = _playtest_walks
        add_child(playtest)
    if walk_seconds > 0.0:
        _walk_left = walk_seconds
        if _wait_left <= 0.0 and _switches.is_empty():
            player.scripted_move = Vector2(0, 1)
    elif DisplayServer.get_name() != "headless" and screenshot_path == "":
        Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
    if _open_menu_at_start:
        _open_menu()


func _on_world_loaded(aabb: AABB) -> void:
    _frame_overview(aabb)
    $ScaleReference.visible = aabb.size == Vector3.ZERO


## Spawn in front (+Z) of whatever was loaded, facing it. The command line's
## --spawn / --yaw / --pitch apply to the first world only (``overrides``).
func _place_player(aabb: AABB, spawn_name := "", overrides := true) -> void:
    var pos := Vector3(0, 0, 5)
    var yaw := 0.0
    if not world.model_aabbs.is_empty():
        aabb = world.model_aabbs[0]  # several exports load in a row: start at the first
    if aabb.size != Vector3.ZERO:
        pos = Vector3(aabb.get_center().x, 0, aabb.end.z + 3.0)
    # Use the first export's own manifest spawn (several exports lay out in a row).
    var spawn := {}
    if spawn_name != "":
        spawn = world.spawn_named(spawn_name)
    else:
        for s in world.spawns:
            if s.get("model", 0) == 0:
                spawn = s
                break
    if not spawn.is_empty():
        pos = spawn["position"]
        yaw = spawn["yaw_deg"]
    if overrides and _spawn_override != null:
        pos = _spawn_override
    if overrides and _yaw_override != null:
        yaw = _yaw_override
    player.spawn(pos, yaw)
    if overrides:
        player.head.rotation.x = deg_to_rad(_pitch)


## Unload the current world and load scene ``scene`` (a catalogue / generated
## name) in its place, the player at its spawn ``spawn_name`` (default: the
## first). A coroutine: it fades to a loading screen (unless already faded, as
## travel does) while the export parses on a worker thread. False (nothing
## changes) if the scene isn't exported or a switch is already under way.
func switch_scene(scene: String, spawn_name := "", queued := false) -> bool:
    if not world.is_exported(scene):
        push_warning("geogen: can't switch to '%s': not exported in %s" % [scene, world.generated_dir])
        return false
    if _switching:
        return false
    _switching = true
    var started := Time.get_ticks_msec()
    var fade_back := _fade.color.a < 0.99
    player.process_mode = Node.PROCESS_MODE_DISABLED
    if fade_back:
        await _fade_to(1.0)
    _loading.text = "Loading %s..." % scene.replace("_", " ")
    _loading.visible = true
    if not player.pose.is_empty():
        player.leave_pose()
    _focus = null
    _focus_npc = null
    _focus_affordance = {}
    var previous := world.scene_name
    world.select_scene(scene)
    world.prime_focus = null
    world.prime_spawn = spawn_name
    await world.load_all_async()
    world.prime_spawn = ""
    world.set_night(clock.night)
    _show_npc_labels()
    var spec := world.player_spec()
    if spec and not _manifest_arg:
        player_spec = spec
        player.apply_spec(spec)
    _place_player(world.world_aabb(), spawn_name, false)
    _remember_scene(scene)
    var p := player.global_position
    print("switched: %s" % JSON.stringify({"from": previous, "scene": scene, "spawn": spawn_name,
        "ms": Time.get_ticks_msec() - started, "x": p.x, "y": p.y, "z": p.z,
        "npcs": world.npcs.size(), "rooms": world.rooms.size()}))
    _loading.visible = false
    player.process_mode = Node.PROCESS_MODE_INHERIT
    if fade_back:
        await _fade_to(0.0)
    _switching = false
    _queued = queued
    _stats_in = 5    # freed/queued nodes are gone after a few frames
    return true


## A travel point fired: fade out, go to its scene (or stay) at its spawn, fade back in.
func _travel(travel: Dictionary) -> void:
    if _travelling:
        return
    _travelling = true
    var scene := str(travel.get("scene", ""))
    var spawn_name := str(travel.get("spawn", ""))
    player.scripted_move = null
    player.process_mode = Node.PROCESS_MODE_DISABLED
    await _fade_to(1.0)
    var ok := true
    if scene == "" or scene == world.scene_name:
        if not player.pose.is_empty():
            player.leave_pose()
        _place_player(world.world_aabb(), spawn_name, false)
        world.stream_focus = player.global_position
        world.arm_travel()
    else:
        ok = await switch_scene(scene, spawn_name)
    var p := player.global_position
    print("travelled: %s" % JSON.stringify({"scene": world.scene_name, "spawn": spawn_name, "ok": ok,
        "via": travel.get("on", "use"), "x": p.x, "y": p.y, "z": p.z, "yaw": rad_to_deg(player.rotation.y),
        "room": world.room_at(p + Vector3(0, 0.5, 0))}))
    if not ok:
        player.process_mode = Node.PROCESS_MODE_INHERIT
    await _fade_to(0.0)
    _travelling = false


func _fade_to(alpha: float) -> void:
    var tween := create_tween()
    tween.tween_property(_fade, "color:a", alpha, FADE_SECONDS)
    await tween.finished


## F7 / F8: the previous / next exported scene in catalogue order (wrapping).
func _step_scene(step: int) -> void:
    var names := world.scene_list().filter(func(s): return s["exported"]).map(func(s): return s["name"])
    if names.is_empty():
        return
    var i := names.find(world.scene_name)
    switch_scene(names[posmod(i + step, names.size())] if i >= 0 else names[0])


func _open_menu() -> void:
    _menu.open(world.scene_list(), world.scene_name)
    player.process_mode = Node.PROCESS_MODE_DISABLED
    Input.mouse_mode = Input.MOUSE_MODE_VISIBLE


func _menu_closed() -> void:
    player.process_mode = Node.PROCESS_MODE_INHERIT
    if DisplayServer.get_name() != "headless" and screenshot_path == "":
        Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _show_npc_labels() -> void:
    for npc in world.npcs:
        if is_instance_valid(npc):
            npc.label.visible = _npc_labels


## --list-scenes
func _print_scene_list() -> void:
    var scenes := world.scene_list()
    for s in scenes:
        print("%-20s %-9s %-13s %s" % [s["name"], s["group"], "exported" if s["exported"] else "not exported",
            s["description"]])
    print("scenes: %s" % JSON.stringify({"default": world.catalogue.get("default", ""), "scenes": scenes}))


## The last scene picked with this generated dir (interactive runs only), or "".
func _remembered_scene() -> String:
    if DisplayServer.get_name() == "headless":
        return ""
    var cfg := ConfigFile.new()
    if cfg.load(SETTINGS_PATH) != OK:
        return ""
    return str(cfg.get_value("last_scene", _settings_key(), ""))


func _remember_scene(scene: String) -> void:
    if DisplayServer.get_name() == "headless":
        return
    var cfg := ConfigFile.new()
    cfg.load(SETTINGS_PATH)
    cfg.set_value("last_scene", _settings_key(), scene)
    cfg.save(SETTINGS_PATH)


func _settings_key() -> String:
    return "dir_" + ProjectSettings.globalize_path(world.generated_dir).md5_text()


## --switch: run the queued switches on time, then the stats; quit when nothing else is pending.
func _run_switches(delta: float) -> void:
    if _stats_in > 0:
        _stats_in -= 1
        if _stats_in == 0:
            _print_switch_stats()
            if not _queued:
                return
            if _switches.is_empty() and _walk_left > 0.0 and _wait_left <= 0.0:
                player.scripted_move = Vector2(0, 1)
            if _switches.is_empty() and _walk_left <= 0.0 and not _print_status and screenshot_path == "" \
                    and _simulate <= 0.0 and quit_after_frames <= 0:
                get_tree().quit()
        return
    if _switches.is_empty() or _switching or _travelling:
        return
    _switch_clock += delta
    if _switch_clock < _switches[0]["at"]:
        return
    var next: Dictionary = _switches.pop_front()
    if world.is_exported(next["name"]):
        switch_scene(next["name"], "", true)
    else:
        print("switched: %s" % JSON.stringify({"scene": next["name"], "error": "not exported"}))
        _queued = true
        _stats_in = 1


func _print_switch_stats() -> void:
    var nav_regions := 0
    for map in NavigationServer3D.get_maps():
        nav_regions += NavigationServer3D.map_get_regions(map).size()
    print("switch stats: %s" % JSON.stringify({"scene": world.scene_name,
        "nodes": int(Performance.get_monitor(Performance.OBJECT_NODE_COUNT)),
        "orphans": int(Performance.get_monitor(Performance.OBJECT_ORPHAN_NODE_COUNT)),
        "objects": int(Performance.get_monitor(Performance.OBJECT_COUNT)),
        "resources": int(Performance.get_monitor(Performance.OBJECT_RESOURCE_COUNT)),
        "nav_regions": nav_regions,
        "bodies": get_tree().root.find_children("*", "CollisionObject3D", true, false).size(),
        "world_children": world.get_child_count()}))


func _frame_overview(aabb: AABB) -> void:
    if aabb.size == Vector3.ZERO:
        return
    var center := aabb.get_center()
    var dist := aabb.size.length() * 1.1
    overview.look_at_from_position(center + Vector3(0.7, 0.55, 1.0).normalized() * dist, center)


func _unhandled_input(event: InputEvent) -> void:
    if _menu != null and _menu.visible:
        return
    if event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_F6:
        _open_menu()
    elif event is InputEventKey and event.pressed and not event.echo and event.physical_keycode in [KEY_F7, KEY_F8]:
        _step_scene(-1 if event.physical_keycode == KEY_F7 else 1)
    elif event.is_action_pressed("geogen_toggle_overlay"):
        overlay.visible = not overlay.visible
    elif event.is_action_pressed("geogen_toggle_colliders"):
        world.show_colliders = not world.show_colliders
    elif event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_E:
        _press_use()
    elif event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_L:
        if _focus != null:
            _focus.toggle_lock(keys)
    elif event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_F5:
        _npc_labels = not _npc_labels
        _show_npc_labels()
    elif event is InputEventKey and event.pressed and event.physical_keycode == KEY_F4:
        if overview.current:
            player.camera.make_current()
        else:
            overview.make_current()


func _physics_process(delta: float) -> void:
    if player != null:
        world.stream_focus = player.global_position
    _sim_greet()
    if _simulate > 0.0:
        _sim_clock += delta
        if _sim_clock >= _simulate:
            _simulate = 0.0
            var reports := []
            for npc in world.npcs:
                if is_instance_valid(npc):
                    reports.append(npc.report())
            print("npc summary: %s" % JSON.stringify(reports))
            var traffic_reports := []
            for t in world.traffic:
                if is_instance_valid(t):
                    traffic_reports.append(t.report())
            print("traffic summary: %s" % JSON.stringify(traffic_reports))
            if not world.trains.is_empty():
                print("train summary: %s" % JSON.stringify(world.trains.filter(func(t): return is_instance_valid(t)).map(func(t): return t.report())))
            if screenshot_path != "":
                _frames = 0
                quit_after_frames = 3
            elif _print_status:
                _frames = 0
                quit_after_frames = 1      # print the status on the way out
            else:
                get_tree().quit()
    _update_focus()
    if _nav_query.size() == 2:
        _frames_nav += 1
        # Wait until the navigation map has synced the baked regions (and streamed tiles are baked).
        var synced := NavigationServer3D.map_get_iteration_id(get_world_3d().navigation_map) > 1 and _frames_nav > 5
        if world.streamer != null:
            if world.streamer.navigation_busy():
                _frames_nav = 0
            synced = _frames_nav > 5   # a few physics frames for the map to pick up new tiles
        if synced or _frames_nav > 1800:     # big maps take a while to sync
            _print_nav_path(_nav_query[0], _nav_query[1])
            get_tree().quit()
        return
    if _lock_aim and _focus != null:
        _lock_aim = false
        print("lock: %s -> %s" % [_focus.asset.name, _focus.toggle_lock(keys)])
    if _use_aim and (_focus != null or not _focus_affordance.is_empty()):
        _use_aim = false
        if _focus != null:
            print("used: %s (%s)" % [_focus.asset.name, _focus.interaction_name])
        else:
            print("used: %s (%s)" % [_focus_affordance["asset"], _focus_affordance["type"]])
        _press_use()
    if not _switches.is_empty() or _stats_in > 0 or _switching:
        return    # --walk / --wait start in the last world switched to
    if _wait_left > 0.0:
        _wait_left -= delta
        if _wait_left <= 0.0 and _walk_left > 0.0:
            player.scripted_move = Vector2(0, 1)
        return
    if _walk_left <= 0.0:
        return
    _walk_left -= delta
    if _walk_left <= 0.0:
        player.scripted_move = null
        var p := player.global_position
        var result := {"x": p.x, "y": p.y, "z": p.z, "scene": world.scene_name,
            "on_floor": player.is_on_floor(), "room": world.room_at(p + Vector3(0, 0.5, 0))}
        if world.streamer != null:
            result["stream"] = world.stream_report()
        print("walk result: %s" % JSON.stringify(result))
        get_tree().quit()


func _process(delta: float) -> void:
    _run_switches(delta)
    _follow_npc()
    if _simulate > 0.0:
        return   # --screenshot waits for the simulation
    if overlay.visible:
        var p := player.global_position
        var room := world.room_at(p + Vector3(0, 0.5, 0))
        var stream := world.stream_report()
        var chunk_info := "" if stream.is_empty() else "   chunks %d full / %d lod / %d interiors%s" % [
            stream["full"].size(), stream["lod"].size(), stream["interiors"].size(),
            " (+%d loading)" % stream["pending"] if stream["pending"] > 0 else ""]
        overlay.text = "%d fps   %s   %s%s\npos %.2f, %.2f, %.2f   room: %s%s%s\nWASD move  Shift sprint  Space jump  F1 overlay  F2 colliders  F3 fly  F4 overview  F6 scenes  F7/F8 prev/next  Esc mouse" % [
            Engine.get_frames_per_second(), clock.label(),
            world.scene_name if world.scene_name != "" else "all exports", chunk_info,
            p.x, p.y, p.z, room if room != "" else "-",
            "   [fly]" if player.flying else "",
            "   [colliders]" if world.show_colliders else ""]
    if quit_after_frames <= 0:
        return
    _frames += 1
    if _frames >= quit_after_frames:
        if _save_path != "":
            var f := FileAccess.open(_save_path, FileAccess.WRITE)
            f.store_string(JSON.stringify(world.save_state()))
            f.close()
        if _print_status:
            var p := player.global_position
            print("status: %s" % JSON.stringify({"scene": world.scene_name, "pose": player.pose.get("type", "stand"),
                "pose_asset": player.pose.get("asset", ""), "x": p.x, "y": p.y, "z": p.z,
                "room": world.room_at(p + Vector3(0, 0.5, 0)), "eye_y": player.camera.global_position.y,
                "states": world.save_state(), "stream": world.stream_report(),
                "clock": clock.label(), "night": world.night,
                "lamps_on": world.lights_by_name.values().filter(func(l): return is_instance_valid(l) and l.visible).size()}))
        if _print_lights:
            var report := {}
            for name in world.lights_by_name:
                var light: OmniLight3D = world.lights_by_name[name]
                report[name] = {"visible": light.visible, "energy": light.light_energy,
                    "range": light.omni_range, "y": snappedf(light.global_position.y, 0.01)}
            var total := get_tree().get_nodes_in_group("geogen_light").size()
            print("lights: %s" % JSON.stringify({"fixtures": report, "total": total,
                "light3d": get_tree().root.find_children("*", "OmniLight3D", true, false).size()}))
        if _print_skeletons:
            print("skeletons: %s" % JSON.stringify(_skeleton_report()))
        if screenshot_path != "":
            # Draw now: a background window may have skipped frames (macOS), and
            # the viewport texture would still hold an old one.
            RenderingServer.force_draw(false)
            var err := get_viewport().get_texture().get_image().save_png(screenshot_path)
            print("screenshot -> %s (%s)" % [screenshot_path, error_string(err)])
        get_tree().quit()


## --play: start (or freeze at --play=NAME@T) every matching animation in the loaded exports.
func _play_animations() -> void:
    var played := []
    for source: AnimationPlayer in world.find_children("*", "AnimationPlayer", true, false):
        var free := source
        for anim_name in source.get_animation_list():
            if anim_name != _play and not String(anim_name).ends_with("_" + _play):
                continue
            # One player plays one animation: each further match (another character) gets a copy.
            var ap := free if free != null else source.duplicate() as AnimationPlayer
            if free == null:
                source.add_sibling(ap)
            free = null
            if _play_at < 0.0:
                ap.get_animation(anim_name).loop_mode = Animation.LOOP_LINEAR   # imported clips don't loop
            ap.play(anim_name)
            if _play_at >= 0.0:
                ap.seek(_play_at, true)
                ap.pause()
            played.append(anim_name)
    print("play: %s" % JSON.stringify(played))


## --skeletons: bones (global positions), skinned meshes (bounds of the CPU-baked pose) and animations.
func _skeleton_report() -> Dictionary:
    var skeletons := []
    for sk: Skeleton3D in world.find_children("*", "Skeleton3D", true, false):
        var bones := {}
        for b in sk.get_bone_count():
            var p := sk.global_transform * sk.get_bone_global_pose(b).origin
            bones[sk.get_bone_name(b)] = [snappedf(p.x, 0.0001), snappedf(p.y, 0.0001), snappedf(p.z, 0.0001)]
        var meshes := []
        for mi: MeshInstance3D in sk.find_children("*", "MeshInstance3D", true, false):
            if mi.skin == null:
                continue
            var box := _skinned_bounds(mi, sk)
            meshes.append({"name": String(mi.name), "min": [box.position.x, box.position.y, box.position.z],
                "max": [box.end.x, box.end.y, box.end.z]})
        skeletons.append({"path": String(world.get_path_to(sk)), "bones": bones, "meshes": meshes})
    var animations := {}
    for ap: AnimationPlayer in world.find_children("*", "AnimationPlayer", true, false):
        animations[String(world.get_path_to(ap))] = Array(ap.get_animation_list())
    return {"skeletons": skeletons, "animations": animations}


## World bounds of a skinned mesh in its current pose, skinned on the CPU the way the renderer
## does it (vertex -> bind pose -> bone global pose -> mesh instance).
static func _skinned_bounds(mi: MeshInstance3D, sk: Skeleton3D) -> AABB:
    var joint_xforms: Array[Transform3D] = []
    for i in mi.skin.get_bind_count():
        var bone := mi.skin.get_bind_bone(i)
        if bone < 0:
            bone = sk.find_bone(mi.skin.get_bind_name(i))
        joint_xforms.append(sk.get_bone_global_pose(bone) * mi.skin.get_bind_pose(i))
    var box := AABB()
    var first := true
    for surface in mi.mesh.get_surface_count():
        var arrays := mi.mesh.surface_get_arrays(surface)
        var verts: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
        var bones: PackedInt32Array = arrays[Mesh.ARRAY_BONES]
        var weights: PackedFloat32Array = arrays[Mesh.ARRAY_WEIGHTS]
        var per := bones.size() / maxi(verts.size(), 1)
        for v in verts.size():
            var p := Vector3.ZERO
            for k in per:
                var w := weights[v * per + k]
                if w > 0.0:
                    p += (joint_xforms[bones[v * per + k]] * verts[v]) * w
            p = mi.global_transform * p
            if first:
                box = AABB(p, Vector3.ZERO)
                first = false
            else:
                box = box.expand(p)
    return box


## --greet=NAME[@S]: greet the NPC whose name starts with NAME once S seconds of world time have
## passed (headless tests); prints "greeted: {...}".
var _greet_clock := 0.0
func _sim_greet() -> void:
    if _greet.is_empty():
        return
    _greet_clock += get_physics_process_delta_time()
    for prefix in _greet.keys():
        if _greet_clock < float(_greet[prefix]):
            continue
        for npc in world.npcs:
            if is_instance_valid(npc) and String(npc.name).begins_with(prefix):
                if npc.can_greet():
                    print("greeted: %s" % JSON.stringify({"npc": String(npc.name), "ok": npc.greet()}))
                    _greet.erase(prefix)
                break


## What the player is aiming at within reach, and its prompt ("E: Open").
func _update_focus() -> void:
    _focus = null
    _focus_affordance = {}
    _focus_npc = null
    if player != null and player.camera.current and player.pose.is_empty():
        var cam := player.camera
        var from := cam.global_position
        var to := from - cam.global_basis.z * player_spec.reach
        var query := PhysicsRayQueryParameters3D.create(from, to)
        query.exclude = [player.get_rid()]
        var hit := get_world_3d().direct_space_state.intersect_ray(query)
        if hit:
            var it := world.interaction_for(hit["collider"])
            if hit["collider"] is GeogenNpc:
                if (hit["collider"] as GeogenNpc).can_greet():
                    _focus_npc = hit["collider"]
            elif it != null and it.can_use():
                _focus = it
            else:
                _focus_affordance = world.affordance_near(hit["position"])
    if _prompt:
        if player != null and not player.pose.is_empty():
            _prompt.text = "E: Stand up"
        elif _focus_npc != null:
            _prompt.text = "E: %s" % _focus_npc.greet_prompt()
        elif _focus != null:
            _prompt.text = "E: %s" % _focus.prompt(keys)
        elif not _focus_affordance.is_empty():
            _prompt.text = "E: %s" % {"sit": "Sit", "lie": "Lie down", "use": "Use", "stand": "Stand here"}.get(
                _focus_affordance["type"], "Use")
        else:
            _prompt.text = ""


func _press_use() -> void:
    if not player.pose.is_empty():
        player.leave_pose()
    elif _focus_npc != null:
        _focus_npc.greet()
    elif _focus != null:
        _focus.use()
    elif not _focus_affordance.is_empty():
        player.take_pose(_focus_affordance)


## Print the navigation path between two floor points as JSON (length, points).
func _print_nav_path(a: Vector3, b: Vector3) -> void:
    var map := get_world_3d().navigation_map
    var from := NavigationServer3D.map_get_closest_point(map, a)
    var to := NavigationServer3D.map_get_closest_point(map, b)
    var path := NavigationServer3D.map_get_path(map, from, to, true)
    var length := 0.0
    var points := []
    for i in path.size():
        points.append([snappedf(path[i].x, 0.01), snappedf(path[i].y, 0.01), snappedf(path[i].z, 0.01)])
        if i > 0:
            length += path[i].distance_to(path[i - 1])
    print("nav path: %s" % JSON.stringify({"length": length, "reached": path.size() > 0 and path[-1].distance_to(to) < 0.05,
        "end_gap": to.distance_to(b), "points": points}))


## --camera=follow: keep the overview camera on an NPC, from the first of a ring
## of viewpoints (front first) with a clear line of sight under any ceiling.
func _follow_npc() -> void:
    if _follow == null or not overview.current:
        return
    for npc in world.npcs:
        if is_instance_valid(npc) and String(npc.name).begins_with(_follow):
            var head := npc.global_position + Vector3(0, 1.3, 0)
            var space := get_world_3d().direct_space_state
            var best := head + Vector3(0, 6, 0)
            for step in [0, 1, -1, 2, -2, 3, -3, 4]:
                var dir := Basis(Vector3.UP, npc.rotation.y + step * PI / 4.0) * Vector3(0, 0, 1)
                var eye := head + dir * 2.6 + Vector3(0, 0.6, 0)
                var query := PhysicsRayQueryParameters3D.create(head, eye + dir * 0.3)
                query.exclude = [npc.get_rid()]
                if space.intersect_ray(query).is_empty():
                    best = eye
                    break
            overview.look_at_from_position(best, npc.global_position + Vector3(0, 0.8, 0))
            return
