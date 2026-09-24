extends Node3D
## Root of the geogen runtime: loads exports from generated/, spawns a
## first-person player sized from the manifest, and shows a debug overlay.
##
## User args (after `--`):
##   --scene NAME | --scene=NAME   load generated/NAME.glb (default: every export)
##   --generated=DIR               read exports from DIR instead of res://generated
##   --spawn=X,Y,Z  --yaw=DEG      player start (default: the export's first spawn
##                                 point, else in front of the model)
##   --camera=overview             start from the overview camera (F4 toggles)
##   --colliders                   show collision shapes (F2 toggles)
##   --walk=SECONDS                walk forward, print the final position, quit
##   --use=ASSET                   use ASSET's interactions at start (e.g. open "door");
##                                 --use=@aim presses E on whatever the player looks at
##   --wait=SECONDS                wait before the --walk starts (let a door swing)
##   --nav=AX,AZ:BX,BZ             print the navigation path between two floor points, quit
##   --lights                      print the fixture lights (JSON) when quitting
##   --playtest[=N]                check every room/interaction is reachable, walk N routes
##                                 with a bot (default 6), print "playtest: {...}", quit
##
## In game, look at something interactive within reach and press E.
##   --screenshot=PATH             save a frame and quit
##   --quit-after=N                quit after N frames
##   --manifest=PATH               print the player spec read from PATH

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
var _nav_query := []
var _frames_nav := 0
var _playtest_walks := -1
var _wait_left := 0.0
var _focus: GeogenInteraction = null
var _prompt: Label

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
        elif arg == "--use=@aim":
            _use_aim = true
        elif arg.begins_with("--use="):
            _use_assets.append(value)
        elif arg.begins_with("--wait="):
            _wait_left = float(value)
        elif arg.begins_with("--quit-after="):
            quit_after_frames = int(value)
        elif arg.begins_with("--screenshot="):
            screenshot_path = value
            if quit_after_frames <= 0:
                quit_after_frames = 10  # let shadows/SSAO settle
        elif arg.begins_with("--manifest="):
            var spec := PlayerSpec.from_manifest(value)
            if spec:
                player_spec = spec
                _manifest_arg = true
        i += 1
    print("geogen runtime ready (Godot %s)" % Engine.get_version_info().string)

    world.open_before_bake = _playtest_walks >= 0
    world.world_loaded.connect(_on_world_loaded)
    world.interaction_event.connect(func(asset: String, interaction: String, state: String, event: String):
        print("interaction event: %s" % JSON.stringify(
            {"asset": asset, "interaction": interaction, "state": state, "event": event})))
    world.load_all()
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
        if _wait_left <= 0.0:
            player.scripted_move = Vector2(0, 1)
    elif DisplayServer.get_name() != "headless" and screenshot_path == "":
        Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _on_world_loaded(aabb: AABB) -> void:
    _frame_overview(aabb)
    $ScaleReference.visible = aabb.size == Vector3.ZERO


## Spawn in front (+Z) of whatever was loaded, facing it.
func _place_player(aabb: AABB) -> void:
    var pos := Vector3(0, 0, 5)
    var yaw := 0.0
    if not world.model_aabbs.is_empty():
        aabb = world.model_aabbs[0]  # several exports load in a row: start at the first
    if aabb.size != Vector3.ZERO:
        pos = Vector3(aabb.get_center().x, 0, aabb.end.z + 3.0)
    # Manifest spawns only make sense when a single export is loaded.
    if _spawn_override == null and world.model_aabbs.size() == 1 and not world.spawns.is_empty():
        pos = world.spawns[0]["position"]
        yaw = world.spawns[0]["yaw_deg"]
    if _spawn_override != null:
        pos = _spawn_override
    player.spawn(pos, _yaw_override if _yaw_override != null else yaw)


func _frame_overview(aabb: AABB) -> void:
    if aabb.size == Vector3.ZERO:
        return
    var center := aabb.get_center()
    var dist := aabb.size.length() * 1.1
    overview.look_at_from_position(center + Vector3(0.7, 0.55, 1.0).normalized() * dist, center)


func _unhandled_input(event: InputEvent) -> void:
    if event.is_action_pressed("geogen_toggle_overlay"):
        overlay.visible = not overlay.visible
    elif event.is_action_pressed("geogen_toggle_colliders"):
        world.show_colliders = not world.show_colliders
    elif event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_E:
        if _focus != null:
            _focus.use()
    elif event is InputEventKey and event.pressed and event.physical_keycode == KEY_F4:
        if overview.current:
            player.camera.make_current()
        else:
            overview.make_current()


func _physics_process(delta: float) -> void:
    _update_focus()
    if _nav_query.size() == 2:
        _frames_nav += 1
        # Wait until the navigation map has synced the baked regions.
        var synced := NavigationServer3D.map_get_iteration_id(get_world_3d().navigation_map) > 1
        if synced or _frames_nav > 120:
            _print_nav_path(_nav_query[0], _nav_query[1])
            get_tree().quit()
        return
    if _use_aim and _focus != null:
        _use_aim = false
        print("used: %s (%s)" % [_focus.asset.name, _focus.interaction_name])
        _focus.use()
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
        print("walk result: %s" % JSON.stringify({"x": p.x, "y": p.y, "z": p.z,
            "on_floor": player.is_on_floor(), "room": world.room_at(p + Vector3(0, 0.5, 0))}))
        get_tree().quit()


func _process(_delta: float) -> void:
    if overlay.visible:
        var p := player.global_position
        var room := world.room_at(p + Vector3(0, 0.5, 0))
        overlay.text = "%d fps   %s\npos %.2f, %.2f, %.2f   room: %s%s%s\nWASD move  Shift sprint  Space jump  F1 overlay  F2 colliders  F3 fly  F4 overview  Esc mouse" % [
            Engine.get_frames_per_second(),
            world.scene_name if world.scene_name != "" else "all exports",
            p.x, p.y, p.z, room if room != "" else "-",
            "   [fly]" if player.flying else "",
            "   [colliders]" if world.show_colliders else ""]
    if quit_after_frames <= 0:
        return
    _frames += 1
    if _frames >= quit_after_frames:
        if _print_lights:
            var report := {}
            for name in world.lights_by_name:
                var light: OmniLight3D = world.lights_by_name[name]
                report[name] = {"visible": light.visible, "energy": light.light_energy,
                    "range": light.omni_range, "y": snappedf(light.global_position.y, 0.01)}
            var total := get_tree().get_nodes_in_group("geogen_light").size()
            print("lights: %s" % JSON.stringify({"fixtures": report, "total": total,
                "light3d": get_tree().root.find_children("*", "OmniLight3D", true, false).size()}))
        if screenshot_path != "":
            var err := get_viewport().get_texture().get_image().save_png(screenshot_path)
            print("screenshot -> %s (%s)" % [screenshot_path, error_string(err)])
        get_tree().quit()


## What the player is aiming at within reach, and its prompt ("E: Open").
func _update_focus() -> void:
    _focus = null
    if player != null and player.camera.current:
        var cam := player.camera
        var from := cam.global_position
        var to := from - cam.global_basis.z * player_spec.reach
        var query := PhysicsRayQueryParameters3D.create(from, to)
        query.exclude = [player.get_rid()]
        var hit := get_world_3d().direct_space_state.intersect_ray(query)
        if hit:
            var it := world.interaction_for(hit["collider"])
            if it != null and it.can_use():
                _focus = it
    if _prompt:
        _prompt.text = "E: %s" % _focus.prompt() if _focus else ""


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
