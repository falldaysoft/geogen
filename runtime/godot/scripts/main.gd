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

	world.world_loaded.connect(_on_world_loaded)
	world.load_all()
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
	if walk_seconds > 0.0:
		_walk_left = walk_seconds
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
	if aabb.size != Vector3.ZERO:
		pos = Vector3(aabb.get_center().x, 0, aabb.end.z + 3.0)
	if _spawn_override == null and not world.spawns.is_empty():
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
	elif event is InputEventKey and event.pressed and event.physical_keycode == KEY_F4:
		if overview.current:
			player.camera.make_current()
		else:
			overview.make_current()


func _physics_process(delta: float) -> void:
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
		if screenshot_path != "":
			var err := get_viewport().get_texture().get_image().save_png(screenshot_path)
			print("screenshot -> %s (%s)" % [screenshot_path, error_string(err)])
		get_tree().quit()
