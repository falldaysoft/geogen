extends Node3D
## Root of the geogen runtime. For now it reports that it booted and, given
## `-- --manifest=path`, the player spec it read; the world loader and player
## arrive with geogen-3cc.26.

## Quit after this many frames (0 = run forever). Set via `-- --quit-after=N`.
var quit_after_frames := 0
## Save the viewport here, then quit. Set via `-- --screenshot=/path/out.png`.
var screenshot_path := ""
## Player constraints from the manifest (defaults if none was given).
var player_spec := PlayerSpec.new()
var _frames := 0


func _ready() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--quit-after="):
			quit_after_frames = int(arg.get_slice("=", 1))
		elif arg.begins_with("--screenshot="):
			screenshot_path = arg.get_slice("=", 1)
			if quit_after_frames <= 0:
				quit_after_frames = 10  # let shadows/SSAO settle
		elif arg.begins_with("--manifest="):
			var spec := PlayerSpec.from_manifest(arg.get_slice("=", 1))
			if spec:
				player_spec = spec
	print("geogen runtime ready (Godot %s)" % Engine.get_version_info().string)
	print("player spec: %s" % JSON.stringify(player_spec.to_dict()))


func _process(_delta: float) -> void:
	if quit_after_frames <= 0:
		return
	_frames += 1
	if _frames >= quit_after_frames:
		if screenshot_path != "":
			var err := get_viewport().get_texture().get_image().save_png(screenshot_path)
			print("screenshot -> %s (%s)" % [screenshot_path, error_string(err)])
		get_tree().quit()
