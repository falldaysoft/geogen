class_name GeogenSound
extends RefCounted
## Sound hooks from extras.geogen.vehicle.sound (geogen/vehicles.py), synthesised here: no
## audio files. An engine is a short seamless loop (a few harmonics of `base` Hz plus some
## noise for `roughness`) whose pitch follows the vehicle's speed; a horn is a two-tone chord.
## Streams are cached per spec, so a fleet shares a handful of samples.
##
## The HUD's chimes (`CHIMES`, `chime`) are made the same way: a few bell notes per kind.

const RATE := 22050
## HUD chimes: per kind, the notes [Hz, start s, length s] of a soft bell.
const CHIMES := {
	"started": [[523.25, 0.0, 0.22], [783.99, 0.11, 0.3]],
	"objective_done": [[880.0, 0.0, 0.2]],
	"complete": [[523.25, 0.0, 0.2], [659.25, 0.09, 0.2], [783.99, 0.18, 0.2], [1046.5, 0.27, 0.42]],
	"failed": [[311.13, 0.0, 0.26], [233.08, 0.16, 0.42]],
	# What rules `play` (geogen.rules.SOUNDS names these and the four above).
	"notice": [[659.25, 0.0, 0.18], [987.77, 0.08, 0.26]],
	"refused": [[196.0, 0.0, 0.12], [185.0, 0.1, 0.2]],
	"success": [[587.33, 0.0, 0.16], [880.0, 0.1, 0.34]],
	"click": [[1760.0, 0.0, 0.04]],
	# Spawned items (flows.gd): an effector's shot, a hit, an item destroyed.
	"shot": [[1318.51, 0.0, 0.05]],
	"hit": [[220.0, 0.0, 0.06]],
	"destroyed": [[392.0, 0.0, 0.08], [196.0, 0.05, 0.18]],
}
static var _cache := {}


## A looping engine sample for ``spec`` ({base, per_speed, volume_db, roughness}).
static func engine_stream(spec: Dictionary) -> AudioStreamWAV:
	var key := "engine:%s:%s" % [spec.get("base", 45.0), spec.get("roughness", 0.35)]
	if _cache.has(key):
		return _cache[key]
	var base := float(spec.get("base", 45.0))
	var rough := float(spec.get("roughness", 0.35))
	# A whole number of cycles so the loop is seamless.
	var cycles := maxi(int(round(base * 0.5)), 1)
	var n := int(round(cycles * RATE / base))
	var rng := RandomNumberGenerator.new()
	rng.seed = 7
	var noise := 0.0
	var data := PackedByteArray()
	data.resize(n * 2)
	for i in n:
		var t := float(i) / n * cycles * TAU
		var v := 0.55 * sin(t) + 0.25 * sin(2.0 * t + 0.4) + 0.12 * sin(3.0 * t + 1.1) + 0.06 * sin(5.0 * t)
		noise = lerpf(noise, rng.randf_range(-1.0, 1.0), 0.2)
		v = v * (1.0 - rough * 0.5) + noise * rough * 0.5
		data.encode_s16(i * 2, int(clampf(v, -1.0, 1.0) * 26000.0))
	var stream := _wav(data, n)
	stream.loop_mode = AudioStreamWAV.LOOP_FORWARD
	stream.loop_begin = 0
	stream.loop_end = n
	_cache[key] = stream
	return stream


## A horn: its tones together for `seconds`, with soft attack and release.
static func horn_stream(spec: Dictionary) -> AudioStreamWAV:
	var tones: Array = spec.get("tones", [370.0, 440.0])
	var seconds := float(spec.get("seconds", 1.2))
	var key := "horn:%s:%s" % [str(tones), seconds]
	if _cache.has(key):
		return _cache[key]
	var n := int(seconds * RATE)
	var data := PackedByteArray()
	data.resize(n * 2)
	for i in n:
		var t := float(i) / RATE
		var env := minf(1.0, minf(t / 0.05, (seconds - t) / 0.15))
		var v := 0.0
		for f in tones:
			var ph := t * float(f) * TAU
			v += (sin(ph) + 0.3 * sin(2.0 * ph) + 0.15 * sin(3.0 * ph)) / tones.size()
		data.encode_s16(i * 2, int(clampf(v * 0.6 * env, -1.0, 1.0) * 30000.0))
	var stream := _wav(data, n)
	_cache[key] = stream
	return stream


## A chime for a HUD notification of ``kind`` (a key of CHIMES): each note is a sine with a
## little of its octave, struck and left to die away.
static func chime_stream(kind: String) -> AudioStreamWAV:
	var key := "chime:%s" % kind
	if _cache.has(key):
		return _cache[key]
	var notes: Array = CHIMES[kind]
	var seconds := 0.0
	for note in notes:
		seconds = maxf(seconds, float(note[1]) + float(note[2]))
	var n := int(seconds * RATE)
	var data := PackedByteArray()
	data.resize(n * 2)
	for i in n:
		var t := float(i) / RATE
		var v := 0.0
		for note in notes:
			var at := t - float(note[1])
			var length := float(note[2])
			if at < 0.0 or at > length:
				continue
			var env := minf(at / 0.006, 1.0) * exp(-at * 4.5 / length) * minf((length - at) / 0.02, 1.0)
			var ph := at * float(note[0]) * TAU
			v += (sin(ph) + 0.25 * sin(2.0 * ph)) * env
		data.encode_s16(i * 2, int(clampf(v * 0.4, -1.0, 1.0) * 30000.0))
	var stream := _wav(data, n)
	_cache[key] = stream
	return stream


## Play ``kind``'s chime on ``player`` (not positional: it is the HUD's); false if there is none.
static func chime(player: AudioStreamPlayer, kind: String) -> bool:
	if player == null or not CHIMES.has(kind):
		return false
	player.stream = chime_stream(kind)
	player.play()
	return true


static func _wav(data: PackedByteArray, _n: int) -> AudioStreamWAV:
	var stream := AudioStreamWAV.new()
	stream.format = AudioStreamWAV.FORMAT_16_BITS
	stream.mix_rate = RATE
	stream.stereo = false
	stream.data = data
	return stream


## An engine player on ``node`` (a vehicle), or null if it declares no engine sound.
static func add_engine(node: Node3D, info: Dictionary) -> AudioStreamPlayer3D:
	var spec: Dictionary = info.get("sound", {}).get("engine", {})
	if spec.is_empty():
		return null
	var player := AudioStreamPlayer3D.new()
	player.name = "Engine"
	player.stream = engine_stream(spec)
	player.volume_db = float(spec.get("volume_db", -10.0))
	player.unit_size = 6.0
	player.max_distance = 70.0
	player.position = Vector3(0, 0.8, 0)
	player.set_meta("geogen_engine", spec)
	node.add_child(player)
	player.play()
	return player


## Pitch the engine for ``speed`` m/s (idle at rest).
static func set_speed(player: AudioStreamPlayer3D, speed: float) -> void:
	if player == null:
		return
	var spec: Dictionary = player.get_meta("geogen_engine", {})
	var base := float(spec.get("base", 45.0))
	player.pitch_scale = clampf((base + float(spec.get("per_speed", 4.0)) * speed) / base, 0.5, 4.0)


## Sound ``node``'s horn once (if it has one); returns whether it did.
static func horn(node: Node3D, info: Dictionary) -> bool:
	var spec: Dictionary = info.get("sound", {}).get("horn", {})
	if spec.is_empty():
		return false
	var player := node.get_node_or_null("Horn") as AudioStreamPlayer3D
	if player == null:
		player = AudioStreamPlayer3D.new()
		player.name = "Horn"
		player.stream = horn_stream(spec)
		player.volume_db = float(spec.get("volume_db", 0.0))
		player.unit_size = 25.0
		player.max_distance = 400.0
		player.position = Vector3(0, 3.0, 0)
		node.add_child(player)
	player.play()
	return true
