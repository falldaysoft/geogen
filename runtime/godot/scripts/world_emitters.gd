class_name GeogenWorldEmitters
extends RefCounted
## What parts give off (extras.geogen.emitter and .sound; geogen/emitters.py): each is described
## physically and mapped here to a GPUParticles3D and a looping AudioStreamPlayer3D. Both follow
## the part's `running` state (state.emitter_state), which a binding or any rule may set.
## No textures or audio files: sprites and sounds are made here and shared between emitters.

## Seconds of a sound's loop.
const LOOP_SECONDS := 3.0
## Seconds the end of a loop is blended into its start, so it has no seam.
const LOOP_BLEND := 0.25

var world: WorldLoader
var _parts := {}                 # stable ID -> {node, particles, sound}
static var _sprites := {}        # material name -> Texture2D
static var _streams := {}        # sound spec -> AudioStreamWAV


func clear() -> void:
    _parts.clear()


## Forget the parts under what ``inside`` says belongs to a freed chunk.
func forget(inside: Callable) -> void:
    for uid in _parts.keys():
        if inside.call(_parts[uid]["node"]):
            _parts.erase(uid)


## Give every part under ``root`` that declares one its particles and sound.
func add(root: Node) -> void:
    for node: Node3D in root.find_children("*", "Node3D", true, false):
        var extras := WorldLoader.geogen_extras(node)
        var emitter = extras.get("emitter")
        var sound = extras.get("sound")
        if not emitter is Dictionary and not sound is Dictionary:
            continue
        var entry := {"node": node, "particles": null, "sound": null}
        if emitter is Dictionary:
            entry["particles"] = make_particles(emitter)
            node.add_child(entry["particles"])
        if sound is Dictionary:
            entry["sound"] = make_sound(sound)
            node.add_child(entry["sound"])
        var uid := WorldLoader.uid_of(node)
        var block: Dictionary = emitter if emitter is Dictionary else sound
        _set_running(entry, bool(block.get("running", true)))
        if uid != "":
            _parts[uid] = entry


## Each part's `running` state, when a package (or a save) loads and when a chunk adds parts.
func sync() -> void:
    for uid in _parts:
        var running = world.rules.get_value(uid + "#running")
        if running is bool:
            _set_running(_parts[uid], running)


func on_running(event: Dictionary) -> void:
    if event["type"] == "changed" and event["name"] == "running" and _parts.has(event["subject"]):
        _set_running(_parts[event["subject"]], bool(event["data"].get("to", true)))


## What the control API reports: each part by stable ID.
func report() -> Dictionary:
    var out := {}
    for uid in _parts:
        var entry: Dictionary = _parts[uid]
        if not is_instance_valid(entry["node"]):
            continue
        var row := {}
        var particles: GPUParticles3D = entry["particles"]
        if particles != null:
            row["emitting"] = particles.emitting
            row["amount"] = particles.amount
            row["position"] = _triple(particles.global_position)
        var sound: AudioStreamPlayer3D = entry["sound"]
        if sound != null:
            row["sounding"] = sound.playing
            row["range"] = sound.max_distance
        out[uid] = row
    return out


func _set_running(entry: Dictionary, running: bool) -> void:
    if not is_instance_valid(entry["node"]):
        return
    var particles: GPUParticles3D = entry["particles"]
    if particles != null:
        particles.emitting = running      # those already in the air finish their flight
    var sound: AudioStreamPlayer3D = entry["sound"]
    if sound != null:
        sound.autoplay = running          # a world still being set up isn't in the tree yet
        if sound.is_inside_tree() and sound.playing != running:
            if running:
                sound.play()
            else:
                sound.stop()


static func _vec(value: Variant) -> Vector3:
    return Vector3(value[0], value[1], value[2])


static func _triple(v: Vector3) -> Array:
    return [snappedf(v.x, 0.001), snappedf(v.y, 0.001), snappedf(v.z, 0.001)]


# --- particles ---------------------------------------------------------------------------------

static func make_particles(spec: Dictionary) -> GPUParticles3D:
    var lifetime := float(spec["lifetime"])
    var particles := GPUParticles3D.new()
    particles.name = "Emitter"
    particles.position = _vec(spec["offset"])
    particles.amount = maxi(int(ceil(float(spec["rate"]) * lifetime)), 1)
    particles.lifetime = lifetime
    particles.preprocess = lifetime             # already flowing when the player first sees it
    particles.local_coords = false              # what has left stays behind when the part moves
    particles.draw_order = GPUParticles3D.DRAW_ORDER_VIEW_DEPTH
    particles.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
    particles.visibility_aabb = reach(spec)
    particles.process_material = _process_material(spec)
    particles.draw_pass_1 = _quad(spec)
    if spec["material"] == "streak":
        particles.transform_align = GPUParticles3D.TRANSFORM_ALIGN_Z_BILLBOARD_Y_TO_VELOCITY
    return particles


## The box a particle can reach from the emitter in its lifetime (a renderer culls by this).
static func reach(spec: Dictionary) -> AABB:
    var t := float(spec["lifetime"])
    var speed := maxf(float(spec["speed"][0]), float(spec["speed"][1]))
    var pull := Vector3(0, -float(spec["gravity"]), 0) + _vec(spec["wind"])
    var drift := pull * 0.5 * t * t
    var half := _vec(spec["area"]) * 0.5
    var size := maxf(float(spec["size"][0]), float(spec["size"][1]))
    var margin := Vector3.ONE * (speed * t + size)
    var box := AABB(-half - margin, (half + margin) * 2.0)
    return box.merge(AABB(drift - half - margin, (half + margin) * 2.0))


static func _process_material(spec: Dictionary) -> ParticleProcessMaterial:
    var m := ParticleProcessMaterial.new()
    m.direction = _vec(spec["direction"])
    m.spread = float(spec["spread"])
    m.initial_velocity_min = float(spec["speed"][0])
    m.initial_velocity_max = float(spec["speed"][1])
    m.gravity = Vector3(0, -float(spec["gravity"]), 0) + _vec(spec["wind"])
    m.damping_min = float(spec["drag"])
    m.damping_max = float(spec["drag"])
    var area := _vec(spec["area"])
    if area.length_squared() > 0.0:
        m.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_BOX
        m.emission_box_extents = area * 0.5
    # Size over life, as a fraction of the larger of the two sizes (the quad's).
    var birth := float(spec["size"][0])
    var death := float(spec["size"][1])
    var largest := maxf(maxf(birth, death), 0.0001)
    var curve := Curve.new()
    curve.add_point(Vector2(0.0, birth / largest))
    curve.add_point(Vector2(1.0, death / largest))
    var scale := CurveTexture.new()
    scale.curve = curve
    m.scale_curve = scale
    # Colour, fading in quickly and out over the last of its life.
    var c: Array = spec["color"]
    var opacity := float(spec["opacity"])
    var gradient := Gradient.new()
    gradient.offsets = PackedFloat32Array([0.0, 0.12, 0.6, 1.0])
    gradient.colors = PackedColorArray([Color(c[0], c[1], c[2], 0.0), Color(c[0], c[1], c[2], opacity),
        Color(c[0], c[1], c[2], opacity), Color(c[0], c[1], c[2], 0.0)])
    var ramp := GradientTexture1D.new()
    ramp.gradient = gradient
    m.color_ramp = ramp
    return m


static func _quad(spec: Dictionary) -> QuadMesh:
    var kind := String(spec["material"])
    var across := maxf(float(spec["size"][0]), float(spec["size"][1]))
    var quad := QuadMesh.new()
    quad.size = Vector2(across, across)
    var m := StandardMaterial3D.new()
    m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
    m.vertex_color_use_as_albedo = true
    m.vertex_color_is_srgb = true           # `color` means what it means on a material
    m.albedo_texture = _sprite(kind)
    m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
    m.billboard_mode = BaseMaterial3D.BILLBOARD_PARTICLES
    m.billboard_keep_scale = true
    match kind:
        "soft":
            if RenderingServer.get_current_rendering_method() != "gl_compatibility":
                m.proximity_fade_enabled = true    # no hard line where a puff meets a wall
                m.proximity_fade_distance = across * 0.5
        "glow":
            m.blend_mode = BaseMaterial3D.BLEND_MODE_ADD
        "streak":
            # As long as the distance it falls in a frame or two; the particle aligns it to its velocity.
            var speed := maxf(float(spec["speed"][0]), float(spec["speed"][1]))
            quad.size = Vector2(across, maxf(across, speed * 0.035))
            m.billboard_mode = BaseMaterial3D.BILLBOARD_DISABLED
            m.cull_mode = BaseMaterial3D.CULL_DISABLED
    quad.material = m
    return quad


## A round sprite: `droplet` has a firm edge, the others fade from the centre.
static func _sprite(kind: String) -> Texture2D:
    if _sprites.has(kind):
        return _sprites[kind]
    var gradient := Gradient.new()
    if kind == "droplet":
        gradient.offsets = PackedFloat32Array([0.0, 0.6, 1.0])
        gradient.colors = PackedColorArray([Color.WHITE, Color.WHITE, Color(1, 1, 1, 0)])
    else:
        gradient.offsets = PackedFloat32Array([0.0, 0.45, 1.0])
        gradient.colors = PackedColorArray([Color.WHITE, Color(1, 1, 1, 0.45), Color(1, 1, 1, 0)])
    var sprite := GradientTexture2D.new()
    sprite.gradient = gradient
    sprite.width = 64
    sprite.height = 64
    sprite.fill = GradientTexture2D.FILL_RADIAL
    sprite.fill_from = Vector2(0.5, 0.5)
    sprite.fill_to = Vector2(1.0, 0.5)
    _sprites[kind] = sprite
    return sprite


# --- sound -------------------------------------------------------------------------------------

static func make_sound(spec: Dictionary) -> AudioStreamPlayer3D:
    var player := AudioStreamPlayer3D.new()
    player.name = "Sound"
    player.position = _vec(spec["offset"])
    player.stream = ambient_stream(spec)
    player.volume_db = float(spec["volume_db"])
    player.max_distance = float(spec["range"])
    player.unit_size = float(spec["range"]) * 0.25
    return player


## A seamless loop: noise in `band`, a `hum` under it, a slow `flutter` and random `crackle`.
static func ambient_stream(spec: Dictionary) -> AudioStreamWAV:
    var key := "%s:%s:%s:%s" % [spec["band"], spec["hum"], spec["flutter"], spec["crackle"]]
    if _streams.has(key):
        return _streams[key]
    var rate := GeogenSound.RATE
    var n := int(LOOP_SECONDS * rate)
    var blend := int(LOOP_BLEND * rate)
    var rng := RandomNumberGenerator.new()
    rng.seed = 11
    # One-pole filters: `low` follows the signal up to the top of the band, `floor_` takes out what is under it.
    var top := 1.0 - exp(-TAU * float(spec["band"][1]) / rate)
    var bottom := 1.0 - exp(-TAU * float(spec["band"][0]) / rate)
    var low := 0.0
    var floor_ := 0.0
    var crackle := float(spec["crackle"])
    var pop := 0.0
    var samples := PackedFloat32Array()
    samples.resize(n + blend)
    var peak := 0.0001
    for i in n + blend:
        low += (rng.randf_range(-1.0, 1.0) - low) * top
        floor_ += (low - floor_) * bottom
        var v := (low - floor_) * 2.0
        if crackle > 0.0:
            if rng.randf() < crackle * 0.0012:
                pop = rng.randf_range(-1.0, 1.0) * 1.6
            pop *= 0.985
            v += pop
        samples[i] = v
        peak = maxf(peak, absf(v))
    # Whole numbers of cycles in the loop, so that the flutter and hum meet themselves at the join.
    var flutter := float(spec["flutter"])
    var hum_cycles := roundf(float(spec["hum"]) * LOOP_SECONDS)
    var data := PackedByteArray()
    data.resize(n * 2)
    for i in n:
        var v := samples[i] / peak
        if i < blend:                      # the extra tail, faded over the start
            var t := float(i) / blend
            v = v * t + samples[n + i] / peak * (1.0 - t)
        var phase := float(i) / n * TAU
        v *= 1.0 - flutter * 0.5 * (0.5 + 0.3 * sin(phase * 3.0) + 0.2 * sin(phase * 7.0 + 1.3))
        if hum_cycles > 0.0:
            v = v * 0.7 + 0.3 * sin(phase * hum_cycles)
        data.encode_s16(i * 2, int(clampf(v * 0.8, -1.0, 1.0) * 30000.0))
    var stream := AudioStreamWAV.new()
    stream.format = AudioStreamWAV.FORMAT_16_BITS
    stream.mix_rate = rate
    stream.stereo = false
    stream.data = data
    stream.loop_mode = AudioStreamWAV.LOOP_FORWARD
    stream.loop_begin = 0
    stream.loop_end = n
    _streams[key] = stream
    return stream
