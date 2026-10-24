class_name GeogenClock
extends Node
## World time of day. Drives the sun, sky and fog, and the world's night state: lights that
## exports mark `auto: night` (street lamps) and vehicle lamps come on after dark. NPC
## routines and traffic schedules read `hours`.
##
## Time runs with physics (so --timescale speeds it up): a whole day takes `day_length`
## seconds (0 freezes the clock). --time=HH:MM sets the start, --day-length=S the pace.
## `day` counts the midnights passed, so the rules' clock timers (game-clock waits) know
## how far the clock has really gone.
##
## A scene's `atmosphere` (its manifest, geogen/atmosphere.py: a preset's values, resolved)
## is the mood it plays in: `set_atmosphere` takes the fog, grading and glow once, and the
## sun, sky, moon and night values feed `apply` every frame. Without one, or with {}, the
## look is main.tscn's and the constants below (the `clear` preset).

signal night_changed(night: bool)

const MAX_ELEVATION := 58.0      # deg, sun at noon
const NOON_AZIMUTH := 200.0      # deg about +Y where the noon sun stands (the old fixed sun's side)
const SUN_COLOR := Color(1.0, 0.98, 0.94)
const SUN_LOW_COLOR := Color(1.0, 0.62, 0.38)
const MOON_ENERGY := 0.14
const NIGHT_AMBIENT := 0.35
const NIGHT_TOP := Color(0.03, 0.045, 0.1)
const NIGHT_HORIZON := Color(0.08, 0.1, 0.17)
const NIGHT_GROUND := Color(0.03, 0.03, 0.04)
const DAY_FOG := Color(0.66, 0.72, 0.8)
const NIGHT_FOG := Color(0.07, 0.08, 0.13)

var hours := 13.0
var day := 0
var day_length := 1440.0         # s per 24 h (a day in 24 minutes)
var sun: DirectionalLight3D
var environment: Environment
var world: WorldLoader
## 0 at night .. 1 in full daylight.
var daylight := 1.0
var night := false
## --time was given: scenes' `atmosphere.time` leaves the clock alone.
var time_pinned := false
## The loaded scene's atmosphere (resolved values), or {} for the default look.
var atmosphere := {}

var _sky: ProceduralSkyMaterial
var _day_sky := {}
var _base_env := {}
var _moon: DirectionalLight3D
var _timed_scenes := {}        # scenes whose atmosphere.time has set the clock this run


func _ready() -> void:
    if sun != null:
        _moon = DirectionalLight3D.new()
        _moon.name = "Moon"
        _moon.light_color = Color(0.62, 0.72, 1.0)
        _moon.shadow_enabled = false
        _moon.rotation = Vector3(deg_to_rad(-40.0), deg_to_rad(NOON_AZIMUTH + 150.0), 0.0)
        sun.get_parent().add_child.call_deferred(_moon)
    if environment != null and environment.sky != null:
        _sky = environment.sky.sky_material as ProceduralSkyMaterial
        if _sky != null:
            _day_sky = {"top": _sky.sky_top_color, "horizon": _sky.sky_horizon_color,
                "ground": _sky.ground_horizon_color, "bottom": _sky.ground_bottom_color}
    if environment != null:
        _base_env = {"fog_density": environment.fog_density, "fog_sky_affect": environment.fog_sky_affect,
            "fog_aerial_perspective": environment.fog_aerial_perspective,
            "adjustment_enabled": environment.adjustment_enabled,
            "adjustment_saturation": environment.adjustment_saturation,
            "adjustment_contrast": environment.adjustment_contrast,
            "glow_enabled": environment.glow_enabled, "glow_intensity": environment.glow_intensity,
            "glow_bloom": environment.glow_bloom}
    _apply_environment()      # main adds the clock after the first load: its atmosphere is already here
    apply()


func _physics_process(delta: float) -> void:
    if day_length > 0.0:
        var next := hours + delta * 24.0 / day_length
        if next >= 24.0:
            day += int(next / 24.0)
        hours = fposmod(next, 24.0)
    apply()


## "HH:MM" -> hours.
static func parse(text: String) -> float:
    var parts := text.split(":")
    return fposmod(float(parts[0]) + (float(parts[1]) / 60.0 if parts.size() > 1 else 0.0), 24.0)


func label() -> String:
    return "%02d:%02d" % [int(hours), int(fmod(hours, 1.0) * 60.0)]


## Sine of the sun's height: 1 at noon, 0 at 06:00 and 18:00, -1 at midnight.
func sun_height() -> float:
    return sin((hours - 6.0) / 12.0 * PI)


## A scene's atmosphere (its manifest's, resolved by the exporter), or {} for the default look.
## Fog, grading and glow are set here, once; the rest is read by apply(). `scene` names the
## load: its `time` sets the clock the first time that scene loads in this run, unless --time
## was given (a restored save sets the clock after this and wins).
func set_atmosphere(spec: Dictionary, scene := "") -> void:
    atmosphere = spec
    _apply_environment()
    if spec.has("time") and not time_pinned and scene != "" and not _timed_scenes.has(scene):
        hours = parse(String(spec["time"]))
        _timed_scenes[scene] = true
    apply()


## The once-per-load settings: main.tscn's values, then the atmosphere's over them.
func _apply_environment() -> void:
    if environment == null or _base_env.is_empty():
        return
    for key in _base_env:
        environment.set(key, _base_env[key])
    if atmosphere.is_empty():
        return
    environment.fog_density = float(atmosphere.get("fog", _base_env["fog_density"]))
    var haze := float(atmosphere.get("haze", 0.0))
    environment.fog_sky_affect = haze
    environment.fog_aerial_perspective = haze * 0.8
    var saturation := float(atmosphere.get("saturation", 1.0))
    var contrast := float(atmosphere.get("contrast", 1.0))
    if not is_equal_approx(saturation, 1.0) or not is_equal_approx(contrast, 1.0):
        environment.adjustment_enabled = true
        environment.adjustment_saturation = saturation
        environment.adjustment_contrast = contrast
    var glow := float(atmosphere.get("glow", 0.0))
    if glow > 0.0:
        environment.glow_enabled = true
        environment.glow_intensity = glow
        environment.glow_bloom = 0.1 * glow


func _color(table: Dictionary, key: String, fallback: Color) -> Color:
    var value = table.get(key)
    if value is Array and value.size() == 3:
        return Color(float(value[0]), float(value[1]), float(value[2]))
    return fallback


func apply() -> void:
    var h := sun_height()
    daylight = smoothstep(-0.12, 0.18, h)
    var skylight := smoothstep(-0.32, 0.12, h)    # twilight lingers in the sky after sunset
    var night_floor := float(atmosphere.get("night", NIGHT_AMBIENT))
    # A brighter night lifts the night sky with it, so the horizon stays a shade above the land.
    var lift := clampf((night_floor - NIGHT_AMBIENT) / 0.4, 0.0, 1.0)
    var night_top := NIGHT_TOP.lerp(Color(0.1, 0.12, 0.16), lift)
    var night_horizon := NIGHT_HORIZON.lerp(Color(0.16, 0.18, 0.22), lift)
    var sky: Dictionary = atmosphere.get("sky", {})
    var day_horizon: Color = _color(sky, "horizon", _day_sky.get("horizon", DAY_FOG))
    if _moon != null:
        _moon.light_energy = float(atmosphere.get("moon", MOON_ENERGY)) * (1.0 - skylight)
        _moon.visible = skylight < 0.98
    if sun != null:
        var elevation := deg_to_rad(float(atmosphere.get("sun_height", MAX_ELEVATION))) * h
        var azimuth := deg_to_rad(NOON_AZIMUTH) + (hours - 12.0) / 12.0 * PI
        sun.rotation = Vector3(-maxf(elevation, deg_to_rad(2.0)), azimuth, 0.0)
        sun.light_energy = float(atmosphere.get("sun", 1.0)) * smoothstep(-0.02, 0.18, h)
        sun.visible = h > -0.02
        var low := clampf(h * 3.0, 0.0, 1.0)       # warm near the horizon
        sun.light_color = SUN_LOW_COLOR.lerp(_color(atmosphere, "sun_color", SUN_COLOR), low)
    if _sky != null:
        var dusk := Color(0.95, 0.55, 0.35)
        var glow := clampf(1.0 - absf(h + 0.05) * 5.0, 0.0, 1.0) * 0.7   # sunrise / sunset tint
        glow *= 1.0 - float(atmosphere.get("haze", 0.0))                    # no sunset under a lid of cloud
        _sky.sky_top_color = night_top.lerp(_color(sky, "top", _day_sky["top"]), skylight)
        var horizon: Color = night_horizon.lerp(day_horizon, skylight)
        _sky.sky_horizon_color = horizon.lerp(dusk, glow)
        _sky.ground_horizon_color = night_horizon.lerp(day_horizon, skylight)
        _sky.ground_bottom_color = NIGHT_GROUND.lerp(_color(sky, "ground", _day_sky["bottom"]), skylight)
    if environment != null:
        environment.ambient_light_energy = lerpf(night_floor, 1.0, skylight)
        environment.fog_light_color = NIGHT_FOG.lerp(day_horizon, skylight)
    var now_night := daylight < 0.35
    if now_night != night:
        night = now_night
        if world != null:
            world.set_night(night)
        night_changed.emit(night)
