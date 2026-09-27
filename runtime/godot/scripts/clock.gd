class_name GeogenClock
extends Node
## World time of day. Drives the sun, sky and fog, and the world's night state: lights that
## exports mark `auto: night` (street lamps) and vehicle lamps come on after dark. NPC
## routines and traffic schedules read `hours`.
##
## Time runs with physics (so --timescale speeds it up): a whole day takes `day_length`
## seconds (0 freezes the clock). --time=HH:MM sets the start, --day-length=S the pace.

signal night_changed(night: bool)

const MAX_ELEVATION := 58.0      # deg, sun at noon
const NOON_AZIMUTH := 200.0      # deg about +Y where the noon sun stands (the old fixed sun's side)

var hours := 13.0
var day_length := 1440.0         # s per 24 h (a day in 24 minutes)
var sun: DirectionalLight3D
var environment: Environment
var world: WorldLoader
## 0 at night .. 1 in full daylight.
var daylight := 1.0
var night := false

var _sky: ProceduralSkyMaterial
var _day_sky := {}
var _moon: DirectionalLight3D


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
    apply()


func _physics_process(delta: float) -> void:
    if day_length > 0.0:
        hours = fposmod(hours + delta * 24.0 / day_length, 24.0)
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


func apply() -> void:
    var h := sun_height()
    daylight = smoothstep(-0.12, 0.18, h)
    var skylight := smoothstep(-0.32, 0.12, h)    # twilight lingers in the sky after sunset
    if _moon != null:
        _moon.light_energy = 0.14 * (1.0 - skylight)
        _moon.visible = skylight < 0.98
    if sun != null:
        var elevation := deg_to_rad(MAX_ELEVATION) * h
        var azimuth := deg_to_rad(NOON_AZIMUTH) + (hours - 12.0) / 12.0 * PI
        sun.rotation = Vector3(-maxf(elevation, deg_to_rad(2.0)), azimuth, 0.0)
        sun.light_energy = smoothstep(-0.02, 0.18, h)
        sun.visible = h > -0.02
        var low := clampf(h * 3.0, 0.0, 1.0)       # warm near the horizon
        sun.light_color = Color(1.0, 0.62, 0.38).lerp(Color(1.0, 0.98, 0.94), low)
    if _sky != null:
        var night_top := Color(0.03, 0.045, 0.1)
        var night_horizon := Color(0.08, 0.1, 0.17)
        var dusk := Color(0.95, 0.55, 0.35)
        var glow := clampf(1.0 - absf(h + 0.05) * 5.0, 0.0, 1.0) * 0.7   # sunrise / sunset tint
        _sky.sky_top_color = night_top.lerp(_day_sky["top"], skylight)
        var horizon: Color = night_horizon.lerp(_day_sky["horizon"], skylight)
        _sky.sky_horizon_color = horizon.lerp(dusk, glow)
        _sky.ground_horizon_color = night_horizon.lerp(_day_sky["ground"], skylight)
        _sky.ground_bottom_color = Color(0.03, 0.03, 0.04).lerp(_day_sky["bottom"], skylight)
    if environment != null:
        environment.ambient_light_energy = lerpf(0.35, 1.0, skylight)
        environment.fog_light_color = Color(0.07, 0.08, 0.13).lerp(Color(0.66, 0.72, 0.8), skylight)
    var now_night := daylight < 0.35
    if now_night != night:
        night = now_night
        if world != null:
            world.set_night(night)
        night_changed.emit(night)
