class_name GeogenHudCrossing
extends Control
## HUD panel: the pedestrian signal the player is at. On the kerb at either end of a signalled
## crossing (the lane graph's `pedestrian_signals`), or on one, it shows what the heads across
## the road show: WALK, a flashing DON'T WALK (don't start; finish if you're on it) or a steady
## DON'T WALK, with a line saying what that means. The same rule NPCs keep (npc.gd
## `_wait_for_walk`), told to the player rather than enforced. Hidden anywhere else.

const KERB := 1.6               # m beyond an end of a crossing that counts as waiting at it
const SIDE := 0.6               # m either side of the stripes that still counts
const POLL := 0.2               # s between looks at where the player is
const TEXT_SIZE := 30
const SHOWN := {
    "walk": ["WALK", "", Color(0.75, 0.95, 0.85)],
    "flash": ["DON'T WALK", "Don't start crossing", Color(1.0, 0.62, 0.3)],
    "dont": ["DON'T WALK", "Wait for the signal", Color(1.0, 0.45, 0.35)],
}
const ON_CROSSING := {"flash": "Keep going", "dont": "Clear the road"}

var _hud: GeogenHud
var _back: PanelContainer
var _title: Label
var _line: Label
var _rects := []                # [Rect2, id, long side along x], from the loaded graph
var _graph: Dictionary = {}     # the graph _rects were made from
var _poll := 0.0
var _shown := {}                # {crossing, where, state, text, line}, or {} when hidden


func _init() -> void:
    name = "crossing"
    mouse_filter = Control.MOUSE_FILTER_IGNORE
    _back = PanelContainer.new()
    _back.mouse_filter = Control.MOUSE_FILTER_IGNORE
    _back.anchor_left = 0.5
    _back.anchor_right = 0.5
    _back.anchor_top = 0.74
    _back.anchor_bottom = 0.74
    _back.grow_horizontal = Control.GROW_DIRECTION_BOTH
    var style := StyleBoxFlat.new()
    style.bg_color = Color(0.03, 0.04, 0.06, 0.66)
    style.set_corner_radius_all(6)
    style.content_margin_left = 18
    style.content_margin_right = 18
    style.content_margin_top = 8
    style.content_margin_bottom = 8
    _back.add_theme_stylebox_override("panel", style)
    add_child(_back)
    var column := VBoxContainer.new()
    column.mouse_filter = Control.MOUSE_FILTER_IGNORE
    _back.add_child(column)
    _title = Label.new()
    _title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    _title.add_theme_font_size_override("font_size", TEXT_SIZE)
    column.add_child(_title)
    _line = Label.new()
    _line.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    _line.add_theme_color_override("font_color", Color(0.9, 0.9, 0.9))
    column.add_child(_line)
    _back.visible = false


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud


func hud_refresh() -> void:
    _update()


func _process(delta: float) -> void:
    _poll -= delta
    if _poll <= 0.0:
        _poll = POLL
        _update()
    # A flashing don't-walk flashes (on for 0.5 s, off for 0.25 s).
    if _back.visible and _shown.get("state", "") == "flash":
        _title.modulate.a = 1.0 if fposmod(Time.get_ticks_msec() / 1000.0, 0.75) < 0.5 else 0.15
    else:
        _title.modulate.a = 1.0


func _update() -> void:
    _shown = {}
    var player = get_tree().root.find_child("Player", true, false) if _hud != null else null
    if player is Node3D:
        var at := crossing_at(Vector2(player.global_position.x, player.global_position.z))
        var state := "" if at.is_empty() else _state(at["crossing"])
        if state != "":
            var shown: Array = SHOWN[state]
            var line: String = ON_CROSSING.get(state, "") if at["where"] == "crossing" else shown[1]
            _shown = {"crossing": at["crossing"], "where": at["where"], "state": state,
                      "text": shown[0], "line": line}
            _title.text = shown[0]
            _title.add_theme_color_override("font_color", shown[2])
            _line.text = line
            _line.visible = line != ""
    _back.visible = not _shown.is_empty()


## The signalled crossing at `p` (plan x, z): {crossing, where: "kerb" | "crossing"}, or {}.
func crossing_at(p: Vector2) -> Dictionary:
    var graph: Dictionary = _hud.world.traffic_graph if _hud != null else {}
    if graph != _graph:
        _graph = graph
        _rects = []
        for sig in graph.get("pedestrian_signals", []):
            var r: Array = sig["crosswalk"]
            var rect := Rect2(float(r[0]), float(r[1]), float(r[2]) - float(r[0]), float(r[3]) - float(r[1]))
            _rects.append([rect, str(sig["id"]), rect.size.x > rect.size.y])
    for entry in _rects:
        var rect: Rect2 = entry[0]
        if rect.has_point(p):
            return {"crossing": entry[1], "where": "crossing"}
        # The kerbs are beyond the ends of the long side (the way across); the other two sides are road.
        var along: float = p.x if entry[2] else p.y
        var side: float = p.y if entry[2] else p.x
        var lo: float = rect.position.x if entry[2] else rect.position.y
        var hi: float = rect.end.x if entry[2] else rect.end.y
        var s_lo: float = (rect.position.y if entry[2] else rect.position.x) - SIDE
        var s_hi: float = (rect.end.y if entry[2] else rect.end.x) + SIDE
        if side >= s_lo and side <= s_hi and ((along < lo and along > lo - KERB) or (along > hi and along < hi + KERB)):
            return {"crossing": entry[1], "where": "kerb"}
    return {}


func _state(id: String) -> String:
    for t in _hud.world.traffic:
        if is_instance_valid(t):
            return t.pedestrian_state(id)
    return ""


## What is on screen: {visible, crossing?, where?, state?, text?, line?}.
func hud_report() -> Dictionary:
    _update()
    var out := {"visible": _back.visible}
    out.merge(_shown)
    return out
