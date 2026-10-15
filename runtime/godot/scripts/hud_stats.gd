class_name GeogenHudStats
extends PanelContainer
## HUD panel: the numbers a scene declares (`hud: {stats}`, geogen/hud.py), one row each:
## "Lives  17", "Wave  3 / 8".
##
## Each row's `value` (and `of`) is a compiled condition from the manifest's `hud`
## section, evaluated for the HUD's actor by the loaded rules program
## (`GeogenRules.value_of`). It re-reads them when state changes and when the HUD
## refreshes everything (a scene loaded, a save restored); never per frame. Hidden when
## the scene declares no stats.

const LABEL_SIZE := 14
const VALUE_SIZE := 20
const COLORS := {
    "label": Color(0.78, 0.8, 0.84),
    "value": Color(1.0, 0.86, 0.55),
}

var _hud: GeogenHud
var _rows: GridContainer
var _shown: Array = []          # [{label, value, of?, text}]


func _init() -> void:
    name = "stats"
    visible = false
    mouse_filter = Control.MOUSE_FILTER_IGNORE
    var style := StyleBoxFlat.new()
    style.bg_color = Color(0.05, 0.06, 0.08, 0.62)
    style.set_corner_radius_all(6)
    style.border_width_left = 3
    style.border_color = COLORS["value"]
    style.content_margin_left = 12
    style.content_margin_right = 12
    style.content_margin_top = 6
    style.content_margin_bottom = 7
    add_theme_stylebox_override("panel", style)
    _rows = GridContainer.new()
    _rows.columns = 2
    _rows.add_theme_constant_override("h_separation", 14)
    _rows.add_theme_constant_override("v_separation", 2)
    add_child(_rows)


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud
    hud.world.events.listen(func(_event: Dictionary): _changed(), ["changed", "clock"])


func _changed() -> void:
    if visible:
        _hud.queue_refresh()


func hud_refresh() -> void:
    var rules: GeogenRules = _hud.world.rules
    _shown = []
    for stat in rules.hud.get("stats", []):
        var value: Variant = rules.value_of(stat["value"], _hud.actor)
        var entry := {"label": str(stat["label"]), "value": value, "text": shown(value)}
        if stat.has("of"):
            entry["of"] = rules.value_of(stat["of"], _hud.actor)
            entry["text"] += " / " + shown(entry["of"])
        _shown.append(entry)
    visible = not _shown.is_empty()
    for child in _rows.get_children():
        _rows.remove_child(child)
        child.queue_free()
    for entry in _shown:
        _rows.add_child(_label(entry["label"], LABEL_SIZE, COLORS["label"]))
        var value := _label(entry["text"], VALUE_SIZE, COLORS["value"])
        value.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
        _rows.add_child(value)


## How a value reads on screen: whole numbers without a point, others to one place.
static func shown(value: Variant) -> String:
    if value == null:
        return "-"
    if typeof(value) == TYPE_BOOL:
        return "yes" if value else "no"
    if typeof(value) == TYPE_FLOAT or typeof(value) == TYPE_INT:
        var number := float(value)
        return str(int(round(number))) if is_equal_approx(number, round(number)) else "%.1f" % number
    return str(value)


static func _label(text: String, font_size: int, color: Color) -> Label:
    var label := Label.new()
    label.text = text
    label.mouse_filter = Control.MOUSE_FILTER_IGNORE
    label.add_theme_font_size_override("font_size", font_size)
    label.add_theme_color_override("font_color", color)
    label.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.7))
    label.add_theme_constant_override("shadow_offset_x", 1)
    label.add_theme_constant_override("shadow_offset_y", 1)
    label.size_flags_vertical = Control.SIZE_SHRINK_CENTER
    return label


## What is on screen: {visible, stats: [{label, value, of?, text}]}.
func hud_report() -> Dictionary:
    return {"visible": visible, "stats": _shown.duplicate(true)}
