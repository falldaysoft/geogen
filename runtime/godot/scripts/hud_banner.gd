class_name GeogenHudBanner
extends Control
## HUD panel: a scene's banner (`hud: {banners}`, geogen/hud.py) across the middle of the
## screen: the first entry whose `when` holds for the HUD's actor ("The core has fallen").
## It is re-read when state changes and when the HUD refreshes everything; hidden while
## none holds.

const TEXT_SIZE := 34
const TONES := {
    "info": Color(1.0, 0.86, 0.55),
    "good": Color(0.6, 0.92, 0.6),
    "bad": Color(0.98, 0.5, 0.45),
}

var _hud: GeogenHud
var _back: PanelContainer
var _text: Label
var _shown := {}                # {text, tone}, or {} when none holds


func _init() -> void:
    name = "banner"
    mouse_filter = Control.MOUSE_FILTER_IGNORE
    _back = PanelContainer.new()
    _back.mouse_filter = Control.MOUSE_FILTER_IGNORE
    _back.anchor_left = 0.0
    _back.anchor_right = 1.0
    _back.anchor_top = 0.32
    _back.anchor_bottom = 0.32
    _back.grow_vertical = Control.GROW_DIRECTION_BOTH
    var style := StyleBoxFlat.new()
    style.bg_color = Color(0.03, 0.04, 0.06, 0.66)
    style.content_margin_top = 14
    style.content_margin_bottom = 14
    _back.add_theme_stylebox_override("panel", style)
    add_child(_back)
    _text = Label.new()
    _text.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    _text.mouse_filter = Control.MOUSE_FILTER_IGNORE
    _text.add_theme_font_size_override("font_size", TEXT_SIZE)
    _text.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.8))
    _text.add_theme_constant_override("shadow_offset_x", 2)
    _text.add_theme_constant_override("shadow_offset_y", 2)
    _back.add_child(_text)
    _back.visible = false


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud
    hud.world.events.listen(func(_event: Dictionary): hud.queue_refresh(), ["changed", "clock"])


func hud_refresh() -> void:
    var rules: GeogenRules = _hud.world.rules
    _shown = {}
    for banner in rules.hud.get("banners", []):
        if GeogenConditions.truthy(rules.value_of(banner["when"], _hud.actor)):
            _shown = {"text": str(banner["text"]), "tone": str(banner.get("tone", "info"))}
            break
    _back.visible = not _shown.is_empty()
    if _back.visible:
        _text.text = _shown["text"]
        _text.add_theme_color_override("font_color", TONES.get(_shown["tone"], TONES["info"]))


## What is on screen: {visible, text?, tone?}.
func hud_report() -> Dictionary:
    var out := {"visible": _back.visible}
    out.merge(_shown)
    return out
