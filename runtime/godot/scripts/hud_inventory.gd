class_name GeogenHudInventory
extends PanelContainer
## HUD panel (bottom right): what the HUD's actor carries (geogen.18), one row an item: its
## title from the manifest's `items`, and a count when more than one ("Parcel  ×2").
##
## Items are actor state (`actor:<pkg>.item_<id>`), so the panel re-reads them on `changed`
## events and when the HUD refreshes everything (a scene loaded, a save restored). A change
## the actor's own (a give or a take, by a rule or a pickup) also raises a toast in the
## notifications panel: "Picked up: Room key", "Handed over: Parcel". Hidden while empty.
##
## An item whose file declares `use:` (a verb) can be used from here: one of them is selected
## (`USE_KEY`'s verb shows on its row, "F: Eat"), `CYCLE_KEY` selects the next, `USE_KEY`
## raises the rules' `use_item` event for it (`GeogenRules.use_item`); its rules do the rest.

const CYCLE_KEY := KEY_I
const USE_KEY := KEY_F
const TITLE_SIZE := 13
const ROW_SIZE := 17
const COLORS := {"title": Color(0.78, 0.8, 0.84), "item": Color(0.95, 0.8, 0.5), "count": Color(0.9, 0.9, 0.9),
    "use": Color(0.75, 0.9, 0.75), "other": Color(0.5, 0.55, 0.5)}

var _hud: GeogenHud
var _rows: GridContainer
var _shown: Array = []          # [{id, title, count, tags?, use?}]
var _selected := ""             # the usable item USE_KEY uses


func _init() -> void:
    name = "inventory"
    visible = false
    mouse_filter = Control.MOUSE_FILTER_IGNORE
    var style := StyleBoxFlat.new()
    style.bg_color = Color(0.05, 0.06, 0.08, 0.62)
    style.set_corner_radius_all(6)
    style.border_width_right = 3
    style.border_color = COLORS["item"]
    style.content_margin_left = 12
    style.content_margin_right = 12
    style.content_margin_top = 6
    style.content_margin_bottom = 7
    add_theme_stylebox_override("panel", style)
    var column := VBoxContainer.new()
    column.add_theme_constant_override("separation", 2)
    add_child(column)
    var title := Label.new()
    title.text = "Carrying"
    title.add_theme_font_size_override("font_size", TITLE_SIZE)
    title.add_theme_color_override("font_color", COLORS["title"])
    column.add_child(title)
    _rows = GridContainer.new()
    _rows.columns = 3
    _rows.add_theme_constant_override("h_separation", 14)
    column.add_child(_rows)


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud
    hud.world.events.listen(_on_changed, ["changed"])


func _on_changed(event: Dictionary) -> void:
    var key := str(event.get("name", ""))
    var rules: GeogenRules = _hud.world.rules
    var prefix := "actor:%s.item_" % rules.package
    if not key.begins_with(prefix) or str(event.get("actor", "")) != _hud.actor:
        return
    var id := key.trim_prefix(prefix)
    var data: Dictionary = event.get("data", {})
    var was := int(data.get("from", 0) if data.get("from") != null else 0)
    var now := int(data.get("to", 0) if data.get("to") != null else 0)
    var title := str(rules.items.get(id, {}).get("title", id))
    var notes = _hud.panel("notifications")
    if notes != null and now != was:
        var n := absi(now - was)
        notes.item_note("item_got" if now > was else "item_gone", title + ("" if n == 1 else "  ×%d" % n))
    _hud.queue_refresh()


func hud_refresh() -> void:
    _shown = _hud.world.rules.carried(_hud.actor)
    var usable := _usable()
    if not usable.has(_selected):
        _selected = usable[0] if not usable.is_empty() else ""
    for child in _rows.get_children():
        _rows.remove_child(child)
        child.queue_free()
    for entry in _shown:
        var name_label := Label.new()
        name_label.text = entry["title"]
        name_label.add_theme_font_size_override("font_size", ROW_SIZE)
        name_label.add_theme_color_override("font_color", COLORS["item"])
        _rows.add_child(name_label)
        var count := Label.new()
        count.text = "×%d" % entry["count"] if entry["count"] > 1 else ""
        count.add_theme_font_size_override("font_size", ROW_SIZE)
        count.add_theme_color_override("font_color", COLORS["count"])
        _rows.add_child(count)
        var verb := Label.new()
        if entry.has("use"):
            var chosen: bool = entry["id"] == _selected
            verb.text = ("%s: %s" % [OS.get_keycode_string(USE_KEY), entry["use"]]) if chosen else str(entry["use"])
            verb.add_theme_color_override("font_color", COLORS["use"] if chosen else COLORS["other"])
        verb.add_theme_font_size_override("font_size", ROW_SIZE - 3)
        _rows.add_child(verb)
    visible = not _shown.is_empty()


func _usable() -> Array:
    var out := []
    for entry in _shown:
        if entry.has("use"):
            out.append(entry["id"])
    return out


## I selects the next usable item; F uses the selected one.
func hud_key(keycode: int) -> bool:
    var usable := _usable()
    if usable.is_empty():
        return false
    if keycode == CYCLE_KEY:
        _selected = usable[(usable.find(_selected) + 1) % usable.size()]
        _hud.queue_refresh()
        return true
    if keycode == USE_KEY:
        var problem: String = _hud.world.rules.use_item(_selected, _hud.actor)
        if problem != "":
            push_warning("inventory: " + problem)
        return true
    return false


## What is on screen: {visible, items: [{id, title, count, tags?, use?}], selected}.
func hud_report() -> Dictionary:
    return {"visible": visible, "items": _shown.duplicate(true), "selected": _selected}
