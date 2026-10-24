class_name GeogenHudInventory
extends Control
## HUD panel (bottom centre): a hotbar of what the HUD's actor carries (geogen.18, geogen.84.14),
## one square slot an item, as a Minecraft player would expect: the item's icon (rendered from
## its body by the world loader, `item_icon`; its initials until one exists), a count badge
## when more than one, and the selected slot framed, with its title and verb on a line above
## ("Berries   F: Eat").
##
## Items are actor state (`actor:<pkg>.item_<id>`), so the panel re-reads them on `changed`
## events and when the HUD refreshes everything (a scene loaded, a save restored). A change
## the actor's own (a give or a take, by a rule or a pickup) also raises a toast in the
## notifications panel: "Picked up: Room key", "Handed over: Parcel". Hidden while empty.
##
## One slot is selected: `CYCLE_KEY` selects the next, the digits 1..9 a slot outright, and
## `USE_KEY` uses the selected item when its file declares `use:` (a verb): it raises the rules'
## `use_item` event (`GeogenRules.use_item`) and the item's rules do the rest. A tool with no
## verb (an axe) is selected all the same: it is what you hold.

const CYCLE_KEY := KEY_I
const USE_KEY := KEY_F
const SLOT := 52.0              # px, a slot's side
const GAP := 4.0
const PAD := 6.0
const TITLE_SIZE := 16
const GLYPH_SIZE := 20
const COUNT_SIZE := 13
const COLORS := {"back": Color(0.05, 0.06, 0.08, 0.62), "slot": Color(0.12, 0.13, 0.16, 0.85),
    "edge": Color(0.3, 0.32, 0.36, 0.9), "selected": Color(1.0, 0.82, 0.4, 0.95),      # the tracker's gold
    "glyph": Color(0.8, 0.82, 0.86), "count": Color(1, 1, 1), "shade": Color(0, 0, 0, 0.6),
    "title": Color(0.95, 0.8, 0.5), "use": Color(0.75, 0.9, 0.75), "digit": Color(0.55, 0.58, 0.62)}

var _hud: GeogenHud
var _shown: Array = []          # [{id, title, count, tags?, use?}]
var _selected := ""             # the slot's item id
var _icons := {}                # item id -> Texture2D, from the world loader
var _font: Font


func _init() -> void:
    name = "inventory"
    visible = false
    mouse_filter = Control.MOUSE_FILTER_IGNORE
    _font = ThemeDB.fallback_font


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud
    hud.world.events.listen(_on_changed, ["changed"])
    hud.world.item_icons_ready.connect(hud.queue_refresh)


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
    var ids := _ids()
    if not ids.has(_selected):
        _selected = ids[0] if not ids.is_empty() else ""
    _icons = _hud.world.item_icons()
    var width := PAD * 2 + maxf(_shown.size() * (SLOT + GAP) - GAP, SLOT)
    custom_minimum_size = Vector2(width, PAD * 2 + SLOT + TITLE_SIZE + 10)
    size = custom_minimum_size
    visible = not _shown.is_empty()
    queue_redraw()


func _ids() -> Array:
    var out := []
    for entry in _shown:
        out.append(entry["id"])
    return out


func _entry(id: String) -> Dictionary:
    for entry in _shown:
        if entry["id"] == id:
            return entry
    return {}


## The bar's slot rectangles, left to right, in the panel's own space.
func _slot_rects() -> Array:
    var out := []
    var top := TITLE_SIZE + 10 + PAD
    for i in _shown.size():
        out.append(Rect2(PAD + i * (SLOT + GAP), top, SLOT, SLOT))
    return out


## Initials to stand in for a missing icon: "Stone axe" -> "SA", "Berries" -> "Be".
static func glyph_of(title: String) -> String:
    var words := title.split(" ", false)
    if words.size() >= 2:
        return (words[0].left(1) + words[1].left(1)).to_upper()
    return title.left(2).capitalize() if title.length() > 1 else title.to_upper()


func _draw() -> void:
    if _shown.is_empty():
        return
    var rects := _slot_rects()
    var bar := Rect2(0, TITLE_SIZE + 10, size.x, PAD * 2 + SLOT)
    draw_rect(bar, COLORS["back"])
    var chosen := _entry(_selected)
    if not chosen.is_empty():
        var line: String = chosen["title"]
        var verb := ""
        if chosen.has("use"):
            verb = "   %s: %s" % [OS.get_keycode_string(USE_KEY), chosen["use"]]
        var line_width := _font.get_string_size(line, HORIZONTAL_ALIGNMENT_LEFT, -1, TITLE_SIZE).x
        var verb_width := _font.get_string_size(verb, HORIZONTAL_ALIGNMENT_LEFT, -1, TITLE_SIZE).x if verb != "" else 0.0
        var x := (size.x - line_width - verb_width) / 2.0
        var y := float(TITLE_SIZE)
        draw_string(_font, Vector2(x + 1, y + 1), line, HORIZONTAL_ALIGNMENT_LEFT, -1, TITLE_SIZE, COLORS["shade"])
        draw_string(_font, Vector2(x, y), line, HORIZONTAL_ALIGNMENT_LEFT, -1, TITLE_SIZE, COLORS["title"])
        if verb != "":
            draw_string(_font, Vector2(x + line_width + 1, y + 1), verb, HORIZONTAL_ALIGNMENT_LEFT, -1, TITLE_SIZE, COLORS["shade"])
            draw_string(_font, Vector2(x + line_width, y), verb, HORIZONTAL_ALIGNMENT_LEFT, -1, TITLE_SIZE, COLORS["use"])
    for i in _shown.size():
        var entry: Dictionary = _shown[i]
        var rect: Rect2 = rects[i]
        var selected: bool = entry["id"] == _selected
        draw_rect(rect, COLORS["slot"])
        var icon: Texture2D = _icons.get(entry["id"])
        if icon != null:
            draw_texture_rect(icon, rect.grow(-3.0), false)
        else:
            var glyph := glyph_of(entry["title"])
            var w := _font.get_string_size(glyph, HORIZONTAL_ALIGNMENT_LEFT, -1, GLYPH_SIZE).x
            var at := Vector2(rect.position.x + (SLOT - w) / 2.0, rect.position.y + SLOT / 2.0 + GLYPH_SIZE * 0.36)
            draw_string(_font, at, glyph, HORIZONTAL_ALIGNMENT_LEFT, -1, GLYPH_SIZE, COLORS["glyph"])
        if i < 9:
            draw_string(_font, rect.position + Vector2(4, 12), str(i + 1), HORIZONTAL_ALIGNMENT_LEFT, -1, 10, COLORS["digit"])
        if entry["count"] > 1:
            var text := "×%d" % entry["count"]
            var w := _font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, COUNT_SIZE).x
            var at := rect.position + Vector2(SLOT - w - 4, SLOT - 5)
            draw_string(_font, at + Vector2(1, 1), text, HORIZONTAL_ALIGNMENT_LEFT, -1, COUNT_SIZE, COLORS["shade"])
            draw_string(_font, at, text, HORIZONTAL_ALIGNMENT_LEFT, -1, COUNT_SIZE, COLORS["count"])
        draw_rect(rect, COLORS["selected"] if selected else COLORS["edge"], false, 3.0 if selected else 1.0)


## For the keys panel.
func hud_keys() -> Dictionary:
    return {"title": "Inventory", "keys": [
        [OS.get_keycode_string(CYCLE_KEY), "select the next item in the hotbar"],
        ["1-9", "select that slot"],
        [OS.get_keycode_string(USE_KEY), "use the selected item, as it says (eat, crack open)"]]}


## I selects the next slot, a digit its slot, F uses the selected item (when it can be used).
func hud_key(keycode: int) -> bool:
    var ids := _ids()
    if ids.is_empty():
        return false
    if keycode == CYCLE_KEY:
        _selected = ids[(ids.find(_selected) + 1) % ids.size()]
        _hud.queue_refresh()
        return true
    if keycode >= KEY_1 and keycode <= KEY_9:
        var index := keycode - KEY_1
        if index < ids.size():
            _selected = ids[index]
            _hud.queue_refresh()
            return true
        return false
    if keycode == USE_KEY:
        if not _entry(_selected).has("use"):
            return false
        var problem: String = _hud.world.rules.use_item(_selected, _hud.actor)
        if problem != "":
            push_warning("inventory: " + problem)
        return true
    return false


## What is on screen: {visible, items: [{id, title, count, tags?, use?}], selected, slots:
## [{id, rect: [x, y, w, h] (viewport px), icon: whether a rendered icon fills it}]}.
func hud_report() -> Dictionary:
    var slots := []
    var rects := _slot_rects()
    for i in _shown.size():
        var r: Rect2 = rects[i]
        var at := global_position + r.position
        slots.append({"id": _shown[i]["id"], "rect": [at.x, at.y, r.size.x, r.size.y],
            "icon": _icons.has(_shown[i]["id"])})
    return {"visible": visible, "items": _shown.duplicate(true), "selected": _selected, "slots": slots}
