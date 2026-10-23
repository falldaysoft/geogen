class_name GeogenHudKeys
extends Control
## HUD panel (slot "screen"): the keys. **?** opens a screen over the game listing what each
## key does, in groups; ? or Esc closes it. While it is open it is the HUD's modal panel
## (`GeogenHud.set_modal`): it alone gets keys, and main.gd parks the player and frees the mouse.
##
## What it lists comes from two places, so the list can't drift from the keys themselves:
## - `groups`: the game's own keys (moving, playing, debug), set by main.gd, which owns them;
## - every panel's `hud_keys()`, an optional method returning `{title, keys: [[key, what it does]]}`
##   (the inventory's I and F, the tracker's Q, the quest log's J, the dialogue's numbers).
## Groups fill two columns on a wide window and one on a narrow one; the frame scrolls when
## the window is shorter than the list.
##
## ? is Shift+/ on most keyboards and the HUD hands panels the physical key, so `/` with or
## without Shift opens it (`OPEN_KEYS`); a typed ? on a layout that keeps it elsewhere reads
## as KEY_QUESTION (`GeogenHud._key_of`), as does `input.key Question` on the control API.

const OPEN_KEYS := [KEY_SLASH, KEY_QUESTION]
const TITLE_SIZE := 20
const HEAD_SIZE := 12
const TEXT_SIZE := 14
## Two columns of groups from this width up, one below it.
const TWO_COLUMNS_FROM := 560.0
const COLORS := {
    "title": Color(1.0, 0.86, 0.55),
    "head": Color(0.72, 0.74, 0.78),
    "key": Color(1.0, 0.86, 0.55),
    "text": Color(0.96, 0.96, 0.96),
    "note": Color(0.72, 0.74, 0.78),
}

## The game's own groups, in the order shown: [{title, keys: [[key, does], ...]}]. main.gd sets them.
var groups: Array = []

var _hud: GeogenHud
var _open := false
var _frame: PanelContainer
var _scroll: ScrollContainer
var _columns: HBoxContainer
var _hint: Label
var _listed: Array = []         # what is on screen, groups then panels', as hud_report() gives it


func _init() -> void:
    name = "keys"
    visible = false
    z_index = 1                 # over the crosshair and the E prompt
    mouse_filter = Control.MOUSE_FILTER_STOP
    var dim := ColorRect.new()
    dim.color = Color(0, 0, 0, 0.5)
    dim.mouse_filter = Control.MOUSE_FILTER_IGNORE
    dim.set_anchors_preset(Control.PRESET_FULL_RECT)
    add_child(dim)
    var center := CenterContainer.new()
    center.mouse_filter = Control.MOUSE_FILTER_IGNORE
    center.set_anchors_preset(Control.PRESET_FULL_RECT)
    add_child(center)
    _frame = PanelContainer.new()
    var style := StyleBoxFlat.new()
    style.bg_color = Color(0.06, 0.07, 0.09, 0.97)
    style.set_corner_radius_all(8)
    style.border_width_top = 3
    style.border_color = COLORS["title"]
    style.content_margin_left = 18
    style.content_margin_right = 18
    style.content_margin_top = 12
    style.content_margin_bottom = 12
    _frame.add_theme_stylebox_override("panel", style)
    center.add_child(_frame)
    var box := VBoxContainer.new()
    box.add_theme_constant_override("separation", 10)
    _frame.add_child(box)

    var head := HBoxContainer.new()
    box.add_child(head)
    var heading := _label("Keys", TITLE_SIZE + 2, COLORS["title"])
    heading.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    head.add_child(heading)
    var note := _label("what each key does", TEXT_SIZE, COLORS["note"])
    note.autowrap_mode = TextServer.AUTOWRAP_OFF         # never wrapped: the heading gives way
    note.text_overrun_behavior = TextServer.OVERRUN_NO_TRIMMING
    note.size_flags_vertical = Control.SIZE_SHRINK_END
    head.add_child(note)
    box.add_child(_rule())

    _scroll = ScrollContainer.new()
    _scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
    _scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
    box.add_child(_scroll)
    _columns = HBoxContainer.new()
    _columns.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    _columns.add_theme_constant_override("separation", 28)
    _scroll.add_child(_columns)

    box.add_child(_rule())
    _hint = _label("? / Esc: close", TEXT_SIZE - 1, COLORS["note"])
    box.add_child(_hint)


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud


func is_open() -> bool:
    return _open


## Open or close the screen.
func set_open(open: bool) -> void:
    if open == _open:
        return
    _open = open
    visible = open
    _hud.set_modal(self, open)
    _hud.refresh()


func hud_key(keycode: int) -> bool:
    if not _open:
        if keycode not in OPEN_KEYS:
            return false
        set_open(true)
        return true
    if keycode in OPEN_KEYS or keycode == KEY_ESCAPE:
        set_open(false)
    return true             # modal: nothing else gets a key while it is open


## Every group listed: the game's (`groups`), then each panel's `hud_keys()` in the order the
## panels were added.
func listed() -> Array:
    var out: Array = []
    for group in groups:
        out.append({"title": str(group["title"]), "keys": _pairs(group["keys"])})
    for panel in _hud.panels():
        if panel != self and panel.has_method("hud_keys"):
            var group: Dictionary = panel.hud_keys()
            if not group.get("keys", []).is_empty():
                out.append({"title": str(group["title"]), "keys": _pairs(group["keys"])})
    return out


static func _pairs(keys: Array) -> Array:
    var out: Array = []
    for pair in keys:
        out.append([str(pair[0]), str(pair[1])])
    return out


# -- drawing -------------------------------------------------------------------------------------

func hud_refresh() -> void:
    if _hud == null or not _open:
        return
    _listed = listed()
    var view := _hud.get_viewport().get_visible_rect().size
    var room := Vector2(maxf(view.x - 48.0, 40.0), maxf(view.y - 48.0, 40.0))
    var wide := room.x >= TWO_COLUMNS_FROM
    _frame.custom_minimum_size = Vector2(minf(760.0 if wide else 420.0, room.x), minf(560.0, room.y))

    for child in _columns.get_children():
        _columns.remove_child(child)
        child.queue_free()
    var columns: Array = [_column()]
    if wide:
        columns.append(_column())
    for column in columns:
        _columns.add_child(column)
    # Fill the columns in turn so each carries about half the rows.
    var rows := 0
    for group in _listed:
        rows += group["keys"].size() + 2
    var filled := 0
    var at := 0
    for group in _listed:
        if at < columns.size() - 1 and filled >= rows / float(columns.size()):
            at += 1
        _draw_group(columns[at], group)
        filled += group["keys"].size() + 2


static func _column() -> VBoxContainer:
    var column := VBoxContainer.new()
    column.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    column.size_flags_vertical = Control.SIZE_SHRINK_BEGIN
    column.add_theme_constant_override("separation", 12)
    return column


func _draw_group(column: VBoxContainer, group: Dictionary) -> void:
    var box := VBoxContainer.new()
    box.add_theme_constant_override("separation", 3)
    column.add_child(box)
    var head := _label(group["title"].to_upper(), HEAD_SIZE, COLORS["head"])
    box.add_child(head)
    var grid := GridContainer.new()
    grid.columns = 2
    grid.add_theme_constant_override("h_separation", 16)
    grid.add_theme_constant_override("v_separation", 2)
    box.add_child(grid)
    for pair in group["keys"]:
        var key := _label(pair[0], TEXT_SIZE, COLORS["key"])
        key.autowrap_mode = TextServer.AUTOWRAP_OFF
        key.text_overrun_behavior = TextServer.OVERRUN_NO_TRIMMING
        key.size_flags_vertical = Control.SIZE_SHRINK_BEGIN
        grid.add_child(key)
        var does := _label(pair[1], TEXT_SIZE, COLORS["text"])
        does.size_flags_horizontal = Control.SIZE_EXPAND_FILL
        grid.add_child(does)


static func _label(text: String, font_size: int, color: Color) -> Label:
    var label := Label.new()
    label.text = text
    label.mouse_filter = Control.MOUSE_FILTER_IGNORE
    label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    label.max_lines_visible = 2
    label.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
    label.add_theme_font_size_override("font_size", font_size)
    label.add_theme_color_override("font_color", color)
    return label


static func _rule() -> ColorRect:
    var line := ColorRect.new()
    line.color = Color(1, 1, 1, 0.12)
    line.mouse_filter = Control.MOUSE_FILTER_IGNORE
    line.custom_minimum_size = Vector2(1, 1)
    line.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    return line


## What is on screen: {open, groups: [{title, keys: [[key, does]]}] (listed whether open or not),
## columns, hint, rect ([x, y, w, h] of the frame while open, else []), viewport}.
func hud_report() -> Dictionary:
    if _hud == null:
        return {"open": false, "groups": [], "columns": 0, "hint": "", "rect": [], "viewport": [0, 0]}
    var rect := _frame.get_global_rect()
    var view := get_viewport_rect().size if is_inside_tree() else Vector2.ZERO
    return {
        "open": _open,
        "groups": listed(),
        "columns": _columns.get_child_count() if _open else 0,
        "hint": _hint.text if _open else "",
        "rect": [rect.position.x, rect.position.y, rect.size.x, rect.size.y] if _open else [],
        "viewport": [view.x, view.y],
    }
