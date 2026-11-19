class_name GeogenHudUi
extends RefCounted
## The HUD's look, shared by its panels: the palette, panel frames, labels, rules and the shell
## of a modal screen (a dimmer with a framed column in the middle). Each panel keeps its own
## `COLORS` of roles (title, note, done, ...), drawn from this palette where they agree.

## The tracker's gold: titles, keys, the active quest.
const GOLD := Color(1.0, 0.86, 0.55)
## The markers' and the selection's deeper gold.
const AMBER := Color(1.0, 0.82, 0.4)
const TEXT := Color(0.96, 0.96, 0.96)
## Secondary text: notes, hints, group headings.
const NOTE := Color(0.72, 0.74, 0.78)
## A done objective, a complete quest, a failed one.
const DONE := Color(0.62, 0.78, 0.62)
const COMPLETE := Color(0.6, 0.9, 0.6)
const FAILED := Color(0.95, 0.5, 0.45)
## Behind the panels at the screen's edges (stats, tracker, inventory).
const BACK := Color(0.05, 0.06, 0.08, 0.62)
## Behind a modal screen's frame.
const MODAL_BACK := Color(0.06, 0.07, 0.09, 0.97)
## Behind a one-line strip (the banner, the crossing signal).
const STRIP_BACK := Color(0.03, 0.04, 0.06, 0.66)
## A thin dividing line.
const RULE := Color(1, 1, 1, 0.12)
const SHADOW := Color(0, 0, 0, 0.7)


## A panel's frame: its background, corner radius and content margins (one for every side, or
## [left, right, top, bottom]; -1 or none leaves them at the default), and optionally a coloured edge
## on one side ("left" or "top").
static func panel(back: Color, radius: int, margins: Variant = null, edge := "", edge_color := Color(0, 0, 0, 0),
        edge_width := 3) -> StyleBoxFlat:
    var style := StyleBoxFlat.new()
    style.bg_color = back
    style.set_corner_radius_all(radius)
    if margins is int:
        style.set_content_margin_all(margins)
    elif margins != null:
        for i in 4:
            if margins[i] >= 0:
                style.set_content_margin([SIDE_LEFT, SIDE_RIGHT, SIDE_TOP, SIDE_BOTTOM][i], margins[i])
    if edge != "":
        style.set("border_width_" + edge, edge_width)
        style.border_color = edge_color
    return style


## A label that lets the mouse through, in a size and colour.
static func label(text: String, font_size: int, color: Color) -> Label:
    var l := Label.new()
    l.text = text
    l.mouse_filter = Control.MOUSE_FILTER_IGNORE
    l.add_theme_font_size_override("font_size", font_size)
    l.add_theme_color_override("font_color", color)
    return l


## A label wrapped at word boundaries to at most `lines` lines (0: any number), with an ellipsis
## where it is cut.
static func wrapped(l: Label, lines := 2) -> Label:
    l.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    if lines > 0:
        l.max_lines_visible = lines
        l.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
    return l


## A label with the small drop shadow that keeps it legible over the scene.
static func shadowed(l: Label) -> Label:
    l.add_theme_color_override("font_shadow_color", SHADOW)
    l.add_theme_constant_override("shadow_offset_x", 1)
    l.add_theme_constant_override("shadow_offset_y", 1)
    return l


## A label on one line that is never cut.
static func fixed(l: Label) -> Label:
    l.autowrap_mode = TextServer.AUTOWRAP_OFF
    l.text_overrun_behavior = TextServer.OVERRUN_NO_TRIMMING
    l.max_lines_visible = -1
    return l


## A 1 px dividing line across a column (or down a row, `vertical`).
static func rule(vertical := false) -> ColorRect:
    var line := ColorRect.new()
    line.color = RULE
    line.mouse_filter = Control.MOUSE_FILTER_IGNORE
    line.custom_minimum_size = Vector2(1, 1)
    if vertical:
        line.size_flags_vertical = Control.SIZE_EXPAND_FILL
    else:
        line.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    return line


## A modal screen's shell in `owner` (a full-screen Control): the scene dimmed, and in the middle
## a frame with a gold top edge holding a column. Returns the frame; the column is its child.
static func modal(owner: Control) -> PanelContainer:
    var dim := ColorRect.new()
    dim.color = Color(0, 0, 0, 0.5)
    dim.mouse_filter = Control.MOUSE_FILTER_IGNORE
    dim.set_anchors_preset(Control.PRESET_FULL_RECT)
    owner.add_child(dim)
    var center := CenterContainer.new()
    center.mouse_filter = Control.MOUSE_FILTER_IGNORE
    center.set_anchors_preset(Control.PRESET_FULL_RECT)
    owner.add_child(center)
    var frame := PanelContainer.new()
    frame.add_theme_stylebox_override("panel", panel(MODAL_BACK, 8, [18, 18, 12, 12], "top", GOLD))
    center.add_child(frame)
    var box := VBoxContainer.new()
    box.add_theme_constant_override("separation", 10)
    frame.add_child(box)
    return frame
