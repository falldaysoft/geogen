class_name GeogenHud
extends CanvasLayer
## The game HUD: one layer that panels join (the quest tracker; later
## notifications, world markers, the quest log, inventory, dialogue).
##
## The host owns what panels share and nothing a panel shows:
## - a slot per screen corner (`SLOTS`), each a column inside the screen margin,
##   so panels in one corner stack instead of overlapping;
## - sizes that follow the window (`panel_width()`, `slot_height(slot)`);
## - one coalesced refresh: `queue_refresh()` any number of times in a frame
##   re-reads each panel once, after the events of that frame are through;
## - a refresh of every panel when the world's program or state is replaced
##   (a scene loaded or unloaded, a save restored) and when the window resizes,
##   so panels survive scene switches without tracking them;
## - key presses, offered to panels in the order they were added;
## - `report()`: what every panel is displaying, for the control API and tests.
##
## A panel is a Control added with `add_panel(panel, slot)`. It may define
## (all optional):
##   hud_attach(hud)         called once, before it enters the tree: keep `hud`, connect to
##                           what it shows (hud.world.events, hud.world.rules, ...) and call
##                           hud.queue_refresh() from those handlers
##   hud_refresh()           re-read its source and redraw
##   hud_key(keycode) -> bool   a key was pressed; true if the panel used it
##   hud_report() -> Dictionary  what it displays
## Panels read what the console reads; they hold presentation state only.
##
## The layer sits under main.gd's $Overlay (debug text, prompt, fade), so a
## scene switch's fade covers the HUD too.

const SLOTS := ["top_left", "top_right", "bottom_left", "bottom_right"]
const MARGIN := 12
## The debug overlay's three lines are in the top-left corner and run most of the
## way across a small window: the top slots start under them.
const TOP_INSET := 84

var world: WorldLoader
## Whose game this HUD shows.
var actor := "player"
## Returns true while something modal (the scene menu) has the keyboard.
var input_blocked := Callable()

var _panels := {}           # name -> Control, in the order added
var _slots := {}            # slot -> VBoxContainer
var _queued := false


func _init() -> void:
    name = "Hud"
    layer = 0
    for slot in SLOTS:
        var column := VBoxContainer.new()
        column.name = slot.to_pascal_case()
        column.mouse_filter = Control.MOUSE_FILTER_IGNORE
        column.add_theme_constant_override("separation", 8)
        var right: bool = slot.ends_with("right")
        var bottom: bool = slot.begins_with("bottom")
        column.anchor_left = 1.0 if right else 0.0
        column.anchor_right = column.anchor_left
        column.anchor_top = 1.0 if bottom else 0.0
        column.anchor_bottom = column.anchor_top
        column.grow_horizontal = Control.GROW_DIRECTION_BEGIN if right else Control.GROW_DIRECTION_END
        column.grow_vertical = Control.GROW_DIRECTION_BEGIN if bottom else Control.GROW_DIRECTION_END
        column.alignment = BoxContainer.ALIGNMENT_END if bottom else BoxContainer.ALIGNMENT_BEGIN
        var x := -MARGIN if right else MARGIN
        var y := -MARGIN if bottom else TOP_INSET
        column.offset_left = x
        column.offset_right = x
        column.offset_top = y
        column.offset_bottom = y
        add_child(column)
        _slots[slot] = column


func _ready() -> void:
    world.world_loaded.connect(func(_aabb: AABB): queue_refresh())
    world.rules.reset.connect(queue_refresh)
    get_viewport().size_changed.connect(queue_refresh)
    queue_refresh()


## Add a panel to a corner's column. Its node name is its name in `report()` and `panel()`.
func add_panel(panel: Control, slot := "top_right") -> void:
    assert(slot in SLOTS, "hud slot: one of %s" % [SLOTS])
    _panels[String(panel.name)] = panel
    if panel.has_method("hud_attach"):
        panel.hud_attach(self)
    _slots[slot].add_child(panel)
    queue_refresh()


func panel(panel_name: String) -> Control:
    return _panels.get(panel_name)


## Width for a side panel: about a third of the window, within readable limits.
func panel_width() -> float:
    var width := get_viewport().get_visible_rect().size.x
    return minf(clampf(width * 0.36, 170.0, 320.0), maxf(width - 2 * MARGIN, 40.0))


## The height a slot's column can fill before it leaves the screen.
func slot_height(slot := "top_right") -> float:
    var top := float(MARGIN if slot.begins_with("bottom") else TOP_INSET)
    return maxf(get_viewport().get_visible_rect().size.y - top - MARGIN, 40.0)


## Refresh every panel once, after whatever is happening now has finished.
func queue_refresh() -> void:
    if _queued:
        return
    _queued = true
    _refresh.call_deferred()


## Refresh every panel now (tools that read `report()` straight after a change).
func refresh() -> void:
    _refresh()


func _refresh() -> void:
    _queued = false
    for panel_name in _panels:
        if _panels[panel_name].has_method("hud_refresh"):
            _panels[panel_name].hud_refresh()


## What each panel displays: {panel name: its hud_report()}.
func report() -> Dictionary:
    var out := {}
    for panel_name in _panels:
        if _panels[panel_name].has_method("hud_report"):
            out[panel_name] = _panels[panel_name].hud_report()
    return out


func _unhandled_input(event: InputEvent) -> void:
    if not (event is InputEventKey and event.pressed and not event.echo):
        return
    if input_blocked.is_valid() and input_blocked.call():
        return
    for panel_name in _panels:
        if _panels[panel_name].has_method("hud_key") and _panels[panel_name].hud_key(event.physical_keycode):
            get_viewport().set_input_as_handled()
            return
