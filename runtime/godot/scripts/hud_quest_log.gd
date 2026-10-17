class_name GeogenHudQuestLog
extends Control
## HUD panel: the quest log, a screen listing the player's active, completed and
## failed quests with each one's objectives.
##
## `OPEN_KEY` (J) opens it over the game; the same key or Esc closes it. While it
## is open it is the HUD's modal panel (`GeogenHud.set_modal`): it gets every key,
## and main.gd parks the player and frees the mouse. Up / Down (or a click) select
## a quest, Enter (a double click, or the button) makes the tracker follow it.
##
## It has no quest logic. Everything listed comes from `GeogenRules.report_all(actor)`:
## the loaded package's `report()` (what `--quests` and `quests.list` print) and, for
## packages visited earlier in the session, what theirs said when they unloaded. It
## re-reads on quest events, on `changed` events while open (progress counts) and
## when the HUD refreshes everything (a scene loaded, a save restored, the window
## resized); never per frame.
##
## Only quests the player has: the loaded package's and those of packages visited this
## session (or held in a loaded save). Quests nobody has offered yet (available or
## unavailable) and other scenes' aren't listed, so the log doesn't give them away.
##
## Presentation state only: whether it is open and which quest is selected. What
## is tracked belongs to the tracker (hud_quest_tracker.gd): picking an active quest
## of the loaded package tracks it now, one of another package when that loads again.
## Finished quests can be read but not tracked.

const OPEN_KEY := KEY_J
const SECTIONS := [["active", "Active"], ["complete", "Completed"], ["failed", "Failed"]]
const TITLE_SIZE := 20
const HEAD_SIZE := 12
const TEXT_SIZE := 14
const COLORS := {
    "title": Color(1.0, 0.86, 0.55),
    "text": Color(0.96, 0.96, 0.96),
    "note": Color(0.72, 0.74, 0.78),
    "dim": Color(0.55, 0.57, 0.62),
    "active": Color(1.0, 0.86, 0.55),
    "complete": Color(0.6, 0.9, 0.6),
    "failed": Color(0.95, 0.5, 0.45),
    "done": Color(0.62, 0.78, 0.62),
}
const STATUS_TEXT := {"active": "Active", "complete": "Completed", "failed": "Failed"}

var _hud: GeogenHud
var _open := false
var _selected := ""             # an entry's key: "<package>/<quest id>"
var _entries: Array = []        # what is listed, in display order
var _frame: PanelContainer
var _summary: Label
var _list_scroll: ScrollContainer
var _list: VBoxContainer
var _detail_scroll: ScrollContainer
var _detail: VBoxContainer
var _hint: Label
var _rows := {}                 # key -> its row in the list


var _track_button: Button = null      # the selected quest's "Track this quest", while there is one


func _init() -> void:
    name = "log"
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
    var heading := _label("Quests", TITLE_SIZE + 2, COLORS["title"])
    heading.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    head.add_child(heading)
    _summary = _fixed(_label("", TEXT_SIZE, COLORS["note"]))
    _summary.size_flags_vertical = Control.SIZE_SHRINK_END
    head.add_child(_summary)
    box.add_child(_rule(false))

    var body := HBoxContainer.new()
    body.size_flags_vertical = Control.SIZE_EXPAND_FILL
    body.add_theme_constant_override("separation", 14)
    box.add_child(body)
    _list_scroll = ScrollContainer.new()
    _list_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
    body.add_child(_list_scroll)
    _list = VBoxContainer.new()
    _list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    _list.add_theme_constant_override("separation", 3)
    _list_scroll.add_child(_list)
    body.add_child(_rule(true))
    _detail_scroll = ScrollContainer.new()
    _detail_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
    _detail_scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    body.add_child(_detail_scroll)
    _detail = VBoxContainer.new()
    _detail.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    _detail.add_theme_constant_override("separation", 6)
    _detail_scroll.add_child(_detail)

    box.add_child(_rule(false))
    _hint = _label("", TEXT_SIZE - 1, COLORS["note"])
    _hint.max_lines_visible = 2
    box.add_child(_hint)


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud
    hud.world.rules.quest_event.connect(_on_quest_event)
    hud.world.events.listen(_on_changed, ["changed"])


func _on_quest_event(event: Dictionary) -> void:
    if _open and event["actor"] == _hud.actor:
        _hud.queue_refresh()


## State changed: an objective's count may have moved. Only matters while it is open.
func _on_changed(_event: Dictionary) -> void:
    if _open:
        _hud.queue_refresh()


# -- opening and choosing ------------------------------------------------------------------------

func is_open() -> bool:
    return _open


## Open or close the log. Opening selects the tracked quest.
func set_open(open: bool) -> void:
    if open == _open:
        return
    _open = open
    visible = open
    if open:
        _read()
        var tracked := _tracked_key()
        if tracked != "":
            _selected = tracked
    _hud.set_modal(self, open)
    _hud.refresh()


func hud_key(keycode: int) -> bool:
    if not _open:
        if keycode != OPEN_KEY:
            return false
        set_open(true)
        return true
    match keycode:
        OPEN_KEY, KEY_ESCAPE:
            set_open(false)
        KEY_UP, KEY_W:
            step(-1)
        KEY_DOWN, KEY_S:
            step(1)
        KEY_ENTER, KEY_KP_ENTER, KEY_SPACE:
            track_selected()
    return true             # modal: nothing else gets a key while it is open


## Select the next (or previous) quest in the list; stops at the ends.
func step(by: int) -> void:
    _read()
    if _entries.is_empty():
        return
    var at := _index(_selected)
    _selected = _entries[clampi(at + by, 0, _entries.size() - 1)]["key"]
    _hud.refresh()


## Select a quest by "<package>/<id>", or by id alone (the loaded package's first). False if it isn't listed.
func select(pick: String) -> bool:
    _read()
    var found := ""
    for entry in _entries:
        if entry["key"] == pick:
            found = pick
            break
        if entry["id"] == pick and (found == "" or entry["loaded"]):
            found = entry["key"]
    if found == "":
        return false
    _selected = found
    _hud.refresh()
    return true


## Make the tracker follow the selected quest. False if it isn't an active one.
func track_selected() -> bool:
    _read()
    var at := _index(_selected)
    if at < 0 or _entries[at]["status"] != "active":
        return false
    var entry: Dictionary = _entries[at]
    var tracker: GeogenHudQuestTracker = _hud.panel("quests")
    if tracker == null:
        return false
    if entry["loaded"]:
        if not tracker.track(entry["id"]):
            return false
    else:
        tracker.prefer(entry["package"], entry["id"])
    _hud.refresh()
    return true


func _index(key: String) -> int:
    for i in _entries.size():
        if _entries[i]["key"] == key:
            return i
    return -1


func _tracked_key() -> String:
    for entry in _entries:
        if entry["tracked"] and entry["loaded"]:
            return entry["key"]
    return ""


# -- reading -------------------------------------------------------------------------------------

## The quests to list, in display order: active, completed, failed; in each, the loaded
## package's first.
func _read() -> void:
    var rules: GeogenRules = _hud.world.rules
    var tracker: GeogenHudQuestTracker = _hud.panel("quests")
    var all := rules.report_all(_hud.actor)
    _entries = []
    for section in SECTIONS:
        for loaded in [true, false]:
            for quest in all:
                if quest["status"] != section[0] or bool(quest.get("loaded", true)) != loaded:
                    continue
                var entry := {"key": "%s/%s" % [quest["package"], quest["id"]], "package": quest["package"],
                    "id": quest["id"], "title": quest["title"], "status": quest["status"], "loaded": loaded,
                    "description": str(quest.get("description", "")), "objectives": [], "done": 0, "of": 0}
                entry["tracked"] = quest["status"] == "active" and tracker != null \
                    and tracker.tracked_in(quest["package"]) == quest["id"]
                for objective in quest["objectives"]:
                    var line := {"id": objective["id"], "text": objective["text"], "state": objective["state"],
                        "optional": objective["optional"], "progress_text": ""}
                    if objective.has("progress"):
                        line["progress"] = objective["progress"]
                        line["progress_text"] = "%d of %d" % [objective["progress"][0], objective["progress"][1]]
                    entry["objectives"].append(line)
                    if not objective["optional"]:
                        entry["of"] += 1
                        entry["done"] += 1 if objective["state"] == "done" else 0
                _entries.append(entry)
    if _index(_selected) < 0:
        var tracked := _tracked_key()
        _selected = tracked if tracked != "" else ("" if _entries.is_empty() else _entries[0]["key"])


# -- drawing -------------------------------------------------------------------------------------

func hud_refresh() -> void:
    if _hud == null or not _open:
        return
    _read()
    var view := _hud.get_viewport().get_visible_rect().size
    var room := Vector2(maxf(view.x - 48.0, 40.0), maxf(view.y - 48.0, 40.0))
    _frame.custom_minimum_size = Vector2(minf(880.0, room.x), minf(520.0, room.y))
    _list_scroll.custom_minimum_size.x = clampf(_frame.custom_minimum_size.x * 0.38, 110.0, 330.0)

    var counts := {}
    for entry in _entries:
        counts[entry["status"]] = int(counts.get(entry["status"], 0)) + 1
    var parts := []
    for section in SECTIONS:
        if counts.has(section[0]):
            parts.append("%d %s" % [counts[section[0]], section[1].to_lower()])
    _summary.text = "   ".join(parts)

    for child in _list.get_children():
        _list.remove_child(child)
        child.queue_free()
    _rows.clear()
    var packages := {}
    for entry in _entries:
        packages[entry["package"]] = true
    for section in SECTIONS:
        var members := _entries.filter(func(e): return e["status"] == section[0])
        if members.is_empty():
            continue
        var head := _fixed(_label(section[1].to_upper(), HEAD_SIZE, COLORS[section[0]]))
        var pad := MarginContainer.new()
        pad.mouse_filter = Control.MOUSE_FILTER_IGNORE
        pad.add_theme_constant_override("margin_top", 6 if _list.get_child_count() > 0 else 0)
        pad.add_theme_constant_override("margin_left", 6)
        pad.add_child(head)
        _list.add_child(pad)
        for entry in members:
            var row := Row.new(entry, entry["key"] == _selected, packages.size() > 1)
            row.clicked.connect(_on_row_clicked)
            _rows[entry["key"]] = row
            _list.add_child(row)
    if _entries.is_empty():
        _list.add_child(_label("No quests yet", TEXT_SIZE, COLORS["dim"]))

    for child in _detail.get_children():
        _detail.remove_child(child)
        child.queue_free()
    var at := _index(_selected)
    if at >= 0:
        _draw_detail(_entries[at])
    else:
        _detail.add_child(_label("Quests you are given appear here.", TEXT_SIZE, COLORS["dim"]))
    _hint.text = _hint_text(_entries[at] if at >= 0 else {})
    if _rows.has(_selected):
        _show_row.call_deferred(_rows[_selected])


func _show_row(row: Control) -> void:
    if is_instance_valid(row) and row.is_inside_tree():
        _list_scroll.ensure_control_visible(row)


func _hint_text(entry: Dictionary) -> String:
    var keys := "%s / Esc: close" % OS.get_keycode_string(OPEN_KEY)
    if entry.is_empty():
        return keys
    var track := ""
    if entry["status"] == "active" and not entry["tracked"]:
        track = "Enter: track    "
    return "Up / Down: select    %s%s" % [track, keys]


func _draw_detail(entry: Dictionary) -> void:
    var status: String = entry["status"]
    var title := _label(entry["title"], TITLE_SIZE, COLORS["text"])
    title.max_lines_visible = 3
    _detail.add_child(title)
    var state: String = STATUS_TEXT[status]
    if status == "active":
        if entry["tracked"]:
            state += "  -  tracked" if entry["loaded"] else "  -  tracked there"
    if not entry["loaded"]:
        state += "    (%s)" % _scene_title(entry["package"])
    _detail.add_child(_label(state, TEXT_SIZE, COLORS[status]))
    if entry["description"] != "":
        var about := _label(entry["description"], TEXT_SIZE, COLORS["note"])
        about.max_lines_visible = 6
        _detail.add_child(about)
    var head := _label("OBJECTIVES", HEAD_SIZE, COLORS["note"])
    var pad := MarginContainer.new()
    pad.add_theme_constant_override("margin_top", 8)
    pad.add_child(head)
    _detail.add_child(pad)
    for line in entry["objectives"]:
        _detail.add_child(_objective_row(line, status))
    _track_button = null
    if status == "active" and not entry["tracked"]:
        var button := Button.new()
        button.name = "Track"
        _track_button = button
        button.text = "Track this quest" if entry["loaded"] else "Track when you are back there"
        button.focus_mode = Control.FOCUS_NONE       # keys go through hud_key, not the focused control
        button.size_flags_horizontal = Control.SIZE_SHRINK_BEGIN
        button.add_theme_font_size_override("font_size", TEXT_SIZE)
        button.pressed.connect(track_selected)
        var room := MarginContainer.new()
        room.add_theme_constant_override("margin_top", 10)
        room.add_child(button)
        _detail.add_child(room)


## An objective: a tick when done, a ring while open, a dimmed dash while it waits on
## another (or was left undone when the quest ended).
func _objective_row(line: Dictionary, status: String) -> Control:
    var state: String = line["state"]
    var color: Color = COLORS["done"] if state == "done" else (COLORS["text"] if state == "open"
        else (COLORS["failed"] if state == "missed" else COLORS["dim"]))
    var row := HBoxContainer.new()
    row.add_theme_constant_override("separation", 8)
    var mark := Mark.new()
    mark.kind = state
    mark.color = color
    row.add_child(mark)
    var words: String = line["text"] + (" (optional)" if line["optional"] else "")
    if state == "waiting":
        words += "  -  %s" % ("later" if status == "active" else "not done")
    elif state == "missed":
        words += "  -  missed"
    var text := _label(words, TEXT_SIZE, color)
    text.max_lines_visible = 3
    text.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    row.add_child(text)
    if line["progress_text"] != "":
        var count := _fixed(_label(line["progress_text"], TEXT_SIZE, color if state != "open" else COLORS["title"]))
        count.size_flags_vertical = Control.SIZE_SHRINK_BEGIN
        row.add_child(count)
    return row


## A package as people know it: its scene's name, spaced ("quest_yard" -> "Quest yard").
static func _scene_title(package: String) -> String:
    return package.replace("_", " ").capitalize() if package != "" else ""


func _on_row_clicked(key: String, twice: bool) -> void:
    _selected = key
    if twice:
        track_selected()
    _hud.refresh()


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


## A label that is never wrapped or trimmed (counts, headings): the text beside it gives way.
static func _fixed(label: Label) -> Label:
    label.autowrap_mode = TextServer.AUTOWRAP_OFF
    label.text_overrun_behavior = TextServer.OVERRUN_NO_TRIMMING
    label.max_lines_visible = -1
    return label


static func _rule(vertical: bool) -> ColorRect:
    var line := ColorRect.new()
    line.color = Color(1, 1, 1, 0.12)
    line.mouse_filter = Control.MOUSE_FILTER_IGNORE
    line.custom_minimum_size = Vector2(1, 1)
    if vertical:
        line.size_flags_vertical = Control.SIZE_EXPAND_FILL
    else:
        line.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    return line


## What the log holds: {open, quests: [{key: "<package>/<id>", package, id, title, status:
## active | complete | failed, loaded (its package is the loaded one), tracked, done, of
## (required objectives)}] in display order, sections: {active: [keys], complete, failed},
## selected: {key, package, id, title, status, description, loaded, tracked, can_track,
## objectives: [{id, text, state: done | open | waiting, optional, progress?, progress_text}]}
## ({} when there is none), tracked (key of the loaded package's tracked quest), hint,
## rect: [x, y, w, h] ([] while closed), viewport: [w, h], rows: {key: [x, y, w, h]} and
## track_button: [x, y, w, h] ([] without one) to click}. Read whether it is open or not.
func hud_report() -> Dictionary:
    if _hud == null:
        return {"open": false, "quests": [], "sections": {}, "selected": {}, "tracked": "", "hint": "",
            "rect": [], "viewport": [0, 0]}
    _read()
    var out := {"open": _open, "quests": [], "sections": {}, "selected": {}, "tracked": _tracked_key()}
    for section in SECTIONS:
        out["sections"][section[0]] = []
    for entry in _entries:
        var listed: Dictionary = entry.duplicate(true)
        listed.erase("objectives")
        listed.erase("description")
        out["quests"].append(listed)
        out["sections"][entry["status"]].append(entry["key"])
    var at := _index(_selected)
    if at >= 0:
        out["selected"] = _entries[at].duplicate(true)
        out["selected"]["can_track"] = _entries[at]["status"] == "active" and not _entries[at]["tracked"]
    out["hint"] = _hint_text(out["selected"]) if _open else ""
    var rect := _frame.get_global_rect()
    var view := get_viewport_rect().size if is_inside_tree() else Vector2.ZERO
    out["rect"] = [rect.position.x, rect.position.y, rect.size.x, rect.size.y] if _open else []
    out["viewport"] = [view.x, view.y]
    # Where to click (input.mouse): each listed quest's row, and the Track button.
    out["rows"] = {}
    for key in _rows:
        if _open and is_instance_valid(_rows[key]) and _rows[key].is_inside_tree():
            var r: Rect2 = _rows[key].get_global_rect()
            out["rows"][key] = [r.position.x, r.position.y, r.size.x, r.size.y]
    out["track_button"] = []
    if _open and is_instance_valid(_track_button) and _track_button.is_inside_tree():
        var b := _track_button.get_global_rect()
        out["track_button"] = [b.position.x, b.position.y, b.size.x, b.size.y]
    return out


## A quest in the list: its mark, title and how far along it is. Click to select, twice to track.
class Row extends PanelContainer:
    signal clicked(key: String, twice: bool)

    var key := ""

    func _init(entry: Dictionary, selected: bool, show_package: bool) -> void:
        key = entry["key"]
        mouse_filter = Control.MOUSE_FILTER_STOP
        mouse_default_cursor_shape = Control.CURSOR_POINTING_HAND
        var style := StyleBoxFlat.new()
        style.bg_color = Color(1.0, 0.86, 0.55, 0.16) if selected else Color(1, 1, 1, 0.0)
        style.set_corner_radius_all(5)
        style.border_width_left = 3
        style.border_color = COLORS["title"] if selected else Color(0, 0, 0, 0)
        style.content_margin_left = 8
        style.content_margin_right = 8
        style.content_margin_top = 5
        style.content_margin_bottom = 5
        add_theme_stylebox_override("panel", style)
        var row := HBoxContainer.new()
        row.mouse_filter = Control.MOUSE_FILTER_IGNORE
        row.add_theme_constant_override("separation", 8)
        add_child(row)
        var mark := Mark.new()
        mark.kind = entry["status"]
        mark.color = COLORS[entry["status"]]
        row.add_child(mark)
        var box := VBoxContainer.new()
        box.mouse_filter = Control.MOUSE_FILTER_IGNORE
        box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
        box.add_theme_constant_override("separation", 0)
        row.add_child(box)
        box.add_child(GeogenHudQuestLog._label(entry["title"], TEXT_SIZE + 1,
            COLORS["text"] if entry["status"] == "active" else COLORS["note"]))
        var notes := []
        if entry["tracked"]:
            notes.append("Tracked")
        if show_package or not entry["loaded"]:
            notes.append(GeogenHudQuestLog._scene_title(entry["package"]))
        if not notes.is_empty():
            box.add_child(GeogenHudQuestLog._label("  -  ".join(notes), HEAD_SIZE,
                COLORS["title"] if entry["tracked"] else COLORS["dim"]))
        if entry["status"] == "active" and entry["of"] > 0:
            var count := GeogenHudQuestLog._fixed(GeogenHudQuestLog._label(
                "%d / %d" % [entry["done"], entry["of"]], TEXT_SIZE, COLORS["note"]))
            count.size_flags_vertical = Control.SIZE_SHRINK_CENTER
            row.add_child(count)

    func _gui_input(event: InputEvent) -> void:
        if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
            clicked.emit(key, event.double_click)
            accept_event()


## A mark: quests are a ring with a dot (active), a tick in a disc (complete), a cross in a
## disc (failed); objectives a tick (done), a ring (open), a dash (waiting).
class Mark extends Control:
    var kind := "open"
    var color := Color.WHITE

    func _init() -> void:
        custom_minimum_size = Vector2(18, 20)
        size_flags_vertical = Control.SIZE_SHRINK_BEGIN
        mouse_filter = Control.MOUSE_FILTER_IGNORE

    func _draw() -> void:
        var c := Vector2(9, 11)
        var tick := PackedVector2Array([c + Vector2(-3.6, 0.3), c + Vector2(-1.0, 3.1), c + Vector2(4.0, -3.1)])
        match kind:
            "active":
                draw_arc(c, 6.0, 0.0, TAU, 24, color, 1.5, true)
                draw_circle(c, 2.2, color)
            "complete":
                draw_circle(c, 7.5, Color(color, 0.3))
                draw_polyline(tick, color, 1.8, true)
            "missed":
                draw_line(c + Vector2(-3.2, -3.2), c + Vector2(3.2, 3.2), color, 1.8, true)
                draw_line(c + Vector2(-3.2, 3.2), c + Vector2(3.2, -3.2), color, 1.8, true)
            "failed":
                draw_circle(c, 7.5, Color(color, 0.3))
                draw_line(c + Vector2(-3.2, -3.2), c + Vector2(3.2, 3.2), color, 1.8, true)
                draw_line(c + Vector2(-3.2, 3.2), c + Vector2(3.2, -3.2), color, 1.8, true)
            "done":
                draw_circle(c, 6.0, Color(color, 0.35))
                draw_polyline(tick, color, 1.8, true)
            "open":
                draw_arc(c, 4.5, 0.0, TAU, 20, color, 1.5, true)
            _:
                draw_line(c + Vector2(-3.5, 0), c + Vector2(3.5, 0), color, 1.5, true)
