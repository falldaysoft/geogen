class_name GeogenHudQuestTracker
extends PanelContainer
## HUD panel: the tracked quest's title and its current objectives.
##
## It has no quest logic. Everything shown comes from `GeogenRules.report(actor)`
## (what `--quests` and `quests.list` print): a quest's status, each objective's
## state (done | open | missed | waiting) and `progress: [n, of]`. It re-reads that
## when a quest event arrives, when state changes while a quest is shown (progress
## counts), when the HUD refreshes everything (a scene loaded, a save restored, the
## window resized), and once a second while a quest is shown, in case what it shows
## moved with nothing announcing it (progress that reads the clock); never per frame.
##
## What it adds is presentation state:
## - which active quest is tracked (the first one, until `CYCLE_KEY` or
##   `cycle()` picks another; remembered per package across scene switches);
## - a finished quest stays up for `linger` seconds marked complete or failed,
##   then the next active one takes its place. Hidden when there is none.
##
## Objectives that wait on another (`after`) aren't listed; a failed quest lists the ones it
## failed with open (missed), crossed out. Text wraps to two
## lines and the panel clips at the bottom of the screen.

const CYCLE_KEY := KEY_Q
const TITLE_SIZE := 16
const TEXT_SIZE := 14
const COLORS := {
    "title": Color(1.0, 0.86, 0.55),
    "open": Color(0.96, 0.96, 0.96),
    "done": Color(0.62, 0.78, 0.62),
    "note": Color(0.72, 0.74, 0.78),
    "complete": Color(0.6, 0.9, 0.6),
    "failed": Color(0.95, 0.5, 0.45),
}

## Seconds a finished quest stays up before the next one replaces it.
var linger := 4.0:
    set(value):
        linger = maxf(value, 0.0)
        if not _finished.is_empty():
            _linger(linger)

var _hud: GeogenHud
var _tracked := ""              # quest id, in the loaded package
var _choice := {}               # package -> the quest tracked there
var _last := {}                 # {package, quest}: the latest pick, which wins where it is running
var _finished := {}             # {package, quest}: the tracked quest ended and is still shown
var _shown := {}                # what hud_report() returns
var _timer: Timer
var _recheck: Timer             # once a second while shown: has what it would show moved?
var _settle := 0                # frames left of fitting the height after a redraw
var _scroll: ScrollContainer
var _box: VBoxContainer
var _title: Label
var _note: Label
var _rows: VBoxContainer


func _init() -> void:
    name = "quests"
    visible = false
    mouse_filter = Control.MOUSE_FILTER_IGNORE
    var style := StyleBoxFlat.new()
    style.bg_color = Color(0.05, 0.06, 0.08, 0.62)
    style.set_corner_radius_all(6)
    style.border_width_left = 3
    style.border_color = COLORS["title"]
    style.content_margin_left = 12
    style.content_margin_right = 10
    style.content_margin_top = 8
    style.content_margin_bottom = 9
    add_theme_stylebox_override("panel", style)
    # The column clips when it is taller than the screen allows: a ScrollContainer with no bars.
    _scroll = ScrollContainer.new()
    _scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
    _scroll.vertical_scroll_mode = ScrollContainer.SCROLL_MODE_SHOW_NEVER
    _scroll.mouse_filter = Control.MOUSE_FILTER_IGNORE
    add_child(_scroll)
    _box = VBoxContainer.new()
    _box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    _box.add_theme_constant_override("separation", 3)
    _box.minimum_size_changed.connect(_fit)
    _scroll.add_child(_box)
    _title = _label("", TITLE_SIZE, COLORS["title"])
    _box.add_child(_title)
    _note = _label("", TEXT_SIZE - 2, COLORS["note"])
    _box.add_child(_note)
    _rows = VBoxContainer.new()
    _rows.add_theme_constant_override("separation", 3)
    _box.add_child(_rows)
    _timer = Timer.new()
    _timer.one_shot = true
    _timer.timeout.connect(_linger_over)
    add_child(_timer)
    _recheck = Timer.new()
    _recheck.wait_time = 1.0
    _recheck.timeout.connect(_check_shown)
    add_child(_recheck)
    set_process(false)


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud
    hud.world.rules.quest_event.connect(_on_quest_event)
    hud.world.events.listen(_on_changed, ["changed"])


# -- what moves it -------------------------------------------------------------------------------

func _on_quest_event(event: Dictionary) -> void:
    if event["actor"] != _hud.actor:
        return
    var package: String = _hud.world.rules.quest_package(event["quest"])
    match event["event"]:
        "started":
            if _tracked == "" and _finished.is_empty():
                _tracked = event["quest"]
        "complete", "failed":
            if event["quest"] == _tracked:
                _finished = {"package": package, "quest": _tracked}
                _linger(linger)
    _hud.queue_refresh()


## Redraw if the tracked quest's report no longer matches what is shown (its status, objective
## states and counts): progress from the clock moves without a `changed` event.
func _check_shown() -> void:
    if _hud == null or not visible or _tracked == "":
        return
    for entry in _hud.world.rules.report(_hud.actor):
        if entry["id"] != _tracked:
            continue
        var now: Array = entry["objectives"].filter(func(o): return o["state"] != "waiting").map(
            func(o): return [o["id"], o["state"], o.get("progress", [])])
        var shown: Array = _shown.get("lines", []).map(func(l): return [l["id"], l["state"], l.get("progress", [])])
        if entry["status"] != _shown.get("status", "") or now != shown:
            _hud.queue_refresh()
        return


## State changed: an objective's count may have moved. Only matters while something is shown.
func _on_changed(_event: Dictionary) -> void:
    if _tracked != "":
        _hud.queue_refresh()


func _linger(seconds: float) -> void:
    if seconds <= 0.0:
        _timer.stop()
        _linger_over()
    else:
        _timer.start(seconds)


func _linger_over() -> void:
    if _finished.is_empty():
        return
    if _finished["package"] in _hud.world.rules.running():
        _tracked = ""
    _choice.erase(_finished["package"])
    _finished = {}
    _hud.queue_refresh()


# -- choosing the quest --------------------------------------------------------------------------

## Track the next (or previous) active quest. Returns the tracked id ("" if none is active).
func cycle(step := 1) -> String:
    var active := _active(_hud.world.rules.report(_hud.actor))
    if not _finished.is_empty():        # the finished one gives way at once
        _timer.stop()
        _choice.erase(_finished["package"])
        _finished = {}
        _tracked = active[0] if not active.is_empty() else ""
    elif not active.is_empty():
        _tracked = active[posmod(active.find(_tracked) + step, active.size())]
    _hud.refresh()
    return _tracked


## Track one quest by id. False if it isn't an active quest of the HUD's actor.
func track(quest: String) -> bool:
    if not quest in _active(_hud.world.rules.report(_hud.actor)):
        return false
    _timer.stop()
    _finished = {}
    _tracked = quest
    _hud.refresh()
    return true


## The quest tracked in a package ("" if none): the one shown for the loaded package, the
## pick waiting for another to load again.
func tracked_in(package: String) -> String:
    var rules: GeogenRules = _hud.world.rules
    if _tracked != "" and package in rules.running() and rules.quest_package(_tracked) == package:
        return _tracked
    return "" if package in rules.running() else str(_choice.get(package, ""))


## The quest tracked in the loaded world ("" if none), whatever package it belongs to.
func tracked_here() -> String:
    return _tracked if _hud.world.rules.machine != null else ""


## Pick the quest to track in a package that isn't loaded: it is tracked when that loads again.
func prefer(package: String, quest: String) -> void:
    _choice[package] = quest
    _last = {"package": package, "quest": quest}


func hud_key(keycode: int) -> bool:
    if keycode != CYCLE_KEY or _tracked == "":
        return false
    cycle()
    return true


static func _active(report: Array) -> Array:
    return report.filter(func(q): return q["status"] == "active").map(func(q): return q["id"])


# -- drawing -------------------------------------------------------------------------------------

func hud_refresh() -> void:
    var rules: GeogenRules = _hud.world.rules
    var report := rules.report(_hud.actor)
    var active := _active(report)
    var lingering: bool = not _finished.is_empty() and _finished["package"] in rules.running()
    if rules.machine != null and not lingering:
        # Keep the pick while it is active; else the one picked before in a package running
        # here (the scene's, then its content packages': a quest keeps its pick across the
        # scenes its package runs in); else the first.
        if not _tracked in active:
            _tracked = active[0] if not active.is_empty() else ""
            var last: String = _last.get("quest", "")
            if last in active and rules.quest_package(last) == _last.get("package", ""):
                _tracked = last
            else:
                for name in rules.running():
                    var before: String = _choice.get(name, "")
                    if before in active and rules.quest_package(before) == name:
                        _tracked = before
                        break
        if _tracked != "":
            _choice[rules.quest_package(_tracked)] = _tracked
            _last = {"package": rules.quest_package(_tracked), "quest": _tracked}
    var quest := {}
    for entry in report:
        if entry["id"] == _tracked and (lingering or entry["status"] == "active"):
            quest = entry
    if rules.machine == null:
        _tracked = ""               # nothing is loaded: the pick comes back from _choice with the package
    _shown = {"visible": not quest.is_empty(), "tracked": quest.get("id", ""), "package": quest.get("package", ""),
        "title": quest.get("title", ""), "status": quest.get("status", ""), "note": "", "active": active,
        "lines": [], "waiting": 0}
    visible = _shown["visible"]
    if quest.is_empty():
        _recheck.stop()
        return
    if _recheck.is_stopped():
        _recheck.start()
    var status: String = quest["status"]
    var note := ""
    if status == "complete":
        note = "Quest complete"
    elif status == "failed":
        note = "Quest failed"
    elif active.size() > 1:
        note = "%d of %d    %s: next quest" % [active.find(_tracked) + 1, active.size(), OS.get_keycode_string(CYCLE_KEY)]
    _shown["note"] = note
    _title.text = quest["title"]
    _note.text = note
    _note.visible = note != ""
    _note.add_theme_color_override("font_color", COLORS.get(status, COLORS["note"]))
    for row in _rows.get_children():
        _rows.remove_child(row)
        row.queue_free()
    for objective in quest["objectives"]:
        if objective["state"] == "waiting":
            _shown["waiting"] += 1
            continue
        var line := {"id": objective["id"], "state": objective["state"], "optional": objective["optional"],
            "text": objective["text"] + (" (optional)" if objective["optional"] else ""), "progress_text": ""}
        if objective.has("progress"):
            line["progress"] = objective["progress"]
            line["progress_text"] = "%d of %d" % [objective["progress"][0], objective["progress"][1]]
        _shown["lines"].append(line)
        _rows.add_child(_row(line))
    custom_minimum_size.x = _hud.panel_width()
    _fit()
    _settle = 4
    set_process(true)


## Wrapped text only knows its height once it has its width: fit again over the next frames.
func _process(_delta: float) -> void:
    _fit()
    _settle -= 1
    if _settle <= 0:
        set_process(false)


## Clip the column at what the screen allows (called again when wrapping changes its height).
func _fit() -> void:
    if _hud == null:
        return
    var style := get_theme_stylebox("panel")
    var room := _hud.slot_height() - style.content_margin_top - style.content_margin_bottom
    var wanted := _box.get_combined_minimum_size().y
    _scroll.custom_minimum_size.y = minf(wanted, room)
    _shown["clipped"] = wanted > room + 0.5


func _row(line: Dictionary) -> Control:
    var done: bool = line["state"] == "done"
    var color: Color = COLORS["done" if done else ("failed" if line["state"] == "missed" else "open")]
    var row := HBoxContainer.new()
    row.add_theme_constant_override("separation", 7)
    var mark := Mark.new()
    mark.done = done
    mark.missed = line["state"] == "missed"
    mark.color = color
    row.add_child(mark)
    var text := _label(line["text"], TEXT_SIZE, color)
    text.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    row.add_child(text)
    if line["progress_text"] != "":
        var count := _label(line["progress_text"], TEXT_SIZE, color if done else COLORS["title"])
        count.autowrap_mode = TextServer.AUTOWRAP_OFF           # never wrapped or trimmed: the text gives way
        count.text_overrun_behavior = TextServer.OVERRUN_NO_TRIMMING
        count.max_lines_visible = -1
        count.size_flags_vertical = Control.SIZE_SHRINK_BEGIN
        row.add_child(count)
    return row


static func _label(text: String, font_size: int, color: Color) -> Label:
    var label := Label.new()
    label.text = text
    label.mouse_filter = Control.MOUSE_FILTER_IGNORE
    label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    label.max_lines_visible = 2
    label.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
    label.add_theme_font_size_override("font_size", font_size)
    label.add_theme_color_override("font_color", color)
    label.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.7))
    label.add_theme_constant_override("shadow_offset_x", 1)
    label.add_theme_constant_override("shadow_offset_y", 1)
    return label


## What is on screen: {visible, tracked, package, title, status, note, active: [ids],
## lines: [{id, text, state: done | open, optional, progress?: [n, of], progress_text}],
## waiting (objectives not listed yet), clipped, rect: [x, y, w, h], viewport: [w, h]}.
func hud_report() -> Dictionary:
    var out := _shown.duplicate(true)
    if out.is_empty():
        out = {"visible": false, "tracked": "", "package": "", "title": "", "status": "", "note": "", "active": [],
            "lines": [], "waiting": 0}
    out["clipped"] = bool(out.get("clipped", false)) and visible
    var rect := get_global_rect()
    var view := get_viewport_rect().size if is_inside_tree() else Vector2.ZERO
    out["rect"] = [rect.position.x, rect.position.y, rect.size.x, rect.size.y] if visible else []
    out["viewport"] = [view.x, view.y]
    out["on_screen"] = bool(out["visible"]) and is_visible_in_tree()     # not under a modal panel
    return out


## An objective's bullet: a ring while open, a filled disc with a tick when done, a cross when
## its quest failed with it open (missed).
class Mark extends Control:
    var done := false
    var missed := false
    var color := Color.WHITE

    func _init() -> void:
        custom_minimum_size = Vector2(14, 20)
        size_flags_vertical = Control.SIZE_SHRINK_BEGIN
        mouse_filter = Control.MOUSE_FILTER_IGNORE

    func _draw() -> void:
        var c := Vector2(7, 11)
        if missed:
            draw_line(c + Vector2(-3.2, -3.2), c + Vector2(3.2, 3.2), color, 1.8, true)
            draw_line(c + Vector2(-3.2, 3.2), c + Vector2(3.2, -3.2), color, 1.8, true)
        elif done:
            draw_circle(c, 6.0, Color(color, 0.35))
            draw_polyline(PackedVector2Array([c + Vector2(-3.2, 0.2), c + Vector2(-0.8, 2.8), c + Vector2(3.6, -2.8)]),
                color, 1.8, true)
        else:
            draw_arc(c, 4.5, 0.0, TAU, 20, color, 1.5, true)
