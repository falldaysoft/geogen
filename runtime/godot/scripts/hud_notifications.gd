class_name GeogenHudNotifications
extends VBoxContainer
## HUD panel: toasts for what happens to the player's quests, and what rules tell them.
##
## A quest event (`GeogenRules.quest_event`: started, objective_done, complete,
## failed) of the HUD's actor becomes a toast: a heading for the kind and the
## quest's title or the objective's text, with a short chime (sound.gd). Other
## actors' quests raise nothing.
##
## Feedback (`GeogenRules.feedback`) for the HUD's actor, or for nobody in particular:
## a `notify` is a `notice` toast of its text (no heading, no chime of its own), and a
## `play` sounds its chime at once, without a toast (`sounds` counts them).
##
## It has no quest logic, only presentation state: the toasts on screen and
## the ones waiting.
## - At most `limit` are up at once, and one appears no sooner than `gap`
##   seconds after the one before, so several in one moment (the last objective,
##   the quest complete, the next quest started) arrive one by one, in the order
##   they happened, each with its own chime. The rest wait their turn.
## - A toast stays `hold` seconds from when it appears, then fades. New ones
##   join below, so the ones being read only move up.
## - Events are the only source. A scene switch, a reload or a restored save
##   fire none (the state resumes), so they raise no toasts, and toasts already
##   up or waiting carry over a switch. When the program or its state is
##   replaced (`GeogenRules.reset`), a toast about the loaded package that is
##   no longer true there (a save from before the quest was finished) is dropped.

const HEAD_SIZE := 12
const TEXT_SIZE := 15
const FADE_IN := 0.15
## Each kind's heading, colour and chime (GeogenSound.CHIMES).
const KINDS := {
    "started": {"heading": "Quest started", "color": Color(1.0, 0.86, 0.55)},
    "objective_done": {"heading": "Objective complete", "color": Color(0.62, 0.82, 0.95)},
    "complete": {"heading": "Quest complete", "color": Color(0.6, 0.9, 0.6)},
    "failed": {"heading": "Quest failed", "color": Color(0.95, 0.5, 0.45)},
    "notice": {"heading": "", "color": Color(0.85, 0.85, 0.88)},
}

## Seconds a toast stays up before it fades.
var hold := 4.0
## Seconds between one toast appearing and the next.
var gap := 0.6
## Seconds a toast takes to fade out.
var fade := 0.4
## How many are up at once.
var limit := 3
## No chimes (the toasts still say which they would have played).
var muted := false

var _hud: GeogenHud
var _queue: Array[Dictionary] = []      # notes waiting, oldest first
var _showing: Array[Dictionary] = []    # {note, toast, age}, oldest first
var _since := INF                       # seconds since the last one appeared
var _raised := 0                        # notes ever raised: the next one's `seq` is this + 1
var _chimes := 0
var _sounds: Array[String] = []         # what `play` feedback asked for, in order
var _recheck := false                   # the state was replaced: drop what is no longer true
var _player: AudioStreamPlayer


func _init() -> void:
    name = "notifications"
    visible = false
    mouse_filter = Control.MOUSE_FILTER_IGNORE
    add_theme_constant_override("separation", 6)
    _player = AudioStreamPlayer.new()
    _player.volume_db = -8.0
    add_child(_player)
    set_process(false)


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud
    hud.world.rules.quest_event.connect(_on_quest_event)
    hud.world.rules.feedback.connect(_on_feedback)
    hud.world.rules.reset.connect(func(): _recheck = true)


# -- what raises them ----------------------------------------------------------------------------

func _on_quest_event(event: Dictionary) -> void:
    if event["actor"] != _hud.actor or not KINDS.has(event["event"]):
        return
    _raised += 1
    var note := {"seq": _raised, "kind": event["event"], "package": _hud.world.rules.quest_package(event["quest"]),
        "quest": event["quest"], "heading": KINDS[event["event"]]["heading"], "text": event["title"]}
    if event["event"] == "objective_done":
        note["objective"] = event["objective"]
        note["text"] = event.get("text", event["title"])
    _queue.append(note)
    set_process(true)


func _on_feedback(kind: String, value: String, actor: String) -> void:
    if actor != _hud.actor and actor != "":
        return
    if kind == "play":
        _sounds.append(value)
        if not muted and DisplayServer.get_name() != "headless":
            GeogenSound.chime(_player, value)
        return
    _raised += 1
    _queue.append({"seq": _raised, "kind": "notice", "package": _hud.world.rules.package, "quest": "",
        "heading": "", "text": value})
    set_process(true)


## Drop every toast, shown or waiting.
func clear() -> void:
    _queue.clear()
    for entry in _showing:
        _remove(entry)
    _showing.clear()
    _since = INF
    visible = false


func hud_refresh() -> void:
    if _recheck:
        _recheck = false
        _drop_untrue()
    for entry in _showing:
        entry["toast"].custom_minimum_size.x = _hud.panel_width()


## After the state was replaced: toasts about the loaded package say what its state says.
func _drop_untrue() -> void:
    var rules: GeogenRules = _hud.world.rules
    if rules.machine == null:
        return
    var quests := {}
    for quest in rules.report(_hud.actor):
        quests[quest["id"]] = quest
    var holds := func(note: Dictionary) -> bool:
        if not note["package"] in rules.running() or note["kind"] == "notice":
            return true
        if not quests.has(note["quest"]):
            return false
        var quest: Dictionary = quests[note["quest"]]
        match note["kind"]:
            "started":
                return quest["status"] in ["active", "complete", "failed"]
            "objective_done":
                return quest["objectives"].any(func(o): return o["id"] == note["objective"] and o["state"] == "done")
        return quest["status"] == note["kind"]
    _queue = _queue.filter(holds)
    for entry in _showing.duplicate():
        if not holds.call(entry["note"]):
            _remove(entry)
            _showing.erase(entry)
    visible = not _showing.is_empty()


# -- showing them --------------------------------------------------------------------------------

func _process(delta: float) -> void:
    _since += delta
    for entry in _showing.duplicate():
        entry["age"] += delta
        var age: float = entry["age"]
        if age >= hold + fade:
            _remove(entry)
            _showing.erase(entry)
        else:
            entry["toast"].modulate.a = minf(clampf(age / FADE_IN, 0.0, 1.0),
                clampf((hold + fade - age) / maxf(fade, 0.001), 0.0, 1.0))
    while not _queue.is_empty() and _showing.size() < limit and _since >= gap:
        _show(_queue.pop_front())
    visible = not _showing.is_empty()
    if _showing.is_empty() and _queue.is_empty():
        _since = INF
        set_process(false)


func _show(note: Dictionary) -> void:
    var toast := Toast.new(note, KINDS[note["kind"]]["color"])
    toast.custom_minimum_size.x = _hud.panel_width()
    toast.modulate.a = 0.0
    add_child(toast)
    _showing.append({"note": note, "toast": toast, "age": 0.0})
    _since = 0.0
    if muted or note["kind"] == "notice" or not GeogenSound.CHIMES.has(note["kind"]):
        return
    _chimes += 1
    # Headless runs have nobody to hear it, and a chime cut off by quitting is reported as a leak.
    if DisplayServer.get_name() != "headless":
        GeogenSound.chime(_player, note["kind"])


func _remove(entry: Dictionary) -> void:
    remove_child(entry["toast"])
    entry["toast"].queue_free()


## What is on screen and what waits: {visible, showing: [{seq, kind: started |
## objective_done | complete | failed | notice, package, quest, objective?, heading, text, age}],
## queued: [the same, without age], raised (toasts ever raised: the last one's seq),
## chimes (sounded; counted in a headless run too, where nothing plays), sounds (what `play`
## feedback played, in order), hold, gap, limit, rect: [x, y, w, h], viewport: [w, h]}.
func hud_report() -> Dictionary:
    var showing := []
    for entry in _showing:
        var note: Dictionary = entry["note"].duplicate()
        note["age"] = entry["age"]
        showing.append(note)
    var rect := get_global_rect()
    var view := get_viewport_rect().size if is_inside_tree() else Vector2.ZERO
    return {"visible": visible, "showing": showing, "queued": _queue.duplicate(true), "raised": _raised,
        "chimes": _chimes, "sounds": _sounds.duplicate(), "hold": hold, "gap": gap, "limit": limit,
        "rect": [rect.position.x, rect.position.y, rect.size.x, rect.size.y] if visible else [],
        "viewport": [view.x, view.y]}


## One toast: a bar and a mark in the kind's colour, its heading, and what it is about.
class Toast extends PanelContainer:
    func _init(note: Dictionary, color: Color) -> void:
        mouse_filter = Control.MOUSE_FILTER_IGNORE
        var style := StyleBoxFlat.new()
        style.bg_color = Color(0.05, 0.06, 0.08, 0.72)
        style.set_corner_radius_all(6)
        style.border_width_left = 3
        style.border_color = color
        style.content_margin_left = 10
        style.content_margin_right = 10
        style.content_margin_top = 6
        style.content_margin_bottom = 7
        add_theme_stylebox_override("panel", style)
        var row := HBoxContainer.new()
        row.add_theme_constant_override("separation", 8)
        add_child(row)
        var mark := Mark.new()
        mark.kind = note["kind"]
        mark.color = color
        row.add_child(mark)
        var box := VBoxContainer.new()
        box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
        box.add_theme_constant_override("separation", 0)
        row.add_child(box)
        if note["heading"] != "":
            box.add_child(_label(note["heading"].to_upper(), HEAD_SIZE, color))
        box.add_child(_label(note["text"], TEXT_SIZE, Color(0.96, 0.96, 0.96)))

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


## A toast's mark: a ring with a dot (started), a tick (objective), a tick in a disc
## (complete), a cross in a disc (failed), an i in a ring (notice).
class Mark extends Control:
    var kind := "started"
    var color := Color.WHITE

    func _init() -> void:
        custom_minimum_size = Vector2(20, 20)
        size_flags_vertical = Control.SIZE_SHRINK_CENTER
        mouse_filter = Control.MOUSE_FILTER_IGNORE

    func _draw() -> void:
        var c := Vector2(10, 10)
        var tick := PackedVector2Array([c + Vector2(-4.2, 0.4), c + Vector2(-1.2, 3.6), c + Vector2(4.6, -3.6)])
        match kind:
            "started":
                draw_arc(c, 7.0, 0.0, TAU, 28, color, 1.6, true)
                draw_circle(c, 2.6, color)
            "objective_done":
                draw_polyline(tick, color, 2.0, true)
            "complete":
                draw_circle(c, 9.0, Color(color, 0.3))
                draw_arc(c, 9.0, 0.0, TAU, 28, color, 1.4, true)
                draw_polyline(tick, color, 2.0, true)
            "notice":
                draw_arc(c, 8.0, 0.0, TAU, 28, color, 1.4, true)
                draw_line(c + Vector2(0, -1.0), c + Vector2(0, 4.2), color, 2.0, true)
                draw_circle(c + Vector2(0, -4.0), 1.2, color)
            "failed":
                draw_circle(c, 9.0, Color(color, 0.3))
                draw_arc(c, 9.0, 0.0, TAU, 28, color, 1.4, true)
                draw_line(c + Vector2(-3.6, -3.6), c + Vector2(3.6, 3.6), color, 2.0, true)
                draw_line(c + Vector2(-3.6, 3.6), c + Vector2(3.6, -3.6), color, 2.0, true)
