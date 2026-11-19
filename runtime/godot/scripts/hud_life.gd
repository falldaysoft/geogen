class_name GeogenHudLife
extends PanelContainer
## HUD panel: how someone's life is going (geogen.111). Greeting a person with a `life:` shows
## their name, what has happened since you last asked and what they are hoping for now. A new
## chapter while you are about shows its line by itself. It holds no story of its own: everything
## is read from the person (`GeogenLife.report`), and only how much of it you have seen is kept here.

const HOLD := 14.0              ## s the card stays up
const WIDTH := 400.0
const COLORS := {"back": Color(0.07, 0.07, 0.06, 0.82), "edge": Color(0.86, 0.72, 0.42), "name": Color(1.0, 0.88, 0.62),
    "text": Color(0.93, 0.92, 0.88), "dim": Color(0.68, 0.68, 0.64)}

var _hud: GeogenHud
var _rows := VBoxContainer.new()
var _left := 0.0
var _seen := {}                 # stable ID -> how many of their story's lines have been shown
var shown := {}                 # what is up now: {name, lines, hopes} (the API and tests read it)


func _init() -> void:
    name = "life"
    visible = false
    mouse_filter = Control.MOUSE_FILTER_IGNORE
    custom_minimum_size.x = WIDTH
    var style := StyleBoxFlat.new()
    style.bg_color = COLORS["back"]
    style.set_corner_radius_all(6)
    style.border_width_left = 3
    style.border_color = COLORS["edge"]
    style.content_margin_left = 14
    style.content_margin_right = 14
    style.content_margin_top = 10
    style.content_margin_bottom = 11
    add_theme_stylebox_override("panel", style)
    _rows.add_theme_constant_override("separation", 5)
    add_child(_rows)


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud
    hud.world.events.listen(_on_event, ["npc"])


func _on_event(event: Dictionary) -> void:
    var data: Dictionary = event.get("data", {})
    var what := str(data.get("event", event.get("name", "")))
    if what != "greet" and what != "chapter":
        return
    for npc in _hud.world.npcs:
        if is_instance_valid(npc) and String(npc.name) == str(data.get("npc", "")) and GeogenLife.has(npc):
            show_person(npc)


## Put up the card for ``npc``: what is new in their story since it was last shown.
func show_person(npc: GeogenNpc) -> void:
    var life := GeogenLife.report(npc)
    var uid := WorldLoader.npc_uid(npc)
    var story: Array = life["story"]
    var lines := story.slice(int(_seen.get(uid, 0)))
    if lines.is_empty() and not story.is_empty():
        lines = story.slice(-1)
    _seen[uid] = story.size()
    shown = {"name": npc.person_name(), "lines": lines.map(func(l): return str(l["text"])), "hopes": str(life["hopes"])}
    for child in _rows.get_children():
        _rows.remove_child(child)
        child.queue_free()
    _line(shown["name"], COLORS["name"], 18)
    for line in shown["lines"]:
        _line(line, COLORS["text"], 15)
    if shown["hopes"] != "":
        _line("Hoping to: %s" % shown["hopes"], COLORS["dim"], 13)
    _left = HOLD
    visible = true


func _line(text: String, color: Color, size_px: int) -> void:
    var label := Label.new()
    label.text = text
    label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    label.custom_minimum_size.x = WIDTH - 32.0
    label.mouse_filter = Control.MOUSE_FILTER_IGNORE
    label.add_theme_font_size_override("font_size", size_px)
    label.add_theme_color_override("font_color", color)
    _rows.add_child(label)


func _process(delta: float) -> void:
    if not visible:
        return
    _left -= delta
    if _left <= 0.0:
        visible = false
        shown = {}


func hud_refresh() -> void:
    pass
