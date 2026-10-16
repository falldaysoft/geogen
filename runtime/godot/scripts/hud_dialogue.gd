class_name GeogenHudDialogue
extends Control
## HUD panel (slot "screen"): the conversation the HUD's actor has open (geogen.17,
## "Dialogue" in docs/rules.md). A dialogue is open while its actor state
## (`actor:<pkg>.dialogue_<id>`) names a passage: the panel shows that passage's text,
## its `{condition}` parts as their values now, and the choices whose `when` holds, numbered.
##
## Choosing raises the anchor's signal `dialogue {passage, choice}` on the event bus as the
## actor; the compiled rules do the rest (effects, the next passage or closing). A passage
## with no choices offers Continue, and Esc closes: both are choice "". While open it is the
## HUD's modal panel: it alone gets keys (1-9, Up/Down, Enter or Space, Esc), the mouse is
## free to click a choice, and main.gd parks the player.

const TEXT_SIZE := 20
const CHOICE_SIZE := 17
const COLORS := {"text": Color(0.95, 0.95, 0.95), "choice": Color(0.95, 0.8, 0.5),
    "selected": Color(1.0, 1.0, 1.0), "title": Color(0.78, 0.8, 0.84)}

var _hud: GeogenHud
var _back: PanelContainer
var _title: Label
var _text: RichTextLabel
var _list: VBoxContainer
var _open := {}                 # {dialogue, anchor, passage, title, text, choices: [{id, text}]}, or {}
var _selected := 0


func _init() -> void:
    name = "dialogue"
    mouse_filter = Control.MOUSE_FILTER_IGNORE
    set_anchors_preset(Control.PRESET_FULL_RECT)
    _back = PanelContainer.new()
    _back.anchor_left = 0.2
    _back.anchor_right = 0.8
    _back.anchor_top = 0.7               # below the face of whoever is talking
    _back.anchor_bottom = 0.7
    _back.grow_vertical = Control.GROW_DIRECTION_END
    var style := StyleBoxFlat.new()
    style.bg_color = Color(0.04, 0.05, 0.07, 0.85)
    style.set_corner_radius_all(8)
    style.border_width_top = 3
    style.border_color = COLORS["choice"]
    for side in ["left", "right", "top", "bottom"]:
        style.set("content_margin_" + side, 16)
    _back.add_theme_stylebox_override("panel", style)
    add_child(_back)
    var column := VBoxContainer.new()
    column.add_theme_constant_override("separation", 8)
    _back.add_child(column)
    _title = Label.new()
    _title.add_theme_font_size_override("font_size", 13)
    _title.add_theme_color_override("font_color", COLORS["title"])
    column.add_child(_title)
    _text = RichTextLabel.new()
    _text.fit_content = true
    _text.scroll_active = false
    _text.bbcode_enabled = false
    _text.add_theme_font_size_override("normal_font_size", TEXT_SIZE)
    _text.add_theme_color_override("default_color", COLORS["text"])
    column.add_child(_text)
    _list = VBoxContainer.new()
    column.add_child(_list)
    _back.visible = false


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud
    hud.world.events.listen(func(_event: Dictionary): hud.queue_refresh(), ["changed"])


## The open dialogue for the HUD's actor, read from the rules program.
func _read() -> Dictionary:
    var rules: GeogenRules = _hud.world.rules
    for id in rules.dialogues:
        var entry: Dictionary = rules.dialogues[id]
        var passage = rules.get_value(str(entry["state"]), _hud.actor)
        if passage == null or str(passage) == "" or not entry["passages"].has(str(passage)):
            continue
        var shown: Dictionary = entry["passages"][str(passage)]
        var text := ""
        for part in shown.get("text", []):
            text += GeogenHudStats.shown(rules.value_of(part["value"], _hud.actor)) if part is Dictionary else str(part)
        var choices := []
        for choice in shown.get("choices", []):
            if not choice.has("when") or GeogenConditions.truthy(rules.value_of(choice["when"], _hud.actor)):
                choices.append({"id": str(choice["id"]), "text": str(choice["text"])})
        if choices.is_empty():
            choices.append({"id": "", "text": "Continue"})
        return {"dialogue": id, "anchor": str(entry["anchor"]), "passage": str(passage),
            "title": str(entry.get("title", "")), "text": text, "choices": choices}
    return {}


func hud_refresh() -> void:
    var was_open := not _open.is_empty()
    var before := [_open.get("dialogue"), _open.get("passage")]
    _open = _read()
    if [_open.get("dialogue"), _open.get("passage")] != before:
        _selected = 0
    _back.visible = not _open.is_empty()
    if _open.is_empty():
        if was_open:
            _hud.set_modal(self, false)
        return
    _hud.set_modal(self, true)
    _title.text = _open["title"]
    _title.visible = _open["title"] != ""
    _text.text = _open["text"]
    for child in _list.get_children():
        _list.remove_child(child)
        child.queue_free()
    for i in _open["choices"].size():
        var choice: Dictionary = _open["choices"][i]
        var button := Button.new()
        button.text = "%d.  %s" % [i + 1, choice["text"]]
        button.alignment = HORIZONTAL_ALIGNMENT_LEFT
        button.flat = true
        button.add_theme_font_size_override("font_size", CHOICE_SIZE)
        button.add_theme_color_override("font_color", COLORS["selected"] if i == _selected else COLORS["choice"])
        button.pressed.connect(choose.bind(i))
        _list.add_child(button)


## Take choice ``index`` of what is shown (its number less one): raise the anchor's signal.
func choose(index: int) -> bool:
    if _open.is_empty() or index < 0 or index >= _open["choices"].size():
        return false
    _hud.world.events.post("signal", _open["anchor"], "dialogue", _hud.actor,
        {"passage": _open["passage"], "choice": _open["choices"][index]["id"]})
    _hud.queue_refresh()
    return true


## Close it without choosing (choice "").
func close() -> void:
    if not _open.is_empty():
        _hud.world.events.post("signal", _open["anchor"], "dialogue", _hud.actor,
            {"passage": _open["passage"], "choice": ""})
        _hud.queue_refresh()


func hud_key(keycode: int) -> bool:
    if _open.is_empty():
        return false
    if keycode >= KEY_1 and keycode <= KEY_9:
        choose(keycode - KEY_1)
    else:
        match keycode:
            KEY_ESCAPE:
                close()
            KEY_UP, KEY_W:
                _selected = maxi(_selected - 1, 0)
                _hud.queue_refresh()
            KEY_DOWN, KEY_S:
                _selected = mini(_selected + 1, _open["choices"].size() - 1)
                _hud.queue_refresh()
            KEY_ENTER, KEY_KP_ENTER, KEY_SPACE:
                choose(_selected)
    return true             # modal: nothing else gets a key while it is open


## What is on screen: {open, dialogue?, anchor?, passage?, title?, text?, choices?: [{id, text}], selected?}.
func hud_report() -> Dictionary:
    var out := {"open": not _open.is_empty()}
    out.merge(_open.duplicate(true))
    if not _open.is_empty():
        out["selected"] = _selected
    return out
