class_name GeogenSceneMenu
extends CanvasLayer
## The in-game scene switcher (F6): the catalogue's scenes grouped showcase /
## test with their descriptions. Picking one emits ``picked``; main.gd does the
## switch. Scenes that haven't been exported are listed but can't be picked.

signal picked(scene_name: String)
signal closed

const GROUP_TITLES := {"showcase": "Showcase", "test": "Test scenes"}

var _list: ItemList
var _hint: Label


func _init() -> void:
    name = "SceneMenu"
    layer = 10
    visible = false
    var dim := ColorRect.new()
    dim.color = Color(0, 0, 0, 0.45)
    dim.set_anchors_preset(Control.PRESET_FULL_RECT)
    add_child(dim)
    var panel := PanelContainer.new()
    panel.set_anchors_preset(Control.PRESET_CENTER)
    panel.custom_minimum_size = Vector2(900, 480)
    panel.position = -panel.custom_minimum_size / 2.0
    dim.add_child(panel)
    var box := VBoxContainer.new()
    box.add_theme_constant_override("separation", 8)
    panel.add_child(box)
    var title := Label.new()
    title.text = "Scenes"
    title.add_theme_font_size_override("font_size", 20)
    box.add_child(title)
    _list = ItemList.new()
    _list.size_flags_vertical = Control.SIZE_EXPAND_FILL
    _list.item_activated.connect(_activate)
    box.add_child(_list)
    _hint = Label.new()
    _hint.text = "Enter / double-click: go    F6 / Esc: close    F7 / F8: previous / next scene"
    _hint.add_theme_color_override("font_color", Color(0.75, 0.75, 0.75))
    box.add_child(_hint)


## Fill the list from ``scenes`` (WorldLoader.scene_list()) and show it, with ``current`` selected.
func open(scenes: Array[Dictionary], current: String) -> void:
    _list.clear()
    for group in ["showcase", "test"]:
        var members := scenes.filter(func(s): return s["group"] == group)
        if members.is_empty():
            continue
        var header := _list.add_item(GROUP_TITLES[group])
        _list.set_item_selectable(header, false)
        _list.set_item_disabled(header, true)
        _list.set_item_custom_fg_color(header, Color(1.0, 0.8, 0.45))
        for s in members:
            var text := "    %s" % s["name"]
            if s["description"] != "":
                text += "  -  %s" % s["description"]
            if not s["exported"]:
                text += "  (not exported)"
            var i := _list.add_item(text)
            _list.set_item_metadata(i, s["name"])
            _list.set_item_disabled(i, not s["exported"])
            if s["name"] == current:
                _list.set_item_custom_fg_color(i, Color(0.55, 0.9, 1.0))
                _list.select(i)
                _list.ensure_current_is_visible()
    visible = true
    _list.grab_focus()


func close() -> void:
    if visible:
        visible = false
        closed.emit()


func _activate(index: int) -> void:
    var scene = _list.get_item_metadata(index)
    if scene == null or _list.is_item_disabled(index):
        return
    close()
    picked.emit(scene)


func _unhandled_input(event: InputEvent) -> void:
    if not visible:
        return
    if event is InputEventKey and event.pressed and not event.echo \
            and event.physical_keycode in [KEY_ESCAPE, KEY_F6]:
        close()
        get_viewport().set_input_as_handled()
