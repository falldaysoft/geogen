class_name GeogenHudDirect
extends PanelContainer
## HUD panel (bottom left): telling people what to do (the dollhouse, geogen.101.8).
##
## `TOGGLE_KEY` turns directing on and off. While it is on, main.gd looks down from the
## overview camera with ceilings hidden and frees the mouse; one person is selected (a ring at
## their feet), and the panel shows who they are, how their needs stand, what they are doing
## and what they have been told to do. Clicking a thing (or another person) lists what the
## selected person could do with it (`GeogenNpc.choices`): each line is an affordance of the
## thing, with the chain it means ("Eat: Get raw food > Make meal > Eat") or why it can't be
## done now. Choosing one queues it (`GeogenNpc.direct`); orders are done in turn, ahead of
## anything the person would do of their own accord, and with none left they see to themselves
## again. Clicking an order in the queue takes it back.
##
## The menu is worked out when a thing is picked, never on a refresh: asking an NPC for its
## options draws from its random numbers, and a panel that is merely open must not change what
## people decide.

## Directing was turned on or off: main.gd changes the camera, the mouse and the player.
signal mode_changed(on: bool)
## Someone else was selected (null: nobody): main.gd brings the view over to them.
signal selected_changed(npc: GeogenNpc)

const TOGGLE_KEY := KEY_G
const NEXT_KEY := KEY_TAB
const UNDO_KEY := KEY_BACKSPACE
const PICK_PIXELS := 70.0       # a click picks the nearest thing within this many pixels
const PERSON_PIXELS := 45.0     # ...and a person this near the click before any thing
const REFRESH := 0.25           # s between redraws while directing (needs, what they are doing)
const WIDTH := 330.0
const COLORS := {"back": Color(0.05, 0.06, 0.08, 0.72), "name": Color(1.0, 0.82, 0.4), "text": Color(0.9, 0.91, 0.93),
    "dim": Color(0.6, 0.63, 0.68), "bar": Color(0.45, 0.75, 0.5), "low": Color(0.9, 0.45, 0.35),
    "trough": Color(0.16, 0.17, 0.2, 0.9), "ring": Color(1.0, 0.82, 0.3)}

var active := false
var selected: GeogenNpc = null
var thing := ""                 # the picked thing's stable ID ("" for none)
var menu: Array = []            # its choices for the selected person (GeogenNpc.choices)
var message := ""               # the last thing that couldn't be done

var _hud: GeogenHud
var _rows := VBoxContainer.new()
var _since := 0.0
var _built := ""                # what the rows were built for (_signature): rebuilt only when it changes
var _doing_label: Label = null
var _bars := {}                 # need -> ProgressBar
var _lines: Array = []          # the menu's buttons, in order
var _ring: MeshInstance3D = null


func _init() -> void:
    name = "direct"
    visible = false
    mouse_filter = Control.MOUSE_FILTER_STOP        # a click on the panel isn't a click on the world behind it
    custom_minimum_size = Vector2(WIDTH, 0)
    var box := StyleBoxFlat.new()
    box.bg_color = COLORS["back"]
    box.set_corner_radius_all(6)
    box.set_content_margin_all(10)
    add_theme_stylebox_override("panel", box)
    _rows.add_theme_constant_override("separation", 3)
    add_child(_rows)


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud
    hud.world.world_loaded.connect(func(_aabb: AABB): _forget())


func _forget() -> void:
    selected = null
    thing = ""
    menu = []
    message = ""


func _process(delta: float) -> void:
    if not active:
        return
    _since += delta
    if _since < REFRESH:
        return
    _since = 0.0
    # The buttons are rebuilt only when what they show changes: one rebuilt between a press and
    # its release would lose the click. Otherwise only the live parts are brought up to date.
    hud_refresh()


## What the rows are built from, apart from the live parts: who, their orders, the menu, the message.
func _signature() -> String:
    if not _here(selected):
        return "nobody"
    return JSON.stringify([selected.get_instance_id(), selected._orders, thing, menu, message, selected.needs.keys()])


func _show_need(bar: ProgressBar, level: float) -> void:
    var low := level < 0.3
    if bar.value == level and bar.has_meta("low") and bar.get_meta("low") == low:
        return
    bar.value = level
    bar.set_meta("low", low)
    bar.add_theme_stylebox_override("fill", _flat(COLORS["low"] if low else COLORS["bar"]))


## Everyone who is here to be told anything.
func people() -> Array:
    return _hud.world.npcs.filter(func(n): return is_instance_valid(n) and not n.away and not n.absent)


func set_active(on: bool) -> void:
    if on == active:
        return
    active = on
    visible = on
    thing = ""
    menu = []
    message = ""
    if on and not _here(selected):
        var all := people()
        select(all[0] if not all.is_empty() else null)
    _place_ring()
    mode_changed.emit(on)
    if on:
        selected_changed.emit(selected)
    _hud.queue_refresh()


func _here(npc) -> bool:
    return is_instance_valid(npc) and not npc.away and not npc.absent


func select(npc: GeogenNpc) -> void:
    if npc != selected or not active:
        selected_changed.emit.call_deferred(npc)
    selected = npc
    thing = ""
    menu = []
    message = ""
    _place_ring()
    _hud.queue_refresh()


## The next person (or the one before), in the order the world lists them.
func select_next(step := 1) -> void:
    var all := people()
    if all.is_empty():
        select(null)
        return
    select(all[posmod(all.find(selected) + step, all.size())])


## Pick a thing (stable ID; another person's: what they offer) for the selected person: its
## choices become the menu. False with nobody selected or nothing to do with it.
func pick(uid: String) -> bool:
    message = ""
    thing = ""
    menu = []
    if not _here(selected) or uid == "" or uid == WorldLoader.npc_uid(selected):
        _hud.queue_refresh()
        return false
    menu = selected.choices(uid)
    thing = uid if not menu.is_empty() else ""
    if menu.is_empty():
        message = "Nothing for %s to do with that" % _name(selected)
    _hud.queue_refresh()
    return not menu.is_empty()


## A click at a viewport position, seen through ``camera``: the person or thing nearest it on
## screen (what a ray would hit is no guide: seated people have no collider, and hidden
## ceilings still have theirs). ``choose``: a left click (pick it for the selected person; a
## person with nobody selected, or with nothing to offer, is selected instead); otherwise select
## the person there. What was picked or selected (a stable ID), or "".
func click(at: Vector2, camera: Camera3D, choose := true) -> String:
    var person: GeogenNpc = null
    var person_gap := PERSON_PIXELS
    for npc in people():
        var p: Vector3 = npc.global_position + Vector3.UP * 0.9
        if camera.is_position_behind(p):
            continue
        var gap := camera.unproject_position(p).distance_to(at)
        if gap < person_gap:
            person_gap = gap
            person = npc
    if person != null:
        if choose and _here(selected) and person != selected and pick(WorldLoader.npc_uid(person)):
            return thing
        select(person)
        return WorldLoader.npc_uid(person)
    if not choose:
        return ""
    var best := ""
    var best_gap := PICK_PIXELS
    for a in _hud.world.affordances:
        if not is_instance_valid(a.get("node")) or camera.is_position_behind(a["position"]):
            continue
        var gap := camera.unproject_position(a["position"]).distance_to(at)
        if gap < best_gap:
            best_gap = gap
            best = WorldLoader.uid_of(a["node"])
    if best == "":
        thing = ""
        menu = []
        message = ""
        _hud.queue_refresh()
        return ""
    pick(best)
    return best


## Tell the selected person to do line ``index`` of the menu. "" or why not.
func choose(index: int) -> String:
    if not _here(selected) or index < 0 or index >= menu.size():
        return "no such choice"
    var entry: Dictionary = menu[index]
    var why := str(entry["why"]) if not entry["available"] else selected.direct(str(entry["option"]))
    message = "" if why == "" else "%s: %s" % [entry["label"], why]
    if why == "":
        thing = ""
        menu = []
    _hud.queue_refresh()
    return why


## Take back order ``index`` of the selected person's queue (-1: the last one told).
func cancel(index := -1) -> int:
    if not _here(selected) or selected._orders.is_empty():
        return 0
    var gone: int = selected.cancel_orders(selected._orders.size() - 1 if index < 0 else index)
    _hud.queue_refresh()
    return gone


func hud_key(keycode: int) -> bool:
    if keycode == TOGGLE_KEY:
        set_active(not active)
        return true
    if not active:
        return false
    if keycode == NEXT_KEY:
        select_next()
    elif keycode == UNDO_KEY:
        cancel()
    elif keycode == KEY_ESCAPE:
        if menu.is_empty():
            set_active(false)
        else:
            thing = ""
            menu = []
            _hud.queue_refresh()
    elif keycode >= KEY_1 and keycode <= KEY_9:
        choose(keycode - KEY_1)         # (the digits are the menu's while directing, not the hotbar's)
    else:
        return false
    return true


func hud_keys() -> Dictionary:
    return {"title": "Directing people", "keys": [
        ["G", "direct people: look down into the rooms, with the mouse free"],
        ["Click", "a thing or a person: what the selected person can do with it"],
        ["1-9", "choose from that list (or click the line)"],
        ["Right click / Tab", "select the person there / the next person"],
        ["Backspace", "take back the last order (or click it)"],
        ["W A S D, wheel", "move the view, zoom"],
        ["Esc", "close the list; leave"]]}


func hud_refresh() -> void:
    if active and _signature() == _built and _doing_label != null and _here(selected):
        _doing_label.text = _doing(selected)        # nothing but the live parts changed
        for need in _bars:
            _show_need(_bars[need], float(selected.needs.get(need, 0.0)))
        return
    for child in _rows.get_children():
        _rows.remove_child(child)
        child.queue_free()
    _bars = {}
    _lines = []
    _doing_label = null
    _built = _signature()
    if not active:
        return
    if not _here(selected):
        selected = null
        _place_ring()
        _rows.add_child(_label("Nobody here to direct", 15, COLORS["dim"]))
        return
    _rows.add_child(_label(_name(selected), 18, COLORS["name"]))
    _doing_label = _label(_doing(selected), 13, COLORS["dim"])
    _rows.add_child(_doing_label)
    var grid := GridContainer.new()
    grid.columns = 2
    grid.add_theme_constant_override("h_separation", 10)
    for need in selected.needs:
        grid.add_child(_label(str(need).capitalize(), 13, COLORS["text"]))
        var bar := ProgressBar.new()
        bar.show_percentage = false
        bar.max_value = 1.0
        bar.step = 0.0
        bar.custom_minimum_size = Vector2(170, 9)
        bar.size_flags_vertical = Control.SIZE_SHRINK_CENTER
        bar.mouse_filter = Control.MOUSE_FILTER_IGNORE
        bar.add_theme_stylebox_override("background", _flat(COLORS["trough"]))
        _show_need(bar, float(selected.needs[need]))
        _bars[need] = bar
        grid.add_child(bar)
    _rows.add_child(grid)
    if not selected._orders.is_empty():
        _rows.add_child(_label("Told to", 13, COLORS["dim"]))
        for i in selected._orders.size():
            var order := _button("%s%s   x" % ["> " if i == 0 else "   ", selected.option_label(str(selected._orders[i]))])
            order.pressed.connect(cancel.bind(i))
            _rows.add_child(order)
    if not menu.is_empty():
        _rows.add_child(_label(_thing_name(), 15, COLORS["name"]))
        for i in menu.size():
            var entry: Dictionary = menu[i]
            var text := "%d  %s" % [i + 1, entry["label"]]
            if not entry["available"]:
                text += "   (%s)" % entry["why"]
            elif (entry["plan"] as Array).size() > 1:
                text += ":  " + " > ".join((entry["plan"] as Array).map(func(id): return selected.option_label(str(id))))
            var line := _button(text)
            line.disabled = not entry["available"]
            line.pressed.connect(choose.bind(i))
            _lines.append(line)
            _rows.add_child(line)
    elif message != "":
        _rows.add_child(_label(message, 13, COLORS["low"]))
    else:
        _rows.add_child(_label("Click a thing for %s to use.  Tab: next person" % _name(selected).split(" ")[0], 12, COLORS["dim"]))


## What is on screen: {active, selected (NPC node name, "" for nobody), name, doing, needs,
## orders: [{option, label}], thing, menu: [{option, label, available, why, plan, cost}], message,
## lines: the menu lines' rectangles on screen ([x, y, w, h] each, once laid out)}.
func hud_report() -> Dictionary:
    var who := selected if _here(selected) else null
    return {"active": active, "visible": visible, "selected": String(who.name) if who != null else "",
        "name": _name(who) if who != null else "", "doing": _doing(who) if who != null else "",
        "needs": who._rounded_needs() if who != null else {},
        "orders": who._orders.map(func(id): return {"option": id, "label": who.option_label(str(id))}) if who != null else [],
        "thing": thing, "menu": menu.duplicate(true), "message": message,
        "lines": _lines.filter(func(b): return is_instance_valid(b)).map(func(b): return [b.global_position.x,
            b.global_position.y, b.size.x, b.size.y])}


func _name(npc: GeogenNpc) -> String:
    return npc.person_name() if npc.person_name() != "" else String(npc.name).trim_suffix("_npc").capitalize()


## What they are at, in a few words: the option in hand by its label, and whose idea it was.
func _doing(npc: GeogenNpc) -> String:
    if npc._option.is_empty():
        return "Deciding what to do"
    return "%s%s" % [npc.option_label(str(npc._option["id"])), "  (told to)" if npc._ordered else ""]


func _thing_name() -> String:
    for npc in people():
        if WorldLoader.npc_uid(npc) == thing:
            return _name(npc)
    return thing.get_slice(":", thing.get_slice_count(":") - 1).get_file().capitalize()


## A ring on the floor under the selected person, while directing.
func _place_ring() -> void:
    if _ring != null and is_instance_valid(_ring):
        _ring.queue_free()
    _ring = null
    if not active or not _here(selected):
        return
    var torus := TorusMesh.new()
    torus.inner_radius = 0.42
    torus.outer_radius = 0.52
    var material := StandardMaterial3D.new()
    material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
    material.albedo_color = COLORS["ring"]
    torus.material = material
    _ring = MeshInstance3D.new()
    _ring.name = "direct_ring"
    _ring.mesh = torus
    _ring.scale = Vector3(1, 0.15, 1)
    _ring.position = Vector3(0, 0.03, 0)
    _ring.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
    selected.add_child(_ring)


static func _flat(color: Color) -> StyleBoxFlat:
    var box := StyleBoxFlat.new()
    box.bg_color = color
    box.set_corner_radius_all(3)
    return box


static func _label(text: String, font_size: int, color: Color) -> Label:
    var label := Label.new()
    label.text = text
    label.mouse_filter = Control.MOUSE_FILTER_IGNORE
    label.add_theme_font_size_override("font_size", font_size)
    label.add_theme_color_override("font_color", color)
    label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    label.custom_minimum_size = Vector2(WIDTH - 20, 0)
    return label


static func _button(text: String) -> Button:
    var button := Button.new()
    button.text = text
    button.alignment = HORIZONTAL_ALIGNMENT_LEFT
    button.focus_mode = Control.FOCUS_NONE          # keys stay the game's
    button.add_theme_font_size_override("font_size", 13)
    button.clip_text = true
    button.custom_minimum_size = Vector2(WIDTH - 20, 0)
    return button
