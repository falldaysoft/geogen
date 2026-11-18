class_name GeogenHudMotionGallery
extends Control
## HUD panel: the motion gallery, a list of every motion file (motions.json's `catalogue`,
## written from assets/motions/ with every export) that plays the chosen one on the people in
## the scene, all of them at once or one at a time.
##
## `OPEN_KEY` (M) opens it over any scene with people a motion can run on (bodies with
## extras.geogen.rig); the motion_gallery scene opens it by itself. While it is open it is the
## HUD's modal panel (`GeogenHud.set_modal`): main.gd parks the player and frees the mouse, so
## the list can be clicked. Up / Down (or a click) choose a motion, Enter or Space plays it,
## Left / Right choose who performs it (everyone, or one person), L loops it.
##
## Motions come from the catalogue, so a new file in assets/motions/ is listed after the next
## export with no change here. Ones that take things (drink, inspect, take, carry, set_down: a
## motion's `props` say which asset each role is) play at a table in front of each person: the
## scene has `<person>_table` with `<person>_<asset>` on it (motion_gallery: a mug and a plate),
## shown while such a motion plays and hidden otherwise, and the things are put back where they
## stood before each play. A motion's `demo` is played in its place when it has one (take,
## carry and set_down: take the plate, carry it, put it back where it stood; `<role>@start`).
## A prop with `behind` (sit's seat) is `<person>_<asset>` behind the person, shown the same way;
## people a motion moved (sit) are put back where they stood before the next play. Ones that
## need things the stage doesn't hold (press: a TV) are listed greyed; `python -m
## geogen.motion_review --motion NAME` shows them on the reference rig.
##
## Presentation state only: open, the chosen motion and performer, looping, the demo's clock.

const OPEN_KEY := KEY_M
const GALLERY_SCENE := "motion_gallery"
const TITLE_SIZE := 20
const HEAD_SIZE := 12
const TEXT_SIZE := 14
const LOOP_GAP := 0.6               # s between the end of a looped motion and its next start
const COLORS := {
    "title": Color(1.0, 0.86, 0.55),
    "text": Color(0.96, 0.96, 0.96),
    "note": Color(0.72, 0.74, 0.78),
    "dim": Color(0.5, 0.52, 0.56),
    "head": Color(0.55, 0.78, 1.0),
    "chosen": Color(1.0, 0.86, 0.55),
}
const CATEGORY_TITLES := {"idle": "Idles", "emote": "Emotes", "dance": "Dances", "exercise": "Exercise",
    "action": "With things", "reaction": "Reactions"}

var _hud: GeogenHud
var _open := false
var _auto_opened := ""             # the generated dir the gallery scene last opened itself for
var _catalogue: Array = []         # the catalogue's entries, in its order
var _selected := ""                # a motion's name
var _performer := "all"            # "all" or a performer's node name
var _performers: Array[Node3D] = []
var _playing := ""
var _loop := false
var _idle_for := 0.0
var _demo: Array = []              # [{at, body, entry, starts}] still to start, by time
var _demo_clock := 0.0
var _homes := {}                   # a staged thing that isn't a physics prop -> where it started
var _stands := {}                  # a performer -> where it stood when the gallery first played
var _frame: PanelContainer
var _who: HBoxContainer
var _loop_button: CheckButton
var _list_scroll: ScrollContainer
var _list: VBoxContainer
var _about: Label
var _hint: Label
var _rows := {}                    # motion name -> its Button


func _init() -> void:
    name = "motion_gallery"
    visible = false
    z_index = 1
    mouse_filter = Control.MOUSE_FILTER_IGNORE      # the scene shows through beside the list
    _frame = PanelContainer.new()
    var style := StyleBoxFlat.new()
    style.bg_color = Color(0.06, 0.07, 0.09, 0.9)
    style.set_corner_radius_all(8)
    style.border_width_top = 3
    style.border_color = COLORS["title"]
    style.content_margin_left = 14
    style.content_margin_right = 14
    style.content_margin_top = 10
    style.content_margin_bottom = 10
    _frame.add_theme_stylebox_override("panel", style)
    _frame.mouse_filter = Control.MOUSE_FILTER_STOP
    add_child(_frame)
    var column := VBoxContainer.new()
    column.add_theme_constant_override("separation", 6)
    _frame.add_child(column)
    column.add_child(_label("Motions", TITLE_SIZE, COLORS["title"]))
    column.add_child(_label("Who performs it", HEAD_SIZE, COLORS["head"]))
    _who = HBoxContainer.new()
    _who.add_theme_constant_override("separation", 4)
    var who_scroll := ScrollContainer.new()
    who_scroll.vertical_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
    who_scroll.custom_minimum_size = Vector2(0, 34)
    who_scroll.add_child(_who)
    column.add_child(who_scroll)
    _loop_button = CheckButton.new()
    _loop_button.text = "Loop"
    _loop_button.focus_mode = Control.FOCUS_NONE
    _loop_button.add_theme_font_size_override("font_size", TEXT_SIZE)
    _loop_button.toggled.connect(func(on: bool): _loop = on)
    column.add_child(_loop_button)
    _list_scroll = ScrollContainer.new()
    _list_scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
    _list_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
    _list = VBoxContainer.new()
    _list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    _list.add_theme_constant_override("separation", 1)
    _list_scroll.add_child(_list)
    column.add_child(_list_scroll)
    _about = _label("", TEXT_SIZE, COLORS["text"])
    _about.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    column.add_child(_about)
    _hint = _label("Up/Down choose  Enter play  Left/Right who  L loop  M close", HEAD_SIZE, COLORS["note"])
    _hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    column.add_child(_hint)


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud


func is_open() -> bool:
    return _open


## Open or close the gallery. Opening reads the catalogue and the scene's people afresh.
func set_open(open: bool) -> void:
    if open == _open:
        return
    _open = open
    visible = open
    if open:
        _read()
    _hud.set_modal(self, open)
    _hud.refresh()


## For the keys panel.
func hud_keys() -> Dictionary:
    return {"title": "Motion gallery", "keys": [
        [OS.get_keycode_string(OPEN_KEY), "open or close the gallery of motions (people play them)"],
        ["Up / Down", "choose a motion"],
        ["Enter / Space", "play it"],
        ["Left / Right", "who performs it: everyone or one person"],
        ["L", "loop it"]]}


func hud_key(keycode: int) -> bool:
    if not _open:
        if keycode != OPEN_KEY or _find_performers().is_empty():
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
        KEY_LEFT, KEY_A:
            cycle_performer(-1)
        KEY_RIGHT, KEY_D:
            cycle_performer(1)
        KEY_L:
            set_loop(not _loop)
        KEY_ENTER, KEY_KP_ENTER, KEY_SPACE:
            play(_selected)
    return true


# -- choosing and playing ------------------------------------------------------------------------

## Choose the next (or previous) motion in the list; stops at the ends.
func step(by: int) -> void:
    if _catalogue.is_empty():
        return
    var names := _catalogue.map(func(e): return e["name"])
    var at := names.find(_selected)
    select(names[clampi(at + by if at >= 0 else 0, 0, names.size() - 1)])


## Choose a motion by name; false when the catalogue has none by that name.
func select(motion: String) -> bool:
    if _entry(motion).is_empty():
        return false
    _selected = motion
    _hud.queue_refresh()
    return true


## Who performs: "all" or a performer's name; false when nobody is called that.
func set_performer(who: String) -> bool:
    if who != "all" and _performers.all(func(p): return String(p.name) != who):
        return false
    _performer = who
    _hud.queue_refresh()
    return true


func cycle_performer(by: int) -> void:
    var names: Array = ["all"]
    names.append_array(_performers.map(func(p): return String(p.name)))
    set_performer(names[posmod(names.find(_performer) + by, names.size())])


func set_loop(on: bool) -> void:
    _loop = on
    _loop_button.set_pressed_no_signal(on)


## Play `motion` on whoever performs (everyone stops what they are doing first); "" or why not.
## One that takes things plays at each person's table, with the things put back first.
func play(motion: String) -> String:
    var entry := _entry(motion)
    if entry.is_empty():
        return "no motion '%s' in the catalogue" % motion
    var why_not := _why_not(entry)
    if why_not != "":
        return why_not
    var motions := GeogenMotionPlayer.motions(_hud.world.generated_dir)
    if not motions.has(motion):
        return "no motion '%s' in motions.json" % motion
    _selected = motion
    var props: Dictionary = entry.get("props", {})
    for body in _performers:
        var player := GeogenMotionPlayer.on(body)
        if player != null and (_performer == "all" or String(body.name) == _performer):
            # Back where it stood, standing (sit leaves the body at the seat's approach point).
            if not _stands.has(body):
                _stands[body] = body.global_transform
            if player.rig.posture != "stand" or not body.global_transform.is_equal_approx(_stands[body]):
                player.reset(_stands[body])
            else:
                player.break_(null)
    _stage(entry)
    var script: Array = entry.get("demo", [])
    if script.is_empty():
        var only := {"at": 0.0, "start": motion}
        for role in props:
            only[role] = role
        script = [only]
    _demo = []
    _demo_clock = 0.0
    for body in _chosen():
        # Where each staged thing stands now (put back): a demo's `<role>@start`.
        var starts := {}
        for role in props:
            var node := _staged(body, props[role])
            if node != null:
                starts[role] = GeogenMotionRig.Thing.fixed("%s@start" % node.name, node.global_transform)
        for e in script:
            _demo.append({"at": float(e.get("at", 0.0)), "body": body, "entry": e, "starts": starts, "props": props})
    _demo.sort_custom(func(a, b): return a["at"] < b["at"])
    var why := _run_demo()
    if why != "":
        _demo = []
        return why
    _playing = motion
    _idle_for = 0.0
    _hud.queue_refresh()
    return ""


## Start the demo's entries that are due; "" or why one didn't start.
func _run_demo() -> String:
    var motions := GeogenMotionPlayer.motions(_hud.world.generated_dir)
    while not _demo.is_empty() and _demo[0]["at"] <= _demo_clock + 1e-6:
        var item: Dictionary = _demo.pop_front()
        var e: Dictionary = item["entry"]
        var body: Node3D = item["body"]
        var player := GeogenMotionPlayer.on(body)
        var spec: Dictionary = motions.get(str(e.get("start", "")), {})
        if player == null or spec.is_empty():
            continue
        var props: Dictionary = item["props"]
        var values := {}
        for k in e:
            if k in ["at", "start"]:
                continue
            var v = e[k]
            if v is String and v.ends_with("@start") and item["starts"].has(v.trim_suffix("@start")):
                v = item["starts"][v.trim_suffix("@start")]
            elif v is String and props.has(v):
                v = _staged(body, props[v])
                if v == null:
                    return "%s has no %s on a table in front" % [body.name, props[e[k]].get("asset", "?")]
            values[k] = v
        var why := player.play(spec, values)
        if why != "":
            return why
    return ""


## Why the gallery can't play a catalogue entry here ("" when it can).
func _why_not(entry: Dictionary) -> String:
    if entry.get("body_only", false):
        return ""
    var props: Dictionary = entry.get("props", {})
    var roles: Dictionary = entry.get("roles", {})
    var needed := roles.keys().filter(func(r): return not props.has(r) and not (roles[r] is Dictionary and roles[r].get("optional", false)))
    if not needed.is_empty() and (entry.get("demo", []) as Array).is_empty():
        return "'%s' needs %s, which the gallery doesn't stage" % [entry["name"], ", ".join(needed)]
    for body in _chosen():
        for role in props:
            if _staged(body, props[role]) == null:
                return "'%s' needs %s %s %s" % [entry["name"], props[role].get("asset", "?"),
                    "behind" if props[role].has("behind") else "on a table in front of", body.name]
    return ""


## The thing staged for `prop` ({asset}) in front of `body`: the node `<body>_<asset stem>`.
func _staged(body: Node3D, prop: Dictionary) -> Node3D:
    var stem := str(prop.get("asset", "")).get_file().get_basename()
    return _hud.world.find_child("%s_%s" % [body.name, stem], true, false) as Node3D


## Show what `entry`'s props need in front of (or behind) each person: their table and the
## things on it, or a seat; hide the rest ({}: hide everything). Shown things are put back where
## they stood. Hidden ones are out of the physics too (disabled), so nothing walks into them.
func _stage(entry: Dictionary) -> void:
    var stems := {}
    for e in _catalogue:
        for role in e.get("props", {}):
            stems[str(e["props"][role].get("asset", "")).get_file().get_basename()] = false
    var table_shown := false
    for role in entry.get("props", {}):
        var prop: Dictionary = entry["props"][role]
        stems[str(prop.get("asset", "")).get_file().get_basename()] = true
        if not prop.has("behind"):
            table_shown = true
    var props: GeogenProps = _hud.world.props
    for body in _performers:
        var table := _hud.world.find_child("%s_table" % body.name, true, false) as Node3D
        if table != null:
            table.visible = table_shown
            table.process_mode = Node.PROCESS_MODE_INHERIT if table_shown else Node.PROCESS_MODE_DISABLED
        for stem in stems:
            # (what stands on the table shows with it: the mug stays while the plate is taken)
            var show: bool = stems[stem] or (table_shown and _on_table(stem))
            var node := _hud.world.find_child("%s_%s" % [body.name, stem], true, false) as Node3D
            if node == null:
                continue
            var uid := WorldLoader.uid_of(node)
            if props != null and props.props.has(uid):
                if show:
                    props.put_back(uid)
                node = props.props[uid]["body"]
            elif show:                   # not a physics prop: back where the scene put it
                if not _homes.has(node):
                    _homes[node] = node.global_transform
                node.global_transform = _homes[node]
            node.visible = show
            node.process_mode = Node.PROCESS_MODE_INHERIT if show else Node.PROCESS_MODE_DISABLED


## Whether a staged thing of asset `stem` stands on the table (not `behind` the person).
func _on_table(stem: String) -> bool:
    for e in _catalogue:
        for role in e.get("props", {}):
            var prop: Dictionary = e["props"][role]
            if str(prop.get("asset", "")).get_file().get_basename() == stem and not prop.has("behind"):
                return true
    return false


func _chosen() -> Array[Node3D]:
    if _performer == "all":
        return _performers
    return _performers.filter(func(p): return String(p.name) == _performer)


func _process(delta: float) -> void:
    if not _open or _playing == "":
        return
    if not _demo.is_empty():
        _demo_clock += delta
        var why := _run_demo()
        if why != "":
            push_warning("motion gallery: " + why)
            _demo = []
        _idle_for = 0.0
        return
    var busy := false
    for body in _chosen():
        var player := GeogenMotionPlayer.on(body)
        if player != null and not player.rig.running().is_empty():
            busy = true
    if busy:
        _idle_for = 0.0
        return
    _idle_for += delta
    if _loop and _idle_for >= LOOP_GAP:
        play(_playing)
    elif not _loop and _idle_for > 0.0 and _playing != "":
        _playing = ""
        _hud.queue_refresh()


# -- reading -------------------------------------------------------------------------------------

func _read() -> void:
    _catalogue = GeogenMotionPlayer.catalogue(_hud.world.generated_dir)
    _performers = _find_performers()
    if _performer != "all" and _performers.all(func(p): return String(p.name) != _performer):
        _performer = "all"
    if _selected == "" and not _catalogue.is_empty():
        var first = _catalogue.filter(func(e): return e.get("body_only", false))
        _selected = (first[0] if not first.is_empty() else _catalogue[0])["name"]
    if _playing == "":
        _stage({})                   # the tables come out when a motion with things plays


func _find_performers() -> Array[Node3D]:
    var found: Array[Node3D] = []
    if _hud == null or _hud.world == null:
        return found
    for node: Node3D in _hud.world.find_children("*", "Node3D", true, false):
        if WorldLoader.geogen_extras(node).get("rig") is Dictionary:
            found.append(node)
    return found


func _entry(motion: String) -> Dictionary:
    for e in _catalogue:
        if e["name"] == motion:
            return e
    return {}


# -- drawing -------------------------------------------------------------------------------------

func hud_refresh() -> void:
    # The gallery scene opens it by itself, once per load.
    if not _open and _hud.world != null and _hud.world.scene_name == GALLERY_SCENE \
            and _auto_opened != _hud.world.generated_dir + "/" + _hud.world.scene_name:
        _auto_opened = _hud.world.generated_dir + "/" + _hud.world.scene_name
        set_open.call_deferred(true)
        return
    if not _open:
        return
    var rect := get_viewport().get_visible_rect()
    var width := clampf(rect.size.x * 0.3, 240.0, 360.0)
    var top := float(GeogenHud.TOP_INSET if _hud.overlay_shown else GeogenHud.MARGIN)   # under the F1 overlay's lines
    _frame.position = Vector2(GeogenHud.MARGIN, top)
    _frame.size = Vector2(width, rect.size.y - top - GeogenHud.MARGIN)
    _draw_performers()
    _draw_list()
    var entry := _entry(_selected)
    var text := String(entry.get("summary", ""))
    var params: Dictionary = entry.get("params", {})
    if not params.is_empty():
        text += "\n" + ", ".join(params.keys().map(func(k): return "%s %s" % [k, _number(params[k])]))
    var why_not := _why_not(entry) if not entry.is_empty() else ""
    if why_not != "":
        text += "\nNot here yet: %s. python -m geogen.motion_review --motion %s shows it." % [why_not, entry["name"]]
    elif not (entry.get("props", {}) as Dictionary).is_empty():
        text += "\nAt a table: %s." % ", ".join((entry["props"] as Dictionary).values().map(
            func(p): return str(p.get("asset", "")).get_file().get_basename()))
    _about.text = text


func _draw_performers() -> void:
    for child in _who.get_children():
        child.queue_free()
    var names: Array = ["all"]
    names.append_array(_performers.map(func(p): return String(p.name)))
    for who in names:
        var b := Button.new()
        b.text = "Everyone" if who == "all" else who
        b.toggle_mode = true
        b.button_pressed = who == _performer
        b.focus_mode = Control.FOCUS_NONE
        b.add_theme_font_size_override("font_size", HEAD_SIZE)
        b.pressed.connect(set_performer.bind(who))
        _who.add_child(b)


func _draw_list() -> void:
    for child in _list.get_children():
        child.queue_free()
    _rows.clear()
    var category := ""
    for e in _catalogue:
        if e["category"] != category:
            category = e["category"]
            var head := _label(CATEGORY_TITLES.get(category, category.capitalize()), HEAD_SIZE, COLORS["head"])
            _list.add_child(head)
        var b := Button.new()
        var playable := _why_not(e) == ""
        b.text = ("> " if e["name"] == _playing else "  ") + String(e["name"]).replace("_", " ")
        b.alignment = HORIZONTAL_ALIGNMENT_LEFT
        b.flat = e["name"] != _selected
        b.focus_mode = Control.FOCUS_NONE
        b.add_theme_font_size_override("font_size", TEXT_SIZE)
        var color: Color = COLORS["chosen"] if e["name"] == _selected else (COLORS["text"] if playable else COLORS["dim"])
        b.add_theme_color_override("font_color", color)
        b.add_theme_color_override("font_hover_color", COLORS["chosen"])
        b.tooltip_text = String(e.get("summary", ""))
        b.pressed.connect(func():
            select(e["name"])
            play(e["name"]))
        _list.add_child(b)
        _rows[e["name"]] = b
    _scroll_to_selected()


## Once the new rows are laid out, scroll the chosen one into view.
func _scroll_to_selected() -> void:
    await get_tree().process_frame
    await get_tree().process_frame
    if _rows.has(_selected) and is_instance_valid(_rows[_selected]):
        _list_scroll.ensure_control_visible(_rows[_selected])


static func _number(v) -> String:
    if (v is float or v is int) and float(v) == roundf(float(v)):
        return str(int(v))
    return str(v)


func _label(text: String, size_: int, color: Color) -> Label:
    var l := Label.new()
    l.text = text
    l.add_theme_font_size_override("font_size", size_)
    l.add_theme_color_override("font_color", color)
    return l


## What it shows: open, the motions (and which play here), who performs, what plays.
func hud_report() -> Dictionary:
    return {"open": _open, "selected": _selected, "playing": _playing, "loop": _loop,
        "performer": _performer, "performers": _performers.map(func(p): return String(p.name)),
        "motions": _catalogue.map(func(e): return {"name": e["name"], "category": e["category"],
            "plays": _why_not(e) == ""}),
        "rect": [_frame.position.x, _frame.position.y, _frame.size.x, _frame.size.y] if _open else []}
