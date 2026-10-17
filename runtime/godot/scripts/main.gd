extends Node3D
## Root of the geogen runtime: loads exports from generated/, spawns a
## first-person player sized from the manifest, and shows a debug overlay.
##
## User args (after `--`):
##   --scene NAME | --scene=NAME   load generated/NAME.glb (default: generated/catalogue.json's
##                                 default scene, else every export); a chunked export
##                                 (generated/NAME_chunks/) streams around the player
##   --scene=all                   load every export side by side, ignoring the catalogue
##   --stream                      prefer NAME's chunked export when both exist
##   --stream-radius=F,L,I         streaming radii: full exterior, LOD, interiors (m)
##   --generated=DIR               read exports from DIR instead of res://generated
##   --spawn=X,Y,Z  --yaw=DEG      player start (default: the export's first spawn
##                                 point, else in front of the model)
##   --script=FILE / --run=JSON    run control API commands in order once the world is loaded
##                                 (api.gd; docs/schema/geogen-control.v1.schema.json): a JSON list
##                                 of {cmd, args, retry?} steps. Each prints "script: {...}", then
##                                 "script done: {...}" and the runtime quits (exit 1 if a step failed)
##
## Action flags are shorthand for script steps, run after any --script steps in this order:
## --quests, --switch, --use=ASSET, --greet, --lock=@aim, --use=@aim, --wait, --walk, --nav,
## --simulate, --quit-after, --screenshot, --save, --status, --lights, --skeletons.
##   --camera=overview             start from the overview camera (F4 toggles)
##   --colliders                   show collision shapes (F2 toggles)
##   --walk=SECONDS                walk forward (player.move)
##   --greet=NAME[@S]              greet the NPC named NAME... after S s (default 2): it waves back (npcs.greet)
##   --use=ASSET                   use ASSET's interactions (interactions.use; e.g. open "door");
##                                 --use=@aim presses E on whatever the player looks at (player.use)
##   --wait=SECONDS                wait before the --walk starts (let a door swing)
##   --nav=AX,AZ:BX,BZ             the navigation path between two floor points (world.nav_path)
##   --lights                      the fixture lights (world.lights)
##   --keys=K1,K2                  keys the player holds (lock/unlock doors with L)
##   --lock=@aim                   press L on whatever the player looks at (player.lock)
##   --pitch=DEG                   look up (+) or down (-) at spawn
##   --save=PATH / --load=PATH     write the game's state (a geogen-save; state.save) / restore one at start
##   --status                      the player's pose/room/focus (player.status)
##   --npc-trace                   print every NPC decision and step ("npc: {...}")
##   --quests                      every quest's state for the player (quests.list); quest and objective
##                                 changes always print "quest event: {...}"
##   --open-doors=GLOB             open doors whose name matches before the navmesh bake, and hold them
##                                 (stair halls on a walked route; --playtest opens every door)
##   --give=ITEM[:N]               give the player N (default 1) of an item at start (repeatable)
##   --set=KEY=VALUE               set a state variable at start (world:pkg.name, actor:pkg.name for the
##                                 player, <uid>#<state>); VALUE is JSON, or text
##   --events[=TYPE,...]           print the event bus ("event: {...}"): every type, or only these
##                                 (signal, changed, use, enter, leave, travel, clock, start, arrived, npc)
##   --timescale=N                 run the world N times faster (physics ticks scale too)
##   --simulate=SECONDS            run SECONDS of world time and report NPCs, traffic and cost (time.simulate)
##   --npc-limit=N                 spawn at most N of the export's NPCs (crowd measurements)
##   --npc-labels                  show what each NPC is doing and its needs (F5 toggles)
##   --time=HH:MM                  world clock at start (default 13:00): sun, sky, street lamps,
##                                 routines and traffic follow it
##   --day-length=SECONDS          real seconds per 24 h of world time (default 1440; 0 freezes it)
##   --traffic-trace               print traffic claims, overlaps and respawns ("traffic: {...}")
##   --camera=follow[:NAME]        watch an NPC (the first, or the one whose name starts with NAME)
##   --api[=PORT]                  serve the runtime control API on 127.0.0.1:PORT (default 7878; 0 picks
##                                 a free port), printed as "api listening: {...}" (see api.gd)
##
## E uses the focused interaction, sits/lies on the furniture you look at, or greets an NPC
## (E or walking stands you up); L locks/unlocks with a held key; Q tracks the next active
## quest in the HUD (hud.gd).
##   --playtest[=N]                check every room/interaction is reachable, walk N routes
##                                 with a bot (default 6), print "playtest: {...}", quit
##
## In game, look at something interactive within reach and press E.
##   --play=ANIM[@SECONDS]         loop every animation named ANIM (or <node>_ANIM); with @SECONDS,
##                                 freeze it at that time (skeletal clips, interaction animations)
##   --skeletons                   Skeleton3D bones, skinned mesh bounds and animations (world.skeletons)
##   --screenshot=PATH             save a frame (capture.screenshot; after --quit-after, else 10 frames)
##   --quit-after=N                let N physics frames run (wait)
##   --manifest=PATH               print the player spec read from PATH
##   --menu                        open the scene list at start (screenshots)
##   --timings                     print each load's phases ("load timings: {...}", ms) and whether
##                                 the navmesh came from <generated>/.navcache
##   --list-scenes                 print the scene catalogue (name, group, exported, description), quit
##   --trace=FILE                  replay a rules conformance trace (docs/rules.md) without loading a
##                                 world; prints "trace: {name, problems}", exits 1 on problems (repeatable)
##   --switch=NAME[@S]             switch to scene NAME at S s (default 1; repeatable, in order): world.switch,
##                                 then world.stats (node/object counts)
##
## Travel points (extras.geogen.travel: doors you use, portals you walk into) fade out, load
## their target scene (or move within this one) and put the player at the named spawn; prints
## "travelled: {...}".
##
## F6 opens the scene list, F7 / F8 go to the previous / next exported scene. The last scene
## picked is remembered (user://settings.cfg, per generated dir) and loaded when no --scene is
## given (not in headless runs).

var player_spec := PlayerSpec.new()

var _spawn_override = null
var _yaw_override = null
var _manifest_arg := false
var _start_overview := false
var keys: Array = []         # --keys; held_keys() adds what the player carries
var _pitch := 0.0
var _load_path := ""
var _started := false
var _sets: Array[String] = []
var _gives: Array[String] = []
var _open_doors := ""
var _focus_affordance := {}
var _focus_npc: GeogenNpc = null
var _playtest_walks := -1
var _focus: GeogenInteraction = null
## Aimed at but refused by its `when`, with text saying why (`refused`): the prompt says it,
## and E tells the player again.
var _refused_focus: GeogenInteraction = null
## The physics prop aimed at that E would pick up (its uid), or "".
var _focus_prop := ""
## How far in front of the eyes a carried prop is held (m).
const HOLD_DISTANCE := 1.4
var _npc_labels := false
var _follow = null       # NPC name prefix to follow with the overview camera, or null
var _prompt: Label
var _play := ""
var _play_at := -1.0
var clock: GeogenClock
var _start_time := 13.0
var _menu: GeogenSceneMenu
var hud: GeogenHud                # the game HUD (hud.gd): the quest tracker, notifications and later panels
var _list_scenes := false
var _traces: Array[String] = []         # --trace: rules traces to replay instead of loading a world
var _travelling := false
var _switching := false
var _loading: Label
var _open_menu_at_start := false
var _fade: ColorRect
const FADE_SECONDS := 0.25
## An interaction target this far off the aim line (m) still takes the E prompt.
const AIM_SLACK := 0.2

const SETTINGS_PATH := "user://settings.cfg"
var _day_length := 1440.0
var _api_port := -1
## Control API steps to run once the world is loaded (--script / --run, then the action flags').
var _steps: Array = []
var _stop_on_error := true

@onready var world: WorldLoader = $World
@onready var overview: Camera3D = $OverviewCamera
@onready var overlay: Label = $Overlay/Label
var player: Player


func _ready() -> void:
    var args: Array[String] = []
    args.assign(Array(OS.get_cmdline_user_args()))
    var act := {"use": [], "greet": [], "switch": []}     # action flags: become script steps (_flag_steps)
    var i := 0
    while i < args.size():
        var arg := args[i]
        var value := arg.get_slice("=", 1) if "=" in arg else ""
        if arg == "--scene" and i + 1 < args.size():
            i += 1
            world.scene_name = args[i]
        elif arg.begins_with("--scene="):
            world.scene_name = value
        elif arg == "--stream":
            world.prefer_chunks = true
            world.force_chunks = true
        elif arg.begins_with("--stream-radius="):
            world.stream_radii = value.split_floats(",")
        elif arg.begins_with("--generated="):
            world.generated_dir = value
        elif arg.begins_with("--spawn="):
            var p := value.split_floats(",")
            _spawn_override = Vector3(p[0], p[1], p[2])
        elif arg.begins_with("--yaw="):
            _yaw_override = float(value)
        elif arg == "--colliders":
            world.show_colliders = true
        elif arg == "--camera=overview":
            _start_overview = true
        elif arg.begins_with("--walk="):
            act["walk"] = float(value)
        elif arg.begins_with("--script=") or arg.begins_with("--run="):
            var text := FileAccess.get_file_as_string(value) if arg.begins_with("--script=") else arg.trim_prefix("--run=")
            var parsed = GeogenApi.parse_script(text, arg.get_slice("=", 0))
            if parsed is String:
                push_error("geogen: " + parsed)
                get_tree().quit(2)
                return
            _steps.append_array(parsed["steps"])
            _stop_on_error = _stop_on_error and bool(parsed.get("stop_on_error", true))
            args.append_array(_start_args(parsed.get("start", {})))     # parsed like flags, after this one
        elif arg == "--playtest":
            _playtest_walks = 6
        elif arg.begins_with("--playtest="):
            _playtest_walks = int(value)
        elif arg.begins_with("--nav="):
            # X,Z (on the ground) or X,Y,Z (e.g. on a hill), for each end.
            act["nav"] = Array(value.split(":")).map(func(point): return Array(point.split_floats(",")))
        elif arg == "--lights":
            act["lights"] = true
        elif arg.begins_with("--keys="):
            keys = Array(value.split(",", false))
        elif arg.begins_with("--pitch="):
            _pitch = float(value)
        elif arg == "--lock=@aim":
            act["lock"] = true
        elif arg.begins_with("--save="):
            act["save"] = value
        elif arg.begins_with("--load="):
            _load_path = value
        elif arg == "--npc-trace":
            world.npc_trace = true
        elif arg == "--quests":
            act["quests"] = true
        elif arg.begins_with("--set="):
            _sets.append(arg.trim_prefix("--set="))
        elif arg.begins_with("--open-doors="):
            _open_doors = arg.trim_prefix("--open-doors=")
        elif arg.begins_with("--give="):
            _gives.append(arg.trim_prefix("--give="))
        elif arg == "--events" or arg.begins_with("--events="):
            world.events.print_types = ["*"] if value == "" else Array(value.split(","))
        elif arg.begins_with("--timescale="):
            var scale := maxf(float(value), 0.01)
            Engine.time_scale = scale
            world.apply_physics_rate()
        elif arg.begins_with("--simulate="):
            act["simulate"] = float(value)
        elif arg.begins_with("--npc-limit="):
            world.npc_limit = int(value)
        elif arg.begins_with("--time="):
            _start_time = GeogenClock.parse(value)
        elif arg.begins_with("--day-length="):
            _day_length = float(value)
        elif arg == "--traffic-trace":
            world.traffic_trace = true
        elif arg == "--npc-labels":
            _npc_labels = true
        elif arg == "--camera=follow" or arg.begins_with("--camera=follow:"):
            _follow = arg.get_slice(":", 1) if ":" in arg else ""
            _start_overview = true
        elif arg == "--status":
            act["status"] = true
        elif arg == "--use=@aim":
            act["use_aim"] = true
        elif arg.begins_with("--greet="):
            act["greet"].append([value.get_slice("@", 0), float(value.get_slice("@", 1)) if "@" in value else 2.0])
        elif arg.begins_with("--use="):
            act["use"].append(value)
        elif arg.begins_with("--wait="):
            act["wait"] = float(value)
        elif arg.begins_with("--play="):
            _play = value.get_slice("@", 0)
            if "@" in value:
                _play_at = float(value.get_slice("@", 1))
        elif arg == "--skeletons":
            act["skeletons"] = true
        elif arg.begins_with("--quit-after="):
            act["quit_after"] = int(value)
        elif arg.begins_with("--screenshot="):
            act["screenshot"] = value
        elif arg == "--timings":
            world.print_timings = true
        elif arg == "--menu":
            _open_menu_at_start = true
        elif arg == "--list-scenes":
            _list_scenes = true
        elif arg.begins_with("--trace="):
            _traces.append(arg.trim_prefix("--trace="))
        elif arg.begins_with("--switch="):
            act["switch"].append([value.get_slice("@", 0), float(value.get_slice("@", 1)) if "@" in value else 1.0])
        elif arg == "--api":
            _api_port = 7878
        elif arg.begins_with("--api="):
            _api_port = int(value)
        elif arg.begins_with("--manifest="):
            var spec := PlayerSpec.from_manifest(value)
            if spec:
                player_spec = spec
                _manifest_arg = true
        i += 1
    _steps.append_array(_flag_steps(act))
    print("geogen runtime ready (Godot %s)" % Engine.get_version_info().string)
    if not _traces.is_empty():
        set_process(false)
        set_physics_process(false)
        get_tree().quit(GeogenTraceRunner.run_files(_traces))
        return
    if _list_scenes:
        _print_scene_list()
        set_process(false)
        set_physics_process(false)
        get_tree().quit()
        return

    world.open_before_bake = _playtest_walks >= 0 or _open_doors != ""
    world.open_only = _open_doors if _playtest_walks < 0 else ""
    world.prime_focus = _spawn_override
    world.world_loaded.connect(_on_world_loaded)
    world.travel_requested.connect(_travel)
    world.interaction_event.connect(func(asset: String, interaction: String, state: String, event: String):
        print("interaction event: %s" % JSON.stringify(
            {"asset": asset, "interaction": interaction, "state": state, "event": event})))
    world.place_event.connect(func(event: Dictionary): print("place event: %s" % JSON.stringify(event)))
    world.world_loaded.connect(func(_aabb: AABB):       # a new game: once, not on a scene switch or a loaded save
        if not _started:
            _started = true
            if _load_path == "":
                world.events.post("start", "", "", "player")
            else:                               # before --set and --quests, which act on what it restores
                var saved = WorldLoader.read_save(_load_path)
                if saved is Dictionary:
                    world.load_state(saved)
                else:
                    push_warning("geogen: --load=%s: not a save" % _load_path)
            for entry in _sets:
                var key := entry.get_slice("=", 0)
                var text := entry.substr(key.length() + 1)
                var value = JSON.parse_string(text)
                var problem := world.rules.set_value(key, text if value == null else value, "player")
                if problem != "":
                    push_warning("geogen: --set=%s: %s" % [entry, problem])
            for entry in _gives:
                var given := world.rules.give(entry.get_slice(":", 0), int(entry.get_slice(":", 1)) if ":" in entry else 1)
                if given != "":
                    push_warning("geogen: --give=%s: %s" % [entry, given]))
    clock = GeogenClock.new()
    clock.name = "Clock"
    clock.night_changed.connect(func(night: bool): world.events.post("clock", "", "night" if night else "day"))
    clock.hours = _start_time
    clock.day_length = _day_length
    clock.sun = $Sun
    clock.environment = ($WorldEnvironment as WorldEnvironment).environment
    clock.world = world
    world.clock = clock
    if world.scene_name == "all":
        world.scene_name = ""
    else:
        world.use_catalogue_default(_remembered_scene())
    # The HUD's panels listen before the first world loads: a quest that starts with the game is news.
    hud = GeogenHud.new()
    hud.world = world
    hud.input_blocked = func(): return _menu.visible
    hud.add_panel(GeogenHudStats.new(), "top_left")
    hud.add_panel(GeogenHudQuestTracker.new(), "top_right")
    hud.add_panel(GeogenHudNotifications.new(), "bottom_left")
    hud.add_panel(GeogenHudInventory.new(), "bottom_right")
    hud.add_panel(GeogenHudMarkers.new(), "screen")
    hud.move_child(hud.panel("markers"), 0)         # under the corner panels and the quest log
    hud.add_panel(GeogenHudBanner.new(), "screen")
    hud.add_panel(GeogenHudCrossing.new(), "screen")
    hud.add_panel(GeogenHudQuestLog.new(), "screen")
    hud.add_panel(GeogenHudDialogue.new(), "screen")
    hud.modal_changed.connect(_hud_modal)
    world.load_all()
    add_child(clock)          # applies the time (night lights) once the world is loaded
    _show_npc_labels()
    if _play != "":
        _play_animations()
    if not world.npcs.is_empty():
        print("npcs: %s" % JSON.stringify(world.npcs.map(func(n): return String(n.name))))
    _prompt = Label.new()
    _prompt.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    _prompt.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
    _prompt.grow_horizontal = Control.GROW_DIRECTION_BOTH     # stays centred as its text changes
    _prompt.position.y += 40
    _prompt.add_theme_constant_override("outline_size", 6)
    _prompt.add_theme_color_override("font_outline_color", Color(0, 0, 0, 0.75))
    var spec := world.player_spec()
    if spec and not _manifest_arg:
        player_spec = spec
    print("player spec: %s" % JSON.stringify(player_spec.to_dict()))

    player = Player.new()
    player.spec = player_spec
    add_child(player)
    _fade = ColorRect.new()
    _fade.color = Color(0, 0, 0, 0)
    _fade.mouse_filter = Control.MOUSE_FILTER_IGNORE
    _fade.set_anchors_preset(Control.PRESET_FULL_RECT)
    $Overlay.add_child(_fade)
    _loading = Label.new()
    _loading.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    _loading.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
    _loading.add_theme_font_size_override("font_size", 22)
    _loading.visible = false
    $Overlay.add_child(_loading)
    _menu = GeogenSceneMenu.new()
    _menu.picked.connect(func(scene: String): switch_scene(scene))
    _menu.closed.connect(_menu_closed)
    add_child(_menu)
    hud.add_child(_prompt)                    # the E prompt, under the crosshair: part of the HUD, not the F1 overlay
    add_child(hud)
    move_child(hud, $Overlay.get_index())     # under the overlay, so its fade covers the HUD
    _place_player(world.world_aabb())
    _frame_overview(world.world_aabb())
    # The viewport auto-selects the first camera to enter the tree (the
    # overview), so always pick one explicitly.
    (overview if _start_overview else player.camera).make_current()
    if _playtest_walks >= 0:
        var playtest := GeogenPlaytest.new()
        playtest.world = world
        playtest.player = player
        playtest.start = player.global_position
        playtest.start_yaw = rad_to_deg(player.rotation.y)
        playtest.walk_count = _playtest_walks
        add_child(playtest)
    if DisplayServer.get_name() != "headless" and _steps.is_empty():
        Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
    if _open_menu_at_start:
        _open_menu()
    if _api_port >= 0 or not _steps.is_empty():
        var api := GeogenApi.new()
        api.name = "Api"
        api.main = self
        api.port = maxi(_api_port, 0)
        api.listen = _api_port >= 0
        api.script_steps = _steps
        api.stop_on_error = _stop_on_error
        add_child(api)


## A script's ``start`` (how the world starts: the schema's start object) as the flags that set it.
static func _start_args(start: Dictionary) -> Array[String]:
    var out: Array[String] = []
    var joined := func(v): return ",".join(Array(v).map(func(x): return str(x))) if v is Array else str(v)
    for key in start:
        var v = start[key]
        match key:
            "set":
                for k in v:
                    out.append("--set=%s=%s" % [k, JSON.stringify(v[k])])
            "give":
                for item in v:
                    out.append("--give=%s:%d" % [item, int(v[item])])
            _:
                out.append("--%s=%s" % [key.replace("_", "-"), joined.call(v)])
    return out


## The action flags as control API steps, in a fixed order (see the header).
static func _flag_steps(act: Dictionary) -> Array:
    var steps := []
    if act.get("quests", false):
        steps.append({"cmd": "quests.list"})
    var clock := 0.0
    for entry in act["switch"]:
        if entry[1] > clock:
            steps.append({"cmd": "wait", "args": {"seconds": entry[1] - clock}})
            clock = entry[1]
        steps.append({"cmd": "world.switch", "args": {"scene": entry[0]}})
        steps.append({"cmd": "world.stats"})
    for asset in act["use"]:
        steps.append({"cmd": "interactions.use", "args": {"asset": asset, "actor": "player"}})
    var greets: Array = act["greet"].duplicate()
    greets.sort_custom(func(a, b): return a[1] < b[1])
    for entry in greets:
        if entry[1] > clock:
            steps.append({"cmd": "wait", "args": {"seconds": entry[1] - clock}})
            clock = entry[1]
        steps.append({"cmd": "npcs.greet", "args": {"npc": entry[0], "wait": 30.0}})
    if act.get("lock", false):
        steps.append({"cmd": "player.lock", "retry": 2.0})
    if act.get("use_aim", false):
        steps.append({"cmd": "player.use", "retry": 2.0})
    if act.get("wait", 0.0) > 0.0:
        steps.append({"cmd": "wait", "args": {"seconds": act["wait"]}})
    if act.get("walk", 0.0) > 0.0:
        steps.append({"cmd": "player.move", "args": {"forward": 1.0, "seconds": act["walk"]}})
    if act.has("nav"):
        steps.append({"cmd": "wait_ready", "args": {"timeout": 120.0}})
        steps.append({"cmd": "world.nav_path", "args": {"from": act["nav"][0], "to": act["nav"][1]}})
    if act.get("simulate", 0.0) > 0.0:
        steps.append({"cmd": "time.simulate", "args": {"seconds": act["simulate"]}})
    if act.get("quit_after", 0) > 0:
        steps.append({"cmd": "wait", "args": {"frames": act["quit_after"]}})
    if act.has("screenshot"):
        if act.get("quit_after", 0) <= 0:
            steps.append({"cmd": "wait", "args": {"frames": 10}})     # let shadows and SSAO settle
        steps.append({"cmd": "capture.screenshot", "args": {"path": act["screenshot"]}})
    if act.has("save"):
        steps.append({"cmd": "state.save", "args": {"path": act["save"]}})
    for flag in ["status", "lights", "skeletons"]:
        if act.get(flag, false):
            steps.append({"cmd": {"status": "player.status", "lights": "world.lights", "skeletons": "world.skeletons"}[flag]})
    return steps


func _on_world_loaded(aabb: AABB) -> void:
    _frame_overview(aabb)
    $ScaleReference.visible = aabb.size == Vector3.ZERO


## Spawn in front (+Z) of whatever was loaded, facing it. The command line's
## --spawn / --yaw / --pitch apply to the first world only (``overrides``).
func _place_player(aabb: AABB, spawn_name := "", overrides := true) -> void:
    var pos := Vector3(0, 0, 5)
    var yaw := 0.0
    if not world.model_aabbs.is_empty():
        aabb = world.model_aabbs[0]  # several exports load in a row: start at the first
    if aabb.size != Vector3.ZERO:
        pos = Vector3(aabb.get_center().x, 0, aabb.end.z + 3.0)
    # Use the first export's own manifest spawn (several exports lay out in a row).
    var spawn := {}
    if spawn_name != "":
        spawn = world.spawn_named(spawn_name)
    else:
        for s in world.spawns:
            if s.get("model", 0) == 0:
                spawn = s
                break
    if not spawn.is_empty():
        pos = spawn["position"]
        yaw = spawn["yaw_deg"]
    if overrides and _spawn_override != null:
        pos = _spawn_override
    if overrides and _yaw_override != null:
        yaw = _yaw_override
    player.spawn(pos, yaw)
    if overrides:
        player.head.rotation.x = deg_to_rad(_pitch)


## Unload the current world and load scene ``scene`` (a catalogue / generated
## name) in its place, the player at its spawn ``spawn_name`` (default: the
## first). A coroutine: it fades to a loading screen (unless already faded, as
## travel does) while the export parses on a worker thread. False (nothing
## changes) if the scene isn't exported or a switch is already under way.
func switch_scene(scene: String, spawn_name := "") -> bool:
    if not world.is_exported(scene):
        push_warning("geogen: can't switch to '%s': not exported in %s" % [scene, world.generated_dir])
        return false
    if _switching:
        return false
    _switching = true
    var started := Time.get_ticks_msec()
    var fade_back := _fade.color.a < 0.99
    player.process_mode = Node.PROCESS_MODE_DISABLED
    if fade_back:
        await _fade_to(1.0)
    _loading.text = "Loading %s..." % scene.replace("_", " ")
    _loading.visible = true
    if not player.pose.is_empty():
        player.leave_pose()
    _focus = null
    _focus_npc = null
    _focus_affordance = {}
    var previous := world.scene_name
    world.select_scene(scene)
    world.prime_focus = null
    world.prime_spawn = spawn_name
    await world.load_all_async()
    world.prime_spawn = ""
    world.set_night(clock.night)
    _show_npc_labels()
    var spec := world.player_spec()
    if spec and not _manifest_arg:
        player_spec = spec
        player.apply_spec(spec)
    _place_player(world.world_aabb(), spawn_name, false)
    _remember_scene(scene)
    var p := player.global_position
    print("switched: %s" % JSON.stringify({"from": previous, "scene": scene, "spawn": spawn_name,
        "ms": Time.get_ticks_msec() - started, "x": p.x, "y": p.y, "z": p.z,
        "npcs": world.npcs.size(), "rooms": world.rooms.size()}))
    _loading.visible = false
    player.process_mode = Node.PROCESS_MODE_INHERIT
    if fade_back:
        await _fade_to(0.0)
    _switching = false
    return true


## A travel point fired: fade out, go to its scene (or stay) at its spawn, fade back in.
func _travel(travel: Dictionary) -> void:
    if _travelling:
        return
    _travelling = true
    var scene := str(travel.get("scene", ""))
    var spawn_name := str(travel.get("spawn", ""))
    player.scripted_move = null
    player.process_mode = Node.PROCESS_MODE_DISABLED
    await _fade_to(1.0)
    var ok := true
    if scene == "" or scene == world.scene_name:
        if not player.pose.is_empty():
            player.leave_pose()
        _place_player(world.world_aabb(), spawn_name, false)
        world.stream_focus = player.global_position
        world.arm_travel()
        player.process_mode = Node.PROCESS_MODE_INHERIT     # switch_scene does this for the other case
    else:
        ok = await switch_scene(scene, spawn_name)
    var p := player.global_position
    print("travelled: %s" % JSON.stringify({"scene": world.scene_name, "spawn": spawn_name, "ok": ok,
        "via": travel.get("on", "use"), "x": p.x, "y": p.y, "z": p.z, "yaw": rad_to_deg(player.rotation.y),
        "room": world.room_at(p + Vector3(0, 0.5, 0))}))
    if not ok:
        player.process_mode = Node.PROCESS_MODE_INHERIT
    await _fade_to(0.0)
    _travelling = false


func _fade_to(alpha: float) -> void:
    var tween := create_tween()
    tween.tween_property(_fade, "color:a", alpha, FADE_SECONDS)
    await tween.finished


## F7 / F8: the previous / next exported scene in catalogue order (wrapping).
func _step_scene(step: int) -> void:
    var names := world.scene_list().filter(func(s): return s["exported"]).map(func(s): return s["name"])
    if names.is_empty():
        return
    var i := names.find(world.scene_name)
    switch_scene(names[posmod(i + step, names.size())] if i >= 0 else names[0])


func _open_menu() -> void:
    _menu.open(world.scene_list(), world.scene_name)
    player.process_mode = Node.PROCESS_MODE_DISABLED
    Input.mouse_mode = Input.MOUSE_MODE_VISIBLE


## A HUD screen (the quest log) opened or closed: the player stands still and the mouse is free while it is up.
func _hud_modal(open: bool) -> void:
    if player == null:
        return
    if open:
        player.process_mode = Node.PROCESS_MODE_DISABLED
        Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
    else:
        _menu_closed()


func _menu_closed() -> void:
    player.process_mode = Node.PROCESS_MODE_INHERIT
    if DisplayServer.get_name() != "headless" and _steps.is_empty():
        Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _show_npc_labels() -> void:
    for npc in world.npcs:
        if is_instance_valid(npc):
            npc.label.visible = _npc_labels


## --list-scenes
func _print_scene_list() -> void:
    var scenes := world.scene_list()
    for s in scenes:
        print("%-20s %-9s %-13s %s" % [s["name"], s["group"], "exported" if s["exported"] else "not exported",
            s["description"]])
    print("scenes: %s" % JSON.stringify({"default": world.catalogue.get("default", ""), "scenes": scenes}))


## The last scene picked with this generated dir (interactive runs only), or "".
func _remembered_scene() -> String:
    if DisplayServer.get_name() == "headless":
        return ""
    var cfg := ConfigFile.new()
    if cfg.load(SETTINGS_PATH) != OK:
        return ""
    return str(cfg.get_value("last_scene", _settings_key(), ""))


func _remember_scene(scene: String) -> void:
    if DisplayServer.get_name() == "headless":
        return
    var cfg := ConfigFile.new()
    cfg.load(SETTINGS_PATH)
    cfg.set_value("last_scene", _settings_key(), scene)
    cfg.save(SETTINGS_PATH)


func _settings_key() -> String:
    return "dir_" + ProjectSettings.globalize_path(world.generated_dir).md5_text()


func _frame_overview(aabb: AABB) -> void:
    if aabb.size == Vector3.ZERO:
        return
    var center := aabb.get_center()
    var dist := aabb.size.length() * 1.1
    overview.look_at_from_position(center + Vector3(0.7, 0.55, 1.0).normalized() * dist, center)


func _unhandled_input(event: InputEvent) -> void:
    if (_menu != null and _menu.visible) or (hud != null and hud.modal != null):
        return
    if event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_F6:
        _open_menu()
    elif event is InputEventKey and event.pressed and not event.echo and event.physical_keycode in [KEY_F7, KEY_F8]:
        _step_scene(-1 if event.physical_keycode == KEY_F7 else 1)
    elif event.is_action_pressed("geogen_toggle_overlay"):
        overlay.visible = not overlay.visible
    elif event.is_action_pressed("geogen_toggle_colliders"):
        world.show_colliders = not world.show_colliders
    elif event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_E:
        _press_use()
    elif event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_L:
        if _focus != null:
            _focus.toggle_lock(held_keys(), "player")
    elif event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_F5:
        _npc_labels = not _npc_labels
        _show_npc_labels()
    elif event is InputEventKey and event.pressed and event.physical_keycode == KEY_F4:
        if overview.current:
            player.camera.make_current()
        else:
            overview.make_current()


func _physics_process(_delta: float) -> void:
    if player != null:
        world.stream_focus = player.global_position
    _update_focus()


func _process(_delta: float) -> void:
    _follow_npc()
    if overlay.visible:
        var p := player.global_position
        var room := world.room_at(p + Vector3(0, 0.5, 0))
        var stream := world.stream_report()
        var chunk_info := "" if stream.is_empty() else "   chunks %d full / %d lod / %d interiors%s" % [
            stream["full"].size(), stream["lod"].size(), stream["interiors"].size(),
            " (+%d loading)" % stream["pending"] if stream["pending"] > 0 else ""]
        overlay.text = "%d fps   %s   %s%s\npos %.2f, %.2f, %.2f   room: %s%s%s\nWASD move  Shift sprint  Space jump  F1 overlay  F2 colliders  F3 fly  F4 overview  Q quest  F6 scenes  F7/F8 prev/next  Esc mouse" % [
            Engine.get_frames_per_second(), clock.label(),
            world.scene_name if world.scene_name != "" else "all exports", chunk_info,
            p.x, p.y, p.z, room if room != "" else "-",
            "   [fly]" if player.flying else "",
            "   [colliders]" if world.show_colliders else ""]


## --play: start (or freeze at --play=NAME@T) every matching animation in the loaded exports.
func _play_animations() -> void:
    var played := []
    for source: AnimationPlayer in world.find_children("*", "AnimationPlayer", true, false):
        var free := source
        for anim_name in source.get_animation_list():
            if anim_name != _play and not String(anim_name).ends_with("_" + _play):
                continue
            # One player plays one animation: each further match (another character) gets a copy.
            var ap := free if free != null else source.duplicate() as AnimationPlayer
            if free == null:
                source.add_sibling(ap)
            free = null
            if _play_at < 0.0:
                ap.get_animation(anim_name).loop_mode = Animation.LOOP_LINEAR   # imported clips don't loop
            ap.play(anim_name)
            if _play_at >= 0.0:
                ap.seek(_play_at, true)
                ap.pause()
            played.append(anim_name)
    print("play: %s" % JSON.stringify(played))


## What the player is aiming at within reach, and its prompt ("E: Open").
func _update_focus() -> void:
    _focus = null
    _refused_focus = null
    _focus_affordance = {}
    _focus_npc = null
    _focus_prop = ""
    if world.props.held != "":
        pass        # carrying: E drops it, nothing else is aimed at
    elif player != null and player.camera.current and player.pose.is_empty():
        var cam := player.camera
        var from := cam.global_position
        var to := from - cam.global_basis.z * _longest_reach()
        var query := PhysicsRayQueryParameters3D.create(from, to)
        query.exclude = [player.get_rid()]
        var hit := get_world_3d().direct_space_state.intersect_ray(query)
        if hit and from.distance_to(hit["position"]) > player_spec.reach:
            # Beyond the player's reach only an interaction that reaches further counts.
            var far := world.interaction_for(hit["collider"])
            if far == null or from.distance_to(hit["position"]) > _reach_of(far):
                hit = {}
        if hit:
            var it := world.interaction_for(hit["collider"])
            var prop := world.props.prop_of(hit["collider"])
            if prop != "":
                if world.props.can_carry(prop):
                    _focus_prop = prop
            elif hit["collider"] is GeogenNpc:
                if (hit["collider"] as GeogenNpc).can_greet():
                    _focus_npc = hit["collider"]
            elif it != null and it.can_use():
                _focus = it
            elif it != null and it.refusal() != "":
                _refused_focus = it
            else:
                _focus_affordance = world.affordance_near(hit["position"])
        if _focus == null and _focus_npc == null:
            # A seated or lying NPC has no collider: the aim line passing its body finds it,
            # unless something (not the seat it is in) is in the way.
            var limit: float = player_spec.reach
            if hit:
                limit = minf(limit, from.distance_to(hit["position"]) + 0.4)
            for npc in world.npcs:
                if is_instance_valid(npc) and npc.can_greet() and npc.posed() \
                        and npc.aim_distance(from, -cam.global_basis.z) <= limit:
                    _focus_npc = npc
                    _focus_affordance = {}
                    break
        if _focus == null and _focus_npc == null and _focus_affordance.is_empty() and _refused_focus == null \
                and _focus_prop == "":
            _focus = _assisted_focus(from, -cam.global_basis.z)
    if hud != null:
        hud.crosshair.aim(focus_info().get("kind", "") not in ["", "refused"],
            player != null and player.camera.current and player.pose.is_empty())
    if _prompt:
        if player != null and not player.pose.is_empty():
            _prompt.text = "E: Stand up"
        elif world.props.held != "":
            _prompt.text = "E: Drop"
        elif _focus_prop != "":
            _prompt.text = "E: Pick up"
        elif _focus_npc != null:
            _prompt.text = "E: %s" % _focus_npc.greet_prompt()
        elif _focus != null:
            _prompt.text = "E: %s" % _focus.prompt(held_keys())
        elif _refused_focus != null:
            _prompt.text = _refused_focus.refusal()
        elif not _focus_affordance.is_empty():
            _prompt.text = "E: %s" % {"sit": "Sit", "lie": "Lie down", "use": "Use", "stand": "Stand here"}.get(
                _focus_affordance["type"], "Use")
        else:
            _prompt.text = ""


## How far from the eye ``it`` can be used from: its own `reach` when that is longer.
func _reach_of(it: GeogenInteraction) -> float:
    return maxf(player_spec.reach, it.reach)


## The longest reach of anything loaded: how far the focus ray looks.
func _longest_reach() -> float:
    var longest: float = player_spec.reach
    for it in world.interactions:
        if is_instance_valid(it):
            longest = maxf(longest, it.reach)
    return longest


## Aim assist: the usable interaction whose target is nearest the aim line, when the line
## itself lands on nothing usable. Small targets (a light switch is 8 cm wide) are hard to
## hit exactly; within AIM_SLACK of one, and with nothing in the way, E takes it.
func _assisted_focus(from: Vector3, forward: Vector3) -> GeogenInteraction:
    var best: GeogenInteraction = null
    var best_miss := AIM_SLACK
    var space := get_world_3d().direct_space_state
    for it in world.interactions:
        if not is_instance_valid(it) or not it.can_use():
            continue
        for target in it.targets:
            if not is_instance_valid(target) or target.global_position.distance_to(from) > _reach_of(it) + 2.0:
                continue
            var centre := target.global_position
            if target is MeshInstance3D:
                centre = target.global_transform * (target as MeshInstance3D).get_aabb().get_center()
            else:
                var box := WorldLoader._aabb(target)
                if box.size != Vector3.ZERO:
                    centre = box.get_center()
            var along := (centre - from).dot(forward)
            if along <= 0.0 or centre.distance_to(from) > _reach_of(it):
                continue
            var miss := (centre - from - forward * along).length()
            if miss >= best_miss:
                continue
            var query := PhysicsRayQueryParameters3D.create(from, centre)
            query.exclude = [player.get_rid()]
            var hit := space.intersect_ray(query)
            if hit and world.interaction_for(hit["collider"]) != it \
                    and (hit["position"] as Vector3).distance_to(centre) > 0.1:
                continue        # something else is in the way
            best = it
            best_miss = miss
    return best


## For the runtime API: refresh what the player aims at now.
func update_focus() -> void:
    _update_focus()


## What the player aims at (the E prompt's subject), or {}.
func focus_info() -> Dictionary:
    if world.props.held != "":
        return {"kind": "drop", "prop": world.props.held, "prompt": "Drop"}
    if _focus_prop != "":
        return {"kind": "carry", "prop": _focus_prop, "prompt": "Pick up"}
    if _focus_npc != null:
        return {"kind": "npc", "npc": String(_focus_npc.name), "prompt": _focus_npc.greet_prompt()}
    if _focus != null:
        return {"kind": "interaction", "asset": String(_focus.asset.name), "interaction": _focus.interaction_name,
            "prompt": _focus.prompt(held_keys())}
    if _refused_focus != null:
        return {"kind": "refused", "asset": String(_refused_focus.asset.name),
            "interaction": _refused_focus.interaction_name, "prompt": _refused_focus.refusal()}
    if not _focus_affordance.is_empty():
        return {"kind": "affordance", "asset": str(_focus_affordance.get("asset", "")),
            "type": _focus_affordance["type"]}
    return {}


## Press E on whatever is aimed at now (the runtime API); returns what that was, or {}.
func press_use() -> Dictionary:
    _update_focus()
    var used := focus_info()
    if not player.pose.is_empty():
        used = {"kind": "stand"}
    if not used.is_empty():
        _press_use()
    return used


func is_switching() -> bool:
    return _switching or _travelling


## Follow an NPC (name prefix) with the overview camera, or stop following (null).
func set_follow(prefix) -> void:
    _follow = prefix


func _press_use() -> void:
    if not player.pose.is_empty():
        player.leave_pose()
    elif world.props.held != "":
        world.props.drop()
    elif _focus_prop != "":
        world.props.pick(_focus_prop, player, _hold_point)
    elif _focus_npc != null:
        _focus_npc.greet()
    elif _focus != null:
        _focus.use("player")
    elif _refused_focus != null:
        _refused_focus.use("player")        # refused: it says why (WorldLoader -> GeogenRules.tell)
    elif not _focus_affordance.is_empty():
        player.take_pose(_focus_affordance)


## Where a carried prop is held: in front of the eyes.
func _hold_point() -> Vector3:
    var cam := player.camera
    return cam.global_position - cam.global_basis.z * HOLD_DISTANCE


## --camera=follow: keep the overview camera on an NPC, from the first of a ring
## of viewpoints (front first) with a clear line of sight under any ceiling.
func _follow_npc() -> void:
    if _follow == null or not overview.current:
        return
    for npc in world.npcs:
        if is_instance_valid(npc) and String(npc.name).begins_with(_follow):
            var head := npc.global_position + Vector3(0, 1.3, 0)
            var space := get_world_3d().direct_space_state
            var best := head + Vector3(0, 6, 0)
            for step in [0, 1, -1, 2, -2, 3, -3, 4]:
                var dir := Basis(Vector3.UP, npc.global_rotation.y + step * PI / 4.0) * Vector3(0, 0, 1)
                var eye := head + dir * 2.6 + Vector3(0, 0.6, 0)
                var query := PhysicsRayQueryParameters3D.create(head, eye + dir * 0.3)
                query.exclude = [npc.get_rid()]
                if space.intersect_ray(query).is_empty():
                    best = eye
                    break
            overview.look_at_from_position(best, npc.global_position + Vector3(0, 0.8, 0))
            return


## The lock keys the player holds: --keys, plus every item it carries and what each `opens` (geogen.18.3).
func held_keys() -> Array:
    var out := keys.duplicate()
    if world != null and world.rules != null:
        out.append_array(world.rules.carried_keys("player"))
    return out
