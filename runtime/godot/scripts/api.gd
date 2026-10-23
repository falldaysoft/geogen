class_name GeogenApi
extends Node
## Runtime control API (``--api[=PORT]``): a localhost TCP server so tools, tests and agents can
## query and drive one long-lived world instead of launching a process per check.
##
## The same commands run as a script (``--script=FILE`` / ``--run=JSON``): a list of
## ``{"cmd", "args", "retry"?, "id"?}`` steps (or ``{"steps": [...], "stop_on_error"?}``) run in
## order once the world is loaded, each printing ``script: {"step", "cmd", "ok", "result" | "error"}``;
## then ``script done: {...}`` and the runtime quits (exit 1 if a step failed), unless --api also
## serves. The protocol is docs/schema/geogen-control.v1.schema.json, for every runtime.
##
## Newline-delimited JSON. Requests ``{"id": 1, "cmd": "player.status", "args": {...}}`` get
## ``{"id": 1, "ok": true, "result": ...}`` or ``{"id": 1, "ok": false, "error": "..."}``.
## Slow commands (walk_to, step, wait_ready, switch) answer when they finish; other requests
## keep being served meanwhile. After ``events.subscribe`` the connection also gets unsolicited
## ``{"event": "<stream>", "data": {...}}`` lines.
##
## Points are [x, y, z] in world metres, angles in degrees. Commands are thin wrappers over
## WorldLoader / Player / GeogenNpc / GeogenInteraction; the list is in COMMANDS (and `help`).
## Python client: geogen/runtime_client.py.

const COMMANDS := {
    "help": "list commands",
    "ping": "liveness check",
    "wait": "{seconds? | frames?} let the world run that much physics time (or that many physics frames)",
    "batch": "{commands: [{cmd, args}], stop_on_error?} run commands in order, one reply",
    "quit": "stop the runtime",
    "wait_ready": "{timeout?} wait until the world is loaded and the navmesh answers queries",
    "world.info": "scene, bounds, spawns, rooms, npc names, player spec, clock",
    "world.scenes": "the scene catalogue",
    "world.query": "{tag?, type?, name? (glob), uid? (glob), room?, interactive?, limit?} nodes carrying extras.geogen",
    "world.room_at": "{pos} room id at a point",
    "world.raycast": "{from, to} first hit: position, normal, node, room, interaction",
    "world.nav_path": "{from, to} navmesh path: points, length, reached",
    "world.switch": "{scene, spawn?} load another exported scene",
    "world.reload": "{keep_player?} re-read the current scene's export (after a re-export)",
    "world.node": "{name | uid | node_path} one node: transform, world bounds, extras, colliders, children, interactions",
    "world.lights": "the light fixtures: {fixtures: {name: {visible, energy, range, y}}, total, light3d}",
    "world.skeletons": "skeletons (bone positions, skinned mesh bounds in the current pose) and animations",
    "world.stats": "engine object counts: nodes, orphans, objects, resources, nav_regions, bodies, world_children",
    "player.status": "position, yaw, pitch, eye, room, scene, pose, focus, keys, on_floor, walking; stream (chunked worlds)",
    "player.teleport": "{pos, yaw?} put the player's feet at pos",
    "player.look": "{yaw?, pitch?} or {at: point}",
    "player.move": "{forward?, strafe?, seconds} walk with the controls (until a travel point, then where it lands)",
    "player.walk_to": "{pos, timeout?} follow the navmesh there with the real body, opening closed doors on the way "
        + "(a held key unlocks one) -> status: arrived | stuck | timeout | locked | refused, doors",
    "player.use": "{asset?, interaction?} use the aimed thing, or a named asset's interactions",
    "player.lock": "lock or unlock the aimed interaction with the keys the player holds -> {asset, interaction, ok, locked}",
    "player.sit": "{asset?} take the nearest (or the asset's) sit/lie affordance",
    "player.stand": "leave a sit/lie pose",
    "player.carry": "{prop?} pick up the aimed physics prop, or the one with that uid (glob) or node name, from anywhere: "
        + "-> {ok, held}",
    "player.drop": "let go of what the player carries -> {dropped}",
    "world.bodies": "every physics prop: uid, node, position, center (of its box), yaw_deg, live, held, sleeping, speed",
    "flows.list": "spawned items (geogen/flows.py): {sources: [{uid, running, queued, emitted, alive, items}], "
        + "items: [{id, source, item, mode, position, data, age}]}",
    "player.keys": "{keys?} set (or get) the keys given by --keys; held: those plus what the player carries",
    "window.size": "{size?: [w, h]} get or set the window size (headless runs too: layouts follow it)",
    "input.key": "{key} press and release a keyboard key by name (Q, F5, E), as the player would",
    "input.mouse": "{at: [x, y] (viewport pixels), button?: left | right | middle, double?: bool} click there, as the "
        + "player would (a double click is a second press marked double)",
    "interactions.list": "{asset?} interactions with states",
    "interactions.states": "every interaction's saved state, by stable id (<uid>#<interaction>)",
    "interactions.set": "{asset, interaction?, state, instant?, wait?} jump or animate to a state; wait: until the parts arrive, returning their bounds",
    "interactions.use": "{asset, interaction?, wait?} advance to the next state (honours locks)",
    "interactions.lock": "{asset, interaction?, locked} lock or unlock without a key",
    "state.save": "{path?} -> the game as a geogen-save (docs/schema/geogen-save.v1.schema.json): world and actor "
        + "sections, clock, timers; interactions are instance state in world. path: also write it there",
    "state.get": "{key, actor?} a variable's value (world:pkg.name, actor:pkg.name, <uid>#<state>); actor: player",
    "state.set": "{key, value, actor?} set one, as a console would: rules and bindings react",
    "state.list": "-> {world, actors}: every stored value that differs from its default",
    "quests.list": "{actor?} every quest's state for an actor (default player): status, objectives "
        + "(done | open | waiting, progress, markers)",
    "state.load": "{state | path} restore a save (one from before the format still loads); -> the save now",
    "hud.aim": "the crosshair: {visible, active (E would do something), focus}",
    "hud.stats": "what the scene's HUD stats and banner show (geogen/hud.py): {stats: {visible, stats: [{label, value, of?, text}]}, "
        + "banner: {visible, text?, tone?}}",
    "dialogue.show": "the open conversation as shown: {open, dialogue?, anchor?, passage?, title?, text?, choices?: [{id, text}], selected?}",
    "dialogue.choose": "{index? (from 0) | choice? (id) | close?: true} choose one of what is shown, or close it; -> dialogue.show",
    "hud.inventory": "what the HUD's inventory panel shows: {visible, items: [{id, title, count, tags?}]}",
    "inventory.list": "{actor?} what an actor carries (default player): {items: [{id, title, count, tags?}], known: {id: {title, tags?, opens?}}}",
    "inventory.give": "{item, count?, actor?} give (negative count: take) items as a give / take effect would",
    "inventory.use": "{item, actor?} use an item the actor carries, as the inventory's F does: the rules' use_item event",
    "hud.crossing": "the pedestrian signal the player is at (on the kerb of a signalled crossing, or on it): "
        + "{visible, crossing?, where?: kerb | crossing, state?: walk | flash | dont, text?, line?}",
    "hud.quests": "{track?: quest id | 'next' | 'previous', linger?: s} what the HUD's quest tracker displays: "
        + "visible, tracked, title, status, note, active, lines [{id, text, state, optional, progress, "
        + "progress_text}], waiting, clipped, rect, viewport; track picks the quest first, linger sets how "
        + "long a finished quest stays up",
    "hud.log": "{open?: bool, select?: quest id | package/id | 'next' | 'previous', track?: bool} the HUD's quest log: "
        + "{open, quests, sections: {active, complete, failed}, selected: {.., objectives}, tracked, hint, rect}; "
        + "track makes the tracker follow the selected quest",
    "hud.markers": "{enabled?: bool} the HUD's world markers for the tracked quest's open objectives: "
        + "{visible, tracked, markers: [{quest, objective, subject, kind: point | area | portal, position, screen, "
        + "distance, on_screen, clamped, behind, radius?, inside?, scene?}]}",
    "hud.notifications": "{hold?: s, gap?: s, limit?: n, muted?: bool, clear?: bool} the HUD's toasts: "
        + "visible, showing [{seq, kind: started | objective_done | complete | failed | notice, package, quest, "
        + "objective, heading, text, age}], queued (waiting their turn, oldest first), raised, chimes, "
        + "sounds (what rules played), hold, "
        + "gap, limit, rect, viewport; hold sets how long a toast stays, gap the time between two appearing, "
        + "limit how many are up at once; clear drops them all",
    "npcs.list": "every NPC's report (doing, needs, position, ...)",
    "npcs.options": "{npc} the NPC's scored options now",
    "npcs.force": "{npc, option} make its next decision take that option",
    "npcs.greet": "{npc, wait?: s} greet it (wait: until it can be greeted, up to that long) -> {npc, ok}",
    "npcs.pause": "{npc?} freeze one NPC (or all)",
    "npcs.resume": "{npc?}",
    "time.pause": "freeze the world (the API keeps serving)",
    "time.resume": "",
    "time.step": "{frames} run N physics frames, then pause",
    "time.advance": "{seconds, scale?} run S s of world time (at scale, default 8), return NPC reports",
    "time.simulate": "{seconds} run S s of world time at the current speed and report what it cost: "
        + "{npcs, cost: {npcs, script_us, process_ms, physics_ms, physics_p95_ms, frame_ms, frame_p95_ms, "
        + "frame_max_ms, draw_calls, primitives, objects}, traffic, trains}",
    "time.scale": "{scale} world speed",
    "time.clock": "{time? 'HH:MM', day?, day_length?} get or set the world clock",
    "time.skip": "{until? 'HH:MM' | minutes? | hours? | days?} move the world clock forward (until: the next time it "
        + "shows that); the rules' clock-time waits that come due on the way run in order",
    "capture.screenshot": "{path?, from?, at?, node?, view?, zoom?, isolate?, fov?, size?} PNG from the current camera, a pose (from/at) or framed on a node from a view (iso, front, back, left, right, top); base64 without path",
    "capture.views": "{node? | node_path? | bounds?, views?, zoom?, isolate?, path?} contact sheet of views framed on a node or box",
    "capture.camera": "{mode: player|overview|follow, npc?}",
    "events.subscribe": "{streams: [events, interaction, npc, room, place, travel, traffic], types?, subjects?}: "
        + "`events` is the whole bus (events.gd), narrowed to these types and subject IDs",
    "events.history": "{since?, types?, subjects?} -> {count, events}: the bus's last events (numbered from 1; "
        + "since: only those after that number)",
    "events.unsubscribe": "{streams?}",
}
## extras.geogen keys every mesh part may carry; a node with only these isn't listed by world.query.
const PART_KEYS := ["version", "collider", "walkable"]
const TRACE_SECONDS := 0.25          # walk/move position samples
const STREAMS := ["events", "quest", "interaction", "npc", "room", "place", "travel", "traffic"]

var main: Node                 # main.gd (player, world, overview, clock, switch_scene, ...)
var port := 7878
var _server := TCPServer.new()
var _peers: Array[Dictionary] = []   # {peer: StreamPeerTCP, buffer: String, streams: {}}
var _connected_traffic := {}
var _room := ""
var _walker: GeogenWalker = null
var _world_frames := 0               # physics frames the world ran (not counting paused ones)
## Serve TCP (--api); a script-only run doesn't listen.
var listen := true
## Steps to run once the world is loaded (--script / --run, or flags that expand to steps).
var script_steps: Array = []
var stop_on_error := true
## A script's connection: no socket, events print instead.
var _script_entry := {"peer": null, "buffer": "", "streams": {}}
## While time.simulate runs: what each frame and physics tick cost (see _simulate).
var _sim := {}
var _tick_start := 0


func _ready() -> void:
    process_mode = Node.PROCESS_MODE_ALWAYS   # serve while the world is paused
    # Time every physics tick's callbacks (scripts, move_and_slide; not the physics server's own
    # step) while time.simulate runs: Godot's TIME_PHYSICS_PROCESS monitor is the slowest tick of
    # the last second.
    add_child(_TickMark.new(-100000, func(): _tick_start = Time.get_ticks_usec()))
    add_child(_TickMark.new(100000, func():
        if not _sim.is_empty():
            _sim["ticks"].append((Time.get_ticks_usec() - _tick_start) / 1000.0)))
    if listen:
        var err := _server.listen(port, "127.0.0.1")
        if err != OK:
            push_error("geogen api: can't listen on port %d (%s)" % [port, error_string(err)])
            return
        print("api listening: %s" % JSON.stringify({"port": _server.get_local_port()}))
    if not script_steps.is_empty():
        _run_script.call_deferred()
    var world: WorldLoader = main.world
    world.interaction_event.connect(func(asset: String, interaction: String, state: String, event: String):
        _emit("interaction", {"asset": asset, "interaction": interaction, "state": state, "event": event}))
    world.travel_requested.connect(func(travel: Dictionary): _emit("travel", travel))
    world.events.posted.connect(_on_event)
    world.rules.quest_event.connect(func(event: Dictionary): _emit("quest", event))


## The streams are views of the event bus: `events` is all of it, the others one kind each.
func _on_event(event: Dictionary) -> void:
    match event["type"]:
        "enter", "leave":
            _emit("place", {"type": event["type"], "subject": event["subject"], "place": event["data"]["place"],
                "kind": event["data"]["kind"], "actor": event["actor"]})
        "npc":
            _emit("npc", event["data"])


static func _wanted(event: Dictionary, types: Array, subjects: Array) -> bool:
    return (types.is_empty() or event["type"] in types) and (subjects.is_empty() or event["subject"] in subjects)


func _process(delta: float) -> void:
    if not _sim.is_empty():
        _sim["process"] += Performance.get_monitor(Performance.TIME_PROCESS)
        _sim["samples"] += 1
        _sim["frames"].append(delta * 1000.0 / Engine.time_scale)
        _sim["render"][0] += Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME)
        _sim["render"][1] += Performance.get_monitor(Performance.RENDER_TOTAL_PRIMITIVES_IN_FRAME)
        _sim["render"][2] += Performance.get_monitor(Performance.RENDER_TOTAL_OBJECTS_IN_FRAME)
    if not _server.is_listening():
        return
    while _server.is_connection_available():
        var peer := _server.take_connection()
        peer.set_no_delay(true)
        _peers.append({"peer": peer, "buffer": "", "streams": {}})
    for entry in _peers.duplicate():
        var peer: StreamPeerTCP = entry["peer"]
        peer.poll()
        if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
            main.world.events.unlisten(entry.get("listener", 0))
            _peers.erase(entry)
            continue
        var available := peer.get_available_bytes()
        if available > 0:
            var chunk := peer.get_data(available)
            if chunk[0] == OK:
                entry["buffer"] += (chunk[1] as PackedByteArray).get_string_from_utf8()
        while "\n" in entry["buffer"]:
            var line: String = entry["buffer"].get_slice("\n", 0)
            entry["buffer"] = entry["buffer"].substr(line.length() + 1)
            if line.strip_edges() != "":
                _handle(entry, line)


func _physics_process(delta: float) -> void:
    if not get_tree().paused:
        _world_frames += 1
    _wire_streams()
    if main.player == null:
        return
    var room: String = main.world.room_at(main.player.global_position + Vector3(0, 0.5, 0))
    if room != _room:
        _emit("room", {"from": _room, "to": room, "position": _v(main.player.global_position)})
        _room = room
    if _walker != null and not get_tree().paused:
        _walker.step(delta)


# --- protocol -----------------------------------------------------------------

func _handle(entry: Dictionary, line: String) -> void:
    var msg = JSON.parse_string(line)
    if not msg is Dictionary:
        _send(entry, {"id": null, "ok": false, "error": "not a JSON object: %s" % line.left(200)})
        return
    var id = msg.get("id")
    var cmd := str(msg.get("cmd", ""))
    var args = msg.get("args", {})
    if not args is Dictionary:
        _send(entry, {"id": id, "ok": false, "error": "args must be an object"})
        return
    if not COMMANDS.has(cmd):
        _send(entry, {"id": id, "ok": false, "error": "unknown command '%s' (try help)" % cmd})
        return
    var result = await _run(entry, cmd, args)
    if result is Dictionary and result.has("__error"):
        _send(entry, {"id": id, "ok": false, "error": result["__error"]})
    else:
        _send(entry, {"id": id, "ok": true, "result": result})
    if cmd == "quit":
        get_tree().quit.call_deferred()


func _send(entry: Dictionary, data: Dictionary) -> void:
    if entry["peer"] == null:           # a script: subscribed events print
        print("script event: %s" % JSON.stringify(data, "", true, true))
        return
    var peer: StreamPeerTCP = entry["peer"]
    if peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
        # Full precision: a save (state.save) handed back to state.load must be exactly what was
        # saved, or a timer due on the very frame can come due a frame early (geogen.51).
        peer.put_data((JSON.stringify(data, "", true, true) + "\n").to_utf8_buffer())


## Every connection: the script's, then the sockets'.
func _entries() -> Array:
    var out: Array = [_script_entry]
    out.append_array(_peers)
    return out


func _emit(stream: String, data: Dictionary) -> void:
    for entry in _entries():
        if entry["streams"].has(stream):
            _send(entry, {"event": stream, "data": data})


## Traffic comes and goes with scene switches: wire its trace signals when subscribed.
func _wire_streams() -> void:
    var want_traffic := _entries().any(func(e): return e["streams"].has("traffic"))
    if want_traffic:
        var movers: Array = []
        movers.append_array(main.world.traffic)
        movers.append_array(main.world.trains)
        for t in movers:
            if is_instance_valid(t) and not _connected_traffic.has(t):
                _connected_traffic[t] = true
                t.traced.connect(func(event: Dictionary): _emit("traffic", event))


static func _err(message: String) -> Dictionary:
    return {"__error": message}


static func _v(p: Vector3) -> Array:
    return [snappedf(p.x, 0.001), snappedf(p.y, 0.001), snappedf(p.z, 0.001)]


## [x, y, z] (or [x, z] on the ground) -> Vector3, or null.
static func _vec(value):
    if value is Array and value.size() == 3:
        return Vector3(float(value[0]), float(value[1]), float(value[2]))
    if value is Array and value.size() == 2:
        return Vector3(float(value[0]), 0.0, float(value[1]))
    return null


func _run(entry: Dictionary, cmd: String, args: Dictionary):
    var world: WorldLoader = main.world
    var player: Player = main.player
    match cmd:
        "help":
            return COMMANDS
        "ping":
            return {"pong": true, "frame": _world_frames}
        "wait":
            return await _wait(args)
        "batch":
            var results := []
            for c in args.get("commands", []):
                var sub := str(c.get("cmd", "")) if c is Dictionary else ""
                var sub_args = c.get("args", {}) if c is Dictionary else {}
                var r
                if not COMMANDS.has(sub) or sub in ["batch", "quit"]:
                    r = _err("unknown command '%s'" % sub)
                elif not sub_args is Dictionary:
                    r = _err("args must be an object")
                else:
                    r = await _run(entry, sub, sub_args)
                var failed: bool = r is Dictionary and r.has("__error")
                results.append({"cmd": sub, "ok": not failed, "error": r["__error"]} if failed \
                    else {"cmd": sub, "ok": true, "result": r})
                if failed and bool(args.get("stop_on_error", true)):
                    break
            return results
        "quit":
            return {"quitting": true}
        "wait_ready":
            return await _wait_ready(float(args.get("timeout", 60.0)))
        "world.info":
            return _info()
        "world.scenes":
            return {"default": world.catalogue.get("default", ""), "scenes": world.scene_list()}
        "world.query":
            return _query(args)
        "world.room_at":
            var pos = _vec(args.get("pos"))
            return _err("pos: [x, y, z]") if pos == null else {"room": world.room_at(pos)}
        "world.raycast":
            return _raycast(args)
        "world.nav_path":
            var a = _vec(args.get("from", _v(player.global_position)))
            var b = _vec(args.get("to"))
            if a == null or b == null:
                return _err("from/to: [x, y, z]")
            return _nav_path(a, b)
        "world.switch":
            var scene := str(args.get("scene", ""))
            if not world.is_exported(scene):
                return _err("scene '%s' is not exported" % scene)
            _cancel_walk()
            var ok: bool = await main.switch_scene(scene, str(args.get("spawn", "")))
            if not ok:
                return _err("a switch is already under way")
            _connected_traffic.clear()
            return _info()
        "world.reload":
            var keep := bool(args.get("keep_player", true))
            var pos: Vector3 = player.global_position
            var yaw := rad_to_deg(player.rotation.y)
            _cancel_walk()
            var ok: bool = await main.switch_scene(world.scene_name)
            if not ok:
                return _err("can't reload '%s'" % world.scene_name)
            _connected_traffic.clear()
            if keep:
                player.spawn(pos, yaw)
            return _info()
        "world.node":
            var node := _find_node(args)
            return _err("no node %s" % args.get("node_path", args.get("uid", args.get("name", args.get("node", ""))))) \
                if node == null else _node_info(node)
        "player.status":
            return _status()
        "player.lock":
            main.focus.update()
            var it: GeogenInteraction = main.focus.interaction
            if it == null:
                return _err("nothing lockable in view")
            var toggled: bool = it.toggle_lock(main.held_keys(), "player")
            return {"asset": String(it.asset.name), "interaction": it.interaction_name, "ok": toggled, "locked": it.locked}
        "player.teleport":
            var pos = _vec(args.get("pos"))
            if pos == null:
                return _err("pos: [x, y, z]")
            _cancel_walk()
            if not player.pose.is_empty():
                player.leave_pose()
            player.spawn(pos, float(args.get("yaw", rad_to_deg(player.rotation.y))))
            world.stream_focus = pos
            return _status()
        "player.look":
            _look(args)
            return _status()
        "player.move":
            return await _move(args)
        "player.walk_to":
            return await _walk_to(args)
        "player.use":
            return _use(args)
        "player.sit":
            return _sit(args)
        "player.stand":
            player.leave_pose()
            return _status()
        "player.carry":
            var want := str(args.get("prop", ""))
            if want == "":
                main.focus.update()
                want = main.focus.prop
            else:
                for b in world.props.report():
                    if b["uid"].match(want) or b["node"] == want:
                        want = b["uid"]
                        break
            var ok: bool = want != "" and world.props.pick(want, player, main.focus.hold_point)
            return {"ok": ok, "held": world.props.held}
        "player.drop":
            var dropped := world.props.held
            world.props.drop()
            return {"dropped": dropped}
        "world.bodies":
            return world.props.report()
        "world.lights":
            return _lights()
        "world.skeletons":
            return _skeleton_report()
        "world.stats":
            return _stats()
        "flows.list":
            return world.flows.report()
        "player.keys":
            if args.has("keys"):
                main.keys = Array(args["keys"])
            return {"keys": main.keys, "held": main.held_keys()}
        "interactions.states":
            return world.saves.interaction_states()
        "interactions.list":
            var asset := str(args.get("asset", ""))
            return world.interactions.filter(func(it): return asset == "" or String(it.asset.name) == asset) \
                .map(func(it): return _interaction(it))
        "interactions.set", "interactions.use", "interactions.lock":
            var found := _interactions(args)
            if found.is_empty():
                return _err("no interaction %s on '%s'" % [args.get("interaction", ""), args.get("asset", "")])
            for it in found:
                if cmd == "interactions.set":
                    if not it.states.has(str(args.get("state", ""))):
                        return _err("'%s' has no state '%s' (states: %s)" % [it.interaction_name,
                            args.get("state", ""), ", ".join(it.states.keys())])
                    it.set_state(str(args["state"]), bool(args.get("instant", false)), str(args.get("actor", "")))
                elif cmd == "interactions.use":
                    it.use(str(args.get("actor", "")))
                else:
                    it.set_locked(bool(args.get("locked", true)), str(args.get("actor", "")))
            if cmd != "interactions.lock" and bool(args.get("wait", false)):
                var waited := 0.0
                while found.any(func(i): return i.state != i.target) and waited < 10.0:
                    await get_tree().physics_frame
                    if not get_tree().paused:
                        waited += get_physics_process_delta_time()
                return found.map(func(it): return _interaction(it, true))
            return found.map(func(it): return _interaction(it))
        "state.get":
            var got = world.rules.get_value(str(args.get("key", "")), str(args.get("actor", "player")))
            return _err("no state '%s'" % args.get("key", "")) if got == null else {"value": got}
        "state.set":
            var problem: String = world.rules.set_value(str(args.get("key", "")), args.get("value"),
                str(args.get("actor", "player")))
            return _err(problem) if problem != "" else {"value": world.rules.get_value(str(args["key"]),
                str(args.get("actor", "player")))}
        "state.list":
            return world.rules.stored()
        "quests.list":
            return world.rules.report(str(args.get("actor", "player")))
        "hud.aim":
            main.focus.update()
            var aim: Dictionary = main.hud.crosshair.hud_report()
            aim["focus"] = main.focus.info()
            return aim
        "hud.stats":
            main.hud.refresh()
            return {"stats": main.hud.panel("stats").hud_report(), "banner": main.hud.panel("banner").hud_report()}
        "hud.crossing":
            return main.hud.panel("crossing").hud_report()
        "dialogue.show", "dialogue.choose":
            var panel: GeogenHudDialogue = main.hud.panel("dialogue")
            main.hud.refresh()
            if cmd == "dialogue.choose":
                var shown: Dictionary = panel.hud_report()
                if not shown["open"]:
                    return _err("no conversation is open")
                if args.get("close", false):
                    panel.close()
                else:
                    var index := int(args.get("index", -1))
                    if args.has("choice"):
                        index = shown["choices"].map(func(c): return c["id"]).find(str(args["choice"]))
                    if not panel.choose(index):
                        return _err("no choice %s here (shown: %s)" % [args.get("choice", index),
                            ", ".join(shown["choices"].map(func(c): return c["id"]))])
                main.hud.refresh()
            return panel.hud_report()
        "hud.inventory":
            main.hud.refresh()
            return main.hud.panel("inventory").hud_report()
        "inventory.list":
            return {"items": world.rules.carried(str(args.get("actor", "player"))), "known": world.rules.items}
        "inventory.give":
            var problem: String = world.rules.give(str(args.get("item", "")), int(args.get("count", 1)),
                str(args.get("actor", "player")))
            if problem != "":
                return _err(problem)
            return {"items": world.rules.carried(str(args.get("actor", "player")))}
        "inventory.use":
            var problem: String = world.rules.use_item(str(args.get("item", "")), str(args.get("actor", "player")))
            if problem != "":
                return _err(problem)
            main.hud.refresh()
            return {"items": world.rules.carried(str(args.get("actor", "player")))}
        "hud.quests":
            var tracker: GeogenHudQuestTracker = main.hud.panel("quests")
            if args.has("linger"):
                tracker.linger = float(args["linger"])
            var pick := str(args.get("track", ""))
            if pick in ["next", "previous"]:
                tracker.cycle(1 if pick == "next" else -1)
            elif pick != "" and not tracker.track(pick):
                return _err("'%s' isn't an active quest of the player" % pick)
            if args.has("linger") or pick != "":
                main.hud.refresh()
            return tracker.hud_report()
        "hud.log":
            var quest_log: GeogenHudQuestLog = main.hud.panel("log")
            if args.has("open"):
                quest_log.set_open(bool(args["open"]))
            var chosen := str(args.get("select", ""))
            if chosen in ["next", "previous"]:
                quest_log.step(1 if chosen == "next" else -1)
            elif chosen != "" and not quest_log.select(chosen):
                return _err("'%s' isn't a quest in the log" % chosen)
            if bool(args.get("track", false)) and not quest_log.track_selected():
                return _err("the selected quest isn't active, so it can't be tracked")
            if not args.is_empty():
                main.hud.refresh()
                await get_tree().process_frame      # laid out before its rect is read
                await get_tree().process_frame
            return quest_log.hud_report()
        "hud.markers":
            var marks: GeogenHudMarkers = main.hud.panel("markers")
            if args.has("enabled"):
                marks.enabled = bool(args["enabled"])
            main.hud.refresh()
            return marks.hud_report()
        "hud.notifications":
            var toasts: GeogenHudNotifications = main.hud.panel("notifications")
            if args.has("hold"):
                toasts.hold = maxf(float(args["hold"]), 0.0)
            if args.has("gap"):
                toasts.gap = maxf(float(args["gap"]), 0.0)
            if args.has("limit"):
                toasts.limit = maxi(int(args["limit"]), 1)
            if args.has("muted"):
                toasts.muted = bool(args["muted"])
            if bool(args.get("clear", false)):
                toasts.clear()
            return toasts.hud_report()
        "window.size":
            if args.get("size") is Array and args["size"].size() == 2:
                get_window().size = Vector2i(int(args["size"][0]), int(args["size"][1]))
                await get_tree().process_frame      # layouts follow over the next frames
                await get_tree().process_frame
            return {"size": [get_window().size.x, get_window().size.y]}
        "input.key":
            var code := OS.find_keycode_from_string(str(args.get("key", "")))
            if code == KEY_NONE:
                return _err("key: a key's name (Q, F5, Escape), got '%s'" % args.get("key", ""))
            for pressed in [true, false]:
                var press := InputEventKey.new()
                press.keycode = code
                press.physical_keycode = code
                press.pressed = pressed
                Input.parse_input_event(press)
            await get_tree().process_frame          # delivered with the next frame's input
            await get_tree().process_frame
            return {"key": OS.get_keycode_string(code)}
        "input.mouse":
            var at = args.get("at")
            if not at is Array or at.size() != 2:
                return _err("at: [x, y] in viewport pixels")
            var buttons := {"left": MOUSE_BUTTON_LEFT, "right": MOUSE_BUTTON_RIGHT, "middle": MOUSE_BUTTON_MIDDLE}
            var button := str(args.get("button", "left"))
            if not buttons.has(button):
                return _err("button: left | right | middle, got '%s'" % button)
            var where := Vector2(float(at[0]), float(at[1]))
            var presses := [false, true] if bool(args.get("double", false)) else [false]
            for double in presses:
                if double:
                    await get_tree().process_frame  # a person's second click comes frames later
                for pressed in [true, false]:
                    var click := InputEventMouseButton.new()
                    click.button_index = buttons[button]
                    click.position = where
                    click.global_position = where
                    click.pressed = pressed
                    click.double_click = double and pressed
                    click.button_mask = buttons[button] if pressed else 0
                    get_viewport().push_input(click)
            await get_tree().process_frame
            await get_tree().process_frame
            return {"at": [where.x, where.y], "button": button, "double": bool(args.get("double", false))}
        "state.save":
            var saved := world.saves.save()
            if args.has("path"):
                var problem := GeogenWorldSaves.write(str(args["path"]), saved)
                if problem != "":
                    return _err(problem)
            return saved
        "state.load":
            var state = args.get("state")
            if args.has("path"):
                state = GeogenWorldSaves.read(str(args["path"]))
            if not state is Dictionary:
                return _err("state: a save (state.save), or path: a file holding one")
            world.saves.load(state)
            return world.saves.save()
        "npcs.list":
            return world.npcs.filter(func(n): return is_instance_valid(n)).map(func(n): return n.report())
        "npcs.options", "npcs.force", "npcs.greet":
            var npc := _npc(str(args.get("npc", "")))
            if npc == null:
                return _err("no NPC named '%s'" % args.get("npc", ""))
            if cmd == "npcs.options":
                return npc.options().map(func(o): return {"id": o["id"], "action": o["action"],
                    "score": snappedf(o["score"], 0.001), "at": _v(o["at"])})
            if cmd == "npcs.greet":
                var waited := 0.0       # wait: until it can be greeted (not mid-greeting, in reach of nothing else)
                while not npc.can_greet() and waited < float(args.get("wait", 0.0)):
                    await get_tree().physics_frame
                    waited += get_physics_process_delta_time()
                return {"npc": String(npc.name), "ok": npc.greet()}
            var option := str(args.get("option", ""))
            if not npc.force_option(option):
                return _err("'%s' is not one of %s's options now" % [option, npc.name])
            return {"npc": String(npc.name), "forced": option}
        "npcs.pause", "npcs.resume":
            var names := []
            for npc in world.npcs:
                if is_instance_valid(npc) and String(npc.name).begins_with(str(args.get("npc", ""))):
                    npc.process_mode = Node.PROCESS_MODE_DISABLED if cmd == "npcs.pause" else Node.PROCESS_MODE_INHERIT
                    names.append(String(npc.name))
            return {"npcs": names}
        "time.pause", "time.resume":
            get_tree().paused = cmd == "time.pause"
            return _time()
        "time.step":
            var frames := maxi(int(args.get("frames", 1)), 1)
            # Pause again at the start of the frame after the Nth (physics_frame fires before nodes run).
            var until := _world_frames + frames
            get_tree().paused = false
            while _world_frames < until:
                await get_tree().physics_frame
            get_tree().paused = true
            return _time()
        "time.advance":
            return await _advance(float(args.get("seconds", 10.0)), float(args.get("scale", 8.0)))
        "time.simulate":
            return await _simulate(float(args.get("seconds", 10.0)))
        "time.scale":
            _set_scale(float(args.get("scale", 1.0)))
            return _time()
        "time.clock":
            var clock: GeogenClock = main.clock
            if args.has("time"):
                clock.hours = GeogenClock.parse(str(args["time"]))
                clock.apply()
            if args.has("day"):
                clock.day = int(args["day"])
            if args.has("day_length"):
                clock.day_length = float(args["day_length"])
            return _time()
        "time.skip":
            var clock: GeogenClock = main.clock
            var start := clock.day * 1440.0 + clock.hours * 60.0
            var target := start + float(args.get("minutes", 0.0)) + float(args.get("hours", 0.0)) * 60.0 \
                + float(args.get("days", 0.0)) * 1440.0
            if args.has("until"):
                target = start - fmod(start, 1440.0) + GeogenClock.parse(str(args["until"])) * 60.0
                if target <= start + 0.001:
                    target += 1440.0
            if target <= start:
                return {"error": "time.skip needs until, minutes, hours or days ahead"}
            clock.day = int(target / 1440.0)
            clock.hours = fmod(target, 1440.0) / 60.0
            clock.apply()
            main.world.rules._sync_clock()
            return _time()
        "capture.screenshot":
            return await _screenshot(args)
        "capture.views":
            return await _views(args)
        "capture.camera":
            return _camera(args)
        "events.subscribe", "events.unsubscribe":
            var streams: Array = args.get("streams", STREAMS if cmd == "events.unsubscribe" else [])
            for s in streams:
                if not s in STREAMS:
                    return _err("unknown stream '%s' (%s)" % [s, ", ".join(STREAMS)])
                if cmd == "events.subscribe":
                    entry["streams"][s] = true
                else:
                    entry["streams"].erase(s)
            if "events" in streams:                 # the bus does the filtering: one listener per peer
                main.world.events.unlisten(entry.get("listener", 0))
                if cmd == "events.subscribe":
                    entry["listener"] = main.world.events.listen(
                        func(event: Dictionary): _send(entry, {"event": "events", "data": event}),
                        args.get("types", []), args.get("subjects", []))
            return {"streams": entry["streams"].keys()}
        "events.history":
            var bus: GeogenEvents = main.world.events
            var first := bus.count - bus.history.size() + 1          # the number of history[0]
            var out := []
            for index in bus.history.size():
                if first + index > int(args.get("since", 0)) \
                        and _wanted(bus.history[index], args.get("types", []), args.get("subjects", [])):
                    out.append(bus.history[index])
            return {"count": bus.count, "events": out}
    return _err("unhandled command '%s'" % cmd)


# --- world --------------------------------------------------------------------

func _info() -> Dictionary:
    var world: WorldLoader = main.world
    var aabb := world.world_aabb()
    return {"scene": world.scene_name, "loading": world.loading,
        "bounds": {"min": _v(aabb.position), "max": _v(aabb.end)},
        "spawns": world.spawns.map(func(s): return {"name": s.get("name", ""), "position": _v(s["position"]),
            "yaw": s["yaw_deg"]}),
        "rooms": world.rooms.map(func(r): return {"id": r["id"], "type": r["type"],
            "center": _v((r["xform"] as Transform3D).origin), "size": _v(r["size"])}),
        "places": world.places.filter(func(p): return p["kind"] == "place").map(func(p): return {"id": p["id"],
            "uid": p["uid"], "position": _v((p["node"] as Node3D).global_position), "actors": p["actors"].keys()}),
        "npcs": world.npcs.filter(func(n): return is_instance_valid(n)).map(func(n): return String(n.name)),
        "interactions": world.interactions.size(), "affordances": world.affordances.size(),
        "player_spec": main.player_spec.to_dict(), "time": _time(), "stream": world.stream_report()}


func _time() -> Dictionary:
    var clock: GeogenClock = main.clock
    return {"paused": get_tree().paused, "scale": Engine.time_scale, "frame": _world_frames,
        "clock": clock.label(), "hours": snappedf(clock.hours, 0.0001), "day": clock.day, "day_length": clock.day_length,
        "night": main.world.night}


## Whether the navmesh around the player answers queries (the playtest's sync test).
func _nav_ready() -> bool:
    var map: RID = main.get_world_3d().navigation_map
    if NavigationServer3D.map_get_iteration_id(map) <= 0:
        return false
    if main.world.streamer != null and main.world.streamer.navigation_busy():
        return false
    var at: Vector3 = main.player.global_position
    var near := NavigationServer3D.map_get_closest_point(map, at)
    return Vector2(near.x - at.x, near.z - at.z).length() < 1.0


## Ready: loaded, not switching, the navmesh answers near the player, and the navigation map
## has stopped changing (many baked regions merge over several map iterations): 20 physics
## frames on one iteration, or 5 once a streamed world's tiles are all baked.
func _wait_ready(timeout: float) -> Dictionary:
    var started := Time.get_ticks_msec()
    var map: RID = main.get_world_3d().navigation_map
    var iteration := -1
    var settled := 0
    while true:
        var now := NavigationServer3D.map_get_iteration_id(map)
        if now != iteration:
            iteration = now
            settled = 0
        var ready: bool = main.player != null and not main.world.loading and not main.is_switching() and _nav_ready()
        settled = settled + 1 if ready else 0
        if settled >= (5 if main.world.streamer != null else 20):
            break
        if Time.get_ticks_msec() - started > timeout * 1000.0:
            return _err("not ready after %.0f s (loading %s, nav %s)" % [timeout, main.world.loading, _nav_ready()])
        await get_tree().physics_frame
    return _info()


func _query(args: Dictionary) -> Array:
    var tag := str(args.get("tag", ""))
    var type := str(args.get("type", ""))
    var pattern := str(args.get("name", ""))
    var uid := str(args.get("uid", ""))
    var room := str(args.get("room", ""))
    var interactive := bool(args.get("interactive", false))
    var limit := int(args.get("limit", 200))
    var world: WorldLoader = main.world
    var out := []
    for node: Node3D in world.find_children("*", "Node3D", true, false):
        var g := WorldLoader.geogen_extras(node)
        if not _is_object(g):
            continue
        if tag != "" and not (g.get("tags", []) as Array).any(func(t): return t == tag or str(t).begins_with(tag + ".")):
            continue
        if type != "" and str(g.get("type", "")) != type:
            continue
        if pattern != "" and not String(node.name).matchn(pattern):
            continue
        if uid != "" and not str(g.get("uid", "")).match(uid):
            continue
        if interactive and not g.has("interactions"):
            continue
        var pos := node.global_position
        if room != "" and world.room_at(pos + Vector3(0, 0.3, 0)) != room:
            continue
        out.append({"name": String(node.name), "uid": str(g.get("uid", "")), "node_path": String(world.get_path_to(node)), "position": _v(pos),
            "yaw": snappedf(rad_to_deg(node.global_rotation.y), 0.1), "room": world.room_at(pos + Vector3(0, 0.3, 0)),
            "extras": g})
        if out.size() >= limit:
            break
    return out


func _raycast(args: Dictionary):
    var from = _vec(args.get("from"))
    var to = _vec(args.get("to"))
    if from == null or to == null:
        return _err("from/to: [x, y, z]")
    var query := PhysicsRayQueryParameters3D.create(from, to)
    query.exclude = [main.player.get_rid()]
    var hit: Dictionary = main.get_world_3d().direct_space_state.intersect_ray(query)
    if hit.is_empty():
        return {"hit": false}
    var collider: Node = hit["collider"]
    var it: GeogenInteraction = main.world.interaction_for(collider)
    return {"hit": true, "position": _v(hit["position"]), "normal": _v(hit["normal"]),
        "distance": snappedf((hit["position"] as Vector3).distance_to(from), 0.001),
        "node": String(collider.name), "node_path": String(main.world.get_path_to(collider)) if main.world.is_ancestor_of(collider) else String(collider.get_path()),
        "asset": _asset_of(collider), "room": main.world.room_at(hit["position"] + Vector3(0, 0.05, 0)),
        "interaction": _interaction(it) if it != null else null,
        "npc": String(collider.name) if collider is GeogenNpc else null}


## Name of the nearest ancestor that is a gameplay object (the placed asset, a room...), or "".
func _asset_of(node: Node) -> String:
    while node != null and node != main.world:
        if _is_object(WorldLoader.geogen_extras(node)):
            return String(node.name)
        node = node.get_parent()
    return ""


## Extras that say more than how a mesh part collides: tags, type, interactions, affordances...
static func _is_object(extras: Dictionary) -> bool:
    return extras.keys().any(func(k): return not k in PART_KEYS)


func _nav_path(a: Vector3, b: Vector3) -> Dictionary:
    var map: RID = main.get_world_3d().navigation_map
    var from := NavigationServer3D.map_get_closest_point(map, a)
    var to := NavigationServer3D.map_get_closest_point(map, b)
    var path := GeogenWalker.find_path(map, from, to)
    var length := 0.0
    for i in range(1, path.size()):
        length += path[i].distance_to(path[i - 1])
    return {"length": snappedf(length, 0.001), "reached": path.size() > 0 and path[-1].distance_to(to) < 0.05,
        "start_gap": snappedf(from.distance_to(a), 0.001), "end_gap": snappedf(to.distance_to(b), 0.001),
        "points": Array(path).map(func(p): return _v(p))}


# --- player -------------------------------------------------------------------

func _status() -> Dictionary:
    var player: Player = main.player
    var p := player.global_position
    main.focus.update()
    var out := {"position": _v(p), "yaw": snappedf(rad_to_deg(wrapf(player.rotation.y, -PI, PI)), 0.1),
        "pitch": snappedf(rad_to_deg(player.head.rotation.x), 0.1),
        "eye": _v(player.camera.global_position), "on_floor": player.is_on_floor(), "flying": player.flying,
        "room": main.world.room_at(p + Vector3(0, 0.5, 0)), "scene": main.world.scene_name,
        "pose": player.pose.get("type", "stand"), "pose_asset": player.pose.get("asset", ""),
        "focus": main.focus.info(), "keys": main.held_keys(),
        "walking": _walker != null and _walker.status == "walking"}
    if main.world.streamer != null:
        out["stream"] = main.world.stream_report()
    return out


func _look(args: Dictionary) -> void:
    var player: Player = main.player
    var at = _vec(args.get("at"))
    if at != null:
        var d: Vector3 = at - player.camera.global_position
        player.rotation.y = atan2(-d.x, -d.z)
        player.head.rotation.x = atan2(d.y, Vector2(d.x, d.z).length())
        return
    if args.has("yaw"):
        player.rotation.y = deg_to_rad(float(args["yaw"]))
    if args.has("pitch"):
        player.head.rotation.x = deg_to_rad(clampf(float(args["pitch"]), -89.0, 89.0))


func _move(args: Dictionary) -> Dictionary:
    var player: Player = main.player
    _cancel_walk()
    var seconds := float(args.get("seconds", 1.0))
    var start := player.global_position
    player.scripted_move = Vector2(float(args.get("strafe", 0.0)), float(args.get("forward", 1.0)))
    var motion := {"trace": [_v(start)], "blocked_by": null, "elapsed": 0.0, "next_sample": TRACE_SECONDS}
    while motion["elapsed"] < seconds and not main.is_switching():
        await get_tree().physics_frame
        _track(motion)
    player.scripted_move = null
    # Walked into a travel point: the walk ends there; the result is where the trip lands.
    while main.is_switching():
        await get_tree().physics_frame
    var result := _status()
    result.merge({"moved": snappedf(player.global_position.distance_to(start), 0.001),
        "trace": motion["trace"], "blocked_by": motion["blocked_by"]}, true)
    return result


func _walk_to(args: Dictionary):
    var target = _vec(args.get("pos"))
    if target == null:
        return _err("pos: [x, y, z]")
    var player: Player = main.player
    if not player.pose.is_empty():
        player.leave_pose()
    _cancel_walk()
    var route := _nav_path(player.global_position, target)
    var walker := _walk_along(route)
    var timeout := float(args.get("timeout", 30.0))
    var motion := {"trace": [_v(player.global_position)], "blocked_by": null, "elapsed": 0.0,
        "next_sample": TRACE_SECONDS}
    var doors := []           # the doors it opened on the way
    var status := ""
    while walker.status == "walking":
        await get_tree().physics_frame
        _track(motion)
        if motion["elapsed"] > timeout:
            walker.cancel()
            status = "timeout"
            break
        # A closed door just ahead: stop clear of its swing, open it (unlocking it with a key
        # the player holds, as L would), wait for it, and walk on, as a player would.
        var door: Dictionary = main.world.closed_portal_on(walker.path, player.global_position, walker.waypoint,
            _door_reach(null))
        if door.is_empty() or door["distance"] > _door_reach(door["portal"]):
            continue
        walker.cancel()
        var it: GeogenInteraction = door["interaction"]
        var info := {"asset": String(it.asset.name), "interaction": it.interaction_name}
        if it.locked and not it.toggle_lock(main.held_keys(), "player"):
            status = "locked"
        elif it.target != door["portal"]["open"] and not it.use("player"):
            status = "refused"
        if status != "":
            motion["blocked_by"] = {"node": String(it.asset.name), "asset": String(it.asset.name), "point": _v(door["hit"]),
                "interaction": {"name": it.interaction_name, "state": it.state, "asset": String(it.asset.name)}}
            break
        var waited := 0.0
        while it.state != door["portal"]["open"] and waited < 5.0:
            await get_tree().physics_frame
            _track(motion)
            waited += get_physics_process_delta_time()
        doors.append(info)
        route = _nav_path(player.global_position, target)
        walker = _walk_along(route)            # through the doorway straight (through_portals)
    if status == "":
        status = walker.status
    if _walker == walker:
        _walker = null
    var p := player.global_position
    var result := _status()
    result.merge({"status": status, "reached": route["reached"] and status == "arrived",
        "distance": snappedf(Vector2(p.x - target.x, p.z - target.z).length(), 0.001),
        "path_length": route["length"], "path": route["points"], "seconds": snappedf(motion["elapsed"], 0.01),
        "trace": motion["trace"], "doors": doors,
        "blocked_by": motion["blocked_by"] if status != "arrived" else null}, true)
    return result


## How near a closed door walk_to comes before opening it: clear of its leaf's swing (the
## portal's `clearance`), and a little more; with no portal, the furthest any door asks.
func _door_reach(portal) -> float:
    if portal != null:
        return float(portal["clearance"]) + 0.3
    var reach := 1.2
    for p in main.world.portals:
        reach = maxf(reach, float(p["clearance"]) + 0.3)
    return reach


func _walk_along(route: Dictionary) -> GeogenWalker:
    var path := PackedVector3Array()
    for p in route["points"]:
        path.append(Vector3(p[0], p[1], p[2]))
    path = main.world.through_portals(path)
    _walker = GeogenWalker.new(main.player, GeogenWalker.along_links(path, get_tree()))
    return _walker


## Per physics frame of a scripted motion: world time, a position sample every TRACE_SECONDS,
## and the last thing the body ran into (a wall, not the floor it stands on).
func _track(motion: Dictionary) -> void:
    if get_tree().paused:
        return
    var player: Player = main.player
    motion["elapsed"] += get_physics_process_delta_time()
    if motion["elapsed"] >= motion["next_sample"]:
        motion["next_sample"] += TRACE_SECONDS
        motion["trace"].append(_v(player.global_position))
    for i in player.get_slide_collision_count():
        var hit := player.get_slide_collision(i)
        if hit.get_normal().y > 0.7:
            continue
        var collider := hit.get_collider() as Node
        if collider == null:
            continue
        var it: GeogenInteraction = main.world.interaction_for(collider)
        motion["blocked_by"] = {"node": String(collider.name), "asset": _asset_of(collider),
            "point": _v(hit.get_position()), "normal": _v(hit.get_normal()),
            "interaction": {"name": it.interaction_name, "state": it.state, "asset": String(it.asset.name)}
                if it != null else null}


func _cancel_walk() -> void:
    if _walker != null:
        _walker.cancel()
        _walker = null


func _use(args: Dictionary):
    if not args.has("asset"):
        var used: Dictionary = main.focus.press()
        return _err("nothing usable in view") if used.is_empty() else used
    var found := _interactions(args)
    if found.is_empty():
        return _err("no interaction %s on '%s'" % [args.get("interaction", ""), args["asset"]])
    var results := []
    for it in found:
        var ok: bool = it.use("player")
        var info := _interaction(it)
        info["ok"] = ok
        results.append(info)
    return {"used": results}


func _sit(args: Dictionary):
    var player: Player = main.player
    var asset := str(args.get("asset", ""))
    var best := {}
    var best_d := INF
    for a in main.world.affordances:
        if not a["type"] in ["sit", "lie"] or not is_instance_valid(a.get("node")):
            continue
        if asset != "" and String(a["node"].name) != asset and str(a.get("asset", "")) != asset:
            continue
        var d := (a["position"] as Vector3).distance_to(player.global_position)
        if d < best_d:
            best_d = d
            best = a
    if best.is_empty():
        return _err("no sit/lie affordance%s" % (" on '%s'" % asset if asset != "" else ""))
    _cancel_walk()
    player.leave_pose()
    player.take_pose(best)
    return _status()


# --- interactions and NPCs ----------------------------------------------------

func _interactions(args: Dictionary) -> Array[GeogenInteraction]:
    var found: Array[GeogenInteraction] = main.world.interactions_of(str(args.get("asset", "")))
    var name := str(args.get("interaction", ""))
    if name != "":
        found = found.filter(func(it): return it.interaction_name == name)
    return found


func _interaction(it: GeogenInteraction, parts := false) -> Dictionary:
    var info := {"asset": String(it.asset.name), "interaction": it.interaction_name, "state": it.state,
        "target": it.target, "moving": it.state != it.target, "locked": it.locked, "lock_key": it.lock_key,
        "states": it.states.keys(), "prompt": it.prompt(main.keys), "refused": it.refusal(), "position": _v(it.asset.global_position),
        "room": main.world.room_at(it.asset.global_position + Vector3(0, 0.3, 0))}
    if parts:
        # Where the moving parts are now (e.g. does an open leaf clear the wardrobe beside it?).
        info["parts"] = it.moving_nodes().map(func(n): return {"name": String(n.name), "bounds": _box(_bounds(n))})
        # What the player aims at to use it.
        info["targets"] = it.targets.filter(func(n): return is_instance_valid(n)).map(
            func(n): return {"name": String(n.name), "bounds": _box(_bounds(n))})
    return info


func _npc(prefix: String) -> GeogenNpc:
    for npc in main.world.npcs:
        if is_instance_valid(npc) and String(npc.name) == prefix:
            return npc
    for npc in main.world.npcs:
        if is_instance_valid(npc) and prefix != "" and String(npc.name).begins_with(prefix):
            return npc
    return null


# --- capture ------------------------------------------------------------------

## Screenshot. Without from/at/node it's what the current camera sees; with them a separate
## camera (the player and overlay are left alone) looks from ``from`` at ``at``, or frames ``node``
## from ``view``. ``path`` saves a PNG, else the PNG comes back base64.
func _screenshot(args: Dictionary):
    var image = await _capture(args)
    if image is Dictionary:
        return image   # an error
    return _deliver(image, str(args.get("path", "")))


## Contact sheet: ``views`` (default iso, front, right, top) framed on ``node`` or ``bounds``.
func _views(args: Dictionary):
    var views: Array = args.get("views", ["iso", "front", "right", "top"])
    var images: Array[Image] = []
    for view in views:
        var shot := args.duplicate()
        shot["view"] = view
        var image = await _capture(shot)
        if image is Dictionary:
            return image
        images.append(image)
    var w := images[0].get_width()
    var h := images[0].get_height()
    var cols := int(ceil(sqrt(float(images.size()))))
    var rows := int(ceil(float(images.size()) / cols))
    var sheet := Image.create(w * cols, h * rows, false, Image.FORMAT_RGBA8)
    for i in images.size():
        var image := images[i]
        image.convert(Image.FORMAT_RGBA8)
        sheet.blit_rect(image, Rect2i(0, 0, w, h), Vector2i((i % cols) * w, (i / cols) * h))
    var result = _deliver(sheet, str(args.get("path", "")))
    if result is Dictionary and not result.has("__error"):
        result["views"] = views
    return result


func _deliver(image: Image, path: String):
    if path != "":
        var err := image.save_png(path)
        if err != OK:
            return _err("can't save %s: %s" % [path, error_string(err)])
        return {"path": path, "width": image.get_width(), "height": image.get_height()}
    return {"png_base64": Marshalls.raw_to_base64(image.save_png_to_buffer()),
        "width": image.get_width(), "height": image.get_height()}


const VIEW_DIRECTIONS := {
    "iso": Vector3(0.7, 0.55, 1.0), "iso_back": Vector3(-0.7, 0.55, -1.0),
    "front": Vector3(0, 0.12, 1), "back": Vector3(0, 0.12, -1),
    "right": Vector3(1, 0.12, 0), "left": Vector3(-1, 0.12, 0), "top": Vector3(0, 1, 0),
}


## One frame as an Image (or an error Dictionary).
func _capture(args: Dictionary):
    if DisplayServer.get_name() == "headless":
        return _err("headless runs don't render: launch the runtime with a window to take screenshots")
    if args.get("size") is Array:
        get_window().size = Vector2i(int(args["size"][0]), int(args["size"][1]))
    var viewport := get_viewport()
    var previous := viewport.get_camera_3d()
    var camera: Camera3D = null
    var from = _vec(args.get("from"))
    var at = _vec(args.get("at"))
    var node: Node3D = _find_node(args) if args.has("node") or args.has("node_path") else null
    if args.has("node") and node == null:
        return _err("no node %s" % args["node"])
    if from != null or node != null or args.has("bounds"):
        camera = Camera3D.new()
        camera.fov = float(args.get("fov", 50.0))
        camera.near = 0.05
        camera.far = 2000.0
        main.add_child(camera)
        if node != null or args.has("bounds"):
            var box := AABB()
            if node != null:
                box = _bounds(node)
            else:
                var lo: Variant = _vec(args["bounds"].get("min"))
                var hi: Variant = _vec(args["bounds"].get("max"))
                if lo == null or hi == null:
                    camera.queue_free()
                    return _err("bounds: {min: [x, y, z], max: [x, y, z]}")
                box = AABB(lo as Vector3, (hi as Vector3) - (lo as Vector3))
            var view := str(args.get("view", "iso"))
            if not VIEW_DIRECTIONS.has(view):
                camera.queue_free()
                return _err("view: %s" % ", ".join(VIEW_DIRECTIONS.keys()))
            var dir: Vector3 = VIEW_DIRECTIONS[view].normalized()
            var radius := maxf(box.size.length() / 2.0, 0.1)
            var distance := radius / sin(deg_to_rad(camera.fov) / 2.0) / maxf(float(args.get("zoom", 1.0)), 0.01)
            var target: Vector3 = box.get_center() if at == null else at
            camera.look_at_from_position(target + dir * distance, target,
                Vector3(0, 0, -1) if view == "top" else Vector3.UP)
        else:
            camera.look_at_from_position(from, at if at != null else from + Vector3(0, 0, -1))
        camera.make_current()
    var overlay: CanvasLayer = main.get_node("Overlay")
    var overlay_was := overlay.visible
    overlay.visible = false
    # isolate: only the node is drawn (and the ground), so walls and roofs don't hide it.
    var hidden: Array[Node3D] = []
    if node != null and bool(args.get("isolate", false)):
        for vi: VisualInstance3D in main.find_children("*", "VisualInstance3D", true, false):
            if vi.visible and not node.is_ancestor_of(vi) and vi != node and main.world.is_ancestor_of(vi) \
                    and not vi is Light3D:
                vi.visible = false
                hidden.append(vi)
        if main.player != null:
            for npc in main.world.npcs:
                if is_instance_valid(npc) and npc.visible and not node.is_ancestor_of(npc):
                    npc.visible = false
                    hidden.append(npc)
    for i in 3:   # the new camera's frame, with shadows settled
        await RenderingServer.frame_post_draw
    var image := viewport.get_texture().get_image()
    overlay.visible = overlay_was
    for n in hidden:
        n.visible = true
    if camera != null:
        if previous != null:
            previous.make_current()
        camera.queue_free()
    if image == null or image.is_empty():
        return _err("the viewport returned no image")
    return image


func _camera(args: Dictionary):
    var mode := str(args.get("mode", "player"))
    match mode:
        "player":
            main.set_follow(null)
            main.player.camera.make_current()
        "overview":
            main.set_follow(null)
            main.overview.make_current()
        "follow":
            if _npc(str(args.get("npc", ""))) == null:
                return _err("no NPC named '%s'" % args.get("npc", ""))
            main.set_follow(str(args.get("npc", "")))
            main.overview.make_current()
        _:
            return _err("mode: player, overview or follow")
    return {"mode": mode}


# --- nodes --------------------------------------------------------------------

## A node by ``node_path`` under the world ("cottage/world/cottage/door", as query results give
## it) or by ``name`` / ``node`` (a name or glob; the first match, gameplay objects first).
func _find_node(args: Dictionary) -> Node3D:
    var world: WorldLoader = main.world
    if args.has("node_path"):
        return world.get_node_or_null(NodePath(str(args["node_path"]))) as Node3D
    if args.has("uid"):          # a stable ID (extras.geogen.uid); the scene's package may be left off
        return world.node_by_uid(str(args["uid"])) as Node3D
    var pattern := str(args.get("name", args.get("node", "")))
    if pattern == "":
        return null
    var best: Node3D = null
    for node: Node3D in world.find_children(pattern, "Node3D", true, false):
        if _is_object(WorldLoader.geogen_extras(node)):
            return node          # prefer the gameplay object over a same-named part
        if best == null:
            best = node
    return best


## World bounds of the visible meshes under (and at) a node.
static func _bounds(node: Node) -> AABB:
    var box := AABB()
    var first := true
    var meshes: Array = node.find_children("*", "MeshInstance3D", true, false)
    if node is MeshInstance3D:
        meshes.append(node)
    for mi: MeshInstance3D in meshes:
        if mi.is_in_group("geogen_collider_debug") or not mi.is_visible_in_tree():
            continue
        var b: AABB = mi.global_transform * mi.get_aabb()
        box = b if first else box.merge(b)
        first = false
    if first and node is Node3D:
        box = AABB((node as Node3D).global_position, Vector3.ZERO)
    return box


static func _box(box: AABB) -> Dictionary:
    return {"min": _v(box.position), "max": _v(box.end), "size": _v(box.size), "center": _v(box.get_center())}


func _node_info(node: Node3D) -> Dictionary:
    var world: WorldLoader = main.world
    var xform := node.global_transform
    var colliders := []
    for shape: CollisionShape3D in node.find_children("*", "CollisionShape3D", true, false):
        if shape.shape == null:
            continue
        var debug := shape.shape.get_debug_mesh()
        colliders.append({"node": String(shape.get_parent().name), "shape": shape.shape.get_class(),
            "disabled": shape.disabled, "bounds": _box(shape.global_transform * debug.get_aabb()) if debug else null})
    var children := []
    for child in node.get_children():
        if child is Node3D and not child is CollisionObject3D:
            children.append({"name": String(child.name), "type": child.get_class(),
                "object": _is_object(WorldLoader.geogen_extras(child))})
    return {"name": String(node.name), "uid": WorldLoader.uid_of(node),
        "node_path": String(world.get_path_to(node)), "type": node.get_class(),
        "position": _v(xform.origin), "rotation_deg": _v(xform.basis.get_euler() * (180.0 / PI)),
        "scale": _v(xform.basis.get_scale()), "visible": node.is_visible_in_tree(),
        "bounds": _box(_bounds(node)), "room": world.room_at(xform.origin + Vector3(0, 0.3, 0)),
        "asset": _asset_of(node), "extras": WorldLoader.geogen_extras(node),
        "interactions": world.interactions_of(String(node.name)).map(func(it): return _interaction(it, true)),
        "colliders": colliders.slice(0, 50), "collider_count": colliders.size(),
        "children": children.slice(0, 100), "child_count": children.size()}


# --- time ---------------------------------------------------------------------

func _set_scale(scale: float) -> void:
    scale = maxf(scale, 0.01)
    Engine.time_scale = scale
    main.world.apply_physics_rate()


## Run ``seconds`` of world time fast, then restore the speed (and pause, if it was paused).
func _advance(seconds: float, scale: float) -> Dictionary:
    var was_paused := get_tree().paused
    var was_scale := Engine.time_scale
    _set_scale(scale)
    get_tree().paused = false
    var elapsed := 0.0
    var started := Time.get_ticks_msec()
    while elapsed < seconds:
        await get_tree().physics_frame
        elapsed += get_physics_process_delta_time()
    _set_scale(was_scale)
    get_tree().paused = was_paused
    return {"advanced": snappedf(elapsed, 0.01), "real_ms": Time.get_ticks_msec() - started, "time": _time(),
        "player": _status(), "npcs": main.world.npcs.filter(func(n): return is_instance_valid(n)).map(
            func(n): return n.report())}


# --- scripts ------------------------------------------------------------------

## Run ``script_steps`` in order, printing each reply; then quit, unless the API also serves.
func _run_script() -> void:
    await get_tree().physics_frame      # the player is placed and the bodies are in the physics space
    var failed := false
    var index := 0
    for step in script_steps:
        var cmd := str(step.get("cmd", "")) if step is Dictionary else ""
        var args = step.get("args", {}) if step is Dictionary else {}
        var line := {"step": index, "cmd": cmd}
        if step is Dictionary and step.has("id"):
            line["id"] = step["id"]
        var r
        if not COMMANDS.has(cmd):
            r = _err("unknown command '%s' (try help)" % cmd)
        elif not args is Dictionary:
            r = _err("args must be an object")
        else:
            # retry: a step that fails is tried again each physics frame for up to that many
            # seconds of world time (the aimed thing isn't in view yet, an NPC can't be greeted yet).
            var retry := float(step.get("retry", 0.0))
            var waited := 0.0
            while true:
                r = await _run(_script_entry, cmd, args)
                if not (r is Dictionary and r.has("__error")) or waited >= retry:
                    break
                await get_tree().physics_frame
                waited += get_physics_process_delta_time()
        var ok: bool = not (r is Dictionary and r.has("__error"))
        line["ok"] = ok
        if ok:
            line["result"] = r
        else:
            line["error"] = r["__error"]
        print("script: %s" % JSON.stringify(line, "", true, true))
        index += 1
        if not ok:
            failed = true
            if stop_on_error:
                break
        if cmd == "quit":
            break
    print("script done: %s" % JSON.stringify({"steps": index, "ok": not failed}))
    if not listen or (script_steps.size() > 0 and str(script_steps[-1].get("cmd", "")) == "quit"):
        get_tree().quit(1 if failed else 0)


## Steps from a script file or text: a JSON list, or {"steps": [...], "stop_on_error": bool}.
## Returns the steps, or a string saying what is wrong.
static func parse_script(text: String, where: String):
    var data = JSON.parse_string(text)
    if data is Dictionary and data.get("steps") is Array:
        return data
    if data is Array:
        return {"steps": data}
    return "%s: a script is a JSON list of {cmd, args} steps, or {steps: [...]}" % where


func _wait(args: Dictionary) -> Dictionary:
    var frames := int(args.get("frames", 0))
    var seconds := float(args.get("seconds", 0.0))
    var elapsed := 0.0
    var count := 0
    while (frames > 0 and count < frames) or (frames <= 0 and elapsed < seconds):
        await get_tree().physics_frame
        count += 1
        if not get_tree().paused:
            elapsed += get_physics_process_delta_time()
    return {"waited": snappedf(elapsed, 0.001), "frames": count, "time": _time()}


## Run ``seconds`` of world time at the current speed and measure what it cost: the NPCs'
## script time per physics frame, the engine's process and physics time per frame (animation is
## in process_ms) and the whole frame. Compare runs with and without NPCs; frame_ms needs a
## window and --disable-vsync.
func _simulate(seconds: float) -> Dictionary:
    _sim = {"process": 0.0, "samples": 0, "render": [0.0, 0.0, 0.0], "frames": [], "ticks": []}
    var started := Time.get_ticks_usec()
    var first_frame := Engine.get_process_frames()
    var elapsed := 0.0
    while elapsed < seconds:
        await get_tree().physics_frame
        if not get_tree().paused:
            elapsed += get_physics_process_delta_time()
    var sim := _sim
    _sim = {}
    var world: WorldLoader = main.world
    var reports := world.npcs.filter(func(n): return is_instance_valid(n)).map(func(n): return n.report())
    var script_us := 0.0
    for r in reports:
        script_us += float(r["cost_us"])
    var samples := maxi(sim["samples"], 1)
    return {"seconds": snappedf(elapsed, 0.01), "npcs": reports,
        "cost": {"npcs": reports.size(), "script_us": snappedf(script_us / maxi(reports.size(), 1), 0.1),
            "process_ms": snappedf(sim["process"] / samples * 1000.0, 0.001),
            "physics_ms": snappedf(_mean(sim["ticks"]), 0.001),
            "physics_p95_ms": snappedf(_percentile(sim["ticks"], 0.95), 0.01),
            "frame_ms": snappedf(float(Time.get_ticks_usec() - started) / 1000.0
                / maxi(Engine.get_process_frames() - first_frame, 1), 0.001),
            "frame_p95_ms": snappedf(_percentile(sim["frames"], 0.95), 0.01),
            "frame_max_ms": snappedf(_percentile(sim["frames"], 1.0), 0.01),
            "draw_calls": roundi(sim["render"][0] / samples), "primitives": roundi(sim["render"][1] / samples),
            "objects": roundi(sim["render"][2] / samples)},
        "traffic": world.traffic.filter(func(t): return is_instance_valid(t)).map(func(t): return t.report()),
        "trains": world.trains.filter(func(t): return is_instance_valid(t)).map(func(t): return t.report())}


## Calls ``on_tick`` every physics frame at ``priority`` (lowest first).
class _TickMark extends Node:
    var on_tick: Callable

    func _init(priority: int, callback: Callable) -> void:
        process_physics_priority = priority
        process_mode = Node.PROCESS_MODE_PAUSABLE
        on_tick = callback

    func _physics_process(_delta: float) -> void:
        on_tick.call()


## The mean of ``values`` (0 when empty).
static func _mean(values: Array) -> float:
    var total := 0.0
    for v in values:
        total += v
    return total / maxi(values.size(), 1)


## The q-quantile (0..1) of ``values`` (0 when empty).
static func _percentile(values: Array, q: float) -> float:
    if values.is_empty():
        return 0.0
    var sorted := values.duplicate()
    sorted.sort()
    return sorted[clampi(int(ceil(q * sorted.size())) - 1, 0, sorted.size() - 1)]


# --- reports ------------------------------------------------------------------

func _lights() -> Dictionary:
    var world: WorldLoader = main.world
    var report := {}
    for name in world.lights.by_name:
        var light: OmniLight3D = world.lights.by_name[name]
        report[name] = {"visible": light.visible, "energy": light.light_energy,
            "range": light.omni_range, "y": snappedf(light.global_position.y, 0.01)}
    return {"fixtures": report, "total": get_tree().get_nodes_in_group("geogen_light").size(),
        "light3d": get_tree().root.find_children("*", "OmniLight3D", true, false).size()}


func _stats() -> Dictionary:
    var nav_regions := 0
    for map in NavigationServer3D.get_maps():
        nav_regions += NavigationServer3D.map_get_regions(map).size()
    return {"scene": main.world.scene_name,
        "nodes": int(Performance.get_monitor(Performance.OBJECT_NODE_COUNT)),
        "orphans": int(Performance.get_monitor(Performance.OBJECT_ORPHAN_NODE_COUNT)),
        "objects": int(Performance.get_monitor(Performance.OBJECT_COUNT)),
        "resources": int(Performance.get_monitor(Performance.OBJECT_RESOURCE_COUNT)),
        "nav_regions": nav_regions,
        "bodies": get_tree().root.find_children("*", "CollisionObject3D", true, false).size(),
        "world_children": main.world.get_child_count()}


## world.skeletons: bones (global positions), skinned meshes (bounds of the CPU-baked pose) and animations.
func _skeleton_report() -> Dictionary:
    var skeletons := []
    for sk: Skeleton3D in main.world.find_children("*", "Skeleton3D", true, false):
        var bones := {}
        for b in sk.get_bone_count():
            var p := sk.global_transform * sk.get_bone_global_pose(b).origin
            bones[sk.get_bone_name(b)] = [snappedf(p.x, 0.0001), snappedf(p.y, 0.0001), snappedf(p.z, 0.0001)]
        var meshes := []
        for mi: MeshInstance3D in sk.find_children("*", "MeshInstance3D", true, false):
            if mi.skin == null:
                continue
            var box := _skinned_bounds(mi, sk)
            meshes.append({"name": String(mi.name), "min": [box.position.x, box.position.y, box.position.z],
                "max": [box.end.x, box.end.y, box.end.z]})
        skeletons.append({"path": String(main.world.get_path_to(sk)), "bones": bones, "meshes": meshes})
    var animations := {}
    for ap: AnimationPlayer in main.world.find_children("*", "AnimationPlayer", true, false):
        animations[String(main.world.get_path_to(ap))] = Array(ap.get_animation_list())
    return {"skeletons": skeletons, "animations": animations}


## World bounds of a skinned mesh in its current pose, skinned on the CPU the way the renderer
## does it (vertex -> bind pose -> bone global pose -> mesh instance).
static func _skinned_bounds(mi: MeshInstance3D, sk: Skeleton3D) -> AABB:
    var joint_xforms: Array[Transform3D] = []
    for i in mi.skin.get_bind_count():
        var bone := mi.skin.get_bind_bone(i)
        if bone < 0:
            bone = sk.find_bone(mi.skin.get_bind_name(i))
        joint_xforms.append(sk.get_bone_global_pose(bone) * mi.skin.get_bind_pose(i))
    var box := AABB()
    var first := true
    for surface in mi.mesh.get_surface_count():
        var arrays := mi.mesh.surface_get_arrays(surface)
        var verts: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
        var bones: PackedInt32Array = arrays[Mesh.ARRAY_BONES]
        var weights: PackedFloat32Array = arrays[Mesh.ARRAY_WEIGHTS]
        var per := bones.size() / maxi(verts.size(), 1)
        for v in verts.size():
            var p := Vector3.ZERO
            for k in per:
                var w := weights[v * per + k]
                if w > 0.0:
                    p += (joint_xforms[bones[v * per + k]] * verts[v]) * w
            p = mi.global_transform * p
            if first:
                box = AABB(p, Vector3.ZERO)
                first = false
            else:
                box = box.expand(p)
    return box
