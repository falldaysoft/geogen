class_name GeogenApi
extends Node
## Runtime control API (``--api[=PORT]``): a localhost TCP server so tools, tests and agents can
## query and drive one long-lived world instead of launching a process per check.
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
    "batch": "{commands: [{cmd, args}], stop_on_error?} run commands in order, one reply",
    "quit": "stop the runtime",
    "wait_ready": "{timeout?} wait until the world is loaded and the navmesh answers queries",
    "world.info": "scene, bounds, spawns, rooms, npc names, player spec, clock",
    "world.scenes": "the scene catalogue",
    "world.query": "{tag?, type?, name? (glob), room?, interactive?, limit?} nodes carrying extras.geogen",
    "world.room_at": "{pos} room id at a point",
    "world.raycast": "{from, to} first hit: position, normal, node, room, interaction",
    "world.nav_path": "{from, to} navmesh path: points, length, reached",
    "world.switch": "{scene, spawn?} load another exported scene",
    "world.reload": "{keep_player?} re-read the current scene's export (after a re-export)",
    "world.node": "{name | node_path} one node: transform, world bounds, extras, colliders, children, interactions",
    "player.status": "position, yaw, pitch, room, pose, focus, keys, on_floor",
    "player.teleport": "{pos, yaw?} put the player's feet at pos",
    "player.look": "{yaw?, pitch?} or {at: point}",
    "player.move": "{forward?, strafe?, seconds} walk with the controls",
    "player.walk_to": "{pos, timeout?} follow the navmesh there with the real body",
    "player.use": "{asset?, interaction?} use the aimed thing, or a named asset's interactions",
    "player.sit": "{asset?} take the nearest (or the asset's) sit/lie affordance",
    "player.stand": "leave a sit/lie pose",
    "player.keys": "{keys?} set (or get) the keys the player holds",
    "interactions.list": "{asset?} interactions with states",
    "interactions.set": "{asset, interaction?, state, instant?, wait?} jump or animate to a state; wait: until the parts arrive, returning their bounds",
    "interactions.use": "{asset, interaction?, wait?} advance to the next state (honours locks)",
    "interactions.lock": "{asset, interaction?, locked} lock or unlock without a key",
    "state.save": "interaction states keyed asset/interaction",
    "state.load": "{state} restore a save",
    "npcs.list": "every NPC's report (doing, needs, position, ...)",
    "npcs.options": "{npc} the NPC's scored options now",
    "npcs.force": "{npc, option} make its next decision take that option",
    "npcs.greet": "{npc} greet it",
    "npcs.pause": "{npc?} freeze one NPC (or all)",
    "npcs.resume": "{npc?}",
    "time.pause": "freeze the world (the API keeps serving)",
    "time.resume": "",
    "time.step": "{frames} run N physics frames, then pause",
    "time.advance": "{seconds, scale?} run S s of world time (at scale, default 8), return NPC reports",
    "time.scale": "{scale} world speed",
    "time.clock": "{time? 'HH:MM', day_length?} get or set the world clock",
    "capture.screenshot": "{path?, from?, at?, node?, view?, zoom?, isolate?, fov?, size?} PNG from the current camera, a pose (from/at) or framed on a node from a view (iso, front, back, left, right, top); base64 without path",
    "capture.views": "{node? | node_path? | bounds?, views?, zoom?, isolate?, path?} contact sheet of views framed on a node or box",
    "capture.camera": "{mode: player|overview|follow, npc?}",
    "events.subscribe": "{streams: [interaction, npc, room, travel, traffic]}",
    "events.unsubscribe": "{streams?}",
}
## extras.geogen keys every mesh part may carry; a node with only these isn't listed by world.query.
const PART_KEYS := ["version", "collider", "walkable"]
const TRACE_SECONDS := 0.25          # walk/move position samples
const STREAMS := ["interaction", "npc", "room", "travel", "traffic"]

var main: Node                 # main.gd (player, world, overview, clock, switch_scene, ...)
var port := 7878
var _server := TCPServer.new()
var _peers: Array[Dictionary] = []   # {peer: StreamPeerTCP, buffer: String, streams: {}}
var _connected_npcs := {}            # GeogenNpc -> true (traced wired)
var _connected_traffic := {}
var _room := ""
var _walker: GeogenWalker = null
var _world_frames := 0               # physics frames the world ran (not counting paused ones)


func _ready() -> void:
    process_mode = Node.PROCESS_MODE_ALWAYS   # serve while the world is paused
    var err := _server.listen(port, "127.0.0.1")
    if err != OK:
        push_error("geogen api: can't listen on port %d (%s)" % [port, error_string(err)])
        return
    print("api listening: %s" % JSON.stringify({"port": _server.get_local_port()}))
    var world: WorldLoader = main.world
    world.interaction_event.connect(func(asset: String, interaction: String, state: String, event: String):
        _emit("interaction", {"asset": asset, "interaction": interaction, "state": state, "event": event}))
    world.travel_requested.connect(func(travel: Dictionary): _emit("travel", travel))


func _process(_delta: float) -> void:
    while _server.is_connection_available():
        var peer := _server.take_connection()
        peer.set_no_delay(true)
        _peers.append({"peer": peer, "buffer": "", "streams": {}})
    for entry in _peers.duplicate():
        var peer: StreamPeerTCP = entry["peer"]
        peer.poll()
        if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
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
    var peer: StreamPeerTCP = entry["peer"]
    if peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
        peer.put_data((JSON.stringify(data) + "\n").to_utf8_buffer())


func _emit(stream: String, data: Dictionary) -> void:
    for entry in _peers:
        if entry["streams"].has(stream):
            _send(entry, {"event": stream, "data": data})


## NPCs and traffic come and go with scene switches: wire their trace signals when subscribed.
func _wire_streams() -> void:
    var want_npc := _peers.any(func(e): return e["streams"].has("npc"))
    var want_traffic := _peers.any(func(e): return e["streams"].has("traffic"))
    if want_npc:
        for npc in main.world.npcs:
            if is_instance_valid(npc) and not _connected_npcs.has(npc):
                _connected_npcs[npc] = true
                npc.traced.connect(func(event: Dictionary): _emit("npc", event))
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
            _connected_npcs.clear()
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
            _connected_npcs.clear()
            _connected_traffic.clear()
            if keep:
                player.spawn(pos, yaw)
            return _info()
        "world.node":
            var node := _find_node(args)
            return _err("no node %s" % args.get("node_path", args.get("name", args.get("node", "")))) \
                if node == null else _node_info(node)
        "player.status":
            return _status()
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
        "player.keys":
            if args.has("keys"):
                main.keys = Array(args["keys"])
            return {"keys": main.keys}
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
                    it.set_state(str(args["state"]), bool(args.get("instant", false)))
                elif cmd == "interactions.use":
                    it.use()
                else:
                    it.locked = bool(args.get("locked", true))
            if cmd != "interactions.lock" and bool(args.get("wait", false)):
                var waited := 0.0
                while found.any(func(i): return i.state != i.target) and waited < 10.0:
                    await get_tree().physics_frame
                    if not get_tree().paused:
                        waited += get_physics_process_delta_time()
                return found.map(func(it): return _interaction(it, true))
            return found.map(func(it): return _interaction(it))
        "state.save":
            return world.save_state()
        "state.load":
            if not args.get("state") is Dictionary:
                return _err("state: {asset/interaction: {state, locked}}")
            world.load_state(args["state"])
            return world.save_state()
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
        "time.scale":
            _set_scale(float(args.get("scale", 1.0)))
            return _time()
        "time.clock":
            var clock: GeogenClock = main.clock
            if args.has("time"):
                clock.hours = GeogenClock.parse(str(args["time"]))
                clock.apply()
            if args.has("day_length"):
                clock.day_length = float(args["day_length"])
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
            return {"streams": entry["streams"].keys()}
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
        "npcs": world.npcs.filter(func(n): return is_instance_valid(n)).map(func(n): return String(n.name)),
        "interactions": world.interactions.size(), "affordances": world.affordances.size(),
        "player_spec": main.player_spec.to_dict(), "time": _time(), "stream": world.stream_report()}


func _time() -> Dictionary:
    var clock: GeogenClock = main.clock
    return {"paused": get_tree().paused, "scale": Engine.time_scale, "frame": _world_frames,
        "clock": clock.label(), "hours": snappedf(clock.hours, 0.0001), "day_length": clock.day_length,
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


func _wait_ready(timeout: float) -> Dictionary:
    var started := Time.get_ticks_msec()
    var settled := 0
    while true:
        var ready: bool = main.player != null and not main.world.loading and not main.is_switching() and _nav_ready()
        settled = settled + 1 if ready else 0
        if settled >= 5:    # a few frames for tiled regions to merge
            break
        if Time.get_ticks_msec() - started > timeout * 1000.0:
            return _err("not ready after %.0f s (loading %s, nav %s)" % [timeout, main.world.loading, _nav_ready()])
        await get_tree().process_frame
    return _info()


func _query(args: Dictionary) -> Array:
    var tag := str(args.get("tag", ""))
    var type := str(args.get("type", ""))
    var pattern := str(args.get("name", ""))
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
        if interactive and not g.has("interactions"):
            continue
        var pos := node.global_position
        if room != "" and world.room_at(pos + Vector3(0, 0.3, 0)) != room:
            continue
        out.append({"name": String(node.name), "node_path": String(world.get_path_to(node)), "position": _v(pos),
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
    var path := NavigationServer3D.map_get_path(map, from, to, true)
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
    main.update_focus()
    return {"position": _v(p), "yaw": snappedf(rad_to_deg(wrapf(player.rotation.y, -PI, PI)), 0.1),
        "pitch": snappedf(rad_to_deg(player.head.rotation.x), 0.1),
        "eye": _v(player.camera.global_position), "on_floor": player.is_on_floor(), "flying": player.flying,
        "room": main.world.room_at(p + Vector3(0, 0.5, 0)), "scene": main.world.scene_name,
        "pose": player.pose.get("type", "stand"), "pose_asset": player.pose.get("asset", ""),
        "focus": main.focus_info(), "keys": main.keys,
        "walking": _walker != null and _walker.status == "walking"}


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
    while motion["elapsed"] < seconds:
        await get_tree().physics_frame
        _track(motion)
    player.scripted_move = null
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
    var path := PackedVector3Array()
    for p in route["points"]:
        path.append(Vector3(p[0], p[1], p[2]))
    var walker := GeogenWalker.new(player, path)
    _walker = walker
    var timeout := float(args.get("timeout", 30.0))
    var motion := {"trace": [_v(player.global_position)], "blocked_by": null, "elapsed": 0.0,
        "next_sample": TRACE_SECONDS}
    while walker.status == "walking":
        await get_tree().physics_frame
        _track(motion)
        if motion["elapsed"] > timeout:
            walker.cancel()
            walker.status = "timeout"
    if _walker == walker:
        _walker = null
    var p := player.global_position
    var result := _status()
    result.merge({"status": walker.status, "reached": route["reached"] and walker.status == "arrived",
        "distance": snappedf(Vector2(p.x - target.x, p.z - target.z).length(), 0.001),
        "path_length": route["length"], "path": route["points"], "seconds": snappedf(motion["elapsed"], 0.01),
        "trace": motion["trace"], "blocked_by": motion["blocked_by"] if walker.status != "arrived" else null}, true)
    return result


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
            "interaction": {"name": it.interaction_name, "state": it.state} if it != null else null}


func _cancel_walk() -> void:
    if _walker != null:
        _walker.cancel()
        _walker = null


func _use(args: Dictionary):
    if not args.has("asset"):
        var used: Dictionary = main.press_use()
        return _err("nothing usable in view") if used.is_empty() else used
    var found := _interactions(args)
    if found.is_empty():
        return _err("no interaction %s on '%s'" % [args.get("interaction", ""), args["asset"]])
    var results := []
    for it in found:
        var ok: bool = it.use()
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
        "states": it.states.keys(), "prompt": it.prompt(main.keys), "position": _v(it.asset.global_position),
        "room": main.world.room_at(it.asset.global_position + Vector3(0, 0.3, 0))}
    if parts:
        # Where the moving parts are now (e.g. does an open leaf clear the wardrobe beside it?).
        info["parts"] = it.moving_nodes().map(func(n): return {"name": String(n.name), "bounds": _box(_bounds(n))})
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
    return {"name": String(node.name), "node_path": String(world.get_path_to(node)), "type": node.get_class(),
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
    Engine.physics_ticks_per_second = int(60 * maxf(scale, 1.0))
    Engine.max_physics_steps_per_frame = int(8 * maxf(scale, 1.0))


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
