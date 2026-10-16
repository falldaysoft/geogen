class_name GeogenInteraction
extends Node
## One interaction exported in an asset node's extras.geogen.interactions:
## a small state machine whose states pose parts by rotating about a pivot
## or translating along an axis (asset frame). See
## docs/schema/geogen-extras.v1.schema.json.
##
## Exported transforms show the `initial` state, so each part's rest pose is
## M(initial value)^-1 * (asset^-1 * part); a value v poses it at
## asset * M(v) * rest. Values move at constant speed (the widest move takes
## `duration` seconds), so use() mid-motion simply reverses. A `visible` motion shows
## and hides its parts (and turns their colliders off): shown as soon as it heads for a
## state that shows them, hidden when it arrives at one that doesn't.

signal state_entered(state: String, event: String)
## What the event bus carries (events.gd): an actor used it (before it moves), the
## state it is heading to changed, it reached a state, it raised a signal (a
## state's emit, or "locked" for a refused use), its lock was turned.
signal used(actor: String)
signal target_changed(from: String, to: String, actor: String)
signal reached(state: String, emit: String)
signal raised(event: String, actor: String)
signal lock_changed(now_locked: bool, actor: String)
## A use its `when` stopped, by an interaction that says why (`refused`).
signal refused_use(text: String, actor: String)

var asset: Node3D
var interaction_name := ""
var states := {}
var state := ""          ## last state reached
var target := ""         ## state currently heading to
var duration := 0.9
var targets: Array[Node3D] = []
## Lock: key id ("" = no lock) and whether it's currently locked.
var lock_key := ""
var locked := false
## How it was exported: a save only holds what differs from these.
var initial := ""
var initial_locked := false
## Who caused the move under way: "player", an NPC's ID, or "" (a rule, a timer, a tool).
var actor := ""
var _dwell := 0.0          # seconds spent in the current state (for 'after')
## Freeze timed transitions (playtests keep doors open).
var hold := false
## The interaction's `when` (set by WorldLoader from the rules program): guard.call(actor) is
## whether that actor may use it now. Unset: always.
var guard := Callable()
## What the prompt says, and a refused use tells the user, while `guard` says no ("": nothing).
var refused := ""
## Metres from the eye it can be used from (0: the player's reach).
var reach := 0.0

## [{parts: [Node3D], type, axis: Vector3, pivot: Vector3, values: {state: float},
##   value: float, rest: {Node3D: Transform3D}, speed: float}]
var _motions: Array[Dictionary] = []
## Visibility motions: [{parts: [Node3D], values: {state: 1 shown | 0 hidden}}]
var _shows: Array[Dictionary] = []


static func from_extras(asset_node: Node3D, name_: String, data: Dictionary) -> GeogenInteraction:
    var it := GeogenInteraction.new()
    it.name = "%s_%s" % [asset_node.name, name_]
    it.asset = asset_node
    it.interaction_name = name_
    it.states = data.get("states", {})
    it.duration = maxf(float(data.get("duration", 0.9)), 0.01)
    it.state = str(data.get("initial", ""))
    it.target = it.state
    it.initial = it.state
    it.refused = str(data.get("refused", ""))
    it.reach = float(data.get("reach", 0.0))
    var lock = data.get("lock")
    if lock is Dictionary:
        it.lock_key = str(lock.get("key", ""))
        it.locked = bool(lock.get("locked", false))
        it.initial_locked = it.locked
    for n in data.get("targets", []):
        var node := _find(asset_node, n)
        if node:
            it.targets.append(node)
    var widest := 0.0
    for m: Dictionary in data.get("motions", []):
        if m.get("type") == "visible":
            var shown := {"type": "visible", "values": m.get("values", {}), "parts": []}
            for n in m.get("nodes", []):
                var part := _find(asset_node, n)
                if part != null:
                    shown["parts"].append(part)
            it._shows.append(shown)
            continue
        var a: Array = m.get("axis", [0, 1, 0])
        var p: Array = m.get("pivot", [0, 0, 0])
        var motion := {
            "type": m.get("type", "rotate"),
            "axis": Vector3(a[0], a[1], a[2]).normalized(),
            "pivot": Vector3(p[0], p[1], p[2]),
            "values": m.get("values", {}),
            "parts": [],
            "rest": {},
        }
        motion["value"] = float(motion["values"].get(it.state, 0.0))
        var initial_inv: Transform3D = it._matrix(motion, motion["value"]).affine_inverse()
        for n in m.get("nodes", []):
            var part := _find(asset_node, n)
            if part == null:
                push_warning("geogen: interaction %s: no node '%s'" % [it.name, n])
                continue
            motion["parts"].append(part)
            motion["rest"][part] = initial_inv * (asset_node.global_transform.affine_inverse() * part.global_transform)
        var vals: Array = motion["values"].values()
        if not vals.is_empty():
            widest = maxf(widest, vals.max() - vals.min())
        it._motions.append(motion)
    for motion in it._motions:
        var vals: Array = motion["values"].values()
        motion["speed"] = maxf(widest, 1e-6) / it.duration if not vals.is_empty() else 0.0
    it._apply_visibility()
    return it


func _ready() -> void:
    _apply_visibility.call_deferred()      # again once the loader has given the parts their colliders


static func _find(root: Node, node_name: String) -> Node3D:
    if root.name == node_name:
        return root as Node3D
    return root.find_child(node_name, true, false) as Node3D


## Every node whose transform this interaction drives.
func moving_nodes() -> Array[Node3D]:
    var result: Array[Node3D] = []
    for motion in _motions + _shows:
        for part in motion["parts"]:
            result.append(part)
    return result


## Prompt for the action use() would take, or "" if not usable now.
func prompt(keys: Array = [], by := "player") -> String:
    var t: Dictionary = states.get(target, {})
    if not t.has("next") or not allowed(by):
        return ""
    if locked:
        return "Unlock (L)" if opens(keys) else "Locked"
    var text := str(t.get("prompt", "Use"))
    if opens(keys):
        text += "   L: Lock"
    return text


## Why ``by`` can't use it now: its `refused` text while its `when` says no, else "".
func refusal(by := "player") -> String:
    if refused == "" or not states.get(target, {}).has("next") or allowed(by):
        return ""
    return refused


func can_use(by := "player") -> bool:
    return states.get(target, {}).has("next") and allowed(by)


## Whether its `when` lets ``by`` use it now (rules can still drive it either way).
func allowed(by := "") -> bool:
    return not guard.is_valid() or bool(guard.call(by))


## Returns false (and emits a "locked" event) when locked. ``by`` is the actor using it.
func use(by := "") -> bool:
    var t: Dictionary = states.get(target, {})
    if not t.has("next"):
        return false
    if locked:
        state_entered.emit(state, "locked")
        raised.emit("locked", by)
        return false
    if not allowed(by):
        if refused != "":
            refused_use.emit(refused, by)
        return false
    used.emit(by)
    _head_to(str(t["next"]), by)
    return true


func _head_to(new_state: String, by: String) -> void:
    if new_state == target:
        return
    var from := target
    target = new_state
    actor = by
    _apply_visibility()
    target_changed.emit(from, new_state, by)


## Whether one of ``keys`` (ids or globs: a master key's "key_room_*") fits the lock.
func opens(keys: Array) -> bool:
    if lock_key == "":
        return false
    for key in keys:
        if lock_key.match(str(key)):
            return true
    return false


## Lock or unlock with a key the user holds. Only closed (initial-state) doors lock.
func toggle_lock(keys: Array, by := "") -> bool:
    if not opens(keys):
        return false
    if not locked and state != target:
        return false  # still moving
    locked = not locked
    state_entered.emit(state, "locked" if locked else "unlocked")
    lock_changed.emit(locked, by)
    return true


## Lock or unlock without a key (a rule, a tool).
func set_locked(value: bool, by := "") -> void:
    if value == locked:
        return
    locked = value
    state_entered.emit(state, "locked" if locked else "unlocked")
    lock_changed.emit(locked, by)


## Serializable state (for save/restore).
func snapshot() -> Dictionary:
    return {"state": target, "locked": locked}


func restore(data: Dictionary) -> void:
    locked = bool(data.get("locked", locked))
    set_state(str(data.get("state", state)), true)


## How far a move or an after-dwell has got, as a save timer would say it (docs/rules.md
## "Saving"): {kind: arrive, state, from, in, actor} on the way to `state` from `from`,
## {kind: dwell, state, in, actor} waiting to move on, else {}. `in`: seconds left.
func progress() -> Dictionary:
    if state != target:
        var left := 0.0
        for motion in _motions:
            if motion["speed"] > 0.0:
                left = maxf(left, absf(float(motion["values"].get(target, 0.0)) - motion["value"]) / motion["speed"])
        return {"kind": "arrive", "state": target, "from": state, "in": left, "actor": actor}
    var s: Dictionary = states.get(state, {})
    if s.has("after") and s.has("then") and not hold:
        return {"kind": "dwell", "state": state, "in": maxf(float(s["after"]) - _dwell, 0.0), "actor": actor}
    return {}


## Carry on from a save timer (see `progress`) due in `left` seconds, once restored to the
## state it names: part of the way there from its `from` (an arrive without one, or that no
## longer matches, just arrives), or part of the way through the dwell. Silent, like restore.
func resume(timer: Dictionary, left: float) -> void:
    var who = timer.get("actor")
    if timer.get("kind") == "dwell":
        var s: Dictionary = states.get(state, {})
        if state == target and target == str(timer.get("state")) and s.has("after"):
            _dwell = clampf(float(s["after"]) - left, 0.0, float(s["after"]))
            actor = "" if who == null else str(who)
        return
    var from := str(timer.get("from", ""))
    if timer.get("kind") != "arrive" or target != str(timer.get("state")) or from == target \
            or not states.has(from) or _motions.is_empty():
        return
    var whole := 0.0      # the move's time from `from` (motions share one speed)
    for motion in _motions:
        if motion["speed"] > 0.0:
            whole = maxf(whole, absf(float(motion["values"].get(target, 0.0))
                - float(motion["values"].get(from, 0.0))) / motion["speed"])
    for motion in _motions:
        var start := float(motion["values"].get(from, 0.0))
        motion["value"] = move_toward(start, float(motion["values"].get(target, 0.0)),
            motion["speed"] * maxf(whole - left, 0.0))
        _apply(motion)
    state = from
    actor = "" if who == null else str(who)
    _apply_emission()
    _apply_visibility()


## Jump or animate to a state. Jumping (restoring a save, a tool) is silent: no
## events, since nothing happened. ``by`` is the actor behind an animated move.
func set_state(new_state: String, instant := false, by := "") -> void:
    if not instant:
        _head_to(new_state, by)
        return
    target = new_state
    if instant:
        for motion in _motions:
            motion["value"] = float(motion["values"].get(new_state, 0.0))
            _apply(motion)
        state = new_state
        _apply_emission()
        _apply_visibility()
        state_entered.emit(state, "")   # listeners (light switches) sync; no gameplay event


func _physics_process(delta: float) -> void:
    if state == target:
        # Dwelling: auto-advance after 'after' seconds (e.g. a door closing itself).
        var s: Dictionary = states.get(state, {})
        if s.has("after") and s.has("then") and not hold:
            _dwell += delta
            if _dwell >= float(s["after"]):
                _dwell = 0.0
                _head_to(str(s["then"]), actor)
        return
    _dwell = 0.0
    var arrived := true
    for motion in _motions:
        var goal := float(motion["values"].get(target, 0.0))
        var value: float = motion["value"]
        if not is_equal_approx(value, goal):
            value = move_toward(value, goal, motion["speed"] * delta)
            motion["value"] = value
            _apply(motion)
        if not is_equal_approx(value, goal):
            arrived = false
    if arrived and state != target:
        state = target
        var s: Dictionary = states.get(state, {})
        _apply_emission()
        _apply_visibility()
        state_entered.emit(state, str(s.get("emit", "")))
        reached.emit(state, str(s.get("emit", "")))
        if s.has("emit"):
            raised.emit(str(s["emit"]), actor)
        if s.has("then") and not s.has("after"):
            _head_to(str(s["then"]), actor)


## States named on/off light up or dim the targets' emissive materials (TV screens).
func _apply_emission() -> void:
    if not (states.has("on") and states.has("off")):
        return
    for t in targets:
        for mi: MeshInstance3D in [t] + t.find_children("*", "MeshInstance3D", true, false) if t is MeshInstance3D \
                else t.find_children("*", "MeshInstance3D", true, false):
            for i in mi.get_surface_override_material_count():
                var mat := mi.get_active_material(i) as BaseMaterial3D
                if mat == null or not mat.emission_enabled:
                    continue
                if not mi.has_meta("geogen_base_emission"):
                    mi.set_meta("geogen_base_emission", mat.emission_energy_multiplier)
                    mat = mat.duplicate()
                    mi.set_surface_override_material(i, mat)
                var base: float = mi.get_meta("geogen_base_emission")
                (mi.get_surface_override_material(i) as BaseMaterial3D).emission_energy_multiplier = \
                    base * (12.0 if state == "on" else 1.0)


## Parts of visibility motions show while the state reached or the one heading for shows them.
func _apply_visibility() -> void:
    for shown in _shows:
        var values: Dictionary = shown["values"]
        var on := float(values.get(target, 0.0)) >= 0.5 or float(values.get(state, 0.0)) >= 0.5
        for part: Node3D in shown["parts"]:
            part.visible = on
            for col: CollisionShape3D in part.find_children("*", "CollisionShape3D", true, false):
                col.set_deferred("disabled", not on)


func _apply(motion: Dictionary) -> void:
    var m := _matrix(motion, motion["value"])
    for part: Node3D in motion["parts"]:
        part.global_transform = asset.global_transform * m * motion["rest"][part]


func _matrix(motion: Dictionary, value: float) -> Transform3D:
    var axis: Vector3 = motion["axis"]
    if motion["type"] == "translate":
        return Transform3D(Basis.IDENTITY, axis * value)
    var pivot: Vector3 = motion["pivot"]
    var basis := Basis(axis, deg_to_rad(value))
    return Transform3D(basis, pivot - basis * pivot)
