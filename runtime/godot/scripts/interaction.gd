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
## `duration` seconds), so use() mid-motion simply reverses.

signal state_entered(state: String, event: String)

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
var _dwell := 0.0          # seconds spent in the current state (for 'after')
## Freeze timed transitions (playtests keep doors open).
var hold := false

## [{parts: [Node3D], type, axis: Vector3, pivot: Vector3, values: {state: float},
##   value: float, rest: {Node3D: Transform3D}, speed: float}]
var _motions: Array[Dictionary] = []


static func from_extras(asset_node: Node3D, name_: String, data: Dictionary) -> GeogenInteraction:
    var it := GeogenInteraction.new()
    it.name = "%s_%s" % [asset_node.name, name_]
    it.asset = asset_node
    it.interaction_name = name_
    it.states = data.get("states", {})
    it.duration = maxf(float(data.get("duration", 0.9)), 0.01)
    it.state = str(data.get("initial", ""))
    it.target = it.state
    var lock = data.get("lock")
    if lock is Dictionary:
        it.lock_key = str(lock.get("key", ""))
        it.locked = bool(lock.get("locked", false))
    for n in data.get("targets", []):
        var node := _find(asset_node, n)
        if node:
            it.targets.append(node)
    var widest := 0.0
    for m: Dictionary in data.get("motions", []):
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
    return it


static func _find(root: Node, node_name: String) -> Node3D:
    if root.name == node_name:
        return root as Node3D
    return root.find_child(node_name, true, false) as Node3D


## Every node whose transform this interaction drives.
func moving_nodes() -> Array[Node3D]:
    var result: Array[Node3D] = []
    for motion in _motions:
        for part in motion["parts"]:
            result.append(part)
    return result


## Prompt for the action use() would take, or "" if not usable now.
func prompt(keys: Array = []) -> String:
    var t: Dictionary = states.get(target, {})
    if not t.has("next"):
        return ""
    if locked:
        return "Unlock (L)" if keys.has(lock_key) else "Locked"
    var text := str(t.get("prompt", "Use"))
    if lock_key != "" and keys.has(lock_key):
        text += "   L: Lock"
    return text


func can_use() -> bool:
    return states.get(target, {}).has("next")


## Returns false (and emits a "locked" event) when locked.
func use() -> bool:
    var t: Dictionary = states.get(target, {})
    if not t.has("next"):
        return false
    if locked:
        state_entered.emit(state, "locked")
        return false
    target = str(t["next"])
    return true


## Lock or unlock with a key the user holds. Only closed (initial-state) doors lock.
func toggle_lock(keys: Array) -> bool:
    if lock_key == "" or not keys.has(lock_key):
        return false
    if not locked and state != target:
        return false  # still moving
    locked = not locked
    state_entered.emit(state, "locked" if locked else "unlocked")
    return true


## Serializable state (for save/restore).
func snapshot() -> Dictionary:
    return {"state": target, "locked": locked}


func restore(data: Dictionary) -> void:
    locked = bool(data.get("locked", locked))
    set_state(str(data.get("state", state)), true)


## Jump or animate to a state (e.g. restoring a save).
func set_state(new_state: String, instant := false) -> void:
    target = new_state
    if instant:
        for motion in _motions:
            motion["value"] = float(motion["values"].get(new_state, 0.0))
            _apply(motion)
        state = new_state
        _apply_emission()
        state_entered.emit(state, "")   # listeners (light switches) sync; no gameplay event


func _physics_process(delta: float) -> void:
    if state == target:
        # Dwelling: auto-advance after 'after' seconds (e.g. a door closing itself).
        var s: Dictionary = states.get(state, {})
        if s.has("after") and s.has("then") and not hold:
            _dwell += delta
            if _dwell >= float(s["after"]):
                _dwell = 0.0
                target = str(s["then"])
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
        state_entered.emit(state, str(s.get("emit", "")))
        if s.has("then") and not s.has("after"):
            target = str(s["then"])


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
