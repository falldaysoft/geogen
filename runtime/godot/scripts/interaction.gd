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
func prompt() -> String:
    var t: Dictionary = states.get(target, {})
    if not t.has("next"):
        return ""
    return str(t.get("prompt", "Use"))


func can_use() -> bool:
    return states.get(target, {}).has("next")


func use() -> void:
    var t: Dictionary = states.get(target, {})
    if t.has("next"):
        target = str(t["next"])


## Jump or animate to a state (e.g. restoring a save).
func set_state(new_state: String, instant := false) -> void:
    target = new_state
    if instant:
        for motion in _motions:
            motion["value"] = float(motion["values"].get(new_state, 0.0))
            _apply(motion)
        state = new_state


func _physics_process(delta: float) -> void:
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
        state_entered.emit(state, str(s.get("emit", "")))
        if s.has("then"):
            target = str(s["then"])


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
