class_name GeogenPlayerFocus
extends Node
## What the player aims at and what E does with it (main.focus): an interaction within reach (with
## aim assist for small targets), one its `when` refuses, a physics prop to pick up, an NPC to
## greet or an affordance to sit or lie on; the E prompt under the crosshair; and pressing E.

## An interaction target this far off the aim line (m) still takes the E prompt.
const AIM_SLACK := 0.2
## How much further than the eye's height a prop can be picked up from (m): bending down.
const PICK_UP_BEND := 0.25
## How far in front of the eyes a carried prop is held (m).
const HOLD_DISTANCE := 1.4

var main: Node                    # main.gd: player, player_spec, world, hud, held_keys()
var prompt: Label                 # the E prompt (part of the HUD)
var interaction: GeogenInteraction = null
## Aimed at but refused by its `when`, with text saying why (`refused`): the prompt says it,
## and E tells the player again.
var refused: GeogenInteraction = null
## The physics prop aimed at that E would pick up (its uid), or "".
var prop := ""
var npc: GeogenNpc = null
var affordance := {}


## Forget what was aimed at (the world is changing).
func clear() -> void:
    interaction = null
    npc = null
    affordance = {}


## What the player is aiming at within reach, and its prompt ("E: Open").
func update() -> void:
    interaction = null
    refused = null
    affordance = {}
    npc = null
    prop = ""
    if main.world.props.held != "":
        pass        # carrying: E drops it, nothing else is aimed at
    elif main.player != null and main.player.camera.current and main.player.pose.is_empty():
        var cam: Camera3D = main.player.camera
        var from: Vector3 = cam.global_position
        var to: Vector3 = from - cam.global_basis.z * maxf(_longest_reach(), _pick_up_reach())
        var query := PhysicsRayQueryParameters3D.create(from, to)
        query.exclude = [main.player.get_rid()]
        var hit: Dictionary = main.get_world_3d().direct_space_state.intersect_ray(query)
        if hit and from.distance_to(hit["position"]) > main.player_spec.reach:
            # Beyond the player's reach only an interaction that reaches further counts, or a
            # prop to pick up within bending-down reach.
            var far: GeogenInteraction = main.world.interaction_for(hit["collider"])
            var far_prop: String = main.world.props.prop_of(hit["collider"])
            var distance: float = from.distance_to(hit["position"])
            if not (far != null and distance <= _reach_of(far)) and not (far_prop != ""
                    and main.world.props.can_carry(far_prop) and distance <= _pick_up_reach()):
                hit = {}
        if hit:
            var it: GeogenInteraction = main.world.interaction_for(hit["collider"])
            var hit_prop: String = main.world.props.prop_of(hit["collider"])
            if hit_prop != "":
                if main.world.props.can_carry(hit_prop):
                    prop = hit_prop
            elif hit["collider"] is GeogenNpc:
                if (hit["collider"] as GeogenNpc).can_greet():
                    npc = hit["collider"]
            elif it != null and it.can_use():
                interaction = it
            elif it != null and it.refusal() != "":
                refused = it
            else:
                affordance = main.world.affordance_near(hit["position"])
                if not affordance.is_empty() and not _in_sight(from, affordance["position"] + Vector3.UP * 0.05,
                        affordance["node"]):
                    affordance = {}     # near where the aim landed, but behind it (a booth through a wall)
        if interaction == null and npc == null:
            # A seated or lying NPC has no collider: the aim line passing its body finds it,
            # unless something (not the seat it is in) is in the way.
            var limit: float = main.player_spec.reach
            if hit:
                limit = minf(limit, from.distance_to(hit["position"]) + 0.4)
            for other in main.world.npcs:
                if not is_instance_valid(other) or not other.can_greet() or not other.posed():
                    continue
                var along: float = other.aim_distance(from, -cam.global_basis.z)
                # Past what the aim hit only from in front: there it is the seat behind them;
                # from behind, the chair's back is in the way.
                var beyond: bool = hit and along > from.distance_to(hit["position"])
                if along <= limit and (not beyond or other.faces(from)) \
                        and _in_sight(from, other.head_point(), null, 0.3):
                    npc = other
                    affordance = {}
                    break
        if interaction == null and npc == null and affordance.is_empty() and refused == null \
                and prop == "":
            interaction = _assisted(from, -cam.global_basis.z)
    if main.hud != null:
        main.hud.crosshair.aim(info().get("kind", "") not in ["", "refused"],
            main.player != null and main.player.camera.current and main.player.pose.is_empty())
    if prompt:
        if main.player != null and not main.player.pose.is_empty():
            prompt.text = "E: Stand up"
        elif main.world.props.held != "":
            prompt.text = "E: Drop"
        elif prop != "":
            prompt.text = "E: Pick up"
        elif npc != null:
            prompt.text = "E: %s" % npc.greet_prompt()
        elif interaction != null:
            prompt.text = "E: %s" % interaction.prompt(main.held_keys())
        elif refused != null:
            prompt.text = refused.refusal()
        elif not affordance.is_empty():
            prompt.text = "E: %s" % {"sit": "Sit", "lie": "Lie down", "use": "Use", "stand": "Stand here"}.get(
                affordance["type"], "Use")
        else:
            prompt.text = ""


## How far from the eye ``it`` can be used from: its own `reach` when that is longer.
func _reach_of(it: GeogenInteraction) -> float:
    return maxf(main.player_spec.reach, it.reach)


## Whether the eye at ``from`` sees ``point``: nothing is in the way, or only ``owner`` (the
## asset the point is on: aiming at a chair, the line to its seat may graze its arm) or
## something within ``slack`` of the point.
func _in_sight(from: Vector3, point: Vector3, owner: Node, slack := 0.1) -> bool:
    var query := PhysicsRayQueryParameters3D.create(from, point)
    query.exclude = [main.player.get_rid()]
    var hit: Dictionary = main.get_world_3d().direct_space_state.intersect_ray(query)
    if hit.is_empty() or (hit["position"] as Vector3).distance_to(point) <= slack:
        return true
    return owner != null and is_instance_valid(owner) and (hit["collider"] == owner
        or owner.is_ancestor_of(hit["collider"]))


## How far from the eye a prop can be picked up: the player bends down for one on the floor
## (a can a metre away), so further than `reach` (1.5 m, less than the eye's height).
func _pick_up_reach() -> float:
    return maxf(main.player_spec.reach, main.player_spec.eye_height + PICK_UP_BEND)


## The longest reach of anything loaded: how far the focus ray looks.
func _longest_reach() -> float:
    var longest: float = main.player_spec.reach
    for it in main.world.interactions:
        if is_instance_valid(it):
            longest = maxf(longest, it.reach)
    return longest


## Aim assist: the usable interaction whose target is nearest the aim line, when the line
## itself lands on nothing usable. Small targets (a light switch is 8 cm wide) are hard to
## hit exactly; within AIM_SLACK of one, and with nothing in the way, E takes it.
func _assisted(from: Vector3, forward: Vector3) -> GeogenInteraction:
    var best: GeogenInteraction = null
    var best_miss := AIM_SLACK
    var space: PhysicsDirectSpaceState3D = main.get_world_3d().direct_space_state
    for it in main.world.interactions:
        if not is_instance_valid(it) or not it.can_use():
            continue
        for target in it.targets:
            if not is_instance_valid(target) or target.global_position.distance_to(from) > _reach_of(it) + 2.0:
                continue
            var centre: Vector3 = target.global_position
            if target is MeshInstance3D:
                centre = target.global_transform * (target as MeshInstance3D).get_aabb().get_center()
            else:
                var box := WorldLoader._aabb(target)
                if box.size != Vector3.ZERO:
                    centre = box.get_center()
            var along: float = (centre - from).dot(forward)
            if along <= 0.0 or centre.distance_to(from) > _reach_of(it):
                continue
            var miss: float = (centre - from - forward * along).length()
            if miss >= best_miss:
                continue
            var query := PhysicsRayQueryParameters3D.create(from, centre)
            query.exclude = [main.player.get_rid()]
            var hit := space.intersect_ray(query)
            if hit and main.world.interaction_for(hit["collider"]) != it \
                    and (hit["position"] as Vector3).distance_to(centre) > 0.1:
                continue        # something else is in the way
            best = it
            best_miss = miss
    return best


## What the player aims at (the E prompt's subject), or {}.
func info() -> Dictionary:
    if main.world.props.held != "":
        return {"kind": "drop", "prop": main.world.props.held, "prompt": "Drop"}
    if prop != "":
        return {"kind": "carry", "prop": prop, "prompt": "Pick up"}
    if npc != null:
        return {"kind": "npc", "npc": String(npc.name), "prompt": npc.greet_prompt()}
    if interaction != null:
        return {"kind": "interaction", "asset": String(interaction.asset.name), "interaction": interaction.interaction_name,
            "prompt": interaction.prompt(main.held_keys())}
    if refused != null:
        return {"kind": "refused", "asset": String(refused.asset.name),
            "interaction": refused.interaction_name, "prompt": refused.refusal()}
    if not affordance.is_empty():
        return {"kind": "affordance", "asset": str(affordance.get("asset", "")),
            "type": affordance["type"]}
    return {}


## Press E on whatever is aimed at now (the runtime API); returns what that was, or {}.
func press() -> Dictionary:
    update()
    var used := info()
    if not main.player.pose.is_empty():
        used = {"kind": "stand"}
    if not used.is_empty():
        _press()
    return used


func _press() -> void:
    if not main.player.pose.is_empty():
        main.player.leave_pose()
    elif main.world.props.held != "":
        main.world.props.drop()
    elif prop != "":
        main.world.props.pick(prop, main.player, hold_point)
    elif npc != null:
        npc.greet()
    elif interaction != null:
        interaction.use("player")
    elif refused != null:
        refused.use("player")        # refused: it says why (WorldLoader -> GeogenRules.tell)
    elif not affordance.is_empty():
        main.player.take_pose(affordance)


## Where a carried prop is held: in front of the eyes.
func hold_point() -> Vector3:
    var cam: Camera3D = main.player.camera
    return cam.global_position - cam.global_basis.z * HOLD_DISTANCE
