class_name GeogenWalker
extends RefCounted
## Steers the player body along a navmesh path, the way a person at the keys
## would: face the next waypoint and walk forward; if no progress is made for a
## moment, sidestep toward the side the path continues on, and give up after a
## few tries. Shared by the playtest bot and the runtime API's walk_to.
##
##   var walker := GeogenWalker.new(player, path)
##   # each physics frame:
##   match walker.step(delta): "walking", "arrived", "stuck"

const WAYPOINT_RADIUS := 0.35
const STALL_SECONDS := 0.6      # no progress this long: sidestep
const SIDESTEP_SECONDS := 0.35
const MAX_SIDESTEPS := 3        # then the walk counts as stuck

var player: Player
var path: PackedVector3Array
var waypoint := 1
var status := "walking"
var _stuck_time := 0.0
var _sidesteps := 0
var _sidestep_left := 0.0
var _sidestep_dir := 1.0
var _best_distance := INF


func _init(player_: Player, path_: PackedVector3Array) -> void:
    player = player_
    path = path_
    if path.size() < 2:
        status = "arrived" if path.size() == 1 else "stuck"


## Advance one physics frame; returns the status ("walking", "arrived" or "stuck").
## Leaves player.scripted_move set while walking and clears it once done.
func step(delta: float) -> String:
    if status != "walking":
        return status
    if waypoint >= path.size():
        return _finish("arrived")
    var target := path[waypoint]
    var to := target - player.global_position
    var flat := Vector2(to.x, to.z)
    if flat.length() < WAYPOINT_RADIUS:
        waypoint += 1
        _best_distance = INF
        return status
    player.rotation.y = atan2(-to.x, -to.z)
    if _sidestep_left > 0.0:
        # Slide off whatever we're caught on (a jamb, a leaf edge), then retry.
        _sidestep_left -= delta
        player.scripted_move = Vector2(_sidestep_dir, 0.3)
        return status
    player.scripted_move = Vector2(0, 1)
    if flat.length() < _best_distance - 0.05:
        _best_distance = flat.length()
        _stuck_time = 0.0
    else:
        _stuck_time += delta
        if _stuck_time > STALL_SECONDS:
            _stuck_time = 0.0
            _sidesteps += 1
            if _sidesteps > MAX_SIDESTEPS:
                return _finish("stuck")
            # Step toward the side the path continues on (alternate if unsure).
            var after := path[mini(waypoint + 1, path.size() - 1)] - player.global_position
            var right := Vector2(-cos(player.rotation.y), sin(player.rotation.y))
            var lateral := Vector2(after.x, after.z).dot(right)
            _sidestep_dir = signf(lateral) if absf(lateral) > 0.05 else (1.0 if _sidesteps % 2 else -1.0)
            _sidestep_left = SIDESTEP_SECONDS
    return status


## Stop where the player is (e.g. the caller timed out).
func cancel() -> void:
    _finish("cancelled")


func _finish(result: String) -> String:
    status = result
    player.scripted_move = null
    return status
