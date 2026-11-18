class_name GeogenNpc
extends CharacterBody3D
## An NPC exported in a node's extras.geogen.npc (see geogen/npc.py and
## assets/npcs/). This script is a generic interpreter: everything specific
## comes from the definition and from what the world advertises.
##
## - Needs fall at their declared rates; doing something adds its
##   `advertises` to them.
## - Deciding: every option (affordances in the home region with a free
##   slot, plus the NPC's own activities) scores
##       sum((1 - need) x advertised) x preference(tags)
##       - distance x metres - recency x exp(-age / memory) + noise x random
##   with weights from the definition's `scoring`; the best one runs.
## - Running: an option's action is a list of steps (actions.yaml) from a
##   fixed vocabulary: go_to, face, pose, wait, use. A path through a closed
##   door (a `portal`) pushes that door's `pass` action first.
##
## The body asset (a child of the exported node) is reparented under this
## CharacterBody3D; poses are root transforms the body asset declares.

signal traced(event: Dictionary)
## An actor greeted this NPC: `signal_name` is its definition's `attention.greet.signal`,
## which WorldLoader posts on the event bus as a signal from the NPC (docs/rules.md).
signal greeted(signal_name: String, by: String)

const LAYER := 2                  # NPC bodies; the navmesh bakes layer 1 only
const WAYPOINT_RADIUS := 0.15
const STALL_SECONDS := 0.6        # no progress this long: sidestep
const SIDESTEP_SECONDS := 0.4
const FAIL_SECONDS := 4.0         # no progress this long: give up on the step
const REPLAN_SECONDS := 1.5       # no progress this long on a detour round a character: plan again
const MAX_REPLANS := 2            # ... this often per go_to
const OWN_BONUS := 1.6            # score multiplier of what an NPC owns (geogen/npc.py OWN_BONUS)
const PREFER_BONUS := 1.3         # ...and of what it prefers
const HANDLE_STAND := 0.45        # m from a door's face to stand and open it by its handle (the leaf swings away)
const HANDLE_GRIP := 0.07         # m short of a door's handle the wrist stops (the fingers are round it)
const LOW_STAND := 0.05           # m left between the body and something low it takes
const LOW_HAND := 0.72            # m over the feet: a thing lower than this is taken with the knees bent
const MAX_CROUCH := 0.6           # of the leg's length: as far as the hips go down for it
const HANDS_OFF := 0.07           # m the open hands come off the sides of a bulky thing
const PRESS_REACH := 1.6          # of the arm bones' length from the shoulders (the hand and a bent spine add the rest)
const HANDLE_RANGE := 0.8         # m from the shoulders: a handle further away isn't reached for
const SHOULDER_DROP := 0.25       # m from the head point down to the shoulders
const STAIR_HOLD := 2.0           # m before a stair to wait while someone is coming along it
const STAIR_PATIENCE := 45.0      # s waiting for a stair before giving up on the step
const STAIR_REACH := 0.8          # m from a stair's walking line that counts as on it
const LEVEL_TOLERANCE := NpcGeometry.LEVEL_TOLERANCE
const USE_TIMEOUT := 6.0
const POSE_SECONDS := 0.5         # hold after changing pose (bodies without a transition clip)
const CLOSE_REACH := 0.5          # m: furthest a `go_to` with `close` steps off the navmesh (up to a counter)
const CLOSE_RADIUS := 0.03        # m: how near the spot that last step ends
const REACT_DRIFT := 0.1          # m: a reaction that moved the NPC further sends it back to its last go_to
const SEAT_CLEAR := 0.03          # m the calves keep in front of a seat's front edge
const PORTAL_LOOKAHEAD := 2.5     # react to closed doors this far ahead
const AVOID_AHEAD := 1.6          # steer around characters (player, NPCs) this far ahead
const NEARBY := 4.0               # m: `go_to: nearby` picks a navmesh point within this of where the NPC stands
const NEARBY_MIN := 1.5           # m: ...and at least this far, so it is a few steps away
const NEARBY_DETOUR := 1.4        # ...reached by a walk at most this many times the straight line
const AVOID_MARGIN := 0.2
const DETOUR_HORIZON := 8.0       # m of path ahead whose waypoints are moved off characters (farther ones as they come up)
const CROWD_CELL := 2.0           # m, cells of the shared character grid
const THROTTLE_NEAR := 10.0       # m: off-screen NPCs nearer the player than this still step every frame
const THROTTLE_FAR := 60.0        # m: beyond this NPCs step every other frame even on screen
const SLEEP_DISTANCE := 80.0      # m: off screen beyond this an NPC sleeps: steps every SLEEP_EVERY frames, clips paused
const SLEEP_EVERY := 8
## Moving bodies NPCs steer around (the player joins it too).
const CHARACTER_GROUP := "geogen_character"
## Traffic (traffic.gd): wait for moving vehicles, walk around stopped ones.
const VEHICLE_GROUP := "geogen_vehicle"
const VEHICLE_MARGIN := 0.35
const VEHICLE_LOOKAHEAD := 3.0    # s: a vehicle that would cross our next steps this soon has priority
const VEHICLE_PROBE := 2.5        # m of our path checked against vehicles
const VEHICLE_PATIENCE := 3.0     # s waiting at the kerb before stepping out anyway (traffic yields)
# Groups (geogen.87.3, definition `group`: {leader, radius}).
const GROUP_REFILL := 0.15        # `together` per s while a groupmate is within the group's radius
const GROUP_BONUS := 1.5          # score multiplier of a seat on the same thing as, or beside, one a groupmate holds
const GROUP_BESIDE := 1.2         # m: a groupmate's affordance this near counts as beside
const GROUP_WANDER := 0.75        # members wander within this share of the group's radius of the leader
const GROUP_REJOIN_GAP := 1.4     # m from the groupmate a rejoin walks to (nearer is standing on them)
const GROUP_FAR := 0.35           # a member's utility of things beyond the group's radius from its leader (or where it's going)
const GROUP_GAP := 1.0           # m: a member's wander doesn't end nearer another character than this
const GROUP_REFRESH := 2.0
const COMPANY_ROOM := 0.5        # an offer is an option once a need has room for this share of what it gives
const PLANNER_LIMITS := {"depth": 5, "patience": 60.0, "nodes": 2000}    # for one told to, with no `planner:` of its own
const ORDER_WAIT := 30.0          # s an order to join someone waits for them to be free
const PARTNER_GAP := 0.9          # m from the one it goes to talk to (`go_to: partner`)
const PARTNER_FAR := 3.0          # m: further off than this when it gets there, they have gone
const PARTNER_WAIT := 90.0        # s the one who offers waits for the other at most (`wait: partner`)
const PICK_SECONDS := 0.7         # s to take hold of a prop, or to set one down
const CUT_PATIENCE := 3.0         # s an interrupt waits for the body to be ready (mid sit-down), then lapses
const SEEN_NEAR := 0.4            # m: a ray toward a thing may stop this short of it (it hits the thing itself)
const ON_SEAT := 0.35             # m (flat) from a seat's anchor a loose prop is on the seat
const CARRY_HEIGHT := 0.6         # of the body's height: where the middle of a thing carried in both hands is
const CARRY_BODY := 0.14          # m from the body's middle to the near side of such a thing
const SET_OUT_NEAR := 0.3        # m ahead of the seat a set-out item's top is first looked for
const SET_OUT_FAR := 0.7         # and the furthest
const SET_OUT_ABOVE := 0.7       # m above the seat the look down starts
const FIND_OWN := 0.5             # a `find` step's own things (its bed) count as this much as far
const CARRY_PALM := 0.02          # m off its side the hand bone is (the palm is on it)
const STAND_SLACK := 0.2          # m short of its reach a spot must be from where it will stand (walks end a little off)
const REACH_SLACK := 0.15         # m: the navmesh lies a little above the floor it is baked from
const PUT_GAP := 0.01             # m above the spot a prop is let go at
const HELD_EVERY := 0.25          # s between looking at what the NPC carries (geogen.87.2)        # s between looking up the groupmates (chunks stream NPCs in and out)

var world: WorldLoader
var definition := {}
var needs := {}                   # need -> 0..1
var rng := RandomNumberGenerator.new()
var body: Node3D
var home := PackedVector2Array()  # world XZ polygon
var home_y := 0.0
## Places outside home its activities' `visit` makes trips to: name -> {polygon (world XZ), y}.
var visits := {}
## Affordances more than this above or below the home's floor belong to another storey.
const FLOOR_TOLERANCE := 1.5
var step_height := 0.3
var clock := 0.0                  # seconds since spawn
var trace_enabled := false
var label: Label3D
## Per-NPC counters for tests: used {id: count}, decisions, failures,
## max_stall (s), outside (s outside home + margin), passes (doors).
var _queued := {}                 # an action asked for by name (do_action), taken at the next decision
var _noticed := {}                # report: the last stimulus it noticed {stimulus, subject, at, by, clock}
var _notice_point := Vector3.ZERO # ...where the head turns
var _notice_until := -INF         # ...until then
var _cut_until := -INF            # an interrupt is wanted until then (it waits for a pose to settle)
var _handled := {}                # prop uid -> the clock times it picked it up (reluctance, npc_wants.gd)
var _wants_seen := []             # report: the want options at its last decision [{id, score}]
var _wants_left := []             # report: want options it has had enough of (reluctance), at the last look
var _want_doing := {}             # report: the want it is on now (subject, place, score...)
var _intent := ""                 # where a step's `look:` has the head turned: subject | place | anchor | ""
var _intent_ctx := {}             # ...and the action context that names them
var _tried := {}                  # prop uid -> the spots (Vector3) it has put it at, for its wants (npc_wants.gd)
var _memory := {}                 # prop uid -> what it last perceived of it (npc_memory.gd; with `memory:`)
var _looked_at := -INF            # clock of its last look round (memory)
var stats := {"used": {}, "decisions": 0, "failures": [], "max_stall": 0.0, "outside": 0.0, "passes": 0}
var _stuck_at = null     # where the last wedged failure happened (see _unstick)
var _wedged := false     # the step failing now made no progress at all (_walk), not e.g. a busy stair

var _stack: Array[Dictionary] = []   # frames: {action, steps, index, started, ctx}
var _find_seen := []            # the last `find` step's candidates [id, scored distance] (for the trace)
var _carry_on := {}
var _set_out := {}               # {item, node}: a carried item's body set out on the table in front (set_out)
var _item_state := {}            # item id -> the state its body shows (show), while it is carried              # {uid, local}: a thing in both hands, held where it was against the chest when picked up
var _option := {}
var _recent := {}                    # option id -> clock when last finished
var _retry_at := {}                  # failed option id -> clock when it may be tried again
var _forced := ""                    # option id the next decision takes (force_option)
var relations := {}                  # how well it knows the NPCs it has done things with: stable ID -> 0..1 (`relations:`)
var _partner: GeogenNpc = null       # who it is doing something with (an offer taken up: either side)
var _answering := ""                 # ...as the one who offers: the action it runs meanwhile on its feet
var free_will := true                # false: with nothing it was told to do, it waits (it still sleeps by its routine)
var _orders: Array = []              # what it was told to do, in order: option ids (direct())
var _order_since := -1.0             # clock when the first of them began to wait for someone busy
var _ordered := false                # what it is doing now is the first of them
var _order_last := {}                # how the last one ended: {option, ok, reason}
var _chain := {}                      # the chain it is on (`planner:`): {goal, steps: [options], index, cost, score}
var _fill := 0.0                     # how much of the option's `advertises` its duration has given so far (0..1)
var _fill_total := 0.0               # the duration (s) that fills it; 0: all of it when the option is done
var _shape: CollisionShape3D
var _gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity")
var _pose := "stand"
var _pose_yaw := 0.0                 # world heading the current pose was set to (rad)
var animator: GeogenNpcAnimator = null   # the body's skeletal clips, poses and locomotion (null: no body)
## "from>to" (poses) -> {clip, times, positions, rotations}: the clip that moves the body between
## them and its root path, in the frame of where the NPC ends up (the clip's extras `root`).
var _cost_usec := 0                 # script time spent in _physics_process (report: cost_us per frame)
var _cost_frames := 0
# Every character (player and NPCs) as {node, pos, radius, npc}, gathered once a physics frame
# and shared by all NPCs (geogen.11.18): scanning the group and each body's shape per NPC per
# call grew with the square of the crowd.
var _frame_delta := 1.0 / 60.0       # this physics frame's delta (move_and_slide moves one of these)
var _skipped := 0.0                  # s of frames skipped while throttled, made up by the next step
var _phase := 0                      # which frames a throttled NPC steps on (alternates between NPCs)
static var _phases := 0
static var _crowd: Array[Dictionary] = []
static var _crowd_frame := -1
static var _crowd_grid := {}          # Vector2i cell -> Array of _crowd entries
static var _crowd_radius := 0.0       # the largest character radius
# Step state
var _timer := 0.0
var _path := PackedVector3Array()
var _waypoint := 0
var _best := INF
var _stall := 0.0
var _sidestep := 0.0
var _sidestep_dir := 1.0
var _yaw_goal := 0.0
var _detour := -1                 # index in _path of the last detour waypoint
var _vehicle_wait := 0.0          # s spent waiting for traffic at this waypoint
var _assertive := -1              # waypoint we're walking to regardless of moving traffic
var _vehicle_detours := 0         # detours round stopped vehicles on this path
var _signal_wait := 0.0           # s spent on the kerb waiting for the walk signal this time
var _streets := GeogenNpcStreets.new()   # the crossings and pedestrian signals (world set in _ready)
var _was_at := Vector2.ZERO       # where we were a frame ago (plan)
## Off the map during a routine's `away` block (vanished into a building), back when it ends.
var away := false
## Out of the world while its placement's `present` condition is false (a story has moved it
## on); back, where it was, when it holds again.
var absent := false
var _look: GeogenHeadLook = null   # turns the head toward the player (attention.look)
var _hands: GeogenHands = null     # hands onto what is really there: tables, door handles (hands.gd)
var _wave_run: GeogenMotionRig.Run = null   # a wave from a seat (greet)
var _handle_at: GeogenMotionRig.Target = null   # the door handle the hand is going for
var _motions: GeogenMotionPlayer = null   # the body's motion rig, made when first needed (_motion_player)
var _run: GeogenMotionRig.Run = null      # the motion the step in hand waits for (`do:`, a press)
var _kept: Array[GeogenMotionRig.Run] = []   # motions started with `keep: true`: until the option ends
var _pressing := false             # a `use:` step's finger is on its way to the interaction's touch point
var _taking := false               # a `pick_up:` / `put_down:` step's hand is at work (motions take, set_down)
var _carry_run: GeogenMotionRig.Run = null   # the hand holding what it took (motion carry)
var _emitted := {}                 # what the body's motions have emitted since the press began
var _feet: GeogenFeet = null       # standing feet onto the ground that is really there (feet.gd)
var _playing := false             # a `play:` step holds the body's animation
var _close := false               # the walk's last bit goes off the navmesh, up to the target (`close`)
var _looping := false             # the `play:` step loops its clip: busy at work, it only glances up
var _glance_at := 0.0             # clock of the next glance up while busy
var _glance_until := -INF         # clock the glance up ends
var _greeted_at := -INF           # clock of the last greeting taken (attention.greet debounce)
var _waved_at := -INF             # clock of the last wave back (attention.greet cooldown)
var _attend_until := -INF         # keep looking at whoever greeted us until then (acknowledge)
const BOB_HEIGHT := 0.06            # m a body without a skeleton hops to acknowledge a greeting
var _yielded_at := -INF
const YIELD_COOLDOWN := 3.0       # s between stepping aside
const YIELD_FAR := 6.0            # m: beyond the yield distance plus this, nobody is walking at us (0.6 s at 10 m/s)
const GOAL_STANDOFF := 0.6        # m in front of a thing to stand when using it for a quest
const KERB_STEP := 0.7            # wait for the walk signal when the road of a crossing is this close ahead (m)
const VANISH_DISTANCE := 12.0     # don't vanish in front of the player: wait until they're this far


## Build an NPC from an exported npc node (its body child moves under the NPC).
static func spawn(world_: WorldLoader, node: Node3D, data: Dictionary) -> GeogenNpc:
    var npc := GeogenNpc.new()
    npc.world = world_
    npc.definition = data
    npc.name = "%s_npc" % node.name
    var xform := node.global_transform
    var yaw := atan2(xform.basis.z.x, xform.basis.z.z)
    node.get_parent().add_child(npc)
    npc.global_transform = Transform3D(Basis(Vector3.UP, yaw), xform.origin)
    npc.body = node.get_node_or_null(str(data.get("body", {}).get("node", "body")))
    if npc.body != null:
        npc.animator = GeogenNpcAnimator.load(npc.body)
        npc.body.reparent(npc, false)
        npc.body.transform = Transform3D.IDENTITY
        npc.animator.attach()
        var look: Dictionary = data.get("attention", {}).get("look", {})
        var skeletons := npc.body.find_children("*", "Skeleton3D", true, false)
        if not skeletons.is_empty():
            npc._skeleton = skeletons[0]
            npc._look = GeogenHeadLook.new()
            npc._look.name = "HeadLook"
            if not look.is_empty():
                npc._look.yaw_limit = deg_to_rad(float(look["yaw"]))
                npc._look.pitch_limit = deg_to_rad(float(look["pitch"]))
                npc._look.speed = float(look["speed"])
            skeletons[0].add_child(npc._look)
            var rests := {}             # hands on the real table top or sill, not the default one
            for clip in WorldLoader.geogen_extras(npc.body).get("clips", []):
                if clip.has("rests"):
                    rests[clip["name"]] = clip["rests"]
            npc._hands = GeogenHands.attach(skeletons[0], rests, npc.animator.current, npc.get_rid())
            npc._feet = GeogenFeet.attach(skeletons[0], npc._feet_free, npc.get_rid())   # feet on steps and slopes
    var h: Dictionary = data.get("home", {})
    for p in h.get("polygon", []):
        var w := xform * Vector3(p[0], 0.0, p[1])
        npc.home.append(Vector2(w.x, w.z))
    npc.home_y = (xform * Vector3(0, float(h.get("y", 0.0)), 0)).y
    var spots: Dictionary = data.get("visits", {})
    for name in spots:
        var poly := PackedVector2Array()
        for p in spots[name]["polygon"]:
            var w := xform * Vector3(p[0], 0.0, p[1])
            poly.append(Vector2(w.x, w.z))
        npc.visits[name] = {"polygon": poly, "y": (xform * Vector3(0, float(spots[name]["y"]), 0)).y}
    node.set_meta("geogen_npc", npc)
    return npc


func _ready() -> void:
    _phase = _phases
    _phases = (_phases + 1) % 2
    _streets.world = world
    collision_layer = LAYER
    collision_mask = 1 | LAYER | Player.LAYER
    add_to_group(CHARACTER_GROUP)
    floor_snap_length = step_height
    floor_block_on_wall = true
    _shape = CollisionShape3D.new()
    var cyl := CylinderShape3D.new()   # flat bottom: climbs steps (see Player)
    cyl.radius = float(definition.get("radius", 0.25))
    cyl.height = float(definition.get("height", 1.75))
    cyl.margin = Player.SHAPE_MARGIN
    _shape.shape = cyl
    _shape.position.y = cyl.height / 2.0
    add_child(_shape)
    rng.seed = hash("%s/%d" % [name, int(definition.get("seed", 0))])
    for need in definition.get("needs", {}):
        needs[need] = float(definition["needs"][need].get("initial", 0.5))
    label = Label3D.new()
    label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
    label.position.y = cyl.height + 0.35
    label.font_size = 28
    label.pixel_size = 0.004
    label.no_depth_test = true
    label.visible = false
    add_child(label)
    if not (definition.get("reacts", {}) as Dictionary).is_empty() and world != null and world.events != null:
        world.events.posted.connect(_on_event)      # perception (geogen.99): only for who reacts to something


# --- deciding -----------------------------------------------------------

## Every option this NPC could take now, scored (best first).
## The routine blocks active at the world clock's hour.
func routine_blocks() -> Array:
    var out := []
    if world == null or world.clock == null:
        return out
    var h: float = world.clock.hours
    for b in definition.get("routine", []):
        var a := float(b["from"])
        var z := float(b["to"])
        if (a <= z and h >= a and h < z) or (a > z and (h >= a or h < z)):
            out.append(b)
    return out


## Whether an affordance is one a `hold` block keeps the NPC on: one of the block's `only`
## (its bed), not what it fell back to when it couldn't get there (it tries again after that).
func _held_on(a: Dictionary) -> bool:
    var uid := WorldLoader.uid_of(a["node"])
    for b in routine_blocks():
        if not b.get("hold", false):
            continue
        var only: Array = b.get("only", [])
        if only.is_empty() or only.any(func(t): return t in a.get("tags", []) \
                or (t == "owns" and _under(uid, definition.get("owns", [])))):
            return true
    return false


func _away_block() -> bool:
    return routine_blocks().any(func(b): return b.get("away", false))


## An active block that `hold`s: the NPC stays on what it took until the block ends (its bed).
func _hold_block() -> bool:
    return routine_blocks().any(func(b): return b.get("hold", false))


## Back from an `away` block: at an entrance if there is one, else somewhere in the NPC's haunts.
func _reappear() -> void:
    var at = null
    var exits := world.exit_points()
    if not exits.is_empty():
        at = exits[rng.randi() % exits.size()]
    if at == null:
        at = _random_home_point()
    if at == null:
        return      # try again next frame (navmesh not ready)
    global_position = at
    away = false
    visible = true
    _solid_after_move()
    add_to_group(CHARACTER_GROUP)
    stats["returns"] = int(stats.get("returns", 0)) + 1
    _trace({"event": "return", "pos": _v(global_position)})


## A routine block's `only` keeps them to affordances with one of its tags (`owns`: the NPC's
## own things) and none of its activities; with nothing to take, they are as without it.
## ``chains``: for the planner, affordances whose `requires` don't hold now are options too
## (a chain's earlier step may see to that).
func options(restricted := true, chains := false, told := false) -> Array[Dictionary]:
    var result: Array[Dictionary] = []
    var actions: Dictionary = definition.get("actions", {})
    var margin := float(definition.get("home_margin", 1.0))
    var only: Array = definition.get("affordance_tags", [])
    var block_only := []
    if restricted:
        for b in routine_blocks():
            block_only.append_array(b.get("only", []))
    # A routine block `at: work`: the affordances under the NPC's work, wherever that is.
    var work := str(definition.get("work", ""))
    var at_work := work != "" and routine_blocks().any(func(b): return b.get("at", "") == "work")
    for a in world.affordances + world.affordance_uses:
        if not actions.has(a.get("action", "")) or not is_instance_valid(a.get("node")):
            continue
        if a.has("follow") and not a.get("boardable", true):
            continue    # a ride's seat, away from the ground or moving
        if a.has("prop"):
            a = _prop_affordance(a)
            if a.is_empty():
                continue    # someone has it, or it is nowhere this body reaches
        if not only.is_empty() and not only.any(func(t): return t in a.get("tags", [])):
            continue
        if (definition.get("skips", []) as Array).any(func(t): return t in a.get("tags", [])):
            continue                # left to someone else
        if not told and clock < float(_retry_at.get(a["id"], -1.0)):
            continue                # (told to: whoever said so may insist)
        var uid := WorldLoader.uid_of(a["node"])
        if _owned_by_another(uid):
            continue    # someone's armchair
        if not block_only.is_empty() and not block_only.any(func(t): return t in a.get("tags", []) \
                or (t == "owns" and _under(uid, definition.get("owns", [])))):
            continue    # not what this time of day is for (the bed at night)
        if at_work:
            if not _under(uid, [work]):
                continue
        elif home_distance(a["approach"]) > margin:
            continue
        elif absf((a["approach"] as Vector3).y - home_y) > FLOOR_TOLERANCE:
            continue    # another storey above or below the home region
        if world.npc_reserved(a["id"], self) >= int(a.get("slots", 1)):
            continue
        var gate = a.get("interaction")
        if gate is GeogenInteraction and not (gate as GeogenInteraction).allowed(WorldLoader.actor_id(self)):
            continue    # its `when` says no for this actor (a bin with nothing to throw away)
        if a.has("requires") and not chains and not _meets(a):
            continue    # it needs something carried, or a state, it hasn't got (geogen.101.1)
        if not _for_me(a, chains):
            continue    # a link in a chain, to one who doesn't plan; or for needs it hasn't got
        if _poses_at_anchor(actions[a["action"]]) and _something_on(a["position"], uid):
            continue    # a seat or a bed with a thing lying on it: it sits somewhere else
        result.append({"id": a["id"], "action": a["action"], "affordance": a,
            "advertises": a.get("advertises", {}), "tags": a.get("tags", []), "at": a["approach"]})
    if not block_only.is_empty() and result.is_empty():
        return options(false, chains, told)
    var activities: Dictionary = definition.get("activities", {}) if block_only.is_empty() else {}
    for n in activities:
        var act: Dictionary = activities[n]
        if clock < float(_retry_at.get("self/%s" % n, -1.0)):
            continue    # failed, or done and cooling down (a rare trip)
        if act.has("visit") and not visits.has(act["visit"]):
            continue
        if act["action"] == "rejoin" and (_rejoin_target() == null or _near_groupmate()):
            continue    # nobody to go back to, or with them already
        result.append({"id": "self/%s" % n, "action": act["action"], "activity": act,
            "advertises": act.get("advertises", {}), "tags": [], "at": global_position})
    if block_only.is_empty():
        result.append_array(GeogenWants.options(self))      # what its wants are unhappy about
        result.append_array(_people_options(actions, margin, told))   # what the people around offer
    var w: Dictionary = definition.get("scoring", {})
    var prefs: Dictionary = definition.get("preferences", {}).duplicate()
    var act_mult := {}
    for b in routine_blocks():      # routines bias the choice by time of day
        for t in b.get("preferences", {}):
            prefs[t] = float(prefs.get(t, 1.0)) * float(b["preferences"][t])
        for n in b.get("activities", {}):
            act_mult["self/%s" % n] = float(act_mult.get("self/%s" % n, 1.0)) * float(b["activities"][n])
    var lead = _group_lead()
    var lead_at: Vector3 = _heading_for(lead) if lead != null else Vector3.ZERO
    for o in result:
        var utility := float(o.get("base", 0.0))
        var plain := 0.0
        for need in o["advertises"]:
            if needs.has(need):
                plain += GeogenPlanner.urgency(needs[need], definition["needs"][need]) * float(o["advertises"][need])
        utility += plain
        var before := utility
        for tag in o["tags"]:
            utility *= float(prefs.get(tag, 1.0))
        if o.has("person"):
            utility *= (0.5 + knows(o["person"])) * float(o["offer"].get("weight", 1.0))
        if o.has("affordance"):
            var uid := WorldLoader.uid_of(o["affordance"]["node"])
            if _under(uid, definition.get("owns", [])):
                utility *= OWN_BONUS
            elif _under(uid, definition.get("prefers", [])):
                utility *= PREFER_BONUS
            if _beside_groupmate(o["affordance"]) or _beside_friend(o["affordance"]):
                utility *= GROUP_BONUS    # the seat next to a friend's, the same gondola
            if lead != null and NpcGeometry.flat(o["at"] - lead_at).length() > _group_radius():
                utility *= GROUP_FAR      # a member keeps near its leader; the leader goes where it likes
        utility *= float(act_mult.get(o["id"], 1.0)) * float(o.get("activity", {}).get("weight", 1.0))
        var distance := global_position.distance_to(o["at"])
        var recency := 0.0
        if _recent.has(o["id"]) and not o.has("want"):     # a want is about now: what fell again is a new problem
            recency = exp(-(clock - float(_recent[o["id"]])) / maxf(float(w.get("memory", 120.0)), 1.0))
        # For the planner: how much it likes this option (every multiplier above), less for what it has just done.
        o["liking"] = (utility / before if before > 0.0 else 1.0) * maxf(0.1, 1.0 - float(w.get("recency", 0.4)) * recency)
        o["score"] = utility - float(w.get("distance", 0.02)) * distance \
            - float(w.get("recency", 0.4)) * recency + float(w.get("noise", 0.05)) * rng.randf()
    result.sort_custom(func(x, y): return x["score"] > y["score"])
    return result


## What the people around offer (`offers:`, geogen.101.6): an option `with/<their stable ID>/<offer>`
## for each offer of every NPC within this one's home that is free for it now, and that fills a
## need this one has room in.
func _people_options(actions: Dictionary, margin: float, told := false) -> Array[Dictionary]:
    var result: Array[Dictionary] = []
    if _partner != null:
        return result
    for other in world.npcs:
        if other == self or not is_instance_valid(other) or not other.definition.has("offers"):
            continue
        if not other.free_for_company() or home_distance(other.global_position) > margin \
                or absf(other.global_position.y - home_y) > FLOOR_TOLERANCE:
            continue
        var offers: Dictionary = other.definition["offers"]
        for n in offers:
            var offer: Dictionary = offers[n]
            var id := "with/%s/%s" % [WorldLoader.npc_uid(other), n]
            if not actions.has(offer["action"]) or (not told and clock < float(_retry_at.get(id, -1.0))):
                continue
            # Only for a need with room for half of what it gives: nobody is bothered for a top-up
            # (unless it was told to).
            if not told and not (offer["advertises"] as Dictionary).keys().any(func(need): return needs.has(need) \
                    and needs[need] <= 1.0 - COMPANY_ROOM * float(offer["advertises"][need])):
                continue
            result.append({"id": id, "action": offer["action"], "person": other, "offer": offer,
                "advertises": offer["advertises"], "tags": ["social"], "at": other.global_position})
    return result


## Whether another NPC may take up one of this one's offers now: here, awake, with nobody
## already, and either seated (it stays seated) or on its feet with nothing in hand to finish
## (wandering, idling: not halfway through a plan).
func free_for_company() -> bool:
    if away or absent or _partner != null or not _orders.is_empty() or _hold_block() or _moving_pose() or _stack.size() > 1 or not _ride.is_empty():
        return false
    if _pose == "sit":
        return true
    return _pose == "stand" and _chain.is_empty() and (_option.is_empty() or _option.has("activity"))


## How well this NPC knows another, 0..1 (`relations:`): what doing things together has made it,
## or to start with `household` for one of its household and `stranger` for anyone else.
func knows(other: Node) -> float:
    var spec: Dictionary = definition.get("relations", {})
    if spec.is_empty() or not other is GeogenNpc:
        return 0.0
    var uid := WorldLoader.npc_uid(other)
    if relations.has(uid):
        return float(relations[uid])
    var mine := str(definition.get("household", ""))
    return float(spec["household"] if mine != "" and mine == str((other as GeogenNpc).definition.get("household", "")) else spec["stranger"])


## Hold each other for an offer this NPC takes up; the other, on its feet, turns to wait.
func _take_up(other: GeogenNpc, offer: Dictionary) -> void:
    _partner = other
    other._partner = self
    other._answering = str(offer.get("answer", ""))
    _trace({"event": "take_up", "with": WorldLoader.npc_uid(other), "pose": other._pose})
    other._answer()


## As the one whose offer was taken up: on its feet, run the offer's `answer` on top of what it
## was doing (seated, it stays as it is; it answers when it gets up, if the other is still at it).
func _answer() -> void:
    if _pose == "stand" and definition["actions"].has(_answering):
        _react(_answering, {"partner": _partner})


## Let go of the one it is with (either side); they are free too.
func _part() -> void:
    var other := _partner
    _partner = null
    _answering = ""
    if is_instance_valid(other) and other._partner == self:
        other._partner = null
        other._answering = ""


## From the one who took up this NPC's offer: its share of what the offer gives, as theirs fills.
func _share(advertises: Dictionary, share: float, from: GeogenNpc) -> void:
    for need in advertises:
        if needs.has(need):
            needs[need] = clampf(needs[need] + float(advertises[need]) * share, 0.0, 1.0)
    _know_better(from, share)
    if _pose == "sit":                  # seated, the head turns to them (standing, its answer's `look:` does)
        _notice_point = from.head_point()
        _notice_until = clock + 1.0


func _know_better(other: GeogenNpc, share: float) -> void:
    var spec: Dictionary = definition.get("relations", {})
    if not spec.is_empty():
        relations[WorldLoader.npc_uid(other)] = clampf(knows(other) + float(spec["grow"]) * share, 0.0, 1.0)


## Whether someone it knows well (`relations.friend`) holds this affordance's thing or one beside it.
func _beside_friend(a: Dictionary) -> bool:
    var spec: Dictionary = definition.get("relations", {})
    if spec.is_empty():
        return false
    for o in world.npcs:
        if o == self or not is_instance_valid(o) or knows(o) < float(spec["friend"]):
            continue
        var theirs: Dictionary = o._option.get("affordance", {})
        if theirs.is_empty() or not is_instance_valid(theirs.get("node")):
            continue
        if theirs["node"] == a["node"] or (theirs["position"] as Vector3).distance_to(a["position"]) < GROUP_BESIDE:
            return true
    return false


## Whether an affordance is anything to this NPC. One that only `gives` (advertises nothing)
## is a link in a chain: for those who plan (``chains``). One that advertises only needs the NPC
## hasn't got does nothing for it (a shower, to someone with no `hygiene`).
func _for_me(a: Dictionary, chains: bool) -> bool:
    var advertises: Dictionary = a.get("advertises", {})
    if advertises.is_empty():
        return chains or not a.has("gives")
    if chains:
        return true
    for need in advertises:
        if needs.has(need):
            return true
    return false


## Whether an action puts the body on its anchor (a `pose` step other than standing: sit, lie).
static func _poses_at_anchor(action: Dictionary) -> bool:
    return (action["steps"] as Array).any(func(s): return s.has("pose") and not s["pose"] in ["stand", "found"] \
        and str(s.get("at", "anchor")) == "anchor")


## Whether a loose physics prop lies where a body would go at ``anchor`` (a seat, a mattress)
## of thing ``seat``: one resting on that thing there, not on the table the chair is tucked under.
func _something_on(anchor: Vector3, seat: String) -> bool:
    if world.props == null:
        return false
    for uid in world.props.props:
        var body: Node3D = world.props.props[uid]["body"]
        if not is_instance_valid(body) or not body.is_visible_in_tree() or world.props.holder_of(uid) != null:
            continue
        var at := body.global_position
        if absf(at.x - anchor.x) > ON_SEAT or absf(at.z - anchor.z) > ON_SEAT:
            continue                # (most props, most of the time)
        var centre := _prop_centre(uid)
        if NpcGeometry.flat(centre - anchor).length() > ON_SEAT or centre.y < anchor.y - 0.1:
            continue
        var under := str(GeogenPlaceSearch.resting_on(world, uid).get("uid", ""))
        if _under(under, [seat]) or _under(seat, [under]):
            return true
    return false


static func _under(uid: String, ids: Array) -> bool:
    if uid == "":
        return false
    for id in ids:
        if uid == str(id) or uid.begins_with(str(id) + "/"):
            return true
    return false


## Whether another NPC owns a thing and this one doesn't (definition `owns`: only its owners
## use it; a couple both own their bed).
func _owned_by_another(uid: String) -> bool:
    if _under(uid, definition.get("owns", [])):
        return false
    for other in world.npcs:
        if other != self and is_instance_valid(other) and _under(uid, other.definition.get("owns", [])):
            return true
    return false


## Make the next decision take option ``id`` (an affordance id or "self/<activity>") if it is
## still available then. False if it isn't one of the options now.
func force_option(id: String) -> bool:
    if not options().any(func(o): return o["id"] == id):
        return false
    _forced = id
    return true


## A quest this NPC holds (geogen.15.8): the nearest goal of an open objective of an active quest
## (the manifest's objective `goals`: enter a place, use an interaction, use one until it shows a
## state) that isn't met yet and hasn't just failed, as an option with its action context; {}
## when there's none.
func quest_option() -> Dictionary:
    # (Not before the navmesh has synced: there'd be no path yet, and the goal would wait a retry.)
    if world.rules == null or NavigationServer3D.map_get_iteration_id(get_world_3d().navigation_map) == 0:
        return {}
    var best := {}
    var nearest := INF
    # A routine block's quests are worked on only while the block lasts (its rules give them).
    var routine_quests := {}
    for b in definition.get("routine", []):
        for id in b.get("quests", []):
            routine_quests[id] = false
    for b in routine_blocks():
        for id in b.get("quests", []):
            routine_quests[id] = true
    for quest in world.rules.report(WorldLoader.actor_id(self)):
        if quest["status"] != "active" or not routine_quests.get(quest["id"], true):
            continue
        for objective in quest["objectives"]:
            if objective["state"] != "open":
                continue
            for goal in objective.get("goals", []):
                if goal.get("greet", false):
                    continue            # greeting an NPC: the player's quest bot, not an NPC holder
                var id := "quest/%s/%s/%s" % [quest["id"], objective["id"], goal["subject"]]
                if clock < float(_retry_at.get(id, -1.0)):
                    continue
                var ctx := _goal_context(goal)
                if ctx.is_empty():
                    continue
                var d := global_position.distance_to(ctx["approach"])
                if d < nearest:
                    nearest = d
                    best = {"id": id, "action": "quest_go" if goal.get("enter", false) else "quest_use",
                        "advertises": {}, "tags": [], "at": ctx["approach"], "quest": quest["id"], "ctx": ctx}
    return best


## Where to go and what to work for one quest goal; {} if the goal is met or its subject isn't here.
func _goal_context(goal: Dictionary) -> Dictionary:
    var subject := str(goal["subject"])
    if goal.get("enter", false):
        var place = world.node_by_uid(subject)
        if not place is Node3D:
            return {}
        var at := NavigationServer3D.map_get_closest_point(get_world_3d().navigation_map, (place as Node3D).global_position)
        return {"anchor": at, "approach": at, "yaw_deg": rad_to_deg(_yaw()), "duration": [0.0, 0.0]}
    var it = world.interactions_by_key.get("%s#%s" % [subject, goal["interaction"]])
    if not it is GeogenInteraction or not is_instance_valid(it) or it.asset == null:
        return {}
    var state := str(goal["state"]) if goal.has("state") else str(it.states.get(it.target, {}).get("next", ""))
    if goal.has("state") and it.state == state and it.target == state:
        return {}                       # done already
    for p in world.portals:
        if p["interaction"] != it:
            continue
        # A door: stand where one going through it would, on the side its leaf swings away from
        # (the side this NPC is on, when it swings neither way), near enough for the handle.
        var centre: Vector3 = p["center"]
        var side := -float(int(p.get("swing", 0)))
        if side == 0.0:
            side = 1.0 if (global_position - centre).dot(p["normal"]) >= 0.0 else -1.0
        var clear: float = p["clearance"]
        if _hands != null and (p.get("handles", {}) as Dictionary).has(int(side)):
            clear = minf(clear, HANDLE_STAND)
        var stand := NavigationServer3D.map_get_closest_point(get_world_3d().navigation_map,
            centre + (p["normal"] as Vector3) * side * (float(p["depth"]) / 2.0 + clear))
        var to_door := NpcGeometry.flat(centre - stand)
        return {"anchor": centre, "approach": stand, "interaction": it, "state": state, "portal": p, "side": side,
            "yaw_deg": rad_to_deg(atan2(to_door.x, to_door.z)) if to_door.length() > 1e-3 else rad_to_deg(_yaw()),
            "duration": [0.4, 0.8]}
    var thing: Node3D = it.targets[0] if not it.targets.is_empty() else it.asset
    var anchor := thing.global_position
    # Things face +Z (a wall switch out of its wall): stand in front of it, on the navmesh.
    var front := NpcGeometry.flat(it.asset.global_basis.z).normalized()
    var approach := NavigationServer3D.map_get_closest_point(get_world_3d().navigation_map, anchor + front * GOAL_STANDOFF)
    var look := NpcGeometry.flat(anchor - approach)
    var yaw := atan2(look.x, look.z) if look.length() > 1e-3 else _yaw()
    return {"anchor": anchor, "approach": approach, "interaction": it, "state": state,
        "yaw_deg": rad_to_deg(yaw), "duration": [0.4, 0.8]}


func _decide() -> void:
    if not _queued.is_empty():
        _option = {"id": "do/%s" % _queued["action"], "action": _queued["action"], "advertises": {}}
        var asked := {"anchor": global_position, "yaw_deg": rad_to_deg(_yaw()), "approach": global_position}
        asked.merge(_queued["extra"], true)
        _queued = {}
        stats["decisions"] += 1
        _trace({"event": "decide", "chosen": _option["id"]})
        _push(_option["action"], asked)
        return
    if _run_out():
        return
    if _partner != null and _answering != "":
        _answer()                   # got up while the one who came to talk is still at it
        return
    while not _orders.is_empty():       # (one that can't be done leaves the queue: on to the next)
        if _take_order():
            return
    if not free_will and not _hold_block() and definition["actions"].has("idle"):
        _option = {}                    # nothing of its own: stand by until told
        _push("idle", {"anchor": global_position, "yaw_deg": rad_to_deg(_yaw()), "approach": global_position,
            "duration": [1.0, 1.0]})
        return
    var goal := quest_option() if _forced == "" else {}
    if not goal.is_empty():
        stats["decisions"] += 1
        stats["quest_goals"] = int(stats.get("quest_goals", 0)) + 1
        _option = goal
        _trace({"event": "decide", "chosen": goal["id"], "quest": goal["quest"]})
        _push(goal["action"], goal["ctx"])
        return
    var planned := definition.has("planner") and _forced == ""
    var ranked := options(true, planned)
    stats["decisions"] += 1
    _wants_seen = ranked.filter(func(o): return o.has("want")).map(
        func(o): return {"id": o["id"], "score": snappedf(o["score"], 0.001)})
    if ranked.is_empty():
        _drop_plan("nothing to do")
        return
    if planned:
        _option = _next_planned(ranked)
        if _option.is_empty():
            return                  # nothing worth doing now: look again next frame
    else:
        _option = ranked[0]
    if _forced != "":
        for o in ranked:
            if o["id"] == _forced:
                _option = o
        _forced = ""
    _start(ranked)


## Set off on the option chosen (`_option`): it is this NPC's, and its action runs.
func _start(ranked: Array[Dictionary]) -> void:
    if _option.has("affordance"):
        world.npc_reserve(_option["id"], self)
    var ctx := {}
    var a: Dictionary = _option.get("affordance", {})
    if _option.has("want"):
        ctx = GeogenWants.begin(self, _option)
        if ctx.is_empty():
            _trace({"event": "decide", "chosen": _option["id"]})
            _finish_option(false, "nowhere to put it")
            return
    elif _option.has("person"):
        var other: GeogenNpc = _option["person"]
        if not is_instance_valid(other) or not other.free_for_company():
            _trace({"event": "decide", "chosen": _option["id"]})
            _finish_option(false, "they are busy")
            return
        _take_up(other, _option["offer"])
        ctx = {"anchor": other.global_position, "yaw_deg": rad_to_deg(_yaw()), "approach": other.global_position,
            "duration": _option["offer"]["duration"], "partner": other}
    elif not a.is_empty():
        ctx = {"anchor": a["position"], "yaw_deg": a["npc_yaw_deg"], "approach": a["approach"],
            "interaction": a.get("interaction"), "duration": a.get("duration"), "depth": a.get("depth"),
            "held": _held_on(a), "ride": a if a.has("follow") else {}}
        if a.has("prop"):
            # The prop's own affordance: it is what the action takes, and where it lies is where
            # `put_down: place` puts it back.
            ctx["subject"] = a["prop"]
            ctx["place"] = _where_it_lies(a["prop"])
    else:
        ctx = {"anchor": global_position, "yaw_deg": rad_to_deg(_yaw()), "approach": global_position,
            "duration": _option["activity"].get("duration"), "visit": _option["activity"].get("visit", "")}
    _trace({"event": "decide", "chosen": _option["id"], "top": ranked.slice(0, 3).map(
        func(o): return {"id": o["id"], "score": snappedf(o["score"], 0.001)}),
        "needs": _rounded_needs()})
    _push(_option["action"], ctx)


## A need that has run out (its `at_zero`: {action, recover}): what happens then is run where the
## NPC stands, before anything else, and gives `recover` of the need back (geogen.101.2).
func _run_out() -> bool:
    for need in needs:
        var zero: Dictionary = definition["needs"][need].get("at_zero", {})
        if zero.is_empty() or needs[need] > 0.0 or not definition["actions"].has(zero["action"]):
            continue
        _drop_plan("%s ran out" % need)
        _option = {"id": "need/%s" % need, "action": str(zero["action"]), "advertises": {need: float(zero["recover"])}}
        stats["decisions"] += 1
        stats["ran_out"] = int(stats.get("ran_out", 0)) + 1
        _trace({"event": "decide", "chosen": _option["id"], "reason": "ran out"})
        _push(_option["action"], {"anchor": global_position, "yaw_deg": rad_to_deg(_yaw()), "approach": global_position})
        return true
    return false


## The next step of the plan it is on, or of a new one (`planner:`, geogen.101.4): the option
## among ``ranked`` to take now, {} when there is no plan to be had. A plan is a chain of
## options found by GeogenPlanner over what they require, take and give; every affordance in
## it is reserved while the plan lasts, and a step that is no longer possible drops the plan.
func _next_planned(ranked: Array[Dictionary]) -> Dictionary:
    var by_id := {}
    for o in ranked:
        by_id[o["id"]] = o
    if not _chain.is_empty():
        var step: Dictionary = _chain["steps"][_chain["index"]]
        var now: Dictionary = by_id.get(step["id"], {})
        if not now.is_empty() and (not now.has("affordance") or _meets(now["affordance"])):
            return now
        _drop_plan("%s isn't possible any more" % step["id"])
    var found := GeogenPlanner.plan(_snapshot(ranked))
    if not (found.get("waits", []) as Array).is_empty():
        found = {}                  # someone else is about to make it possible: something else meanwhile
    if found.is_empty():
        for o in ranked:            # nothing fills a need: the best of what can be done as things are
            if not o.has("affordance"):
                return o
            var a: Dictionary = o["affordance"]
            if (a.get("advertises", {}) as Dictionary).is_empty() and (a.has("gives") or a.has("sets")):
                continue            # a link in a chain, with no chain to be part of
            if a.has("takes") and not _carries_one_of(a["takes"]):
                continue            # it would get there with nothing to use
            if not a.has("requires") or _meets(a):
                return o
        return {}
    # Who works with others commits to one run at a time, and leaves the rest of the job open.
    var steps: Array = (found.get("run", found["steps"]) as Array).map(func(id): return by_id[id])
    for o in steps:
        if o.has("affordance"):
            world.npc_reserve(o["id"], self)
    _chain = {"goal": found["goal"], "steps": steps, "index": 0, "cost": found["cost"], "score": found["score"]}
    stats["plans"] = int(stats.get("plans", 0)) + 1
    _trace({"event": "plan", "goal": found["goal"], "steps": found["steps"], "cost": found["cost"],
        "score": found["score"], "fills": found["fills"], "goals": found["goals"]})
    return steps[0]


## Whether it carries one of these items.
func _carries_one_of(items: Array) -> bool:
    var actor := WorldLoader.actor_id(self)
    return world.rules != null and items.any(func(id): return world.rules.held(str(id), actor) > 0)


## The states the steps of its plan still to finish will set: {"<uid>#<interaction>": state}.
func promises() -> Dictionary:
    var out := {}
    if _chain.is_empty():
        return out
    for i in range(int(_chain["index"]), (_chain["steps"] as Array).size()):
        var a: Dictionary = _chain["steps"][i].get("affordance", {})
        for name in a.get("sets", {}):
            out["%s#%s" % [WorldLoader.uid_of(a["node"]), name]] = str(a["sets"][name])
    return out


## What the planner is asked: the options as its affordances, the states they name, and this NPC.
func _snapshot(ranked: Array[Dictionary]) -> Dictionary:
    var actor := WorldLoader.actor_id(self)
    var listed := []
    var states := {}
    for o in ranked:
        var span = null
        var entry := {"id": o["id"], "advertises": o["advertises"], "weight": float(o.get("liking", 1.0)),
            "worth": float(o.get("base", 0.0))}
        if o.has("affordance"):
            var a: Dictionary = o["affordance"]
            var at: Vector3 = a["approach"]
            var uid := WorldLoader.uid_of(a["node"])
            entry["at"] = [at.x, at.y, at.z]
            span = a.get("duration")
            for key in ["takes", "gives"]:
                if a.has(key):
                    entry[key] = a[key]
            var requires := {}
            if a.get("requires", {}).has("holding"):
                requires["holding"] = a["requires"]["holding"]
            for part in ["requires", "sets"]:
                var named: Dictionary = a.get(part, {}).get("state", {}) if part == "requires" else a.get("sets", {})
                var keyed := {}
                for name in named:
                    var key := "%s#%s" % [uid, name]
                    keyed[key] = str(named[name])
                    var it = world.interactions_by_key.get(key)
                    if it is GeogenInteraction:
                        states[key] = (it as GeogenInteraction).state
                if keyed.is_empty():
                    continue
                if part == "requires":
                    requires["state"] = keyed
                else:
                    entry["sets"] = keyed
            if not requires.is_empty():
                entry["requires"] = requires
        elif o.has("want") or o.has("person"):
            var at: Vector3 = o["at"]
            entry["at"] = [at.x, at.y, at.z]
            if o.has("person"):
                span = o["offer"]["duration"]
        else:
            span = o["activity"].get("duration")
        if span == null:
            span = definition["actions"][o["action"]].get("duration", [3.0, 6.0])
        entry["duration"] = (float(span[0]) + float(span[1])) / 2.0
        listed.append(entry)
    var holding := {}
    if world.rules != null:
        for e in world.rules.carried(actor):
            holding[str(e["id"])] = int(e["count"])
    var urgency := {}
    for need in needs:
        urgency[need] = GeogenPlanner.urgency(needs[need], definition["needs"][need])
    var p := global_position
    var promised := {}
    for other in world.npcs:
        if other != self and is_instance_valid(other):
            promised.merge(other.promises(), true)
    var me := {"at": [p.x, p.y, p.z], "speed": float(definition.get("speed", 1.2)), "holding": holding,
        "needs": needs, "urgency": urgency}
    if definition.get("planner", {}).has("hands"):
        me["hands"] = int(definition["planner"]["hands"])
    return {"affordances": listed, "states": states, "promised": promised,
        "limits": definition.get("planner", PLANNER_LIMITS), "actor": me}


## Give the plan up: its things are free again, and the next decision plans afresh.
func _drop_plan(reason: String) -> void:
    if _chain.is_empty():
        return
    for o in _chain["steps"]:
        world.npc_release(o["id"], self)
    stats["replans"] = int(stats.get("replans", 0)) + 1
    _trace({"event": "replan", "goal": _chain["goal"], "at": int(_chain["index"]), "reason": reason})
    _chain = {}


## How it is, in one number: the mean of its needs (1 content .. 0 wretched).
func mood() -> float:
    if needs.is_empty():
        return 1.0
    var total := 0.0
    for need in needs:
        total += needs[need]
    return total / needs.size()


func _push(action_name: String, ctx: Dictionary) -> void:
    var action: Dictionary = definition["actions"][action_name]
    if ctx.get("duration") == null:
        ctx["duration"] = action.get("duration", [3.0, 6.0])
    _stack.append({"action": action_name, "steps": action["steps"], "index": 0, "started": false, "ctx": ctx})


func _finish_option(ok: bool, reason := "") -> void:
    _stop_motions()
    if _pose == "stand":
        _ride = {}
    _intent = ""
    if _option.is_empty():
        return
    var id: String = _option["id"]
    _recent[id] = clock
    if ok and not _has_what_it_takes():
        ok = false                      # what it was to use up went missing on the way (taken, dropped)
        reason = "nothing left of what it takes"
    if ok:
        if _option.has("affordance"):
            _carry_out(_option["affordance"])
        _give(1.0 - _fill)              # what its duration hasn't given already (cut short, it still counts as done)
        stats["used"][id] = int(stats["used"].get(id, 0)) + 1
        if _option.has("activity") and _option["activity"].has("cooldown"):
            _retry_at[id] = clock + float(_option["activity"]["cooldown"])
        _trace({"event": "done", "option": id, "needs": _rounded_needs()})
    else:
        stats["failures"].append("%s: %s" % [id, reason])
        _retry_at[id] = clock + float(definition.get("scoring", {}).get("retry", 30.0))
        _trace({"event": "fail", "option": id, "reason": reason})
    _fill = 0.0
    _fill_total = 0.0
    _clear_set_out()
    world.npc_release(id, self)
    if _option.has("found"):
        world.npc_release(str(_option["found"]), self)
    if _option.has("want"):
        GeogenWants.finish(self, _option, ok)
    if _option.has("person"):
        if ok and is_instance_valid(_partner):
            _partner.stats["joined"] = int(_partner.stats.get("joined", 0)) + 1
        _part()
    if not _chain.is_empty() and _chain["steps"][_chain["index"]]["id"] == id:
        if not ok:
            _drop_plan(reason)
        else:
            _chain["index"] += 1
            if _chain["index"] >= (_chain["steps"] as Array).size():
                stats["plans_done"] = int(stats.get("plans_done", 0)) + 1
                _chain = {}
    _option = {}
    if _ordered:                        # the order it was on: over when it failed, or its last step is done
        _ordered = false
        if not ok:
            _order_done(false, reason)
        elif _chain.is_empty():
            _order_done(true, "")


# --- told what to do (geogen.101.8) ---------------------------------------------

## What this NPC could be told to do with a thing (stable ID; a person's: their offers): every
## affordance under it this NPC has an action for, as {option, label, available, why, plan,
## cost}. `plan` is the chain it would take (the option last: a meal means the fridge and the
## range first), `why` the reason when it can't be done now.
func choices(thing: String) -> Array:
    var result := []
    var ranked := options(false, true, true)
    var by_id := {}
    for o in ranked:
        by_id[o["id"]] = o
    var actions: Dictionary = definition.get("actions", {})
    for other in world.npcs:
        if is_instance_valid(other) and other != self and WorldLoader.npc_uid(other) == thing:
            var offers: Dictionary = other.definition.get("offers", {})
            for n in offers:
                if actions.has(offers[n]["action"]):
                    var id := "with/%s/%s" % [thing, n]
                    result.append(_choice(id, option_label(id), by_id, ranked, "" if by_id.has(id) else "they are busy"))
            return result
    for a in world.affordances + world.affordance_uses:
        if not is_instance_valid(a.get("node")) or not actions.has(a.get("action", "")):
            continue
        var uid := WorldLoader.uid_of(a["node"])
        if not _under(uid, [thing]):
            continue
        var why := ""
        if not by_id.has(a["id"]):
            if _owned_by_another(uid):
                why = "someone else's"
            elif world.npc_reserved(a["id"], self) >= int(a.get("slots", 1)):
                why = "in use"
            elif home_distance(a["approach"]) > float(definition.get("home_margin", 1.0)):
                why = "too far from home"
            else:
                why = "not now"
        var entry := _choice(a["id"], _choice_label(a), by_id, ranked, why)
        # One line for the same thing to do at several places of it (a sofa's seats): the first that can be.
        var same := result.filter(func(e): return e["label"] == entry["label"])
        if same.is_empty():
            result.append(entry)
        elif entry["available"] and not same[0]["available"]:
            result[result.find(same[0])] = entry
    return result


## What a menu calls an affordance: what it is for. A link in a chain is named by what it gives
## ("Get raw food"; "Make meal" when it takes something for it), anything else by its use or action.
static func _choice_label(a: Dictionary) -> String:
    var gives: Dictionary = a.get("gives", {})
    if not gives.is_empty() and (a.get("advertises", {}) as Dictionary).is_empty():
        return "%s %s" % ["Make" if a.has("takes") else "Get", str(gives.keys()[0]).replace("_", " ")]
    return _sentence(str(a.get("use", a["action"])))


## "sit_at_table" -> "Sit at table".
static func _sentence(text: String) -> String:
    text = text.replace("_", " ")
    return text.left(1).to_upper() + text.substr(1)


## What a menu calls one of this NPC's options (an option id): an affordance by what it is for,
## someone's offer as "Chat with <them>", something of its own by its name.
func option_label(id: String) -> String:
    if id.begins_with("with/"):
        var who := id.get_slice("/", 1)
        for other in world.npcs:
            if is_instance_valid(other) and WorldLoader.npc_uid(other) == who:
                who = other.person_name() if other.person_name() != "" else String(other.name).trim_suffix("_npc").capitalize()
        return "%s with %s" % [_sentence(id.get_slice("/", 2)), who]
    for a in world.affordances + world.affordance_uses:
        if a["id"] == id:
            return _choice_label(a)
    return _sentence(id.get_slice("/", id.get_slice_count("/") - 1))


func _choice(id: String, label: String, by_id: Dictionary, ranked: Array[Dictionary], why: String) -> Dictionary:
    var plan := []
    var cost := 0.0
    if why == "":
        var found := _order_plan(by_id[id], ranked)
        if found.is_empty():
            why = "what it takes is missing or in use"
        else:
            plan = found["steps"]
            cost = float(found["cost"])
    return {"option": id, "label": label, "available": why == "", "why": why, "plan": plan, "cost": snappedf(cost, 0.1)}


## The chain an order for option ``o`` takes (GeogenPlanner.plan_to: {goal, steps, cost, ...}); {} for none.
func _order_plan(o: Dictionary, ranked: Array[Dictionary]) -> Dictionary:
    return GeogenPlanner.plan_to(_snapshot(ranked), str(o["id"]))


## Tell this NPC to do something (an option id from choices()): it joins the queue of what it
## was told, done in order before anything of its own; what it was doing of its own accord is cut
## short. "" or why not.
func direct(id: String) -> String:
    if away or absent:
        return "not here"
    if not options(false, true, true).any(func(o): return o["id"] == id):
        return "'%s' is not something it can do now" % id
    _orders.append(id)
    _trace({"event": "order", "option": id, "queue": _orders.size()})
    if _orders.size() == 1 and not _ordered and str(_chain.get("goal", _option.get("id", ""))) == id:
        _ordered = true                 # at it already, of its own accord: now it is the order
        if not _chain.is_empty():
            _chain["ordered"] = true
    elif not _ordered and not (_chain.get("ordered", false)):
        _drop_plan("told to do something else")
        if not _stack.is_empty() and _stack[0]["ctx"].get("held", false):
            _stack[0]["ctx"]["held"] = false        # out of bed
        _cut_short()
    return ""


## Take back what it was told: order ``index`` of the queue, or all of them (-1). Cancelling
## the one it is on stops it there.
func cancel_orders(index := -1) -> int:
    var n := _orders.size()
    if index >= n:
        return 0
    var current: bool = index <= 0 and (_ordered or _chain.get("ordered", false))
    if index <= 0:
        _order_since = -1.0
    if index < 0:
        _orders.clear()
    else:
        _orders.remove_at(index)
    if current:
        _ordered = false
        _drop_plan("order taken back")
        _cut_short()
    return n - _orders.size()


## At a decision with orders waiting: set off on the first (the next step of its chain, or a
## new chain to it). False when it can't be done: that order is over, and the NPC decides for itself.
func _take_order() -> bool:
    var ranked := options(false, true, true)
    var by_id := {}
    for o in ranked:
        by_id[o["id"]] = o
    if _chain.get("ordered", false):
        var step: Dictionary = _chain["steps"][_chain["index"]]
        var now: Dictionary = by_id.get(step["id"], {})
        if now.is_empty() or (now.has("affordance") and not _meets(now["affordance"])):
            _drop_plan("%s isn't possible any more" % step["id"])
            _order_done(false, "%s isn't possible any more" % step["id"])
            return false
        _option = now
    else:
        _drop_plan("told to do something else")
        var id := str(_orders[0])
        if not by_id.has(id):
            # Someone who is busy just now (an offer of theirs): wait for them a while, where it stands.
            if id.begins_with("with/") and definition["actions"].has("idle"):
                if _order_since < 0.0:
                    _order_since = clock
                if clock - _order_since < ORDER_WAIT:
                    _option = {}
                    _push("idle", {"anchor": global_position, "yaw_deg": rad_to_deg(_yaw()),
                        "approach": global_position, "duration": [0.5, 0.5]})
                    return true
            _order_done(false, "can't be done now")
            return false
        var found := _order_plan(by_id[id], ranked)
        if found.is_empty():
            _order_done(false, "what it takes is missing or in use")
            return false
        var steps: Array = (found["steps"] as Array).map(func(step_id): return by_id[step_id])
        if steps.size() > 1:
            for o in steps:
                if o.has("affordance"):
                    world.npc_reserve(o["id"], self)
            _chain = {"goal": id, "steps": steps, "index": 0, "cost": found["cost"], "score": 0.0, "ordered": true}
            stats["plans"] = int(stats.get("plans", 0)) + 1
            _trace({"event": "plan", "goal": id, "steps": found["steps"], "cost": found["cost"], "ordered": true})
        _option = steps[0]
    _ordered = true
    stats["decisions"] += 1
    _start(ranked)
    return true


func _order_done(ok: bool, reason: String) -> void:
    if _orders.is_empty():
        return
    _order_since = -1.0
    _order_last = {"option": str(_orders.pop_front()), "ok": ok, "reason": reason}
    stats["orders_done" if ok else "orders_failed"] = int(stats.get("orders_done" if ok else "orders_failed", 0)) + 1
    _trace({"event": "order_done", "option": _order_last["option"], "ok": ok, "reason": reason})


## Whether the NPC still carries one of the things the option in hand `takes` (true when it takes nothing).
func _has_what_it_takes() -> bool:
    var takes: Array = _option.get("affordance", {}).get("takes", [])
    if takes.is_empty() or world.rules == null:
        return true
    var actor := WorldLoader.actor_id(self)
    return takes.any(func(id): return world.rules.held(str(id), actor) > 0)


## A share (0..1) of what the option in hand `advertises`, added to the needs.
func _give(share: float) -> void:
    if share <= 0.0 or _option.is_empty():
        return
    for need in _option["advertises"]:
        if needs.has(need):
            needs[need] = clampf(needs[need] + float(_option["advertises"][need]) * share, 0.0, 1.0)
    if _option.has("person") and is_instance_valid(_partner):      # the one it is with gets as much
        _partner._share(_option["advertises"], share, self)
        _know_better(_partner, share)


## Needs fill over the option's duration (its action's `wait: duration` / `loop: duration`), so
## one cut short or failed has still given what it had time to (geogen.101.2).
func _fill_over(delta: float) -> void:
    # Only for those who plan: the others' needs are tuned to getting it all when the option is done
    # (a diner's rest falls faster than a seat fills it: filled as it sat, it would never recover).
    if not definition.has("planner") or _stack.size() != 1 or _fill_total <= 0.0 or _fill >= 1.0 \
            or not _has_what_it_takes():
        return
    var share := minf(delta / _fill_total, 1.0 - _fill)
    _fill += share
    _give(share)


# --- running --------------------------------------------------------------

## Riding: the affordance (one with `follow`, on a mechanism body) the NPC is posed on, and
## the NPC's offset from its seat, kept every frame as the seat moves (mechanisms.gd).
var _ride: Dictionary = {}
var _ride_offset := Transform3D.IDENTITY
var _solid_at := 0                   # physics frame its collider comes back on (_solid_after_move)


## Moved in one go (off a seat, back from away): its collider comes back on only once it has
## stood a couple of physics frames where it now is. Jolt gives a kinematic body moved in a step
## the speed of that move, from where it was: a collider on at once is, for that step, a wall
## going hundreds of metres a second through whatever is at the seat it left (a swing chair
## was flung over its crown).
func _solid_after_move() -> void:
    _shape.set_deferred("disabled", true)
    _solid_at = Engine.get_physics_frames() + 2


func _physics_process(delta: float) -> void:
    var started := Time.get_ticks_usec()
    _frame_delta = delta
    if _solid_at > 0 and Engine.get_physics_frames() >= _solid_at:
        _solid_at = 0
        _shape.set_deferred("disabled", false)
    if not _ride.is_empty():
        global_transform = GeogenMechanisms.seat_now(_ride) * _ride_offset
    # Out of sight (or far off), step every other frame over both frames' time: half the
    # cost, and nobody sees the coarser motion. NPCs alternate, so each frame takes half.
    # Far off and out of sight, sleep: a step every SLEEP_EVERY frames, the clips paused.
    var rate := _rate()
    if animator != null and animator.player != null and animator.player.active != (rate < SLEEP_EVERY):
        animator.player.active = rate < SLEEP_EVERY
    if rate > 1 and (Engine.get_physics_frames() + _phase) % rate != 0:
        _skipped += delta
        stats["throttled"] = int(stats.get("throttled", 0)) + 1
        if rate == SLEEP_EVERY:
            stats["slept"] = int(stats.get("slept", 0)) + 1
    else:
        _step(delta + _skipped)
        _skipped = 0.0
    _cost_usec += Time.get_ticks_usec() - started
    _cost_frames += 1


func _step(delta: float) -> void:
    clock += delta
    if absent:
        return
    if away:
        if not _away_block():
            _reappear()
        return
    if animator != null:
        animator.advance(delta)
        animator.locomote(Vector2(velocity.x, velocity.z).length(), _pose == "stand", _playing)
    _attend(delta)
    if GeogenMemory.has(self):
        GeogenMemory.step(self)
    if clock < _cut_until and _cut_short():
        _cut_until = -INF
    for need in needs:
        needs[need] = maxf(0.0, needs[need] - float(definition["needs"][need].get("decay", 0.0)) * delta)
    if needs.has("together") and group_leader() != "":
        _keep_together(delta)
    if _skeleton != null and clock >= _held_at:
        _held_at = clock + HELD_EVERY
        _hold(_held_item())
    if home_distance(global_position) > float(definition.get("home_margin", 1.0)) + 0.5:
        stats["outside"] += delta
    if label.visible:
        label.text = ("%s\n" % person_name() if person_name() != "" else "") + "%s\n%s" % [_status(), _needs_text()]
    if _stack.is_empty():
        var map := get_world_3d().navigation_map
        if NavigationServer3D.map_get_iteration_id(map) > 0 \
                and NavigationServer3D.map_get_closest_point(map, global_position).distance_to(global_position) < 1.0:
            if _away_block() and definition.get("actions", {}).has("leave"):
                _option = {"id": "self/leave", "action": "leave", "advertises": {}}
                _trace({"event": "decide", "chosen": "self/leave", "reason": "routine"})
                _push("leave", {"anchor": global_position, "yaw_deg": rad_to_deg(_yaw()),
                    "approach": global_position, "duration": [0.0, 0.0]})
            else:
                _decide()
        return
    var frame: Dictionary = _stack.back()
    if frame["index"] >= frame["steps"].size():
        _stack.pop_back()
        if _stack.is_empty():
            _finish_option(true)
        else:
            var under: Dictionary = _stack.back()
            under["started"] = false           # re-run the step a door or reaction interrupted
            if under.has("from"):
                # Moved off (stepped aside): walk back first, so work at a counter resumes there.
                if global_position.distance_to(under["from"]) > REACT_DRIFT:
                    var i: int = mini(under["index"], under["steps"].size() - 1)
                    while i > 0 and not under["steps"][i].has("go_to"):
                        i -= 1
                    if under["steps"][i].has("go_to"):
                        under["index"] = i
                under.erase("from")
        return
    var step: Dictionary = frame["steps"][frame["index"]]
    if step.has("if") and not definition.get("flags", {}).get(step["if"], false):
        frame["index"] += 1
        return
    if not frame["started"]:
        frame["started"] = true
        if step.has("look"):
            _intent = "" if step["look"] == "none" else str(step["look"])
            _intent_ctx = frame["ctx"]
        var err := _begin(step, frame["ctx"])
        _trace({"event": "step", "action": frame["action"], "step": step, "pos": _v(global_position)})
        if err != "":
            _fail("%s %s: %s" % [frame["action"], JSON.stringify(step), err])
            return
    var result := _tick(step, frame["ctx"], delta)
    if result == "done":
        frame["index"] += 1
        frame["started"] = false
    elif result != "running":
        _fail("%s %s: %s" % [frame["action"], JSON.stringify(step), result])


func _fail(reason: String) -> void:
    _stop_motions()
    _let_go()
    _stack.clear()
    _stair = null
    _playing = false
    _looping = false
    if _pose != "stand":
        # (Up from the seat where it is: the stand pose finds the floor beside it. It needs the
        # `approach` it stands at; without one the failure left the body seated, walking on.)
        _set_pose("stand", {"anchor": global_position, "approach": global_position, "yaw_deg": rad_to_deg(_yaw())}, "approach")
    velocity = Vector3.ZERO
    _unstick(_wedged)
    _wedged = false
    _finish_option(false, reason)


## Wedged (stuck twice running at the same spot, e.g. against a post at the top of some steps):
## step back onto the nearest point of the navmesh instead of failing every plan from there.
func _unstick(wedged: bool) -> void:
    if not wedged:
        _stuck_at = null
        return
    if _stuck_at != null and (_stuck_at as Vector3).distance_to(global_position) < 0.3:
        var map := get_world_3d().navigation_map
        var q := NavigationServer3D.map_get_closest_point(map, global_position)
        if q.distance_to(global_position) > 0.05:
            global_position = q + Vector3.UP * 0.02
        else:
            global_position += -global_basis.z * 0.3   # on the navmesh but snagged: back off
        stats["unstuck"] = int(stats.get("unstuck", 0)) + 1
        _trace({"event": "unstuck", "pos": _v(global_position)})
        _stuck_at = null
        return
    _stuck_at = global_position


## Start a step; returns "" or why it can't run.
func _begin(step: Dictionary, ctx: Dictionary) -> String:
    _timer = 0.0
    if step.has("go_to"):
        var target = _target(step["go_to"], ctx)
        if target == null:
            return "no %s point" % step["go_to"]
        _replans = 0
        _close = bool(step.get("close", false))
        return _plan(target)
    if step.has("face"):
        var dir: Vector3
        if step["face"] == "portal" and ctx.has("portal"):
            dir = -ctx["portal"]["normal"] * float(ctx["side"])
        elif step["face"] == "partner":
            var other = ctx.get("partner")
            if not is_instance_valid(other) or (other as GeogenNpc).away:
                return "nobody there"
            dir = NpcGeometry.flat((other as GeogenNpc).global_position - global_position)
            if dir.length() > PARTNER_FAR and _option.has("person"):
                return "they have gone"
            if dir.length() < 1e-3:
                return ""
        elif step["face"] == "player":
            var who := _player()
            if who == null:
                return ""
            dir = NpcGeometry.flat(who.global_position - global_position)
            if dir.length() < 1e-3:
                return ""
        else:
            dir = Basis(Vector3.UP, deg_to_rad(float(ctx["yaw_deg"]))) * Vector3.BACK
        _yaw_goal = atan2(dir.x, dir.z)
        return ""
    if step.has("set_out"):
        return _set_out_item(str(step["set_out"]), ctx)
    if step.has("show"):
        for id in step["show"]:
            _item_state[str(id)] = str(step["show"][id])
            if _held != null and is_instance_valid(_held) and str(_held.get_meta("held_item", "")) == str(id):
                for body_ in _held.get_children():
                    show_state(body_, _item_state[str(id)])
            if str(_set_out.get("item", "")) == str(id) and is_instance_valid(_set_out.get("node")):
                show_state(_set_out["node"], _item_state[str(id)])
        _trace({"event": "show", "show": step["show"]})
        return ""
    if step.has("find"):
        var found := _find_rest(step["find"], step.get("prefer", {}))
        if found.is_empty():
            ctx["found"] = ""             # nowhere free: it carries on where it is
            return ""
        world.npc_reserve(found["id"], self)
        _option["found"] = found["id"]
        ctx["found"] = found["type"]
        ctx["anchor"] = found["position"]
        ctx["approach"] = found["approach"]
        ctx["yaw_deg"] = found["npc_yaw_deg"]
        ctx["depth"] = found.get("depth")
        _trace({"event": "found", "found": found["id"], "pose": found["type"], "among": _find_seen})
        return ""
    if step.has("pose") and step["pose"] == "found":
        if str(ctx.get("found", "")) == "":
            _timer = POSE_SECONDS         # nothing found: nothing to sit or lie on
            return ""
        _set_pose(str(ctx["found"]), ctx, str(step.get("at", "anchor")))
        return ""
    if step.has("pose"):
        if not definition.get("body", {}).get("poses", {}).has(step["pose"]):
            return "body has no pose '%s'" % step["pose"]
        var ride: Dictionary = ctx.get("ride", {})
        if step["pose"] != "stand" and not ride.is_empty() and not ride.get("boardable", true):
            return "the ride has started"          # it moved off while the NPC walked up: try later
        _set_pose(step["pose"], ctx, str(step.get("at", "anchor")), str(step.get("hold", "")))
        return ""
    if step.has("wait") and str(step["wait"]) == "partner":
        return ""
    if step.has("wait"):
        var r = ctx["duration"] if str(step["wait"]) == "duration" else step["wait"]
        _timer = rng.randf_range(float(r[0]), float(r[1]))
        if str(step["wait"]) == "duration" and step.has("part"):
            # A share of the duration, drawn once for the action: eating in stages over one meal.
            if not ctx.has("drawn"):
                ctx["drawn"] = _timer
            _timer = float(ctx["drawn"]) * float(step["part"])
        if str(step["wait"]) == "duration" and _stack.size() == 1 and not ctx.get("held", false):
            _fill_total = float(ctx.get("drawn", _timer))   # (held until a routine block ends, it all comes at the end, as it did)
        return ""
    if step.has("vanish"):
        return ""
    if step.has("do"):
        return _begin_do(step, ctx)
    if step.has("play"):
        # A skeletal body plays the clip once (then back to standing), or over and over for the
        # step's `loop` time (work at a counter); others just pause, or stand there that long.
        var clip := str(step["play"])
        velocity = Vector3.ZERO
        _timer = 1.0
        if step.has("loop"):
            var r = ctx["duration"] if str(step["loop"]) == "duration" else step["loop"]
            _timer = rng.randf_range(float(r[0]), float(r[1]))
            if str(step["loop"]) == "duration" and _stack.size() == 1 and not ctx.get("held", false):
                _fill_total = _timer
        if animator != null and animator.has(clip) and _pose == "stand":
            if step.has("loop"):
                animator.play_loop(clip)
                _looping = true
                _glance_at = clock + _busy_gap()
            else:
                _timer = animator.play_once(clip)
            _playing = true
        return ""
    if step.has("pick_up"):
        var uid := str(ctx.get("subject", ""))
        if not world.props.can_take(uid):
            return "can't take it (gone, or in someone's hands)"
        var limit := _carry_limit(uid)
        if limit != "":
            return limit
        var at := _prop_centre(uid)
        if not _within_reach(at):
            return "out of reach (%.2f m away, %.2f m up)" % [NpcGeometry.flat(at - global_position).length(), at.y - global_position.y]
        _face_point(at)
        _taking = _take(uid)
        if _taking:
            return ""
        _timer = PICK_SECONDS
        return ""
    if step.has("put_down"):
        if world.props.taken_by(self) == "":
            return "holding nothing"
        var place: Dictionary = ctx.get("place", {})
        if not place.has("position"):
            return "no place to put it"
        if not _within_reach(place["position"]):
            return "the place is out of reach"
        _face_point(place["position"])
        _taking = _set_down(world.props.taken_by(self), place)
        _timer = PICK_SECONDS
        return ""
    if step.has("use_item"):
        var id := _carried_item(str(step["use_item"]))
        if id == "":
            return "carrying nothing to use for %s" % step["use_item"]
        var err: String = world.rules.use_item(id, WorldLoader.actor_id(self))
        if err != "":
            return err
        stats["items_used"] = int(stats.get("items_used", 0)) + 1
        _trace({"event": "use_item", "item": id, "pos": _v(global_position)})
        return ""
    if step.has("use"):
        var it: GeogenInteraction = ctx.get("interaction")
        if it == null:
            return ""   # nothing to operate: the step is a no-op
        var state := _state_name(str(step["use"]), ctx)
        if not it.states.has(state):
            return ""
        _pressing = false
        if it.target != state and _press(it):
            return ""
        if it.target != state:
            # A door with a handle on this side: the hand takes it, and the door moves when it turns.
            var handle: Dictionary = _handle(ctx)
            if not handle.is_empty() and _turn_handle(handle):
                return ""
            if not it.use(WorldLoader.actor_id(self)):
                return "can't use %s (locked?)" % it.name
        return ""
    return "unknown step"


func _tick(step: Dictionary, ctx: Dictionary, delta: float) -> String:
    if step.has("go_to"):
        return _walk(delta)
    if step.has("face"):
        var diff := wrapf(_yaw_goal - _yaw(), -PI, PI)
        var turn := deg_to_rad(float(definition.get("turn_speed", 360.0))) * delta
        rotation.y += clampf(diff, -turn, turn)
        return "done" if absf(diff) <= turn else "running"
    if step.has("do"):
        if str((step.get("params", {}) as Dictionary).get("duration", "")) == "duration":
            _fill_over(delta)
        return "running" if _run != null and not _run.done else "done"
    if step.has("pose"):
        if _moving_pose():
            return "running"              # the transition clip sets the pace
        _timer += delta
        return "done" if _timer >= POSE_SECONDS else "running"
    if step.has("wait") and str(step["wait"]) == "partner":
        _timer += delta
        if _timer > PARTNER_WAIT:
            _part()                       # they never came
        if not is_instance_valid(_partner):
            return "done"
        _face_point(_partner.global_position)       # turned to them as they come up
        var off := wrapf(_yaw_goal - _yaw(), -PI, PI)
        var most := deg_to_rad(float(definition.get("turn_speed", 360.0))) * delta
        rotation.y += clampf(off, -most, most)
        return "running"
    if step.has("wait"):
        _timer -= delta
        if str(step["wait"]) == "duration":
            _fill_over(delta)
        if ctx.get("held", false) and str(step["wait"]) == "duration":
            return "running" if _hold_block() else "done"     # until the block ends (morning)
        if _timer <= 0.0 and not _ride.is_empty() and not _ride.get("boardable", true):
            return "running"              # the ride is going: stay in the seat until it stops at the ground
        return "done" if _timer <= 0.0 else "running"
    if step.has("play"):
        _timer -= delta
        if str(step.get("loop", "")) == "duration":
            _fill_over(delta)
        if ctx.get("held", false) and str(step.get("loop", "")) == "duration" and _hold_block():
            return "running"
        if _timer > 0.0:
            return "running"
        _playing = false
        _looping = false
        return "done"
    if step.has("vanish"):
        if not _away_block():
            return "done"                 # the time to be away is over: stay
        for other in get_tree().get_nodes_in_group(CHARACTER_GROUP):
            if other is Node3D and not other is GeogenNpc \
                    and (other as Node3D).global_position.distance_to(global_position) < VANISH_DISTANCE:
                return "running"          # not in front of the player
        _let_go()
        away = true
        visible = false
        velocity = Vector3.ZERO
        _shape.set_deferred("disabled", true)
        remove_from_group(CHARACTER_GROUP)
        stats["aways"] = int(stats.get("aways", 0)) + 1
        _trace({"event": "away", "pos": _v(global_position)})
        return "done"
    if step.has("find") or step.has("set_out") or step.has("show"):
        return "done"
    if step.has("pick_up") or step.has("put_down"):
        var diff := wrapf(_yaw_goal - _yaw(), -PI, PI)
        rotation.y += clampf(diff, -TAU * delta, TAU * delta)
        if _taking:
            var heard := "taken" if step.has("pick_up") else "put"
            if not _emitted.has(heard) and _run != null and not _run.done:
                return "running"                  # the hand is on its way
            _taking = false
            if step.has("pick_up"):
                var taken := str(ctx.get("subject", ""))
                if prop_in_hand() != taken:
                    var had := _emitted.has("taken")
                    _let_go()
                    return "holding nothing (taken out of its hand)" if had else "can't take it (gone, or in someone's hands)"
                stats["picked_up"] = int(stats.get("picked_up", 0)) + 1
                stats["take_gap"] = _take_gap()
                GeogenWants.note_handled(self, taken)
                _trace({"event": "pick_up", "prop": taken, "pos": _v(global_position)})
                return "done"
            var was: String = world.props.taken_by(self)
            if was != "":
                _let_go()                         # the motion was cut off: it lets go where it is
            _carry_run = null
            _carry_on = {}
            stats["put_down"] = int(stats.get("put_down", 0)) + 1
            _trace({"event": "put_down", "prop": was, "at": _v(ctx["place"]["position"])})
            return "done"
        _timer -= delta
        if _timer > 0.0:
            return "running"
        if step.has("pick_up"):
            var uid := str(ctx.get("subject", ""))
            if not world.props.take(uid, self, _prop_point.bind(uid)):
                return "can't take it (gone, or in someone's hands)"
            stats["picked_up"] = int(stats.get("picked_up", 0)) + 1
            GeogenWants.note_handled(self, uid)
            _trace({"event": "pick_up", "prop": uid, "pos": _v(global_position)})
            return "done"
        var held_uid: String = world.props.taken_by(self)
        if held_uid == "":
            return "holding nothing"          # taken out of its hand as it turned
        var place: Dictionary = ctx["place"]
        var basis := Basis(Vector3.UP, float(place.get("yaw", _yaw())))
        var centre: Vector3 = world.props.box_of(held_uid)["center"]
        # Set down with its box's underside just above the spot: it settles the last few mm.
        var size: Vector3 = world.props.box_of(held_uid)["size"]
        var origin: Vector3 = place["position"] + Vector3.UP * (size.y / 2.0 + PUT_GAP) - basis * centre
        world.props.release(held_uid, Transform3D(basis, origin))
        _carry_on = {}
        stats["put_down"] = int(stats.get("put_down", 0)) + 1
        _trace({"event": "put_down", "prop": held_uid, "at": _v(place["position"])})
        return "done"
    if step.has("use_item"):
        return "done"
    if step.has("use"):
        var it: GeogenInteraction = ctx.get("interaction")
        var state := _state_name(str(step["use"]), ctx)
        if it == null or not it.states.has(state) or (it.state == state and it.target == state):
            return "done"
        if _pressing:
            if not _emitted.has("pressed") and _run != null and not _run.done:
                return "running"                  # the finger is on its way
            _pressing = false
            _note_press(it)
            if it.target != state and not it.use(WorldLoader.actor_id(self)):
                return "can't use %s (locked?)" % it.name
        _timer += delta
        if it.target != state and it.can_use():
            it.use(WorldLoader.actor_id(self))   # someone turned it back: press again
        return "running" if _timer < USE_TIMEOUT else "%s never reached '%s'" % [it.name, state]
    return "unknown step"


# --- motions (assets/motions/, motion_player.gd) ---------------------------------

## The body's motion player, layered over its clips; null for a body with no rig (a capsule, a cat).
func _motion_player() -> GeogenMotionPlayer:
    if _motions == null and body != null:
        _motions = GeogenMotionPlayer.on(body)
        if _motions != null:
            _motions.layered = true
            _motions.rig.emitted.connect(func(_motion: String, what: String): _emitted[what] = true)
    return _motions


## Start a `do:` step's motion; "" or why it can't run. A body with no rig, or an export without
## the motion, just goes on.
func _begin_do(step: Dictionary, ctx: Dictionary) -> String:
    velocity = Vector3.ZERO
    _run = null
    var player := _motion_player()
    if player == null:
        return ""
    var name := str(step["do"])
    var spec: Dictionary = GeogenMotionPlayer.motions(world.generated_dir).get(name, {})
    if spec.is_empty():
        return ""                          # an export without its motions.json: the body just goes on
    var values := {}
    var params: Dictionary = step.get("params", {})
    for key in params:
        values[key] = params[key]
        if str(params[key]) == "duration":
            var r: Array = ctx["duration"]
            values[key] = rng.randf_range(float(r[0]), float(r[1]))
            if _stack.size() == 1 and not ctx.get("held", false):
                _fill_total = float(values[key])
    var roles: Dictionary = step.get("roles", {})
    for role in roles:
        var bound = _motion_role(str(roles[role]), ctx)
        if bound == null:
            return "motion '%s': nothing to be its %s (%s)" % [name, role, roles[role]]
        values[role] = bound
    var run := player.start(spec, values)
    if run == null:
        return "motion '%s' can't run on this body" % name
    if step.get("keep", false):
        _kept.append(run)
    else:
        _run = run
    return ""


## What a `do:` step's role names, as the rig binds it: a node, a Thing or a Target; null if none.
func _motion_role(source: String, ctx: Dictionary) -> Variant:
    match source:
        "subject":
            var uid := str(ctx.get("subject", ""))
            return world.props.props[uid]["node"] if world.props != null and world.props.props.has(uid) else null
        "interaction":
            var it: GeogenInteraction = ctx.get("interaction")
            return _touch_of(it) if it != null else null
        "anchor":
            if not ctx.has("anchor"):
                return null
            var at := Transform3D(Basis(Vector3.UP, deg_to_rad(float(ctx.get("yaw_deg", 0.0)))), ctx["anchor"])
            return GeogenMotionRig.Thing.fixed("anchor", at)
        "place":
            var place: Dictionary = ctx.get("place", {})
            if not place.has("position"):
                return null
            var at := Transform3D(Basis(Vector3.UP, float(place.get("yaw", _yaw()))), place["position"])
            return GeogenMotionRig.Thing.fixed("place", at)
    return null


## Where a body touches interaction ``it`` (its `touch:` grip point), or null.
func _touch_of(it: GeogenInteraction) -> GeogenMotionRig.Target:
    var player := _motion_player()
    if player == null or it.touch == "" or not is_instance_valid(it.asset):
        return null
    var thing := player.thing_of(it.asset)
    return GeogenMotionRig.contact(thing, null, it.touch) if thing.grips.has(it.touch) else null


## Work ``it`` with a finger: the `press` motion to its touch point, the state changing when the
## finger arrives (its `pressed`). False when this body can't: no rig, no touch point, out of reach,
## or not on its feet; the step then uses it from where it stands.
func _press(it: GeogenInteraction) -> bool:
    var button := _touch_of(it)
    if button == null or _pose != "stand":
        return false
    var spec: Dictionary = GeogenMotionPlayer.motions(world.generated_dir).get("press", {})
    var rig := _motions.rig
    var arm: float = float(rig.params.get("height", 1.7)) * float(rig.params.get("arm_ratio", 0.44))
    if spec.is_empty() or button.world().origin.distance_to(head_point() + Vector3.DOWN * SHOULDER_DROP) > arm * PRESS_REACH:
        return false
    _emitted.erase("pressed")
    _run = _motions.start(spec, {"button": button})
    _pressing = _run != null
    if _pressing:
        stats["presses"] = int(stats.get("presses", 0)) + 1
    return _pressing


## How far the index fingertip is from the touch point as the press lands (m, for the report).
func _note_press(it: GeogenInteraction) -> void:
    var button := _touch_of(it)
    var rig := _motions.rig if _motions != null else null
    if rig != null and _handle_at != null and _emitted.has("pressed"):
        var fk := rig.world()                  # the wrist nearer the handle: the hand that took it
        var at := _handle_at.world().origin
        stats["handle_gap"] = snappedf(minf((rig.body_world * (fk[rig.index["LeftHand"]] as Transform3D).origin).distance_to(at),
            (rig.body_world * (fk[rig.index["RightHand"]] as Transform3D).origin).distance_to(at)), 0.001)
    _handle_at = null
    if button == null or rig == null or not _emitted.has("pressed"):
        return
    var tip := -1
    for i in rig.names.size():
        if String(rig.names[i]).begins_with("RightIndex"):
            tip = i                        # the last of the finger's bones
    if tip >= 0:
        var at: Vector3 = rig.body_world * (rig.world()[tip] as Transform3D).origin
        stats["press_gap"] = snappedf(at.distance_to(button.world().origin), 0.001)


## The thing a prop is to the hands. One hand takes the top of its box, palm down; two take its
## sides as this body faces it now, whichever way the prop happens to lie.
func _prop_thing(uid: String) -> GeogenMotionRig.Thing:
    var thing := _motions.thing_of(world.props.props[uid]["node"])
    var box: Dictionary = world.props.box_of(uid)
    var centre: Vector3 = box["center"]
    var size: Vector3 = box["size"]
    if not _two_handed(uid):
        if thing.grips.is_empty():
            var top := centre + Vector3.UP * size.y / 2.0
            thing.set_grips({"top": {"grip": "rim", "at": [top.x, top.y, top.z, 0, 0, 0, 1]}})
        return thing
    var left: Vector3 = thing.world().basis.inverse() * (Basis(Vector3.UP, _yaw()) * Vector3.RIGHT)
    var half := _carry_half(uid) - CARRY_PALM
    var point := func(side: float, out: float) -> Array:
        var p: Vector3 = centre + left * side * (half + out)
        return [p.x, p.y, p.z, 0, 0, 0, 1]
    thing.set_grips({"left": {"grip": "flat", "at": point.call(1.0, CARRY_PALM)},
        "right": {"grip": "flat", "at": point.call(-1.0, CARRY_PALM)},
        "left_off": {"grip": "open", "at": point.call(1.0, CARRY_PALM + HANDS_OFF)},
        "right_off": {"grip": "open", "at": point.call(-1.0, CARRY_PALM + HANDS_OFF)}})
    return thing


## How far from the middle of bulky prop ``uid`` each hand is held (m).
func _carry_half(uid: String) -> float:
    var size: Vector3 = world.props.box_of(uid)["size"]
    return maxf(size.x, size.z) / 2.0 + CARRY_PALM


## How far the hand that took it is from where it took hold (m): the wrist from a side grip, the
## index fingertip from the top of a thing taken in one hand; -1 for nothing held.
func _take_gap() -> float:
    if _motions == null or not _motions.rig.attached_.has("item"):
        return -1.0
    var gaps := _hand_gaps()
    if not gaps.is_empty():
        return gaps.values().max()
    var rig := _motions.rig
    var thing: GeogenMotionRig.Thing = rig.attached_["item"]["thing"]
    if not thing.grips.has("top"):
        return -1.0
    var tip := -1
    for i in rig.names.size():
        if String(rig.names[i]).begins_with("RightIndex"):
            tip = i
    var at: Vector3 = rig.body_world * (rig.world()[tip] as Transform3D).origin
    return snappedf(at.distance_to(thing.world() * (thing.grips["top"]["local"] as Transform3D).origin), 0.001)


## How far each hand is from where it holds what both hands carry ({hand: m}; {} for none).
func _hand_gaps() -> Dictionary:
    if _motions == null or not _motions.rig.attached_.has("item"):
        return {}
    var thing: GeogenMotionRig.Thing = _motions.rig.attached_["item"]["thing"]
    if not thing.grips.has("left"):
        return {}
    var bones := _motions.rig.world()
    var gaps := {}
    for hand in [["LeftHand", "left"], ["RightHand", "right"]]:
        var at: Vector3 = _motions.rig.body_world * (bones[_motions.rig.index[hand[0]]] as Transform3D).origin
        gaps[hand[0]] = snappedf(at.distance_to(thing.world() * (thing.grips[hand[1]]["local"] as Transform3D).origin), 0.001)
    return gaps


## Take prop ``uid`` and keep hold of it (motions take and carry, or take_both and carry_both for
## a bulky one). False when this body has no such motions or is not on its feet.
func _take(uid: String) -> bool:
    var player := _motion_player()
    var motions := GeogenMotionPlayer.motions(world.generated_dir)
    if player == null or _pose != "stand":
        return false
    var both := _two_handed(uid)
    # Low down, the knees bend to get the hands to it.
    var legs: float = float(player.rig.params.get("height", 1.7)) * float(player.rig.params.get("leg_ratio", 0.47))
    var crouch := clampf(LOW_HAND - (_prop_centre(uid).y - global_position.y), 0.0, legs * MAX_CROUCH)
    var take := "take_both" if both else ("take_low" if crouch > 0.02 else "take")
    var carry := "carry_both" if both else "carry"
    if not (motions.has(take) and motions.has(carry)):
        return false
    var thing := _prop_thing(uid)
    var taking := {"item": thing}
    var holding := {"item": thing}
    if both:
        taking["half"] = _carry_half(uid)
        holding["half"] = taking["half"]
    if both or crouch > 0.02:
        taking["crouch"] = crouch
    velocity = Vector3.ZERO
    player.holder = self
    _emitted.erase("taken")
    _run = player.start(motions[take], taking)
    if _run == null:
        return false
    _carry_run = player.start(motions[carry], holding)
    return true


## Put what the hand holds at ``place`` (motion set_down): the box's underside just above the
## spot. False when no motion holds it (it was taken some other way).
func _set_down(uid: String, place: Dictionary) -> bool:
    var motions := GeogenMotionPlayer.motions(world.generated_dir)
    var motion := "set_down_both" if _two_handed(uid) else "set_down"
    if _motions == null or not _motions.rig.attached_.has("item") or not motions.has(motion):
        return false
    var box: Dictionary = world.props.box_of(uid)
    var basis := Basis(Vector3.UP, float(place.get("yaw", _yaw())))
    var at: Vector3 = place["position"] + Vector3.UP * ((box["size"] as Vector3).y / 2.0 + PUT_GAP) - basis * (box["center"] as Vector3)
    _emitted.erase("put")
    _run = _motions.start(motions[motion], {"item": _motions.rig.attached_["item"]["thing"],
        "place": GeogenMotionRig.Thing.fixed("place", Transform3D(basis, at))})
    return _run != null


## Open a door by ``handle`` ({node, local}) with the hand on its side (motions turn_handle and
## turn_handle_left): the door moves when the hand turns it. False when this body can't.
func _turn_handle(handle: Dictionary) -> bool:
    var player := _motion_player()
    if player == null or _pose != "stand":
        return false
    var node: Node3D = handle["node"]
    var point: Vector3 = node.global_transform * (handle["local"] as Vector3)
    var left: bool = (player.body.global_transform.affine_inverse() * point).x > 0.0      # bodies' +X is their left
    var spec: Dictionary = GeogenMotionPlayer.motions(world.generated_dir).get("turn_handle_left" if left else "turn_handle", {})
    if spec.is_empty():
        return false
    var near := point + NpcGeometry.flat(global_position - point).normalized() * HANDLE_GRIP
    var local: Vector3 = node.global_transform.affine_inverse() * near
    var thing := player.thing_of(node)
    thing.set_grips({"knob": {"grip": "power", "at": [local.x, local.y, local.z, 0, 0, 0, 1]}})
    _emitted.erase("pressed")
    _handle_at = GeogenMotionRig.contact(thing, null, "knob")
    _run = player.start(spec, {"handle": _handle_at})
    _pressing = _run != null
    if _pressing:
        stats["handles"] = int(stats.get("handles", 0)) + 1
    return _pressing


## Wave from a seat (motion wave, on the arm alone): false when this body has no such motion.
func _wave_seated() -> bool:
    var player := _motion_player()
    var spec: Dictionary = GeogenMotionPlayer.motions(world.generated_dir).get("wave", {})
    if player == null or spec.is_empty():
        return false
    _wave_run = player.start(spec)
    return _wave_run != null


## The motions it is in the middle of stop as their beats allow (`how`: yield, or drop).
func _stop_motions(how := "yield") -> void:
    _pressing = false
    if _motions != null:
        for run in _kept + ([_run] if _run != null else []):
            if not run.done:
                _motions.stop(run.name, how)
    _run = null
    _kept = []


## What a save keeps of this NPC (the save's `npcs`, by stable ID; docs/rules.md "Saving"): its
## needs, whether it is away, how long ago it last used each thing, and where it is: a seated or
## lying NPC by the affordance it is on (`seat`), so loading puts it back on it, else its
## position and heading. What it was doing otherwise isn't kept: it decides again.
func save_state() -> Dictionary:
    var out := {"needs": _rounded_needs(), "away": away,
        "position": [snappedf(global_position.x, 0.001), snappedf(global_position.y, 0.001), snappedf(global_position.z, 0.001)],
        "yaw": snappedf(rad_to_deg(_yaw()), 0.1)}
    if not relations.is_empty():
        out["relations"] = _rounded_relations()
    if not _orders.is_empty():
        out["orders"] = _orders.duplicate()
    if not free_will:
        out["free_will"] = false
    var recent := {}
    for id in _recent:
        recent[id] = snappedf(clock - float(_recent[id]), 0.1)
    if not recent.is_empty():
        out["recent"] = recent
    if _pose != "stand" and _option.has("affordance"):
        out["seat"] = _option["id"]
    return out


## Take up a save_state(): drop what it is doing, stand where it was (or go back to the seat
## it was on: it walks the last step and sits again), with the needs and memory it had.
func restore_state(data: Dictionary) -> void:
    _drop_everything()
    take_needs(data.get("needs", {}))
    relations = (data.get("relations", {}) as Dictionary).duplicate()
    _orders = (data.get("orders", []) as Array).duplicate()
    free_will = bool(data.get("free_will", true))
    _recent = {}
    var recent: Dictionary = data.get("recent", {})
    for id in recent:
        _recent[id] = clock - float(recent[id])
    var p: Array = data.get("position", [global_position.x, global_position.y, global_position.z])
    global_position = Vector3(p[0], p[1], p[2])
    rotation = Vector3(0, deg_to_rad(float(data.get("yaw", rad_to_deg(_yaw())))) - _parent_yaw(), 0)
    var seat := str(data.get("seat", ""))
    if seat != "":
        for a in world.affordances + world.affordance_uses:
            if a["id"] == seat:
                global_position = a["approach"]
        force_option(seat)
    away = bool(data.get("away", false))
    _show()
    _trace({"event": "restored", "pos": _v(global_position), "seat": seat, "away": away})


## In the world or out of it, by its placement's `present` state (WorldLoader keeps it in step).
func set_present(on: bool) -> void:
    if absent != on:
        return
    absent = not on
    if absent:
        if _pose != "stand" and _option.has("at"):
            global_position = _option["at"]      # off the seat: it comes back standing beside it
        _drop_everything()
    _show()
    _trace({"event": "present" if on else "absent", "pos": _v(global_position)})


## Seen, solid and among the characters unless it is away or absent.
func _show() -> void:
    var out := away or absent
    visible = not out
    if out:
        _shape.set_deferred("disabled", true)
    else:
        _solid_after_move()          # (it may have been put somewhere: a load, back beside its seat)
    if out:
        remove_from_group(CHARACTER_GROUP)
    elif not is_in_group(CHARACTER_GROUP):
        add_to_group(CHARACTER_GROUP)


## Stop whatever it is doing and stand: off its seat, its option given up.
func _drop_everything() -> void:
    _stop_motions("drop")
    _let_go()
    _clear_set_out()
    _intent = ""
    _stack.clear()
    _stair = null
    _playing = false
    _looping = false
    _path = PackedVector3Array()
    velocity = Vector3.ZERO
    _drop_plan("dropped everything")
    _part()
    _orders.clear()
    _ordered = false
    _fill = 0.0
    _fill_total = 0.0
    if not _option.is_empty():
        world.npc_release(str(_option.get("id", "")), self)
        if _option.has("found"):
            world.npc_release(str(_option["found"]), self)
        if _option.has("want"):
            GeogenWants.finish(self, _option, false)
        _option = {}
    if _pose != "stand":
        _pose = "stand"
        _shape.set_deferred("disabled", false)
        if animator != null:
            animator.hold = ""
            animator.transition = {}
            if body != null:
                body.transform = Transform3D.IDENTITY
            animator.play_pose("stand", 0.0)


## Take up saved needs (a save's, or a named person's from the scene it was last seen in).
func take_needs(saved: Dictionary) -> void:
    for need in saved:
        if needs.has(need):
            needs[need] = clampf(float(saved[need]), 0.0, 1.0)


## Whether the feet are the ground's to place: standing, not in a transition clip.
func _feet_free() -> bool:
    return _pose == "stand" and visible and (animator == null or animator.transition.is_empty())


## The handle of a pass action's door on the side the NPC is standing ({node, local}), or {}.
func _handle(ctx: Dictionary) -> Dictionary:
    if _motion_player() == null or not ctx.has("portal"):
        return {}
    var p: Dictionary = ctx["portal"]
    var side := 1 if (global_position - (p["center"] as Vector3)).dot(p["normal"]) >= 0.0 else -1
    var handle: Variant = (p.get("handles", {}) as Dictionary).get(side)
    if handle == null or not is_instance_valid(handle["node"]):
        return {}
    var point: Vector3 = (handle["node"] as Node3D).global_transform * (handle["local"] as Vector3)
    return handle if point.distance_to(head_point() + Vector3.DOWN * SHOULDER_DROP) < HANDLE_RANGE else {}


## A portal's generic open/closed names map to its interaction's states; `goal` is the state a
## quest goal wants (ctx.state).
func _state_name(state: String, ctx: Dictionary) -> String:
    if state == "goal":
        return str(ctx.get("state", ""))
    if ctx.has("portal"):
        return str(ctx["portal"].get(state, state))
    return state


func _target(kind: String, ctx: Dictionary):
    match kind:
        "aside":
            return ctx.get("aside")
        "partner":                  # a pace short of them, on the side it comes from
            var other = ctx.get("partner")
            if not is_instance_valid(other):
                return null
            var from := NpcGeometry.flat(global_position - (other as GeogenNpc).global_position)
            if from.length() <= PARTNER_GAP:
                return global_position
            var spot: Vector3 = (other as GeogenNpc).global_position + from.normalized() * PARTNER_GAP
            return NavigationServer3D.map_get_closest_point(get_world_3d().navigation_map, Vector3(spot.x, global_position.y, spot.z))
        "subject":
            return _stand_by_prop(str(ctx.get("subject", "")))
        "place":
            var place: Dictionary = ctx.get("place", {})
            if place.get("approach") != null:
                return place["approach"]
            # (A prop's own place, where it lay: where to stand to reach it is worked out on the way
            # back, from wherever the NPC is then.)
            return stand_for(place["position"]) if place.has("position") else null
        "approach":
            return ctx["approach"]
        "anchor":
            return ctx["anchor"]
        "near", "far":
            if not ctx.has("portal"):
                return null
            var p: Dictionary = ctx["portal"]
            var side := float(ctx["side"]) * (1.0 if kind == "near" else -1.0)
            # On the side the leaf swings away from, close enough to take the handle and hold it
            # as it opens; else clear of the leaf.
            var clear: float = p["clearance"]
            if kind == "near" and _hands != null and int(p.get("swing", 0)) == -int(side) \
                    and (p.get("handles", {}) as Dictionary).has(int(side)):
                clear = minf(clear, HANDLE_STAND)
            return p["center"] + p["normal"] * side * (p["depth"] / 2.0 + clear)
        "random":
            return _random_home_point()
        "nearby":
            return _nearby_point()
        "group":
            return _rejoin_point()
        "visit":
            var spot: Dictionary = visits.get(str(ctx.get("visit", "")), {})
            return null if spot.is_empty() else _random_point_in(spot["polygon"], spot["y"])
        "exit":
            # The nearest way out (a building entrance); none: stay put and vanish from here.
            var best = null
            for p in world.exit_points():
                if best == null or global_position.distance_to(p) < global_position.distance_to(best):
                    best = p
            return best if best != null else global_position
    return null


# --- carrying physics props (geogen.97.1) --------------------------------------

## Drop the prop in hand, if any, where it is.
func _let_go() -> void:
    _carry_on = {}
    _carry_run = null
    _taking = false
    if _motions != null and not _motions.rig.attached_.is_empty():
        _motions.break_()                  # the hand opens: it falls from where the hand had it
    if world != null and world.props != null:
        world.props.release(world.props.taken_by(self))


## The prop this NPC has taken ("" for none), for the report.
func prop_in_hand() -> String:
    return world.props.taken_by(self) if world != null and world.props != null else ""


## Whether prop ``uid`` is one for both hands: over the definition's `carry.two_hands` on its
## longest side.
func _two_handed(uid: String) -> bool:
    if uid == "" or not world.props.props.has(uid):
        return false
    var size: Vector3 = world.props.box_of(uid)["size"]
    return maxf(size.x, maxf(size.y, size.z)) > float(definition.get("carry", {}).get("two_hands", 0.2))


## Where a taken prop is held: a bulky one (`_two_handed`) in front of the body below the
## chest, a small one with its box's centre at the right hand (a skeletal body; else in front
## of the chest too); upright, turned with the NPC.
func _prop_point(uid: String) -> Transform3D:
    var basis := Basis(Vector3.UP, _yaw())
    var at := global_position + Vector3.UP * (float(definition.get("height", 1.7)) * 0.6) + basis * Vector3(0, 0, 0.3)
    if _two_handed(uid):
        var depth: float = (world.props.box_of(uid)["size"] as Vector3).z
        at = global_position + Vector3.UP * (float(definition.get("height", 1.7)) * CARRY_HEIGHT) \
            + basis * Vector3(0, 0, CARRY_BODY + depth / 2.0)
        # Held against the body, not a height over the feet: recorded in the chest's frame when it
        # was taken (standing), it goes with the chest onto a lap or over a chest lying down.
        var chest = _chest()
        if chest != null:
            if _carry_on.get("uid", "") != uid:
                _carry_on = {"uid": uid, "local": (chest as Transform3D).affine_inverse() * Transform3D(basis, at)}
            var held: Transform3D = (chest as Transform3D) * (_carry_on["local"] as Transform3D)
            return Transform3D(held.basis, held.origin - held.basis * (world.props.box_of(uid)["center"] as Vector3))
    elif _skeleton != null:
        var bone := _skeleton.find_bone("RightHand")
        if bone >= 0:
            at = (_skeleton.global_transform * _skeleton.get_bone_global_pose(bone)).origin + basis * Vector3(0, -0.04, 0.08)
    return Transform3D(basis, at - basis * (world.props.box_of(uid)["center"] as Vector3))


## Put carried item ``id``'s body on what is in front of the NPC (a table under its hands): a
## copy of the item's body standing on the first surface below a point ahead of the chest. Shown
## until the option ends or the item is used up, and not in the hand meanwhile; "" or why not.
func _set_out_item(id: String, ctx: Dictionary) -> String:
    if world.rules == null or world.rules.held(id, WorldLoader.actor_id(self)) <= 0:
        return "nothing left of what it takes (no %s to set out)" % id
    _clear_set_out()
    var template: Node3D = world.item_bodies.get(id)
    if template == null or not is_instance_valid(template):
        return ""                    # an item with no body: nothing to show
    # The top in front of where it sits: going out from the seat a step at a time, the first
    # thing above the seat looking down (a table). None within reach: it stays in the hand.
    var forward := Basis(Vector3.UP, _yaw()) * Vector3.BACK
    var seat: Vector3 = ctx.get("anchor", global_position)
    var hit := {}
    var ahead := SET_OUT_NEAR
    while ahead <= SET_OUT_FAR and hit.is_empty():
        var from := seat + forward * ahead + Vector3.UP * SET_OUT_ABOVE
        var query := PhysicsRayQueryParameters3D.create(from, Vector3(from.x, seat.y + 0.1, from.z))
        query.exclude = [get_rid()]
        hit = get_world_3d().direct_space_state.intersect_ray(query)
        ahead += 0.05
    if hit.is_empty():
        return ""
    var copy := template.duplicate() as Node3D
    for node in copy.find_children("*", "CollisionObject3D", true, false):
        node.get_parent().remove_child(node)
        node.free()
    copy.name = "set_out_%s" % id
    world.add_child(copy)
    copy.global_transform = Transform3D(Basis(Vector3.UP, _yaw()), hit["position"])
    copy.visible = true
    _set_out = {"item": id, "node": copy}
    if _item_state.has(id):
        show_state(copy, str(_item_state[id]))
    _trace({"event": "set_out", "item": id, "at": _v(hit["position"])})
    return ""


## Take what is set out away (the option is over); an item still carried goes back to the hand.
## The shown states of items no longer carried are forgotten (the next meal starts full).
func _clear_set_out() -> void:
    if _set_out.has("node") and is_instance_valid(_set_out["node"]):
        (_set_out["node"] as Node).queue_free()
    _set_out = {}
    if world == null or world.rules == null:
        return
    var actor := WorldLoader.actor_id(self)
    for id in _item_state.keys():
        if world.rules.held(str(id), actor) <= 0:
            _item_state.erase(id)


## Give ``node`` (an item's body, or any exported asset) a state of its own interactions: its
## `visible` motions show or hide their parts as the state says (extras.geogen.interactions).
static func show_state(node: Node, state: String) -> void:
    var found: Variant = WorldLoader.geogen_extras(node).get("interactions")
    if not found is Dictionary:
        return
    for name in found:
        var it: Dictionary = found[name]
        if not (it.get("states", {}) as Dictionary).has(state):
            continue
        for motion in it.get("motions", []):
            if str(motion.get("type", "")) != "visible":
                continue
            var on := float((motion.get("values", {}) as Dictionary).get(state, 1.0)) >= 0.5
            for part in motion.get("nodes", []):
                var target := node.find_child(str(part), true, false) as Node3D
                if target != null:
                    target.visible = on


## The chest bone's world transform as animated (null for a body without one).
func _chest():
    if _skeleton == null:
        return null
    var bone := _skeleton.find_bone("Chest")
    if bone < 0:
        return null
    var t := _skeleton.global_transform * _skeleton.get_bone_global_pose(bone)
    return Transform3D(t.basis.orthonormalized(), t.origin)


## A free affordance of these `types` (sit, lie) this NPC may use for a `find` step: one whose
## action poses the body at it, in its home, on its floor, not another's, not reserved past its
## slots, with nothing lying on it; {} when there is none. The nearest, with what it owns (its bed)
## counting as half as far, a thing with a `prefer` tag that share as far, and a seeded +-40% on
## each distance.
func _find_rest(types: Array, prefer: Dictionary = {}) -> Dictionary:
    var actions: Dictionary = definition.get("actions", {})
    var margin := float(definition.get("home_margin", 1.0))
    var best := {}
    var best_d := INF
    var seen := []
    for a in world.affordances:
        if not str(a.get("type", "")) in types or a.has("prop") or a.has("follow"):
            continue
        if not is_instance_valid(a.get("node")) or not actions.has(a.get("action", "")):
            continue
        if not _poses_at_anchor(actions[a["action"]]):
            continue
        var uid := WorldLoader.uid_of(a["node"])
        if _owned_by_another(uid) or world.npc_reserved(a["id"], self) >= int(a.get("slots", 1)):
            continue
        if home_distance(a["approach"]) > margin or absf((a["approach"] as Vector3).y - home_y) > FLOOR_TOLERANCE:
            continue
        if _something_on(a["position"], uid):
            continue
        var d := global_position.distance_to(a["approach"]) * rng.randf_range(0.6, 1.4)
        if _under(uid, definition.get("owns", [])):
            d *= FIND_OWN
        var tags: Array = WorldLoader.geogen_extras(a["node"]).get("tags", [])
        for tag in prefer:
            if tag in tags:
                d *= float(prefer[tag])
        seen.append([a["id"], snappedf(d, 0.01)])
        if d < best_d:
            best_d = d
            best = a
    _find_seen = seen
    return best


func _prop_centre(uid: String) -> Vector3:
    var body: Node3D = world.props.props[uid]["body"]
    return body.global_transform * (world.props.box_of(uid)["center"] as Vector3)


## A carryable prop's affordance as it is now (`a` is where the prop started): at the prop, with
## the approach where this body stands to reach it; {} while the prop can't be taken (in someone's
## hands, gone) or is out of this body's reach.
func _prop_affordance(a: Dictionary) -> Dictionary:
    var uid: String = a["prop"]
    if not world.props.can_take(uid):
        return {}
    var at := _prop_centre(uid)
    var stand = stand_for(at)
    if stand == null:
        return {}
    var now := a.duplicate()
    now["position"] = at
    now["approach"] = stand
    var toward := NpcGeometry.flat(at - (stand as Vector3))
    if toward.length() > 1e-3:
        now["npc_yaw_deg"] = rad_to_deg(atan2(toward.x, toward.z))
    return now


## Where prop ``uid`` lies, as a `place` for `put_down`: the middle of its box's underside and its
## heading, so putting it down there puts it back as it was (a book upright in its gap on the shelf).
func _where_it_lies(uid: String) -> Dictionary:
    var body: Node3D = world.props.props[uid]["body"]
    var box := world.props.box_of(uid)
    var xform := body.global_transform
    var under: Vector3 = xform * ((box["center"] as Vector3) - Vector3.UP * (box["size"] as Vector3).y / 2.0)
    var forward := xform.basis * Vector3.BACK
    return {"position": under, "yaw": atan2(forward.x, forward.z)}


## Where to stand to pick prop ``uid`` up: on the navmesh, `reach.stand` m short of it on this
## NPC's side (not on top of something lying on the floor).
func _stand_by_prop(uid: String):
    if not world.props.props.has(uid):
        return null
    var centre := _prop_centre(uid)
    var stand = stand_for(centre)
    if stand == null or centre.y - global_position.y >= LOW_HAND:
        return stand
    # Low down: right up to it (a `close` step goes the last bit), for the hands to get down to it.
    var from := NpcGeometry.flat(global_position - centre)
    if from.length() < 0.01:
        return stand
    var size: Vector3 = world.props.box_of(uid)["size"]
    var gap := float(definition.get("radius", 0.25)) + maxf(size.x, size.z) / 2.0 + LOW_STAND
    return Vector3(centre.x, (stand as Vector3).y, centre.z) + from.normalized() * gap


## Where to stand to get the hands to ``point``: the nearest place on the navmesh of the floor
## under it (beside the table it is on); for a point where one can walk, `reach.stand` m short
## of it on this NPC's side. null when nowhere in reach.
func stand_for(point: Vector3):
    var map := get_world_3d().navigation_map
    var reach: Dictionary = definition.get("reach", {})
    var off := float(reach.get("stand", 0.5))
    var heights: Array = reach.get("heights", [0.0, 1.5])
    # Ask from hand height below the point, so the floor it is over is nearer than the one above.
    var low := point - Vector3.UP * (float(heights[0]) + float(heights[1])) / 2.0
    var near := NavigationServer3D.map_get_closest_point(map, low)
    if NpcGeometry.flat(near - point).length() < off:
        var back := NpcGeometry.flat(global_position - point)
        near = NavigationServer3D.map_get_closest_point(map,
            low + (back.normalized() * off if back.length() > 1e-3 else Vector3(off, 0, 0)))
    # (A walk ends a little off the point it was for: what is only just in reach from there isn't.)
    return near if _within_reach(point, near, STAND_SLACK) else null


## Whether a point is one this body's hands get to from ``feet`` (where it stands): no further
## than `reach.distance` on the ground, and between the `reach.heights` above its feet.
func _within_reach(point: Vector3, feet = null, short := 0.0) -> bool:
    var reach: Dictionary = definition.get("reach", {})
    var origin: Vector3 = feet if feet is Vector3 else global_position
    var heights: Array = reach.get("heights", [0.0, 1.5])
    var up := point.y - origin.y
    return NpcGeometry.flat(point - origin).length() <= float(reach.get("distance", 1.0)) - short \
        and up >= float(heights[0]) - REACH_SLACK and up <= float(heights[1])


## "" when this body can carry prop ``uid`` (definition `carry: {mass, size}`), else why not.
func _carry_limit(uid: String) -> String:
    var carry: Dictionary = definition.get("carry", {})
    var data: Dictionary = world.props.props[uid]["data"]
    if float(data.get("mass", 10.0)) > float(carry.get("mass", 10.0)):
        return "too heavy to carry"
    var size: Vector3 = world.props.box_of(uid)["size"]
    if maxf(size.x, maxf(size.y, size.z)) > float(carry.get("size", 0.6)):
        return "too big to carry"
    return ""


func _face_point(point: Vector3) -> void:
    var dir := NpcGeometry.flat(point - global_position)
    if dir.length() > 1e-3:
        _yaw_goal = atan2(dir.x, dir.z)


## Run ``action`` with ``extra`` context as its next decision, once what it is doing is done (the
## control API's npcs.do: a test hands it a `subject` and a `place`). "" or why not.
func do_action(action: String, extra: Dictionary) -> String:
    if not (definition.get("actions", {}) as Dictionary).has(action):
        return "no action '%s'" % action
    _queued = {"action": action, "extra": extra}
    return ""


## An item this NPC carries that a `use_item` step names: an item id, or "[tag=<tag>]" for the
## first carried item with that tag; "" when it carries no such thing.
func _carried_item(spec: String) -> String:
    var actor := WorldLoader.actor_id(self)
    if spec.begins_with("[tag=") and spec.ends_with("]"):
        var tag := spec.substr(5, spec.length() - 6)
        for entry in world.rules.carried(actor):
            if (entry.get("tags", []) as Array).has(tag):
                return str(entry["id"])
        return ""
    return spec if world.rules.held(spec, actor) > 0 else ""


## A navmesh point a few steps from where the NPC stands (`go_to: nearby`): within NEARBY m, at
## least NEARBY_MIN away, and reached without a detour (the walk there no longer than NEARBY_DETOUR
## times the straight line: not round the back of a stall through a gap) when one can be found,
## else wherever it stands.
func _nearby_point() -> Vector3:
    var c := Vector2(global_position.x, global_position.z)
    var poly := PackedVector2Array([c + Vector2(-NEARBY, -NEARBY), c + Vector2(NEARBY, -NEARBY),
                                    c + Vector2(NEARBY, NEARBY), c + Vector2(-NEARBY, NEARBY)])
    var map := get_world_3d().navigation_map
    var fallback = null
    for _i in 10:
        var p = _random_point_in(poly, global_position.y)
        if p == null:
            continue
        var straight := (p as Vector3).distance_to(global_position)
        if straight < NEARBY_MIN:
            fallback = p
            continue
        if _someone_at(p):
            continue
        var path := GeogenWalker.find_path(map, global_position, p)
        var walked := 0.0
        for i in range(1, path.size()):
            walked += path[i - 1].distance_to(path[i])
        if path.size() >= 2 and path[path.size() - 1].distance_to(p) < 0.5 and walked <= NEARBY_DETOUR * straight:
            return p
    return fallback if fallback != null else global_position


## Whether another character (the player, an NPC) stands within NEARBY_MIN of a point.
func _someone_at(p: Vector3) -> bool:
    for other in get_tree().get_nodes_in_group(CHARACTER_GROUP):
        if other != self and other is Node3D and (other as Node3D).global_position.distance_to(p) < NEARBY_MIN:
            return true
    return false


func _random_home_point():
    var lead = _group_lead()
    if lead != null:
        var near = _point_near_leader(lead)
        if near != null:
            return near
    return _wander_point()


func _wander_point():
    var tags: Array = definition.get("wander", {}).get("tags", [])
    if not tags.is_empty():
        return _random_surface_point(tags)
    return _random_point_in(home, home_y)


# --- held items (geogen.87.2) ------------------------------------------------

var _skeleton: Skeleton3D = null
var _held: Node3D = null          # the BoneAttachment3D showing the held item's body
var _held_at := 0.0


## What the NPC shows in its hand: the first item it carries whose file says `held`; "" for none.
func _held_item() -> String:
    if world.rules == null:
        return ""
    for entry in world.rules.carried(WorldLoader.actor_id(self)):
        if world.item_held.has(str(entry["id"])) and str(entry["id"]) != str(_set_out.get("item", "")):
            return str(entry["id"])
    return ""


## Show ``id``'s body on its bone (none: empty hands); nothing changes while it is the same item.
func _hold(id: String) -> void:
    if _held != null and is_instance_valid(_held) and str(_held.get_meta("held_item", "")) == id:
        return
    if _held != null and is_instance_valid(_held):
        _held.queue_free()
    _held = null
    if id == "":
        return
    var body := world.held_body(id)
    var bone := str(world.item_held[id].get("bone", "RightHand"))
    if body == null or _skeleton.find_bone(bone) < 0:
        return
    var attach := BoneAttachment3D.new()
    attach.name = "Held"
    attach.bone_name = bone
    attach.set_meta("held_item", id)
    _skeleton.add_child(attach)
    attach.add_child(body)
    _held = attach
    if _item_state.has(id):
        show_state(body, str(_item_state[id]))


## The item in the NPC's hand now ("" for none), for the report.
func holding() -> String:
    return str(_held.get_meta("held_item", "")) if _held != null and is_instance_valid(_held) else ""


# --- groups (geogen.87.3) ---------------------------------------------------

var _mates: Array = []
var _mates_at := -INF


## The stable ID of this NPC's group leader ("" for none); the leader's is its own.
func group_leader() -> String:
    return str(definition.get("group", {}).get("leader", ""))


func _group_radius() -> float:
    return float(definition.get("group", {}).get("radius", 4.0))


## The other people of this NPC's group who are here now (not away, not absent).
func _groupmates() -> Array:
    if group_leader() == "":
        return []
    if clock - _mates_at > GROUP_REFRESH:
        _mates_at = clock
        _mates = world.npcs.filter(func(o): return o != self and is_instance_valid(o) and o.group_leader() == group_leader())
    return _mates.filter(func(o): return is_instance_valid(o) and not o.away and not o.absent)


## A member's leader when it is here; null for the leader itself or one without a group.
func _group_lead():
    var id := group_leader()
    if id == "" or id == WorldLoader.npc_uid(self):
        return null
    for o in _groupmates():
        if WorldLoader.npc_uid(o) == id:
            return o
    return null


func _near_groupmate() -> bool:
    var r := _group_radius()
    for o in _groupmates():
        var d: Vector3 = o.global_position - global_position
        if Vector2(d.x, d.z).length() <= r and absf(d.y) < FLOOR_TOLERANCE:
            return true
    return false


## Together refills near a groupmate and decays apart (the decay ran already); alone in the
## scene (the rest away), there is nobody to miss.
func _keep_together(delta: float) -> void:
    if _groupmates().is_empty():
        needs["together"] = 1.0
    elif _near_groupmate():
        needs["together"] = minf(1.0, needs["together"] + GROUP_REFILL * delta)
    else:
        stats["apart"] = float(stats.get("apart", 0.0)) + delta


## Who a rejoin walks to: a member's leader (or, with the leader away, its nearest groupmate);
## the leader's nearest member. Null when nobody is here.
func _rejoin_target():
    var lead = _group_lead()
    if lead != null:
        return lead
    var best = null
    for o in _groupmates():
        if best == null or global_position.distance_to(o.global_position) < global_position.distance_to(best.global_position):
            best = o
    return best


## The navmesh point beside the groupmate a rejoin goes to (on this side of them), or null.
func _rejoin_point():
    var mate = _rejoin_target()
    if mate == null:
        return null
    var at: Vector3 = mate.global_position
    var side := NpcGeometry.flat(global_position - at)
    side = side.normalized() if side.length() > 1e-3 else Vector3(rng.randf() - 0.5, 0, rng.randf() - 0.5).normalized()
    return NavigationServer3D.map_get_closest_point(get_world_3d().navigation_map, at + side * GROUP_REJOIN_GAP)


## A member's wander: one of its usual points within reach of the leader (where it is, or where
## it is going), else any navmesh point near there.
func _point_near_leader(lead: Node3D):
    var r := _group_radius() * GROUP_WANDER
    var at := _heading_for(lead)
    var c := Vector2(at.x, at.z)
    var poly := PackedVector2Array([c + Vector2(-r, -r), c + Vector2(r, -r), c + Vector2(r, r), c + Vector2(-r, r)])
    for i in 12:
        # Its usual kind of spot first (the paths), then anywhere on the navmesh near the leader.
        var p = _wander_point() if i < 8 else _random_point_in(poly, at.y)
        if p != null and Vector2((p as Vector3).x, (p as Vector3).z).distance_to(c) <= r and not _taken_spot(p):
            return p
    return null


## Whether another character stands within GROUP_GAP of a point (not a spot to walk to).
func _taken_spot(p: Vector3) -> bool:
    for other in get_tree().get_nodes_in_group(CHARACTER_GROUP):
        if other != self and other is Node3D and NpcGeometry.flat((other as Node3D).global_position - p).length() < GROUP_GAP:
            return true
    return false


## Where a groupmate is, or where it is going when it has taken an affordance.
static func _heading_for(mate) -> Vector3:
    var a: Dictionary = mate._option.get("affordance", {})
    return a["approach"] if not a.is_empty() else mate.global_position


## Whether a groupmate holds (or is on its way to) this affordance's thing or one beside it.
func _beside_groupmate(a: Dictionary) -> bool:
    for o in _groupmates():
        var theirs: Dictionary = o._option.get("affordance", {})
        if theirs.is_empty() or not is_instance_valid(theirs.get("node")):
            continue
        if theirs["node"] == a["node"] or (theirs["position"] as Vector3).distance_to(a["position"]) < GROUP_BESIDE:
            return true
    return false


## A random navmesh point inside a region (world XZ polygon at floor height y), or null.
func _random_point_in(poly: PackedVector2Array, y: float):
    if poly.size() < 3:
        return null
    var lo := poly[0]
    var hi := poly[0]
    for p in poly:
        lo = Vector2(minf(lo.x, p.x), minf(lo.y, p.y))
        hi = Vector2(maxf(hi.x, p.x), maxf(hi.y, p.y))
    for _i in 20:
        var p := Vector2(rng.randf_range(lo.x, hi.x), rng.randf_range(lo.y, hi.y))
        if Geometry2D.is_point_in_polygon(p, poly):
            var map := get_world_3d().navigation_map
            var q := NavigationServer3D.map_get_closest_point(map, Vector3(p.x, y, p.y))
            if Vector2(q.x - p.x, q.z - p.y).length() < 0.5 and absf(q.y - y) < 0.5:
                return q
    return null


var _wander_tris := PackedVector3Array()   # up-facing triangles (world) of the wander surfaces
var _wander_area := PackedFloat32Array()   # cumulative areas


## A random point (area-weighted) on the up-facing faces of nodes tagged ``tags``, inside the
## home region, snapped onto the navmesh.
func _random_surface_point(tags: Array):
    if _wander_tris.is_empty():
        var total := 0.0
        for tag in tags:
            for node in get_tree().get_nodes_in_group(tag):
                for mi: MeshInstance3D in ([node] if node is MeshInstance3D else []) + node.find_children("*", "MeshInstance3D", true, false):
                    if mi.mesh == null or mi.is_in_group("geogen_collider_debug"):
                        continue                   # (collider overlays aren't more pavement)
                    var faces := mi.mesh.get_faces()
                    var xf := mi.global_transform
                    for k in range(0, faces.size(), 3):
                        var a := xf * faces[k]
                        var b := xf * faces[k + 1]
                        var c := xf * faces[k + 2]
                        var n := (b - a).cross(c - a)
                        if absf(n.y) < 0.9 * n.length() or n.length() < 1e-4:
                            continue
                        if home.size() >= 3 and not Geometry2D.is_point_in_polygon(Vector2((a.x + b.x + c.x) / 3.0, (a.z + b.z + c.z) / 3.0), home):
                            continue
                        total += n.length() / 2.0
                        _wander_tris.append_array([a, b, c])
                        _wander_area.append(total)
    if _wander_area.is_empty():
        return null
    var map := get_world_3d().navigation_map
    for _i in 10:
        var pick := rng.randf() * _wander_area[-1]
        var k := _wander_area.bsearch(pick)
        var u := rng.randf()
        var v := rng.randf()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        var a := _wander_tris[3 * k]
        var p := a + (_wander_tris[3 * k + 1] - a) * u + (_wander_tris[3 * k + 2] - a) * v
        var q := NavigationServer3D.map_get_closest_point(map, p)
        if Vector2(q.x - p.x, q.z - p.z).length() < 0.6 and absf(q.y - p.y) < 0.5:
            return q
    return null


func _plan(target: Vector3) -> String:
    var map := get_world_3d().navigation_map
    var from := NavigationServer3D.map_get_closest_point(map, global_position)
    var to := NavigationServer3D.map_get_closest_point(map, target)
    _path = GeogenWalker.find_path(map, from, to)
    _stairs = []
    _stair = null
    _stair_wait = 0.0
    _waypoint = 1
    _detour = -1
    _assertive = -1
    _vehicle_wait = 0.0
    _vehicle_detours = 0
    _signal_wait = 0.0
    _best = INF
    _stall = 0.0
    _sidestep = 0.0
    if _path.is_empty():
        return "no path"
    if Vector2(to.x - target.x, to.z - target.z).length() > 0.8:
        return "target off the navmesh (%.1f m)" % Vector2(to.x - target.x, to.z - target.z).length()
    var over := _streets.crosswalk_route(map, _path, from, to)
    if not over.is_empty():
        _path = over
        stats["crosswalks"] = int(stats.get("crosswalks", 0)) + 1
    # A navigation link on the way (a spiral stair) is two points, a straight line through the
    # stair: walk its way round instead. Last, so the reroute above sees the navmesh's own path.
    _path = GeogenWalker.along_links(_path, get_tree(), _stairs)
    if _close:
        _close_in(to, target)
    return ""


## `go_to` with `close: true`: the navmesh keeps a player's width off everything, but a counter
## is worked at from closer. Past the path's end, step straight up to the target, when it is
## within CLOSE_REACH and this body's width can move there without touching anything.
func _close_in(end: Vector3, target: Vector3) -> void:
    var gap := Vector3(target.x - end.x, 0.0, target.z - end.z)
    if gap.length() < 0.02 or gap.length() > CLOSE_REACH:
        return
    var cyl := CylinderShape3D.new()
    cyl.radius = float(definition.get("radius", 0.25))
    cyl.height = float(definition.get("height", 1.75)) - 0.3
    var query := PhysicsShapeQueryParameters3D.new()
    query.shape = cyl
    query.transform = Transform3D(Basis.IDENTITY, end + Vector3.UP * (0.2 + cyl.height / 2.0))
    query.motion = gap
    query.exclude = [get_rid()]
    query.collision_mask = 1
    var free := get_world_3d().direct_space_state.cast_motion(query)
    if free.size() == 2 and float(free[0]) >= 1.0:
        _path.append(end + gap)
        stats["close_ins"] = int(stats.get("close_ins", 0)) + 1


# Stairs are walked in single file. A spiral stair's navigation link is one body wide, and two
# NPCs meeting on it push each other off the treads: whoever gets to an end while someone is
# coming the other way waits there, a little back from the end so the other can step off.
var _stairs := []                  # [{link, first, last, down}]: the path's stretches along links
var _stair: NavigationLink3D       # the link we're on (others read it: see _stair_free)
var _stair_down := false
var _stair_wait := 0.0
var _replans := 0


## May we walk on? False while the path is about to take a stair someone is coming along the
## other way. Taking one marks us as on it, until the path leaves it or is dropped.
func _stair_free() -> bool:
    var ahead := 0.0
    var at := global_position
    for i in range(_waypoint, _path.size()):
        ahead += at.distance_to(_path[i])
        at = _path[i]
        if ahead > STAIR_HOLD:
            break
        for span in _stairs:
            if i < span["first"] or i > span["last"] or not is_instance_valid(span["link"]):
                continue
            if _stair == span["link"]:
                return true
            for other in get_tree().get_nodes_in_group(CHARACTER_GROUP):
                if other is GeogenNpc and other != self and other._stair == span["link"] \
                        and other._stair_down != span["down"]:
                    return false
            if _player_in_the_way(span["link"], span["down"]):
                return false
            _stair = span["link"]
            _stair_down = span["down"]
            return true
    _stair = null      # no stair in the next few steps: off the one we were on, if any
    return true


## The player holds no claim on a stair, but they're on it as much as anyone: within
## STAIR_REACH of its walking line, standing there or coming the other way, they keep an NPC
## from taking it (one going the same way it follows). An NPC already on it carries on: the
## player can back off, it can't.
func _player_in_the_way(link: NavigationLink3D, down: bool) -> bool:
    var who := _player()
    if who == null:
        return false
    var walk: PackedVector3Array = link.get_meta("walk", PackedVector3Array())
    var line := walk if walk.size() >= 2 else PackedVector3Array([link.start_position, link.end_position])
    var p := who.global_position
    var near := INF
    for k in range(1, line.size()):
        var a := link.global_transform * line[k - 1]
        var b := link.global_transform * line[k]
        near = minf(near, p.distance_to(Geometry3D.get_closest_point_to_segment(p, a, b)))
    if near > STAIR_REACH:
        return false
    var climb: float = (who as CharacterBody3D).velocity.y if who is CharacterBody3D else 0.0
    if absf(climb) < 0.1:
        return true                     # standing on it
    return (climb < 0.0) != down


## Is waypoint ``index`` on a stair's way round (``after_first``: and not where it starts)?
## There the line is the only way: nothing is moved aside or walked around.
func _on_stair(index: int, after_first := false) -> bool:
    for span in _stairs:
        if index >= span["first"] + (1 if after_first else 0) and index <= span["last"]:
            return true
    return false


## A waypoint added in front of ``index`` moves the stair stretches after it along.
func _insert_waypoint(index: int, point: Vector3) -> void:
    _path.insert(index, point)
    for span in _stairs:
        if span["first"] >= index:
            span["first"] += 1
        if span["last"] >= index:
            span["last"] += 1


## On the kerb with a signalled crossing's road in the next step: wait for walk. Not once we've
## stepped off (someone caught by the flashing don't-walk finishes crossing), and whatever the
## path: the rule is where we stand and where we're headed. True while waiting.
func _wait_for_walk(delta: float) -> bool:
    var here := Vector2(global_position.x, global_position.z)
    var to := NpcGeometry.flat(_path[_waypoint] - global_position)
    var step := to.normalized() * minf(to.length(), KERB_STEP)
    var id := _streets.crossing_from_kerb(here, here + Vector2(step.x, step.z))
    var state := "" if id == "" else _streets.pedestrian_state(id)
    if state == "walk" or state == "":
        _signal_wait = 0.0
        return false
    if _signal_wait == 0.0:
        stats["signal_waits"] = int(stats.get("signal_waits", 0)) + 1
        _trace({"event": "signal_wait", "crossing": id, "state": state})
    _signal_wait += delta
    stats["signal_wait"] = maxf(float(stats.get("signal_wait", 0.0)), _signal_wait)
    # Stand facing the crossing.
    if to.length() > 0.05:
        var turn := deg_to_rad(float(definition.get("turn_speed", 360.0))) * delta
        rotation.y += clampf(wrapf(atan2(to.x, to.z) - _yaw(), -PI, PI), -turn, turn)
    velocity.x = 0.0
    velocity.z = 0.0
    _move(delta)
    return true


## Count our steps off a kerb onto a signalled crossing: `signal_crossings`, and
## `signal_violations` for the ones made while it showed a steady don't-walk.
func _watch_crossing() -> void:
    var here := Vector2(global_position.x, global_position.z)
    # (A frame's step: not where we were before sitting down, or before being put back on the navmesh.)
    var id := _streets.crossing_from_kerb(_was_at, here) if _was_at.distance_to(here) < 0.5 else ""
    _was_at = here
    if id == "":
        return
    var state := _streets.pedestrian_state(id)
    var key := "signal_violations" if state == "dont" else "signal_crossings"
    stats[key] = int(stats.get(key, 0)) + 1
    _trace({"event": "signal_step", "crossing": id, "state": state, "position": _v(global_position)})


func _walk(delta: float) -> String:
    if _waypoint >= _path.size():
        velocity.x = 0.0
        velocity.z = 0.0
        _stair = null
        _move(delta)
        return "done"
    # A closed door ahead: go through it first.
    var door := _portal_ahead()
    if not door.is_empty():
        stats["passes"] += 1
        _push("pass", door)
        return "running"
    _watch_crossing()
    if _wait_for_walk(delta):
        return "running"
    if _waypoint != _assertive and not _clear_of_vehicles():
        # Traffic coming: wait at the kerb (not stalling), but not forever: after a while step
        # out anyway until the next waypoint; vehicles yield to people in their way.
        _vehicle_wait += delta
        if _vehicle_wait > VEHICLE_PATIENCE:
            _assertive = _waypoint
        velocity.x = 0.0
        velocity.z = 0.0
        _move(delta)
        return "running"
    _vehicle_wait = 0.0
    if not _stair_free():
        # Someone is coming along the stair: wait for them (not stalling), but not all day.
        _stair_wait += delta
        velocity.x = 0.0
        velocity.z = 0.0
        _move(delta)
        return "stuck: stair in use" if _stair_wait > STAIR_PATIENCE else "running"
    _stair_wait = 0.0
    if not _detour_around_characters():
        # Someone is standing on our destination: wait (the stall timer gives up eventually).
        velocity.x = 0.0
        velocity.z = 0.0
        _move(delta)
        _stall += delta
        return "stuck: destination occupied" if _stall > FAIL_SECONDS else "running"
    var target := _path[_waypoint]
    var to := target - global_position
    var flat := Vector2(to.x, to.z)
    var speed := float(definition.get("speed", 1.2))
    # (On its own level: a waypoint a turn of a spiral stair above isn't reached from below.)
    # Stepping up to a counter (`close`), the last waypoint is the spot itself.
    var radius := CLOSE_RADIUS if _close and _waypoint == _path.size() - 1 else WAYPOINT_RADIUS
    if flat.length() < maxf(radius, speed * delta) and absf(to.y) < LEVEL_TOLERANCE:
        _waypoint += 1
        _best = INF
        _stall = 0.0
        return "running"
    var dir := Vector3(to.x, 0, to.z).normalized()
    if not _on_stair(_waypoint, true):
        dir = _avoid(dir)
    var heading := atan2(dir.x, dir.z)
    var turn := deg_to_rad(float(definition.get("turn_speed", 360.0))) * delta
    rotation.y += clampf(wrapf(heading - _yaw(), -PI, PI), -turn, turn)
    if _sidestep > 0.0:
        _sidestep -= delta
        dir = (dir + dir.cross(Vector3.UP) * _sidestep_dir * 1.5).normalized()
    velocity.x = dir.x * speed
    velocity.z = dir.z * speed
    _move(delta)
    if flat.length() < _best - 0.03:
        _best = flat.length()
        _stall = 0.0
    else:
        _stall += delta
        stats["max_stall"] = maxf(stats["max_stall"], _stall)
        if _stall > FAIL_SECONDS:
            _wedged = true
            return "stuck at %s" % str(_v(global_position))
        if _stall > REPLAN_SECONDS and _waypoint <= _detour and _replans < MAX_REPLANS:
            # Held up on a way round someone (who may have gone since): forget it, plan afresh.
            _replans += 1
            var err := _plan(_path[-1])
            return "running" if err == "" else err
        if _stall > STALL_SECONDS and _sidestep <= 0.0 and fmod(_stall, STALL_SECONDS * 2.0) < delta * 1.5:
            _sidestep = SIDESTEP_SECONDS
            _sidestep_dir = -_sidestep_dir
    return "running"


## Characters aren't in the navmesh, so walk around them: path waypoints
## inside a character's reach move out to its edge, and a segment that
## crosses one gets an arc of waypoints around it, on the shorter side whose
## points are all on the navmesh (which is already shrunk by the agent
## radius, so a point on it has room for us). Returns false while a
## character is standing on the destination itself (wait for them).
func _detour_around_characters() -> bool:
    var radius := float(definition.get("radius", 0.25))
    var map := get_world_3d().navigation_map
    var last := _path.size() - 1
    _characters()
    var widest := radius + _crowd_radius + AVOID_MARGIN
    var goal := Vector2(_path[last].x, _path[last].z)
    for entry in _characters_in(goal - Vector2(widest, widest), goal + Vector2(widest, widest)):
        var c: Vector3 = entry["pos"]
        if entry["node"] != self and NpcGeometry.flat(_path[last] - c).length() < radius + float(entry["radius"]) + AVOID_MARGIN \
                and NpcGeometry.same_level(_path[last], c):
            return false   # someone is standing where we're going
    if _waypoint <= _detour:
        return true
    # The waypoints of the next DETOUR_HORIZON m (not the destination); farther ones are
    # seen to as they come up. Only characters within reach of these, or of the way to the
    # next one, matter.
    var end := _waypoint
    var ahead := 0.0
    var at := global_position
    var lo := Vector2(at.x, at.z)
    var hi := lo
    while end < last and ahead < DETOUR_HORIZON:
        ahead += at.distance_to(_path[end])
        at = _path[end]
        lo = lo.min(Vector2(at.x, at.z))
        hi = hi.max(Vector2(at.x, at.z))
        end += 1
    for entry in _characters_in(lo - Vector2(widest, widest), hi + Vector2(widest, widest)):
        if entry["node"] == self:
            continue
        var c: Vector3 = entry["pos"]
        var reach := radius + float(entry["radius"]) + AVOID_MARGIN
        # Waypoints inside its reach move out to the edge.
        for i in range(_waypoint, end):
            if NpcGeometry.flat(_path[i] - c).length() < reach and NpcGeometry.same_level(_path[i], c) and not _on_stair(i):
                var out = NpcGeometry.edge_point(c, _path[i], reach + 0.05, map)
                if out != null:
                    _path[i] = out
        var a := global_position
        var b := _path[_waypoint]
        var seg := NpcGeometry.flat(b - a)
        if seg.length() < 0.05 or not NpcGeometry.same_level(a, c) or _on_stair(_waypoint, true):
            continue
        var dir := seg.normalized()
        var rel := NpcGeometry.flat(c - a)
        var along := rel.dot(dir)
        if along <= 0.0 or along > minf(seg.length() + reach, AVOID_AHEAD + reach):
            continue
        if (rel - dir * along).length() >= reach:
            continue
        var arc := NpcGeometry.arc_around(c, a, b, reach + 0.05, map)
        if arc.is_empty():
            continue   # boxed in: press on and let the stall timer decide
        for k in arc.size():
            _insert_waypoint(_waypoint + k, arc[k])
        _detour = _waypoint + arc.size() - 1
        _best = INF
        return true    # the rest of the path has moved along: the others next frame
    return true


## Vehicles aren't in the navmesh either. A moving one whose next few seconds of travel crosses
## the next steps of our path makes us wait (unless we're already in its way: then press on and
## it yields); a stopped one in the way gets a detour round its nearer end.
func _clear_of_vehicles() -> bool:
    var radius := float(definition.get("radius", 0.25))
    var a := global_position
    var seg := NpcGeometry.flat(_path[_waypoint] - a)
    if seg.length() < 0.05:
        return true
    var probe := a + seg.normalized() * minf(seg.length(), VEHICLE_PROBE)
    var map := get_world_3d().navigation_map
    for car in get_tree().get_nodes_in_group(VEHICLE_GROUP):
        var box: Dictionary = car.get_meta("geogen_vehicle_box", {})
        if box.is_empty():
            continue
        var xf: Transform3D = car.global_transform
        if absf(xf.origin.y - a.y) > 2.0:
            continue
        # Too far off for its box, swept ahead, to reach the probed stretch of our path.
        var speed_now := float(car.get_meta("geogen_speed", 0.0))
        if NpcGeometry.flat(xf.origin - a).length() > absf(float(box["offset"])) + float(box["half"]) + float(box["half_w"]) \
                + speed_now * VEHICLE_LOOKAHEAD + VEHICLE_PROBE + radius + VEHICLE_MARGIN:
            continue
        var fwd := NpcGeometry.flat(xf.basis.z).normalized()
        var right := Vector3(fwd.z, 0.0, -fwd.x)
        var center: Vector3 = xf.origin + fwd * float(box["offset"])
        var hl := float(box["half"]) + radius + VEHICLE_MARGIN
        var hw := float(box["half_w"]) + radius + VEHICLE_MARGIN
        var speed := float(car.get_meta("geogen_speed", 0.0))
        if speed > 0.5:
            var reach := speed * VEHICLE_LOOKAHEAD
            var swept_c := center + fwd * reach / 2.0
            if NpcGeometry.in_box(a, swept_c, fwd, right, hl + reach / 2.0, hw):
                continue          # already in its way: keep going, it yields to us
            if NpcGeometry.segment_hits_box(a, probe, swept_c, fwd, right, hl + reach / 2.0, hw):
                return false
        elif _waypoint > _detour and _vehicle_detours < 2 and NpcGeometry.segment_hits_box(a, probe, center, fwd, right, hl, hw):
            # (Also when pressed up against it: the corners on our side lead round it.)
            _vehicle_detours += 1
            var detour := NpcGeometry.around_box(a, _path[_waypoint], center, fwd, right, hl + 0.05, hw + 0.05, map)
            if not detour.is_empty():
                for k in detour.size():
                    _insert_waypoint(_waypoint + k, detour[k])
                _detour = _waypoint + detour.size() - 1
                _best = INF
    return true


## Bend ``dir`` around characters standing in the way (static geometry is the navmesh's job).
func _avoid(dir: Vector3) -> Vector3:
    var radius := float(definition.get("radius", 0.25))
    var steer := Vector3.ZERO
    var p := Vector2(global_position.x, global_position.z)
    _characters()
    var near := AVOID_AHEAD + radius + _crowd_radius + AVOID_MARGIN
    for entry in _characters_in(p - Vector2(near, near), p + Vector2(near, near)):
        if entry["node"] == self:
            continue
        var c: Vector3 = entry["pos"]
        if not NpcGeometry.same_level(c, global_position):
            continue
        var rel := NpcGeometry.flat(c - global_position)
        var along := rel.dot(dir)
        if along <= 0.0 or along > AVOID_AHEAD:
            continue
        var lateral := rel - dir * along
        var reach := radius + float(entry["radius"]) + AVOID_MARGIN
        if lateral.length() >= reach:
            continue
        # Pass on the side it's already off to (the right if dead ahead).
        var away := -lateral.normalized() if lateral.length() > 0.02 else dir.cross(Vector3.UP)
        steer += away * (1.0 - lateral.length() / reach) * (1.0 - along / AVOID_AHEAD) * 3.0
    return (dir + steer).normalized() if steer != Vector3.ZERO else dir


func _move(delta: float) -> void:
    if is_on_floor():
        velocity.y = 0.0
        if Player.step_up(self, Vector3(velocity.x, 0, velocity.z) * delta, step_height):
            return
    else:
        velocity.y -= _gravity * delta
    var scale := delta / _frame_delta
    if scale > 1.01:      # a throttled step covers skipped frames too
        velocity *= scale
        move_and_slide()
        velocity /= scale
    else:
        move_and_slide()


## Whether to step every other frame: off-screen and not near the player, or far off. Not
## while greeting, in a transition or on a stair, where timing and single file matter.
func _throttled() -> bool:
    return _rate() > 1


## How many physics frames apart it steps: 1 near or in view, 2 off screen (or far), SLEEP_EVERY
## far off and out of sight. Never while moving between poses, on a stair or attending someone.
func _rate() -> int:
    if away or _moving_pose() or _stair != null or clock < _attend_until:
        return 1
    var camera := get_viewport().get_camera_3d()
    var who := _player()
    if camera == null or who == null:
        return 1
    var d := NpcGeometry.flat(who.global_position - global_position).length()
    var seen := camera.is_position_in_frustum(global_position + Vector3.UP * 0.9)
    if d > SLEEP_DISTANCE and not seen:
        return SLEEP_EVERY
    if d > THROTTLE_FAR:
        return 2
    return 2 if d > THROTTLE_NEAR and not seen else 1


## The first closed portal the remaining path crosses within PORTAL_LOOKAHEAD, as a pass context.
func _portal_ahead() -> Dictionary:
    if _stack.back()["action"] == "pass":
        return {}
    var door := world.closed_portal_on(_path, global_position, _waypoint, PORTAL_LOOKAHEAD)
    if door.is_empty():
        return {}
    return {"portal": door["portal"], "side": door["side"], "interaction": door["interaction"],
        "anchor": door["portal"]["center"], "yaw_deg": 0.0, "duration": [0.0, 0.0]}


# --- body -----------------------------------------------------------------

## A keyframed move between poses is playing (sit_down, stand_up, ...).
func _moving_pose() -> bool:
    return animator != null and animator.moving()


## World heading (rad). The NPC sits under its placement's parent, which may be turned
## (a cottage facing south), while affordance and path headings are in world space.
func _yaw() -> float:
    return rotation.y + _parent_yaw()


func _parent_yaw() -> float:
    var parent := get_parent_node_3d()
    if parent == null:
        return 0.0
    var z := parent.global_basis.z
    return atan2(z.x, z.z)


## Put the body in a pose: sit/lie at the anchor (collider off), stand where it stands.
func _set_pose(pose_name: String, ctx: Dictionary, at: String, hold := "") -> void:
    var pose: Dictionary = definition["body"]["poses"][pose_name]
    var yaw := deg_to_rad(float(ctx["yaw_deg"]))
    var place: Vector3 = ctx["anchor"] if at == "anchor" else ctx["approach"]
    var ride: Dictionary = ctx.get("ride", {})
    if not ride.is_empty():
        # A seat on a mechanism body: where it (and the way off it) is now, not at decision time.
        place = ride["position"] if at == "anchor" else ride["approach"]
        yaw = deg_to_rad(float(ride["npc_yaw_deg"]))
    var was := _pose
    var before := body.global_transform if body != null else Transform3D.IDENTITY
    if pose_name == "stand":
        var map := get_world_3d().navigation_map
        place = NavigationServer3D.map_get_closest_point(map, place)
    global_position = place
    rotation = Vector3(0, yaw - _parent_yaw(), 0)
    _pose_yaw = yaw
    velocity = Vector3.ZERO
    if pose_name == "stand":
        _solid_after_move()
    else:
        _shape.set_deferred("disabled", true)
    _pose = pose_name
    if at == "anchor" and pose.has("reach") and ctx.get("depth") != null:
        # Deep seats: sit forward until the shins clear the front edge.
        var forward := Basis(Vector3.UP, yaw) * Vector3.BACK
        global_position += forward * maxf(0.0, float(ctx["depth"]) + SEAT_CLEAR - float(pose["reach"]))
    if pose_name != "stand" and at == "anchor" and not ride.is_empty():
        # Go with the seat from here on.
        _ride = ride
        _ride_offset = GeogenMechanisms.seat_now(ride).affine_inverse() * global_transform
    else:
        _ride = {}
    if animator != null:
        animator.hold = hold if pose_name != "stand" else ""
    if body != null:
        var o: Array = pose["offset"]
        var r: Array = pose["rotation"]
        var s: Array = pose["scale"]
        var basis := Basis.from_euler(Vector3(deg_to_rad(r[0]), deg_to_rad(r[1]), deg_to_rad(r[2])))
        var target := Transform3D(basis.scaled_local(Vector3(s[0], s[1], s[2])), Vector3(o[0], o[1], o[2]))
        if animator.change_pose(was, pose_name, global_transform.affine_inverse() * before, target, self):
            _timer = POSE_SECONDS             # no extra hold once it has played
            stats["transitions"] = int(stats.get("transitions", 0)) + 1


# --- attention (the player) ---------------------------------------------------

## The player (the one character that isn't an NPC), or null.
func _player() -> Node3D:
    for entry in _characters():
        if not entry["npc"] and is_instance_valid(entry["node"]):
            return entry["node"]
    return null


## The characters this physics frame (see ``_crowd``): positions as the frame began.
func _characters() -> Array[Dictionary]:
    var frame := Engine.get_physics_frames()
    if frame != _crowd_frame:
        _crowd_frame = frame
        _crowd = []
        _crowd_grid = {}
        _crowd_radius = 0.0
        for other in get_tree().get_nodes_in_group(CHARACTER_GROUP):
            if other is CharacterBody3D:
                var entry := {"node": other, "pos": other.global_position,
                    "radius": NpcGeometry.radius_of(other), "npc": other is GeogenNpc}
                _crowd.append(entry)
                _crowd_radius = maxf(_crowd_radius, entry["radius"])
                var cell := Vector2i(floori(entry["pos"].x / CROWD_CELL), floori(entry["pos"].z / CROWD_CELL))
                if not _crowd_grid.has(cell):
                    _crowd_grid[cell] = []
                _crowd_grid[cell].append(entry)
    return _crowd


## The characters this physics frame whose cells overlap the plan rectangle ``lo``..``hi`` (x, z).
func _characters_in(lo: Vector2, hi: Vector2) -> Array:
    _characters()
    var out := []
    for cx in range(floori(lo.x / CROWD_CELL), floori(hi.x / CROWD_CELL) + 1):
        for cz in range(floori(lo.y / CROWD_CELL), floori(hi.y / CROWD_CELL) + 1):
            var cell = _crowd_grid.get(Vector2i(cx, cz))
            if cell != null:
                out.append_array(cell)
    return out


# --- perception (geogen.99) ------------------------------------------------------

## A bus event: is it one of the stimuli this NPC reacts to (definition `reacts`, from
## npcs/stimuli.yaml), not its own doing, and does it hear or see it (`senses`)?
func _on_event(event: Dictionary) -> void:
    if absent or away or event.get("actor", "") == WorldLoader.actor_id(self):
        return
    var reacts: Dictionary = definition.get("reacts", {})
    for stimulus in reacts:
        var r: Dictionary = reacts[stimulus]
        if r["event"] != event["type"]:
            continue
        var node = world.node_by_uid(str(event.get("subject", "")))
        if r.has("of") and not (node is Node and GeogenPlaceSearch._tags(node).any(
                func(t): return GeogenWants._tag_is(str(t), str(r["of"])))):
            continue
        var data: Dictionary = event.get("data", {})
        var point: Vector3
        if data.get("at") is Array:
            point = Vector3(data["at"][0], data["at"][1], data["at"][2])
        elif node is Node3D:
            point = (node as Node3D).global_position
        else:
            continue
        var by := _perceives(point, float(r["loudness"]) * float(data.get("strength", 1.0)))
        if by == "":
            continue
        stats["noticed"] = int(stats.get("noticed", 0)) + 1
        _noticed = {"stimulus": stimulus, "subject": str(event.get("subject", "")), "at": _v(point), "by": by,
            "clock": snappedf(clock, 0.01)}
        var look: Array = r["look"]
        _notice_point = point
        _notice_until = clock + rng.randf_range(float(look[0]), float(look[1]))
        if r["interrupt"]:
            _cut_until = clock + CUT_PATIENCE
        if GeogenMemory.has(self):
            GeogenMemory.noticed(self, str(event.get("subject", "")), point)
        var traced_event := {"event": "notice"}
        traced_event.merge(_noticed)
        _trace(traced_event)
        return


## How this NPC perceives something at ``point`` that carries ``loud`` of its hearing range:
## "heard" (within that; only `hearing.muffled` of it with something in the way), "seen" (within
## `sight.range` and its cone, with a clear line), or "".
func _perceives(point: Vector3, loud: float) -> String:
    var senses: Dictionary = definition.get("senses", {})
    var eyes := head_point()
    var distance := eyes.distance_to(point)
    var hit := get_world_3d().direct_space_state.intersect_ray(
        PhysicsRayQueryParameters3D.create(eyes, point, 1, [get_rid()]))
    var clear := hit.is_empty() or (hit["position"] as Vector3).distance_to(point) < SEEN_NEAR
    var hearing: Dictionary = senses.get("hearing", {})
    if not hearing.is_empty() and distance <= float(hearing["range"]) * loud * (1.0 if clear else float(hearing["muffled"])):
        return "heard"
    return "seen" if clear and _in_view(point) else ""


## Whether ``point`` is within `sight.range` of the eyes and the cone ahead (a clear line aside).
func _in_view(point: Vector3) -> bool:
    var sight: Dictionary = definition.get("senses", {}).get("sight", {})
    if sight.is_empty() or head_point().distance_to(point) > float(sight["range"]):
        return false
    var to := NpcGeometry.flat(point - global_position)
    var forward := Basis(Vector3.UP, _yaw()) * Vector3.BACK
    return to.length() < 1e-3 or rad_to_deg(forward.angle_to(to)) <= float(sight["cone"]) / 2.0


## Whether it sees ``point``: in view, with nothing but ``exclude`` (the thing itself) in the way.
func _sees(point: Vector3, exclude: Array[RID] = []) -> bool:
    if not _in_view(point):
        return false
    var skip: Array[RID] = [get_rid()]
    skip.append_array(exclude)
    var hit := get_world_3d().direct_space_state.intersect_ray(
        PhysicsRayQueryParameters3D.create(head_point(), point, 1, skip))
    return hit.is_empty() or (hit["position"] as Vector3).distance_to(point) < SEEN_NEAR


## Cut what it is doing short, for something it noticed: only an option of its own choosing
## (an affordance or an activity) whose action is `interruptible`, with empty hands, off a ride
## and outside a routine's hold. Seated or lying, it goes to the step that stands it up; on its
## feet, the action just ends. The option counts as done: it decides again. False while it
## can't yet (a reaction or a door is on top, the body is mid-move); true once done or not to be.
func _cut_short() -> bool:
    if _stack.size() > 1 or _moving_pose():
        return false
    if _stack.is_empty() or _option.is_empty() or not (_option.has("affordance") or _option.has("activity")):
        return true
    var frame: Dictionary = _stack[0]
    if not definition["actions"][frame["action"]].get("interruptible", false) or prop_in_hand() != "" \
            or not _ride.is_empty() or frame["ctx"].get("held", false):
        return true
    var steps: Array = frame["steps"]
    var to: int = steps.size()
    if _pose != "stand":
        for i in range(frame["index"], steps.size()):
            if steps[i].get("pose", "") == "stand":
                to = i
                break
        if to == frame["index"] and frame["started"]:
            return true                    # getting up already
    velocity = Vector3.ZERO
    _path = PackedVector3Array()
    _playing = false
    _looping = false
    _stop_motions()
    frame["index"] = to
    frame["started"] = false
    stats["cut_short"] = int(stats.get("cut_short", 0)) + 1
    _trace({"event": "cut_short", "option": _option["id"], "for": _noticed.get("stimulus", "")})
    return true


## Whether this NPC meets affordance ``a``'s `requires` now: carrying one of its `holding`
## items, and its asset's interactions showing its `state`s.
func _meets(a: Dictionary) -> bool:
    var need: Dictionary = a.get("requires", {})
    var actor := WorldLoader.actor_id(self)
    var holding: Array = need.get("holding", [])
    if not holding.is_empty() and (world.rules == null or not holding.any(func(id): return world.rules.held(str(id), actor) > 0)):
        return false
    var states: Dictionary = need.get("state", {})
    for name in states:
        var it = world.interactions_by_key.get("%s#%s" % [WorldLoader.uid_of(a["node"]), name])
        if not it is GeogenInteraction or (it as GeogenInteraction).state != str(states[name]):
            return false
    return true


## An affordance's action is done: the states it `sets`, and what it `takes` and `gives` (to what
## this NPC carries).
func _carry_out(a: Dictionary) -> void:
    for name in a.get("sets", {}):
        var it = world.interactions_by_key.get("%s#%s" % [WorldLoader.uid_of(a["node"]), name])
        if it is GeogenInteraction:
            (it as GeogenInteraction).set_state(str(a["sets"][name]), false, WorldLoader.actor_id(self))
    if world.rules == null:
        return
    var actor := WorldLoader.actor_id(self)
    for id in a.get("takes", []):
        if world.rules.held(str(id), actor) > 0:
            world.rules.give(str(id), -1, actor)
            break
    var gives: Dictionary = a.get("gives", {})
    for id in gives:
        world.rules.give(str(id), int(gives[id]), actor)
    if a.has("takes") or not gives.is_empty():
        _trace({"event": "carry", "option": a["id"], "takes": a.get("takes", []), "gives": gives,
            "carrying": world.rules.carried(actor).map(func(e): return e["id"])})


## The point a step's `look:` has the head turned to now (the action's subject or place), or null.
func _intent_point():
    if _intent == "" or _pose != "stand":
        return null
    if _intent == "partner":
        var other = _intent_ctx.get("partner")
        return (other as GeogenNpc).head_point() if is_instance_valid(other) else null
    if _intent == "anchor":
        return _intent_ctx.get("anchor")
    if _intent == "subject":
        var uid := str(_intent_ctx.get("subject", ""))
        return _prop_centre(uid) if world.props != null and world.props.props.has(uid) else null
    var place: Dictionary = _intent_ctx.get("place", {})
    if not place.has("position"):
        return null
    # The middle of the thing once it is there, not the top it stands on.
    var held: String = world.props.taken_by(self) if world.props != null else ""
    var lift: float = (world.props.box_of(held)["size"] as Vector3).y / 2.0 if held != "" else 0.0
    return (place["position"] as Vector3) + Vector3.UP * lift


## Busy at work (a looping `play` step), the head comes up only now and then, for `busy_glance` s
## every `busy_every` s; otherwise always.
func _glancing(look: Dictionary) -> bool:
    if not _looping:
        return true
    if clock >= _glance_at:
        _glance_until = clock + float(look.get("busy_glance", 1.2))
        _glance_at = _glance_until + _busy_gap()
    return clock < _glance_until


func _busy_gap() -> float:
    var every: Array = definition.get("attention", {}).get("look", {}).get("busy_every", [5.0, 12.0])
    return rng.randf_range(float(every[0]), float(every[1]))


## Look at the player when they're near and ahead; step out of their way when standing in it.
func _attend(delta: float) -> void:
    var attention: Dictionary = definition.get("attention", {})
    var who := _player()
    if _look != null:
        var look: Dictionary = attention.get("look", {})
        var want := 0.0
        var intent = _intent_point()
        # (Out of range and not just greeted: nothing to look at; skip the camera lookups.)
        # Lying down, only who just greeted them: the head turns on the pillow.
        if clock < _notice_until and clock >= _attend_until and _pose in ["stand", "sit"]:
            want = 1.0                         # something it noticed, for a moment (`reacts.look`)
            _look.target = _notice_point
        elif intent != null and clock >= _attend_until:
            want = 1.0                         # a step's `look:` what it is about; a greeter comes first
            _look.target = intent
        elif who != null and not look.is_empty() and (_pose in ["stand", "sit"] or (_pose == "lie" and clock < _attend_until)) \
                and (clock < _attend_until
                or NpcGeometry.flat(who.global_position - global_position).length() < float(look["range"]) + 1.0):
            var eye: Vector3 = who.global_position + Vector3(0, 1.6, 0)
            var cam = who.get("camera")
            if cam is Camera3D:
                eye = (cam as Camera3D).global_position
            var to := NpcGeometry.flat(eye - global_position)
            var forward := Basis(Vector3.UP, _yaw()) * Vector3.BACK
            # Just greeted: look their way wherever they stand (the head's own limits clamp it).
            if clock < _attend_until or (to.length() < float(look["range"])
                    and rad_to_deg(forward.angle_to(to)) < float(look["cone"]) / 2.0 and _glancing(look)):
                want = 1.0
                _look.target = eye
        if _wave_run != null and not _wave_run.done and (_pose != "sit" or _moving_pose()):
            _motions.stop("wave", "drop")      # they got up: the arm goes with the body's clip
            _wave_run = null
        _look.want = want
        if want > 0.0:
            stats["looking"] = float(stats.get("looking", 0.0)) + delta
    var yield_spec: Dictionary = attention.get("yield", {})
    if who == null or yield_spec.is_empty() or _pose != "stand" or _moving_pose() \
            or clock - _yielded_at < YIELD_COOLDOWN:
        return
    if NpcGeometry.flat(global_position - who.global_position).length() > float(yield_spec["distance"]) + YIELD_FAR:
        return                             # too far off to be walking at us, even sprinting
    if not _stack.is_empty() and (_stack.back()["action"] in ["greet", yield_spec["action"]]
            or _stack.back()["steps"][mini(_stack.back()["index"], _stack.back()["steps"].size() - 1)].has("go_to")):
        return                             # walking already avoids the player; reactions run out
    var body_velocity = who.get("velocity")
    if not body_velocity is Vector3:
        return
    var v := NpcGeometry.flat(body_velocity)
    var offset := NpcGeometry.flat(global_position - who.global_position)
    if v.length() < 0.3 or offset.length() > float(yield_spec["distance"]) + v.length() * 0.6 \
            or v.normalized().dot(offset.normalized()) < 0.7:
        return                             # not walking at us
    var side := signf(v.cross(offset).y)
    if side == 0.0:
        side = 1.0 if rng.randf() < 0.5 else -1.0
    var perp := Vector3(v.z, 0, -v.x).normalized() * -side
    var aside := global_position + perp * float(yield_spec["step"])
    var map := get_world_3d().navigation_map
    aside = NavigationServer3D.map_get_closest_point(map, aside)
    _yielded_at = clock
    stats["yields"] = int(stats.get("yields", 0)) + 1
    _trace({"event": "yield", "to": _v(aside)})
    _react(str(yield_spec["action"]), {"aside": aside})


## Whether the player can greet this NPC now (their E prompt): it has a greeting and is here,
## whatever it is doing (standing, walking, seated, lying down).
func can_greet() -> bool:
    return not definition.get("attention", {}).get("greet", {}).is_empty() and not away and not absent


## Seated or lying: out of its standing collider.
func posed() -> bool:
    return _pose != "stand"


## How far along an aim line (from `from`, unit `forward`) it passes this NPC's posed body, or
## INF if it misses. A seated or lying NPC has no collider for the player's aim ray to hit, so
## the E prompt asks this instead: the body is the line from its hips to its head (the root
## to head height without a skeleton), `radius` thick.
func aim_distance(from: Vector3, forward: Vector3) -> float:
    var radius := float(definition.get("radius", 0.25))
    var line := _body_line()
    var best := INF
    for i in 5:
        var p: Vector3 = line[0].lerp(line[1], i / 4.0)
        var along := (p - from).dot(forward)
        if along > 0.0 and (p - from - forward * along).length() <= radius:
            best = minf(best, along)
    return best


## Whether ``point`` is in front of the NPC (the side it faces), not behind it.
func faces(point: Vector3) -> bool:
    return (Basis(Vector3.UP, _yaw()) * Vector3.BACK).dot(NpcGeometry.flat(point - global_position)) > 0.0


## The posed body's head: what the player must have a clear line to, to greet it.
func head_point() -> Vector3:
    return _body_line()[1]


## The body as a line, hips to head (root to head height without a skeleton).
func _body_line() -> Array:
    var radius := float(definition.get("radius", 0.25))
    var a := global_position + Vector3.UP * 0.3
    var b := global_position + Vector3.UP * (float(definition.get("height", 1.75)) - radius)
    var skeletons := body.find_children("*", "Skeleton3D", true, false) if body != null else []
    if not skeletons.is_empty():
        var skeleton := skeletons[0] as Skeleton3D
        var hips := skeleton.find_bone("Hips")
        var head := skeleton.find_bone("Head")
        if hips >= 0 and head >= 0:
            a = skeleton.global_transform * skeleton.get_bone_global_pose(hips).origin
            b = skeleton.global_transform * skeleton.get_bone_global_pose(head).origin
    return [a, b]


func greet_prompt() -> String:
    var prompt := str(definition.get("attention", {}).get("greet", {}).get("prompt", "Say hello"))
    return "%s (%s)" % [prompt, person_name()] if person_name() != "" else prompt


## A named person's name (a `kind: character` placement), or "".
func person_name() -> String:
    return str(definition.get("name", ""))


## An actor (the player by default) greets this NPC. Rules hear it as the NPC's `greeted`
## signal, with `by` as the actor, in any pose and in the middle of any action: only a second
## press within `debounce` s is ignored. What the NPC does about it fits what it is doing:
## standing, it runs its greet action (turns to them and waves) and then carries on; seated, it
## stays put, looks at them for `acknowledge` s and waves from the chair; lying down, it turns
## its head to them for `acknowledge` s; a body with no skeleton (a capsule) gives a little bob;
## in the middle of sitting down it only takes the greeting. It waves once per `cooldown` s.
func greet(by := "player") -> bool:
    var spec: Dictionary = definition.get("attention", {}).get("greet", {})
    if not can_greet() or clock - _greeted_at < float(spec.get("debounce", 0.5)):
        return false
    _greeted_at = clock
    stats["greets"] = int(stats.get("greets", 0)) + 1
    _attend_until = clock + float(spec.get("acknowledge", 3.0))
    var response := "none"
    var action := str(spec["action"])
    var reacting: bool = not _stack.is_empty() and _stack.back()["action"] == action
    if clock - _waved_at >= float(spec["cooldown"]) and not _moving_pose() and not reacting:
        if _pose == "stand":
            response = "action"
            _react(action, {})
        elif _pose == "sit" and _wave_seated():
            response = "gesture"
        elif _pose == "lie" and _look != null and not definition.get("attention", {}).get("look", {}).is_empty():
            response = "look"                    # _attend turns the head while _attend_until lasts
        elif _look == null and body != null:
            response = "bob"
            var rest := body.position
            var bob := create_tween()
            bob.tween_property(body, "position", rest + Vector3.UP * BOB_HEIGHT, 0.12)
            bob.tween_property(body, "position", rest, 0.18)
    if response != "none":
        _waved_at = clock
        stats["waves"] = int(stats.get("waves", 0)) + 1
    _trace({"event": "greet", "pos": _v(global_position), "pose": _pose, "response": response})
    var signal_name := str(spec.get("signal", ""))
    if signal_name != "":
        greeted.emit(signal_name, by)
    return true


## Run a reaction on top of whatever the NPC is doing; the interrupted step starts again after.
func _react(action_name: String, extra: Dictionary) -> void:
    if not _stack.is_empty():
        _stack.back()["started"] = false
        _stack.back()["from"] = global_position
    _playing = false                       # the reaction's own steps animate the body
    _looping = false
    _stop_motions()
    velocity = Vector3.ZERO
    _path = PackedVector3Array()
    _stairs = []
    _stair = null
    var ctx := {"anchor": global_position, "yaw_deg": rad_to_deg(_yaw()), "approach": global_position}
    ctx.merge(extra)
    _push(action_name, ctx)


# --- helpers ------------------------------------------------------------------

## 0 inside the home polygon, else metres to its edge (XZ).
func home_distance(p: Vector3) -> float:
    if home.size() < 3:
        return 0.0
    var q := Vector2(p.x, p.z)
    if Geometry2D.is_point_in_polygon(q, home):
        return 0.0
    var best := INF
    for i in home.size():
        var c := Geometry2D.get_closest_point_to_segment(q, home[i], home[(i + 1) % home.size()])
        best = minf(best, c.distance_to(q))
    return best


## Current action and step, e.g. "sit (armchair): wait".
func _status() -> String:
    if _stack.is_empty():
        return "deciding"
    var frame: Dictionary = _stack.back()
    var step: Dictionary = frame["steps"][mini(frame["index"], frame["steps"].size() - 1)]
    var verb := ""
    for k in step:
        if k != "if" and k != "at":
            verb = "%s %s" % [k, step[k]]
    return "%s: %s" % [_option.get("id", frame["action"]), verb]


func _needs_text() -> String:
    var parts := []
    for need in needs:
        parts.append("%s %d%%" % [need, roundi(needs[need] * 100)])
    return "  ".join(parts)


func _rounded_needs() -> Dictionary:
    var out := {}
    for need in needs:
        out[need] = snappedf(needs[need], 0.01)
    return out


func _trace(event: Dictionary) -> void:
    event["t"] = snappedf(clock, 0.01)
    event["npc"] = String(name)
    traced.emit(event)
    if trace_enabled:
        print("npc: %s" % JSON.stringify(event))


static func _v(p: Vector3) -> Array:
    return [snappedf(p.x, 0.01), snappedf(p.y, 0.01), snappedf(p.z, 0.01)]


## Summary for tests and --simulate.
func _quests() -> Dictionary:
    var out := {}
    if world != null and world.rules != null:
        for quest in world.rules.report(WorldLoader.actor_id(self)):
            if quest["status"] in ["active", "complete", "failed"]:
                out[quest["id"]] = quest["status"]
    return out


func _rounded_relations() -> Dictionary:
    var out := {}
    for uid in relations:
        out[uid] = snappedf(float(relations[uid]), 0.001)
    return out


func _plan_report() -> Dictionary:
    if _chain.is_empty():
        return {}
    return {"goal": _chain["goal"], "steps": (_chain["steps"] as Array).map(func(o): return o["id"]),
        "index": _chain["index"], "cost": _chain["cost"], "score": _chain["score"]}


func report() -> Dictionary:
    var p := global_position
    return {"npc": String(name), "uid": WorldLoader.npc_uid(self), "actor": WorldLoader.actor_id(self), "name": person_name(),
        "character": str(definition.get("character", "")), "used": stats["used"], "distinct": stats["used"].size(),
        "decisions": stats["decisions"], "failures": stats["failures"], "max_stall": snappedf(stats["max_stall"], 0.01),
        "unstuck": int(stats.get("unstuck", 0)), "throttled": int(stats.get("throttled", 0)), "slept": int(stats.get("slept", 0)),
        "outside": snappedf(stats["outside"], 0.01), "passes": stats["passes"], "pose": _pose, "clip": animator.current() if animator != null else "",
        "hands": _hands.report() if _hands != null else {},
        "feet": _feet.report() if _feet != null else {},
        "position": _v(p), "head": _v(head_point()), "needs": _rounded_needs(), "doing": _status(), "away": away, "absent": absent,
        "aways": int(stats.get("aways", 0)), "returns": int(stats.get("returns", 0)),
        "crosswalks": int(stats.get("crosswalks", 0)), "greets": int(stats.get("greets", 0)),
        "close_ins": int(stats.get("close_ins", 0)),
        # Signalled crossings: waits on the kerb for walk (and the longest, s), steps onto one, and
        # steps onto one showing a steady don't-walk.
        "signal_waits": int(stats.get("signal_waits", 0)), "signal_wait": snappedf(float(stats.get("signal_wait", 0.0)), 0.1),
        "signal_crossings": int(stats.get("signal_crossings", 0)),
        "signal_violations": int(stats.get("signal_violations", 0)),
        "waves": int(stats.get("waves", 0)), "handles": int(stats.get("handles", 0)), "handle_gap": float(stats.get("handle_gap", -1.0)), "hand_gaps": _hand_gaps(), "take_gap": float(stats.get("take_gap", -1.0)), "presses": int(stats.get("presses", 0)), "press_gap": float(stats.get("press_gap", -1.0)),
        # Its group's leader (stable ID, "" for none) and the seconds it spent apart from them all.
        "group": group_leader(), "apart": snappedf(float(stats.get("apart", 0.0)), 0.1),
        "holding": holding(),
        # Company (`offers:`): who it is with now (stable ID, "" for nobody), how often its offers were taken
        # up and seen through, and how well it knows those it has been with ({stable ID: 0..1}).
        "with": WorldLoader.npc_uid(_partner) if is_instance_valid(_partner) else "", "joined": int(stats.get("joined", 0)),
        "relations": _rounded_relations(),
        # Told what to do (direct()): the queue (option ids, the first being done or next), how the last one
        # ended ({option, ok, reason}; {} for none yet), and how many were seen through and given up.
        "orders": _orders.duplicate(), "order": _order_last, "free_will": free_will, "orders_done": int(stats.get("orders_done", 0)),
        "orders_failed": int(stats.get("orders_failed", 0)),
        # Planning (`planner:`): the chain it is on ({goal, steps, index, cost, score}; {} for none), how many it
        # has made, finished and given up; needs that ran out (`at_zero`); its mood (the mean of its needs).
        "plan": _plan_report(), "plans": int(stats.get("plans", 0)), "plans_done": int(stats.get("plans_done", 0)),
        "replans": int(stats.get("replans", 0)), "ran_out": int(stats.get("ran_out", 0)), "mood": snappedf(mood(), 0.01),
        # The physics prop in its hands (stable ID, "" for none), and how many it has picked up and put down.
        "wants": int(stats.get("wants", 0)),
        # Its wants in detail (GeogenWants.report: seen, left, doing, handled), where a step's `look:` has the
        # head turned (subject | place | "") and how far the head is turned now (deg, yaw + pitch).
        # Perception: stimuli it has noticed, the last one ({stimulus, subject, at, by: heard | seen, clock}) and
        # the times that cut short what it was doing.
        "noticed": int(stats.get("noticed", 0)), "notice": _noticed, "cut_short": int(stats.get("cut_short", 0)),
        # Memory: the props it knows of ({uid: {at, by: seen | heard, resting, age}}) and its searches for them.
        "memory": GeogenMemory.report(self), "searches": int(stats.get("searches", 0)),
        "want": GeogenWants.report(self), "looking_at": _intent if _intent_point() != null else "",
        "head_turn": snappedf(rad_to_deg(_look.turned()), 0.1) if _look != null else 0.0,
        "prop": prop_in_hand(), "picked_up": int(stats.get("picked_up", 0)), "put_down": int(stats.get("put_down", 0)),
        # Items it has used (`use_item` steps) and what it carries now (item ids).
        "items_used": int(stats.get("items_used", 0)),
        "carrying": world.rules.carried(WorldLoader.actor_id(self)).map(func(e): return str(e["id"])) if world.rules != null else [],
        # Quests it holds ({id: status}) and the quest goals it has set off to work on.
        "quests": _quests(), "quest_goals": int(stats.get("quest_goals", 0)),
        "yields": int(stats.get("yields", 0)), "transitions": int(stats.get("transitions", 0)),
        "cost_us": snappedf(float(_cost_usec) / maxi(_cost_frames, 1), 0.1), "looking": snappedf(float(stats.get("looking", 0.0)), 0.1),
        # World heading (deg) and how far it is from the pose's (sitting: the seat's facing).
        "yaw": snappedf(rad_to_deg(wrapf(_yaw(), -PI, PI)), 0.1),
        "pose_yaw_error": snappedf(rad_to_deg(absf(wrapf(_yaw() - _pose_yaw, -PI, PI))), 0.1)}
