class_name Player
extends CharacterBody3D
## First-person controller sized from a geogen PlayerSpec: body radius and
## height, eye height, step-up height and max walkable slope all come from the
## export manifest. The origin is at the feet.
##
## The body is a cylinder, not a capsule: a capsule's rounded bottom lands on
## a step's edge at a steep angle and slides off, so ledges of exactly
## step_height (the door step) can't be climbed. A flat bottom stands on them.

const WALK_SPEED := 4.0
const SPRINT_SPEED := 7.0
const FLY_SPEED := 10.0
const JUMP_VELOCITY := 4.5
const ACCEL := 12.0
const MOUSE_SENSITIVITY := 0.0025
## Extra lift when probing a step, so a ledge of exactly step_height isn't
## rejected by the collision margin; the actual climb is still capped.
const STEP_PROBE_MARGIN := 0.05

var spec := PlayerSpec.new()
## Noclip flying for inspecting geometry.
var flying := false
## When non-null, replaces keyboard movement input (x = strafe, y = forward).
## Used by scripted walk tests.
var scripted_move = null

var head: Node3D
var camera: Camera3D
var _shape: CollisionShape3D
var _gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity")


func _init() -> void:
	name = "Player"
	_shape = CollisionShape3D.new()
	_shape.shape = CylinderShape3D.new()
	add_child(_shape)
	head = Node3D.new()
	head.name = "Head"
	add_child(head)
	camera = Camera3D.new()
	camera.name = "Camera"
	camera.near = 0.05
	camera.fov = 75.0
	head.add_child(camera)


func _ready() -> void:
	_ensure_input_actions()
	apply_spec(spec)


func apply_spec(new_spec: PlayerSpec) -> void:
	spec = new_spec
	var body := _shape.shape as CylinderShape3D
	body.radius = spec.radius
	body.height = spec.height
	_shape.position.y = spec.height / 2.0
	head.position.y = spec.eye_height
	floor_max_angle = spec.max_slope_rad()
	floor_snap_length = spec.step_height
	floor_stop_on_slope = true
	floor_block_on_wall = true


## Place the player's feet at `pos`, looking along -Z rotated by `yaw_deg`.
func spawn(pos: Vector3, yaw_deg := 0.0) -> void:
	global_position = pos
	rotation = Vector3(0, deg_to_rad(yaw_deg), 0)
	head.rotation = Vector3.ZERO
	velocity = Vector3.ZERO


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-event.relative.x * MOUSE_SENSITIVITY)
		head.rotate_x(-event.relative.y * MOUSE_SENSITIVITY)
		head.rotation.x = clamp(head.rotation.x, deg_to_rad(-89), deg_to_rad(89))
	elif event.is_action_pressed("ui_cancel"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	elif event is InputEventMouseButton and event.pressed:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	elif event.is_action_pressed("geogen_fly"):
		flying = not flying
		_shape.disabled = flying


func _physics_process(delta: float) -> void:
	var input: Vector2 = scripted_move if scripted_move != null else Input.get_vector(
		"geogen_left", "geogen_right", "geogen_back", "geogen_forward")
	if flying:
		_fly(input, delta)
		return

	var basis_flat := Basis(Vector3.UP, rotation.y)
	var wish := basis_flat * Vector3(input.x, 0, -input.y)
	var speed := SPRINT_SPEED if Input.is_action_pressed("geogen_sprint") else WALK_SPEED
	var target := wish * speed
	velocity.x = move_toward(velocity.x, target.x, ACCEL * speed * delta)
	velocity.z = move_toward(velocity.z, target.z, ACCEL * speed * delta)

	if is_on_floor():
		if Input.is_action_just_pressed("geogen_jump"):
			velocity.y = JUMP_VELOCITY
		elif _try_step_up(Vector3(velocity.x, 0, velocity.z) * delta):
			return
	else:
		velocity.y -= _gravity * delta
	move_and_slide()


## Walk up ledges no taller than spec.step_height: if the way ahead is blocked,
## try the same move from step_height higher and drop back onto the ledge.
func _try_step_up(motion: Vector3) -> bool:
	if motion.length_squared() < 1e-8:
		return false
	var from := global_transform
	if not test_move(from, motion):
		return false
	var up := Vector3.UP * (spec.step_height + STEP_PROBE_MARGIN)
	# Rise only as far as the headroom allows (e.g. under a door head), so
	# low lips like a threshold are still climbable where a full step isn't.
	var ceiling := KinematicCollision3D.new()
	if test_move(from, up, ceiling):
		up = ceiling.get_travel() - Vector3.UP * 0.005
		if up.y < 0.01:
			return false  # no headroom
	var raised := from.translated(up)
	if test_move(raised, motion):
		return false  # a wall, not a step
	var ahead := raised.translated(motion)
	var hit := KinematicCollision3D.new()
	if not test_move(ahead, -up, hit):
		return false  # no floor to land on within the step height
	if hit.get_normal().angle_to(Vector3.UP) > floor_max_angle:
		return false
	var landed := ahead.origin + hit.get_travel()
	if landed.y - from.origin.y > spec.step_height + 0.01:
		return false
	global_position = landed
	velocity.y = 0.0
	return true


func _fly(input: Vector2, delta: float) -> void:
	var dir := camera.global_basis * Vector3(input.x, 0, -input.y)
	if Input.is_action_pressed("geogen_jump"):
		dir += Vector3.UP
	if Input.is_action_pressed("geogen_crouch"):
		dir += Vector3.DOWN
	global_position += dir * FLY_SPEED * delta
	velocity = Vector3.ZERO


static func _ensure_input_actions() -> void:
	var bindings := {
		"geogen_forward": [KEY_W, KEY_UP],
		"geogen_back": [KEY_S, KEY_DOWN],
		"geogen_left": [KEY_A, KEY_LEFT],
		"geogen_right": [KEY_D, KEY_RIGHT],
		"geogen_jump": [KEY_SPACE],
		"geogen_crouch": [KEY_CTRL, KEY_C],
		"geogen_sprint": [KEY_SHIFT],
		"geogen_fly": [KEY_F3],
		"geogen_toggle_overlay": [KEY_F1],
		"geogen_toggle_colliders": [KEY_F2],
	}
	for action in bindings:
		if InputMap.has_action(action):
			continue
		InputMap.add_action(action)
		for key in bindings[action]:
			var ev := InputEventKey.new()
			ev.physical_keycode = key
			InputMap.action_add_event(action, ev)
