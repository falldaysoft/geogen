class_name GeogenHudMarkers
extends Control
## HUD panel: world markers over what the tracked quest's open objectives are about.
##
## It has no quest logic. What is marked comes from `GeogenRules.report(actor)`: an open
## objective's `markers` (the world positions of its subjects, from the manifest, so a
## subject in a chunk that isn't loaded is marked too) and `area`. Which quest is the
## tracker's business (`GeogenHudQuestTracker.tracked_in`): the markers follow what it shows.
##
## Kinds:
## - point: an indicator over the subject with the distance to it;
## - area (`marker: {area: r}`): a ring of that radius on the ground, and an indicator with
##   the distance to its edge while the player is outside it. The ring isn't centred on the
##   subject (`AREA_SHIFT` of its radius off, in a direction fixed by the subject's id), so
##   it says "somewhere in here" and not where;
## - portal: a quest of another package (a scene visited earlier this session, whose state
##   the rules keep), or an objective whose subjects are in another scene, can't be marked
##   here, so the travel points of the loaded scene that lead there are marked instead:
##   those that go straight there, else the first steps of the shortest routes over the
##   exported scenes' travel points (`route` lists the scenes on the way). One quest per
##   package: the one tracked there, else its first active one.
## Subjects of a counted objective that are done already (`subjects_done`) aren't marked.
## `marker: hidden` objectives have no markers (the report leaves them out).
##
## An indicator that is off screen, or behind the camera, is clamped to the edge of the
## screen (`edge_margin()` in) with an arrow pointing the way to turn. Markers fade with
## distance (`FADE_FROM` to `FADE_TO` m, down to `FADE_MIN`), and one hidden behind
## something (a ray from the camera, a few times a second) is drawn at `OCCLUDED` strength.
##
## The list is rebuilt on quest events, state changes and the HUD's refreshes; only the
## projection to the screen runs per frame, and only while there is something to mark.

const AREA_SHIFT := 0.45
const RING_POINTS := 48
const ICON := 11.0
const TEXT_SIZE := 15
const COLORS := {
    "point": Color(1.0, 0.82, 0.4),             # the tracker's gold
    "area": Color(1.0, 0.82, 0.4),
    "portal": Color(1.0, 0.82, 0.4),            # told apart by its arch
}
const TEXT := Color(1.0, 0.96, 0.84)
const FADE_FROM := 40.0
const FADE_TO := 160.0
const FADE_MIN := 0.45
const OCCLUDED := 0.5
const OCCLUSION_EVERY := 0.2           # seconds between occlusion rays
const SHADE := Color(0, 0, 0, 0.75)

## False hides every marker (the list is still kept and reported).
var enabled := true:
    set(value):
        enabled = value
        _show()

var _hud: GeogenHud
var _markers: Array[Dictionary] = []    # {quest, package, objective, subject, kind, at: Vector3, ...}
var _placed: Array[Dictionary] = []     # the same, with where each is on screen now
var _travel_points := -1                # how many travel points there were when the list was built
var _since_check := 0.0
var _sample := false                    # ring heights are due (taken in a physics frame)
var _occluded := {}                     # marker index -> hidden behind something, from the last rays
var _since_rays := 0.0
var _routes := {}                       # scene -> [scenes its travel points go to], from the manifests
var _routes_stamp := ""


func _init() -> void:
    name = "markers"
    mouse_filter = Control.MOUSE_FILTER_IGNORE
    set_process(false)
    set_physics_process(false)


func hud_attach(hud: GeogenHud) -> void:
    _hud = hud
    hud.world.rules.quest_event.connect(func(_event: Dictionary): _hud.queue_refresh())
    hud.world.events.listen(func(_event: Dictionary): _hud.queue_refresh(), ["changed"])
    hud.modal_changed.connect(func(_open: bool): _show())


func _show() -> void:
    visible = enabled and (_hud == null or _hud.modal == null)


# -- what is marked ------------------------------------------------------------------------------

func hud_refresh() -> void:
    var world: WorldLoader = _hud.world
    var rules: GeogenRules = world.rules
    var tracker: GeogenHudQuestTracker = _hud.panel("quests")
    _markers.clear()
    var offset := _model_offset()
    var tracked := tracker.tracked_here() if tracker != null else ""
    for quest in rules.report(_hud.actor):
        if quest["id"] != tracked or quest["status"] != "active":
            continue
        for objective in quest["objectives"]:
            var points: Array = objective.get("markers", [])
            if points.is_empty():
                # Open, but somewhere else: a content package's next step in another scene.
                if objective["state"] == "open" and not world.scene_name in objective.get("scenes", [world.scene_name]):
                    _portals(quest, quest["package"], objective)
                continue
            var base := {"quest": quest["id"], "package": quest["package"], "objective": objective["id"],
                "title": quest["title"], "text": objective["text"]}
            if objective.has("area"):
                # One ring for the objective: round the middle of its subjects, shifted off it.
                var middle := Vector3.ZERO
                for point in points:
                    middle += _vec(point["position"]) / points.size()
                var radius := float(objective["area"])
                var turn := float(hash(str(points[0]["subject"])) % 3600) / 3600.0 * TAU
                var entry := base.duplicate()
                entry.merge({"kind": "area", "subject": points[0]["subject"], "radius": radius, "ring": [],
                    "at": middle + offset + Vector3(cos(turn), 0.0, sin(turn)) * radius * AREA_SHIFT})
                _markers.append(entry)
            else:
                for point in points:
                    var entry := base.duplicate()
                    entry.merge({"kind": "point", "subject": point["subject"], "at": _vec(point["position"]) + offset})
                    _markers.append(entry)
    _add_portals(rules, tracker)
    _travel_points = world.travel_points.size()
    _sample = _markers.any(func(m): return m["kind"] == "area")
    set_physics_process(_sample)
    set_process(true)           # one more pass even when empty: it clears what was drawn
    _show()
    _place_all()
    queue_redraw()


## Quests of the other packages this session has loaded: mark the way to their scene.
func _add_portals(rules: GeogenRules, tracker: GeogenHudQuestTracker) -> void:
    var world: WorldLoader = _hud.world
    var by_package := {}            # package -> its active quests, in order
    for quest in rules.report_all(_hud.actor):
        if not quest["loaded"] and quest["status"] == "active":
            if not by_package.has(quest["package"]):
                by_package[quest["package"]] = []
            by_package[quest["package"]].append(quest)
    for package in by_package:
        var active: Array = by_package[package]
        var quest: Dictionary = active[0]
        var pick := tracker.tracked_in(package) if tracker != null else ""
        for other in active:
            if other["id"] == pick:
                quest = other
        var objective := {}
        for candidate in quest["objectives"]:
            var somewhere: bool = candidate.get("marked", false) or not candidate.get("scenes", []).is_empty()
            if candidate["state"] == "open" and somewhere and objective.is_empty():
                objective = candidate
        if not objective.is_empty():
            _portals(quest, package, objective)


## Mark the travel points that go straight to a scene an objective's subjects are in (a
## content package's quest names things in scenes it isn't named after; else its package's).
func _portals(quest: Dictionary, package: String, objective: Dictionary) -> void:
    var world: WorldLoader = _hud.world
    var scenes: Array = objective.get("scenes", [])
    if scenes.is_empty():
        scenes = [package]
    var route := {}                 # first step -> the scenes on the way, when none goes straight there
    if not world.travel_points.any(func(p): return str(p.get("travel", {}).get("scene", "")) in scenes):
        route = _first_steps(world.scene_name, scenes)
    for point in world.travel_points:
        var node = point.get("node")
        var travel: Dictionary = point.get("travel", {})
        var scene := str(travel.get("scene", ""))
        if not (scene in scenes or route.has(scene)) or scene == world.scene_name \
                or not is_instance_valid(node) or not node is Node3D:
            continue
        var uid := WorldLoader.uid_of(node)
        if _markers.any(func(m): return m["kind"] == "portal" and m["subject"] == uid):
            continue                # one marker a portal: the tracked quest's, which is marked first
        var box := WorldLoader._aabb(node)
        _markers.append({"quest": quest["id"], "package": package, "objective": objective["id"],
            "title": quest["title"], "text": objective["text"], "kind": "portal", "scene": scene,
            "subject": uid, "label": scene.capitalize(), "route": route.get(scene, [scene]),
            "at": box.get_center() if box.size != Vector3.ZERO else (node as Node3D).global_position})


## The first steps of the shortest routes from ``here`` to any of ``targets`` over the
## exported scenes' travel points: {first scene: [scenes on the way, ending at a target]}.
func _first_steps(here: String, targets: Array) -> Dictionary:
    var graph := _travel_graph()
    var best := {}
    var seen := {here: true}
    var frontier := []
    for first in graph.get(here, []):
        if not seen.has(first):
            seen[first] = true
            frontier.append([first])
    while not frontier.is_empty() and best.is_empty():
        var next := []
        for path in frontier:
            var last: String = path[-1]
            if last in targets:
                if not best.has(path[0]):
                    best[path[0]] = path
                continue
            for onward in graph.get(last, []):
                if not seen.has(onward):
                    seen[onward] = true
                    next.append(path + [onward])
        frontier = next
    return best


## scene -> the scenes its travel points go to, from every manifest in the exports
## directory, read again when the catalogue index is rewritten.
func _travel_graph() -> Dictionary:
    var world: WorldLoader = _hud.world
    var index := "%s/catalogue.json" % world.generated_dir
    var stamp := "%s@%d" % [world.generated_dir, FileAccess.get_modified_time(index) if FileAccess.file_exists(index) else 0]
    if stamp == _routes_stamp:
        return _routes
    _routes_stamp = stamp
    _routes = {}
    var dir := DirAccess.open(world.generated_dir)
    if dir == null:
        return _routes
    var files := Array(dir.get_files())
    for sub in dir.get_directories():
        if sub.ends_with("_chunks"):
            files.append("%s/%s.chunks.json" % [sub, sub.trim_suffix("_chunks")])
    for file in files:
        if not (file.ends_with(".manifest.json") or file.ends_with(".chunks.json")):
            continue
        var path := "%s/%s" % [world.generated_dir, file]
        if not FileAccess.file_exists(path):
            continue
        var data = JSON.parse_string(FileAccess.get_file_as_string(path))
        if not data is Dictionary:
            continue
        var scene: String = str(file).get_file().trim_suffix(".manifest.json").trim_suffix(".chunks.json")
        var goes: Array = _routes.get(scene, [])
        for travel in data.get("travel", []):
            var there := str(travel.get("scene", ""))
            if there != "" and there != scene and not there in goes:
                goes.append(there)
        _routes[scene] = goes
    return _routes


## Where the loaded model was put: the origin, unless several exports are laid out in a row.
func _model_offset() -> Vector3:
    var world: WorldLoader = _hud.world
    if world.scene_name != "":
        return Vector3.ZERO
    for child in world.get_children():
        if child is Node3D and String(child.name) == world.rules.package:
            return (child as Node3D).position
    return Vector3.ZERO


static func _vec(values: Array) -> Vector3:
    return Vector3(float(values[0]), float(values[1]), float(values[2]))


# -- where it is on screen -----------------------------------------------------------------------

## How far in from the screen's edge a clamped indicator sits.
func edge_margin() -> float:
    var view := get_viewport_rect().size
    return clampf(minf(view.x, view.y) * 0.07, 4.0, 44.0)


func _process(delta: float) -> void:
    # Streamed chunks bring their travel points with them: look again when the count moves.
    _since_check += delta
    if _since_check > 1.0:
        _since_check = 0.0
        if _hud.world.travel_points.size() != _travel_points:
            _hud.queue_refresh()
    _since_rays += delta
    if _since_rays >= OCCLUSION_EVERY:
        _since_rays = 0.0
        set_physics_process(true)           # the rays are cast in a physics frame
    _place_all()
    queue_redraw()
    if _markers.is_empty():
        set_process(false)


## The ground under each ring, from the colliders (in a physics frame: it casts rays).
func _physics_process(_delta: float) -> void:
    set_physics_process(false)
    if not is_inside_tree():
        return
    _occlusion()
    if not _sample:
        return
    _sample = false
    var space := _hud.world.get_world_3d().direct_space_state
    for marker in _markers:
        if marker["kind"] != "area":
            continue
        var centre: Vector3 = marker["at"]
        var radius: float = marker["radius"]
        var reach := maxf(3.0, radius * 0.6)
        var ring := PackedVector3Array()
        for i in RING_POINTS:
            var turn := TAU * i / RING_POINTS
            var p := centre + Vector3(cos(turn), 0.0, sin(turn)) * radius
            var hit := space.intersect_ray(PhysicsRayQueryParameters3D.create(
                p + Vector3.UP * reach, p + Vector3.DOWN * (reach + 30.0)))
            ring.append(Vector3(p.x, hit["position"].y + 0.05, p.z) if not hit.is_empty() else p)
        marker["ring"] = ring


## Which point and portal markers something stands in front of, from the camera.
func _occlusion() -> void:
    var camera := get_viewport().get_camera_3d()
    _occluded.clear()
    if camera == null:
        return
    var space := _hud.world.get_world_3d().direct_space_state
    var eye := camera.global_position
    var exclude: Array[RID] = []
    var player = get_tree().root.find_child("Player", true, false)
    if player is CollisionObject3D:
        exclude.append((player as CollisionObject3D).get_rid())
    for i in _markers.size():
        var marker: Dictionary = _markers[i]
        if marker["kind"] == "area":
            continue
        var at: Vector3 = marker["at"]
        var query := PhysicsRayQueryParameters3D.create(eye, at)
        query.exclude = exclude
        var hit := space.intersect_ray(query)
        # Hitting the subject itself, or something within a step of it, isn't being hidden.
        _occluded[i] = not hit.is_empty() and eye.distance_to(hit["position"]) < eye.distance_to(at) - 1.0


func _place_all() -> void:
    _placed.clear()
    var camera := get_viewport().get_camera_3d() if is_inside_tree() else null
    if camera == null:
        return
    var view := get_viewport_rect().size
    for i in _markers.size():
        var placed := _place(_markers[i], camera, view)
        var fade := clampf(1.0 - (placed["distance"] - FADE_FROM) / (FADE_TO - FADE_FROM), FADE_MIN, 1.0)
        placed["occluded"] = bool(_occluded.get(i, false))
        placed["alpha"] = fade * (OCCLUDED if placed["occluded"] else 1.0)
        _placed.append(placed)


## One marker's place on screen: {screen, distance, on_screen, clamped, behind, inside, turn}.
## `turn` is the way a clamped one points (radians, screen axes).
func _place(marker: Dictionary, camera: Camera3D, view: Vector2) -> Dictionary:
    var out := marker.duplicate()
    var at: Vector3 = marker["at"]
    var eye := camera.global_position
    var local := camera.global_transform.affine_inverse() * at
    var behind := local.z >= -camera.near
    var distance := eye.distance_to(at)
    var inside := false
    if marker["kind"] == "area":
        var flat := Vector2(at.x - eye.x, at.z - eye.z).length()
        inside = flat <= marker["radius"]
        distance = maxf(flat - marker["radius"], 0.0)
    var margin := edge_margin()
    var field := Rect2(Vector2(margin, margin), view - Vector2(margin, margin) * 2.0)
    var centre := view / 2.0
    var screen := centre
    if not behind:
        screen = camera.unproject_position(at)
    var on_screen := not behind and Rect2(Vector2.ZERO, view).has_point(screen)
    var clamped := behind or not field.has_point(screen)
    var turn := 0.0
    if clamped:
        # Which way to turn: where it is on screen from the middle; behind, its side of the view
        # (straight behind reads as "turn round": the bottom edge).
        var way := screen - centre
        if behind:
            way = Vector2(local.x, -local.y)
            way = Vector2(way.x, absf(way.y) + 0.35 * maxf(absf(local.z), 0.001))
        if way.length() < 0.001:
            way = Vector2.DOWN
        way = way.normalized()
        var half := field.size / 2.0
        var span := INF
        if absf(way.x) > 0.0001:
            span = minf(span, half.x / absf(way.x))
        if absf(way.y) > 0.0001:
            span = minf(span, half.y / absf(way.y))
        screen = centre + way * span
        turn = way.angle()
    out.merge({"screen": screen, "distance": distance, "on_screen": on_screen, "clamped": clamped,
        "behind": behind, "inside": inside, "turn": turn})
    return out


# -- drawing -------------------------------------------------------------------------------------

func _draw() -> void:
    var camera := get_viewport().get_camera_3d() if is_inside_tree() else null
    for marker in _placed:
        var color := Color(COLORS[marker["kind"]], float(marker.get("alpha", 1.0)))
        if marker["kind"] == "area":
            if camera != null:
                _draw_ring(marker, camera, color)
            if marker["inside"]:
                continue                    # in the ring there is nowhere more exact to point
        var at: Vector2 = marker["screen"]
        var label := _distance_text(marker["distance"])
        if marker["kind"] == "portal":
            label = "%s   %s" % [marker["label"], label]
        if marker["clamped"]:
            var way := Vector2.from_angle(marker["turn"])
            _draw_arrow(at, way, color)
            _draw_icon(at - way * (ICON + 22.0), marker["kind"], color)
            _draw_label(at - way * (ICON + 22.0), label, way, color.a)
        else:
            _draw_icon(at, marker["kind"], color)
            _draw_label(at, label, Vector2.UP, color.a)


## The ring on the ground, through whatever is in front of it, where it is ahead of the camera.
func _draw_ring(marker: Dictionary, camera: Camera3D, color: Color) -> void:
    var ring: PackedVector3Array = marker["ring"] if marker["ring"] is PackedVector3Array else PackedVector3Array()
    if ring.is_empty():             # not sampled yet: level with its middle
        for i in RING_POINTS:
            var turn := TAU * i / RING_POINTS
            ring.append(marker["at"] + Vector3(cos(turn), 0.0, sin(turn)) * marker["radius"])
    var inverse := camera.global_transform.affine_inverse()
    var lines := PackedVector2Array()
    for i in ring.size():
        var a := ring[i]
        var b := ring[(i + 1) % ring.size()]
        if (inverse * a).z > -camera.near or (inverse * b).z > -camera.near:
            continue
        lines.append(camera.unproject_position(a))
        lines.append(camera.unproject_position(b))
    if lines.size() >= 2:
        draw_multiline(lines, SHADE, 5.0, true)
        draw_multiline(lines, Color(color, 0.95), 2.5, true)


func _draw_icon(at: Vector2, kind: String, color: Color) -> void:
    match kind:
        "area":
            draw_arc(at, ICON, 0.0, TAU, 32, SHADE, 5.0, true)
            draw_arc(at, ICON, 0.0, TAU, 32, color, 2.5, true)
            draw_circle(at, 2.0, color)
        "portal":
            # an arch: a doorway to go through
            var arch := PackedVector2Array()
            arch.append(at + Vector2(-ICON * 0.75, ICON))
            for i in 13:
                var turn := PI + PI * i / 12.0
                arch.append(at + Vector2(cos(turn) * ICON * 0.75, sin(turn) * ICON * 0.75 - ICON * 0.2))
            arch.append(at + Vector2(ICON * 0.75, ICON))
            draw_polyline(arch, SHADE, 7.0, true)
            draw_polyline(arch, color, 3.5, true)
        _:
            var diamond := PackedVector2Array([at + Vector2(0, -ICON), at + Vector2(ICON * 0.8, 0),
                at + Vector2(0, ICON), at + Vector2(-ICON * 0.8, 0)])
            var outline := diamond.duplicate()
            outline.append(diamond[0])
            draw_colored_polygon(diamond, color)
            draw_polyline(outline, SHADE, 2.0, true)
            draw_circle(at, 2.2, SHADE)


func _draw_arrow(at: Vector2, way: Vector2, color: Color) -> void:
    var side := way.orthogonal()
    var tip := at + way * 8.0
    var points := PackedVector2Array([tip, tip - way * 20.0 + side * 9.0, tip - way * 14.0, tip - way * 20.0 - side * 9.0])
    var outline := points.duplicate()
    outline.append(points[0])
    draw_colored_polygon(points, color)
    draw_polyline(outline, SHADE, 2.0, true)


## The text beside an indicator: under one on screen, on the screen's side of a clamped one.
func _draw_label(at: Vector2, text: String, way: Vector2, alpha := 1.0) -> void:
    var font := ThemeDB.fallback_font
    var extent := font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, TEXT_SIZE)
    var view := get_viewport_rect().size
    var middle := at + Vector2(0, ICON + 5.0 + extent.y / 2.0)
    if way != Vector2.UP:
        middle = at - Vector2(way.x * (ICON + 8.0 + extent.x / 2.0), way.y * (ICON + 5.0 + extent.y / 2.0))
    var origin := middle + Vector2(-extent.x / 2.0, extent.y / 2.0 - font.get_descent(TEXT_SIZE))
    origin.x = clampf(origin.x, 2.0, maxf(view.x - extent.x - 2.0, 2.0))
    origin.y = clampf(origin.y, extent.y, maxf(view.y - 4.0, extent.y))
    draw_string_outline(font, origin, text, HORIZONTAL_ALIGNMENT_LEFT, -1, TEXT_SIZE, 6, Color(SHADE, SHADE.a * alpha))
    draw_string(font, origin, text, HORIZONTAL_ALIGNMENT_LEFT, -1, TEXT_SIZE, Color(TEXT, alpha))


static func _distance_text(metres: float) -> String:
    if metres >= 1000.0:
        return "%.1f km" % (metres / 1000.0)
    return "%d m" % int(round(metres))


## What is displayed: {visible, tracked, package, viewport: [w, h], edge_margin, markers: [{quest,
## package, objective, subject, kind: point | area | portal, position: [x, y, z], screen: [x, y],
## distance, distance_text, on_screen, clamped, behind, turn_deg?, radius?, inside?, scene?, label?}]}.
## `position` is what is pointed at (a ring's middle, not its subject); `distance` is to it (to a
## ring's edge); `screen` is where the indicator is drawn.
func hud_report() -> Dictionary:
    _place_all()
    var view := get_viewport_rect().size if is_inside_tree() else Vector2.ZERO
    var tracker: GeogenHudQuestTracker = _hud.panel("quests") if _hud != null else null
    var package: String = _hud.world.rules.package if _hud != null and _hud.world.rules.machine != null else ""
    var out := {"visible": visible and not _placed.is_empty(), "package": package,
        "tracked": tracker.tracked_in(package) if tracker != null and package != "" else "",
        "viewport": [view.x, view.y], "edge_margin": edge_margin() if is_inside_tree() else 0.0, "markers": []}
    for marker in _placed:
        var at: Vector3 = marker["at"]
        var entry := {"quest": marker["quest"], "package": marker["package"], "objective": marker["objective"],
            "subject": marker["subject"], "kind": marker["kind"], "position": [at.x, at.y, at.z],
            "screen": [marker["screen"].x, marker["screen"].y], "distance": marker["distance"],
            "distance_text": _distance_text(marker["distance"]), "on_screen": marker["on_screen"],
            "clamped": marker["clamped"], "behind": marker["behind"], "alpha": marker["alpha"],
            "occluded": marker["occluded"]}
        if marker["clamped"]:
            entry["turn_deg"] = rad_to_deg(marker["turn"])
        if marker["kind"] == "area":
            entry["radius"] = marker["radius"]
            entry["inside"] = marker["inside"]
        if marker["kind"] == "portal":
            entry["scene"] = marker["scene"]
            entry["label"] = marker["label"]
            entry["route"] = marker["route"]
        out["markers"].append(entry)
    return out
