class_name GeogenWorldLights
extends RefCounted
## Light fixtures (extras.geogen.light; WorldLoader.lights): a tuned light under each, switchable
## fixtures shown and hidden by their `lit` state (state.light_state: a switch's binding sets it, and
## any rule may), `auto: night` lamps that follow the clock, and the glow of lamp shades.

var world: WorldLoader
## Fixture lights by their fixture node's name (light switches refer to them).
var by_name := {}
var _fixtures := {}                  # stable ID -> {light, fixture}: the switchable ones
var _auto: Array[Dictionary] = []    # [{light, fixture}] for `auto: night` fixtures


func clear() -> void:
    by_name.clear()
    _fixtures.clear()
    _auto.clear()


## Forget the fixtures under what ``inside`` says belongs to a freed chunk.
func forget(inside: Callable) -> void:
    _auto = _auto.filter(func(a): return not inside.call(a.get("fixture")))
    for key in by_name.keys():
        if inside.call(by_name[key]):
            by_name.erase(key)
    for key in _fixtures.keys():
        if inside.call(_fixtures[key]["fixture"]):
            _fixtures.erase(key)


## `auto: night` lights and their glowing glass on or off.
func set_night(on: bool) -> void:
    for a in _auto:
        if is_instance_valid(a["light"]):
            a["light"].visible = on
            set_emission(a["fixture"], on)


## Ceiling fixtures etc. carry extras.geogen.light: add a light just below them.
func add(root: Node) -> void:
    for node in root.find_children("*", "Node3D", true, false):
        var spec = WorldLoader.geogen_extras(node).get("light")
        if not spec is Dictionary:
            continue
        # Exports carry a KHR_lights_punctual light (imported as a Light3D child);
        # tune it with our energy/range/shadows. Older exports get one added.
        var light: OmniLight3D = null
        for child in node.get_children():
            if child is OmniLight3D:
                light = child
            elif WorldLoader.geogen_extras(child).get("type") == "light":
                for grand in child.get_children():
                    if grand is OmniLight3D:
                        light = grand
                if light == null and child is OmniLight3D:
                    light = child
        if light == null:
            light = OmniLight3D.new()
            var offset = spec.get("offset", [0, -0.5, 0])
            light.position = Vector3(offset[0], offset[1], offset[2]) if offset is Array else Vector3(0, offset, 0)
            node.add_child(light)
        light.name = "Light"
        var c: Array = spec.get("color", [1, 1, 1])
        light.light_color = Color(c[0], c[1], c[2])
        light.light_energy = float(spec.get("energy", 1.0))
        light.omni_range = float(spec.get("range", 5.0))
        light.omni_attenuation = 1.0
        light.shadow_enabled = true
        light.add_to_group("geogen_light")
        if spec.get("auto") == "night":
            # Street lamps: many of them, so no shadows; on when the clock says it's dark.
            light.shadow_enabled = false
            _auto.append({"light": light, "fixture": node})
            light.visible = world.night
            set_emission(node, world.night)
        else:
            var uid := WorldLoader.uid_of(node)
            if uid != "":
                _fixtures[uid] = {"light": light, "fixture": node}
        by_name[String(node.name)] = light
        # The fixture itself mustn't shadow its own light.
        for mi: MeshInstance3D in node.find_children("*", "MeshInstance3D", true, false):
            mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF


## A fixture's `lit` state (state.light_state) shows or hides its light and glow: a switch's binding
## sets it, and so may any rule. Synced when a package (or a save) loads and when a chunk adds fixtures.
func sync_fixtures() -> void:
    for uid in _fixtures:
        var lit = world.rules.get_value(uid + "#lit")
        if lit is bool:
            _light_fixture(uid, lit)


func on_lit(event: Dictionary) -> void:
    if event["type"] == "changed" and event["name"] == "lit" and _fixtures.has(event["subject"]):
        _light_fixture(event["subject"], bool(event["data"].get("to", true)))


func _light_fixture(uid: String, lit: bool) -> void:
    var entry: Dictionary = _fixtures[uid]
    if not is_instance_valid(entry["fixture"]):
        return
    entry["light"].visible = lit
    set_emission(entry["fixture"], lit)


## Dim or restore a fixture's glowing materials (lamp shades) with its light.
static func set_emission(fixture: Node, on: bool) -> void:
    for mi: MeshInstance3D in fixture.find_children("*", "MeshInstance3D", true, false):
        for i in mi.mesh.get_surface_count() if mi.mesh else 0:
            var mat := mi.get_active_material(i) as BaseMaterial3D
            if mat == null or not (mat.emission_enabled or mi.has_meta("geogen_dimmed")):
                continue
            if mi.get_surface_override_material(i) == null:
                mat = mat.duplicate()
                mi.set_surface_override_material(i, mat)
            mat.emission_enabled = on
            mi.set_meta("geogen_dimmed", not on)
