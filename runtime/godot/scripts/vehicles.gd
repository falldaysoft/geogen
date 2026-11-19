class_name GeogenVehicles
extends RefCounted
## What road traffic (traffic.gd) and trains (train.gd) share: their lamps, lit brighter after
## dark, and the world time their schedules and signals run on.


## Collect the emissive materials of a vehicle's lamp parts (extras.geogen.vehicle `lamps`:
## {head | tail | ...: [part names]}) into ``lamps``: material -> [day emission energy, kind].
## The materials are shared by a fleet, so each is taken once.
static func collect_lamps(node: Node, info: Dictionary, lamps: Dictionary) -> void:
    for kind in info.get("lamps", {}):
        for part_name in info["lamps"][kind]:
            var part := node.find_child(str(part_name), true, false)
            if part == null:
                continue
            for mi: MeshInstance3D in ([part] if part is MeshInstance3D else []) + part.find_children("*", "MeshInstance3D", true, false):
                for i in mi.mesh.get_surface_count() if mi.mesh else 0:
                    var mat := mi.get_active_material(i) as BaseMaterial3D
                    if mat != null and mat.emission_enabled and not lamps.has(mat):
                        lamps[mat] = [mat.emission_energy_multiplier, kind]


## Head lamps five times, the others three times as bright at night; back to the day's at day.
static func light_lamps(lamps: Dictionary, night: bool) -> void:
    for mat in lamps:
        var boost := 5.0 if lamps[mat][1] == "head" else 3.0
        (mat as BaseMaterial3D).emission_energy_multiplier = float(lamps[mat][0]) * (boost if night else 1.0)


## Seconds of world time: the clock's position in its (compressed) day, else ``since_load``.
static func world_time(world: WorldLoader, since_load: float) -> float:
    if world != null and world.clock != null and world.clock.day_length > 0.0:
        return world.clock.hours / 24.0 * world.clock.day_length
    return since_load
