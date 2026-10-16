"""The YAML vocabulary: every key a part, asset, scene or placement accepts, with its meaning.

This is the single source for both ``validation.py`` (unknown-key warnings) and
``geogen.describe`` (``--describe``, docs/catalogue). Add a key here when a loader starts
reading it.
"""

from typing import Any, NamedTuple


class Key(NamedTuple):
    """One YAML key: its value type, what it does, and its default (None: see the generator)."""

    type: str
    doc: str
    default: Any = None


class Primitive(NamedTuple):
    doc: str
    keys: dict[str, Key]
    size: str = "required"        # required | optional (taken from the geometry)
    example: str = ""             # a part definition (YAML, inside a 1 m unit container)


# Keys every part takes, whatever its primitive.
PART_KEYS: dict[str, Key] = {
    "primitive": Key("name", "which generator builds the part"),
    "size": Key("[x, y, z]", "extent as fractions of the asset's `size` (metres when the asset size is [1, 1, 1])"),
    "anchor": Key("anchor", "where in the asset box the part sits: center, bottom_center, top_front_left, ... "
                            "(bottom anchors rest the part's base there)", "center"),
    "offset": Key("[x, y, z]", "shift from the anchor in fractions of the asset size; metres after `attach_to`",
                  [0, 0, 0]),
    "attach_to": Key("part", "parent part to connect to (instead of anchor)"),
    "at": Key("point", "the parent's attachment point: top, bottom, left, right, front, back", "top"),
    "from": Key("point", "the point on this part that meets the parent's", "bottom"),
    "rotation": Key("[x, y, z] deg", "XYZ Euler rotation"),
    "material": Key("material", "a material from assets/materials/ (see Materials)"),
    "tint": Key("colour", "vertex colour over the material: a palette name or [r, g, b]"),
    "when": Key("expr", "drop the part, and parts attached to it, when it resolves to 0 / false (`\"{paint == 'taxi' and seats >= 4}\"`)"),
    "array": Key("{count, step: [x, y, z]}", "repeat the part as <part>_0..n-1, each `step` further along"),
    "subtract": Key("[part, ...]", "cut these parts out of this one (CSG difference)"),
    "cutter": Key("bool", "helper part: only used by another part's `subtract`, then removed", False),
    "cut_host": Key("bool", "opening cutter applied to the surface the asset is placed on (doors, windows)", False),
    "reveal": Key("bool | {bottom: bool}", "the box behind an opening's frame that the host wall lines", False),
    "subdivide": Key("int | {levels, crease}", "Loop subdivision, stretched back to the part box"),
    "displace": Key("{amplitude, scale, octaves, seed, ridged}", "fractal noise along the normals"),
    "tags": Key("[tag, ...]", "dotted gameplay tags exported in extras.geogen"),
    "collider": Key("auto | none | box | hull | mesh", "collision shape exported for the part", "auto"),
    "walkable": Key("bool", "the part is floor the player and NPCs walk on", False),
    "joint": Key("{type: hinge | slide, pivot, limits | range, with}",
                 "shorthand for a closed/open interaction named after the part"),
}

_BEVEL_SEGMENTS = Key("int", "arc steps of the bevel")
_CREASE = Key("deg", "edges sharper than this stay hard", 40.0)
_FIT = Key("stretch | none", "scale the profile to the part size, or take it in metres", "stretch")
_SEED = Key("int", "random seed", 0)

PRIMITIVES: dict[str, Primitive] = {
    "cube": Primitive(
        "Rounded box filling the part size. Surfaces: front, back, left, right, top, bottom.",
        {"bevel": Key("m", "edge radius; 0 for sharp boxes (do this for thin slabs)", 0.02),
         "bevel_segments": Key("int", "arc steps per 45 degrees (1 for parts under 12 cm)", 2)},
        example="{primitive: cube, size: [0.8, 0.5, 0.6], bevel: 0.04, material: wood}"),
    "cylinder": Primitive(
        "Upright cylinder; radius is half the smaller of x and z.",
        {"bevel": Key("m", "rounded rim radius (sharp without it)", 0.0),
         "bevel_segments": Key("int", "arc steps of the rim", 4)},
        example="{primitive: cylinder, size: [0.6, 0.8, 0.6], bevel: 0.03, material: metal}"),
    "sphere": Primitive("Sphere of diameter min(size).", {},
                        example="{primitive: sphere, size: [0.8, 0.8, 0.8], material: red_metal}"),
    "ellipsoid": Primitive("Sphere stretched to all three size components.", {},
                           example="{primitive: ellipsoid, size: [1, 0.5, 0.7], material: fabric_linen}"),
    "cone": Primitive("Upright cone; base radius is half the smaller of x and z.", {},
                      example="{primitive: cone, size: [0.7, 0.9, 0.7], material: roof_shingle}"),
    "capsule": Primitive("Upright cylinder with hemispherical ends; radius min(x, z) / 2.", {},
                         example="{primitive: capsule, size: [0.4, 1, 0.4], material: paint_sage}"),
    "torus": Primitive("Ring lying in XZ, filling x and z.",
                       {"tube": Key("m", "tube radius (default: size y / 2)", 0.0)},
                       example="{primitive: torus, size: [0.9, 0.2, 0.9], material: chrome}"),
    "plane": Primitive("Flat sheet in XZ (y ignored). Surface: top.",
                       {"subdivisions_x": Key("int", "cells along x", 1),
                        "subdivisions_z": Key("int", "cells along z", 1)},
                       example="{primitive: plane, size: [1, 0, 1], material: grass}"),
    "extrude": Primitive(
        "Extrude a 2D shape (with holes) along an axis; watertight, optional rounded cap edges.",
        {"shape": Key("shape", "{rect: [w, h], radius} | {circle: r} | {polygon: [[x, y], ...]} | "
                               "{outer: ..., holes: [...]} | [[x, y], ...]"),
         "axis": Key("x | y | z", "extrusion axis: y for slabs (profile in XZ), z for panels (XY)", "y"),
         "fit": _FIT,
         "bevel": Key("m", "rounded cap edge radius", 0.0),
         "bevel_segments": Key("int", "arc steps of the cap bevel", 3),
         "crease_angle": _CREASE,
         "caps": Key("bool", "close both ends", True)},
        example="{primitive: extrude, size: [1, 0.08, 0.7], shape: {rect: [1, 0.7], radius: 0.15}, "
                "bevel: 0.01, material: wood}"),
    "lathe": Primitive(
        "Revolve an (r, y) profile around Y: turned legs, vases, bowls.",
        {"profile": Key("polyline", "[[r, y], ...] | {spline: [[r, y], ...]} | {segments: [...]}; r = 0 closes at "
                                    "the axis, repeating the first point makes a ring"),
         "fit": _FIT,
         "segments": Key("int", "radial steps (capped at one per 2.5 cm of circumference)", 48),
         "sweep": Key("deg", "partial revolution", 360.0),
         "cap_bottom": Key("bool", "flat disc on an open bottom end", True),
         "cap_top": Key("bool", "flat disc on an open top end", True),
         "crease_angle": _CREASE},
        example="{primitive: lathe, size: [0.5, 1, 0.5], "
                "profile: {spline: [[0.6, 0], [0.3, 0.3], [0.8, 0.7], [0.4, 1]]}, material: marble}"),
    "sweep": Primitive(
        "Sweep a profile shape along a 3D path (rails, pipes, mouldings), with mitred corners.",
        {"profile": Key("shape", "cross-section in metres (x = right of travel, y = up)"),
         "path": Key("[[x, y, z], ...] | {spline: [...], samples}", "the centre line"),
         "closed": Key("bool", "join the end back to the start", False),
         "up": Key("[x, y, z]", "the profile's up direction", [0, 1, 0]),
         "twist": Key("deg", "twist over the whole path", 0.0),
         "level": Key("bool", "keep the profile level across (no roll round a helix: troughs, banked tracks)", False),
         "scale": Key("s | [start, end]", "profile scale along the path", [1, 1]),
         "crease_angle": _CREASE,
         "center": Key("bool", "false keeps the path in the asset frame (anchor ignored)", True)},
        size="optional",
        example="{primitive: sweep, profile: {circle: 0.03}, "
                "path: {spline: [[-0.5, 0, 0], [-0.2, 0.5, 0], [0.2, 0.5, 0.3], [0.5, 0, 0.3]]}, material: metal}"),
    "text": Primitive(
        "Extruded lettering reading along +X and facing +Z, centred on its bounds.",
        {"text": Key("string", "the text; newlines split lines"),
         "font": Key("sans | serif", "bundled font", "sans"),
         "height": Key("m", "cap height", 0.2),
         "depth": Key("m", "extrusion depth", 0.02),
         "align": Key("left | center | right", "line alignment", "center"),
         "line_spacing": Key("factor", "line pitch", 1.0),
         "tracking": Key("em", "extra letter spacing", 0.0),
         "bevel": Key("m", "rounded face edges", 0.0)},
        size="optional",
        example="{primitive: text, text: Geogen, height: 0.25, depth: 0.05, material: metal}"),
    "roof": Primitive(
        "Pitched or flat roof over a wall-top footprint: size = footprint + rise, eaves overhang beyond it.",
        {"style": Key("gable | hip | shed | flat", "roof form", "gable"),
         "overhang": Key("m", "eaves beyond the footprint", 0.35),
         "thickness": Key("m", "roof slab thickness", 0.12),
         "ridge_axis": Key("auto | x | z", "ridge direction (auto: the longer side)", "auto"),
         "ridge_cap": Key("bool", "cap strip along the ridge", True)},
        example="{primitive: roof, size: [1, 0.35, 0.8], style: hip, overhang: 0.06, thickness: 0.03, "
                "material: roof_shingle}"),
    "prism": Primitive(
        "Triangular prism: gable-end infill, wedges and ramps.",
        {"apex": Key("center | back | front", "where the ridge sits", "center")},
        example="{primitive: prism, size: [1, 0.4, 0.2], material: brick}"),
    "stairs": Primitive(
        "A climbable flight (one watertight solid) with swept handrails; climbs toward -Z.",
        {"style": Key("straight | l | spiral", "flight shape"),
         "rise": Key("m", "total height climbed"),
         "width": Key("m", "flight width"),
         "max_riser": Key("m", "step count = ceil(rise / max_riser)"),
         "tread": Key("m", "going of each step"),
         "turn": Key("left | right", "turn direction (l and spiral)"),
         "landing_at": Key("0-1", "where the landing sits (l)"),
         "waist": Key("m", "slab thickness under the steps"),
         "railing": Key("both | left | right | outer | none", "handrail sides"),
         "rail_height": Key("m", "handrail height"),
         "railing_material": Key("material", "handrail material")},
        size="optional",
        example="{primitive: stairs, rise: 1, width: 0.6, tread: 0.2, material: concrete}"),
    "room": Primitive(
        "Hollow box room with openings. Surfaces (facing in): north_wall, south_wall, east_wall, west_wall, "
        "floor, ceiling. Prefer a `floorplan:` asset for real buildings.",
        {"wall_thickness": Key("m", "wall thickness", 0.1),
         "has_floor": Key("bool", "build the floor", True),
         "has_ceiling": Key("bool", "build the ceiling", True),
         "openings": Key("[{wall, position, bottom, width, height}]", "door and window holes (fractions)")},
        example="{primitive: room, size: [1, 0.6, 0.8], wall_thickness: 0.04, has_ceiling: false, "
                "openings: [{wall: south, position: 0.5, width: 0.25, height: 0.7}], material: wall_plaster}"),
    "tree": Primitive(
        "A tree grown in the part box: trunk (hull collider), branches and noise-displaced foliage.",
        {"style": Key("deciduous | conifer | palm", "crown shape"),
         "seed": _SEED,
         "trunk_height": Key("0-1", "bare trunk as a fraction of the height"),
         "trunk_radius": Key("m", "trunk radius at the ground"),
         "attractors": Key("int", "crown density (space colonisation points)"),
         "step": Key("m", "branch segment length"),
         "leaf_size": Key("m", "leaf cluster size"),
         "segments": Key("int", "radial steps of the branches"),
         "foliage_material": Key("material", "leaf material"),
         "fronds": Key("int", "frond count (palm)")},
        example="{primitive: tree, size: [1, 1, 1], anchor: bottom_center, seed: 3, trunk_radius: 0.04, "
                "step: 0.09, leaf_size: 0.14, material: bark, foliage_material: foliage_green}"),
    "rock": Primitive(
        "A random convex cage, subdivided, displaced and flat-bottomed.",
        {"seed": _SEED,
         "points": Key("int", "cage points"),
         "levels": Key("int", "subdivision levels"),
         "roughness": Key("0-1", "displacement strength"),
         "crease": Key("deg | null", "hard-edge angle (null: fully smooth)")},
        example="{primitive: rock, size: [1, 0.6, 0.8], seed: 4, material: rock}"),
    "paths": Primitive(
        "Path network: centre lines buffered, unioned and extruded (loops leave holes). Asset frame.",
        {"paths": Key("[[[x, z], ...] | {spline, closed} | {circle: r, segments, center}]", "the centre lines"),
         "width": Key("m", "path width"),
         "thickness": Key("m", "slab thickness"),
         "bevel": Key("m", "rounded top edge")},
        size="optional",
        example="{primitive: paths, paths: [[[-0.5, -0.4], [0, 0], [0.5, -0.3]], {circle: 0.3, center: [0, 0.1]}], "
                "width: 0.12, thickness: 0.03, material: gravel}"),
    "fence": Primitive(
        "Posts, rails and pickets along a path, with gaps for gates. Asset frame.",
        {"path": Key("[[x, z], ...] | {circle: r, segments, center}", "the fence line"),
         "closed": Key("bool", "join the end back to the start"),
         "height": Key("m", "fence height"),
         "gaps": Key("[[from, to], ...]", "openings in metres along the path"),
         "post": Key("m", "post thickness"),
         "post_spacing": Key("m", "distance between posts"),
         "picket": Key("m", "picket thickness"),
         "spacing": Key("m", "distance between pickets"),
         "rail": Key("m", "rail thickness")},
        size="optional",
        example="{primitive: fence, path: [[-0.5, 0.3], [0.5, 0.3], [0.5, -0.3]], height: 0.4, post: 0.03, "
                "post_spacing: 0.5, picket: 0.01, spacing: 0.06, rail: 0.015, material: metal}"),
    "terrain": Primitive(
        "A closed heightfield solid (walkable): island or rolling ground with material bands. Asset frame.",
        {"extent": Key("[x, z]", "ground size in metres"),
         "resolution": Key("m", "grid cell size"),
         "height": Key("m", "peak above sea level"),
         "sea_level": Key("m", "y of the sea surface"),
         "depth": Key("m", "solid depth below the sea bed"),
         "seed": _SEED,
         "scale": Key("m", "noise feature size"),
         "octaves": Key("int", "noise octaves"),
         "falloff": Key("{radius, edge, power}", "island coastline (fractions of the smaller side)"),
         "island": Key("bool", "false for rolling ground without a coast"),
         "pads": Key("[{center, radius, blend, height}]", "levelled ground for buildings"),
         "materials": Key("{sand, grass, rock, path}", "materials by height and slope"),
         "paths": Key("[{points, width}]", "gravel ribbons draped on the ground"),
         "shore": Key("{depth, height, openings}", "collider-only barrier where the water gets deep")},
        size="optional",
        example="{primitive: terrain, extent: [60, 60], resolution: 1, height: 7, depth: 1, scale: 22, seed: 2, "
                "falloff: {radius: 0.4, edge: 0.15}}"),
    "water": Primitive(
        "A thin water sheet from a shape (no collider); the Godot runtime gives it waves and foam.",
        {"shape": Key("shape", "outline in metres ({outer, holes} for a sea round an island)"),
         "fit": Key("stretch | none", "water takes its shape in metres", "none"),
         "barrier": Key("m", "invisible shore wall of this height along every edge"),
         "shallow_color": Key("[r, g, b]", "tint in the shallows"),
         "deep_color": Key("[r, g, b]", "tint over deep water"),
         "depth_scale": Key("m", "depth at which the deep colour is reached"),
         "wave_scale": Key("factor", "wave height"),
         "wave_speed": Key("factor", "wave speed"),
         "foam": Key("0-1", "shoreline foam")},
        example="{primitive: water, size: [1, 0.02, 1], shape: {circle: 0.5}}"),
}

# Keys read by more than one primitive's shared plumbing (water is an extrude underneath).
_SHARED_KEYS = {"bevel", "bevel_segments", "crease_angle", "caps", "axis"}

ASSET_KEYS: dict[str, str] = {
    "name": "node name of the asset",
    "origin": "anchor of the asset's origin in its box (usually bottom_center)",
    "size": "[x, y, z] container in metres; part sizes and offsets are fractions of it",
    "bounds": "`geometry`: report the real extent (for unit containers with parts in metres)",
    "params": "{name: {default, choices?, type: text?}}: values `{expr}` interpolation can use anywhere",
    "presets": "{name: {param: value}}: named param bundles, picked with `params: {preset: name}`",
    "preset": "the default preset",
    "derived": "{name: expr}: values computed from params, in order",
    "parts": "{name: part}: the primitives the asset is built from",
    "attachments": "{name: {anchor, offset, facing}}: named points other assets attach to (`attach_to` + `at`)",
    "surfaces": "{name: {from: part.surface, cut, reveal, flip}}: 2D regions other assets are placed `on` "
                "(`flip: true` faces the other way, e.g. a cutter's face seen from inside the room)",
    "tags": "dotted gameplay tags (furniture.chair, door.exterior)",
    "interactions": "{name: {targets, initial, states, motions}}: state machines that move parts (doors, drawers)",
    "state": "{name: {type: bool | number | text | choice, default, choices}}: variables each placed instance "
             "holds (read as `thing.name` in conditions)",
    "signals": "{name: {emit | changed | enter | leave: <name>, of, to}}: named outputs wiring can listen to",
    "inputs": "{name: {set: <state or interaction>, to, of}}: named things wiring can drive",
    "rules": "[{id, on: <event>, when: <condition>, do: [effects], after, once, if}]: what happens when "
             "(geogen.wiring); `if: \"{expr}\"` leaves the rule out of the build when 0",
    "wire": "[{from: self | <part>.<signal>, to: self | <part>.<input>, value, when, after, if}]: a signal drives "
            "an input",
    "bind": "{<state>: <expression>}: standing relations; a bound variable is never set or saved",
    "affordances": "[{type: sit | lie | use | stand | look, at, facing, ...}]: what actors can do here",
    "places": "{name: {box | polygon | around}}: trigger volumes",
    "clearance": "{front, back, left, right}: free space the furnishing solver keeps around it",
    "container": "{capacity, part}: the asset holds items",
    "light": "{type: omni, color, energy, range, offset, auto: night}: the asset is a light fixture",
    "switch": "{light: <name>}: the asset's `switch` interaction (on | off) turns that light fixture on and off: "
              "the nearest placement of that name; compiled to a binding of the fixture's `lit` state",
    "portal": "a doorway NPC paths pass through (opening the interaction first)",
    "travel": "{scene, spawn, prompt, on: use | enter, target}: moves the player to another scene",
    "vehicle": "{class, wheels, steer, lamps, max_speed, ...}: the asset drives (meta.vehicle)",
    "physics": "{mass, friction, bounce, carry, spawn, initial, shape, damping}: a rigid body (geogen.physics)",
    "pickup": "<item> or {item, count, prompt}: something to take (E gives the item, hides it)",
    "source": "{items, rate, at, max_alive, count, initial, velocity, spread, seed}: makes items (geogen.flows)",
    "track": "{path, speed, spacing, mode: belt | walk, loop, end, accepts}: items follow it (geogen.flows)",
    "effector": "{range, rate, targets, effect: {damage} | {take}, at, aim, projectile}: acts on items (geogen.flows)",
    "poses": "NPC root transforms per pose name (body assets)",
    "room": "legacy single room (prefer `floorplan`)",
    "floorplan": "rooms, doors and windows instead of parts; surfaces are <room>.<surface>",
    "building": "{storeys: [{floorplan, repeat, furnish}], roof, facade}: stacked floor plans",
    "body": "a skinned humanoid: skeleton, ring-lofted chains, hair, clothing, poses and clips",
    "_params": "internal",
}

SCENE_KEYS: dict[str, str] = {
    "name": "node name of the scene",
    "size": "[x, y, z] box that slot anchors resolve in",
    "params": "{name: {default}}: values `{expr}` interpolation can use",
    "slots": "{name: {anchor, offset | position, facing}}: named places for placements",
    "place": "{name: placement}: the objects in the scene",
    "compose": "same as `place`",
    "attachments": "named points on the scene, for parent scenes",
    "tags": "dotted gameplay tags on the scene root",
    "state": "{name: {scope: world | actor | instance, type, default}}: `world.x`, `actor.x` or each copy's `self.x`",
    "signals": "{name: {emit | changed | enter | leave}}: each copy's outputs",
    "inputs": "{name: {set, to}}: what a parent drives in each copy",
    "rules": "[{id, on: <event>, when: <condition>, do: [effects], after, once, if}]: what happens when "
             "(geogen.wiring); `if: \"{expr}\"` leaves the rule out of the build when 0",
    "wire": "[{from: <thing>.<signal>, to: <thing>.<input>, value, when, after, if}]: a signal drives an input",
    "bind": "{<state>: <expression>}: standing relations; a bound variable is never set or saved",
    "quests": "[quests/name.yaml, ...]: the `kind: quest` files this scene contributes (docs/quests.md); "
              "`{file, if: \"{expr}\"}` contributes one only when the expression isn't 0",
    "dialogues": "[dialogues/name.yaml | {file, if}]: conversations anchored to things (geogen.dialogue)",
    "hud": "{stats: [{label, value, of}], banners: [{when, text, tone}]}: on-screen numbers (geogen.hud)",
    "physics": "{ticks: n}: physics steps a second while the scene is loaded (runtime default 60); fast "
               "rolling bodies on curved meshes need 120",
    "spawns": "{name: {position | anchor, facing}}: where the player can start",
    "places": "{name: {box | polygon | around}}: trigger volumes (enter / leave events)",
    "city": "a street grid with zoned lots, buildings, street furniture, parking and traffic",
    "routes": "{name: {path, lanes, speed, loop}}: roads with traffic lanes",
    "tracks": "{name: {path, speed, mode, loop, end, surface}}: paths spawned items follow (geogen.flows)",
    "railways": "{name: {path, loop, gauge, stations}}: track beds with platforms",
}

SLOT_KEYS: dict[str, str] = {
    "anchor": "anchor in the scene box (bottom_center, bottom_front_left, ...)",
    "offset": "[x, y, z] metres from the anchor",
    "position": "[x, y, z] metres, instead of anchor + offset",
    "facing": "center | outward | north (+Z) | south | east (+X) | west",
}

PLACE_KEYS: dict[str, str] = {
    "box": "[x, y, z] m: a box standing on `position`",
    "polygon": "[[x, z], ...]: a footprint, from `y` up `height`",
    "around": "a reference: the box round it",
    "position": "[x, y, z]: the middle of a box's bottom",
    "anchor": "scenes: an anchor of the scene box, instead",
    "offset": "scenes: m from the anchor",
    "yaw": "degrees a box is turned",
    "y": "a polygon's base (0)",
    "height": "a polygon's height (2.5)",
    "margin": "m round an `around` box, or [x, y, z]",
    "headroom": "m above an `around` box, to stand on it",
    "detect": "[actors, props]: props count too, in its `occupants` state",
    "sink": "true or {accepts: [item names]}: spawned items entering it are taken in (geogen.flows)",
    "tags": "dotted tags",
}

PLACEMENT_KEYS: dict[str, str] = {
    "asset": "asset file to load, relative to assets/ (door.yaml, vehicles/car.yaml)",
    "scene": "nested scene file (scenes/cottage.yaml)",
    "params": "{name: value}: overrides of the asset's params (or `preset: name`)",
    "slot": "place at a named slot of this scene",
    "attach_to": "another placement to attach to, with `at` (a reference: `table`, or `plot/table` inside a "
                 "placed scene)",
    "at": "attachment name(s) on the `attach_to` object (a list places one copy each), "
          "or {u, v, depth} on an `on` surface (0-1 fractions; {abs: m} for metres)",
    "on": "<object>.<surface> to place on (walls, floors, table tops); openings cut their host. The object is a "
          "reference: `house.floor`, `plot/house.floor` inside a placed scene, `suite/[tag=furniture.desk].top`",
    "facing": "turn the object: center | outward | north | south | east | west",
    "yaw": "degrees about +Y, added to the facing",
    "state": "initial interaction state: `open`, or {interaction: state}; {variable: value} starts a declared "
             "variable of the asset with another value",
    "when": "drop the placement when the expression is 0 / false (comparisons, and / or / not allowed)",
    "tags": "extra tags on the placed object",
    "furnish": "true or a seed: furnish a floor plan's rooms from their room types",
    "scatter": "{seed, rect | path | on | on_terrain | on_tag, count, spacing, avoid, yaw, scale}: many copies "
               "(`avoid` takes references, selectors included: [house, 'bench_*', '[tag=furniture]'])",
    "npc": "an NPC definition (npcs/resident.yaml), with `on`/`home`",
    "home": "<object>.<surface> (the object is a reference, like `on`) or {rect, y}: the area an NPC lives in",
    "seed": "random seed (which person an archetype or NPC is)",
    "pickup": "<item> or {item, count, prompt}: something to take (as the asset key)",
    "quests": "an NPC's quests (files or {file, if}): it holds and works on them",
    "archetype": "a random person (characters/archetypes/adult.yaml)",
    "traffic": "a moving fleet (traffic/town.yaml) on the scene's lanes",
    "count": "fleet size for a traffic placement",
    "train": "a train (trains/commuter.yaml), with `railway`",
    "railway": "the railway a train runs on",
    "timetable": "{headway, dwell, offset}: overrides of a train's timetable",
    "travel": "{scene, spawn, prompt}: overrides of the asset's travel target (spawn: a spawn's name, or its "
              "full path in the target: street/hotel/entrance_spawn)",
}

ANCHORS_DOC = ("center, or any of top|bottom + front|back + left|right|center, e.g. bottom_center, top_front_left, "
               "left_center (x: left = -X, y: up, z: front = +Z)")


def part_known_keys() -> set[str]:
    return set(PART_KEYS).union(*(p.keys for p in PRIMITIVES.values()), _SHARED_KEYS)
