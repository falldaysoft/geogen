# Vocabulary

## Asset keys

| Key | Meaning |
| --- | --- |
| `name` | node name of the asset |
| `origin` | anchor of the asset's origin in its box (usually bottom_center) |
| `size` | [x, y, z] container in metres; part sizes and offsets are fractions of it |
| `bounds` | `geometry`: report the real extent (for unit containers with parts in metres) |
| `params` | {name: {default, choices?, type: text?}}: values `{expr}` interpolation can use anywhere |
| `presets` | {name: {param: value}}: named param bundles, picked with `params: {preset: name}` |
| `preset` | the default preset |
| `derived` | {name: expr}: values computed from params, in order |
| `parts` | {name: part}: the primitives the asset is built from |
| `attachments` | {name: {anchor, offset, facing}}: named points other assets attach to (`attach_to` + `at`) |
| `surfaces` | {name: {from: part.surface, cut, reveal}}: 2D regions other assets are placed `on` |
| `tags` | dotted gameplay tags (furniture.chair, door.exterior) |
| `interactions` | {name: {targets, initial, states, motions}}: state machines that move parts (doors, drawers) |
| `state` | {name: {type: bool \| number \| text \| choice, default, choices}}: variables each placed instance holds (read as `thing.name` in conditions) |
| `signals` | {name: {emit \| changed \| enter \| leave: <name>, of, to}}: named outputs wiring can listen to |
| `inputs` | {name: {set: <state or interaction>, to, of}}: named things wiring can drive |
| `affordances` | [{type: sit \| lie \| use \| stand \| look, at, facing, ...}]: what actors can do here |
| `places` | {name: {box \| polygon \| around}}: trigger volumes |
| `clearance` | {front, back, left, right}: free space the furnishing solver keeps around it |
| `container` | {capacity, part}: the asset holds items |
| `light` | {type: omni, color, energy, range, offset, auto: night}: the asset is a light fixture |
| `portal` | a doorway NPC paths pass through (opening the interaction first) |
| `travel` | {scene, spawn, prompt, on: use \| enter, target}: moves the player to another scene |
| `vehicle` | {class, wheels, steer, lamps, max_speed, ...}: the asset drives (meta.vehicle) |
| `poses` | NPC root transforms per pose name (body assets) |
| `room` | legacy single room (prefer `floorplan`) |
| `floorplan` | rooms, doors and windows instead of parts; surfaces are <room>.<surface> |
| `building` | {storeys: [{floorplan, repeat, furnish}], roof, facade}: stacked floor plans |
| `body` | a skinned humanoid: skeleton, ring-lofted chains, hair, clothing, poses and clips |

## Scene keys

| Key | Meaning |
| --- | --- |
| `name` | node name of the scene |
| `size` | [x, y, z] box that slot anchors resolve in |
| `params` | {name: {default}}: values `{expr}` interpolation can use |
| `slots` | {name: {anchor, offset \| position, facing}}: named places for placements |
| `place` | {name: placement}: the objects in the scene |
| `compose` | same as `place` |
| `attachments` | named points on the scene, for parent scenes |
| `tags` | dotted gameplay tags on the scene root |
| `state` | {name: {scope: world \| actor, type, default, choices}}: the package's shared (`world.name`) and per-actor (`actor.name`) variables |
| `quests` | [quests/name.yaml, ...]: the `kind: quest` files this scene contributes (docs/quests.md) |
| `spawns` | {name: {position \| anchor, facing}}: where the player can start |
| `places` | {name: {box \| polygon \| around}}: trigger volumes (enter / leave events) |
| `city` | a street grid with zoned lots, buildings, street furniture, parking and traffic |
| `routes` | {name: {path, lanes, speed, loop}}: roads with traffic lanes |
| `railways` | {name: {path, loop, gauge, stations}}: track beds with platforms |

## Slot keys

| Key | Meaning |
| --- | --- |
| `anchor` | anchor in the scene box (bottom_center, bottom_front_left, ...) |
| `offset` | [x, y, z] metres from the anchor |
| `position` | [x, y, z] metres, instead of anchor + offset |
| `facing` | center \| outward \| north (+Z) \| south \| east (+X) \| west |

## Placement keys

| Key | Meaning |
| --- | --- |
| `asset` | asset file to load, relative to assets/ (door.yaml, vehicles/car.yaml) |
| `scene` | nested scene file (scenes/cottage.yaml) |
| `params` | {name: value}: overrides of the asset's params (or `preset: name`) |
| `slot` | place at a named slot of this scene |
| `attach_to` | another placement to attach to, with `at` (a reference: `table`, or `plot/table` inside a placed scene) |
| `at` | attachment name(s) on the `attach_to` object (a list places one copy each), or {u, v, depth} on an `on` surface (0-1 fractions; {abs: m} for metres) |
| `on` | <object>.<surface> to place on (walls, floors, table tops); openings cut their host. The object is a reference: `house.floor`, `plot/house.floor` inside a placed scene, `suite/[tag=furniture.desk].top` |
| `facing` | turn the object: center \| outward \| north \| south \| east \| west |
| `yaw` | degrees about +Y, added to the facing |
| `state` | initial interaction state: `open`, or {interaction: state}; {variable: value} starts a declared variable of the asset with another value |
| `when` | drop the placement when the expression is 0 / false (comparisons, and / or / not allowed) |
| `tags` | extra tags on the placed object |
| `furnish` | true or a seed: furnish a floor plan's rooms from their room types |
| `scatter` | {seed, rect \| path \| on \| on_terrain \| on_tag, count, spacing, avoid, yaw, scale}: many copies (`avoid` takes references, selectors included: [house, 'bench_*', '[tag=furniture]']) |
| `npc` | an NPC definition (npcs/resident.yaml), with `on`/`home` |
| `home` | <object>.<surface> (the object is a reference, like `on`) or {rect, y}: the area an NPC lives in |
| `seed` | random seed (which person an archetype or NPC is) |
| `archetype` | a random person (characters/archetypes/adult.yaml) |
| `traffic` | a moving fleet (traffic/town.yaml) on the scene's lanes |
| `count` | fleet size for a traffic placement |
| `train` | a train (trains/commuter.yaml), with `railway` |
| `railway` | the railway a train runs on |
| `timetable` | {headway, dwell, offset}: overrides of a train's timetable |
| `travel` | {scene, spawn, prompt}: overrides of the asset's travel target (spawn: a spawn's name, or its full path in the target: street/hotel/entrance_spawn) |

## Place keys

| Key | Meaning |
| --- | --- |
| `box` | [x, y, z] m: a box standing on `position` |
| `polygon` | [[x, z], ...]: a footprint, from `y` up `height` |
| `around` | a reference: the box round it |
| `position` | [x, y, z]: the middle of a box's bottom |
| `anchor` | scenes: an anchor of the scene box, instead |
| `offset` | scenes: m from the anchor |
| `yaw` | degrees a box is turned |
| `y` | a polygon's base (0) |
| `height` | a polygon's height (2.5) |
| `margin` | m round an `around` box, or [x, y, z] |
| `headroom` | m above an `around` box, to stand on it |
| `tags` | dotted tags |

## Kinds

Data files with a top-level `kind:`.

### archetype

A seeded distribution over a body's params: each seed is a different person.

Use: `place: {anna: {archetype: characters/archetypes/young_woman.yaml, seed: 3}}`

Keys: `body`, `description`, `extends`, `name`, `params`, `tags`, `variants`

| File | What it is |
| --- | --- |
| [characters/archetypes/adult.yaml](../../assets/characters/archetypes/adult.yaml) | Base archetype: anyone. |
| [characters/archetypes/hotel_staff.yaml](../../assets/characters/archetypes/hotel_staff.yaml) | Hotel staff in uniform: white top, black trousers, black shoes, hair short or tied back. |
| [characters/archetypes/office_worker.yaml](../../assets/characters/archetypes/office_worker.yaml) | Anyone dressed for the office: smart tops in sober colours, dark trousers. |
| [characters/archetypes/older_man.yaml](../../assets/characters/archetypes/older_man.yaml) | A man in his sixties or seventies: grey or no hair, a bit of a belly. |
| [characters/archetypes/older_woman.yaml](../../assets/characters/archetypes/older_woman.yaml) | A woman in her sixties or seventies: greying hair, fuller figure, sensible shoes. |
| [characters/archetypes/townsfolk.yaml](../../assets/characters/archetypes/townsfolk.yaml) | Whoever you meet in the street: a weighted mix of the other archetypes. |
| [characters/archetypes/young_man.yaml](../../assets/characters/archetypes/young_man.yaml) | A man in his twenties or thirties in everyday clothes. |
| [characters/archetypes/young_woman.yaml](../../assets/characters/archetypes/young_woman.yaml) | A woman in her twenties or thirties in everyday clothes. |

### garment

A piece of clothing fitted to the humanoid; outfits are picked with its `outfit` param.

Use: `place: {x: {asset: characters/humanoid.yaml, params: {outfit: dress}}}`

Keys: `chains`, `cover`, `cut`, `layer`, `material`, `name`, `thickness`, `type`, `weights`

| File | What it is |
| --- | --- |
| [clothing/boots.yaml](../../assets/clothing/boots.yaml) |  |
| [clothing/dress.yaml](../../assets/clothing/dress.yaml) | Sleeveless dress: a tight bodice plus the skirt shell, one colour. |
| [clothing/flats.yaml](../../assets/clothing/flats.yaml) |  |
| [clothing/jeans.yaml](../../assets/clothing/jeans.yaml) |  |
| [clothing/long_sleeve_top.yaml](../../assets/clothing/long_sleeve_top.yaml) |  |
| [clothing/shorts.yaml](../../assets/clothing/shorts.yaml) |  |
| [clothing/skirt.yaml](../../assets/clothing/skirt.yaml) | A-line skirt to just above the knee: a loose shell from the waist, weighted from the body under it (the thighs below the hips) so it swings with the legs. |
| [clothing/sneakers.yaml](../../assets/clothing/sneakers.yaml) |  |
| [clothing/sweater.yaml](../../assets/clothing/sweater.yaml) |  |
| [clothing/t_shirt.yaml](../../assets/clothing/t_shirt.yaml) |  |
| [clothing/tank_top.yaml](../../assets/clothing/tank_top.yaml) |  |
| [clothing/trousers.yaml](../../assets/clothing/trousers.yaml) |  |

### npc

An NPC: body, needs, preferences, activities and daily routine; the runtime interprets it.

Use: `place: {resident: {npc: npcs/resident.yaml, on: house.floor, home: house.floor, seed: 1}}`

Keys: `activities`, `affordance_tags`, `attention`, `body`, `flags`, `home_margin`, `needs`, `preferences`, `radius`, `routine`, `scoring`, `seed`, `speed`, `turn_speed`, `wander`

| File | What it is |
| --- | --- |
| [npcs/diner.yaml](../../assets/npcs/diner.yaml) | Out for a meal or a drink: sits at a table or in a booth, or at the bar, and moves between them now and then. |
| [npcs/hotel_guest.yaml](../../assets/npcs/hotel_guest.yaml) | A hotel guest in their room: sleeps, sits, looks out of the window, wanders the room. |
| [npcs/hotel_staff.yaml](../../assets/npcs/hotel_staff.yaml) | Hotel front-of-house staff: works the reception desk (its staff-side `stand` affordance, tagged `staff`), takes breaks on the lobby seating and walks the lobby now and then. |
| [npcs/park_visitor.yaml](../../assets/npcs/park_visitor.yaml) | Someone spending time in the park: walks the gravel paths, sits on benches and at the picnic tables, stops on the bandstand to look around. |
| [npcs/pedestrian.yaml](../../assets/npcs/pedestrian.yaml) | Somebody out and about: strolls the sidewalks, stops to look around, rests on benches. |
| [npcs/resident.yaml](../../assets/npcs/resident.yaml) | A person who lives in a house: rests in chairs, looks out of the windows, browses the shelves and steps out of the front door now and then. |

### npc_actions

The actions NPCs run at affordances, as steps (go_to, face, pose, wait, use, play, vanish).

Use: `referenced by affordances: {type: sit, action: sit}`

Keys: `face`, `go_to`, `play`, `pose`, `use`, `vanish`, `wait`

| File | What it is |
| --- | --- |
| [npcs/actions.yaml](../../assets/npcs/actions.yaml) | What NPCs do, as data. |

### player_spec

Player-scale constraints (capsule, eye and step height, door and corridor widths).

Use: `read by layout QA, the furnishing solver and the Godot runtime`

Keys: `corridor_min_width`, `door_min_height`, `door_min_width`, `eye_height`, `height`, `max_slope_deg`, `radius`, `reach`, `step_height`

| File | What it is |
| --- | --- |
| [player.yaml](../../assets/player.yaml) | Player-scale constraints shared by generation and the Godot runtime. |

### quest

A quest: objectives done by conditions or events, compiled to actor state and rules.

Use: `quests: [quests/lighthouse_gallery.yaml]   (a scene key; see docs/quests.md)`

Keys: `available`, `description`, `fail`, `holder`, `id`, `objectives`, `ordered`, `rewards`, `start`, `title`

| File | What it is |
| --- | --- |
| [quests/lighthouse_gallery.yaml](../../assets/quests/lighthouse_gallery.yaml) | Climb to the lighthouse gallery: the "go there" quest, one place to reach. |

### scene_catalogue

The Godot runtime's scenes (showcase | test) and its default.

Use: `python -m geogen.main --export-catalogue`

Keys: `default`, `scenes`

| File | What it is |
| --- | --- |
| [runtime_scenes.yaml](../../assets/runtime_scenes.yaml) | The scenes the Godot runtime offers. `--export-catalogue` exports them into runtime/godot/generated/ and writes generated/catalogue.json; with no. |

### skeleton

The humanoid skeleton: params, bone heads and landmarks.

Use: `body: {skeleton: skeletons/humanoid.yaml}`

Keys: `bones`, `derived`, `group_scale`, `landmarks`, `name`, `params`, `profile`

| File | What it is |
| --- | --- |
| [skeletons/humanoid.yaml](../../assets/skeletons/humanoid.yaml) | Humanoid skeleton: Godot SkeletonProfileHumanoid bones (humanoid_profile.json), placed by body proportions. |

### traffic

A moving fleet: weighted vehicles, count, driving behaviour, turn weights.

Use: `place: {cars: {traffic: traffic/town.yaml, seed: 1, count: 10}}`

Keys: `count`, `description`, `driving`, `fleet`, `radius`, `schedule`, `seed`, `spacing`, `turns`, `yield`

| File | What it is |
| --- | --- |
| [traffic/town.yaml](../../assets/traffic/town.yaml) | Traffic for a town: which vehicles drive the lanes and how (geogen/traffic.py). |

### train

A train: consist, speed and timetable, run on one of the scene's `railways:`.

Use: `place: {train: {train: trains/commuter.yaml, railway: main}}`

Keys: `accel`, `consist`, `decel`, `description`, `speed`, `timetable`, `warning`

| File | What it is |
| --- | --- |
| [trains/commuter.yaml](../../assets/trains/commuter.yaml) | A commuter train: a locomotive and three carriages, every four minutes. |

## Room types

A floor-plan room's `type:`; `furnish: true` fills the room from it.

| Room type | What it is | Furnished with |
| --- | --- | --- |
| `back_of_house` | Back of house: an office corner and storage shelving. | bookshelf, desk, desk_chair |
| `bar` | A bar: a back bar against the longest plain wall, the counter across a staff aisle from it with stools on the room side, plus a couple of armchairs and some art. | armchair, back_bar, bar_counter, bar_stool, planter, wall_art |
| `bathroom` | Home bathroom: bath (when it fits), toilet and basin. | bathtub, toilet, vanity, wall_mirror |
| `bedroom` | Home bedroom: bed against the long wall, nightstands, wardrobe by the door. | bed, curtains, nightstand, rug, table_lamp, wall_art, wardrobe |
| `bedroom_single` | Small bedroom: a single bed, one nightstand and a wardrobe. | bed, curtains, nightstand, table_lamp, wardrobe |
| `bookshop` | A bookshop: bookcases along the walls, the till, and a reading chair (the middle of the floor stays open so the back of the shop is reachable). | armchair, bookshelf, till_counter |
| `cafe` | A cafe: the counter with the espresso machine at the back, a cake display beside it, and small tables for two by the window. | cafe_counter, chair, display_case, planter, table |
| `clothing` | A clothes shop: rails of garments on the floor, shelving on the walls, a long mirror and the till. | armchair, clothes_rail, till_counter, wall_mirror, wall_shelving |
| `corridor` | Suite entry corridor: a luggage rack if there's room. | luggage_rack |
| `dining_room` | A restaurant dining room: booths along a wall, laid tables with chairs, a host stand by the door and a chalkboard menu. | booth, chair, host_stand, menu_board, planter, table, table_setting |
| `grocer` | A grocer: shelving round the walls, gondola shelves down the middle and the till by the door. | gondola, till_counter, wall_shelving |
| `hotel_bathroom` | Hotel ensuite bathroom archetype. | shower, toilet, towel_rail, vanity, wall_mirror |
| `hotel_bedroom` | Hotel bedroom archetype: items in priority order, placed by the furnishing solver (geogen/layout/furnish.py) inside any floor-plan room of this type. | armchair, bed, curtains, desk, desk_chair, floor_lamp, nightstand, rug, table_lamp, tv, tv_console, wall_art, wardrobe |
| `kitchen` | Kitchen-diner: a table with chairs round it (no fitted kitchen units yet). | chair, planter, table |
| `lift_lobby` | Lift lobby on guest floors: plants and a seat while waiting. | armchair, planter |
| `living_room` | Living room: sofa facing the TV across a coffee table, an armchair and a lamp. | armchair, bookshelf, coffee_table, curtains, floor_lamp, sofa, tv, tv_console |
| `lobby` | Hotel entrance lobby: a seating group and plants, keeping the route from the street door to reception clear (door approach zones are kept free). | coffee_table, planter, sofa |
| `lounge` | Lounge: two sofas facing across a coffee table, armchairs, lamps, plants. | armchair, coffee_table, floor_lamp, planter, sofa |
| `office` | Open office: desks with chairs along the walls, shelving and plants. | bookshelf, desk, desk_chair, planter |
| `reception` | Reception: the counter facing the lobby with a chair behind it. | desk_chair, planter, reception_desk |
| `restaurant` | Restaurant: dining tables with four chairs each, spaced for service. | chair, planter, table |
| `restaurant_kitchen` | A restaurant kitchen: the range under its hood on a plain wall, prep tables beside it, a fridge, and the pass by the door to the dining room. | fridge, kitchen_range, pass, prep_counter |
| `restroom` | Public restroom: toilet cubicles side by side on the wall furthest from the door (each with a toilet inside), a basin and mirror by the door, and a hand dryer beside them. | hand_dryer, toilet, toilet_stall, vanity, wall_mirror |
| `shop` | Shop floor: a counter facing the entrance and shelving along the walls. | bookshelf, planter, reception_desk |

## Materials

`material: <name>` on a part.

| Material | Texture |
| --- | --- |
| `asphalt` | asphalt |
| `bark` | wood |
| `bedding_white` | fabric |
| `brick` | brick |
| `car_glass` | metal |
| `car_paint` | painted_wall |
| `car_trim` | painted_wall |
| `carpet` | cut_pile_carpet |
| `ceiling_white` | painted_wall |
| `ceramic_white` | painted_wall |
| `chrome` | metal |
| `cloth` | fabric |
| `concrete` | concrete |
| `denim` | fabric |
| `dirt` | dirt |
| `door_paint` | painted_wall |
| `fabric_linen` | fabric |
| `fabric_velvet` | fabric |
| `floor` | concrete |
| `foliage_green` | painted_wall |
| `foliage_palm` | painted_wall |
| `foliage_pine` | painted_wall |
| `glass` | metal |
| `glow_accent` | painted_wall |
| `grass` | grass |
| `gravel` | dirt |
| `hair` | hair |
| `hardwood_floor` | hardwood_floor |
| `headlight` | painted_wall |
| `laminate` | hardwood_floor |
| `lamp_glass` | painted_wall |
| `lamp_iron` | painted_wall |
| `lamp_shade` | fabric |
| `leather` | skin |
| `marble` | marble |
| `metal` | metal |
| `mirror` | metal |
| `paint_greige` | painted_wall |
| `paint_navy` | painted_wall |
| `paint_sage` | painted_wall |
| `paint_terracotta` | painted_wall |
| `paint_white` | painted_wall |
| `palm_bark` | wood |
| `plastic_black` | painted_wall |
| `play_paint` | painted_wall |
| `play_yellow` | painted_wall |
| `portal_glow` | painted_wall |
| `red_metal` | metal |
| `road_paint_white` | painted_wall |
| `road_paint_yellow` | painted_wall |
| `rock` | rock |
| `roof_shingle` | roof |
| `rubber_play` | concrete |
| `rug` | cut_pile_carpet |
| `sand` | dirt |
| `screen` | painted_wall |
| `signal_amber` | painted_wall |
| `signal_green` | painted_wall |
| `signal_red` | painted_wall |
| `skin` | skin |
| `stone` | concrete |
| `taillight` | painted_wall |
| `tile_floor_grey` | tile |
| `tile_subway` | tile |
| `tile_white` | tile |
| `trim_white` | painted_wall |
| `tyre` | painted_wall |
| `wall` | plaster |
| `wall_plaster` | plaster |
| `water` | painted_wall |
| `wood` | wood |
| `wood_dark` | wood |
