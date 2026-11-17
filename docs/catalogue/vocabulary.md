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
| `attachments` | {name: {anchor, offset, facing, grip, hands}}: named points other assets attach to (`attach_to` + `at`); `grip` says how a body touches it (press, power, pinch, sit, look, put: motions, geogen/motion.py) |
| `surfaces` | {name: {from: part.surface, cut, reveal, flip}}: 2D regions other assets are placed `on` (`flip: true` faces the other way, e.g. a cutter's face seen from inside the room) |
| `tags` | dotted gameplay tags (furniture.chair, door.exterior) |
| `interactions` | {name: {targets, initial, states, motions, touch}}: state machines that move parts (doors, drawers) or drive a mechanism body's motor (a `drive` motion); `touch` names the attachment a body touches to work it (motions, geogen/motion.py) |
| `state` | {name: {type: bool \| number \| text \| choice, default, choices}}: variables each placed instance holds (read as `thing.name` in conditions) |
| `signals` | {name: {emit \| changed \| enter \| leave: <name>, of, to}}: named outputs wiring can listen to |
| `inputs` | {name: {set: <state or interaction>, to, of}}: named things wiring can drive |
| `rules` | [{id, on: <event>, when: <condition>, do: [effects], after, once, if}]: what happens when (geogen.wiring); `if: "{expr}"` leaves the rule out of the build when 0 |
| `wire` | [{from: self \| <part>.<signal>, to: self \| <part>.<input>, value, when, after, if}]: a signal drives an input |
| `bind` | {<state>: <expression>}: standing relations; a bound variable is never set or saved |
| `affordances` | [{type: sit \| lie \| use \| stand \| look, at, facing, body, ...}]: what actors can do here (`body`: a mechanism body the seat rides on; `requires: {holding, state}`, `takes`, `gives`: what doing it needs and does to what the actor carries, the links NPCs plan chains through; `uses: {name: {requires, takes, gives, advertises, duration, tags}}`: other things done at the same place, sharing its slots: a chair's `eat`) |
| `places` | {name: {box \| polygon \| around}}: trigger volumes |
| `clearance` | {front, back, left, right}: free space the furnishing solver keeps around it |
| `container` | {capacity, part}: the asset holds items |
| `light` | {type: omni, color, energy, range, offset, auto: night}: the asset is a light fixture |
| `switch` | {light: <name>}: the asset's `switch` interaction (on \| off) turns that light fixture on and off: the nearest placement of that name; compiled to a binding of the fixture's `lit` state |
| `portal` | a doorway NPC paths pass through (opening the interaction first) |
| `travel` | {scene, spawn, prompt, on: use \| enter, target}: moves the player to another scene |
| `vehicle` | {class, wheels, steer, lamps, max_speed, ...}: the asset drives (meta.vehicle) |
| `physics` | {mass, friction, bounce, carry, spawn, initial, shape, damping}: a rigid body (geogen.physics) |
| `mechanism` | {bodies: {name: {parts, mass, joint: {type: hinge \| slide \| pin, to, axis, pivot, limits, motor}}}}: parts as jointed rigid bodies, motors run by `drive` motions (geogen.mechanism) |
| `pickup` | <item> or {item, count, prompt}: something to take (E gives the item, hides it) |
| `readable` | <prompt> or true: a `read` interaction a dialogue can open on (geogen.dialogue) |
| `source` | {items, rate, at, max_alive, count, initial, velocity, spread, seed}: makes items (geogen.flows) |
| `track` | {path, speed, spacing, mode: belt \| walk, loop, end, accepts}: items follow it (geogen.flows) |
| `effector` | {range, rate, targets, effect: {damage} \| {take}, at, aim, projectile}: acts on items (geogen.flows) |
| `poses` | NPC root transforms per pose name (body assets) |
| `floorplan` | rooms, doors and windows instead of parts; surfaces are <room>.<surface> |
| `building` | {storeys: [{floorplan, repeat, furnish}], roof, facade}: stacked floor plans |
| `body` | a skinned body on a skeleton (a person, or an animal on a skeleton with its own bones): ring-lofted chains, a coat colour and markings, hair, clothing, poses and clips |

## Scene keys

| Key | Meaning |
| --- | --- |
| `name` | node name of the scene |
| `size` | [x, y, z] box that slot anchors resolve in |
| `params` | {name: {default}}: values `{expr}` interpolation can use |
| `presets` | {name: {param: value}}: named param bundles, picked with `params: {preset: name}` |
| `preset` | the default preset |
| `derived` | {name: expr}: values computed from params, in order |
| `slots` | {name: {anchor, offset \| position, facing}}: named places for placements |
| `place` | {name: placement}: the objects in the scene |
| `attachments` | named points on the scene, for parent scenes |
| `tags` | dotted gameplay tags on the scene root |
| `state` | {name: {scope: world \| actor \| instance, type, default}}: `world.x`, `actor.x` or each copy's `self.x` |
| `signals` | {name: {emit \| changed \| enter \| leave}}: each copy's outputs |
| `inputs` | {name: {set, to}}: what a parent drives in each copy |
| `rules` | [{id, on: <event>, when: <condition>, do: [effects], after, once, if}]: what happens when (geogen.wiring); `if: "{expr}"` leaves the rule out of the build when 0 |
| `wire` | [{from: <thing>.<signal>, to: <thing>.<input>, value, when, after, if}]: a signal drives an input |
| `bind` | {<state>: <expression>}: standing relations; a bound variable is never set or saved |
| `quests` | [quests/name.yaml, ...]: the `kind: quest` files this scene contributes (docs/quests.md); `{file, if: "{expr}"}` contributes one only when the expression isn't 0 |
| `dialogues` | [dialogues/name.yaml \| {file, if}]: conversations anchored to things (geogen.dialogue) |
| `hud` | {stats: [{label, value, of}], banners: [{when, text, tone}]}: on-screen numbers (geogen.hud) |
| `play` | {as: people, households: [names]}: the scene is played by directing its people from above (the runtime starts in its direct mode with no walking player, and the HUD is the selected person's); households: only the NPCs placed with one of these `household:` names can be told what to do (all, if none) |
| `ground` | false: the runtime keeps no ground plane at y = 0 while the scene is loaded, so the scene's own floors are all there is and things can go below them (a pit); default true |
| `physics` | {ticks: n}: physics steps a second while the scene is loaded (runtime default 60); fast rolling bodies on curved meshes need 120 |
| `atmosphere` | {preset: clear \| overcast \| grim \| storm, time, fog, haze, saturation, contrast, sun, sun_color, sun_height, sky, moon, night, glow}: the lighting mood the runtime plays the scene in (geogen.atmosphere); a preset's values, overridden |
| `spawns` | {name: {position \| anchor, facing}}: where the player can start |
| `places` | {name: {box \| polygon \| around}}: trigger volumes (enter / leave events) |
| `population` | {name: {npc, count: n \| [lo, hi], on: <places>.<surface>, spacing, seed}}: NPCs living on each place a reference or selector names (residents per house), drawn deterministically |
| `city` | a street grid with zoned lots, buildings, street furniture, parking and traffic |
| `routes` | {name: {path, lanes, speed, loop}}: roads with traffic lanes |
| `tracks` | {name: {path, speed, mode, loop, end, surface}}: paths spawned items follow (geogen.flows) |
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
| `character` | a named person (people/marco.yaml), with defaults the placement's keys override |
| `home` | <object>.<surface> (the object is a reference, like `on`) or {rect, y}: the area an NPC lives in |
| `seed` | random seed (which person an archetype or NPC is) |
| `pickup` | <item> or {item, count, prompt}: something to take (as the asset key) |
| `readable` | <prompt> or true: something to read (as the asset key) |
| `quests` | an NPC's quests (files or {file, if}): it holds and works on them |
| `group` | {size: n \| [lo, hi], radius, spread}: on an NPC scatter, each copy is a group of people who stay loosely together (the leader counts in size; radius: how near is together, spread: how far apart they start) |
| `household` | an NPC's household (a name): NPCs with the same one know each other from the start (`relations`) |
| `owns` | an NPC's things (references): only it uses them, and it likes them best |
| `prefers` | things an NPC likes best (references), still free for others |
| `work` | an NPC's work (a reference): where its routine's `at: work` blocks keep it |
| `visits` | {name: <object>.<surface> \| {rect, y}}: places outside an NPC's home its activities' `visit` goes to |
| `present` | a condition (run time): an NPC is in the scene only while it holds (a story moves a character) |
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
| `detect` | [actors, props]: props count too, in its `occupants` state |
| `sink` | true or {accepts: [item names]}: spawned items entering it are taken in (geogen.flows) |
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

### character

A named person: an NPC role (`npc:`) with its own name, look and defaults.

Use: `place: {marco: {character: people/marco.yaml}}`

Keys: `activities`, `affordance_tags`, `attention`, `body`, `carry`, `flags`, `home`, `home_margin`, `household`, `memory`, `name`, `needs`, `npc`, `offers`, `owns`, `planner`, `preferences`, `prefers`, `quests`, `radius`, `reach`, `reacts`, `relations`, `routine`, `scoring`, `seed`, `senses`, `speed`, `turn_speed`, `visits`, `wander`, `wants`, `work`

| File | What it is |
| --- | --- |
| [people/ana.yaml](../../assets/people/ana.yaml) | Ana, who lives at the cottage (scenes/cottage.yaml): the one who asks callers to put the lights out. |
| [people/bea.yaml](../../assets/people/bea.yaml) | Bea, who lives at the cottage (scenes/cottage.yaml): early to bed. |
| [people/hana.yaml](../../assets/people/hana.yaml) | Hana Mori, a house-proud resident: a named person on top of the resident role (npcs/resident.yaml). |
| [people/marco.yaml](../../assets/people/marco.yaml) | Marco Bellini, the chef at the Corner Bistro: a named person on top of the chef role (npcs/chef.yaml: the range, the prep counter, a look into the dining room now and then, home at night). |
| [people/rosa.yaml](../../assets/people/rosa.yaml) | Rosa Lind, a regular: lunch at the Corner Bistro, the rest of the day in the park. |
| [people/smokey.yaml](../../assets/people/smokey.yaml) | Smokey, a small grey cat: a named animal on top of the cat role (npcs/cat.yaml), as named people are on theirs. |
| [people/tomas.yaml](../../assets/people/tomas.yaml) | Tomas Lind, a resident who puts things up out of the way: a named person on top of the resident role (npcs/resident.yaml). |

### dialogue

A conversation: passages of text with choices that can test conditions, run effects and jump.

Use: `dialogues: [dialogues/receptionist.yaml]   (a scene key; see docs/rules.md)`

Keys: `anchor`, `id`, `once`, `open`, `passages`, `start`, `title`

| File | What it is |
| --- | --- |
| [dialogues/hotel_plaque.yaml](../../assets/dialogues/hotel_plaque.yaml) | The plinth outside the hotel (scenes/hotel_showcase.yaml): it reads differently once you have taken the parcel for room 207 (quests/deliver_parcel.yaml). |
| [dialogues/receptionist.yaml](../../assets/dialogues/receptionist.yaml) | The hotel receptionist (hotel.yaml, in scenes/hotel_showcase.yaml): offers the parcel for room 207 (quests/deliver_parcel.yaml), says where to take it, and thanks you after. |

### garment

A piece of clothing fitted to the humanoid; outfits are picked with its `outfit` param.

Use: `place: {x: {asset: characters/humanoid.yaml, params: {outfit: dress}}}`

Keys: `chains`, `cover`, `cut`, `ease`, `layer`, `material`, `name`, `part`, `sole`, `thickness`, `type`, `weights`

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

### item

A spawned item: the body a source copies while the game runs, its mode and the data it carries.

Use: `source: {items: items/marble.yaml, rate: 1}   (an asset key; see geogen.flows)`

Keys: `body`, `data`, `gait`, `held`, `lifetime`, `mode`, `name`, `opens`, `radius`, `speed`, `stuck`, `tags`, `title`, `use`

| File | What it is |
| --- | --- |
| [items/carrot.yaml](../../assets/items/carrot.yaml) | Carrots from a farm plot (farm_plot.yaml): what harvesting one gives; carried, and sold at the farm's counter. |
| [items/carrot_seeds.yaml](../../assets/items/carrot_seeds.yaml) | Carrot seeds from the farm's seed machine: a packet (seed_packet.yaml) to carry to a plot and drop on it, which plants it. |
| [items/chocolate_bar.yaml](../../assets/items/chocolate_bar.yaml) | Chocolate bar from the vending machine (vending_machine.yaml): a physics item the player can carry off and put down anywhere. |
| [items/cola.yaml](../../assets/items/cola.yaml) | Cola from the vending machine (vending_machine.yaml): a physics item the player can carry off and put down anywhere. |
| [items/crisps.yaml](../../assets/items/crisps.yaml) | Crisps from the vending machine (vending_machine.yaml): a physics item the player can carry off and put down anywhere. |
| [items/dirty_plate.yaml](../../assets/items/dirty_plate.yaml) | The plate a meal leaves: washed up at a kitchen sink (kitchen_sink.yaml). |
| [items/key_room_207.yaml](../../assets/items/key_room_207.yaml) | The key to hotel room 207: its id is the room door's lock key (hotel guest doors lock with key_room_<n>), so carrying it lets L unlock that door. |
| [items/lemonade.yaml](../../assets/items/lemonade.yaml) | Lemonade from the vending machine (vending_machine.yaml): a physics item the player can carry off and put down anywhere. |
| [items/meal.yaml](../../assets/items/meal.yaml) | A cooked meal: made from raw food at a range or a microwave, eaten sitting at a table (chair.yaml), which leaves a dirty plate. |
| [items/parcel.yaml](../../assets/items/parcel.yaml) | A parcel that rides belts (conveyor.yaml) and is worth a point where it's taken in. |
| [items/pumpkin.yaml](../../assets/items/pumpkin.yaml) | Pumpkin from a farm plot (farm_plot.yaml): what harvesting one gives; carried, and sold at the farm's counter. |
| [items/pumpkin_seeds.yaml](../../assets/items/pumpkin_seeds.yaml) | Pumpkin seeds from the farm's seed machine: a packet (seed_packet.yaml) to carry to a plot and drop on it, which plants it. |
| [items/raider.yaml](../../assets/items/raider.yaml) | A raiding rover: walks a track at its own pace, takes 6 damage to stop, is worth 2 when it is and does 1 harm if it gets through. |
| [items/raw_food.yaml](../../assets/items/raw_food.yaml) | Raw food: what a household's fridge gives (fridge.yaml), and a range or a microwave takes to make a meal (geogen.101.5). |
| [items/room_key.yaml](../../assets/items/room_key.yaml) | A hotel room key: carried (give / take, has("room_key")), and what room_key.yaml gives whoever picks it up. |
| [items/rover.yaml](../../assets/items/rover.yaml) | A delivery rover that drives along walk tracks at its own pace. |
| [items/wheat.yaml](../../assets/items/wheat.yaml) | Wheat from a farm plot (farm_plot.yaml): what harvesting one gives; carried, and sold at the farm's counter. |
| [items/wheat_seeds.yaml](../../assets/items/wheat_seeds.yaml) | Wheat seeds from the farm's seed machine: a packet (seed_packet.yaml) to carry to a plot and drop on it, which plants it. |
| [items/wrapper.yaml](../../assets/items/wrapper.yaml) | A screwed-up wrapper: what eating something from a stall leaves in the hand (a scene's rule on use_item gives one); trashcan.yaml's `bin` interaction takes it. |

### motion

An NPC interaction animation: beats (poses, IK onto object contacts, gaze, root) that hold body channels by strength and say what an interruption does.

Use: `MotionRig(actor, scene).start('sit', seat='chair'); review: python -m geogen.motion_review`

Keys: `beats`, `category`, `demo`, `extends`, `holds`, `initial`, `layer`, `name`, `params`, `posture`, `priority`, `props`, `roles`, `summary`, `while`

| File | What it is |
| --- | --- |
| [motions/angry.yaml](../../assets/motions/angry.yaml) | Angry (geogen.103.17): fists clenched at the sides, leaning in, then the right fist raised and shaken `shakes` times. |
| [motions/arms_crossed.yaml](../../assets/motions/arms_crossed.yaml) | Arms crossed (geogen.103.17): an idle. |
| [motions/beckon.yaml](../../assets/motions/beckon.yaml) | Beckon (geogen.103.17): "come here". |
| [motions/bow.yaml](../../assets/motions/bow.yaml) | A bow with the right hand on the chest (geogen.103.5). |
| [motions/carry.yaml](../../assets/motions/carry.yaml) | Hold a taken thing where it is carried (geogen.103.4): the hand stays put in front of the hip, shaped by the thing's grip, whenever nothing else uses the hand (it only prefers it: start it with take and it holds on from the lift). |
| [motions/celebrate_jump.yaml](../../assets/motions/celebrate_jump.yaml) | Celebrate (geogen.103.17): a crouch, then a jump with both fists flung up, and a landing that bends the knees. |
| [motions/check_watch.yaml](../../assets/motions/check_watch.yaml) | Check the time (geogen.103.17): an idle. |
| [motions/cheer.yaml](../../assets/motions/cheer.yaml) | Cheer (geogen.103.17): both fists punched up overhead, pumped `pumps` times, with a bounce at the knees. |
| [motions/clap.yaml](../../assets/motions/clap.yaml) | Clap (geogen.103.17): both hands up in front of the chest, palms facing, meeting `claps` times, then down. |
| [motions/cry.yaml](../../assets/motions/cry.yaml) | Cry (geogen.103.17): the head drops into both hands, the shoulders heaving, `sobs` times; then the hands come down. |
| [motions/dance_arms_up.yaml](../../assets/motions/dance_arms_up.yaml) | Dance: hands up (geogen.103.17): a loop. |
| [motions/dance_bounce.yaml](../../assets/motions/dance_bounce.yaml) | Dance: bounce (geogen.103.17): a loop. |
| [motions/dance_sway.yaml](../../assets/motions/dance_sway.yaml) | Dance: sway (geogen.103.17): a loop. |
| [motions/drink.yaml](../../assets/motions/drink.yaml) | Drink (geogen.103.17): take a mug from the table in front by its handle, bring it to the mouth, tip it for a sip or two, and put it back where it stood. |
| [motions/eat.yaml](../../assets/motions/eat.yaml) | Eat from a plate, seated (geogen.103.5; the design's interruption example): bring food from the plate to the mouth (the hand pinched round an unseen fork), `bites` times. |
| [motions/facepalm.yaml](../../assets/motions/facepalm.yaml) | Facepalm (geogen.103.17): the head drops into the right palm, which covers the eyes; a slow shake of the head in it, then the hand goes down. |
| [motions/fist_pump.yaml](../../assets/motions/fist_pump.yaml) | Fist pump (geogen.103.17): "yes!" The right fist comes up beside the shoulder and is pulled down hard to the hip, with a dip at the knees. |
| [motions/glance.yaml](../../assets/motions/glance.yaml) | Look at something for a while, then look back (geogen.103.4). |
| [motions/hands_behind_back.yaml](../../assets/motions/hands_behind_back.yaml) | Hands behind the back (geogen.103.17): an idle, standing at ease. |
| [motions/hands_on_hips.yaml](../../assets/motions/hands_on_hips.yaml) | Hands on the hips (geogen.103.17): an idle. |
| [motions/inspect.yaml](../../assets/motions/inspect.yaml) | Inspect (geogen.103.17): take a thing from the table in front, hold it up and turn it over in the hand, looking at it, then put it back where it stood. |
| [motions/jumping_jack.yaml](../../assets/motions/jumping_jack.yaml) | Jumping jacks (geogen.103.12): hop out to a star, arms overhead, and back, `reps` times. |
| [motions/laugh.yaml](../../assets/motions/laugh.yaml) | Laugh (geogen.103.17): the head goes back and the chest shakes with it, a hand on the belly,. |
| [motions/look_around.yaml](../../assets/motions/look_around.yaml) | Look around (geogen.103.17): an idle. |
| [motions/nod.yaml](../../assets/motions/nod.yaml) | Nod yes (geogen.103.17): the head dips and comes back up `nods` times. |
| [motions/point.yaml](../../assets/motions/point.yaml) | Point (geogen.103.17): the right arm out ahead, index finger pointing, the head turned along it; hold, then down. |
| [motions/press.yaml](../../assets/motions/press.yaml) | Press a button: reach a fingertip to it, push, take the hand back (geogen.103.4). |
| [motions/rub_hands.yaml](../../assets/motions/rub_hands.yaml) | Rub the hands (geogen.103.17): an idle fidget. |
| [motions/salute.yaml](../../assets/motions/salute.yaml) | Salute (geogen.103.17): the right hand comes up flat, fingertips to the brow; hold, then it snaps down. |
| [motions/scratch_head.yaml](../../assets/motions/scratch_head.yaml) | Scratch the head (geogen.103.17): an idle, a puzzled moment. |
| [motions/set_down.yaml](../../assets/motions/set_down.yaml) | Put a carried thing down (geogen.103.4): take's mirror. |
| [motions/shake_head.yaml](../../assets/motions/shake_head.yaml) | Shake the head no (geogen.103.17): side to side `shakes` times. |
| [motions/shift_weight.yaml](../../assets/motions/shift_weight.yaml) | Shift the weight from foot to foot (geogen.103.17): an idle. |
| [motions/shiver.yaml](../../assets/motions/shiver.yaml) | Shiver (geogen.103.17): cold. |
| [motions/shrug.yaml](../../assets/motions/shrug.yaml) | Shrug (geogen.103.17): shoulders up, forearms out to the sides with the palms up, the head tipped; hold a moment, then let it all drop. |
| [motions/sigh.yaml](../../assets/motions/sigh.yaml) | Sigh (geogen.103.17): an idle. |
| [motions/sit.yaml](../../assets/motions/sit.yaml) | Sit on a seat and get up again (geogen.103; the design's worked example 1). |
| [motions/sit_ground.yaml](../../assets/motions/sit_ground.yaml) | Sit on the ground (geogen.103.17): down onto the floor with the knees up and the arms round them, a while there, and back up. |
| [motions/squat.yaml](../../assets/motions/squat.yaml) | A squat (geogen.103.12): sit back and down over planted feet with the arms reaching forward, then stand, `reps` times. |
| [motions/stagger.yaml](../../assets/motions/stagger.yaml) | A stagger (geogen.103.5): the reaction a break inertializes into (a shove, a near fall). |
| [motions/stretch.yaml](../../assets/motions/stretch.yaml) | Stretch (geogen.103.17): an idle. |
| [motions/surprised.yaml](../../assets/motions/surprised.yaml) | Surprised (geogen.103.17): a start: the hands fly up in front of the chest, palms out, the body rocking back; then it relaxes. |
| [motions/take.yaml](../../assets/motions/take.yaml) | Take a thing in one hand (geogen.103.4; the design's worked example 2): reach over it with the hand open, close the hand on the grip the thing offers (a plate's rim: rim), lift it to where it is carried. |
| [motions/tap_foot.yaml](../../assets/motions/tap_foot.yaml) | Tap the foot (geogen.103.17): an idle, impatient. |
| [motions/think.yaml](../../assets/motions/think.yaml) | Think (geogen.103.17): the right hand to the chin, the left forearm across the waist under the right elbow, the head tipped and the eyes up; hold, then both arms down. |
| [motions/thumbs_up.yaml](../../assets/motions/thumbs_up.yaml) | Thumbs up (geogen.103.17): the right fist comes out in front of the chest, thumb up, with a little nod; hold, then down. |
| [motions/watch.yaml](../../assets/motions/watch.yaml) | Watch something for a while (geogen.103.4): the gaze on it and nothing else. |
| [motions/wave.yaml](../../assets/motions/wave.yaml) | Wave hello with the right hand (geogen.103.5): an overlay. |
| [motions/yawn.yaml](../../assets/motions/yawn.yaml) | Yawn (geogen.103.17): an idle. |

### motion_scenario

A scripted motion run (stage, person, timed starts and interruptions, camera) rendered to video, filmstrip and timeline for review.

Use: `python -m geogen.motion_review sit_on_chair -o /tmp/review`

Keys: `actor`, `camera`, `description`, `duration`, `fps`, `name`, `scene`, `script`, `until`

| File | What it is |
| --- | --- |
| [motions/scenarios/bow_wave.yaml](../../assets/motions/scenarios/bow_wave.yaml) | A bow; a wave borrows the right hand from it (geogen.103.5): the bow pauses, and picks up from. |
| [motions/scenarios/drink.yaml](../../assets/motions/scenarios/drink.yaml) | Take a mug from the table in front, drink, put it back (geogen.103.17): the cup sits near the table's edge at the right hand, its handle turned to it. |
| [motions/scenarios/eat_glance.yaml](../../assets/motions/scenarios/eat_glance.yaml) | Eating at the table when something falls in the hall, look only (the design's `thud: {look: 2}`). |
| [motions/scenarios/eat_interrupted.yaml](../../assets/motions/scenarios/eat_interrupted.yaml) | Eating at the table when something falls in the hall, with an interrupt (the design's. |
| [motions/scenarios/eat_wave.yaml](../../assets/motions/scenarios/eat_wave.yaml) | A wave mid-meal (geogen.103.5): the wave borrows the right hand, eat waits where it was, and picks up when the hand comes back (chewing: chew on, its `resume`), not from the first bite. |
| [motions/scenarios/emotes.yaml](../../assets/motions/scenarios/emotes.yaml) | Emotes in a row (geogen.103.17): a clap with a nod over it (the nod only takes the gaze), a cheer, a shrug, a thumbs up, and down onto the floor and back up. |
| [motions/scenarios/idles.yaml](../../assets/motions/scenarios/idles.yaml) | Idles together (geogen.103.17): arms crossed while looking around (the look only takes the gaze), then a foot tapping with the hands on the hips, then the weight shifting. |
| [motions/scenarios/inspect.yaml](../../assets/motions/scenarios/inspect.yaml) | Pick up a mug from the table in front, look it over, put it back (geogen.103.17). |
| [motions/scenarios/jumping_jacks.yaml](../../assets/motions/scenarios/jumping_jacks.yaml) | Jumping jacks on a rug (geogen.103.12): a body-only motion, four repetitions. |
| [motions/scenarios/press_tv.yaml](../../assets/motions/scenarios/press_tv.yaml) | Stand beside a TV on a table and press its power button (bending to reach it, the head clear of the screen). |
| [motions/scenarios/sit_interrupted.yaml](../../assets/motions/scenarios/sit_interrupted.yaml) | Sit down; get interrupted while seated (the seated beat yields: she rises through `rise`). |
| [motions/scenarios/sit_on_chair.yaml](../../assets/motions/scenarios/sit_on_chair.yaml) | Walk up to a chair (the actor starts at its approach point), sit for two seconds, stand up. |
| [motions/scenarios/sit_watch_tv.yaml](../../assets/motions/scenarios/sit_watch_tv.yaml) | Two motions at once: sitting holds the body, a glance at the TV holds only the gaze. |
| [motions/scenarios/squat.yaml](../../assets/motions/scenarios/squat.yaml) | Squats on a rug (geogen.103.12): a body-only motion, three repetitions. |
| [motions/scenarios/squat_shoved.yaml](../../assets/motions/scenarios/squat_shoved.yaml) | A shove mid-squat (geogen.103.5): a break drops the squat with no exit and the body inertializes into a stagger, then stands. |
| [motions/scenarios/take_carry_set_down.yaml](../../assets/motions/scenarios/take_carry_set_down.yaml) | Take a plate from the near edge of a table, carry it for a moment, set it down in the middle (geogen.103.4): three motions in turn on the right hand, the plate going with it from the grasp to the release. |
| [motions/scenarios/watch_tv.yaml](../../assets/motions/scenarios/watch_tv.yaml) | Sit and watch the TV; a glance at the lamp takes the gaze for a moment and gives it back (watch only prefers the gaze; glance claims it). |

### npc

An NPC: body, needs, preferences, activities and daily routine; the runtime interprets it.

Use: `place: {resident: {npc: npcs/resident.yaml, on: house.floor, home: house.floor, seed: 1}}`

Keys: `activities`, `affordance_tags`, `attention`, `body`, `carry`, `flags`, `home_margin`, `memory`, `needs`, `offers`, `planner`, `preferences`, `radius`, `reach`, `reacts`, `relations`, `routine`, `scoring`, `seed`, `senses`, `speed`, `turn_speed`, `wander`, `wants`

| File | What it is |
| --- | --- |
| [npcs/cat.yaml](../../assets/npcs/cat.yaml) | A house cat: it wanders its home and stands about, tail swishing. |
| [npcs/chef.yaml](../../assets/npcs/chef.yaml) | A cook: works the range and the prep counter (affordances tagged `cook`), rests on a stool or chair now and then, looks into the dining room every few minutes and goes home for the night. |
| [npcs/diner.yaml](../../assets/npcs/diner.yaml) | Out for a meal or a drink: sits at a table or in a booth, or at the bar, and moves between them now and then. |
| [npcs/hotel_guest.yaml](../../assets/npcs/hotel_guest.yaml) | A hotel guest in their room: sleeps the night through in its bed, sits, looks out of the window, wanders the room. |
| [npcs/hotel_staff.yaml](../../assets/npcs/hotel_staff.yaml) | Hotel front-of-house staff: works the reception desk (its staff-side `stand` affordance, tagged `staff`), takes breaks on the lobby seating and walks the lobby now and then. |
| [npcs/householder.yaml](../../assets/npcs/householder.yaml) | Someone who lives in a home and keeps themselves going: gets hungry, tired and grubby, and sees to it with whatever the home has. |
| [npcs/park_visitor.yaml](../../assets/npcs/park_visitor.yaml) | Someone spending time in the park: walks the gravel paths, sits on benches and at the picnic tables, stops on the bandstand to look around. |
| [npcs/pedestrian.yaml](../../assets/npcs/pedestrian.yaml) | Somebody out and about: strolls the sidewalks, stops to look around, rests on benches. |
| [npcs/resident.yaml](../../assets/npcs/resident.yaml) | A person who lives in a house: rests in chairs, looks out of the windows, browses the shelves and steps out of the front door now and then. |

### npc_actions

The actions NPCs run at affordances, as steps (go_to, face, pose, wait, use, play, vanish).

Use: `referenced by affordances: {type: sit, action: sit}`

Keys: `face`, `find`, `go_to`, `pick_up`, `play`, `pose`, `put_down`, `set_out`, `show`, `use`, `use_item`, `vanish`, `wait`

| File | What it is |
| --- | --- |
| [npcs/actions.yaml](../../assets/npcs/actions.yaml) | What NPCs do, as data. |

### npc_stimuli

What NPCs with senses can notice: bus events and how far they carry.

Use: `an NPC's reacts: {thud: {interrupt: true}}`

Keys: `stimuli`

| File | What it is |
| --- | --- |
| [npcs/stimuli.yaml](../../assets/npcs/stimuli.yaml) | What can be noticed, as data: each stimulus is a kind of event on the world's bus (runtime/godot/scripts/events.gd) with how far it carries. |

### package

A content package: state, rules and quests that span the scenes it names.

Use: `packages: [name]   (in runtime_scenes.yaml; see docs/rules.md, Content packages)`

Keys: `bind`, `description`, `id`, `quests`, `rules`, `scenes`, `state`, `title`, `wire`

| File | What it is |
| --- | --- |
| [packages/jetty_trip.yaml](../../assets/packages/jetty_trip.yaml) | A trip from the hub to the island's jetty and back: a quest across two scenes (docs/quests.md, "Ordered, across scenes"). |

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
| [quests/catch_train.yaml](../../assets/quests/catch_train.yaml) | Catch the train: the "timed, can fail" quest. |
| [quests/climb_lighthouse.yaml](../../assets/quests/climb_lighthouse.yaml) | Climb to the lighthouse gallery, for an NPC: whoever a scene gives it to (an NPC placement's `quests:`) walks up the spiral stair and out onto the gallery deck. |
| [quests/deliver_parcel.yaml](../../assets/quests/deliver_parcel.yaml) | Fetch and deliver (geogen.15.7): the hotel receptionist hands over a parcel and the key to room 207 (dialogues/receptionist.yaml, in scenes/hotel_showcase.yaml); let yourself in with the key (L unlocks the door) and give the parcel to the guest. |
| [quests/farm_start.yaml](../../assets/quests/farm_start.yaml) | The farm's starter quest: the whole loop once. |
| [quests/jetty_trip.yaml](../../assets/quests/jetty_trip.yaml) | The jetty trip (packages/jetty_trip.yaml gives it when the player takes the hub's portal to the island). |
| [quests/lighthouse_gallery.yaml](../../assets/quests/lighthouse_gallery.yaml) | Climb to the lighthouse gallery: the "go there" quest, one place to reach. |
| [quests/lights_out.yaml](../../assets/quests/lights_out.yaml) | Switch off every light in the cottage: the "change the world" quest. |
| [quests/yard_dash.yaml](../../assets/quests/yard_dash.yaml) | The dash: stepping onto the start square gives it, and a rule of the quest yard fails it six seconds later unless the finish square was reached. |
| [quests/yard_lamps.yaml](../../assets/quests/yard_lamps.yaml) | Lamps out: the steward of the quest yard gives it when greeted. |
| [quests/yard_round.yaml](../../assets/quests/yard_round.yaml) | The round: visit the three numbered plinths of the quest yard in order. |

### scene_catalogue

The Godot runtime's scenes (showcase | test) and its default.

Use: `python -m geogen.main --export-catalogue`

Keys: `default`, `packages`, `scenes`

| File | What it is |
| --- | --- |
| [runtime_scenes.yaml](../../assets/runtime_scenes.yaml) | The scenes the Godot runtime offers. `--export-catalogue` exports them into runtime/godot/generated/ and writes generated/catalogue.json; with no. |

### skeleton

The humanoid skeleton: params, bone heads, landmarks and body channels.

Use: `body: {skeleton: skeletons/humanoid.yaml}`

Keys: `bones`, `channels`, `derived`, `group_scale`, `hand_shapes`, `landmarks`, `name`, `params`, `profile`

| File | What it is |
| --- | --- |
| [skeletons/humanoid.yaml](../../assets/skeletons/humanoid.yaml) | Humanoid skeleton: Godot SkeletonProfileHumanoid bones (humanoid_profile.json), placed by body proportions. |
| [skeletons/quadruped.yaml](../../assets/skeletons/quadruped.yaml) | Four-legged skeleton: a spine from the hips forward to the head, a leg at each corner, a tail, two ears and two eyes. |

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

### waves

Waves a source sends in turn: each a list of item groups (count, gap) after a delay.

Use: `source: {waves: waves/raid.yaml}   (an asset key; see geogen.flows)`

Keys: `auto`, `waves`

| File | What it is |
| --- | --- |
| [waves/raid.yaml](../../assets/waves/raid.yaml) | Two short waves of raiders, for flow_test: the second follows the first by itself. |

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
| `kitchen` | Kitchen-diner: fridge, range and sink along the walls, and a table with chairs round it. | chair, fridge, kitchen_range, kitchen_sink, planter, table |
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
| `bread_crumb` | painted_wall |
| `bread_crust` | painted_wall |
| `brick` | brick |
| `brick_paving` | brick |
| `car_glass` | metal |
| `car_paint` | painted_wall |
| `car_trim` | painted_wall |
| `cardboard` | painted_wall |
| `carpet` | cut_pile_carpet |
| `carrot_root` | painted_wall |
| `ceiling_white` | painted_wall |
| `ceramic_white` | painted_wall |
| `chrome` | metal |
| `cloth` | fabric |
| `concrete` | concrete |
| `crop_leaf` | painted_wall |
| `denim` | fabric |
| `dirt` | dirt |
| `door_paint` | painted_wall |
| `fabric_linen` | fabric |
| `fabric_velvet` | fabric |
| `floor` | concrete |
| `foliage_green` | painted_wall |
| `foliage_palm` | painted_wall |
| `foliage_pine` | painted_wall |
| `fur` | hair |
| `glass` | metal |
| `glow_accent` | painted_wall |
| `grass` | grass |
| `gravel` | dirt |
| `hair` | hair |
| `hardwood_floor` | hardwood_floor |
| `headlight` | painted_wall |
| `jam` | painted_wall |
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
| `peanut_butter` | painted_wall |
| `plastic_black` | painted_wall |
| `play_paint` | painted_wall |
| `play_yellow` | painted_wall |
| `portal_glow` | painted_wall |
| `pumpkin_skin` | painted_wall |
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
| `wheat_straw` | painted_wall |
| `wood` | wood |
| `wood_dark` | wood |
