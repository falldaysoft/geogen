# Scenes

Composed scenes: view one with `python -m geogen.main -s NAME`, nest one with `scene: <file>`. `--inspect` prints the resolved tree.

| Scene | What it is | Runtime |
| --- | --- | --- |
| [dining_set](#dining_set) | Dining set: table with 4 chairs. | test |
| [archetype_gallery](#archetype_gallery) | Character archetypes (assets/characters/archetypes/): one row per archetype, eight seeded people each (a path scatter draws a seed per copy). |  |
| [city](#city) | The busy city showcase: a 5 x 4 grid with avenues through the middle, the hotel, an apartment block and an office tower as landmarks, two parks, shops along the avenues and houses on the side streets. | showcase |
| [clothing_fit](#clothing_fit) | Clothing-fit stress test: the exaggerated `hourglass` body (big bust, wide hips, narrow waist) in every outfit. |  |
| [cottage](#cottage) | Brick cottage: house shell with a front door and windows on every side. | showcase |
| [crossroads](#crossroads) | A small street grid for traffic: 2 x 2 blocks (parks, no buildings) with an avenue, parked cars, moving traffic (assets/traffic/town.yaml) and a few pedestrians (npcs/pedestrian.yaml). | test |
| [hotel_floor_furnished](#hotel_floor_furnished) | A generated hotel guest floor with every room furnished from its archetype. |  |
| [hotel_room](#hotel_room) | Furnished hotel suite (hand-placed; the furnishing solver, geogen-3cc.6, will do this from room archetypes). | test |
| [hotel_room_auto](#hotel_room_auto) | The hotel suite furnished automatically from its rooms' archetypes (assets/room_types/*.yaml); compare scenes/hotel_room.yaml (by hand). | test |
| [hotel_showcase](#hotel_showcase) | The showcase hotel (assets/hotel.yaml: lobby, three furnished guest floors, stairs and a working lift) on a short street: a paved forecourt with lamps, benches and planters, taxis waiting at the kerb, a receptionist at the desk, guests upstairs, people strolling past, and a travel door back to the hub. | showcase |
| [house_plot](#house_plot) | House plot - a yard with a house. |  |
| [hub](#hub) | The hub: a round plaza with one labelled portal per showcase scene, spread evenly round the far (-Z) side (angles from -Z: +-84, 21 degrees apart, for nine portals; re-space them when a showcase is added). | showcase |
| [humanoid_lineup](#humanoid_lineup) | Humanoid bodies for review: feminine, neutral and masculine presets next to a chair and a door for scale. | test |
| [humanoid_walk](#humanoid_walk) | One feminine humanoid for clip review. |  |
| [island](#island) | The island showcase: a small island in the sea. | showcase |
| [level_crossing](#level_crossing) | A railway crossing a road: a two-lane road (a route), a line with two stations, barriers where they cross, traffic on the road and a commuter train. | test |
| [lighthouse_keepers](#lighthouse_keepers) | NPCs on a spiral stair (test). |  |
| [main_street](#main_street) | The main street showcase: one lively high street (the east-west avenue of a small 3 x 2 grid). | showcase |
| [park](#park) | The park showcase: a railed park entered by a gate at the south end. | showcase |
| [people_gallery](#people_gallery) | A row of humanoid variants for review: body types stacked on the gender presets (params: {preset: [feminine, curvy]}), proportions, hair styles and colours, skin tones and outfits. |  |
| [quest_yard](#quest_yard) | A plaza for trying quests and the quest HUD (docs/quests.md). | showcase |
| [rail_loop_test](#rail_loop_test) | Test scene for city exits and a loop railway: a 2 x 2 grid whose avenues run on out of town across a railway looping round it (four level crossings), two stations on the straights and two trains. | test |
| [restaurant](#restaurant) | The restaurant showcase: assets/restaurant_building.yaml on a street corner. | showcase |
| [room_with_furniture](#room_with_furniture) | Room with furniture - demonstrates nested scene composition. |  |
| [seated_fit](#seated_fit) | Seats for clothing/pose fit review: an armchair, a dining chair and a sofa in a row (clipping QA, geogen/clipping.py). |  |
| [street](#street) | Street scene - hierarchical composition. |  |
| [street_side](#street_side) | One side of a street - sidewalk with house plots behind it. |  |
| [terrain_test](#terrain_test) | Terrain review: an island with palms kept to gentle ground above the beach, rocks on the steep slopes and a flattened pad for a building. |  |
| [town](#town) | A small town district: a 3 x 2 street grid with a commercial avenue down the middle (shop rows with flats above, the odd office), a hotel and an apartment block as landmarks, a park, and houses on the side streets. | showcase |
| [trees](#trees) | Tree demo scene. |  |
| [vehicle_lineup](#vehicle_lineup) | Vehicles for review: the car presets in a few paints, a van and a bus. |  |

## dining_set

`scene: dining_set.yaml` · [source](../../assets/dining_set.yaml)

Dining set: table with 4 chairs
Chairs are attached at the table's seat attachment points

**Places**: `table: {asset: table.yaml}`; `chairs: {asset: chair.yaml}`

## archetype_gallery

`scene: scenes/archetype_gallery.yaml` · [source](../../assets/scenes/archetype_gallery.yaml)

Character archetypes (assets/characters/archetypes/): one row per archetype,
eight seeded people each (a path scatter draws a seed per copy). Faces +Z.

**Places**: `young_woman: {archetype: characters/archetypes/young_woman.yaml}`; `young_man: {archetype: characters/archetypes/young_man.yaml}`; `older_woman: {archetype: characters/archetypes/older_woman.yaml}`; `older_man: {archetype: characters/archetypes/older_man.yaml}`; `office_worker: {archetype: characters/archetypes/office_worker.yaml}`; `hotel_staff: {archetype: characters/archetypes/hotel_staff.yaml}`

## city

`scene: scenes/city.yaml` · [source](../../assets/scenes/city.yaml)

The busy city showcase: a 5 x 4 grid with avenues through the middle, the
hotel, an apartment block and an office tower as landmarks, two parks,
shops along the avenues and houses on the side streets. The avenues run on
out of town (city exits) and cross a railway that loops round the whole
district, with two stations and two trains on the timetable. Dense traffic
with signals, parked cars and a crowd on the pavements. Streamed export.

**Places**: `hub_portal: {asset: portal.yaml}`; `commuter_a: {train: trains/commuter.yaml}`; `commuter_b: {train: trains/commuter.yaml}`; `traffic: {traffic: traffic/town.yaml}`; `pedestrians: {npc: npcs/pedestrian.yaml}`

**Slots**: hub_portal

**Spawns**: from_hub, start

**Features**: city, railways

## clothing_fit

`scene: scenes/clothing_fit.yaml` · [source](../../assets/scenes/clothing_fit.yaml)

Clothing-fit stress test: the exaggerated `hourglass` body (big bust, wide
hips, narrow waist) in every outfit. Review with

```yaml
python -m geogen.main -s clothing_fit -r out.png --views front,iso --zoom 3.2
```

**Places**: `nude: {asset: characters/humanoid.yaml}`; `casual: {asset: characters/humanoid.yaml}`; `jeans_tee: {asset: characters/humanoid.yaml}`; `dress: {asset: characters/humanoid.yaml}`; `skirt_sweater: {asset: characters/humanoid.yaml}`; `shorts_tee: {asset: characters/humanoid.yaml}`; `casual_f: {asset: characters/humanoid.yaml}`

**Slots**: s0, s1, s2, s3, s4, s5, s6

## cottage

`scene: scenes/cottage.yaml` · [source](../../assets/scenes/cottage.yaml)

Brick cottage: house shell with a front door and windows on every side.
Each window/door cuts its own opening into the wall it is placed on.

| Param | Default | Notes |
| --- | --- | --- |
| `hub_portal` | `1` | the portal back to the hub (0 where the cottage is part of something bigger) |
| `gallery` | `1` | the people gallery on the front lawn (0 for a plain cottage) |
| `quest` | `1` | the resident's "lights out" quest (0 where the cottage is part of something bigger) |

**Places**: `house: {asset: house_peaked.yaml}`; `door: {asset: door.yaml}`; `window_front_left: {asset: window.yaml}`; `window_front_right: {asset: window.yaml}`; `window_left: {asset: window.yaml}`; `window_right: {asset: window.yaml}`; `window_back_left: {asset: window.yaml}`; `window_back_right: {asset: window.yaml}`; `dining: {scene: scenes/dining_set.yaml}`; `bookshelf: {asset: bookshelf.yaml}`; `armchair: {asset: armchair.yaml}`; `dining_light: {asset: ceiling_light.yaml}`; `sitting_light: {asset: ceiling_light.yaml}`; `reading_lamp: {asset: floor_lamp.yaml}`; `dining_switch: {asset: light_switch.yaml}`; `sitting_switch: {asset: light_switch.yaml}`; `lamp_switch: {asset: light_switch.yaml}`; `resident: {npc: npcs/resident.yaml}`; `gallery: {scene: scenes/people_gallery.yaml}`; `hub_portal: {asset: portal.yaml}`

**Slots**: lawn, west_lawn

**Spawns**: from_hub, front_path

## crossroads

`scene: scenes/crossroads.yaml` · [source](../../assets/scenes/crossroads.yaml)

A small street grid for traffic: 2 x 2 blocks (parks, no buildings) with an
avenue, parked cars, moving traffic (assets/traffic/town.yaml) and a few
pedestrians (npcs/pedestrian.yaml). Used by the
traffic runtime tests (tests/test_traffic_runtime.py); quick to export.

**Places**: `traffic: {traffic: traffic/town.yaml}`; `pedestrians: {npc: npcs/pedestrian.yaml}`

**Spawns**: start

**Features**: city

## hotel_floor_furnished

`scene: scenes/hotel_floor_furnished.yaml` · [source](../../assets/scenes/hotel_floor_furnished.yaml)

A generated hotel guest floor with every room furnished from its archetype.

**Places**: `floor: {asset: hotel_floor.yaml}`

**Spawns**: lift

## hotel_room

`scene: scenes/hotel_room.yaml` · [source](../../assets/scenes/hotel_room.yaml)

Furnished hotel suite (hand-placed; the furnishing solver, geogen-3cc.6,
will do this from room archetypes). Positions are metres on each room's
floor surface: u runs west -> east, v south -> north. Furniture keeps
clear of the door swing arcs recorded in the rooms' door_swings.

**Places**: `suite: {asset: hotel_suite.yaml}`; `bed: {asset: bed.yaml}`; `nightstand_south: {asset: nightstand.yaml}`; `nightstand_north: {asset: nightstand.yaml}`; `lamp_south: {asset: table_lamp.yaml}`; `lamp_north: {asset: table_lamp.yaml}`; `art: {asset: wall_art.yaml}`; `rug: {asset: rug.yaml}`; `tv_console: {asset: tv_console.yaml}`; `tv: {asset: tv.yaml}`; `wardrobe: {asset: wardrobe.yaml}`; `floor_lamp: {asset: floor_lamp.yaml}`; `curtains: {asset: curtains.yaml}`; `shower: {asset: shower.yaml}`; `toilet: {asset: toilet.yaml}`; `vanity: {asset: vanity.yaml}`; `mirror: {asset: wall_mirror.yaml}`; `towel_rail: {asset: towel_rail.yaml}`

**Spawns**: entry

## hotel_room_auto

`scene: scenes/hotel_room_auto.yaml` · [source](../../assets/scenes/hotel_room_auto.yaml)

The hotel suite furnished automatically from its rooms' archetypes
(assets/room_types/*.yaml); compare scenes/hotel_room.yaml (by hand).

**Places**: `suite: {asset: hotel_suite.yaml}`

**Spawns**: entry

## hotel_showcase

`scene: scenes/hotel_showcase.yaml` · [source](../../assets/scenes/hotel_showcase.yaml)

The showcase hotel (assets/hotel.yaml: lobby, three furnished guest floors,
stairs and a working lift) on a short street: a paved forecourt with lamps,
benches and planters, taxis waiting at the kerb, a receptionist at the desk,
guests upstairs, people strolling past, and a travel door back to the hub.

The hotel's street door is on its south side (-Z, entrance_spawn at z = -10).

**Places**: `hotel: {asset: hotel.yaml}`; `forecourt: {asset: sidewalk.yaml}`; `road: {asset: road.yaml}`; `far_walk: {asset: sidewalk.yaml}`; `hub_door: {asset: travel_door.yaml}`; `taxi_1: {asset: vehicles/car.yaml}`; `taxi_2: {asset: vehicles/car.yaml}`; `lamp_fw: {asset: street_lamp.yaml}`; `lamp_fe: {asset: street_lamp.yaml}`; `lamp_aw: {asset: street_lamp.yaml}`; `lamp_ae: {asset: street_lamp.yaml}`; `bench_w: {asset: bench.yaml}`; `bench_e: {asset: bench.yaml}`; `planter_w: {asset: planter.yaml}`; `planter_e: {asset: planter.yaml}`; `tree_1: {asset: maple_tree.yaml}`; `tree_2: {asset: maple_tree.yaml}`; `tree_3: {asset: maple_tree.yaml}`; `tree_4: {asset: maple_tree.yaml}`; `receptionist: {npc: npcs/hotel_staff.yaml}`; `guest_101: {npc: npcs/hotel_guest.yaml}`; `guest_207: {npc: npcs/hotel_guest.yaml}`; `pedestrians: {npc: npcs/pedestrian.yaml}`

**Slots**: bench_e, bench_w, far_walk, forecourt, front_desk, guest_101, guest_207, hub_door, lamp_ae, lamp_aw, lamp_fe, lamp_fw, planter_e, planter_w, road, taxi_1, taxi_2, tree_1, tree_2, tree_3, tree_4

**Spawns**: from_hub

## house_plot

`scene: scenes/house_plot.yaml` · [source](../../assets/scenes/house_plot.yaml)

House plot - a yard with a house
The plot is oriented with "front" facing +X (toward the street/road)
Contains a yard (ground) and a house

**Places**: `house: {scene: scenes/cottage.yaml}`

**Slots**: house

## hub

`scene: scenes/hub.yaml` · [source](../../assets/scenes/hub.yaml)

The hub: a round plaza with one labelled portal per showcase scene, spread
evenly round the far (-Z) side (angles from -Z: +-84, 21 degrees apart, for nine portals;
re-space them when a showcase is added). Each portal lands at its scene's
`from_hub` spawn, and every showcase has a portal or travel door back here
that lands in front of its own portal (`from_<name>`). It's the runtime
catalogue's default scene; tests/test_hub.py checks the round trips.

**Places**: `plaza: {asset: plaza.yaml}`; `name_plinth: {asset: info_plinth.yaml}`; `portal_cottage: {asset: portal.yaml}`; `portal_hotel: {asset: portal.yaml}`; `portal_restaurant: {asset: portal.yaml}`; `portal_main_street: {asset: portal.yaml}`; `portal_quests: {asset: portal.yaml}`; `portal_city: {asset: portal.yaml}`; `portal_town: {asset: portal.yaml}`; `portal_park: {asset: portal.yaml}`; `portal_island: {asset: portal.yaml}`; `signpost: {asset: signpost.yaml}`; `lamp_e: {asset: street_lamp.yaml}`; `lamp_w: {asset: street_lamp.yaml}`; `lamp_ne: {asset: street_lamp.yaml}`; `lamp_nw: {asset: street_lamp.yaml}`; `bench_e: {asset: bench.yaml}`; `bench_w: {asset: bench.yaml}`; `planter_e: {asset: planter.yaml}`; `planter_w: {asset: planter.yaml}`; `planter_se: {asset: planter.yaml}`; `planter_ne: {asset: planter.yaml}`; `planter_nw: {asset: planter.yaml}`

**Slots**: bench_e, bench_w, centre, lamp_e, lamp_ne, lamp_nw, lamp_w, planter_e, planter_ne, planter_nw, planter_se, planter_w, portal_city, portal_cottage, portal_hotel, portal_island, portal_main_street, portal_park, portal_quests, portal_restaurant, portal_town, sign

**Spawns**: from_city, from_cottage, from_hotel, from_island, from_main_street, from_park, from_quests, from_restaurant, from_town, start

## humanoid_lineup

`scene: scenes/humanoid_lineup.yaml` · [source](../../assets/scenes/humanoid_lineup.yaml)

Humanoid bodies for review: feminine, neutral and masculine presets next to
a chair and a door for scale.

**Places**: `feminine: {asset: characters/humanoid.yaml}`; `neutral: {asset: characters/humanoid.yaml}`; `masculine: {asset: characters/humanoid.yaml}`; `chair: {asset: chair.yaml}`; `door: {asset: door.yaml}`

**Slots**: a, b, c, chair, door

## humanoid_walk

`scene: scenes/humanoid_walk.yaml` · [source](../../assets/scenes/humanoid_walk.yaml)

One feminine humanoid for clip review:

```yaml
python -m geogen.main -s humanoid_walk -r strip.png --clip walk --filmstrip 8 --view side
```

**Places**: `person: {asset: characters/humanoid.yaml}`

## island

`scene: scenes/island.yaml` · [source](../../assets/scenes/island.yaml)

The island showcase: a small island in the sea. You arrive by the hub portal
on the south beach beside a timber jetty with a rowing boat at its landing;
gravel paths climb to two cottages (the western one has a resident) and on
along the ridge to the lighthouse on the eastern headland, which you can
climb to the gallery. Palms line the beaches, pines cover the hill, rocks
sit on the steep ground. The ground is assets/island_grounds.yaml (its pads
and paths are in this scene's frame; heights below come from it).

**Places**: `grounds: {asset: island_grounds.yaml}`; `jetty: {asset: jetty.yaml}`; `boat: {asset: rowing_boat.yaml}`; `cottage_a: {scene: scenes/cottage.yaml}`; `cottage_b: {scene: scenes/cottage.yaml}`; `lighthouse: {asset: lighthouse.yaml}`; `bench_beach: {asset: bench.yaml}`; `hub_portal: {asset: portal.yaml}`; `palms: {asset: palm_tree.yaml}`; `pines: {asset: pine_tree.yaml}`; `rocks: {asset: rock_large.yaml}`

**Slots**: bench_beach, boat, cottage_a, cottage_b, hub_portal, jetty, lighthouse

**Spawns**: from_hub, start

## level_crossing

`scene: scenes/level_crossing.yaml` · [source](../../assets/scenes/level_crossing.yaml)

A railway crossing a road: a two-lane road (a route), a line with two
stations, barriers where they cross, traffic on the road and a commuter train.
Trains lower the barriers (runtime/godot/scripts/train.gd); cars wait for them.

**Places**: `traffic: {traffic: traffic/town.yaml}`; `commuter: {train: trains/commuter.yaml}`

**Spawns**: start

**Features**: routes, railways

## lighthouse_keepers

`scene: scenes/lighthouse_keepers.yaml` · [source](../../assets/scenes/lighthouse_keepers.yaml)

NPCs on a spiral stair (test). The lighthouse with two keepers whose homes
are at the other end of the stair from where they start: the keeper starts
on the ground and lives on the gallery deck, so their first walk climbs the
stair; the lookout starts on the gallery and lives on the ground in front of
the door, so theirs comes down it.

**Places**: `lighthouse: {asset: lighthouse.yaml}`; `keeper: {npc: npcs/resident.yaml}`; `lookout: {npc: npcs/resident.yaml}`

**Slots**: keeper, lookout

**Spawns**: start

## main_street

`scene: scenes/main_street.yaml` · [source](../../assets/scenes/main_street.yaml)

The main street showcase: one lively high street (the east-west avenue of a
small 3 x 2 grid). Shop rows line it on both sides with furnished shops
(a cafe, a grocer, a bookshop, a clothes shop), fascia signs and awnings;
the side streets have houses. Parked cars, traffic with signals where the
avenue crosses the side streets, a bus shelter, pedestrians, and a portal
back to the hub.

**Places**: `bus_shelter: {asset: bus_shelter.yaml}`; `hub_portal: {asset: portal.yaml}`; `traffic: {traffic: traffic/town.yaml}`; `pedestrians: {npc: npcs/pedestrian.yaml}`

**Slots**: bus_stop, hub_portal

**Spawns**: from_hub, start

**Features**: city

## park

`scene: scenes/park.yaml` · [source](../../assets/scenes/park.yaml)

The park showcase: a railed park entered by a gate at the south end. Gravel
paths loop round a pond with a fountain and lead to a playground, a
bandstand and a picnic area; benches, lamps, bins and trees throughout, and
visitors who keep to the paths and use the benches and picnic tables.
(The grounds, paths and railings are assets/park_grounds.yaml.)

**Places**: `grounds: {asset: park_grounds.yaml}`; `gate: {asset: park_gate.yaml}`; `pond: {asset: pond.yaml}`; `playground: {asset: playground.yaml}`; `bandstand: {asset: bandstand.yaml}`; `signpost: {asset: signpost.yaml}`; `hub_portal: {asset: portal.yaml}`; `bench_0: {asset: bench.yaml}`; `bench_1: {asset: bench.yaml}`; `bench_2: {asset: bench.yaml}`; `bench_3: {asset: bench.yaml}`; `bench_x0: {asset: bench.yaml}`; `bench_x1: {asset: bench.yaml}`; `picnic_0: {asset: picnic_table.yaml}`; `picnic_1: {asset: picnic_table.yaml}`; `picnic_2: {asset: picnic_table.yaml}`; `lamp_0: {asset: street_lamp.yaml}`; `lamp_1: {asset: street_lamp.yaml}`; `lamp_2: {asset: street_lamp.yaml}`; `lamp_3: {asset: street_lamp.yaml}`; `lamp_4: {asset: street_lamp.yaml}`; `lamp_5: {asset: street_lamp.yaml}`; `lamp_6: {asset: street_lamp.yaml}`; `lamp_7: {asset: street_lamp.yaml}`; `lamp_8: {asset: street_lamp.yaml}`; `bin_0: {asset: trashcan.yaml}`; `bin_1: {asset: trashcan.yaml}`; `bin_2: {asset: trashcan.yaml}`; `bin_3: {asset: trashcan.yaml}`; `maples_ne: {asset: maple_tree.yaml}`; `pines_nw: {asset: pine_tree.yaml}`; `maples_se: {asset: maple_tree.yaml}`; `pines_sw: {asset: pine_tree.yaml}`; `maples_w: {asset: maple_tree.yaml}`; `bushes: {asset: bush.yaml}`; `visitors: {npc: npcs/park_visitor.yaml}`

**Slots**: bandstand, bench_0, bench_1, bench_2, bench_3, bench_x0, bench_x1, bin_0, bin_1, bin_2, bin_3, gate, hub_portal, lamp_0, lamp_1, lamp_2, lamp_3, lamp_4, lamp_5, lamp_6, lamp_7, lamp_8, picnic_0, picnic_1, picnic_2, playground, pond, sign

**Spawns**: from_hub

## people_gallery

`scene: scenes/people_gallery.yaml` · [source](../../assets/scenes/people_gallery.yaml)

A row of humanoid variants for review: body types stacked on the gender
presets (params: {preset: [feminine, curvy]}), proportions, hair styles and
colours, skin tones and outfits. The cottage places it on the front lawn; faces +Z.

**Places**: `petite_bob: {asset: characters/humanoid.yaml}`; `slim_ponytail: {asset: characters/humanoid.yaml}`; `curvy_long: {asset: characters/humanoid.yaml}`; `tall_bun: {asset: characters/humanoid.yaml}`; `heavy_crop_f: {asset: characters/humanoid.yaml}`; `lanky_buzz: {asset: characters/humanoid.yaml}`; `athletic_crop: {asset: characters/humanoid.yaml}`; `stocky_bald: {asset: characters/humanoid.yaml}`; `heavy_grey: {asset: characters/humanoid.yaml}`; `tall_long_m: {asset: characters/humanoid.yaml}`; `hourglass: {asset: characters/humanoid.yaml}`

**Slots**: athletic_crop, curvy_long, heavy_crop_f, heavy_grey, hourglass, lanky_buzz, petite_bob, slim_ponytail, stocky_bald, tall_bun, tall_long_m

## quest_yard

`scene: scenes/quest_yard.yaml` · [source](../../assets/scenes/quest_yard.yaml)

A plaza for trying quests and the quest HUD (docs/quests.md).

```yaml
The round  - starts with the game: three plinths in order, and an optional detour.
Lamps out  - greet the steward (E): three switches at the board (a count), then report back.
The dash   - step onto the start square: reach the finish square within six seconds, or fail.
```

Progress lasts for the session: restart the runtime to play them again.
The portal on the north side leads back to the hub.

**Places**: `plaza: {asset: plaza.yaml}`; `hub_portal: {asset: portal.yaml}`; `name_plinth: {asset: info_plinth.yaml}`; `plinth_1: {asset: info_plinth.yaml}`; `plinth_2: {asset: info_plinth.yaml}`; `plinth_3: {asset: info_plinth.yaml}`; `start_plinth: {asset: info_plinth.yaml}`; `finish_plinth: {asset: info_plinth.yaml}`; `pad_1: {asset: floor_pad.yaml}`; `pad_2: {asset: floor_pad.yaml}`; `pad_3: {asset: floor_pad.yaml}`; `pad_start: {asset: floor_pad.yaml}`; `pad_finish: {asset: floor_pad.yaml}`; `board: {asset: switch_board.yaml}`; `lamp_1: {asset: floor_lamp.yaml}`; `lamp_2: {asset: floor_lamp.yaml}`; `lamp_3: {asset: floor_lamp.yaml}`; `switch_1: {asset: light_switch.yaml}`; `switch_2: {asset: light_switch.yaml}`; `switch_3: {asset: light_switch.yaml}`; `steward: {npc: npcs/resident.yaml}`

**Slots**: board, finish_sign, hub_portal, lamp_1, lamp_2, lamp_3, marker_1, marker_2, marker_3, name, pad_1, pad_2, pad_3, pad_finish, pad_start, start_sign, steward, switch_1, switch_2, switch_3

**Spawns**: from_hub, start

## rail_loop_test

`scene: scenes/rail_loop_test.yaml` · [source](../../assets/scenes/rail_loop_test.yaml)

Test scene for city exits and a loop railway: a 2 x 2 grid whose avenues
run on out of town across a railway looping round it (four level
crossings), two stations on the straights and two trains. See scenes/city.yaml.

**Places**: `train_a: {train: trains/commuter.yaml}`; `train_b: {train: trains/commuter.yaml}`; `traffic: {traffic: traffic/town.yaml}`

**Spawns**: start

**Features**: city, railways

## restaurant

`scene: scenes/restaurant.yaml` · [source](../../assets/scenes/restaurant.yaml)

The restaurant showcase: assets/restaurant_building.yaml on a street corner.
Diners at the tables, in the booths and at the bar, a host by the door, a
bartender and a chef at work, people walking past, and a portal back to
the hub on the grass beside the building.

The building is centred on the origin: street door on the south side
(entrance_spawn at z = -8.75), dining room and bar at the front, kitchen
and restrooms behind.

**Places**: `building: {asset: restaurant_building.yaml}`; `south_walk: {asset: sidewalk.yaml}`; `west_walk: {asset: sidewalk.yaml}`; `south_road: {asset: road.yaml}`; `west_road: {asset: road.yaml}`; `hub_portal: {asset: portal.yaml}`; `lamp_1: {asset: street_lamp.yaml}`; `lamp_2: {asset: street_lamp.yaml}`; `lamp_3: {asset: street_lamp.yaml}`; `bench: {asset: bench.yaml}`; `fascia: {asset: fascia_sign.yaml}`; `planter_1: {asset: planter.yaml}`; `planter_2: {asset: planter.yaml}`; `car_1: {asset: vehicles/car.yaml}`; `car_2: {asset: vehicles/car.yaml}`; `host: {npc: npcs/hotel_staff.yaml}`; `bartender: {npc: npcs/hotel_staff.yaml}`; `chef: {npc: npcs/hotel_staff.yaml}`; `diner_1: {npc: npcs/diner.yaml}`; `diner_2: {npc: npcs/diner.yaml}`; `diner_3: {npc: npcs/diner.yaml}`; `diner_4: {npc: npcs/diner.yaml}`; `diner_5: {npc: npcs/diner.yaml}`; `diner_6: {npc: npcs/diner.yaml}`; `pedestrians: {npc: npcs/pedestrian.yaml}`

**Slots**: bartender, bench, car_1, car_2, chef, diner_1, diner_2, diner_3, diner_4, diner_5, diner_6, fascia, host, hub_portal, lamp_1, lamp_2, lamp_3, planter_1, planter_2, south_road, south_walk, west_road, west_walk

**Spawns**: from_hub

## room_with_furniture

`scene: scenes/room_with_furniture.yaml` · [source](../../assets/scenes/room_with_furniture.yaml)

Room with furniture - demonstrates nested scene composition
A room containing a dining set

**Places**: `dining_set: {scene: scenes/dining_set.yaml}`

**Slots**: back_wall_center, corner_back_left, corner_back_right, room_center

## seated_fit

`scene: scenes/seated_fit.yaml` · [source](../../assets/scenes/seated_fit.yaml)

Seats for clothing/pose fit review: an armchair, a dining chair and a sofa in a
row (clipping QA, geogen/clipping.py). Render people sitting on them with

```yaml
python -m geogen.main -s seated_fit -r out.png --affordances --views front,side
```

**Places**: `armchair: {asset: armchair.yaml}`; `chair: {asset: chair.yaml}`; `sofa: {asset: sofa.yaml}`

**Slots**: armchair, chair, sofa

## street

`scene: scenes/street.yaml` · [source](../../assets/scenes/street.yaml)

Street scene - hierarchical composition
A residential street with houses on both sides

Structure:

```yaml
street
├── road (center)
├── left_side (street_side scene, mirrored)
│   ├── sidewalk
│   ├── plot_1 → house_plot → house
│   └── plot_2 → house_plot → house
├── right_side (street_side scene)
│   ├── sidewalk
│   ├── plot_1 → house_plot → house
│   └── plot_2 → house_plot → house
└── street_furniture + trees
```

**Places**: `ground_plane: {asset: ground.yaml}`; `road: {asset: road.yaml}`; `left_side: {scene: scenes/street_side.yaml}`; `right_side: {scene: scenes/street_side.yaml}`; `lamp_1: {asset: street_lamp.yaml}`; `lamp_2: {asset: street_lamp.yaml}`; `lamp_3: {asset: street_lamp.yaml}`; `lamp_4: {asset: street_lamp.yaml}`; `fire_hydrant: {asset: fire_hydrant.yaml}`; `park_bench: {asset: bench.yaml}`; `maple_1: {asset: maple_tree.yaml}`; `maple_2: {asset: maple_tree.yaml}`; `pine_1: {asset: pine_tree.yaml}`; `maple_3: {asset: maple_tree.yaml}`

**Slots**: bench, ground, hydrant, lamp_back_left, lamp_back_right, lamp_front_left, lamp_front_right, left_side, right_side, road, tree_left_1, tree_left_2, tree_right_1, tree_right_2

## street_side

`scene: scenes/street_side.yaml` · [source](../../assets/scenes/street_side.yaml)

One side of a street - sidewalk with house plots behind it
Oriented with sidewalk along the front (+Z edge), plots behind
This scene is meant to be placed and rotated by the parent street

**Places**: `sidewalk: {asset: sidewalk.yaml}`; `plot_1: {scene: scenes/house_plot.yaml}`; `plot_2: {scene: scenes/house_plot.yaml}`

**Slots**: furniture_back, furniture_front, plot_back, plot_front, sidewalk

## terrain_test

`scene: scenes/terrain_test.yaml` · [source](../../assets/scenes/terrain_test.yaml)

Terrain review: an island with palms kept to gentle ground above the beach,
rocks on the steep slopes and a flattened pad for a building.

**Places**: `island: {asset: island_terrain.yaml}`; `trees: {asset: maple_tree.yaml}`; `rocks: {asset: rock_large.yaml}`

**Spawns**: beach

## town

`scene: scenes/town.yaml` · [source](../../assets/scenes/town.yaml)

A small town district: a 3 x 2 street grid with a commercial avenue down
the middle (shop rows with flats above, the odd office), a hotel and an
apartment block as landmarks, a park, and houses on the side streets.
Lamps, trees, benches and hydrants line the sidewalks; cars are parked along
the kerbs, traffic drives the streets and townsfolk stroll the sidewalks.

**Places**: `commuter: {train: trains/commuter.yaml}`; `hub_portal: {asset: portal.yaml}`; `traffic: {traffic: traffic/town.yaml}`; `pedestrians: {npc: npcs/pedestrian.yaml}`

**Slots**: park_portal

**Spawns**: from_hub, start

**Features**: city, railways

## trees

`scene: scenes/trees.yaml` · [source](../../assets/scenes/trees.yaml)

Tree demo scene

**Places**: `ground_plane: {asset: ground.yaml}`; `pine_tree: {asset: pine_tree.yaml}`; `maple_tree: {asset: maple_tree.yaml}`

**Slots**: ground, maple, pine

## vehicle_lineup

`scene: scenes/vehicle_lineup.yaml` · [source](../../assets/scenes/vehicle_lineup.yaml)

Vehicles for review: the car presets in a few paints, a van and a bus. Faces +X.

**Places**: `sedan: {asset: vehicles/car.yaml}`; `hatchback: {asset: vehicles/car.yaml}`; `estate: {asset: vehicles/car.yaml}`; `taxi: {asset: vehicles/car.yaml}`; `sedan_red: {asset: vehicles/car.yaml}`; `van: {asset: vehicles/van.yaml}`; `bus: {asset: vehicles/bus.yaml}`

**Slots**: bus, estate, hatchback, sedan, sedan_red, taxi, van
