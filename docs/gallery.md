# Gallery

What geogen makes today, rendered offscreen or captured in the Godot runtime. `docs/make_screenshots.sh` regenerates every picture here.

**Town**: a `city:` scene. It has a street grid with an avenue and sidewalks, zoned lots and street furniture. Buildings are generated to fit their lots:
- shop rows with flats above along the avenue;
- houses on the side streets;
- a hotel and an apartment block as landmarks;
- a park.

![Town](images/geogen_town.png)

**Walking the town in Godot**: the district streams in chunks around the player (left). Buildings have furnished interiors, like the hotel lobby (right):

| Streamed street | Hotel lobby |
|---|---|
| ![Godot town street](images/godot_town_street.png) | ![Godot hotel lobby](images/godot_hotel_lobby.png) |

**Hotel**: a multi-storey building generated from floor-plan layouts: a lobby floor, guest floors, scissor stairs, a working lift, a façade and a roof (left). A guest room furnished by the furnishing solver, shown with the ceiling cut away (right):

| Hotel | Furnished room (cutaway) |
|---|---|
| ![Hotel](images/geogen_hotel.png) | ![Hotel room](images/geogen_hotel_room.png) |

**Park**: seeded scatter placement of space-colonisation trees and displaced rocks around a cottage and a lamp-lit path:

![Park](images/geogen_park.png)

**Street**: a composed scene. Each side of the road is a nested `street_side` scene whose house plots place cottages, and the lamps, trees, hydrant and bench are placed in named slots.

![Street](images/geogen_street.png)

**Cottage**: a hollow brick shell with a gable roof and chimney. The door and windows each cut their own opening into the wall they're placed on. Contact sheet from `--views` (iso, front, side, top):

![Cottage contact sheet](images/geogen_cottage_views.png)

**Dining set**: turned (`lathe`) legs and spindles, a rounded (`extrude`) tabletop, and four chairs attached to the table's `seat_*` attachment points:

![Dining set](images/geogen_dining_set.png)

| Chair | Room |
|---|---|
| ![Chair](images/geogen_chair.png) | ![Room](images/geogen_room.png) |

**Godot runtime**: the cottage exported with `--export-godot` and walked in first person (left); F2 shows the collision shapes (right):

| First person | Collider wireframes |
|---|---|
| ![Godot first person](images/godot_first_person.png) | ![Godot colliders](images/godot_colliders.png) |

**NPCs**: the cottage has a resident, a young woman drawn from a character archetype (skinned body, painted face, hair and clothes), walking with procedural clips. Everything she does is data: the furniture, windows and door advertise what can be done there, and the Godot runtime scores those offers against her needs and runs the chosen action's steps. Here she walks over to the armchair, rests in it, and later steps out of the front door:

| Walking | Resting | On the doorstep |
|---|---|---|
| ![NPC walking](images/godot_npc_walking.png) | ![NPC in the armchair](images/godot_npc_armchair.png) | ![NPC on the doorstep](images/godot_npc_doorstep.png) |

**Fairgoers**: the amusement park's crowd come in groups of one to three that stay loosely together: the others wander near the leader, drift back when they've been apart a while, and take the seats beside each other's. They buy from the stalls and carry what they bought in their hands while they eat it, then bin the wrapper:

![Fairgoers in the amusement park](images/godot_fairgoers.png)

**People**: humanoid characters are generated from a skeleton and a body shape. Presets, body types, skin tones, hair styles and outfits are all parameters, and character archetypes draw them at random, so a crowd is a set of seeds. Bodies are skinned, posed with IK, animated with procedural walk and idle clips, and dressed in garments that follow the body:

![People gallery](images/geogen_people.png)

**Vehicles**: cars (sedan, hatchback, estate, taxi), a van and a bus, plus rolling stock for trains. In the runtime they drive a precomputed lane graph with signals, give way to pedestrians and wait at level crossings:

![Vehicle lineup](images/geogen_vehicles.png)

**Interactive viewer** (Qt/OpenGL) viewport: shadows, PBR, grid, picking:

![Viewer](images/geogen_viewer.png)
