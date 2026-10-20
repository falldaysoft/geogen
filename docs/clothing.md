# Clothing

How dressed humanoids are built, how clothes are kept on the body when it moves, what was tried
on the way, and what is still open. Task: geogen.11.5.5 (and the bugs it blocked, geogen.11.5.7 and
geogen.11.5.22). Written 2026-10-12; arms added 2026-10-13.

The code is in `src/geogen/generators/` (`humanoid.py`, `ringloft.py`, `drape.py`, `arms.py`), the garments in
`assets/clothing/*.yaml`, the outfits in `assets/characters/humanoid.yaml`, and the measurements
in `src/geogen/clipping.py`.

## Where it stands

| Part of the 11.5.5 proposal | State |
| --- | --- |
| 1. Garments that span the body's hollows instead of shrink-wrapping them | Done for tight garments, as a stiffness (`ease`) on the garment. Not done as an SDF: see "Tight garments". |
| 2. Offline cloth simulation over poses and clips | Replaced by a geometric drape (`drape.py`). A simulation was tried first and not finished: see "What was tried". |
| 3. A runtime rig fitted to the drape: skinning plus a few blend shapes | Done for skirts and dresses: linear blend skinning and at most 8 morph targets, keyed by every clip. No extra bones, no secondary motion. |
| 4. Poses and clips that keep the arms clear of the body and the clothes | Done (`arms.py`): each body's upper arms swing out by what its own hips, thighs and skirt need, in every static pose and every clip. See "Arms". |
| 5. Deleting body faces that clothes always hide | Done (geogen.11.5.24): faces well inside the draped cloth in every pose and sampled frame are left out of the drawn body (about 370-400 on a skirted build); the closed body stays beside it, build-only, for fitting and QA. See "Hidden faces". |
| 6. A test: nothing past 2 mm over presets, outfits and sampled clip frames | Done. Cloth against body and seats on five skirted bodies (`tests/test_clipping.py::test_nothing_pokes_through_a_skirt_in_any_pose_or_clip`), and the task's acceptance: all 18 people of `clothing_fit` and `people_gallery`, standing, seated, lying and through the walk, idle and the four transitions (`test_review_scenes_pass_the_penetration_check`). A heavy build in a skirt is still left out. |

Clipping of skirts and dresses, before and after (worst depth in mm, and how many vertices):

| Check | Before | After |
| --- | --- | --- |
| Body through the cloth, seated (five presets) | up to 73 (petite 66, heavy 73) | 0, except heavy: 4 to 5 (one vertex of an overhanging belly) |
| Body through the cloth, walking | not measured; 60 to 110 reported, mostly knees leaving by the hem (not a fault, see "Measuring") | 0 on the five tested bodies |
| Cloth in the seat (armchair, chair, sofa) | 4 to 9, up to 38 vertices | 0 |
| Seated hands in the skirt | up to 15 (hourglass dress) | 0 |
| Hands and forearms in thighs, hips and skirts through sit_down, stand_up, lie_down, get_up (18 people) | 7 to 54, up to 180 vertices, on every one | 0 |
| Hands in the hips or the skirt, lying (wide builds) | 15 to 28, up to 250 vertices | 0 |
| Hands brushing a skirt in the walk and idle (hourglass) | 3 to 14 | 0 |
| Skin at the hip crease of a skirt (three builds were expected failures) | a 2 to 3 mm sliver | none; the expected failures are removed |

Costs: a skirt is about 1,100 triangles (was 670), so a dressed character with one is up to about
4,650 (the test's limit was 4,200). Building a character takes about 0.6 s in trousers and
1.4 s in a skirt (2026-10-18, after the fast skeleton maths of geogen.11.5.29; it was 2.2 s and
3.3 s): the drape is about 0.4 s of that and the arms 0.3 s, or 0.65 s with a skirt to keep out of.
With sculpted hands (2026-10-19) it is 0.93 s in jeans and 1.58 s in a skirt (drape ~0.3 s, the
arms ~0.75 s: the skirt is a second obstacle every probe tests). Probing the skirt through its
rings instead of its triangles (`arms._Tube`) brought that to 1.31 s (the arms 0.30 s in jeans,
0.44 s in a skirt: the skirt is cheap to probe now, but more frames need a swing). Nothing is added at runtime: the arms' swing is
in the clips' keys.

## The system

A character is built by `humanoid.build_humanoid` from the `body:` block of
`assets/characters/humanoid.yaml`. The body is a set of chains: tubes lofted ring by ring along
bones (`ringloft.py`), unioned into one skinned mesh. Clothes come in two kinds.

### Tight garments

A tight garment (`cover:` in its YAML: T-shirt, trousers, shoes, a dress's bodice) is not a mesh
of its own. It is a `ringloft.Span` over part of a chain: the body's own rings grow by the
garment's `thickness` between two `(bone, t)` positions and their faces take the garment's
material slot. So the body under it does not exist, and it cannot poke through. Where two
garments overlap the higher `layer` wins. Edges become hem ring pairs 2 mm apart.

Shoes add a sole (`sole: {height, color}`): not a span but a slab under the shoe's outline,
unioned on after decimation (`humanoid.build_sole`), so the shoe has a flat tread and an edge.

Rules that keep the waist closed (geogen.11.5.19): `lap_waist` makes a top and a bottom overlap
by 3 cm, `smooth_hem` levels the top's hem after decimation and gives it trunk weights only, the
leg chain has `root: Hips` so the top of the thigh tube stays with the pelvis.

**Stiffness (`ease`, new).** A span used to follow the skin exactly, so a top clung to the
underside of a bust and to a narrow waist. A garment now takes `ease: <metres>`: the radius of a
ball rolled along the outside of the body's front, back and side outlines. The cloth goes where
the ball touches, so it spans hollows the ball cannot get into (`ringloft._roll`,
`_span_hollows`). It applies to the torso chain only. Values: tank top 0.10, T-shirt, long sleeve
and dress 0.12, sweater 0.22. The values are small on purpose: 0.12 spans the hollow under the bust and still
follows the waist, while 0.3 and above made a top on the `hourglass` build a straight box from
bust to hip (seen before the ball was allowed to roll off the garment's ends, which also lifted
its hems; not re-measured since).

This is the proposal's "body SDF plus morphological closing" done on the three outlines of the
rings rather than on a volume. It cannot span a hollow that runs round the body rather than
along it, and it does not remesh.

### Loose garments

A loose garment (`chains:` in its YAML: the skirt, a dress's skirt) is a separate shell, lofted
like the body and kept as the `clothes` node. An open hem (`caps: [round, hem]`) rolls the tube
over a rim and runs it back inside itself, so it has an outer wall and a lining and stays
watertight.

- `hug_body` moves the shell's walls out to the body under them plus a clearance at rest, so one
  authored skirt fits every build. The body is measured over the patch of cloth round each
  vertex (half-way to the neighbouring rings and vertices), since the wall between them is
  straight and a hip that bulges between two rings would show through. The waist rings are
  authored close to the body (8 mm), so the room a build gets is the body's own plus the
  clearance, and a slim body's dress no longer juts out at the waist seam.
- `leg_weights` skins it: the waist to the Hips, handing over to the two thighs between a height
  above the hip joints and one below. See "Weights" below for why it is shaped as it is. The start
  cap above the first key ring, tucked under the top, takes the body's weights on bones other than
  the pelvis and legs (the spine), from the few body vertices right under it, unsmoothed: on the
  Hips, or smoothed with the pelvis below, a back bent far forward came out through it.
- `drape.fit` works out where the cloth should really be in each pose and stores the difference
  as morph targets.

### The drape (`drape.py`)

There is no cloth simulation, at build time or at runtime. `drape_frame` re-shapes the shell in
one pose, one ring of its outer wall at a time. A ring skinned into a pose still lies near a
plane. In that plane:

1. **What it goes round.** The posed body is cut by the plane, and the convex hull of the cut is
   what the ring has to clear. The hull makes cloth span the gap between the legs. A ring that
   rides the thighs goes round the thighs, the knees and the back of the pelvis; a ring that rides
   the pelvis goes round the pelvis and the belly as well. Body within 5 cm either side of the
   plane counts too (the cloth's faces run on to the next ring), fading out toward that distance
   so nothing pops in or out as the body moves.
2. **Clearance.** No ring vertex comes nearer the hull than it was at rest (10 mm where it had
   that much), plus 3 mm where the body has moved, plus an allowance where rings fan apart round a
   bend and the faces between them cut the corner.
3. **Looseness.** What the skinned ring has to spare beyond that, it keeps. Two things take it
   away. Gravity: in a ring that stands on edge (a seated lap), looseness hangs below the body, so
   cloth on top lies on the thighs. Length: a ring never gets longer than it was cut, so legs
   pushing out one side pull the rest in.
4. **The seat.** In a seated pose nothing below the hips and behind the knees is below the seat
   plane, which is the seat flesh's underside less 12 mm of cloth.

Everything that is not a ring vertex (lining, rolled hem, the buried top cap) moves with the ring
vertex nearest it. The waistband ring is left as skinned, except that it never comes nearer the
body than at rest (bent far forward, skinning pulls the inside of the bend into the belly). The
rest pose comes out unchanged exactly, so a standing character needs no morph.

`fit` drapes every static pose and sampled frames of every clip (16 of the walk, 8 of idle, 10 of
the others), maps each displacement back to the rest frame through the vertex's blended skinning
matrix, and reduces the lot to at most 8 morph targets `drape_0..7`. A frame is draped once per
pose of the joints it depends on (those moving the shell or the pelvis and thighs it reads, and
their ancestors), so a wave, the work clips' arms and breathing reuse the standing drape: 57
drapes of 119 frames. Of the body, the drape skins only those triangles and the seat flesh.

- up to three static poses get a target of their own, so they are exact (sit, sit at table and
  the crouch: poses held for minutes);
- the rest are the principal components of what the clips' frames still lack, with the walk's
  frames counted three times.

Each clip gets a weights track (`Clip.morphs["clothes"]`), exported as a glTF `weights` channel.
Weights run from -1 to 1. The Godot runtime needs nothing new: its animation player plays them
and its `RESET` rests them at 0. The report is in `meta.humanoid.drape`: the number of targets,
and per clip the largest correction and the largest error the targets leave (typically 2 to 6 mm
in the walk and the transitions, 6 to 12 mm on heavy builds).

### Arms (`arms.py`)

Poses and clips are authored once for every build. A hand that hangs beside a slim hip is inside
a wide one, and a blend from standing to sitting takes every body's hands through its thighs.
`clear_arms` runs last in the build, when the clips exist and the skirt is draped. It changes
nothing but the upper arms' rotation keys.

- **What is kept clear:** the forearm and hand of each side (what `clipping.measure` reports as
  `hands`), by the vertices on their convex hulls.
- **Of what:** the body without its arms, head and feet, and the loose shell, in the pose of the
  frame, drape morphs included. For the body, depth is the distance to the nearest triangle,
  signed by the triangle the point is most squarely in front of or behind among those that meet
  there. A draped shell is the solid its rings bound (`drape.solid_rings`: the key rings and the
  start cap's, capped at the first and last; `arms._Tube`): each ring is tabled per frame, round
  its centre, by how far it reaches and how far off its plane it is in each direction, and a
  point is measured between the two rings it lies between, in their planes. That is over 10x
  quicker than triangles and agrees with clipping QA's solid (the outer wall with its hem capped)
  to within a few mm, except under a seated skirt. A loose shell that isn't draped is its outer
  wall with its openings capped, as triangles; a point found inside one is confirmed with the
  winding number (the nearest triangle goes wrong at a rolled hem's lip).
- **How:** if anything is more than 1 mm inside, the arm turns about the shoulder joint so that
  the hand moves outward, away from the body's centre plane, until everything is 4 mm clear. The
  angle is found in a few steps, each as far as the deepest vertex still lacks.
- **Over time:** static poses are solved exactly. A clip is solved at 14 frames (the walk 12,
  idle 3); each frame then takes the most its neighbours need, blurred, and never less than it
  needs itself, so the swing comes on before it is needed and goes off after. A transition's
  first and last frames take the swing of the static poses it runs between, so the hand-over to
  the pose clip doesn't pop. A clip that is another played backwards (get_up, lie_down) reuses
  its answer.

A pose that is clear already is left exactly as authored: a slim body's stand, sit and lie are
untouched. Typical swings are 7 to 12 degrees in the transitions and 5 to 10 lying for wide
hips; `meta.humanoid.arms` records the largest per clip. The body option `clear_arms: false`
turns it off.

Tried and dropped: swinging the way the surface the hand is in faces (up off a lap, out from a
hip). It is the shortest way out in any one frame, but the direction changes from frame to frame
and blending two swings about different axes puts the hand back in the body between them
(55 mm in the hourglass body's sit_down). Outward is the same in every frame, so more of it is
always further out. The surface's direction is still the fallback when outward clears nothing.

### Weights

How the skirt is skinned decides how much the drape has to correct, and so how well 8 targets can
hold it.

- The hand-over from pelvis to thighs is centred on the hip joints and starts above where the
  thighs' tops come when they lift level. Rings then fan round the hip like an elbow's. A ring
  that stays level while a thigh rises through it sees the whole length of the thigh appear in
  its plane at once, which no smooth correction can follow.
- Each vertex follows both thighs, half each, with a quarter more of its own side's
  (`LEG_SPLIT`). With all of its own side's, the hem follows each leg rigidly and folds at the
  centre line. With none, the cloth hangs correctly between the legs but the pushes of the knees
  travel round the skirt through the stride, which takes more targets than there are (walk error
  7 to 10 mm against 2 to 5 mm).

### Measuring

`clipping.measure(character, pose | clip, t, seat)` reports, in millimetres: body vertices outside
a shell that covered them at rest (`shell_poke`), hands inside the body (not their own upper arm) or a shell, bare heels in
shoes, and, at the waist of separates, skin showing, overlap and the hem's smoothness. With a
seat it poses the character on it as the runtime does and reports what is inside the seat.

A vertex that is nearer the hem's opening than the cloth has left by the opening and is no longer
counted: a knee below a skirt that rode up. Those were the 60 to 110 mm "pokes" while walking.

`tests/data/clipping_baseline.json` records five presets by four outfits; regenerate it with
`python tests/test_clipping.py`.

## What was tried

In order. The first four are earlier work, from the commit history and the task notes.

1. **Weights from the body under the shell** (`transfer_weights`, f3b8ce0). Each shell vertex
   takes the weights of the nearest body vertices. The hem near the knees picked up shin weights
   and swung down with the lower legs when seated. Still available as `weights: body`.
2. **Procedural leg weights and one `seated` morph** (e0f334c). Hips at the waist, each side to
   its own thigh below the hip joints, and a single morph that lifted whatever hung below the
   seat plane straight up onto it. Cloth in seats fell from 40-68 mm to 4-5 mm. The seated skirt
   still stood off the lap as a box, the body still came through it on some builds (51 to 73 mm),
   and nothing was corrected in any other pose.
3. **A cloth simulation** (branch `experiment/xpbd-drape`, d220bba). A quasi-static position-based
   solve per pose: the body moves from the bind pose into the pose while stretch, bend and tether
   constraints and a signed distance field keep the cloth on it. It needed the shell densified to
   32 sides and 4 cm rings (about four times the triangles), was never wired into the build, and
   handled static poses only. It is still on the branch, which predates `hug_body` and the waist
   work and would need rebasing.
4. **Room at the hips** (30332bf): `HIP_ROOM` in `hug_body` and extra rings (`samples`) where the
   skirt bends. This closed the waist in most poses; three builds kept a sliver of thigh at the
   hip crease. The extra rings are gone again (the shell is evenly dense now), and so is
   `HIP_ROOM` (geogen.11.5.38.5): what it covered for was the wall between two rings cutting
   through a heavy body's hips, and the ring vertices authored just past `HUG_REACH` being left
   where they were. `hug_body` now measures the body over each vertex's patch of wall and fits
   the whole wall below the first key ring.
5. **The geometric drape** (this work). Things tried inside it and dropped:
   - *Looseness measured from the rest pose* (the hull plus what the ring had to spare at rest).
     A leg that swings forward then pushes the cloth as far forward again as it was loose,
     instead of using the looseness up. Corrections in the walk were 15 cm and could not be
     compressed (25 mm error).
   - *Principal components alone.* Static poses were several millimetres off, enough to show the
     thighs through a seated skirt. They now get targets of their own first.
   - *Carrying the seat along under a sit-down*, so the cloth is held off the seat in the frames
     before the character lands. The clamp sweeping through the cloth is not something a few
     targets can follow: errors of 30 to 50 mm, visible wobble. The seat now applies only at the
     seated end of a transition.
   - *Stopping the seat at its front edge* (where the seat flesh is measured to, 30% of the thigh
     behind the knee), so the cloth hangs free behind the knees. It looks right, but seats differ
     in depth, and on the armchair and sofa the hanging cloth was 3 to 6 cm inside the cushion.
     The seat is taken as far as the knees.
   - *Blending the pelvis and thigh sections* for rings in between, and *counting more rings as
     pelvis rings*, both to cover a belly that overhangs the lap. Both made it worse (10 to 16 mm
     against 5). See the first open item.
   - *Hard thresholds* on which body points a ring answers for. Points popped in and out as the
     body moved and the drape jumped 20 to 50 mm between frames. They fade now.

## Hidden faces

A character in a skirt or dress no longer carries the pelvis and thighs under it (geogen.11.5.24).
While `drape.fit` drapes each static pose and sampled clip frame it also measures, for every
body vertex the drape reads (the pelvis, thighs and knees), how deep inside the draped shell it
is, through the shell's rings as `arms._Tube` probes them. A vertex is hidden when it is at
least `HIDE_DEPTH` (8 mm, plus the most the baked targets are off by anywhere) inside the cloth
in every frame and `HIDE_END` (4 cm) from the shell's two ends, so nothing near the hem's
opening goes. A body face all of whose vertices are hidden is left out of the drawn `body`
(about 370 to 400 triangles on a skirted build; the heavy build keeps nearly all of them, since
its belly comes near the cloth seated). The vertices stay, so `meta.humanoid.garments` and the
morphs keep their indices; `meta.humanoid.hidden_faces` is the count.

The drawn body is then open, and four things need it closed: `affordance_qa`'s `pose_clash`
(a manifold3d volume), `fit_seats` and `fit_seated_pose` (the underside of the seat flesh),
`drape.Draper` on a posed instance (the hull of the pelvis and thighs' section) and `clipping`'s
hand check (winding numbers against the body). So the build keeps the whole body as a second
skinned node, `body_closed`, hidden and marked `build_only` in its meta: renders and
`iter_meshes` skip it (hidden), the export leaves it out entirely (`build_only`, listed in
`extras.BUILD_ONLY`), and `humanoid.closed_body(root)` is what those four read, falling back to
`body` when there is none (no loose garment, or nothing hidden). The clipping QA's `shell_poke`
still checks every covered vertex, hidden ones included, so a dropped face that came to show
would fail `test_nothing_pokes_through_a_skirt_in_any_pose_or_clip`.

## What is open

Tracked as geogen.11.5.25 (build time: geogen.11.5.23).

- **A belly over a lap.** On the `heavy` build, leaning forward on a seat or crouching, the
  belly's underside shows through the cloth by 4 to 10 mm. Rings on the thighs pass under the
  belly and rings on the pelvis go round it, and the faces between the two cut through it. This
  is as much the body's own belly meeting its thighs as a clothing fault. The heavy build is in
  the baseline but not in the strict test.
- **Hems over wide hips.** A top's hem still stands out over the hips of the `hourglass` build
  like a peplum: it ends on the widest part of the body and steps out by its own thickness over
  the trousers' there. Stiffness doesn't change a garment's ends.
- **Arms only swing at the shoulder.** The elbow never bends to get a hand out of the way, the
  upper arm isn't kept out of the bust or the ribs (it isn't measured either), and a hand is not
  kept out of the other hand, the hair, or furniture. A wide skirt holds the arms of the
  `hourglass` body out about 20 degrees while it walks, which a bent elbow would do better.
- **Arms between sampled frames.** The 14 frames a transition is solved at are not the frames
  it is checked at (`REVIEW_FRAMES`), and those are clean, but nothing checks every key.
- **Between sampled frames and in blends.** The runtime cross-fades clips and scales the walk's
  speed. Weights blend linearly, which is right for morphs, but only sampled frames are checked.
- **The moment of sitting down.** For about a tenth of the clip before the character lands, the
  back of the skirt can be in the seat (see the seat entries under "What was tried").
- **Hair** has no pose correctives. Sculpted hair (geogen.11.5.32) no longer lies in the dining
  chair's top rail, but a fall of long hair can still meet a high chair back.
- **The bare crotch, seated** (`outfit: none` only, every garment hides it). The bottom of the
  pelvis tube stays with the Hips while the thighs swing up, so a seated body without clothes
  shows a small faceted lump between the thighs (largest on `hourglass`, the `nude` of
  `clothing_fit`; a tooth or two on other builds). It doesn't cut into the thighs: the heavy
  build's crotch was reported as interpenetrating (geogen.11.5.22) and measures the same as the
  default build now. Accepted as is (geogen.35).
- **Bare heels in flats on the heavy build.** `feet_skin` counts 2 to 6 heel vertices of
  `masculine+heavy` outside its flats in every pose (in the baseline, not the strict tests).
- **Only single-tube shells are draped.** A loose garment with several chains or `cut` shapes is
  left as skinned (`build_humanoid` skips it). A loose top, a coat or wide trouser legs would need
  the rings found another way, and the choice of body parts to go round (`drape.LEGS`, `TRUNK`)
  is the skirt's.
- **No secondary motion**: a skirt does not swing or lag. That needs bones or a runtime solve.
- **Crowd cost.** Eight morph targets on about 600 vertices per skirted character, and about 450
  more triangles than before. Character LODs are geogen.11.5.17.
- **Knees and hips losing volume in deep bends** (geogen.11.5.22) is a fault of the body's own
  skinning and is untouched. The drape's method (a corrective per pose, reduced to a few targets)
  would apply to the body mesh too, with a different rule for where the surface should be.

## Looking at it

```bash
# every outfit on the stress-test body
python -m geogen.main -s clothing_fit -r out.png --view front --zoom 3.4 --no-ground
# a pose or a clip frame of one character
python -m geogen.main -s humanoid_lineup -r out.png --pose sit
python -m geogen.main -s humanoid_walk -r strip.png --clip walk --filmstrip 8 --view side --zoom 2.6
# in the runtime, every body frozen in a pose (re-import first after pulling new scripts)
python -m geogen.main -s clothing_fit -e /tmp/gen/clothing_fit.glb
godot --path runtime/godot -- --generated=/tmp/gen --scene=clothing_fit --play=pose_sit@0.2 --screenshot=/tmp/sit.png
```

`drape.Draper(character, character.meta["humanoid"]["shells"]).positions(actor, seat)` gives the
drape of any posed instance before it is reduced to morph targets, for telling a fault of the
drape from a fault of the reduction.
