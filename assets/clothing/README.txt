Garments (kind: garment) worn by characters/humanoid.yaml outfits.

tight: `cover: {chain: [[bone, t], [bone, t]]}` thickens the body's own rings over
  that range (one mesh, no poke-through); `layer` orders overlaps (higher wins).
  `ease: <m>` is its stiffness on the torso: the cloth goes where a ball of that radius rolled
  over the body would touch, spanning hollows (under a bust) instead of clinging to them.
loose: `chains:` ring-lofted like the body/hair (expressions over the body's params),
  minus `cut` boxes. `weights: legs` skins a skirt to the pelvis and both thighs
  (humanoid.leg_weights); the default `body` takes the weights of the body underneath.
Colour comes from the outfit entry (vertex colours over the garment material).

part: top | bottom | dress | shoes says what a garment is worn as. An outfit's top and bottom
  overlap by 3 cm at least (humanoid.lap_waist): a tight bottom runs on up under the top's hem,
  and a top's hem comes down under a loose bottom's waistband. The top's waist hem is levelled
  and follows the pelvis, so it stays one smooth loop when the thighs lift (humanoid.smooth_hem).
  A cover end may be {level: <chain>}: level with where the garment's cover of that chain starts
  (the thigh tubes' tops down to the torso's hem line).
loose shells are fitted out to the body under them (humanoid.hug_body). A shell that is one
  tube gets pose correctives (generators/drape.py): its rings are re-shaped round the posed body
  in every pose and clip (on the lap and the seat when seated) and baked to a few morph targets
  the clips key. Keep its rings close (`spacing` ~6 cm) so it can bend. See docs/clothing.md.
