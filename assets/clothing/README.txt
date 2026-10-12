Garments (kind: garment) worn by characters/humanoid.yaml outfits.

tight: `cover: {chain: [[bone, t], [bone, t]]}` thickens the body's own rings over
  that range (one mesh, no poke-through); `layer` orders overlaps (higher wins).
loose: `chains:` ring-lofted like the body/hair (expressions over the body's params),
  minus `cut` boxes; their skin weights are transferred from the body underneath
  (nearest body vertices, smoothed), so a skirt swings with the thighs it covers.
Colour comes from the outfit entry (vertex colours over the garment material).

part: top | bottom | dress | shoes says what a garment is worn as. An outfit's top and bottom
  overlap by 3 cm at least (humanoid.lap_waist): a tight bottom runs on up under the top's hem,
  and a top's hem comes down under a loose bottom's waistband. The top's waist hem is levelled
  and follows the pelvis, so it stays one smooth loop when the thighs lift (humanoid.smooth_hem).
  A cover end may be {level: <chain>}: level with where the garment's cover of that chain starts
  (the thigh tubes' tops down to the torso's hem line).
loose shells are fitted out to the body under them (humanoid.hug_body), with room round the hips
  for the thigh tops to lift into; `samples: [[bone, t], ...]` adds rings where a skirt bends
  over the seat.
