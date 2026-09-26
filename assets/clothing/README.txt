Garments (kind: garment) worn by characters/humanoid.yaml outfits.

tight: `cover: {chain: [[bone, t], [bone, t]]}` thickens the body's own rings over
  that range (one mesh, no poke-through); `layer` orders overlaps (higher wins).
loose: `chains:` ring-lofted like the body/hair (expressions over the body's params),
  minus `cut` boxes; their skin weights are transferred from the body underneath
  (nearest body vertices, smoothed), so a skirt swings with the thighs it covers.
Colour comes from the outfit entry (vertex colours over the garment material).
