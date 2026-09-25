#!/bin/bash
# Regenerate the README screenshots in docs/images/.
# Needs the venv (.venv) and, for the Godot shots, Godot 4.7 ($GODOT or the
# macOS app bundle). Run from the repository root: docs/make_screenshots.sh
set -euo pipefail

OUT=docs/images
PY=${PYTHON:-.venv/bin/python}
GODOT=${GODOT:-/Applications/Godot.app/Contents/MacOS/Godot}
mkdir -p "$OUT"

render() { "$PY" -m geogen.main "$@" > /dev/null; }

render -s chair      -r "$OUT/geogen_chair.png"      --resolution 1280x720
render -s dining_set -r "$OUT/geogen_dining_set.png" --resolution 1280x720 --zoom 1.8
render -s room       -r "$OUT/geogen_room.png"       --resolution 1280x720
render -s street     -r "$OUT/geogen_street.png"     --resolution 1280x720 --zoom 1.35
render -s cottage    -r "$OUT/geogen_cottage_views.png" --views --resolution 1400x1000
render -s street --viewer-screenshot "$OUT/geogen_viewer.png"
render -s town       -r "$OUT/geogen_town.png"       --resolution 1400x800 --zoom 2.2
render -s park       -r "$OUT/geogen_park.png"       --resolution 1280x720 --zoom 1.8
render -s hotel      -r "$OUT/geogen_hotel.png"      --resolution 1280x720 --view iso_back
render -s hotel_room_auto -r "$OUT/geogen_hotel_room.png" --resolution 1280x720 --cutaway --view top --zoom 1.3

# Godot runtime: export the cottage, then screenshot the first-person view
# and the collider wireframes.
render -s cottage --export-godot
"$GODOT" --headless --path runtime/godot --import > /dev/null 2>&1
"$GODOT" --path runtime/godot -- --scene cottage --spawn=1.2,0,6.8 --yaw=8 \
    --screenshot="$PWD/$OUT/godot_first_person.png" > /dev/null
"$GODOT" --path runtime/godot -- --scene cottage --colliders --spawn=4,0,6 --yaw=30 \
    --screenshot="$PWD/$OUT/godot_colliders.png" > /dev/null

# The cottage resident (NPC), sped up to a moment in their day: resting in the
# armchair, then out on the doorstep. Deterministic with --fixed-fps.
"$GODOT" --fixed-fps 60 --path runtime/godot -- --scene cottage --timescale=4 --simulate=40 --camera=follow \
    --screenshot="$PWD/$OUT/godot_npc_armchair.png" > /dev/null
"$GODOT" --fixed-fps 60 --path runtime/godot -- --scene cottage --timescale=4 --simulate=80 --camera=follow \
    --screenshot="$PWD/$OUT/godot_npc_doorstep.png" > /dev/null

# Streamed town district: the avenue, and the hotel lobby interior.
render -s town --export-godot --stream
"$GODOT" --path runtime/godot -- --scene town --stream --spawn=-3,0,-2 --yaw=90 --quit-after=60 \
    --screenshot="$PWD/$OUT/godot_town_street.png" > /dev/null
"$GODOT" --path runtime/godot -- --scene town --stream --spawn=-0.05,0.15,-12.5 --yaw=0 --quit-after=60 \
    --screenshot="$PWD/$OUT/godot_hotel_lobby.png" > /dev/null

# Retina captures come out at 2x; keep the repo images modest.
"$PY" - <<'PYEOF'
from PIL import Image
for name, width in [("geogen_cottage_views", 1400), ("geogen_viewer", 900)]:
    path = f"docs/images/{name}.png"
    im = Image.open(path)
    im.thumbnail((width, width * 2))
    im.save(path, optimize=True)
PYEOF
echo "Screenshots written to $OUT"
