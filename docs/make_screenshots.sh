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

# Godot runtime: export the cottage, then screenshot the first-person view
# and the collider wireframes.
render -s cottage --export-godot
"$GODOT" --headless --path runtime/godot --import > /dev/null 2>&1
"$GODOT" --path runtime/godot -- --scene cottage --spawn=1.2,0,6.8 --yaw=8 \
    --screenshot="$PWD/$OUT/godot_first_person.png" > /dev/null
"$GODOT" --path runtime/godot -- --scene cottage --colliders --spawn=4,0,6 --yaw=30 \
    --screenshot="$PWD/$OUT/godot_colliders.png" > /dev/null

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
