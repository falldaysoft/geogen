# Geogen Godot runtime

Reference runtime for geogen exports (glTF + `extras.geogen`). See the beads
under epic `geogen-3cc` for the roadmap (player, world loader, import plugin).

- **Godot version:** 4.7-stable (pinned; `config/features` in `project.godot`)
- **Renderer:** Forward+
- **Units:** 1 unit = 1 metre, +Y up (same as geogen and glTF)

## Layout

| Path | Purpose |
|---|---|
| `project.godot` | Project settings; main scene is `scenes/main.tscn` |
| `scenes/main.tscn` | Sky, sun (shadows), fog, 400 m ground with collision, 1 m scale-reference cube, empty `World` node for loaded content, camera |
| `scripts/main.gd` | Root script; parses user args after `--` |
| `scripts/player_spec.gd` | `PlayerSpec`: player capsule/step/slope/reach read from a geogen manifest |
| `generated/` | Geogen exports land here (git-ignored except `.gitkeep`) |

## Running

On macOS the binary is `/Applications/Godot.app/Contents/MacOS/Godot`; below it
is written as `godot`.

```bash
# Open in the editor
godot --editor --path runtime/godot

# Run the main scene
godot --path runtime/godot

# First-time / CI: build the .godot import cache
godot --headless --path runtime/godot --import

# Smoke test: boot headless and quit after 10 frames
godot --headless --path runtime/godot -- --quit-after=10

# Screenshot the main scene and quit (needs a GPU, not --headless)
godot --path runtime/godot -- --screenshot=/tmp/godot_main.png

# Load the player spec from an export manifest (prints it)
python -m geogen.main -s chair -e runtime/godot/generated/chair.glb
godot --headless --path runtime/godot -- --quit-after=2 --manifest=res://generated/chair.manifest.json
```

## Manifest

Every glTF export from geogen writes `<name>.manifest.json` next to the model:

```json
{"format": "geogen-manifest", "version": 1, "name": "chair", "model": "chair.glb",
 "units": "m", "up": "+Y",
 "player": {"kind": "player_spec", "version": 1, "radius": 0.3, "height": 1.8, "...": "..."}}
```

`player` comes from `assets/player.yaml` in the geogen repo, the single source
of truth for player scale. `tests/test_player.py::test_godot_reads_manifest`
checks that this runtime reads it back exactly.
