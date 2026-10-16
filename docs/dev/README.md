# Developer notes

How each part of geogen works, for whoever changes it next. `CLAUDE.md` at the repo root is the short version every session reads: commands, conventions, testing policy and the changes that need matching changes elsewhere. The contracts with runtimes are `docs/schema/`, `docs/rules.md`, `docs/quests.md` and `runtime/godot/README.md`.

| File | Covers |
|---|---|
| [core.md](core.md) | Scene graph, meshes and mesh ops, subdivision, skinning, skeletons, UV projection, 2D profiles; textures and materials |
| [generators.md](generators.md) | Primitives and generators (profiles, sweeps, text, terrain, stairs, roofs, CSG, floor plans, buildings, facades, nature); the furniture library and interior finishes |
| [layout.md](layout.md) | Asset and scene YAML: attachments, slots, surfaces, params and expressions, cities, scatter, references, named places, projects, the registry, `--describe`, furnishing and layout QA |
| [characters.md](characters.md) | Humanoid bodies, clips, hair, clothing, archetypes, clipping QA (with [../clothing.md](../clothing.md)) |
| [transport.md](transport.md) | Vehicles, the traffic lane graph and signals, moving traffic, railways and trains, pedestrians |
| [gameplay.md](gameplay.md) | Interactions, NPCs, time of day, light fixtures, physics props, spawned items; conditions, state, rules, wiring, content packages, quests, items, dialogue, HUD |
| [runtime.md](runtime.md) | Rendering, export, extras, chunks and caching, the Qt viewer, the Godot runtime |
