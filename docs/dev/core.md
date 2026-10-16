# Core: scene graph, meshes, textures, materials

The core data model and the per-module notes for `src/geogen/core/`, `textures/`, `materials/` and `lighting/`.

## Core System (`src/geogen/core/`)

- **SceneNode** (`node.py`): Hierarchical scene graph with parent-child relationships. Each node has a local Transform, optional Mesh, and children. Provides `world_transform()` for combined transformation matrices, `flatten()` to merge all geometry, and `iter_meshes()` for traversal. Supports attachment points via `get_attachment()`.

- **Mesh** (`mesh.py`): Geometry container storing vertices, faces, normals, UVs, and optional Material. Has `transform()` method, `merge()` classmethod for combining meshes, and conversion to/from trimesh.
  - **Material groups:** optional per-face `face_materials` index into `materials`, and `merge()` of differently-materialled meshes produces these groups. `groups()` splits them for the renderers, and the exporter writes one glTF primitive each.
  - **Vertex colours:** optional per-vertex RGBA `colors`, which tint the albedo in both renderers and export as `COLOR_0`.
  - **Propagation:** weld, normals, UV projection, subdivision, displacement and CSG/decimation all carry groups (CSG through a manifold vertex property); a difference's cut faces take the target's primary material. Derived meshes copy them with `with_attributes_of`.

- **Transform** (`transform.py`): TRS (Translation-Rotation-Scale) transformation. Rotation uses XYZ Euler angles in radians. Matrix order is Scale -> Rotate -> Translate.

- **geometry** (`geometry.py`): Helper functions for face winding and normal computation. Uses CCW winding convention.

- **meshops** (`meshops.py`): `compute_normals(mesh, crease_angle)` (smooth below the angle, split hard edges above; ignores UV seams), `ensure_normals`, `weld_vertices`, `collapse_short_edges` (CSG slivers; hair uses it), `compute_tangents` (glTF-style xyzw), and `validate(mesh) -> MeshReport` (degenerate faces, NaNs, boundary/non-manifold edges, inconsistent winding). Use `validate` whenever you touch a generator.

- **subdiv** (`subdiv.py`): `subdivide(mesh, levels, crease_angle)` (Loop, crease/boundary rules, sharp crease turns stay corners) and `displace(mesh, amplitude, scale, octaves, seed, ridged)` (fractal 3D gradient noise along smoothed normals; closed meshes stay closed). In YAML any part takes `subdivide: 3` / `{levels, crease}` (result is stretched back to the part box; `LayoutLoader(detail)` adds/removes levels) and `displace: {amplitude, scale, octaves, seed, ridged}`; UVs are box-projected by the undisplaced normals so noisy surfaces get clean seams. Used by rocks (`seed` param), bushes and bed pillows.

- **skin** (`skin.py`): skeletal skinning.
  - A skeleton is a plain joint `SceneNode` hierarchy; `Mesh.joints`/`weights` (Nx4) bind vertices to up to 4 joints, carried through normals/weld/groups/merge (not CSG or decimation yet).
  - `SceneNode.skin = Skin.bind(mesh_node, joints)` stores inverse bind matrices; `world_mesh()` then CPU-skins (linear blend) the current pose, so renders, the viewer and QA see it posed. Keep the skinned node a sibling of the root joint.
  - `SceneNode.clips`: `Clip(name, {joint: Track(times, rotations xyzw, translations)}, loop)`; `clip.apply(owner, t)` / `pose_clips(root, name, t)` pose joints.
  - Export writes glTF skins (JOINTS_0/WEIGHTS_0, inverseBindMatrices), no colliders or LODs for skinned meshes, joint nodes keep their own names (not uniquified), and each clip becomes an animation `<owner>_<clip>` listed in the owner's `extras.geogen.clips`.
  - Test scene: `-s skin_test` (`scenes/skin_test.py`, 3-joint tube with a `sway` clip); render a clip frame with `-r out.png --clip sway@1.0`.
  - Humanoid target: `assets/skeletons/humanoid_profile.json` (Godot `SkeletonProfileHumanoid`: 56 bones, T-pose facing +Z, left = +X, Root on the floor, each bone +Y toward its tail, with per-bone roll in `reference_pose`).
  - CSG and `decimate` carry weights as one dense column per used joint (manifold vertex properties), returning the top 4; vertices from unskinned operands copy their nearest skinned neighbour.

- **skeleton** (`skeleton.py`): `load_skeleton(path, params) -> Skeleton` from a `kind: skeleton` YAML (`assets/skeletons/humanoid.yaml`).
  - The YAML has `params` (height 1.70, head_units 6.5, shoulder_width, hip_width, leg_ratio, arm_ratio, neck_length, hand_size, foot_length), sequential `derived` expressions, world rest `bones` heads (Left* mirror to Right*), `group_scale` for unlisted profile groups (fingers at the profile's offsets) and `landmarks` (head_top, chin, heel, toe_tip, crotch).
  - All 56 profile bones always exist, with rest orientations exactly the profile's (params only move heads).
  - `Skeleton.fk(pose)`, `build() -> (root joint node, name -> node)` for `core.skin`, `length(bone)`.
  - `Pose.from_spec({bones: {Name: [x, y, z] deg}, root, frame}, skeleton)`: the default `frame: body` rotates about the character's own axes (X = its left, Y up, Z forward), e.g. `LeftUpperLeg: [-90, 0, 0]` lifts the thigh forward and `LeftUpperArm: [0, 0, -90]` lowers the arm; `frame: bone` uses each bone's rolled local axes (what library clips contain). `Pose.apply(skeleton, nodes)` poses built nodes.

- **uvmap** (`uvmap.py`): Metric UV projection — `box_project`, `planar_project`, `cylindrical_project`, and `texel_density` (1.0 == metric).

- **profile** (`profile.py`): 2D shapes for extrude/lathe: `Shape(outer, holes)` (auto CCW/CW), `rect(w, h, radius)`, `circle`, `ellipse`, `regular_polygon`, `fillet`, `arc`, `bezier`, `catmull_rom`, `offset`/`Shape.difference` (shapely), `triangulate` (earcut), and YAML builders `loop_from_spec`/`shape_from_spec`/`polyline_from_spec`.

## Textures (`src/geogen/textures/`)

- **TextureGenerator** (`base.py`): Abstract base class for procedural textures. Generates PIL Images and optional PBR maps (normal, roughness, AO). Uses numpy RNG with optional seed.

- **NoiseTextureGenerator**: Shared base for noise-based textures. Subclass implements `_get_noise_layers()`, `_compute_pattern()`, `_apply_color_shift()`. `BrickTextureGenerator` breaks this contract (returns a `(pattern, is_mortar)` tuple) and overrides `generate()`.

- Implementations: `wood`, `metal`, `floor`, `wall`, `asphalt`, `brick`, `concrete`, `dirt`, `grass`, `rock`, `roof` — each in its own module.

- **Interior pack** (`interior.py`): `fabric` (plain/twill weave, `threads` per repeat), `cut_pile_carpet`, `tile` (`tiles_x`/`tiles_y` per repeat, `grout_width`, `layout: grid|offset`), `marble`. Materials: `fabric_linen`, `fabric_velvet`, `bedding_white`, `carpet`, `tile_white`, `tile_subway`, `tile_floor_grey`, `marble`, `chrome`, `mirror`, `laminate`, `paint_{white,sage,navy,terracotta,greige}`. Any colour param in a material YAML may name a `PAINT_PALETTE` colour (`materials/loader.py`) instead of `[r, g, b]`.

## Materials (`src/geogen/materials/`)

- **Material** (`material.py`): Combines a TextureGenerator with PBR properties (roughness, metallic, normal_strength, ao_strength). Caches generated textures. `tile_size` (metres per texture repeat, YAML scalar or `[u, v]`) converts metric mesh UVs to texture space via `texture_uv_scale`; `gltf_images()` packs base colour / metallic-roughness / normal / occlusion for both the renderer and exporter.

- **MaterialLoader** (`loader.py`): Loads material definitions from YAML files in `assets/materials/`. `pbr: {opacity: 0.3}` makes a material alpha-blended (glass: see-through in renderer, viewer, glTF `BLEND` + double-sided, Godot); `pbr: {emissive: [r, g, b], emissive_strength}` makes it glow (`lamp_shade`, `screen`).

## Lighting (`src/geogen/lighting/`)

- **Light classes**: `DirectionalLight` (sun-like), `PointLight` (omni). Both have color and intensity.

- **SceneLighting**: Container with ambient color and lights list. Provides `get_shader_data()` for shader uniforms. Preset: `outdoor_lighting()` (the Qt viewer's).
