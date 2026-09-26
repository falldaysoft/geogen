"""Export scene graphs to game-engine formats (glTF/GLB, OBJ).

glTF keeps the node hierarchy with local transforms, so an exported house
still has separately addressable doors and windows. Meshes carry explicit
crease-angle normals, and PBR metallic-roughness materials with base colour,
metallic-roughness, normal and occlusion textures. Metric UVs are converted
to texture space using each material's ``tile_size`` so textures repeat at
the right real-world scale (samplers use REPEAT wrapping).

glTF exports also write ``<name>.manifest.json`` alongside the model: the
contract the Godot runtime reads (units, up axis, model file, player spec,
rooms, spawns).

Skinned meshes (``SceneNode.skin``, see ``core.skin``) export as glTF skins
with JOINTS_0/WEIGHTS_0, and their clips as animations on the joint nodes.

Gameplay metadata (schema: docs/schema/geogen-extras.v1.schema.json) rides
along in glTF node extras as ``extras.geogen`` -- tags, room ids, trigger
volumes, spawn points, walkable flags and the collider chosen for each mesh.
Colliders are extra child nodes named with Godot's import suffixes, so the
editor importer builds physics with no plugin: ``<name>-convcolonly`` (box or
convex hull) and ``<name>-colonly`` (trimesh). Runtimes that don't know the
suffixes should skip nodes whose extras say ``type: collider``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import trimesh

from .core import meshops
from .core.mesh import Mesh
from .core.node import SceneNode
from .core.transform import quat_from_matrix as _quat
from .materials.material import Material
from .player import PlayerSpec, load_player_spec

logger = logging.getLogger(__name__)

FORMATS = {".glb", ".gltf", ".obj"}
MANIFEST_FORMAT = "geogen-manifest"
MANIFEST_VERSION = 1
EXTRAS_VERSION = 1
EXTRAS_SCHEMA = "docs/schema/geogen-extras.v1.schema.json"

# Colliders smaller than this in every dimension are skipped by ``auto``.
MIN_COLLIDER_SIZE = 0.03
COLLIDER_SUFFIX = {"box": "-convcolonly", "hull": "-convcolonly", "mesh": "-colonly"}


def _pbr_material(material: Material, cache: dict[int, trimesh.visual.material.PBRMaterial]):
    key = id(material)
    if key not in cache:
        images = material.gltf_images()
        cache[key] = trimesh.visual.material.PBRMaterial(
            name=material.name,
            baseColorTexture=images["base_color"],
            metallicRoughnessTexture=images["metallic_roughness"],
            normalTexture=images.get("normal"),
            occlusionTexture=images.get("occlusion"),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            emissiveFactor=list(material.emissive_factor) if any(material.emissive) else None,
            alphaMode="BLEND" if material.transparent else None,
            baseColorFactor=[1.0, 1.0, 1.0, material.opacity] if material.transparent else None,
            doubleSided=True if material.transparent else None,
        )
    return cache[key]


def to_trimesh(mesh: Mesh, cache: dict | None = None) -> trimesh.Trimesh:
    """Convert a geogen Mesh to a trimesh with normals, texture-space UVs and PBR material."""
    cache = {} if cache is None else cache
    mesh = meshops.ensure_normals(mesh)
    visual = None
    if mesh.material is not None and mesh.uvs is not None:
        uv = mesh.uvs * np.asarray(mesh.material.texture_uv_scale)
        visual = trimesh.visual.TextureVisuals(uv=uv, material=_pbr_material(mesh.material, cache))
    tm = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_normals=mesh.normals,
        visual=visual,
        process=False,
    )
    return tm


def resolve_collider(node: SceneNode) -> str:
    """Collider for a mesh node: its ``collider`` meta, or an ``auto`` choice.

    ``auto`` skips tiny parts, uses a box when the mesh fills its bounds, a
    convex hull when it is nearly convex, and the triangle mesh otherwise
    (e.g. walls with openings, hollow shells).
    """
    kind = str(node.meta.get("collider", "auto"))
    if kind != "auto" or node.mesh is None:
        return kind
    mesh = node.mesh
    extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    if extent.max() < MIN_COLLIDER_SIZE:
        return "none"
    if extent.min() < 1e-4:
        return "mesh"  # flat (plane) geometry has no volume to compare
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    volume = abs(tm.volume)
    if volume >= 0.97 * float(np.prod(extent)):
        return "box"
    try:
        hull_volume = tm.convex_hull.volume
    except Exception:  # degenerate input: fall back to the exact mesh
        return "mesh"
    return "hull" if hull_volume > 0 and volume >= 0.9 * hull_volume else "mesh"


def collider_mesh(mesh: Mesh, kind: str) -> trimesh.Trimesh:
    """Collision geometry (in the mesh's own frame) for ``kind`` box|hull|mesh."""
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    if kind == "box":
        return trimesh.creation.box(bounds=tm.bounds)
    if kind == "hull":
        return tm.convex_hull
    return trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)


def node_extras(node: SceneNode, collider: str | None = None) -> dict:
    """``extras`` for one node: {"geogen": {...}} or {} when there is nothing to say."""
    data: dict = {}
    if node.tags:
        data["tags"] = list(node.tags)
    for key, value in node.meta.items():
        if key not in ("collider", "traffic"):     # traffic graphs go in the manifest
            data[key] = value.tolist() if hasattr(value, "tolist") else value
    if collider is not None:
        data["collider"] = collider
    if not data:
        return {}
    return {"geogen": {"version": EXTRAS_VERSION, **data}}


LOD_MIN_TRIANGLES = 200   # meshes smaller than this don't get LODs


def to_trimesh_scene(root: SceneNode, colliders: bool = True, lods: list[float] | None = None) -> trimesh.Scene:
    """Build a trimesh Scene mirroring the SceneNode hierarchy.

    Nodes carry ``extras.geogen``; with ``colliders`` each mesh node that gets
    a collider has a ``<name>-colonly`` / ``<name>-convcolonly`` child.
    Nodes sharing a Mesh object (instances, see ``SceneNode.instance``) share
    one glTF mesh, and so do their colliders and LODs.
    """
    scene = trimesh.Scene(base_frame="world")
    cache: dict = {}
    shared: dict[tuple, str] = {}           # (kind, id(mesh), ...) -> geometry name
    attributes: dict[str, dict] = {}        # geometry name -> {COLOR_0|JOINTS_0|WEIGHTS_0: per-vertex array}
    skins: list[tuple[SceneNode, object]] = []
    groups_of: dict[tuple, int] = {}        # mesh key -> material group count
    collider_kinds: dict[tuple, str] = {}
    lod_sizes: dict[tuple, tuple] = {}
    used: set[str] = {"world"}

    def unique(name: str) -> str:
        candidate, i = name, 1
        while candidate in used:
            i += 1
            candidate = f"{name}_{i}"
        used.add(candidate)
        return candidate

    # Name every node up front so interactions can refer to exported names.
    names: dict[int, str] = {id(n): unique(n.name) for n in root.iter_nodes()}

    def name_of(node: SceneNode) -> str:
        return names[id(node)]

    def add(key: tuple, build, node_name: str, parent: str, matrix=None, metadata=None) -> None:
        """Add a node showing geometry ``key``, building the geometry only the first time."""
        geom = shared.get(key)
        if geom is None:
            shared[key] = node_name
            scene.add_geometry(build(), node_name=node_name, geom_name=node_name, parent_node_name=parent,
                               transform=matrix, metadata=metadata)
        else:
            scene.graph.update(frame_from=parent, frame_to=node_name,
                               matrix=matrix if matrix is not None else np.eye(4), geometry=geom,
                               **({"metadata": metadata} if metadata else {}))

    def vertex_attributes(geom: str, mesh: Mesh) -> None:
        extra = {"COLOR_0": mesh.colors, "JOINTS_0": mesh.joints, "WEIGHTS_0": mesh.weights}
        extra = {k: v for k, v in extra.items() if v is not None}
        if extra:
            attributes[geom] = extra

    def add_mesh(key: tuple, mesh: Mesh, node_name: str, parent: str, matrix=None, metadata=None) -> None:
        """A mesh node; extra material groups become ``material_group`` children, folded into the
        node's glTF mesh as extra primitives after export (``merge_material_groups``)."""
        if key in shared:       # already exported: just another node showing it
            add(key, None, node_name, parent, matrix, metadata)
            for k in range(1, groups_of.get(key, 1)):
                add((*key, "group", k), None, unique(f"{node_name}__group{k}"), node_name,
                    metadata={"geogen": {"version": EXTRAS_VERSION, "type": "material_group"}})
            return
        # Normals first: crease splitting adds vertices, and extra attributes must match them.
        groups = [(m, meshops.ensure_normals(sub)) for m, sub in mesh.groups()]
        groups_of[key] = len(groups)
        first = groups[0][1]
        add(key, lambda: to_trimesh(first, cache), node_name, parent, matrix, metadata)
        vertex_attributes(shared[key], first)
        for k, (_, sub) in enumerate(groups[1:], start=1):
            gkey = (*key, "group", k)
            add(gkey, lambda sub=sub: to_trimesh(sub, cache), unique(f"{node_name}__group{k}"), node_name,
                metadata={"geogen": {"version": EXTRAS_VERSION, "type": "material_group"}})
            vertex_attributes(shared[gkey], sub)

    def collider_of(node: SceneNode) -> str:
        if node.skin is not None:
            return "none"     # a rest-pose collider would be wrong once the skin moves
        key = (id(node.mesh), str(node.meta.get("collider", "auto")))
        if key not in collider_kinds:
            collider_kinds[key] = resolve_collider(node)
        return collider_kinds[key]

    def visit(node: SceneNode, parent_name: str) -> None:
        name = name_of(node)
        matrix = node.transform.to_matrix()
        has_mesh = node.mesh is not None and len(node.mesh.faces) > 0
        collider = collider_of(node) if has_mesh else None
        extras = node_extras(node, collider)
        npc = extras.get("geogen", {}).get("npc")
        if isinstance(npc, dict) and isinstance(npc.get("body"), dict):
            # The body child may be renamed on export (other 'body' nodes): point at the exported name.
            body = next((c for c in node.children if c.name == npc["body"].get("node", "body")), None)
            if body is not None:
                extras["geogen"]["npc"] = {**npc, "body": {**npc["body"], "node": name_of(body)}}
        vehicle = extras.get("geogen", {}).get("vehicle")
        if isinstance(vehicle, dict):
            extras["geogen"]["vehicle"] = _vehicle_exported(vehicle, node, name_of)
        if node.interactions:
            exported = {}
            for interaction in node.interactions:
                try:
                    exported[interaction.name] = interaction.to_extras(name_of)
                except KeyError:   # drives nodes outside this export (a chunk split); runtime can't use it
                    logger.warning("Skipping interaction '%s' on '%s': its nodes aren't all in this export",
                                   interaction.name, node.name)
            if exported:
                extras.setdefault("geogen", {"version": EXTRAS_VERSION})["interactions"] = exported
        if node.clips:
            extras.setdefault("geogen", {"version": EXTRAS_VERSION})["clips"] = [
                {"name": clip.name, "animation": f"{name}_{clip.name}", "duration": round(clip.duration, 6),
                 "loop": clip.loop, **clip.meta} for clip in node.clips]
        if has_mesh:
            mesh = node.mesh
            add_mesh(("mesh", id(mesh)), mesh, name, parent_name, matrix, extras or None)
            if node.skin is not None:
                skins.append((node, node.skin))
            # Decimation doesn't carry skin weights yet, so skinned meshes get no LODs.
            if lods and len(mesh.faces) >= LOD_MIN_TRIANGLES and node.skin is None:
                for level, ratio in enumerate(lods, start=1):
                    key = ("lod", id(mesh), ratio)
                    if key not in lod_sizes:
                        reduced = meshops.decimate(mesh, ratio)
                        lod_sizes[key] = (reduced, len(reduced.faces))
                    reduced, faces = lod_sizes[key]
                    if faces >= len(mesh.faces):
                        continue
                    lod_name = unique(f"{name}_LOD{level}")
                    add_mesh(key, reduced, lod_name, name,
                             metadata={"geogen": {"version": EXTRAS_VERSION, "type": "lod", "level": level,
                                                  "ratio": ratio}})
            if colliders and collider in COLLIDER_SUFFIX:
                col_name = unique(f"{name}{COLLIDER_SUFFIX[collider]}")
                add(("collider", id(mesh), collider), lambda: collider_mesh(mesh, collider), col_name, name,
                    metadata={"geogen": {"version": EXTRAS_VERSION, "type": "collider", "shape": collider}})
        else:
            scene.graph.update(frame_from=parent_name, frame_to=name, matrix=matrix,
                               **({"metadata": extras} if extras else {}))
        for child in node.children:
            visit(child, name)

    visit(root, "world")
    scene.metadata["geogen_names"] = names      # node id -> exported name (for animations)
    scene.metadata["geogen_attributes"] = attributes    # geometry name -> extra vertex attributes
    scene.metadata["geogen_skins"] = [
        {"node": names[id(node)], "name": skin.name, "inverse_bind": skin.inverse_bind,
         "joints": [names.get(id(j)) for j in skin.joints]} for node, skin in skins]
    # Exported names are unique (the post passes look nodes up by name), but joints must keep their
    # own names (Hips, Spine, ...) in every character so skeleton profiles and retargeting match.
    scene.metadata["geogen_joint_names"] = {names[id(j)]: j.name for _, skin in skins for j in skin.joints
                                            if id(j) in names}
    return scene


def manifest_path(model_path: str | Path) -> Path:
    """Where the manifest for an exported model lives (``chair.glb`` -> ``chair.manifest.json``)."""
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}.manifest.json")


def gameplay_summary(root: SceneNode) -> dict[str, list[dict]]:
    """Rooms and spawns in world space, for the manifest.

    Rooms come from ``room_volume`` nodes (centre, size, and the volume's
    world rotation about Y); spawns from ``spawn`` nodes (position and
    ``forward``, the node's +Z axis: the direction the player should face).
    Spawns are ordered shallowest first, so a scene's own spawns come before
    those of nested assets (e.g. every building's ``entrance_spawn`` in a city).
    Scenes with streets or routes add ``traffic``: the lane graph (geogen/traffic.py).
    """
    rooms, spawns = [], []
    for node in root.iter_nodes():
        kind = node.meta.get("type")
        if kind not in ("room_volume", "spawn"):
            continue
        world = node.world_transform()
        position = [round(float(v), 6) for v in world[:3, 3]]
        forward = world[:3, 2] / max(np.linalg.norm(world[:3, 2]), 1e-12)
        if kind == "room_volume":
            room = dict(node.meta.get("room", {}))
            rooms.append({
                **room,
                "node": node.name,
                "center": position,
                "size": [float(v) for v in node.meta.get("size", [0, 0, 0])],
                "yaw_deg": round(float(np.degrees(np.arctan2(world[0, 2], world[2, 2]))), 6),
            })
        else:
            depth, parent = 0, node.parent
            while parent is not None:
                depth, parent = depth + 1, parent.parent
            spawns.append((depth, {"name": node.name, "position": position,
                                   "forward": [round(float(v), 6) for v in forward]}))
    spawns.sort(key=lambda item: item[0])
    out: dict = {"rooms": rooms, "spawns": [spawn for _, spawn in spawns]}
    from .traffic import build_traffic

    traffic = build_traffic(root)       # the lane graph (geogen-traffic v1), world space
    if traffic is not None:
        out["traffic"] = traffic
    return out


def write_manifest(model_path: str | Path, player: PlayerSpec | None = None,
                   root: SceneNode | None = None) -> Path:
    """Write the runtime manifest next to an exported glTF model."""
    model_path = Path(model_path)
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "name": model_path.stem,
        "model": model_path.name,
        "units": "m",
        "up": "+Y",
        "extras": {"key": "geogen", "version": EXTRAS_VERSION, "schema": EXTRAS_SCHEMA},
        "player": (player or load_player_spec()).to_dict(),
        **(gameplay_summary(root) if root is not None else {"rooms": [], "spawns": []}),
    }
    out = manifest_path(model_path)
    # Written last and atomically: runtimes watch the manifest to know an export finished.
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(out)
    return out


# glTF KHR_lights_punctual point intensity (candela) per unit of our light energy.
CANDELA_PER_ENERGY = 12.0


ANIMATION_KEYS = 12       # samples per transition (rotation about a pivot moves the origin on an arc)


def interaction_animations(root: SceneNode, names: dict[int, str]) -> list[dict]:
    """One animation per interaction transition (``next``/``then`` edges), for engines
    that don't read extras.geogen: ``<asset>/<interaction>/<from>-><to>``, sampling the
    moving parts' local translation/rotation over the interaction's duration.
    """
    from .layout.interactions import _to_transform

    animations = []
    for owner in root.iter_nodes():
        for it in owner.interactions:
            edges = []
            for state_name, state in it.states.items():
                for to in (state.next, state.then):
                    if to and (state_name, to) not in edges:
                        edges.append((state_name, to))
            owner_world = owner.world_transform()
            owner_inv = np.linalg.inv(owner_world)
            for a, b in edges:
                channels: dict[int, dict] = {}
                times = np.linspace(0.0, float(it.duration), ANIMATION_KEYS)
                for motion in it.motions:
                    va, vb = it.value(motion, a), it.value(motion, b)
                    undo = np.linalg.inv(motion.matrix(motion.applied))
                    for part in motion.parts:
                        if id(part) not in names:
                            continue
                        in_owner = owner_inv @ part.world_transform()
                        parent_world = part.parent.world_transform() if part.parent is not None else np.eye(4)
                        to_parent = np.linalg.inv(parent_world) @ owner_world
                        poses = []
                        for t in np.linspace(0.0, 1.0, ANIMATION_KEYS):
                            v = va + (vb - va) * t
                            local = to_parent @ motion.matrix(v) @ undo @ in_owner
                            tr = _to_transform(local, part.transform.scale)
                            poses.append((tr.translation, _quat(tr.to_matrix()[:3, :3] / tr.scale[None, :])))
                        channels[id(part)] = {"node": names[id(part)], "times": times,
                                              "translation": np.array([p[0] for p in poses]),
                                              "rotation": np.array([p[1] for p in poses]),
                                              "scale": part.transform.scale}
                if channels:
                    animations.append({"name": f"{names[id(owner)]}/{it.name}/{a}->{b}",
                                       "channels": list(channels.values())})
    return animations


def add_animations(glb: bytes, animations: list[dict]) -> bytes:
    """Append glTF animations (see ``interaction_animations``) to a GLB.

    Animated nodes are switched from ``matrix`` to TRS, as the spec requires.
    """
    import struct

    if not animations:
        return glb
    gltf, rest, header = _split_glb(glb)
    bin_len, bin_type = struct.unpack("<II", rest[:8]) if rest else (0, 0x004E4942)
    data = bytearray(rest[8:8 + bin_len])
    views, accessors = gltf.setdefault("bufferViews", []), gltf.setdefault("accessors", [])
    index = {n.get("name"): i for i, n in enumerate(gltf.get("nodes", []))}

    def accessor(values: np.ndarray, kind: str, with_bounds: bool = False) -> int:
        blob = np.ascontiguousarray(values, dtype="<f4").tobytes()
        data.extend(b"\0" * (-len(data) % 4))
        views.append({"buffer": 0, "byteOffset": len(data), "byteLength": len(blob)})
        data.extend(blob)
        acc = {"bufferView": len(views) - 1, "componentType": 5126, "count": int(values.shape[0]), "type": kind}
        if with_bounds:
            acc["min"] = [float(values.min())]
            acc["max"] = [float(values.max())]
        accessors.append(acc)
        return len(accessors) - 1

    out = gltf.setdefault("animations", [])
    for anim in animations:
        samplers, channels = [], []
        for ch in anim["channels"]:
            node = index.get(ch["node"])
            if node is None:
                continue
            _to_trs(gltf["nodes"][node], ch["scale"])
            time = accessor(ch["times"], "SCALAR", with_bounds=True)
            for path, values, kind in (("translation", ch["translation"], "VEC3"),
                                       ("rotation", ch["rotation"], "VEC4")):
                samplers.append({"input": time, "output": accessor(values, kind), "interpolation": "LINEAR"})
                channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})
        if channels:
            out.append({"name": anim["name"], "samplers": samplers, "channels": channels})
    data.extend(b"\0" * (-len(data) % 4))
    if not gltf.get("buffers"):
        gltf["buffers"] = [{"byteLength": 0}]
    gltf["buffers"][0]["byteLength"] = len(data)
    return _join_glb(gltf, struct.pack("<II", len(data), bin_type) + bytes(data), header)


def _to_trs(node: dict, scale: np.ndarray) -> None:
    """Replace a glTF node's ``matrix`` with translation/rotation/scale."""
    if "matrix" not in node:
        return
    m = np.array(node.pop("matrix"), dtype=np.float64).reshape(4, 4).T     # column-major
    s = np.linalg.norm(m[:3, :3], axis=0)
    node["translation"] = [float(v) for v in m[:3, 3]]
    node["rotation"] = [float(v) for v in _quat(m[:3, :3] / np.where(s == 0, 1, s)[None, :])]
    if not np.allclose(s, 1.0):
        node["scale"] = [float(v) for v in s]


# glTF accessor layout per extra vertex attribute: (numpy dtype, componentType, clamp to 0-1).
VERTEX_ATTRIBUTES = {"COLOR_0": ("<f4", 5126, True), "JOINTS_0": ("<u2", 5123, False),
                     "WEIGHTS_0": ("<f4", 5126, False)}


def add_vertex_attributes(glb: bytes, attributes: dict[str, dict[str, np.ndarray]]) -> bytes:
    """Add per-vertex VEC4 attributes (COLOR_0, JOINTS_0, WEIGHTS_0) to the primitives of the named glTF meshes."""
    import struct

    if not attributes:
        return glb
    gltf, rest, header = _split_glb(glb)
    bin_len, bin_type = struct.unpack("<II", rest[:8])
    data = bytearray(rest[8:8 + bin_len])
    for mesh in gltf.get("meshes", []):
        for semantic, values in attributes.get(mesh.get("name"), {}).items():
            dtype, component, clamp = VERTEX_ATTRIBUTES[semantic]
            if clamp:
                values = np.clip(values, 0, 1)
            blob = np.ascontiguousarray(values, dtype=dtype).tobytes()
            data.extend(b"\0" * (-len(data) % 4))
            gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(data), "byteLength": len(blob),
                                        "target": 34962})
            data.extend(blob)
            gltf["accessors"].append({"bufferView": len(gltf["bufferViews"]) - 1, "componentType": component,
                                      "count": int(len(values)), "type": "VEC4"})
            for primitive in mesh.get("primitives", []):
                primitive.setdefault("attributes", {})[semantic] = len(gltf["accessors"]) - 1
    data.extend(b"\0" * (-len(data) % 4))
    gltf["buffers"][0]["byteLength"] = len(data)
    return _join_glb(gltf, struct.pack("<II", len(data), bin_type) + bytes(data), header)


def add_skins(glb: bytes, skins: list[dict]) -> bytes:
    """Add glTF skins (joints, inverseBindMatrices) and point each skinned node at its skin.

    A skin whose joints aren't all in this export (a chunk split) is skipped:
    the mesh then shows its bind pose, unskinned.
    """
    import struct

    if not skins:
        return glb
    gltf, rest, header = _split_glb(glb)
    bin_len, bin_type = struct.unpack("<II", rest[:8])
    data = bytearray(rest[8:8 + bin_len])
    nodes = gltf.get("nodes", [])
    index = {n.get("name"): i for i, n in enumerate(nodes)}
    parent_of = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    out = gltf.setdefault("skins", [])
    for skin in skins:
        joints = [index.get(name) for name in skin["joints"]]
        if skin["node"] not in index or None in joints:
            logger.warning("Skipping skin on '%s': its joints aren't all in this export", skin["node"])
            continue
        # Column-major MAT4s.
        blob = np.ascontiguousarray(np.transpose(skin["inverse_bind"], (0, 2, 1)), dtype="<f4").tobytes()
        data.extend(b"\0" * (-len(data) % 4))
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(data), "byteLength": len(blob)})
        data.extend(blob)
        gltf["accessors"].append({"bufferView": len(gltf["bufferViews"]) - 1, "componentType": 5126,
                                  "count": len(joints), "type": "MAT4"})
        joint_set = set(joints)
        roots = [j for j in joints if parent_of.get(j) not in joint_set]
        entry = {"name": skin["name"], "joints": joints, "inverseBindMatrices": len(gltf["accessors"]) - 1}
        if len(roots) == 1:
            entry["skeleton"] = roots[0]
        out.append(entry)
        nodes[index[skin["node"]]]["skin"] = len(out) - 1
    data.extend(b"\0" * (-len(data) % 4))
    gltf["buffers"][0]["byteLength"] = len(data)
    return _join_glb(gltf, struct.pack("<II", len(data), bin_type) + bytes(data), header)


def rename_nodes(glb: bytes, renames: dict[str, str]) -> bytes:
    """Rename glTF nodes (exported name -> new name); run after every pass that finds nodes by name."""
    renames = {k: v for k, v in renames.items() if k != v}
    if not renames:
        return glb
    gltf, rest, header = _split_glb(glb)
    for node in gltf.get("nodes", []):
        if node.get("name") in renames:
            node["name"] = renames[node["name"]]
    return _join_glb(gltf, rest, header)


def clip_animations(root: SceneNode, names: dict[int, str]) -> list[dict]:
    """One animation per ``SceneNode.clips`` entry, ``<owner>_<clip>``, targeting the joint nodes.

    Channels not animated by a track (rotation or translation) hold the joint's current value.
    """
    animations = []
    for owner in root.iter_nodes():
        if not owner.clips:
            continue
        by_name = {n.name: n for n in owner.iter_nodes()}
        for clip in owner.clips:
            channels = []
            for joint_name, track in clip.tracks.items():
                joint = by_name.get(joint_name)
                if joint is None or id(joint) not in names:
                    continue
                count = len(track.times)
                m = joint.transform.to_matrix()
                rotation = track.rotations if track.rotations is not None else np.tile(
                    _quat(m[:3, :3] / joint.transform.scale[None, :]), (count, 1))
                translation = track.translations if track.translations is not None else np.tile(
                    joint.transform.translation, (count, 1))
                channels.append({"node": names[id(joint)], "times": track.times, "rotation": rotation,
                                 "translation": translation, "scale": joint.transform.scale})
            if channels:
                animations.append({"name": f"{names[id(owner)]}_{clip.name}", "channels": channels})
    return animations


def merge_material_groups(glb: bytes) -> bytes:
    """Fold ``material_group`` child nodes into their parent's glTF mesh as extra primitives.

    The child nodes are detached (left unreferenced); shared meshes are extended once.
    """
    gltf, rest, header = _split_glb(glb)
    nodes = gltf.get("nodes", [])
    parent_of = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    done: set[tuple[int, int]] = set()
    changed = False
    for index, node in enumerate(nodes):
        extras = (node.get("extras") or {}).get("geogen") or {}
        if extras.get("type") != "material_group" or index not in parent_of:
            continue
        parent = nodes[parent_of[index]]
        if "mesh" in parent and "mesh" in node and (parent["mesh"], node["mesh"]) not in done:
            gltf["meshes"][parent["mesh"]]["primitives"].extend(gltf["meshes"][node["mesh"]]["primitives"])
            done.add((parent["mesh"], node["mesh"]))
        parent["children"] = [c for c in parent["children"] if c != index]
        if not parent["children"]:
            del parent["children"]
        changed = True
    return _join_glb(gltf, rest, header) if changed else glb


def add_lod_extension(glb: bytes) -> bytes:
    """Turn ``<name>_LOD<n>`` child nodes (extras type lod) into MSFT_lod on their base node.

    The LOD nodes are detached from the hierarchy (MSFT_lod references them
    by index) so engines without the extension simply show the full mesh.
    """
    gltf, rest, header = _split_glb(glb)
    nodes = gltf.get("nodes", [])
    used = False
    for base in nodes:
        lod_ids = [i for i in base.get("children", [])
                   if nodes[i].get("extras", {}).get("geogen", {}).get("type") == "lod"]
        if not lod_ids:
            continue
        lod_ids.sort(key=lambda i: nodes[i]["extras"]["geogen"]["level"])
        base["children"] = [i for i in base["children"] if i not in lod_ids]
        if not base["children"]:
            del base["children"]
        # Detached LOD nodes sit at the base's origin: give them its transform.
        for i in lod_ids:
            for key in ("matrix", "translation", "rotation", "scale"):
                if key in base:
                    nodes[i][key] = base[key]
        base.setdefault("extensions", {})["MSFT_lod"] = {"ids": lod_ids}
        coverage = [0.5 ** (2 * (k + 1)) for k in range(len(lod_ids) + 1)]
        base.setdefault("extras", {})["MSFT_screencoverage"] = coverage
        used = True
    if not used:
        return glb
    gltf.setdefault("extensionsUsed", [])
    if "MSFT_lod" not in gltf["extensionsUsed"]:
        gltf["extensionsUsed"].append("MSFT_lod")
    return _join_glb(gltf, rest, header)


def _split_glb(glb: bytes):
    import struct

    magic, version, _ = struct.unpack("<III", glb[:12])
    json_len, json_type = struct.unpack("<II", glb[12:20])
    return json.loads(glb[20:20 + json_len]), glb[20 + json_len:], (magic, version, json_type)


def _join_glb(gltf: dict, rest: bytes, header) -> bytes:
    import struct

    magic, version, json_type = header
    payload = json.dumps(gltf, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    body = struct.pack("<II", len(payload), json_type) + payload + rest
    return struct.pack("<III", magic, version, 12 + len(body)) + body


def externalize_images(glb: bytes, textures_dir: str | Path, uri_prefix: str = "") -> bytes:
    """Move a GLB's embedded images to ``textures_dir`` (named by content hash) and refer to them by URI.

    Chunk files share one texture folder this way instead of each embedding
    the same maps; the binary buffer is repacked without the image data.
    """
    import hashlib
    import struct

    gltf, rest, header = _split_glb(glb)
    images = gltf.get("images") or []
    if not rest or not any("bufferView" in image for image in images):
        return glb
    bin_len, bin_type = struct.unpack("<II", rest[:8])
    data = rest[8:8 + bin_len]
    views = gltf["bufferViews"]
    textures_dir = Path(textures_dir)
    textures_dir.mkdir(parents=True, exist_ok=True)
    moved: set[int] = set()
    for image in images:
        if "bufferView" not in image:
            continue
        view = views[image["bufferView"]]
        start = view.get("byteOffset", 0)
        blob = data[start:start + view["byteLength"]]
        extension = {"image/jpeg": ".jpg"}.get(image.get("mimeType", ""), ".png")
        file_name = hashlib.sha1(blob).hexdigest()[:20] + extension
        if not (textures_dir / file_name).exists():
            (textures_dir / file_name).write_bytes(blob)
        moved.add(image.pop("bufferView"))
        image["uri"] = uri_prefix + file_name
    packed, remap, kept = bytearray(), {}, []
    for index, view in enumerate(views):
        if index in moved:
            continue
        start = view.get("byteOffset", 0)
        packed += b"\0" * (-len(packed) % 4)
        remap[index] = len(kept)
        kept.append({**view, "byteOffset": len(packed)})
        packed += data[start:start + view["byteLength"]]
    packed += b"\0" * (-len(packed) % 4)
    for accessor in gltf.get("accessors", []):
        if "bufferView" in accessor:
            accessor["bufferView"] = remap[accessor["bufferView"]]
    gltf["bufferViews"] = kept
    gltf["buffers"][0]["byteLength"] = len(packed)
    return _join_glb(gltf, struct.pack("<II", len(packed), bin_type) + bytes(packed), header)


def add_punctual_lights(glb: bytes) -> bytes:
    """Add KHR_lights_punctual lights for nodes whose extras.geogen has a ``light``.

    trimesh can't write the extension, so the GLB's JSON chunk is patched: a
    point light per fixture on a ``<node>_light`` child at the light's offset.
    Engines with KHR support (Godot, Blender, three.js) get real lights.
    """
    import struct

    magic, version, _ = struct.unpack("<III", glb[:12])
    json_len, json_type = struct.unpack("<II", glb[12:20])
    gltf = json.loads(glb[20:20 + json_len])
    rest = glb[20 + json_len:]
    lights = []
    nodes = gltf.get("nodes", [])
    for index in range(len(nodes)):
        spec = nodes[index].get("extras", {}).get("geogen", {}).get("light")
        if not isinstance(spec, dict):
            continue
        lights.append({
            "type": "point",
            "name": f"{nodes[index].get('name', 'light')}_light",
            "color": [float(c) for c in spec.get("color", [1.0, 1.0, 1.0])],
            "intensity": float(spec.get("energy", 1.0)) * CANDELA_PER_ENERGY,
            "range": float(spec.get("range", 5.0)),
        })
        child = {"name": lights[-1]["name"], "translation": [float(v) for v in spec.get("offset", [0, 0, 0])],
                 "extensions": {"KHR_lights_punctual": {"light": len(lights) - 1}},
                 "extras": {"geogen": {"version": EXTRAS_VERSION, "type": "light"}}}
        nodes.append(child)
        nodes[index].setdefault("children", []).append(len(nodes) - 1)
    if not lights:
        return glb
    gltf.setdefault("extensions", {})["KHR_lights_punctual"] = {"lights": lights}
    used = gltf.setdefault("extensionsUsed", [])
    if "KHR_lights_punctual" not in used:
        used.append("KHR_lights_punctual")
    payload = json.dumps(gltf, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    body = struct.pack("<II", len(payload), json_type) + payload + rest
    return struct.pack("<III", magic, version, 12 + len(body)) + body


def _vehicle_exported(vehicle: dict, node: SceneNode, name_of) -> dict:
    """``meta.vehicle`` with part names replaced by their exported (uniquified) node names."""
    by_name: dict[str, SceneNode] = {}
    for n in node.iter_nodes(include_self=False):
        by_name.setdefault(n.name, n)

    def rename(part: str) -> str:
        return name_of(by_name[part]) if part in by_name else part

    out = dict(vehicle)
    out["wheels"] = [{**w, "part": rename(w["part"])} for w in vehicle.get("wheels", [])]
    for key in ("steer", "paint", "bogies", "couplers"):
        if key in vehicle:
            out[key] = [rename(p) for p in vehicle[key]]
    if "lamps" in vehicle:
        out["lamps"] = {k: [rename(p) for p in v] for k, v in vehicle["lamps"].items()}
    return out


def export_scene(root: SceneNode, path: str | Path, player: PlayerSpec | None = None,
                 lods: list[float] | None = None, manifest: bool = True,
                 textures_dir: str | Path | None = None, animations: bool = True) -> Path:
    """Export ``root`` to ``path``; the format is chosen by the file extension.

    glTF/GLB exports also get a manifest (see ``write_manifest``); ``player``
    overrides the project's default player spec in it. ``lods`` (GLB only),
    e.g. ``[0.5, 0.25]``, adds decimated levels per mesh as MSFT_lod.
    ``manifest=False`` skips the manifest (chunk files are indexed by
    ``chunks.export_chunks`` instead). GLBs also carry a glTF animation per
    interaction transition (``animations=False`` to skip) for engines that
    don't read extras.geogen, and one per skeletal clip (``SceneNode.clips``,
    always written). Skinned nodes become glTF skins. ``textures_dir`` (GLB only) writes
    images there, shared by content hash, instead of embedding them.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        raise ValueError(f"Unsupported export format '{suffix}'. Use one of {sorted(FORMATS)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # OBJ has no node extras or hierarchy, so collider meshes would just be clutter.
    scene = to_trimesh_scene(root, colliders=suffix != ".obj", lods=lods if suffix == ".glb" else None)
    if suffix == ".obj":
        # OBJ has no hierarchy: bake world transforms. trimesh writes the
        # .mtl and texture images alongside the .obj.
        scene.export(str(path), file_type="obj", include_normals=True, include_texture=True)
    elif suffix == ".glb":
        # Write-then-rename so a runtime watching the file never reads half a GLB.
        tmp = path.with_name(path.name + ".tmp")
        glb = add_lod_extension(add_punctual_lights(scene.export(file_type="glb")))
        glb = merge_material_groups(add_vertex_attributes(glb, scene.metadata["geogen_attributes"]))
        glb = add_skins(glb, scene.metadata["geogen_skins"])
        names = scene.metadata["geogen_names"]
        glb = add_animations(glb, (interaction_animations(root, names) if animations else [])
                             + clip_animations(root, names))
        glb = rename_nodes(glb, scene.metadata["geogen_joint_names"])
        if textures_dir is not None:
            import os

            prefix = Path(os.path.relpath(textures_dir, path.parent)).as_posix() + "/"
            glb = externalize_images(glb, textures_dir, prefix)
        tmp.write_bytes(glb)
        tmp.replace(path)
        if manifest:
            write_manifest(path, player, root)
    else:
        scene.export(str(path), file_type=suffix[1:])
        if manifest:
            write_manifest(path, player, root)
    return path
