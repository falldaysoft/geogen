class_name GeogenWorldNavigation
extends RefCounted
## A loaded export's navmesh (WorldLoader.navigation): baked by the scene builder, or the one baked
## last time the same export loaded with the same settings (<generated>/navcache), synced in step
## with physics frames. Streamed exports bake per tile instead (chunk_streamer.gd).

var world: WorldLoader


## A whole export's navmesh syncs in step with the physics frames. Godot syncs maps and
## regions on worker threads by default, so under load the first synced frame (when NPCs start
## deciding) varied run to run and long simulations drifted apart (geogen.50). It is all
## there at load, so the one synchronous sync costs no more; streamed exports stay async.
func in_step(nav: Node) -> void:
    NavigationServer3D.map_set_use_async_iterations(world.get_world_3d().navigation_map, false)
    for region in [nav] + nav.find_children("*", "NavigationRegion3D", true, false):
        if region is NavigationRegion3D:
            NavigationServer3D.region_set_use_async_iterations(region.get_rid(), false)


## Bake the model's navmesh, or reuse the one baked last time the same export loaded with the
## same settings (<generated>/navcache/<name>-<key>.scn, keyed by the export's files, the
## player spec, the bake options and the baking scripts' source).
func build(root: Node3D, manifest_path: String, model_path: String) -> void:
    var spec := PlayerSpec.from_manifest(manifest_path)
    var path := _cache_path(root, manifest_path, model_path, spec)
    if path != "" and FileAccess.file_exists(path):
        var packed := ResourceLoader.load(path, "PackedScene", ResourceLoader.CACHE_MODE_IGNORE) as PackedScene
        if packed != null:
            var nav := packed.instantiate()
            root.add_child(nav)
            for region in [nav] + nav.find_children("*", "NavigationRegion3D", true, false):
                if region is NavigationRegion3D:
                    region.add_to_group("geogen_navigation")
            in_step(nav)
            world.load_timings["nav_cache"] = "hit"
            return
    # Only some doors opened (open_only): the rest stay out of the bake as usual (their doorways
    # navigable), while the opened ones' leaves are obstacles where they stand.
    var shut: Array[CollisionObject3D] = []
    if world.open_before_bake and world.open_only != "":
        for it in world.interactions:
            if (it.asset.is_ancestor_of(root) or root.is_ancestor_of(it.asset)) and not it.hold:
                for body in it.asset.find_children("*", "AnimatableBody3D", true, false):
                    if body.collision_layer & 1:
                        body.collision_layer &= ~1
                        shut.append(body)
    var nav := GeogenSceneBuilder.build_navigation(root, spec, world.open_before_bake, world.nav_ground_margin())
    in_step(nav)
    for body in shut:
        body.collision_layer |= 1
    world.load_timings["nav_cache"] = "miss"
    if path == "":
        return
    for child in nav.find_children("*", "", true, false):
        child.owner = nav
    var packed := PackedScene.new()
    if packed.pack(nav) != OK:
        return
    var dir := path.get_base_dir()
    DirAccess.make_dir_recursive_absolute(dir)
    var prefix := path.get_file().get_slice("-", 0) + "-"
    for old in DirAccess.get_files_at(dir):
        if old.begins_with(prefix):
            DirAccess.remove_absolute(dir.path_join(old))    # older bakes of this export
    var err := ResourceSaver.save(packed, path, ResourceSaver.FLAG_COMPRESS)
    if err != OK:
        push_warning("geogen: can't cache navmesh %s (%s)" % [path, error_string(err)])


func _cache_path(root: Node3D, manifest_path: String, model_path: String, spec: PlayerSpec) -> String:
    if not world.nav_cache:
        return ""
    var model_md5 := FileAccess.get_md5(model_path)
    if model_md5 == "":
        return ""
    # A res:// export by its res:// path, so a bake made before exporting the project (the web
    # build's, tools/export_web.py) matches the exported project's.
    var where := model_path if model_path.begins_with("res://") else ProjectSettings.globalize_path(model_path)
    var key := JSON.stringify([WorldLoader.NAV_CACHE_VERSION, where,
        model_md5, FileAccess.get_md5(manifest_path),
        spec.to_dict() if spec else {}, world.open_before_bake, world.open_only, world.nav_ground_margin(), var_to_str(root.global_transform),
        FileAccess.get_md5("res://addons/geogen/scene_builder.gd"), FileAccess.get_md5("res://scripts/world_loader.gd"),
        FileAccess.get_md5("res://scripts/world_navigation.gd")])
    return world.generated_dir.path_join("navcache").path_join("%s-%s.scn" % [String(root.name).replace("-", "_"),
        key.md5_text()])
