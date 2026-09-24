@tool
extends EditorPlugin
## Registers the post-import step so .glb files exported by geogen get their
## gameplay nodes (room areas, spawn markers, tag groups) when the editor imports them.

var _post_import: GeogenPostImport


func _enter_tree() -> void:
	_post_import = GeogenPostImport.new()
	add_scene_post_import_plugin(_post_import)


func _exit_tree() -> void:
	remove_scene_post_import_plugin(_post_import)
	_post_import = null
