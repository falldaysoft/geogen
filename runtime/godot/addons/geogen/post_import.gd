@tool
class_name GeogenPostImport
extends EditorScenePostImportPlugin
## Editor import: once a glTF scene is fully imported, apply GeogenSceneBuilder
## (rooms, spawns, tag groups). Runs on the final scene, after the importer
## has replaced its intermediate nodes, so the added nodes and groups are saved.
## (The runtime WorldLoader calls the builder directly for runtime loads.)


func _post_process(scene: Node) -> void:
	if GeogenSceneBuilder.is_geogen_scene(scene):
		var summary := GeogenSceneBuilder.build(scene)
		print("geogen: imported %s (%d rooms, %d spawns, %d tagged)" % [
			scene.name, summary["rooms"], summary["spawns"], summary["tagged"]])
