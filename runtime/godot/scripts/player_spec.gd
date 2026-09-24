class_name PlayerSpec
extends RefCounted
## Player-scale constraints, read from a geogen export manifest
## (`<name>.manifest.json`, key "player"). Mirrors geogen.player.PlayerSpec;
## the source of truth is assets/player.yaml in the geogen repo.

const MANIFEST_FORMAT := "geogen-manifest"
const SPEC_KIND := "player_spec"
const SPEC_VERSION := 1

# Fallback values, used when no manifest is available. Keep in sync with
# assets/player.yaml (tests/test_player.py::test_godot_reads_manifest checks
# the manifest path, not these).
var radius := 0.3
var height := 1.8
var eye_height := 1.65
var step_height := 0.3
var max_slope_deg := 40.0
var door_min_width := 0.85
var door_min_height := 2.0
var corridor_min_width := 1.2
var reach := 1.5


## Load the player spec from a manifest file. Returns null (and pushes an
## error) if the file is missing or isn't a compatible geogen manifest.
static func from_manifest(path: String) -> PlayerSpec:
	var text := FileAccess.get_file_as_string(path)
	if text.is_empty():
		push_error("PlayerSpec: cannot read manifest %s" % path)
		return null
	var manifest = JSON.parse_string(text)
	if not manifest is Dictionary or not manifest.get("format") in [MANIFEST_FORMAT, "geogen-chunks"]:
		push_error("PlayerSpec: %s is not a geogen manifest" % path)
		return null
	return from_dict(manifest.get("player", {}))


static func from_dict(data: Dictionary) -> PlayerSpec:
	var spec := PlayerSpec.new()
	if data.get("kind", SPEC_KIND) != SPEC_KIND or int(data.get("version", SPEC_VERSION)) != SPEC_VERSION:
		push_error("PlayerSpec: unsupported player spec %s v%s" % [data.get("kind"), data.get("version")])
		return null
	for key in data:
		if key in ["kind", "version"]:
			continue
		if key in spec:
			spec.set(key, float(data[key]))
		else:
			push_warning("PlayerSpec: ignoring unknown key '%s'" % key)
	return spec


## Walkable slope in radians, for CharacterBody3D.floor_max_angle.
func max_slope_rad() -> float:
	return deg_to_rad(max_slope_deg)


func to_dict() -> Dictionary:
	return {
		"radius": radius, "height": height, "eye_height": eye_height,
		"step_height": step_height, "max_slope_deg": max_slope_deg,
		"door_min_width": door_min_width, "door_min_height": door_min_height,
		"corridor_min_width": corridor_min_width, "reach": reach,
	}
