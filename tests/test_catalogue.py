"""The runtime scene catalogue (assets/runtime_scenes.yaml) and its catalogue.json index."""

import json

import pytest

from geogen.catalogue import (CATALOGUE_PATH, Catalogue, CatalogueEntry, export_catalogue, load_catalogue,
                              write_index)


def test_catalogue_scenes_are_registered(scene_registry):
    catalogue = load_catalogue()
    assert catalogue.default in catalogue.entries
    missing = [name for name in catalogue.entries if name not in scene_registry]
    assert missing == []
    assert {e.group for e in catalogue.entries.values()} <= {"showcase", "test"}


def test_registry_skips_the_catalogue(scene_registry):
    assert CATALOGUE_PATH.stem not in scene_registry


def test_select_by_group_and_name():
    catalogue = load_catalogue()
    assert all(e.group == "showcase" for e in catalogue.select("showcase"))
    assert [e.name for e in catalogue.select("cottage,door")] == ["cottage", "door"]
    assert len(catalogue.select()) == len(catalogue.entries)
    with pytest.raises(ValueError, match="not in the catalogue"):
        catalogue.select("no_such_scene")


def test_bad_catalogues_are_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("kind: scene_catalogue\ndefault: b\nscenes:\n  a: {group: showcase}\n")
    with pytest.raises(ValueError, match="default"):
        load_catalogue(path)
    path.write_text("kind: scene_catalogue\ndefault: a\nscenes:\n  a: {group: demo}\n")
    with pytest.raises(ValueError, match="group"):
        load_catalogue(path)
    path.write_text("kind: scene_catalogue\ndefault: a\nscenes:\n  a: {strem: true}\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_catalogue(path)


def test_index_lists_entries_and_what_is_exported(tmp_path):
    catalogue = Catalogue("a", {"a": CatalogueEntry("a", "showcase", "A"),
                                "big": CatalogueEntry("big", "showcase", "Big", stream=True)})
    (tmp_path / "a.manifest.json").write_text("{}")
    index = json.loads(write_index(catalogue, tmp_path).read_text())
    assert index["format"] == "geogen-catalogue" and index["default"] == "a"
    by_name = {s["name"]: s for s in index["scenes"]}
    assert by_name["a"]["exported"] and by_name["a"]["manifest"] == "a.manifest.json"
    assert not by_name["big"]["exported"] and by_name["big"]["manifest"] == "big_chunks/big.chunks.json"


@pytest.fixture(scope="module")
def catalogue_dir(tmp_path_factory, built_scene):
    out = tmp_path_factory.mktemp("generated")
    catalogue = Catalogue("dining_set", {"door": CatalogueEntry("door"),
                                         "dining_set": CatalogueEntry("dining_set", "showcase")})
    export_catalogue(catalogue, built_scene, out, log=lambda *_: None)
    return out


def test_runtime_loads_the_catalogue_default(run_godot, catalogue_dir):
    out = run_godot(f"--generated={catalogue_dir}", "--quit-after=3")
    assert "default scene dining_set" in out
    assert "loaded dining_set.glb" in out and "loaded door.glb" not in out


def test_runtime_scene_all_ignores_the_catalogue(run_godot, catalogue_dir):
    out = run_godot(f"--generated={catalogue_dir}", "--scene=all", "--quit-after=3")
    assert "loaded dining_set.glb" in out and "loaded door.glb" in out


def _json_lines(out: str, prefix: str) -> list[dict]:
    return [json.loads(line.removeprefix(prefix)) for line in out.splitlines() if line.startswith(prefix)]


def test_runtime_lists_scenes(run_godot, catalogue_dir):
    out = run_godot(f"--generated={catalogue_dir}", "--list-scenes")
    listing = _json_lines(out, "scenes: ")[0]
    assert listing["default"] == "dining_set"
    assert {s["name"]: s["exported"] for s in listing["scenes"]} == {"door": True, "dining_set": True}
    assert "loaded" not in out   # listing doesn't load a world


def test_switching_scenes_leaves_nothing_behind(run_godot, catalogue_dir):
    out = run_godot(f"--generated={catalogue_dir}", "--switch=door@0.5", "--switch=dining_set@1",
                    "--switch=door@1.5", "--switch=dining_set@2", "--switch=missing@2.5")
    switched = _json_lines(out, "switched: ")
    assert [s["scene"] for s in switched] == ["door", "dining_set", "door", "dining_set", "missing"]
    assert switched[-1]["error"] == "not exported"
    stats = _json_lines(out, "switch stats: ")
    by_scene = {}
    for s in stats[:4]:
        by_scene.setdefault(s["scene"], []).append(s)
    for scene, visits in by_scene.items():
        first, second = visits
        # The second visit to a scene has exactly the nodes, bodies and nav regions of the first.
        for key in ("nodes", "bodies", "nav_regions", "world_children"):
            assert first[key] == second[key], (scene, key, first, second)
        assert abs(first["objects"] - second["objects"]) <= 5, (scene, first, second)
        assert second["orphans"] == 0
    assert "leaked" not in out


def test_switch_then_walk_in_the_new_world(run_godot, catalogue_dir):
    out = run_godot(f"--generated={catalogue_dir}", "--switch=door@0.5", "--walk=0.6")
    assert "loaded dining_set.glb" in out and "loaded door.glb" in out
    start = _json_lines(out, "switched: ")[0]
    end = _json_lines(out, "walk result: ")[0]
    assert end["on_floor"]
    moved = ((end["x"] - start["x"]) ** 2 + (end["z"] - start["z"]) ** 2) ** 0.5
    assert moved > 1.0, (start, end)


def test_second_load_reuses_the_baked_navmesh(run_godot, built_scene, tmp_path):
    from geogen.export import export_scene

    export_scene(built_scene("dining_set"), tmp_path / "dining_set.glb")
    args = ("--scene=dining_set", f"--generated={tmp_path}", "--timings")
    first = _json_lines(run_godot(*args, "--quit-after=3"), "load timings: ")[0]
    assert first["phases"]["nav_cache"] == "miss" and "gltf_parse" in first["phases"]
    assert list((tmp_path / ".navcache").glob("dining_set-*.scn"))
    out = run_godot(*args, "--nav=-2,3:2,-3")
    second = _json_lines(out, "load timings: ")[0]
    assert second["phases"]["nav_cache"] == "hit"
    assert second["phases"]["navigation"] < first["phases"]["navigation"]
    assert _json_lines(out, "nav path: ")[0]["reached"]         # the cached navmesh paths around the table
    # A changed export (another scene under the same name) misses and replaces the old bake.
    export_scene(built_scene("door"), tmp_path / "dining_set.glb")
    third = _json_lines(run_godot(*args, "--quit-after=3"), "load timings: ")[0]
    assert third["phases"]["nav_cache"] == "miss"
    assert len(list((tmp_path / ".navcache").glob("dining_set-*.scn"))) == 1
