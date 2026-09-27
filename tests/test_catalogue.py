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
