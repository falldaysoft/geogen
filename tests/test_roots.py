"""Content roots (geogen/roots.py): a project's assets layered over the core's."""

from pathlib import Path

import numpy as np
import pytest

from geogen import roots
from geogen.layout.cache import source_fingerprint
from geogen.layout.composer import SceneComposer
from geogen.quests import collect
from geogen.registry import SceneRegistry
from geogen.rules import Machine

EXAMPLE = Path(__file__).parent.parent / "examples" / "sandwich"

WIDGET = """
# A project asset in a project material.
name: widget
tags: [puzzle.widget]
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
parts:
  block: {primitive: cube, size: [0.3, 0.3, 0.3], anchor: bottom_center, material: unobtainium}
interactions:
  press:
    targets: [block]
    states:
      up:   {next: down}
      down: {next: up}
    motions:
      - {parts: [block], translate: y, values: {down: -0.05}}
"""

MATERIAL = """
name: unobtainium
texture: {type: painted_wall, params: {base_color: [20, 200, 180], seed: 3}}
size: [32, 32]
tile_size: 1.0
pbr: {roughness: 0.4, metallic: 0.0, normal_strength: 0.0, ao_strength: 0.3}
"""

SCENE = """
name: vault
size: [6, 3, 6]
quests: [quests/press.yaml]
slots:
  a: {position: [0, 0, 0]}
  b: {position: [2, 0, 0]}
place:
  widget: {asset: puzzles/widget.yaml, slot: a}      # the project's
  chair:  {asset: chair.yaml, slot: b}               # the core's
"""

QUEST = """
kind: quest
version: 1
title: Press it
start: auto
objectives:
  - {id: press, text: Press the widget, done: 'widget.press == "down"'}
"""


@pytest.fixture
def project(tmp_path):
    """A project directory; the roots go back to the core alone afterwards."""
    files = {"geogen.yaml": "kind: project\nname: vaults\n", "assets/puzzles/widget.yaml": WIDGET,
             "assets/materials/unobtainium.yaml": MATERIAL, "assets/scenes/vault.yaml": SCENE,
             "assets/quests/press.yaml": QUEST}
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    yield tmp_path
    roots.use()


def test_the_core_stands_alone_by_default():
    roots.use()
    assert roots.roots() == [roots.CORE]
    assert roots.find("chair.yaml") == roots.CORE / "chair.yaml"
    assert roots.find("nothing.yaml") == roots.CORE / "nothing.yaml"      # for the caller's own error


def test_a_project_is_searched_before_the_core(project):
    core_only = source_fingerprint()
    (found,) = roots.use(project)
    assert (found.name, found.root) == ("vaults", (project / "assets").resolve())
    assert roots.roots() == [found.root, roots.CORE]
    assert roots.find("puzzles/widget.yaml") == found.root / "puzzles" / "widget.yaml"
    assert roots.find("chair.yaml", roots.CORE) == roots.CORE / "chair.yaml"
    listed = [roots.relative(p) for p in roots.glob("quests/*.yaml")]
    assert "quests/press.yaml" in listed and "quests/lights_out.yaml" in listed
    assert source_fingerprint() != core_only          # disk caches don't outlive a project edit


def test_a_directory_that_is_not_a_root_stands_alone(project, tmp_path):
    roots.use(project)
    assert roots.find("chair.yaml", tmp_path) == tmp_path / "chair.yaml"
    assert roots.glob("quests/*.yaml", tmp_path) == []


def test_a_project_scene_uses_its_own_and_core_assets(project):
    roots.use(project)
    registry = SceneRegistry()
    registry.discover()
    assert "vault" in registry and "cottage" in registry and "widget" not in registry      # puzzles/ isn't scanned
    scene = registry["vault"]()
    names = {n.name for n in scene.iter_nodes()}
    assert {"widget", "chair"} <= names
    block = next(n for n in scene.iter_nodes() if n.name == "block")
    assert block.mesh.material.name == "unobtainium"
    quests = collect(scene, "vault")
    machine = Machine(quests.program(), {"player": {"kind": "player"}})
    machine.use("vault:widget#press", "player")
    machine.advance(1.0)
    assert quests.report(machine, "player")["press"]["status"] == "complete"


def test_shadowing_a_core_file_must_be_explicit(project):
    (project / "assets" / "chair.yaml").write_text(WIDGET)
    with pytest.raises(roots.RootsError, match=r"also in geogen's assets: chair.yaml.*overrides"):
        roots.use(project)
    (project / "geogen.yaml").write_text("kind: project\noverrides: [chair.yaml]\n")
    (found,) = roots.use(project)
    assert roots.find("chair.yaml") == found.root / "chair.yaml"
    registry = SceneRegistry()
    registry.discover()
    assert registry.path("chair") == found.root / "chair.yaml"
    (project / "geogen.yaml").write_text("kind: project\noverrides: [chair.yaml, gone.yaml]\n")
    with pytest.raises(roots.RootsError, match="doesn't have: gone.yaml"):
        roots.use(project)


def test_a_project_scene_cannot_take_a_core_scenes_name(project):
    (project / "assets" / "scenes" / "chair.yaml").write_text(SCENE)       # the core's chair is assets/chair.yaml
    roots.use(project)
    with pytest.raises(roots.RootsError, match="scene name 'chair' is both"):
        SceneRegistry().discover()


@pytest.mark.parametrize("config, message", [
    ("kind: quest\n", "kind: project"),
    ("kind: project\nasets: stuff\n", "unknown key"),
    ("kind: project\nassets: nowhere\n", "doesn't exist"),
])
def test_bad_project_files_say_what_is_wrong(project, config, message):
    (project / "geogen.yaml").write_text(config)
    with pytest.raises(roots.RootsError, match=message):
        roots.use(project)


def test_the_project_is_found_from_the_environment_or_the_working_directory(project, monkeypatch):
    monkeypatch.setattr(roots, "_projects", None)
    monkeypatch.setenv(roots.ENV, str(project))
    assert [p.name for p in roots.projects()] == ["vaults"]
    monkeypatch.setattr(roots, "_projects", None)
    monkeypatch.delenv(roots.ENV)
    monkeypatch.chdir(project / "assets" / "scenes")
    assert [p.name for p in roots.projects()] == ["vaults"]


def test_the_example_project_makes_a_sandwich():
    """examples/sandwich: a scene, an asset and a quest of its own on the core's kitchen."""
    try:
        roots.use(EXAMPLE)
        registry = SceneRegistry()
        registry.discover()
        scene = registry["sandwich_kitchen"]()
        quests = collect(scene, "sandwich_kitchen")
        machine = Machine(quests.program(), {"player": {"kind": "player"}})
        uses = ["loaf#slices", "loaf#slices",                      # the second comes too soon and goes back
                "peanut_butter#lid", "peanut_butter#scoop", "jam#lid", "jam#scoop", "loaf#slices"]
        for use in uses:
            machine.use(f"sandwich_kitchen:{use}", "player")
            machine.advance(1.0)
        assert machine.get("sandwich_kitchen:sandwich#top_slice") == "placed"
        assert quests.report(machine, "player")["make_sandwich"]["status"] == "complete"
    finally:
        roots.use()


# --- scene folders (geogen.39.2) --------------------------------------------------------------------

PUZZLE = """
name: puzzle
size: [6, 3, 6]
quests: [quests/press.yaml]
slots:
  a: {position: [0, 0, 0]}
  b: {position: [2, 0, 0]}
place:
  widget: {asset: widget.yaml, slot: a}              # the folder's own, in the folder's own material
  chair:  {asset: chair.yaml, slot: b}               # the core's
"""

OTHER = """
name: other
size: [6, 3, 6]
slots: {a: {position: [0, 0, 0]}}
place:
  thing: {%s, slot: a}
"""


@pytest.fixture
def folders(tmp_path):
    """A project with a scene folder `puzzle`: its scene, a private asset, material and quest."""
    files = {"geogen.yaml": "kind: project\nname: vaults\n", "assets/scenes/puzzle/scene.yaml": PUZZLE,
             "assets/scenes/puzzle/widget.yaml": WIDGET, "assets/scenes/puzzle/materials/unobtainium.yaml": MATERIAL,
             "assets/scenes/puzzle/quests/press.yaml": QUEST}
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    roots.use(tmp_path)
    yield tmp_path / "assets"
    roots.use()


def _registry():
    registry = SceneRegistry()
    registry.discover()
    return registry


def test_a_scene_folder_is_a_scene_with_files_of_its_own(folders):
    registry = _registry()
    assert "puzzle" in registry and "scene" not in registry and "widget" not in registry
    assert registry.path("puzzle") == folders / "scenes" / "puzzle" / "scene.yaml"
    scene = registry["puzzle"]()
    assert {"widget", "chair"} <= {n.name for n in scene.iter_nodes()}
    block = next(n for n in scene.iter_nodes() if n.name == "block")
    assert block.mesh.material.name == "unobtainium"
    quests = collect(scene, "puzzle")
    machine = Machine(quests.program(), {"player": {"kind": "player"}})
    machine.use("puzzle:widget#press", "player")
    machine.advance(1.0)
    assert quests.report(machine, "player")["press"]["status"] == "complete"
    assert roots.scope() is None                                # the folder is only searched while it is composed


@pytest.mark.parametrize("names, message", [
    ("asset: scenes/puzzle/widget.yaml", "private to the scene folder 'puzzle'"),
    ("asset: widget.yaml", "widget.yaml"),                      # not found: it isn't this scene's
])
def test_another_scene_cannot_name_a_folders_files(folders, names, message):
    (folders / "scenes" / "other.yaml").write_text(OTHER % names)
    with pytest.raises((roots.RootsError, FileNotFoundError), match=message):
        _registry()["other"]()


def test_another_scene_cannot_use_a_folders_material(folders):
    (folders / "plain.yaml").write_text(WIDGET.replace("name: widget", "name: plain"))
    (folders / "scenes" / "other.yaml").write_text(OTHER % "asset: plain.yaml")
    with pytest.warns(UserWarning, match="Material 'unobtainium' not found"):
        scene = _registry()["other"]()
    assert next(n for n in scene.iter_nodes() if n.name == "block").mesh.material is None


def test_a_folders_scene_can_be_placed_and_its_files_are_not_listed(folders):
    (folders / "scenes" / "other.yaml").write_text(OTHER % "scene: scenes/puzzle/scene.yaml")
    scene = _registry()["other"]()
    assert next(n for n in scene.iter_nodes() if n.name == "block").mesh.material.name == "unobtainium"
    listed = [roots.relative(p) for p in roots.glob("**/*.yaml")]
    assert "scenes/puzzle/scene.yaml" in listed and "chair.yaml" in listed
    assert not [name for name in listed if name.startswith("scenes/puzzle/") and name != "scenes/puzzle/scene.yaml"]
    from geogen.describe import asset_files

    assert asset_files()["puzzle"] == ("scene", folders / "scenes" / "puzzle" / "scene.yaml")


def test_two_folders_keep_files_of_the_same_name_apart(folders):
    second = folders / "scenes" / "second"
    (second / "materials").mkdir(parents=True)
    (second / "scene.yaml").write_text(PUZZLE.replace("name: puzzle", "name: second").replace("quests: [quests/press.yaml]\n", ""))
    (second / "widget.yaml").write_text(WIDGET.replace("[0.3, 0.3, 0.3]", "[0.6, 0.6, 0.6]"))
    (second / "materials" / "unobtainium.yaml").write_text(MATERIAL.replace("[20, 200, 180]", "[200, 30, 30]"))
    (folders / "scenes" / "both.yaml").write_text(
        "name: both\nsize: [9, 3, 6]\nslots: {a: {position: [0, 0, 0]}, b: {position: [5, 0, 0]}}\n"
        "place:\n  one: {scene: scenes/puzzle/scene.yaml, slot: a}\n  two: {scene: scenes/second/scene.yaml, slot: b}\n")
    scene = _registry()["both"]()
    blocks = [n for n in scene.iter_nodes() if n.name == "block"]
    sizes = sorted(round(float(np.ptp(b.mesh.vertices[:, 0])), 2) for b in blocks)
    assert sizes == [0.3, 0.6]
    assert blocks[0].mesh.material is not blocks[1].mesh.material


def test_a_folders_file_cannot_take_a_shared_name(folders):
    (folders / "scenes" / "puzzle" / "chair.yaml").write_text(WIDGET)
    with pytest.raises(roots.RootsError, match="scene folder 'puzzle'.*chair.yaml"):
        _registry()["puzzle"]()
    assert roots.scope() is None


# --- a project's runtime catalogue and exports (geogen.39.3) -----------------------------------------

def test_a_project_exports_its_own_catalogue_to_its_own_directory(project, tmp_path):
    """Its runtime_scenes.yaml is beside the core's, not instead of it: its own scenes and
    default (core scenes may be listed), exported under the project, not runtime/godot."""
    import json

    from geogen import catalogue

    roots.use()
    assert catalogue.active() == (catalogue.CATALOGUE_PATH, catalogue.CORE_GENERATED)
    roots.use(project)
    assert catalogue.active() == (None, (project / "generated").resolve())
    with pytest.raises(ValueError, match="no runtime catalogue"):
        catalogue.load_catalogue()
    (project / "assets" / "runtime_scenes.yaml").write_text(
        "kind: scene_catalogue\ndefault: vault\nscenes:\n  vault: {group: showcase, description: A vault}\n"
        "  chair: {description: The core's chair}\n")
    (project / "geogen.yaml").write_text("kind: project\nname: vaults\ngenerated: build/runtime\n")
    (found,) = roots.use(project)                               # (the catalogue shares the core's file name: allowed)
    file, generated = catalogue.active()
    assert file == found.root / "runtime_scenes.yaml" and generated == (project / "build" / "runtime").resolve()
    listed = catalogue.load_catalogue()
    assert (listed.default, list(listed.entries)) == ("vault", ["vault", "chair"])
    registry = SceneRegistry()
    registry.discover()
    catalogue.export_catalogue(listed, lambda name: registry[name](), generated, log=lambda *_: None)
    index = json.loads((generated / "catalogue.json").read_text())
    assert index["default"] == "vault" and [s["exported"] for s in index["scenes"]] == [True, True]
    manifest = json.loads((generated / "vault.manifest.json").read_text())
    assert manifest["state"] and all(key.startswith("vault:") for key in manifest["state"]["interactions"])


def test_the_example_project_has_a_catalogue_of_its_own():
    from geogen import catalogue

    try:
        (found,) = roots.use(EXAMPLE)
        listed = catalogue.load_catalogue()
        assert listed.default == "sandwich_kitchen" and "hub" not in listed.entries
        assert catalogue.active()[1] == EXAMPLE / "generated"
    finally:
        roots.use()
