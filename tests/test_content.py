"""Content packages (geogen/content.py): state, rules and quests that span scenes they don't own."""

import json
from pathlib import Path

import pytest

from geogen import content
from geogen.catalogue import Catalogue, CatalogueEntry, export_catalogue, load_catalogue, write_index
from geogen.conditions import compile_condition
from geogen.content import ContentError, Scene, compile_package, export_packages
from geogen.layout.composer import SceneComposer
from geogen.rules import Machine, check_program
from test_state import GATE, LAMP, LEVER
from test_wiring import PLATE

SCHEMA = Path(__file__).parent.parent / "docs/schema/geogen-content.v1.schema.json"

# Two scenes that know nothing of each other: a lever in one, a gate, a lamp and a plate in the other.
CAMP = """
name: camp
size: [8, 3, 8]
state:
  fire: {scope: world, type: bool, default: true}
slots:
  a: {position: [0, 0, 2]}
place:
  lever: {asset: lever.yaml, slot: a}
spawns:
  start: {position: [0, 0, 3.2], facing: north}
"""

ANNEX = """
name: annex
size: [10, 3, 10]
slots:
  a: {position: [0, 0, -3]}
  b: {position: [3, 0, 0]}
  c: {position: [-3, 0, 0]}
place:
  gate:  {asset: gate.yaml, slot: a}
  lamp:  {asset: lamp.yaml, slot: b}
  plate: {asset: plate.yaml, slot: c}
spawns:
  start: {position: [0, 0, 3], facing: south}
"""

# The package: pulling the camp's lever is remembered, and opens the annex's gate and lights its lamp.
ERRAND = """
kind: package
version: 1
title: The errand
description: Pull the lever at camp, then stand on the plate in the annex.
scenes: [camp, annex]
state:
  lever_pulled: {scope: world, type: bool, default: false}
  pulls:        {scope: actor, type: number, default: 0}
rules:
  - id: note_pull
    on: {signal: "camp:lever.pulled"}
    do: [{set: world.lever_pulled, to: true}]
  - id: count_pull
    on: {use: "camp:lever", actor: player}
    when: 'world("camp").fire'
    do: [{set: actor.pulls, to: "actor.pulls + 1"}]
  - id: open_gate
    when: "world.lever_pulled"
    do: [{input: "annex:gate.open"}]
bind:
  'ref("annex:lamp").lit': "world.lever_pulled"
quests: [quests/errand.yaml]
"""

QUEST = """
kind: quest
version: 1
title: Run the errand
start: auto
ordered: true
objectives:
  - {id: pull, text: Pull the lever at camp, done: {signal: "camp:lever.pulled"}}
  - {id: plate, text: Stand on the plate in the annex, done: {enter: "annex:plate/top"}}
"""

FILES = (("lever", LEVER), ("gate", GATE), ("lamp", LAMP), ("plate", PLATE), ("camp", CAMP), ("annex", ANNEX),
         ("packages/errand", ERRAND), ("quests/errand", QUEST))
ACTORS = {"player": {"kind": "player"}}


def write_assets(assets):
    for name, text in FILES:
        path = assets / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    return assets


def deploy(assets, out, packages=("errand",)):
    """Export both scenes and the packages into ``out``, as --export-catalogue does."""
    catalogue = Catalogue("camp", {"camp": CatalogueEntry("camp"), "annex": CatalogueEntry("annex")},
                          {name: assets / "packages" / f"{name}.yaml" for name in packages})
    export_catalogue(catalogue, lambda name: SceneComposer(assets_dir=assets).compose(assets / f"{name}.yaml"),
                     out, log=lambda *_: None)
    return catalogue


@pytest.fixture(scope="module")
def deployed(tmp_path_factory):
    assets = write_assets(tmp_path_factory.mktemp("content_assets"))
    out = tmp_path_factory.mktemp("generated_content")
    return assets, out, deploy(assets, out)


def program_of(out, scene):
    """A scene's own rules program, from its manifest."""
    manifest = json.loads((out / f"{scene}.manifest.json").read_text())
    state, rules = manifest.get("state", {}), manifest.get("rules", {})
    return {"format": "geogen-rules", "version": 1, "variables": state.get("variables", {}),
            "interactions": state.get("interactions", {}), "bindings": rules.get("bindings", []),
            "rules": rules.get("rules", [])}


def test_a_package_is_exported_with_ids_and_compiled_conditions(deployed):
    _, out, _ = deployed
    errand = json.loads((out / "errand.content.json").read_text())
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(errand, json.loads(SCHEMA.read_text()))
    assert errand["format"] == "geogen-content" and errand["package"] == "errand"
    assert errand["scenes"] == ["camp", "annex"] and errand["title"] == "The errand"
    assert errand["variables"]["world:errand.lever_pulled"] == {"type": "bool", "default": False}
    assert "actor:errand.quest_errand" in errand["variables"] and "camp:lever#pull" not in errand["variables"]

    rules = {rule["id"]: rule for rule in errand["rules"]}
    assert list(rules)[:3] == ["errand:note_pull", "errand:count_pull", "errand:open_gate"]
    # references are stable IDs of the scene they are in, and the rule runs there
    note = rules["errand:note_pull"]
    assert note["on"] == {"type": "signal", "subject": "camp:lever", "name": "pulled"} and note["scene"] == "camp"
    assert note["do"] == [{"set": "world:errand.lever_pulled", "to": True}]
    assert rules["errand:count_pull"]["when"] == {"get": "world", "package": "camp", "name": "fire"}
    gate = rules["errand:open_gate"]
    assert gate["scene"] == "annex" and gate["do"] == [{"set": "annex:gate#gate", "to": "open"}]
    assert gate["when"] == {"get": "world", "package": "errand", "name": "lever_pulled"}
    assert errand["bindings"] == [{"set": "annex:lamp#lit", "scene": "annex",
                                   "to": {"get": "world", "package": "errand", "name": "lever_pulled"}}]
    # the quest's own bookkeeping reads only the package's state, so it runs wherever the package does
    assert "scene" not in rules["errand:quest:errand:start"]
    assert rules["errand:quest:errand:pull"]["scene"] == "camp"
    assert rules["errand:quest:errand:plate"]["scene"] == "annex"
    assert rules["errand:quest:errand:plate"]["on"] == {"type": "enter", "subject": "annex:plate/top"}
    quest = errand["quests"]["quests"]["errand"]
    assert quest["status"] == "actor:errand.quest_errand" and errand["quests"]["package"] == "errand"
    assert [o["subjects"] for o in quest["objectives"]] == [["camp:lever"], ["annex:plate/top"]]


def test_the_catalogue_lists_deployed_packages(deployed, tmp_path):
    _, out, _ = deployed
    index = json.loads((out / "catalogue.json").read_text())
    assert index["packages"] == [{"name": "errand", "file": "errand.content.json", "exported": True,
                                  "scenes": ["camp", "annex"]}]
    # the catalogue file names them; a package is a file beside it
    assets = write_assets(tmp_path)
    (assets / "runtime_scenes.yaml").write_text(
        "kind: scene_catalogue\ndefault: camp\nscenes: {camp: {}, annex: {}}\npackages: [errand]\n")
    catalogue = load_catalogue(assets / "runtime_scenes.yaml")
    assert catalogue.packages == {"errand": assets / "packages" / "errand.yaml"}
    # before its scenes are exported it is listed as not exported, and nothing is wrong
    assert export_packages(catalogue, tmp_path / "out", log=lambda *_: None) == []
    listed = json.loads(write_index(catalogue, tmp_path / "out").read_text())["packages"]
    assert listed == [{"name": "errand", "file": "errand.content.json", "exported": False, "scenes": []}]
    (assets / "runtime_scenes.yaml").write_text(
        "kind: scene_catalogue\ndefault: camp\nscenes: {camp: {}}\npackages: [erand]\n")
    with pytest.raises(ValueError, match="package 'erand': no file"):
        load_catalogue(assets / "runtime_scenes.yaml")


def test_the_real_catalogue_deploys_packages_that_exist():
    catalogue = load_catalogue()
    for name, path in catalogue.packages.items():
        assert content.load_package(path)["id"] == name


def test_state_set_in_one_scene_is_read_in_another(deployed):
    """The reference for what a runtime does: `apply` on load, `carried` on unload, `resume` on the next."""
    _, out, _ = deployed
    errand = json.loads((out / "errand.content.json").read_text())

    camp = content.apply([errand], "camp", program_of(out, "camp"))
    assert check_program(camp) == []
    assert "errand:open_gate" not in [r["id"] for r in camp["rules"]] and camp["bindings"] == []
    machine = Machine(camp, ACTORS)
    machine.post({"type": "start", "actor": "player"})
    assert machine.get("actor:errand.quest_errand", "player") == "active"
    machine.use("camp:lever#pull", "player")
    machine.advance(1.0)
    assert machine.get("world:errand.lever_pulled") is True
    assert machine.get("actor:errand.pulls", "player") == 1
    assert machine.get("actor:errand.quest_errand_pull", "player") is True

    kept = content.carried(machine.save(), [errand])
    assert kept["world"] == {"world:errand.lever_pulled": True}
    assert kept["actors"]["player"]["actor:errand.pulls"] == 1
    assert not any(key.startswith("camp:") for key in kept["world"])       # the lever stays the camp's

    annex = content.apply([errand], "annex", program_of(out, "annex"))
    assert check_program(annex) == []
    machine = Machine(annex)
    machine.load(content.resume(None, kept, [errand]))
    machine.add_actor("player", "player")
    assert machine.get("annex:gate#gate") == "closed"
    machine.post({"type": "actor", "subject": "player", "actor": "player"})    # arriving settles the watches
    assert machine.get("annex:gate#gate") == "open"                 # the rule that read the camp's lever
    assert machine.get("annex:lamp#lit") is True                    # and the binding
    assert machine.get("actor:errand.quest_errand_pull", "player") is True
    assert machine.get("actor:errand.quest_errand", "player") == "active"
    machine.post({"type": "enter", "subject": "annex:plate/top", "actor": "player"})
    assert machine.get("actor:errand.quest_errand", "player") == "complete"

    # back at camp the camp's own save comes back, with the package's values as they are now
    kept = content.carried(machine.save(), [errand])
    stale = {"format": "geogen-save", "version": 1, "world": {"camp:lever#pull": "down", "world:errand.lever_pulled": False},
             "actors": {"player": {"actor:errand.quest_errand": "active"}}}
    stale["world"]["watch:errand:open_gate"] = True
    back = content.resume(stale, kept, [errand], actors=["player"])
    # the gate's watch is judged afresh next time the annex loads: the annex's things start over
    assert back["world"] == {"camp:lever#pull": "down", "world:errand.lever_pulled": True}
    assert machine.save()["world"]["watch:errand:open_gate"] is True
    assert back["actors"]["player"]["actor:errand.quest_errand"] == "complete"
    # a package the scene isn't in changes nothing
    assert content.apply([{**errand, "scenes": ["annex"]}], "camp", program_of(out, "camp")) == \
        {**program_of(out, "camp")}


def edited(old, new, text=ERRAND):
    assert old in text
    return text.replace(old, new)


@pytest.mark.parametrize("text, wanted", [
    (edited("camp:lever.pulled", "camp:levr.pulled"),
     ["errand.yaml:12: rules.0.on", "no 'levr' in 'camp'", "Did you mean 'lever'?"]),
    (edited('signal: "camp:lever.pulled"', "signal: lever.pulled"),
     ["rules.0.on", "'lever.pulled': say which scene it is in", "camp:lever.pulled"]),
    (edited("camp:lever.pulled", "town:lever.pulled"),
     ["rules.0.on", "'town' isn't one of this package's scenes", "annex, camp"]),
    (edited("annex:gate.open", "annex:gate.opne"), ["rules.2.do.0", "'annex:gate' has no input 'opne'", "'open'"]),
    (edited('ref("annex:lamp").lit', 'ref("annex:lamp").lti'), ["bind.", "has no state 'lti'", "'lit'"]),
    (edited("to: true", "to: 3"), ["rules.0.do.0", "is a bool: it can't be 3"]),
    (edited("world.lever_pulled, to", "world.lever_pulld, to"),
     ["rules.0.do.0", "no world state 'lever_pulld' is declared in 'errand'", "'lever_pulled'"]),
    (edited('world("camp").fire', 'world("camp").fyre'), ["rules.1.when", "no world state 'fyre' is declared in 'camp'"]),
    (edited('when: "world.lever_pulled"', 'when: \'ref("camp:lever").pull == "down"\''),
     ["errand.yaml:18: rules.2", "rule 'open_gate' touches annex and camp", "only one scene is loaded"]),
    (edited('"world.lever_pulled"\nquests', '\'ref("camp:lever").pull == "down"\'\nquests'),
     ["bind.", "touches annex and camp"]),
    (edited('\'ref("annex:lamp").lit\': "world.lever_pulled"', 'world.lamp_cool: \'ref("annex:lamp").colour == "cool"\'',
            edited("  pulls: ", "  lamp_cool: {scope: world, type: bool, default: false}\n  pulls: ")),
     ["bind.", "isn't always loaded", "set it with a rule"]),
    (edited("scenes: [camp, annex]", "scenes: [camp, anex]"), ["errand.yaml:6: scenes.1", "no exported scene 'anex'"]),
    (edited("title: The errand", "titel: The errand"), ["unknown key 'titel'", "'title'"]),
    (edited("quests/errand.yaml", "quests/erand.yaml"), ["quests.0", "no quest file 'quests/erand.yaml'"]),
], ids=["unknown_thing", "no_scene", "not_its_scene", "unknown_input", "unknown_state", "wrong_type",
        "undeclared", "undeclared_in_scene", "rule_in_two_scenes", "binding_in_two_scenes", "binding_of_own_state",
        "unknown_scene", "unknown_key", "no_quest_file"])
def test_mistakes_are_reported_where_they_were_written(deployed, tmp_path, text, wanted):
    _, out, _ = deployed
    assets = write_assets(tmp_path)
    path = assets / "packages" / "errand.yaml"
    path.write_text(text)
    scenes = {name: Scene.exported(name, out / f"{name}.manifest.json") for name in ("camp", "annex")}
    with pytest.raises(ContentError) as raised:
        compile_package(content.load_package(path), scenes, path)
    for part in wanted:
        assert part in str(raised.value), str(raised.value)


def test_a_bad_reference_fails_the_export_and_the_package_is_not_deployed(deployed, tmp_path):
    assets, _, _ = deployed
    mine = write_assets(tmp_path / "assets")
    (mine / "packages" / "errand.yaml").write_text(edited("camp:lever.pulled", "camp:levr.pulled"))
    (mine / "quests" / "errand.yaml").write_text(QUEST.replace("annex:plate/top", "annex:plate/tap"))
    out = tmp_path / "out"
    (out).mkdir()
    (out / "errand.content.json").write_text("{}")          # an older export of it
    with pytest.raises(ValueError) as raised:
        deploy(mine, out)
    message = str(raised.value)
    assert "bad content packages" in message
    assert "errand.yaml:12: rules.0.on: 'camp:levr.pulled': no 'levr' in 'camp'. Did you mean 'lever'?" in message
    assert "quests/errand.yaml:9: objectives.1.done: 'annex:plate/tap': no 'tap' in 'plate' in 'annex'" in message
    assert not (out / "errand.content.json").exists()       # the runtime never runs content that doesn't fit
    index = json.loads((out / "catalogue.json").read_text())
    assert index["packages"][0]["exported"] is False and (out / "camp.manifest.json").exists()


def test_a_package_cannot_take_a_scenes_name_or_its_quests(deployed, tmp_path):
    assets, out, _ = deployed
    scenes = {name: Scene.exported(name, out / f"{name}.manifest.json") for name in ("camp", "annex")}
    spec = content.load_package(assets / "packages" / "errand.yaml")
    with pytest.raises(ContentError, match="'camp' is also a scene"):
        compile_package({**spec, "id": "camp"}, scenes, assets_dir=assets)
    with pytest.raises(ContentError, match="isn't.*one of world, actor"):
        compile_package({**spec, "id": "world"}, scenes, assets_dir=assets)
    taken = {**scenes, "annex": Scene("annex", scenes["annex"].state, scenes["annex"].rules,
                                      {"quests": {"errand": {}}}, scenes["annex"].tree)}
    with pytest.raises(ContentError, match="scene 'annex' already has a quest 'errand'"):
        compile_package(spec, taken, assets_dir=assets)


def test_a_built_scene_stands_in_for_an_export(deployed):
    """Scene.built: compile a package against scenes that haven't been written anywhere."""
    assets, out, _ = deployed
    scenes = {name: Scene.built(name, SceneComposer(assets_dir=assets).compose(assets / f"{name}.yaml"))
              for name in ("camp", "annex")}
    spec = content.load_package(assets / "packages" / "errand.yaml")
    assert compile_package(spec, scenes, assets / "packages" / "errand.yaml") == \
        json.loads((out / "errand.content.json").read_text())


def test_conditions_read_another_packages_variables():
    assert compile_condition('world("town").bridge_down and actor("town").has_ticket') == {
        "op": "and", "args": [{"get": "world", "package": "town", "name": "bridge_down"},
                              {"get": "actor", "package": "town", "name": "has_ticket"}]}
