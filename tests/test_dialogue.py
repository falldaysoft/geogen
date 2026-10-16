"""Dialogue (geogen.17; geogen/dialogue.py, "Dialogue" in docs/rules.md)."""

import json
from pathlib import Path

import pytest

from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from geogen.quests import QuestError, collect
from geogen.rules import Machine

ASSETS = Path(__file__).parent.parent / "assets"
CHAT = """
kind: dialogue
version: 1
anchor: clerk
start:
  - {when: 'has("parcel")', passage: again}
  - hello
passages:
  hello:
    text: "Morning. You carry {held(\\"parcel\\")} parcels."
    choices:
      - {id: take, text: "I'll take the parcel", do: [{give: parcel}], goto: thanks}
      - {id: ask, text: "Anything else?", when: 'has("parcel")', goto: again}
      - {text: Goodbye}
  thanks:
    text: Thank you!
  again:
    text: "You have it already."
"""
SCENE = """
name: desk
size: [6, 2, 6]
dialogues: [{dialogues}]
slots:
  desk: {{position: [0, 0, 0], facing: south}}
place:
  clerk: {{npc: npcs/resident.yaml, slot: desk, home: {{rect: [-2, -2, 2, 2]}}}}
spawns:
  start: {{position: [0, 0, 2], facing: north}}
"""


def compose(tmp_path, chat=CHAT):
    (tmp_path / "chat.yaml").write_text(chat)
    scene = tmp_path / "desk.yaml"
    scene.write_text(SCENE.format(dialogues=tmp_path / "chat.yaml"))
    return SceneComposer(assets_dir=ASSETS).compose(scene)


def choose(machine, passage, choice, actor="player"):
    machine.post({"type": "signal", "subject": "desk:clerk", "name": "dialogue", "actor": actor,
                  "data": {"passage": passage, "choice": choice}})


def test_a_dialogue_opens_on_a_greeting_and_its_choices_run_effects(tmp_path):
    found = collect(compose(tmp_path), "desk")
    entry = found.dialogues["chat"]
    assert entry["anchor"] == "desk:clerk" and entry["state"] == "actor:desk.dialogue_chat"
    hello = entry["passages"]["hello"]
    assert hello["text"][0] == "Morning. You carry " and "value" in hello["text"][1]
    assert [c["id"] for c in hello["choices"]] == ["take", "ask", "3"] and "when" in hello["choices"][1]
    machine = Machine(found.program(), {"player": {"kind": "player"}})
    state = lambda: machine.get("actor:desk.dialogue_chat", "player")
    choose(machine, "hello", "take")                                       # not open: nothing happens
    assert state() == "" and machine.get("actor:desk.item_parcel", "player") == 0
    machine.post({"type": "signal", "subject": "desk:clerk", "name": "greeted", "actor": "player"})
    assert state() == "hello"
    choose(machine, "hello", "ask")                                        # its `when` doesn't hold
    assert state() == "hello"
    choose(machine, "hello", "take")
    assert state() == "thanks" and machine.get("actor:desk.item_parcel", "player") == 1
    choose(machine, "thanks", "")                                          # "" closes it
    assert state() == ""
    machine.post({"type": "signal", "subject": "desk:clerk", "name": "greeted", "actor": "player"})
    assert state() == "again"                                              # start picks by condition
    assert machine.errors == []


def test_a_dialogue_exports_its_passages(tmp_path):
    export_scene(compose(tmp_path), tmp_path / "desk.glb")
    manifest = json.loads((tmp_path / "desk.manifest.json").read_text())
    assert set(manifest["dialogues"]["chat"]["passages"]) == {"hello", "thanks", "again"}
    assert "actor:desk.dialogue_chat" in manifest["state"]["variables"]


@pytest.mark.parametrize("change,message", [
    (("goto: thanks", "goto: thank"), "no passage 'thank'"),
    (("  again:\n    text: \"You have it already.\"\n", "  again:\n    text: \"You have it already.\"\n  lost:\n    text: Lost\n"),
     "nothing leads to passage 'lost'"),
    (("anchor: clerk", "anchor: clark"), "anchor"),
    (("do: [{give: parcel}]", "do: [{give: parcle}]"), "no item 'parcle'"),
])
def test_a_bad_dialogue_is_reported(tmp_path, change, message):
    with pytest.raises(QuestError, match=message):
        collect(compose(tmp_path, CHAT.replace(*change)), "desk")


def test_the_dialogue_panel_shows_passages_and_takes_choices_in_godot(godot_runtime, tmp_path):
    export_scene(compose(tmp_path), tmp_path / "desk.glb")
    rt = godot_runtime("desk", tmp_path)
    assert rt.call("dialogue.show") == {"open": False}
    rt.call("npcs.greet", npc="clerk")
    rt.call("time.advance", seconds=0.5)
    shown = rt.call("dialogue.show")
    assert shown["open"] and shown["passage"] == "hello" and shown["text"] == "Morning. You carry 0 parcels."
    assert [c["id"] for c in shown["choices"]] == ["take", "3"]              # "ask" needs the parcel
    rt.call("input.key", key="1")                                           # the number keys choose
    rt.call("time.advance", seconds=0.2)
    shown = rt.call("dialogue.show")
    assert shown["passage"] == "thanks" and shown["choices"] == [{"id": "", "text": "Continue"}]
    assert rt.call("inventory.list")["items"][0]["count"] == 1
    assert rt.call("dialogue.choose", index=0) == {"open": False}
    rt.call("npcs.greet", npc="clerk")
    rt.call("time.advance", seconds=0.5)
    assert rt.call("dialogue.show")["passage"] == "again"
    rt.call("input.key", key="Escape")
    rt.call("time.advance", seconds=0.2)
    assert rt.call("dialogue.show") == {"open": False}


YARD = """
name: yard
size: [10, 3, 10]
dialogues: [{plaque}, {gate}]
slots:
  plinth: {{position: [2, 0, 0], facing: south}}
places:
  arch: {{box: [3, 3, 3], position: [-3, 0, 0]}}
place:
  plinth: {{asset: info_plinth.yaml, slot: plinth, readable: Read the plaque}}
spawns:
  start: {{position: [0, 0, 3], facing: north}}
"""
PLAQUE = """
kind: dialogue
version: 1
anchor: plinth
passages:
  words: {text: "Founded 1907."}
"""
GATE = """
kind: dialogue
version: 1
anchor: arch
once: true
passages:
  hush: {text: "The yard falls quiet as you pass under the arch."}
"""


def test_objects_and_places_open_their_dialogues(tmp_path):
    # geogen.17.3: readable: gives a plaque a `read` interaction to open on; a place opens its
    # narration on entering, here only the first time (once).
    for name, text in (("plaque", PLAQUE), ("gate", GATE)):
        (tmp_path / f"{name}.yaml").write_text(text)
    scene = tmp_path / "yard.yaml"
    scene.write_text(YARD.format(plaque=tmp_path / "plaque.yaml", gate=tmp_path / "gate.yaml"))
    root = SceneComposer(assets_dir=ASSETS).compose(scene)
    plinth = next(n for n in root.iter_nodes() if n.name == "plinth")
    assert [(i.name, i.states["ready"].prompt) for i in plinth.interactions] == [("read", "Read the plaque")]
    machine = Machine(collect(root, "yard").program(), {"player": {"kind": "player"}})
    plaque = lambda: machine.get("actor:yard.dialogue_plaque", "player")
    gate = lambda: machine.get("actor:yard.dialogue_gate", "player")
    assert machine.use("yard:plinth#read", "player") and plaque() == "words"
    machine.post({"type": "signal", "subject": "yard:plinth", "name": "dialogue", "actor": "player",
                  "data": {"passage": "words", "choice": ""}})
    assert plaque() == "" and machine.use("yard:plinth#read", "player") and plaque() == "words"   # again
    machine.post({"type": "enter", "subject": "yard:arch", "actor": "player"})
    assert gate() == "hush"
    machine.post({"type": "signal", "subject": "yard:arch", "name": "dialogue", "actor": "player",
                  "data": {"passage": "hush", "choice": ""}})
    machine.post({"type": "enter", "subject": "yard:arch", "actor": "player"})
    assert gate() == ""                                                    # once
    assert machine.errors == []


def test_reading_a_plaque_opens_its_dialogue_in_godot(godot_runtime, tmp_path):
    for name, text in (("plaque", PLAQUE), ("gate", GATE)):
        (tmp_path / f"{name}.yaml").write_text(text)
    scene = tmp_path / "yard.yaml"
    scene.write_text(YARD.format(plaque=tmp_path / "plaque.yaml", gate=tmp_path / "gate.yaml"))
    export_scene(SceneComposer(assets_dir=ASSETS).compose(scene), tmp_path / "out" / "yard.glb")
    rt = godot_runtime("yard", tmp_path / "out")
    rt.call("player.use", asset="plinth", interaction="read")
    rt.call("time.advance", seconds=0.3)
    shown = rt.call("dialogue.show")
    assert shown["open"] and shown["text"] == "Founded 1907.", shown
