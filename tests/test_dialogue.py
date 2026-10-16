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
