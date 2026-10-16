"""Things actors carry (geogen.18; geogen/inventory.py, "Items" in docs/rules.md)."""

import json
from pathlib import Path

import pytest

from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from geogen.quests import collect
from geogen.rules import Machine

ASSETS = Path(__file__).parent.parent / "assets"
YARD = """
name: pickup_yard
size: [6, 2, 6]
slots:
  a: {position: [-1, 0, 0]}
  b: {position: [1, 0, 0]}
place:
  key: {asset: room_key.yaml, slot: a}
  parcel: {asset: parcel.yaml, slot: b, pickup: {item: parcel, count: 2}}
spawns:
  start: {position: [0, 0, 2], facing: north}
"""


@pytest.fixture(scope="module")
def yard(tmp_path_factory):
    path = tmp_path_factory.mktemp("pickups") / "pickup_yard.yaml"
    path.write_text(YARD)
    return SceneComposer(assets_dir=ASSETS).compose(path)


def test_a_pickup_gives_its_item_to_whoever_takes_it(yard):
    # The key is a pickup by its asset (pickup: room_key); the parcel by its placement.
    key = next(n for n in yard.iter_nodes() if n.name == "key")
    pickup = next(i for i in key.interactions if i.name == "pickup")
    assert pickup.initial == "here" and pickup.states["here"].prompt == "Take the room key"
    found = collect(yard, "pickup_yard")
    assert found.items == {"room_key": {"title": "Room key", "tags": ["key"]}, "parcel": {"title": "Parcel"}}
    machine = Machine(found.program(), {"player": {"kind": "player"}, "pickup_yard:npc": {"kind": "npc"}})
    assert machine.use("pickup_yard:key#pickup", "player")
    machine.advance(0.5)
    assert machine.get("pickup_yard:key#pickup") == "taken"
    assert machine.get("actor:pickup_yard.item_room_key", "player") == 1
    assert machine.use("pickup_yard:parcel#pickup", "pickup_yard:npc")
    machine.advance(0.5)
    assert machine.get("actor:pickup_yard.item_parcel", "pickup_yard:npc") == 2
    assert machine.get("actor:pickup_yard.item_parcel", "player") == 0
    assert not machine.use("pickup_yard:key#pickup", "player")            # gone: nothing to take twice
    assert machine.errors == []


def test_a_pickup_exports_its_interaction_and_rule(yard, tmp_path):
    export_scene(yard, tmp_path / "pickup_yard.glb")
    manifest = json.loads((tmp_path / "pickup_yard.manifest.json").read_text())
    rules = {r["id"]: r for r in manifest["rules"]["rules"]}
    assert rules["pickup_yard:key#pickup"]["on"] == {"type": "use", "subject": "pickup_yard:key", "name": "pickup"}
    assert "actor:pickup_yard.item_room_key" in manifest["state"]["variables"]


def test_a_bad_pickup_is_reported(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(YARD.replace("item: parcel", "item: parcle"))
    with pytest.raises(ValueError, match="parcle.*items/parcle.yaml"):
        SceneComposer(assets_dir=ASSETS).compose(path)


def test_taking_a_pickup_in_godot_hides_it_and_gives_the_item(godot_runtime, yard, tmp_path):
    export_scene(yard, tmp_path / "pickup_yard.glb")
    rt = godot_runtime("pickup_yard", tmp_path)
    rt.call("player.use", asset="key", interaction="pickup")
    rt.call("time.advance", seconds=1)
    value = lambda key: rt.call("state.get", key=key)["value"]
    assert value("actor:pickup_yard.item_room_key") == 1
    assert value("pickup_yard:key#pickup") == "taken"
    assert value("actor:pickup_yard.item_parcel") == 0
    assert rt.call("world.node", name="fob")["visible"] is False          # the key is gone from the world
    assert rt.call("world.node", name="box")["visible"] is True           # the parcel is still there
