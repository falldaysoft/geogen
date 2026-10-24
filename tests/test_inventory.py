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


LOCKER = """
name: locker
origin: bottom_center
size: [0.8, 1.6, 0.6]
parts:
  body: {primitive: cube, size: [0.8, 1.6, 0.6], anchor: bottom_center, material: metal}
  leaf: {primitive: cube, size: [0.7, 1.4, 0.03], anchor: bottom_center, offset: [0, 0.1, 0.3], material: red_metal}
interactions:
  swing:
    initial: shut
    lock: {key: room_key, locked: true}
    targets: [leaf]
    states:
      shut: {prompt: Open, next: open}
      open: {prompt: Close, next: shut}
    motions: [{part: leaf, translate: [0, 0, 1], values: {shut: 0, open: 0.2}}]
"""


def test_a_carried_key_opens_its_lock_in_godot(godot_runtime, tmp_path):
    # geogen.18.3: a lock whose key is an item opens for a player carrying that item: L unlocks it.
    (tmp_path / "locker.yaml").write_text(LOCKER)
    scene = tmp_path / "lock_yard.yaml"
    scene.write_text(YARD.replace("name: pickup_yard", "name: lock_yard").replace(
        "  parcel:", f"  locker: {{asset: {tmp_path / 'locker.yaml'}, slot: c}}\n  parcel:").replace(
        "  b: {position: [1, 0, 0]}", "  b: {position: [1, 0, 0]}\n  c: {position: [0, 0, -1.5]}"))
    root = SceneComposer(assets_dir=ASSETS).compose(scene)
    out = tmp_path / "generated"
    export_scene(root, out / "lock_yard.glb")
    assert "room_key" in json.loads((out / "lock_yard.manifest.json").read_text())["items"]
    rt = godot_runtime("lock_yard", out)
    rt.call("player.teleport", pos=[0, 0, -0.4])
    rt.call("player.look", at=[0, 0.9, -1.2])
    assert rt.call("hud.aim")["focus"]["prompt"] == "Locked"
    rt.call("player.use", asset="key", interaction="pickup")
    rt.call("time.advance", seconds=0.5)
    assert "room_key" in rt.call("player.status")["keys"]
    assert rt.call("hud.aim")["focus"]["prompt"] == "Unlock (L)"
    rt.call("input.key", key="L")
    assert rt.call("hud.aim")["focus"]["prompt"].startswith("Open")


def test_the_hud_shows_what_the_player_carries(godot_runtime, yard, tmp_path):
    # geogen.18.4: the inventory panel lists carried items by title, with a count; each give or
    # take raises a toast; the API gives and lists.
    export_scene(yard, tmp_path / "pickup_yard.glb")
    rt = godot_runtime("pickup_yard", tmp_path, "--give=parcel:3")
    assert rt.call("inventory.list")["items"] == [{"id": "parcel", "title": "Parcel", "count": 3}]
    rt.call("player.use", asset="key", interaction="pickup")
    rt.call("time.advance", seconds=0.5)
    shown = rt.call("hud.inventory")
    assert shown["visible"] and [(i["title"], i["count"]) for i in shown["items"]] == [("Parcel", 3), ("Room key", 1)]
    rt.call("inventory.give", item="parcel", count=-5)                      # never below none
    assert [i["id"] for i in rt.call("hud.inventory")["items"]] == ["room_key"]
    toasts = rt.call("hud.notifications")
    notes = [(n["heading"], n["text"]) for n in toasts["showing"] + toasts["queued"]]
    assert ("Picked up", "Room key") in notes and ("Handed over", "Parcel  ×3") in notes, notes
    with pytest.raises(Exception, match="no item 'coin'"):
        rt.call("inventory.give", item="coin")


def test_an_export_carries_each_items_body_as_an_icon(yard, tmp_path):
    # geogen.84.14: the exporter adds a hidden item_icons node with every compiled item's body,
    # extras item_icon {item}, once (a second export of the same root adds no more).
    from geogen.export import read_gltf
    from geogen.inventory import attach_icons

    export_scene(yard, tmp_path / "pickup_yard.glb")
    gltf = read_gltf(export_scene(yard, tmp_path / "again.glb").read_bytes())
    icons = sorted(n["name"] for n in gltf["nodes"] if n.get("extras", {}).get("geogen", {}).get("item_icon"))
    assert icons == ["parcel_icon", "room_key_icon"]
    assert sum(n["name"] == "item_icons" for n in gltf["nodes"]) == 1
    assert attach_icons(yard, {"parcel": {}, "room_key": {}}) == ["parcel", "room_key"]      # found, not added


SNACK_PROJECT = {
    "items/snack.yaml": "kind: item\ntitle: Snack\ntags: [food]\nbody: {asset: snack_bar.yaml}\nuse: Eat\n",
    "scenes/snack_yard.yaml": """
name: snack_yard
size: [6, 2, 6]
state:
  fed: {scope: actor, type: number, default: 0}
rules:
  - id: eat
    on: {use_item: snack}
    do: [{take: snack}, {set: actor.fed, to: "actor.fed + 1"}, {notify: "Yum"}]
place:
  key: {asset: room_key.yaml}
spawns:
  start: {position: [0, 0, 2], facing: north}
""",
}


def test_the_hotbar_selects_with_i_and_digits_and_f_uses(godot_runtime, tmp_path):
    # geogen.84.14: the inventory is a hotbar: one slot per item, one selected; I selects the next,
    # a digit its slot, F uses the selected item when its file says how (and does nothing otherwise).
    from geogen import roots

    project = tmp_path / "snacks"
    for name, text in SNACK_PROJECT.items():
        (project / "assets" / name).parent.mkdir(parents=True, exist_ok=True)
        (project / "assets" / name).write_text(text)
    try:
        roots.use(project)
        scene = SceneComposer(assets_dir=ASSETS).compose(project / "assets" / "scenes" / "snack_yard.yaml")
        export_scene(scene, tmp_path / "snack_yard.glb")
    finally:
        roots.use()
    rt = godot_runtime("snack_yard", tmp_path, "--give=snack:2", "--give=room_key")
    rt.call("time.advance", seconds=0.2)
    shown = rt.call("hud.inventory")
    assert [s["id"] for s in shown["slots"]] == ["room_key", "snack"] and shown["selected"] == "room_key"
    assert all(len(s["rect"]) == 4 and s["icon"] is False for s in shown["slots"])      # headless: initials
    rt.call("input.key", key="F")                                                        # a key has no use
    assert rt.call("state.get", key="actor:snack_yard.fed")["value"] == 0
    rt.call("input.key", key="I")
    assert rt.call("hud.inventory")["selected"] == "snack"
    rt.call("input.key", key="F")
    rt.call("time.advance", seconds=0.2)
    assert rt.call("state.get", key="actor:snack_yard.fed")["value"] == 1
    assert [i["count"] for i in rt.call("hud.inventory")["items"] if i["id"] == "snack"] == [1]
    rt.call("input.key", key="1")
    assert rt.call("hud.inventory")["selected"] == "room_key"
    rt.call("input.key", key="I")
    rt.call("input.key", key="I")                                                        # wraps round
    assert rt.call("hud.inventory")["selected"] == "room_key"
