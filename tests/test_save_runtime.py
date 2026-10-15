"""Saving in the Godot runtime (WorldLoader.save_state / load_state): `--save` / `--load` and the control
API's `state.save` / `state.load` write and take up a geogen-save (docs/schema/geogen-save.v1.schema.json)."""

import json
from pathlib import Path

import pytest

from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from test_state import GATE, LAMP, LEVER
from test_wiring import PLATE, QUEST, SCENE

pytestmark = pytest.mark.xdist_group("test_save_runtime")

SCHEMA = Path(__file__).parent.parent / "docs/schema/geogen-save.v1.schema.json"
# The yard of test_wiring, with a spawn: a lever that opens a gate, a lamp, two plates.
YARD = SCENE + "spawns:\n  start: {position: [0, 0, 5], facing: south}\n"
GATE_KEY, LEVER_KEY = "yard:gate#gate", "yard:lever#pull"
SECTIONS = {"format", "version", "package", "now", "clock", "world", "actors", "actor_info", "timers"}


def validate(save):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(save, json.loads(SCHEMA.read_text()))
    return save


@pytest.fixture(scope="module")
def yard_dir(tmp_path_factory):
    assets = tmp_path_factory.mktemp("save_assets")
    for name, text in (("lever", LEVER), ("gate", GATE), ("lamp", LAMP), ("plate", PLATE), ("yard", YARD),
                       ("quests/stand", QUEST)):
        path = assets / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    out = tmp_path_factory.mktemp("generated_save")
    export_scene(SceneComposer(assets_dir=assets).compose(assets / "yard.yaml"), out / "yard.glb")
    return out


@pytest.fixture(scope="module")
def rt(yard_dir, godot_runtime):
    client = godot_runtime("yard", yard_dir)
    client.call("wait_ready")
    return client


def played(rt):
    """Something of every kind away from how it starts."""
    rt.call("interactions.set", asset="gate", state="open", instant=True)        # an interaction
    rt.call("interactions.lock", asset="gate", locked=True)                      # and its lock
    rt.call("state.set", key="yard:lamp#colour", value="cool")                   # instance variables
    rt.call("state.set", key="yard:plate_2#occupants", value=3)
    rt.call("state.set", key="world:yard.power", value=False)                    # a world variable
    rt.call("state.set", key="actor:yard.pulls", value=5)                        # an actor variable
    return rt.call("state.save")


def test_a_save_has_world_and_actor_sections(rt):
    fresh = validate(rt.call("state.save"))
    assert set(fresh) == SECTIONS and (fresh["format"], fresh["version"], fresh["package"]) == ("geogen-save", 1, "yard")
    assert fresh["world"] == {} and fresh["actors"] == {"player": {}}             # nothing differs from its default
    saved = validate(played(rt))
    # flat: interactions are instance state beside the declared variables, the keys name the owners
    assert saved["world"] == {GATE_KEY: "open", GATE_KEY + "_locked": True, "yard:lamp#colour": "cool",
                              "yard:plate_2#occupants": 3, "world:yard.power": False}
    assert saved["actors"] == {"player": {"actor:yard.pulls": 5}}
    assert saved["actor_info"]["player"]["kind"] == "player"
    assert "yard:lamp#lit" not in saved["world"]                                 # a binding: never saved


def test_a_save_round_trips_and_resets_what_it_leaves_out(rt, tmp_path):
    played(rt)
    path = tmp_path / "save.json"
    saved = rt.call("state.save", path=str(path))
    assert validate(json.loads(path.read_text())) == saved
    # move everything, and something the save doesn't mention
    rt.call("interactions.lock", asset="gate", locked=False)
    rt.call("interactions.set", asset="gate", state="closed", instant=True)
    rt.call("interactions.set", asset="lever", state="down", instant=True)
    rt.call("state.set", key="yard:lamp#colour", value="warm")
    rt.call("state.set", key="yard:lamp#level", value=9)
    rt.call("state.set", key="world:yard.power", value=True)
    rt.call("state.set", key="actor:yard.pulls", value=0)
    assert rt.call("state.save")["world"] == {LEVER_KEY: "down", "yard:lamp#level": 9, "yard:plate_2#occupants": 3}
    restored = rt.call("state.load", path=str(path))
    assert {k: restored[k] for k in ("world", "actors", "timers")} == {k: saved[k] for k in ("world", "actors", "timers")}
    (gate,) = rt.call("interactions.list", asset="gate")
    assert gate["state"] == "open" and gate["locked"] is True
    assert rt.call("interactions.list", asset="lever")[0]["state"] == "up"        # as it started
    assert rt.call("state.get", key="yard:lamp#level")["value"] == 2
    assert rt.call("state.get", key="world:yard.power")["value"] is False
    assert rt.call("state.get", key="yard:lamp#lit")["value"] is False            # bindings are recomputed
    assert rt.call("state.get", key="actor:yard.pulls")["value"] == 5
    with pytest.raises(Exception, match="a save"):
        rt.call("state.load", path=str(tmp_path / "missing.json"))


def test_a_save_from_before_the_format_still_loads(rt):
    saved = played(rt)
    program = {k: saved[k] for k in ("format", "version", "now", "clock", "actors", "actor_info", "timers")}
    program["world"] = {k: v for k, v in saved["world"].items() if "#gate" not in k}
    old = {GATE_KEY: {"state": "closed", "locked": False}, "gate/gate": {"state": "ignored"},
           "geogen:rules": {**program, "actors": {"player": {"actor:yard.pulls": 8}}}}
    rt.call("state.set", key="actor:yard.pulls", value=0)
    now = rt.call("state.load", state=old)
    assert validate(now)["actors"]["player"] == {"actor:yard.pulls": 8}
    assert GATE_KEY not in now["world"] and GATE_KEY + "_locked" not in now["world"]
    assert now["world"]["world:yard.power"] is False and now["world"]["yard:lamp#colour"] == "cool"
    rt.call("state.load", state={"gate/gate": {"state": "open"}})                 # before stable IDs: by node name
    assert rt.call("state.save")["world"][GATE_KEY] == "open"


def test_save_and_load_from_the_command_line(run_godot, yard_dir, tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    base = ("--scene", "yard", f"--generated={yard_dir}")
    run_godot(*base, "--set=world:yard.power=false", "--use=lever", "--quit-after=90", f"--save={first}")
    saved = validate(json.loads(first.read_text()))
    assert saved["world"][LEVER_KEY] == "down" and saved["world"][GATE_KEY] == "open"     # the wire opened it
    assert saved["world"]["world:yard.power"] is False
    player = saved["actors"]["player"]
    assert player["actor:yard.pulls"] == 1 and player["actor:yard.quest_stand"] == "active"
    run_godot(*base, f"--load={first}", "--quit-after=5", f"--save={second}")
    again = validate(json.loads(second.read_text()))
    assert again["world"] == saved["world"] and again["actors"] == saved["actors"]
    assert again["now"] > saved["now"]


def test_a_load_restores_the_world_clock(rt):
    rt.call("time.clock", time="21:30", day_length=0)          # stopped, so it reads back exactly
    saved = rt.call("state.save")
    assert saved["clock"]["time"] == "21:30" and saved["clock"]["night"] is True
    rt.call("time.clock", time="08:00")
    rt.call("state.load", state=saved)
    assert rt.call("time.clock")["clock"] == "21:30"           # the sky and sun, not only the rules' clock
    rt.call("time.step", frames=90)                            # past the next minute: no re-sync back
    assert rt.call("state.save")["clock"]["time"] == "21:30"
    rt.call("time.resume")


def test_set_and_quests_act_on_the_loaded_save(run_godot, yard_dir, tmp_path):
    path = tmp_path / "pulled.json"
    base = ("--scene", "yard", f"--generated={yard_dir}")
    run_godot(*base, "--set=actor:yard.pulls=4", "--quit-after=5", f"--save={path}")
    out = run_godot(*base, f"--load={path}", "--set=world:yard.power=false", "--quests", "--quit-after=5",
                    f"--save={path}")
    saved = validate(json.loads(path.read_text()))
    assert saved["world"]["world:yard.power"] is False                         # --set wasn't overwritten
    assert saved["actors"]["player"]["actor:yard.pulls"] == 4                  # and the load happened
    assert any(line.startswith("quests: ") for line in out.splitlines())


# ---- an NPC's state and a pending timer, in the quest yard -------------------------------------------

STEWARD = "quest_yard:steward"
DASH = "actor:quest_yard.quest_yard_dash"


@pytest.fixture(scope="module")
def quest_rt(scene_export, godot_runtime):
    client = godot_runtime("quest_yard", scene_export("quest_yard"))
    client.call("wait_ready")
    return client


def test_npc_state_and_pending_timers_survive(quest_rt):
    rt = quest_rt
    rt.call("state.set", key=DASH, value="complete", actor=STEWARD)              # an NPC's own value
    rt.call("interactions.set", asset="switch_2", state="off", instant=True)
    rt.call("time.pause")
    rt.call("player.teleport", pos=[-10, 0.1, 3])                                # the start square: six seconds to run
    rt.call("time.step", frames=30)
    assert rt.call("state.get", key=DASH)["value"] == "active"
    saved = validate(rt.call("state.save"))
    assert saved["actors"][STEWARD][DASH] == "complete" and saved["actors"]["player"][DASH] == "active"
    assert saved["actor_info"][STEWARD]["kind"] == "npc"
    assert saved["world"]["quest_yard:switch_2#switch"] == "off"
    (timer,) = [t for t in saved["timers"] if t["kind"] == "rule"]
    assert timer["rule"].endswith("dash_time_up") and timer["actor"] == "player"
    assert 4.0 < timer["at"] - saved["now"] < 6.0

    rt.call("time.advance", seconds=8)                                           # time runs out
    assert rt.call("state.get", key=DASH)["value"] == "failed"
    rt.call("state.set", key=DASH, value="available", actor=STEWARD)
    rt.call("interactions.set", asset="switch_2", state="on", instant=True)
    assert not [t for t in rt.call("state.save")["timers"] if t["kind"] == "rule"]

    restored = rt.call("state.load", state=saved)
    assert restored["timers"] == saved["timers"] and restored["now"] == pytest.approx(saved["now"])
    assert restored["actors"][STEWARD] == saved["actors"][STEWARD]
    assert rt.call("state.get", key=DASH)["value"] == "active"
    assert rt.call("state.get", key=DASH, actor=STEWARD)["value"] == "complete"
    assert rt.call("interactions.list", asset="switch_2")[0]["state"] == "off"
    rt.call("time.advance", seconds=3)
    assert rt.call("state.get", key=DASH)["value"] == "active"                    # the restored timer hasn't fired yet
    rt.call("time.advance", seconds=5)
    assert rt.call("state.get", key=DASH)["value"] == "failed"                    # and now it has, a second time
    rt.call("time.resume")


# A gate that takes 2 s to open and closes itself 3 s after it is open (geogen.3.14).
SELF_CLOSING = GATE.replace("      open:   {emit: opened}", "      open:   {emit: opened, then: closed, after: 3}")
DOOR_YARD = """
name: door_yard
size: [8, 3, 8]
slots:
  middle: {position: [0, 0, 0]}
place:
  gate: {asset: gate.yaml, slot: middle}
spawns:
  start: {position: [0, 0, 4], facing: south}
"""
DOOR_KEY = "door_yard:gate#gate"


@pytest.fixture(scope="module")
def door_rt(tmp_path_factory, godot_runtime):
    assets = tmp_path_factory.mktemp("door_assets")
    (assets / "gate.yaml").write_text(SELF_CLOSING)
    (assets / "door_yard.yaml").write_text(DOOR_YARD)
    out = tmp_path_factory.mktemp("generated_door")
    export_scene(SceneComposer(assets_dir=assets).compose(assets / "door_yard.yaml"), out / "door_yard.glb")
    client = godot_runtime("door_yard", out)
    client.call("wait_ready")
    return client


def gate(rt):
    (it,) = rt.call("interactions.list", asset="gate")
    return it


def test_a_save_keeps_a_move_and_a_dwell_part_way(door_rt):
    rt = door_rt
    rt.call("time.pause")
    rt.call("interactions.set", asset="gate", state="open")                    # 2 s on its way
    rt.call("time.step", frames=60)
    halfway = validate(rt.call("state.save"))
    (timer,) = [t for t in halfway["timers"] if t.get("key") == DOOR_KEY]
    assert (timer["kind"], timer["state"], timer["from"]) == ("arrive", "open", "closed")
    assert timer["at"] - halfway["now"] == pytest.approx(1.0, abs=0.05)
    rt.call("time.step", frames=60 + 60)                                         # open, and 1 s into its 3
    waiting = validate(rt.call("state.save"))
    (timer,) = [t for t in waiting["timers"] if t.get("key") == DOOR_KEY]
    assert (timer["kind"], timer["state"]) == ("dwell", "open")
    assert timer["at"] - waiting["now"] == pytest.approx(2.0, abs=0.05)

    # Half-open on its way: still moving after half a second, open by a second and a bit.
    rt.call("interactions.set", asset="gate", state="closed", instant=True)
    rt.call("state.load", state=halfway)
    again = [t for t in rt.call("state.save")["timers"] if t.get("key") == DOOR_KEY]
    assert again[0]["kind"] == "arrive" and again[0]["at"] - halfway["now"] == pytest.approx(1.0, abs=0.05)
    assert gate(rt)["moving"] and gate(rt)["target"] == "open"
    rt.call("time.step", frames=30)
    assert gate(rt)["moving"]
    rt.call("time.step", frames=40)
    assert gate(rt)["state"] == "open" and not gate(rt)["moving"]

    # Waiting to close: it doesn't wait the whole 3 s again, only the 2 that were left.
    rt.call("interactions.set", asset="gate", state="closed", instant=True)
    rt.call("state.load", state=waiting)
    assert gate(rt)["state"] == "open"
    rt.call("time.step", frames=100)                                             # 1.7 s
    assert gate(rt)["target"] == "open"
    rt.call("time.step", frames=30)                                              # 2.2 s
    assert gate(rt)["target"] == "closed"
    rt.call("time.resume")
