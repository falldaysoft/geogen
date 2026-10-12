"""Rules and quests running in the Godot runtime (runtime/godot/scripts/rules.gd): the manifest's
program drives real interactions, and quest progress is reported and saved."""

import json

import pytest

from geogen.export import export_scene, manifest_path
from geogen.layout.composer import SceneComposer
from test_state import GATE, LAMP, LEVER
from test_wiring import PLATE, QUEST, SCENE

pytestmark = pytest.mark.xdist_group("test_quests_runtime")

QUEST_KEY = "actor:yard.quest_stand"
# The yard of test_wiring, with a spawn: slots a..e hold the lever, gate, lamp and two plates.
YARD = SCENE + "spawns:\n  start: {position: [0, 0, 5], facing: south}\n"


@pytest.fixture(scope="module")
def yard_dir(tmp_path_factory):
    assets = tmp_path_factory.mktemp("yard_assets")
    for name, text in (("lever", LEVER), ("gate", GATE), ("lamp", LAMP), ("plate", PLATE), ("yard", YARD),
                       ("quests/stand", QUEST)):
        path = assets / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    out = tmp_path_factory.mktemp("generated_quests")
    export_scene(SceneComposer(assets_dir=assets).compose(assets / "yard.yaml"), out / "yard.glb")
    return out


@pytest.fixture(scope="module")
def rt(yard_dir, godot_runtime):
    client = godot_runtime("yard", yard_dir)
    client.call("wait_ready")
    return client


def quest(rt, actor="player"):
    (entry,) = rt.call("quests.list", actor=actor)
    return entry


def objectives(rt):
    return {o["id"]: o["state"] for o in quest(rt)["objectives"]}


def test_a_given_quest_runs_on_real_interactions(rt, yard_dir):
    manifest = json.loads(manifest_path(yard_dir / "yard.glb").read_text())
    assert {"state", "rules", "quests"} <= set(manifest)
    assert quest(rt)["status"] == "available" and objectives(rt) == {"lit": "waiting", "open": "waiting"}
    assert rt.call("state.get", key="yard:gate#gate")["value"] == "closed"

    rt.subscribe("quest", "events", types=["signal", "changed"])
    rt.call("player.use", asset="lever")                     # the rule counts the pull and gives the quest
    started = rt.wait_event(lambda m: m["event"] == "quest")["data"]
    assert started == {"quest": "stand", "title": "Stand in the light", "actor": "player", "event": "started",
                       "status": "active", "from": "available"}
    assert rt.call("state.get", key="actor:yard.pulls")["value"] == 1
    assert objectives(rt) == {"lit": "open", "open": "open"}

    # The lever arrives, its signal drives the gate's input, and the real gate opens: an objective.
    done = rt.wait_event(lambda m: m["event"] == "quest" and m["data"].get("objective") == "open", timeout=20)["data"]
    assert done["event"] == "objective_done" and done["text"] == "Open the gate"
    assert rt.call("interactions.set", asset="gate", state="open", wait=True)       # let it finish moving
    assert rt.call("state.get", key="yard:gate#gate")["value"] == "open"
    assert objectives(rt) == {"lit": "open", "open": "done"}

    # Standing on the plate lights the lamp (a binding over the plate's own counter): the last objective.
    rt.call("player.teleport", pos=[3, 0.1, 3], yaw=0)
    finished = rt.wait_event(lambda m: m["event"] == "quest" and m["data"].get("status") == "complete", timeout=20)
    assert finished["data"]["from"] == "active"
    rt.unsubscribe()
    assert rt.call("state.get", key="yard:plate#occupants")["value"] == 1
    assert rt.call("state.get", key="yard:lamp#lit")["value"] is True
    assert quest(rt)["status"] == "complete" and objectives(rt) == {"lit": "done", "open": "done"}
    # what the program did is on the bus, marked as its own
    history = rt.call("events.history", types=["changed", "signal"])["events"]
    by_key = {e.get("key"): e for e in history if e["type"] == "changed"}
    assert by_key["yard:lamp#lit"]["origin"] == "rules" and by_key["yard:lamp#lit"]["data"]["to"] is True
    assert by_key[QUEST_KEY]["actor"] == "player" and by_key[QUEST_KEY]["origin"] == "rules"
    assert "origin" not in by_key["yard:lever#pull"]                         # the player did that
    assert [e for e in history if e["type"] == "signal" and e["name"] == "switched_on"]     # the lamp's declared signal
    # nobody else's progress moved
    assert quest(rt, actor="npc:someone")["status"] == "available"


def test_state_is_read_and_set_like_a_console(rt):
    stored = rt.call("state.list")
    assert stored["actors"]["player"][QUEST_KEY] == "complete" and stored["world"]["yard:plate#occupants"] == 1
    assert not [k for values in stored["actors"].values() for k in values if k.startswith(("once:", "watch:"))]
    assert rt.call("state.set", key="world:yard.power", value=False)["value"] is False
    assert rt.call("state.get", key="yard:lamp#lit")["value"] is False       # the binding followed
    rt.call("state.set", key="world:yard.power", value=True)
    assert rt.call("state.set", key="yard:gate#gate", value="closed")["value"] == "closed"     # drives the node
    assert rt.call("interactions.set", asset="gate", state="closed", wait=True)
    for bad, message in (({"key": "world:yard.nope", "value": 1}, "isn't declared"),
                         ({"key": "actor:yard.pulls", "value": "many"}, "number"),
                         ({"key": "yard:gate#gate", "value": "ajar"}, "choice")):
        with pytest.raises(Exception, match=message):
            rt.call("state.set", **bad)


def test_progress_survives_a_save_and_a_reload(rt):
    saved = rt.call("state.save")
    assert saved["geogen:rules"]["actors"]["player"][QUEST_KEY] == "complete"
    rt.call("state.set", key=QUEST_KEY, value="active")
    rt.call("state.set", key="actor:yard.pulls", value=7)
    assert quest(rt)["status"] == "active"
    rt.call("state.load", state=saved)
    assert quest(rt)["status"] == "complete" and rt.call("state.get", key="actor:yard.pulls")["value"] == 1
    # the package's state is kept while another world is loaded (here: the same scene, reloaded)
    rt.call("world.reload")
    assert quest(rt)["status"] == "complete" and objectives(rt) == {"lit": "done", "open": "done"}


def test_the_island_quest_completes_on_the_gallery(scene_export, run_godot):
    """The 'go there' archetype end to end: starts with the game, done on stepping onto the gallery."""
    out = run_godot("--scene=island", f"--generated={scene_export('island')}", "--quests", "--spawn=39.5,15.5,7",
                    "--walk=1", engine_args=("--fixed-fps", "60"))
    events = [json.loads(l.removeprefix("quest event: ")) for l in out.splitlines() if l.startswith("quest event: ")]
    assert [(e["event"], e.get("objective")) for e in events] == [
        ("started", None), ("objective_done", "gallery"), ("complete", None)]
    assert all(e["actor"] == "player" and e["quest"] == "lighthouse_gallery" for e in events)
    (listed,) = json.loads(next(l for l in out.splitlines() if l.startswith("quests: ")).removeprefix("quests: "))
    assert listed["status"] == "active" and listed["objectives"][0]["state"] == "open"
    assert listed["objectives"][0]["markers"] == [
        {"subject": "island:lighthouse/gallery", "position": [39.5, 15.454, 7.0]}]


@pytest.fixture(scope="module")
def cottage(scene_export, godot_runtime):
    client = godot_runtime("cottage", scene_export("cottage"))
    client.call("wait_ready")
    return client


def test_the_resident_gives_the_lights_out_quest_and_it_counts_the_switches(cottage):
    """The 'change the world' archetype end to end: greeting the resident gives it, each real
    switch moves the count, and the last one completes it."""
    rt = cottage

    def lights():
        (entry,) = rt.call("quests.list")
        (objective,) = entry["objectives"]
        return entry["status"], objective["state"], objective["progress"]

    assert rt.call("quests.list")[0]["id"] == "lights_out" and lights() == ("available", "waiting", [0, 3])
    rt.call("player.use", asset="dining_switch")             # nobody asked yet: the count moves, the quest doesn't
    assert lights() == ("available", "waiting", [1, 3])
    rt.call("player.use", asset="dining_switch")
    assert lights() == ("available", "waiting", [0, 3])

    rt.subscribe("quest", "events", types=["signal"], subjects=["cottage:resident"])
    for _ in range(30):                                       # they greet standing: wait out a sit or a turn
        if rt.call("npcs.greet", npc="resident")["ok"]:
            break
        rt.call("time.advance", seconds=2)
    else:
        pytest.fail("the resident never stood still to be greeted")
    greeted = rt.wait_event(lambda m: m["event"] == "events")["data"]
    assert (greeted["type"], greeted["subject"], greeted["name"], greeted["actor"]) == (
        "signal", "cottage:resident", "greeted", "player")
    started = rt.wait_event(lambda m: m["event"] == "quest")["data"]
    assert started == {"quest": "lights_out", "title": "Lights out", "actor": "player", "event": "started",
                       "status": "active", "from": "available"}
    status, state, progress = lights()
    assert (status, state, progress) == ("active", "open", [0, 3])
    markers = rt.call("quests.list")[0]["objectives"][0]["markers"]
    assert sorted(m["subject"] for m in markers) == ["cottage:dining_switch", "cottage:lamp_switch",
                                                     "cottage:sitting_switch"]

    # One by aiming at it and pressing E, as a player would; the others by name.
    light = "cottage/world/cottage/dining_light/Light"
    rt.call("player.teleport", pos=[0.9, 0.32, 1.9], yaw=180)
    rt.call("player.look", at=[0.85, 1.42, 2.67])
    focus = rt.call("player.status")["focus"]
    assert focus == {"kind": "interaction", "asset": "dining_switch", "interaction": "switch", "prompt": "Lights off"}
    assert rt.call("world.node", node_path=light)["visible"] is True
    assert rt.call("player.use")["asset"] == "dining_switch"
    assert lights() == ("active", "open", [1, 3])
    rt.call("interactions.set", asset="dining_switch", state="off", wait=True)    # the rocker arrives
    assert rt.call("world.node", node_path=light)["visible"] is False             # and the light is out
    rt.call("player.use", asset="lamp_switch")
    assert lights() == ("active", "open", [2, 3])
    rt.call("player.use", asset="sitting_switch")
    events = [rt.wait_event(lambda m: m["event"] == "quest")["data"] for _ in range(2)]
    assert [(e["event"], e.get("objective")) for e in events] == [("objective_done", "lights"), ("complete", None)]
    assert events[0]["text"] == "Switch off every light in the cottage" and events[1]["from"] == "active"
    rt.unsubscribe()
    assert lights() == ("complete", "done", [3, 3])
    rt.call("player.use", asset="sitting_switch")             # a light back on: done stays done
    assert lights() == ("complete", "done", [2, 3])
    assert quest(rt, actor="cottage:resident")["status"] == "available"       # the resident holds nothing
