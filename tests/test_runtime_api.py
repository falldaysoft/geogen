"""The Godot runtime's control API (--api, runtime/godot/scripts/api.gd) driven through
geogen.runtime_client: one runtime per module answers every test's questions.

Cottage layout (see test_godot_runtime.py): the front door is at z = 3 with its step in
front, the `front_path` spawn at z = 6.4, a bookshelf against the back wall and a resident NPC.
"""

import time

import pytest

from geogen.export import export_scene
from geogen.runtime_client import ApiError

pytestmark = pytest.mark.xdist_group("test_runtime_api")

INSIDE = [0.0, 0.32, 0.5]


@pytest.fixture(scope="module")
def rt(tmp_path_factory, built_scene, godot_runtime):
    out = tmp_path_factory.mktemp("generated")
    export_scene(built_scene("cottage"), out / "cottage.glb")
    client = godot_runtime("cottage", out)
    client.call("wait_ready")
    return client


def reset(rt, door="closed"):
    rt.call("time.resume")
    rt.call("time.scale", scale=1)
    rt.call("player.teleport", pos=[0, 0, 6.4], yaw=0)
    rt.call("interactions.set", asset="door", state=door, instant=True)


def test_info_lists_the_world(rt):
    info = rt.call("world.info")
    assert info["scene"] == "cottage"
    assert "front_path" in [s["name"] for s in info["spawns"]]
    assert info["npcs"] == ["resident_npc"]
    assert info["player_spec"]["radius"] > 0
    assert "player.walk_to" in rt.call("help")


def test_query_finds_tagged_objects_not_mesh_parts(rt):
    chairs = rt.call("world.query", tag="furniture.chair")
    assert len(chairs) == 4 and all(c["name"].startswith("chairs_") for c in chairs)
    furniture = rt.call("world.query", tag="furniture")
    assert {"table", "bookshelf"} <= {n["name"] for n in furniture}
    assert all(set(n["extras"]) - {"version", "collider", "walkable"} for n in rt.call("world.query"))
    door = rt.call("world.query", name="door", interactive=True)
    assert [d["name"] for d in door] == ["door"]


def test_closed_door_stops_walk_and_open_door_lets_player_in(rt):
    reset(rt, "closed")
    blocked = rt.call("player.walk_to", pos=INSIDE, timeout=8)
    assert blocked["status"] in ("stuck", "timeout")
    assert blocked["position"][2] > 2.8          # still outside the leaf
    assert blocked["blocked_by"]["asset"] == "door"
    assert blocked["blocked_by"]["interaction"] == {"name": "swing", "state": "closed"}
    assert len(blocked["trace"]) > 4 and blocked["trace"][0][2] > blocked["trace"][-1][2]

    reset(rt, "closed")
    rt.subscribe("interaction")
    rt.call("interactions.use", asset="door")
    event = rt.wait_event(lambda e: e["event"] == "interaction" and e["data"]["state"] == "open", timeout=10)
    assert event["data"]["asset"] == "door"
    rt.unsubscribe()
    walked = rt.call("player.walk_to", pos=INSIDE, timeout=15)
    assert walked["status"] == "arrived" and walked["distance"] < 0.5
    assert walked["position"][2] < 1.0
    assert walked["blocked_by"] is None


def test_set_and_wait_reports_where_parts_went(rt):
    reset(rt, "closed")
    closed = rt.call("world.node", name="door")
    leaf_closed = next(p for p in closed["interactions"][0]["parts"] if p["name"] == "leaf")
    opened = rt.call("interactions.set", asset="door", state="open", wait=True)[0]
    assert opened["state"] == "open" and not opened["moving"]
    leaf_open = next(p for p in opened["parts"] if p["name"] == "leaf")
    # The leaf swings into the house: thin along z when closed, deep along z when open.
    assert leaf_closed["bounds"]["size"][2] < 0.1 < 0.6 < leaf_open["bounds"]["size"][2]
    assert leaf_open["bounds"]["min"][2] < leaf_closed["bounds"]["min"][2]


def test_node_details(rt):
    door = rt.call("world.node", name="door")
    assert door["asset"] == "door" and door["extras"]["tags"] == ["door.exterior"]
    assert {"leaf", "frame", "step"} <= {c["name"] for c in door["children"]}
    assert door["collider_count"] >= 2
    assert 2.0 < door["bounds"]["size"][1] < 2.6
    by_path = rt.call("world.node", node_path=door["node_path"])
    assert by_path["position"] == door["position"]


def test_batch_runs_in_order_and_stops_on_error(rt):
    reset(rt)
    results = rt.call("batch", commands=[
        {"cmd": "player.teleport", "args": {"pos": [1, 0, 7]}},
        {"cmd": "player.status"},
        {"cmd": "interactions.set", "args": {"asset": "door", "state": "ajar"}},
        {"cmd": "ping"},
    ])
    assert [r["ok"] for r in results] == [True, True, False]
    assert results[1]["result"]["position"] == [1.0, 0.0, 7.0]


def test_time_advance_runs_world_time_fast(rt):
    reset(rt)
    before = rt.call("time.clock")
    out = rt.call("time.advance", seconds=40, scale=8)
    assert out["advanced"] >= 40 and out["real_ms"] < 40_000
    # The clock ran 40 s of world time: 40 min at the default 1440 s day.
    minutes = (out["time"]["hours"] - before["hours"]) % 24 * 60
    assert minutes == pytest.approx(40 * 1440 / before["day_length"], abs=0.5)
    assert out["time"]["scale"] == 1 and not out["time"]["paused"]
    assert out["npcs"][0]["npc"] == "resident_npc"


def test_reload_keeps_the_player(rt):
    reset(rt)
    rt.call("player.teleport", pos=[2, 0, 7])
    info = rt.call("world.reload")
    assert info["scene"] == "cottage"
    assert rt.call("player.status")["position"][:3:2] == [2.0, 7.0]
    rt.call("wait_ready")


def test_screenshots_need_a_window(rt):
    with pytest.raises(ApiError, match="headless"):
        rt.call("capture.views", node="door")


def test_aim_and_use(rt):
    reset(rt, "closed")
    rt.call("player.teleport", pos=[0, 0, 3.9], yaw=0)
    rt.call("player.look", at=[0, 1.0, 2.9])
    status = rt.call("player.status")
    assert status["focus"]["kind"] == "interaction" and status["focus"]["asset"] == "door"
    used = rt.call("player.use")
    assert used["asset"] == "door"
    assert rt.call("interactions.list", asset="door")[0]["target"] == "open"


def test_nav_path_and_raycast(rt):
    reset(rt, "open")
    path = rt.call("world.nav_path", **{"from": [0, 0, 6.4], "to": INSIDE})
    assert path["reached"] and 5 < path["length"] < 9
    hit = rt.call("world.raycast", **{"from": [0, 1.5, 2.0], "to": [0, 1.5, -10]})
    assert hit["hit"] and hit["asset"] == "bookshelf"


def test_sit_and_stand(rt):
    reset(rt, "open")
    rt.call("player.teleport", pos=INSIDE)
    sat = rt.call("player.sit", asset="chairs_0")
    assert sat["pose"] == "sit" and sat["pose_asset"] == "chairs_0"
    assert rt.call("player.stand")["pose"] == "stand"


def test_state_save_and_load(rt):
    reset(rt, "open")
    saved = rt.call("state.save")
    assert saved["door/swing"]["state"] == "open"
    rt.call("interactions.set", asset="door", state="closed", instant=True)
    restored = rt.call("state.load", state=saved)
    assert restored["door/swing"]["state"] == "open"


def test_pause_freezes_the_world_and_step_advances_it(rt):
    reset(rt)
    paused = rt.call("time.pause")
    assert paused["paused"]
    npc = rt.call("npcs.list")[0]["position"]
    time.sleep(0.5)
    assert rt.call("npcs.list")[0]["position"] == npc
    assert rt.call("time.pause")["frame"] == paused["frame"]
    stepped = rt.call("time.step", frames=30)
    assert stepped["paused"] and stepped["frame"] >= paused["frame"] + 30
    rt.call("time.resume")


def test_force_npc_option(rt):
    reset(rt)
    options = rt.call("npcs.options", npc="resident")
    assert options
    choice = options[-1]["id"]          # the least attractive one
    assert rt.call("npcs.force", npc="resident", option=choice)["forced"] == choice
    rt.subscribe("npc")
    rt.call("time.scale", scale=8)
    decided = rt.wait_event(lambda e: e["event"] == "npc" and e["data"].get("event") == "decide", timeout=60)
    rt.call("time.scale", scale=1)
    rt.unsubscribe()
    assert decided["data"]["chosen"] == choice


def test_errors_are_reported(rt):
    with pytest.raises(ApiError, match="unknown command"):
        rt.call("no.such")
    with pytest.raises(ApiError, match="no state 'ajar'"):
        rt.call("interactions.set", asset="door", state="ajar")
    with pytest.raises(ApiError, match="no NPC"):
        rt.call("npcs.greet", npc="nobody")
    assert rt.call("ping")["pong"]


def test_cli_arguments():
    from geogen.runtime_client import parse_args
    assert parse_args(['{"pos": [1, 2, 3]}']) == {"pos": [1, 2, 3]}
    assert parse_args(["pos=1,0,2.5", "asset=door", "wait=true", "frames=10"]) == {
        "pos": [1.0, 0.0, 2.5], "asset": "door", "wait": True, "frames": 10}
