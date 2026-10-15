"""Tracks for spawned items (geogen.flows): belts and walk paths, joined by geometry, the
reference simulation, and the Godot runtime agreeing with it (scenes/flow_test.yaml)."""

import pytest

from geogen import flows
from geogen.layout.loader import LayoutLoader

SCENE = "flow_test"


def test_a_track_is_resampled_and_checked():
    track = flows.parse_track({"path": [[0, 0, 0], [0, 0, 2]], "speed": 0.5}, "t")
    assert track["length"] == pytest.approx(2.0) and len(track["points"]) == 9
    assert track["mode"] == "belt" and track["end"] == "auto"
    loop = flows.parse_track({"path": [[0, 0, 0], [2, 0, 0], [2, 0, 2]], "loop": True}, "t")
    assert loop["points"][0] == loop["points"][-1] and loop["end"] == "stop"
    for spec, message in (({"path": [[0, 0, 0]]}, "two or more"), ({"path": [[0, 0, 0], [0, 0, 1]], "mode": "fly"}, "belt or walk"),
                          ({"path": [[0, 0, 0], [0, 0, 1]], "end": "bin"}, "end is one of"),
                          ({"path": [[0, 0, 0], [0, 0, 1]], "colour": 1}, "unknown keys")):
        with pytest.raises(flows.FlowError, match=message):
            flows.parse_track(spec, "t")


def test_a_conveyor_carries_its_track_on_its_top():
    belt = LayoutLoader().load("assets/conveyor.yaml", params={"length": 2, "height": 0.9})
    track = belt.meta["flows"]["track"]
    assert track["points"][0] == pytest.approx([0, 0.9, -1]) and track["points"][-1] == pytest.approx([0, 0.9, 1])


def test_flow_test_joins_its_belts_and_feeds_its_tracks(built_scene):
    net = flows.network(built_scene(SCENE))
    names = [t["node"].name for t in net["tracks"]]
    a, b, lane = names.index("belt_a"), names.index("belt_b"), names.index("lane")
    assert net["tracks"][a]["next"] == b and net["tracks"][b]["next"] is None and net["tracks"][lane]["next"] is None
    feeds = {s["node"].name: s["track"] for s in net["sources"]}
    assert feeds == {"parcels": a, "garage": lane}
    assert flows.check_network(built_scene(SCENE)) == []


def test_the_reference_moves_items_along_queues_them_and_takes_them_in(built_scene):
    scene = built_scene(SCENE)
    names = [t["node"].name for t in flows.network(scene)["tracks"]]
    out = flows.simulate(scene, 60 * 30)
    received = dict(zip(names, out["received"]))
    # Parcels: 0.6 a second over 4 + 3 m of belt at 0.6 m/s: the first lands after ~12.5 s.
    assert 8 <= received["belt_b"] <= 12, out["received"]
    assert received["lane"] >= 3, out["received"]                 # rovers: 0.25 a second, ~11 s to drive it
    # Stopping the second belt backs parcels up on both: spaced by their radii, never overlapping.
    held = flows.simulate(scene, 60 * 40, running={"belt_b": False})
    assert dict(zip(names, held["received"]))["belt_b"] == 0
    on_a = sorted(it["s"] for it in held["items"] if names[it["track"]] == "belt_a")
    assert len(on_a) >= 8 and all(b - a >= 0.3 for a, b in zip(on_a, on_a[1:])), on_a


def test_bad_track_networks_are_reported(tmp_path):
    from geogen.layout.composer import SceneComposer

    (tmp_path / "items").mkdir()
    (tmp_path / "items/parcel.yaml").write_text("kind: item\nbody: parcel.yaml\nmode: belt\n")
    (tmp_path / "parcel.yaml").write_text(open("assets/parcel.yaml").read())
    (tmp_path / "spawner.yaml").write_text(open("assets/spawner.yaml").read())
    (tmp_path / "yard.yaml").write_text("name: yard\nsize: [4, 2, 4]\nplace:\n  s: {asset: spawner.yaml}\n")
    found = flows.check_network(SceneComposer(assets_dir=tmp_path).compose(tmp_path / "yard.yaml"))
    assert found and "need a track that starts within 0.5 m" in found[0]


# --- in Godot -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime(godot_runtime, scene_export):
    client = godot_runtime(SCENE, scene_export(SCENE))
    client.call("wait_ready", timeout=60)
    return client


def test_godot_steps_tracks_as_the_reference_does(runtime, built_scene):
    runtime.call("time.advance", seconds=20.0, scale=1)
    report = runtime.call("flows.list")
    expected = flows.simulate(built_scene(SCENE), int(report["ticks"]))
    got = sorted((it["id"], it["track"], it["item"]) for it in report["items"])
    want = sorted((it["id"], it["track"], it["item"]) for it in expected["items"])
    assert got == want, (got, want)
    s_got = {it["id"]: it["s"] for it in report["items"]}
    for it in expected["items"]:
        assert s_got[it["id"]] == pytest.approx(it["s"], abs=1e-3), it
    assert [t["received"] for t in report["tracks"]] == expected["received"]
    assert [s["emitted"] for s in report["sources"]] == expected["emitted"]


def test_godot_rules_count_what_the_tracks_take_in(runtime):
    report = runtime.call("flows.list")
    points = runtime.call("state.get", key="world:flow_test.delivered")["value"]
    by_uid = {t["uid"]: t["received"] for t in report["tracks"]}
    assert points == by_uid["flow_test:belt_b"] + 5 * by_uid["flow_test:lane"]
    shown = runtime.call("hud.stats")["stats"]["stats"]
    assert [s["label"] for s in shown] == ["Parcels", "Rovers", "Points"]


def test_godot_stopping_a_belt_backs_items_up(runtime):
    runtime.call("state.set", key="flow_test:belt_b#running", value=False)
    before = runtime.call("flows.list")
    runtime.call("time.advance", seconds=8.0, scale=1)
    after = runtime.call("flows.list")
    b = next(t for t in after["tracks"] if t["uid"] == "flow_test:belt_b")
    assert b["received"] == next(t for t in before["tracks"] if t["uid"] == "flow_test:belt_b")["received"]
    held = sorted(it["s"] for it in after["items"] if it.get("track") == b["index"])
    assert held and all(y - x >= 0.3 for x, y in zip(held, held[1:])), held
    runtime.call("state.set", key="flow_test:belt_b#running", value=True)
