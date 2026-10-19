"""The quest bot (geogen/quest_bot.py, geogen.15.9): every quest in a scene played with the
player's real body over the control protocol, so each is a regression test of the scenes it
walks through. A quest it can't finish is reported with what stopped it."""

import pytest

from geogen import quest_bot


@pytest.mark.parametrize("scene, started", [
    # How each quest got going: already active (start: auto), given by an NPC the bot greeted,
    # or started by the bot when no one it asked gave it (the dash is given by a place).
    ("quest_yard", {"yard_round": "active", "yard_lamps": "by greeting quest_yard:steward", "yard_dash": "by the bot"}),
    ("cottage", {"lights_out": "by greeting cottage:resident"}),
    ("state_test", {"two_levers": "active"}),          # levers at the marked puzzles, then a place
    ("level_crossing", {"catch_train": "active"}),     # wait on the (polygon) platform for the train
])
def test_the_bot_finishes_every_quest(godot_runtime, scene_export, scene, started):
    rt = godot_runtime(scene, scene_export(scene))
    rt.call("wait_ready")
    report = quest_bot.play(rt)
    assert report["ok"], report
    assert {k: v["started"] for k, v in report["quests"].items()} == started


@pytest.mark.exhaustive
def test_the_bot_delivers_the_parcel_to_room_207(godot_runtime, scene_export):
    # The receptionist gives the quest and the key in a conversation (accept); the bot walks up
    # through the stair halls' doors, unlocks 207 with the key and greets the guest on her bed.
    rt = godot_runtime("hotel_showcase", scene_export("hotel_showcase"))
    report = quest_bot.play(rt, ["deliver_parcel"], timeout=600)
    assert report["ok"], report
    parcel = report["quests"]["deliver_parcel"]
    assert parcel["started"] == "by greeting hotel_showcase:hotel/receptionist"
    assert ["accept"] == [a for step in parcel["steps"] for a in step.get("answers", []) if a]


@pytest.mark.exhaustive
def test_the_bot_takes_the_jetty_trip_across_scenes(godot_runtime, built_scene, tmp_path_factory):
    # From the hub: through its island portal (the objective's subject is in another scene), out
    # along the jetty, and back through the island's hub portal (a marker on a travel point).
    from geogen.catalogue import export_catalogue, load_catalogue

    out = tmp_path_factory.mktemp("generated_quest_bot")
    catalogue = load_catalogue()
    export_catalogue(catalogue, built_scene, out, catalogue.select("hub,island"), log=lambda *_: None)
    rt = godot_runtime("hub", out)
    report = quest_bot.play(rt, ["jetty_trip"], timeout=600)
    assert report["ok"], report
    assert rt.call("player.status")["scene"] == "hub"


class _Runtime:
    """Just enough of a control client for a quest whose objective gives the bot nothing to do."""

    def __init__(self):
        self.calls = []

    def call(self, cmd, **args):
        self.calls.append(cmd)
        if cmd == "quests.list":
            return [{"id": "riddle", "package": "demo", "status": "active", "title": "A riddle",
                     "objectives": [{"id": "answer", "text": "Answer the riddle", "optional": False, "state": "open"}]}]
        if cmd == "player.status":
            return {"scene": "demo", "position": [0, 0, 0]}
        return {}


def test_an_objective_without_goals_is_reported():
    report = quest_bot.play(_Runtime())
    assert not report["ok"]
    riddle = report["quests"]["riddle"]
    assert riddle["status"] == "active" and riddle["objectives"] == {"answer": "open"}
    assert "no goal the bot can work on" in riddle["problem"]
