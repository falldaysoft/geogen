"""The cottage resident in the Godot runtime (headless, deterministic, sped up).

The NPC is driven entirely by data (assets/npcs/, affordances, the door's
portal); these tests assert on its ``npc summary`` after a simulated span.
"""

import json

import pytest

from geogen.export import export_scene
from geogen.main import _build_registry
from geogen.runtime_client import step

FAST = ("--fixed-fps", "60")   # unthrottled: frames run back to back


@pytest.fixture(scope="module")
def cottage_dir(scene_export):
    return scene_export("cottage")


def simulate(godot_script, generated, seconds, scene="cottage", before=(), **start) -> dict:
    """The scene's one NPC's report after ``seconds`` of world time at 8x."""
    run = godot_script([*before, step("time.simulate", seconds=seconds)], f"--scene={scene}",
                       f"--generated={generated}", start={"timescale": 8, **start}, engine_args=FAST)
    (report,) = run["time.simulate"]["npcs"]
    return report


def test_resident_idles_before_deciding(godot_script, cottage_dir):
    # Straight after load the resident stands where the scene placed it.
    report = simulate(godot_script, cottage_dir, 0.02)
    assert report["npc"] == "resident_npc"
    assert report["pose"] == "stand"
    assert report["position"][1] == pytest.approx(0.32, abs=0.03)   # on the floorboards


def test_humanoid_resident_plays_clips(godot_script, cottage_dir):
    # The resident is a skinned humanoid: she walks to her first window (walk clip), idles
    # there, sits down in the armchair ~14 s in (the sit_down transition clip, then pose_sit on
    # her own AnimationPlayer) and later stands up (stand_up) and idles again.
    assert simulate(godot_script, cottage_dir, 0.9)["clip"] == "walk"
    assert simulate(godot_script, cottage_dir, 4)["clip"] == "idle"
    lowering = simulate(godot_script, cottage_dir, 15)
    assert (lowering["pose"], lowering["clip"], lowering["transitions"]) == ("sit", "sit_down", 1)
    sitting = simulate(godot_script, cottage_dir, 20)
    assert (sitting["pose"], sitting["clip"]) == ("sit", "pose_sit")
    later = simulate(godot_script, cottage_dir, 100)
    assert (later["pose"], later["clip"], later["transitions"]) == ("stand", "idle", 2)
    assert later["cost_us"] > 0                           # script time per physics frame


def test_at_the_table_her_hands_rest_on_its_top(godot_script, cottage_dir):
    # A dining chair's action is sit_at_table: sit, holding the pose with the forearms on the table
    # (hand_rest.gd). The pose is made for a default table and reach; her hands find the real
    # top ahead of them and the arm IK puts them on it, a little way in from its edge.
    seated = simulate(godot_script, cottage_dir, 140)
    assert (seated["pose"], seated["clip"]) == ("sit", "pose_sit_at_table")
    hands = seated["hands"]
    assert hands["weight"] == 1.0 and set(hands["hands"]) == {"LeftHand", "RightHand"}
    for hand in hands["hands"].values():
        support, at = hand["support"], hand["at"]
        assert support is not None and support[1] == pytest.approx(1.07, abs=0.02)   # the dining table's top
        assert 0.05 < hand["shift"] < 0.3                       # moved forward onto it, not far


def test_she_opens_the_front_door_by_its_handle(godot_script, cottage_dir):
    # The door's portal names its knobs and that it swings into the house: coming in, she stands
    # close, reaches for the outside knob (hands.gd), presses once her hand is there and holds
    # it as the door opens; going on, she pushes it shut by the inside knob within her reach.
    report = simulate(godot_script, cottage_dir, 100)
    assert report["hands"]["reaches"] >= 2


def test_resident_sits_facing_the_seat_in_a_turned_cottage(godot_script, cottage_dir, scene_export):
    # house_plot turns the cottage to face east, so the NPC's parent is rotated: poses and
    # turns must use world headings (she used to sit sideways here, backwards on the island).
    sitting = simulate(godot_script, scene_export("house_plot"), 20, scene="house_plot")
    assert (sitting["pose"], sitting["doing"]) == ("sit", "armchair#0: wait duration")
    assert sitting["pose_yaw_error"] == pytest.approx(0, abs=1)
    plain = simulate(godot_script, cottage_dir, 20)
    assert sitting["yaw"] == pytest.approx(plain["yaw"] + 90, abs=1)


def test_resident_lives_in_the_cottage(godot_script, cottage_dir):
    report = simulate(godot_script, cottage_dir, 600)
    used = report["used"]
    kinds = {key.split("#")[0].split("_")[0] for key in used}
    # Sits (armchair / dining chairs), looks out of windows, steps out of the door.
    assert {"armchair", "chairs", "window", "door"} <= kinds, used
    assert report["distinct"] >= 6
    assert report["passes"] >= 2          # out through the front door and back in
    assert report["failures"] == []
    assert report["max_stall"] < 2.0      # never pinned against geometry
    assert report["unstuck"] == 0         # never wedged (and put back on the navmesh)
    assert report["outside"] == 0.0       # stayed home (the doorstep is within the margin)


def test_resident_walks_around_the_player(godot_script, cottage_dir):
    # The player stands between the table and the armchair, on the armchair's
    # approach, for the whole run: paths bend around them, and only the
    # armchair (whose approach they block) is given up on.
    report = simulate(godot_script, cottage_dir, 300, spawn=[1.5, 0.32, 1.0])
    assert report["distinct"] >= 6
    assert report["max_stall"] < 2.0
    assert all("destination occupied" in f for f in report["failures"]), report["failures"]


def test_resident_greets_and_carries_on(godot_script, cottage_dir):
    # The player, behind her at the window, presses E on her: she turns, waves, and goes back to
    # her day; she keeps an eye on the player while they're close and ahead of her.
    run = godot_script([step("wait", seconds=3), step("npcs.greet", npc="resident", wait=30),
                        step("time.simulate", seconds=9)], "--scene=cottage", f"--generated={cottage_dir}",
                       start={"timescale": 8, "spawn": [1.5, 0.32, 0.03], "yaw": -90}, engine_args=FAST)
    assert run["npcs.greet"] == {"npc": "resident_npc", "ok": True}
    (report,) = run["time.simulate"]["npcs"]
    assert (report["greets"], report["waves"]) == (1, 1)
    assert report["looking"] > 1.0
    assert report["failures"] == []
    assert report["decisions"] >= 1 and report["clip"] != "wave"


def test_resident_steps_aside_for_the_player(godot_script, cottage_dir):
    # The player walks straight at her while she looks out of the window: she steps aside.
    out = godot_script([step("wait", seconds=3), step("player.move", seconds=2.5)], "--scene=cottage",
                       f"--generated={cottage_dir}", "--npc-trace", start={"spawn": [1.0, 0.32, 0.03], "yaw": -90},
                       engine_args=FAST).output
    assert '"event":"yield"' in out
    steps = [json.loads(l.removeprefix("npc: ")) for l in out.splitlines() if l.startswith("npc: ")]
    aside = [s for s in steps if s.get("action") == "step_aside" and s.get("step") == {"face": "player"}]
    assert aside and abs(aside[0]["pos"][2] - 0.03) > 0.2          # moved off the player's line


# --- spiral stairs (scenes/lighthouse_keepers.yaml) --------------------------------------------

def keepers(godot_script, generated, seconds) -> dict:
    run = godot_script([step("time.simulate", seconds=seconds)], "--scene=lighthouse_keepers",
                       f"--generated={generated}", start={"timescale": 8}, engine_args=FAST)
    return {r["npc"].removesuffix("_npc"): r for r in run["time.simulate"]["npcs"]}


def test_npcs_take_a_spiral_stair_by_its_walking_line(godot_script, scene_export):
    # The keeper starts on the ground and lives on the lighthouse gallery, 10 m up; the lookout
    # starts up there and lives outside the door. The navmesh path between has the stair's
    # navigation link in it: two points, a straight line through the treads. Walking that left
    # each stuck against the stair; they follow the link's walking line round instead.
    generated = scene_export("lighthouse_keepers")
    axis = lambda r: (r["position"][0] ** 2 + r["position"][2] ** 2) ** 0.5

    # 14 s in the lookout is half way down, on the treads (0.09 .. 1.19 m from the axis). The
    # stair is one body wide, so the keeper waits at its foot rather than meet them on it.
    early = keepers(godot_script, generated, 14)
    assert 2.0 < early["lookout"]["position"][1] < 8.0 and 0.3 < axis(early["lookout"]) < 1.0, early["lookout"]
    # On the treads her feet go onto the steps (feet.gd); on flat ground his stay where the clip puts them.
    stepping = early["lookout"]["feet"]
    assert stepping["active"] and max(abs(f["offset"]) for f in stepping["feet"].values()) > 0.05, stepping
    assert all(abs(f["offset"]) < 0.02 for f in early["keeper"]["feet"]["feet"].values()), early["keeper"]["feet"]
    assert early["keeper"]["position"][1] < 0.3 and axis(early["keeper"]) > 1.3, early["keeper"]
    assert early["keeper"]["doing"] == "self/wander: go_to random"

    # A minute in they have changed places, and neither was ever stuck.
    late = keepers(godot_script, generated, 60)
    assert late["keeper"]["position"][1] == pytest.approx(10.0, abs=0.1), late["keeper"]
    assert 1.2 < axis(late["keeper"]) < 3.4                               # on the deck, off the stair well
    assert late["lookout"]["position"][1] == pytest.approx(0.0, abs=0.1) and late["lookout"]["position"][2] > 3.0
    for report in late.values():
        assert report["failures"] == [] and report["unstuck"] == 0, report
        assert report["max_stall"] < 2.0, report
        assert report["used"].get("self/wander", 0) >= 1


def test_the_player_on_a_stair_keeps_npcs_off_it(godot_script, scene_export):
    # geogen.15.20: the player holds no stair claim, but standing on the treads (or coming the
    # other way) they keep an NPC from taking it: the keeper waits at the foot and the lookout
    # at the top until the player steps off; then they change places as usual.
    on_the_treads = [-0.11, 1.7, 0.74]                     # a sixth of the way up the walking line
    run = godot_script([step("player.teleport", pos=on_the_treads), step("time.simulate", seconds=20),
                        step("player.teleport", pos=[0, 0, 6]), step("time.simulate", seconds=60)],
                       "--scene=lighthouse_keepers", f"--generated={scene_export('lighthouse_keepers')}",
                       start={"timescale": 8}, engine_args=FAST)
    held, after = ({r["npc"].removesuffix("_npc"): r for r in sim["npcs"]} for sim in run.all("time.simulate"))
    assert held["keeper"]["position"][1] < 0.3, held["keeper"]           # still at the foot
    assert held["lookout"]["position"][1] > 9.5, held["lookout"]         # still on the gallery
    assert after["keeper"]["position"][1] == pytest.approx(10.0, abs=0.1), after["keeper"]
    assert after["lookout"]["position"][1] == pytest.approx(0.0, abs=0.1), after["lookout"]


# --- NPCs holding quests (geogen.15.8) ----------------------------------------------------------

def test_the_resident_switches_off_every_light_for_its_own_quest(godot_script, tmp_path):
    # With `resident_quest`, the resident holds "lights out" from the start (its placement's
    # quests:) and does it, with no player: to each switch, press it off, quest complete.
    from pathlib import Path

    from geogen.layout.composer import SceneComposer

    assets = Path(__file__).parent.parent / "assets"
    root = SceneComposer(assets_dir=assets).compose(assets / "scenes" / "cottage.yaml", params={"resident_quest": 1})
    export_scene(root, tmp_path / "cottage.glb")
    manifest = json.loads((tmp_path / "cottage.manifest.json").read_text())
    goals = manifest["quests"]["quests"]["lights_out"]["objectives"][0]["goals"]
    assert {g["state"] for g in goals} == {"off"} and len(goals) == 3
    report = simulate(godot_script, tmp_path, 60)
    assert report["quests"] == {"lights_out": "complete"}, report
    assert report["quest_goals"] == 3 and report["failures"] == [], report
    assert sum(1 for k in report["used"] if k.startswith("quest/lights_out/")) == 3


LIGHTHOUSE_QUEST = """
name: lighthouse_errand
size: [16, 14, 16]
spawns:
  start: { position: [3.5, 0, 6.5], facing: north }
slots:
  door: { position: [0, 0, 5.0], facing: north }
place:
  lighthouse: { asset: lighthouse.yaml }
  climber:
    npc: npcs/resident.yaml
    slot: door
    home: { rect: [-3.0, 3.5, 3.0, 7.5] }       # the ground outside the door
    seed: 2
    quests: [quests/climb_lighthouse.yaml]
"""


def test_an_npc_climbs_to_the_lighthouse_gallery_for_its_quest(godot_script, tmp_path):
    # An NPC living on the ground outside the lighthouse is given "up the lighthouse": it goes
    # in, up the spiral stair and onto the gallery (a place), which completes the quest.
    from pathlib import Path

    from geogen.layout.composer import SceneComposer

    assets = Path(__file__).parent.parent / "assets"
    scene = tmp_path / "lighthouse_errand.yaml"
    scene.write_text(LIGHTHOUSE_QUEST)
    out = tmp_path / "generated"
    export_scene(SceneComposer(assets_dir=assets).compose(scene), out / "lighthouse_errand.glb")
    report = simulate(godot_script, out, 60, scene="lighthouse_errand")
    assert report["quests"] == {"climb_lighthouse": "complete"}, report
    assert report["quest_goals"] >= 1 and report["failures"] == [] and report["unstuck"] == 0, report


# --- ownership (geogen.11.17) -------------------------------------------------------------------

TWO_RESIDENTS = """
name: two_residents
size: [14, 6, 14]
place:
  cottage: { scene: scenes/cottage.yaml, params: { quest: 0 } }
  guest:
    npc: npcs/resident.yaml
    on: cottage/house.floor
    at: { u: { abs: 2.0 }, v: { abs: 3.2 } }
    home: cottage/house.floor
    seed: 5
    owns: [cottage/armchair]
"""


def test_only_its_owner_sits_in_an_owned_armchair(godot_script, tmp_path):
    # A placement's `owns` (references) becomes the NPC's owned stable IDs: the cottage's own
    # resident never takes the guest's armchair, and the guest likes it best.
    from pathlib import Path

    from geogen.layout.composer import SceneComposer

    path = tmp_path / "two_residents.yaml"
    path.write_text(TWO_RESIDENTS)
    root = SceneComposer(assets_dir=Path(__file__).parent.parent / "assets").compose(path)
    export_scene(root, tmp_path / "out" / "two_residents.glb")
    run = godot_script([step("time.simulate", seconds=240)], "--scene=two_residents",
                       f"--generated={tmp_path / 'out'}", start={"timescale": 8}, engine_args=FAST)
    reports = {r["npc"].removesuffix("_npc"): r for r in run["time.simulate"]["npcs"]}
    armchair = lambda r: sum(n for k, n in r["used"].items() if k.startswith("armchair"))   # noqa: E731
    assert armchair(reports["resident"]) == 0, reports["resident"]["used"]
    assert armchair(reports["guest"]) >= 1, reports["guest"]["used"]


def test_far_off_npcs_sleep_and_carry_on(godot_script, scene_export):
    # Off screen and further than SLEEP_DISTANCE (80 m) an NPC steps every 8th frame with its
    # clips paused (geogen.11.16): the crossroads' pedestrians, with the player 200 m away, sleep
    # most frames and still go about their day; back beside them they are awake.
    run = godot_script([step("player.teleport", pos=[200, 0.15, 200]), step("time.simulate", seconds=30)],
                       "--scene=crossroads", f"--generated={scene_export('crossroads')}",
                       start={"timescale": 4}, engine_args=FAST)
    reports = run["time.simulate"]["npcs"]
    assert reports and all(r["slept"] > 0 for r in reports), [(r["npc"], r["slept"]) for r in reports]
    assert sum(r["decisions"] for r in reports) >= len(reports)
    assert all(r["failures"] == [] for r in reports)


def test_a_cook_steps_up_to_the_counter_and_works_with_hands_on_it(godot_script, scene_export):
    # kitchen_work: Marco at a prep counter and a range. The navmesh keeps a player's width off
    # them; his work actions step up past it (`close`) and loop chop / stir with his left hand on
    # the real top (hands.gd: resting, a centimetre or two at most from where the clip puts it).
    generated = scene_export("kitchen_work")
    run = godot_script([step("npcs.force", npc="marco_npc", option="counter#0"),
                        step("time.simulate", seconds=10, id="chop"),
                        step("npcs.force", npc="marco_npc", option="range#0"),     # once he's done chopping
                        step("time.simulate", seconds=150, id="later")],
                       "--scene=kitchen_work", f"--generated={generated}", start={"timescale": 8}, engine_args=FAST)
    (chop,) = run["chop"]["npcs"]
    assert chop["doing"] == "counter#0: loop duration" and chop["clip"] == "chop", chop["doing"]
    assert chop["close_ins"] >= 1 and chop["name"] == "Marco Bellini"
    hand = chop["hands"]["hands"]["LeftHand"]
    assert hand["support"] is not None and hand["shift"] < 0.03, hand
    (later,) = run["later"]["npcs"]
    # (A wander in so small a room may pick where the player stands; the work never fails.)
    assert not [f for f in later["failures"] if f.startswith(("counter", "range"))] and later["unstuck"] == 0, later
    assert {"counter#0", "range#0"} <= set(later["used"]), later["used"]
