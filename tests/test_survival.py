"""The survival example (examples/survival, geogen.84): a castaway island where everything is found,
combined, chopped, burnt, built and slept in through the core vocabulary alone, and hunger, thirst
and energy fall on the game clock through periodic rules. The project's checks, the loop on the
reference machine by both routes (driftwood and woven walls, or an axe and logs), the needs, the
quests, and a Godot playthrough over the control API."""

import subprocess
import sys
from pathlib import Path

import pytest

from geogen import roots
from geogen.quests import collect
from geogen.registry import SceneRegistry
from geogen.rules import Machine

EXAMPLE = Path(__file__).parent.parent / "examples" / "survival"
SCENE = "castaway"
PLAYER = {"player": {"kind": "player"}}


@pytest.fixture(scope="module")
def built():
    try:
        roots.use(EXAMPLE)
        registry = SceneRegistry()
        registry.discover()
        scene = registry[SCENE]()
        yield scene, collect(scene, SCENE)
    finally:
        roots.use()


def key(thing: str) -> str:
    return f"{SCENE}:{thing}"


def item(name: str) -> str:
    return f"actor:{SCENE}.item_{name}"


def need(name: str) -> str:
    return f"actor:{SCENE}.{name}"


class Castaway:
    """A player on the reference machine, with the things a test does often."""

    def __init__(self, program, time="08:00"):
        self.m = Machine(program, PLAYER)
        self.m.set_clock(time=time, day=0, night=False)

    def use(self, thing, seconds=0.5, ok=True):
        used = self.m.use(key(thing), "player")
        assert used == ok, (thing, used, self.m.feedback[-2:])
        self.m.advance(seconds)
        return used

    def state(self, thing):
        return self.m.get(key(thing))

    def held(self, name):
        return self.m.get(item(name), "player")

    def give(self, **items):
        for name, count in items.items():
            self.m.set(item(name), count, "player")

    def get(self, name):
        return self.m.get(need(name), "player")

    def set(self, name, value):
        self.m.set(need(name), value, "player")

    def clock(self, time, **more):
        self.m.set_clock(time=time, **more)

    def notes(self):
        return [f[1] for f in self.m.feedback if f[0] == "notify"]


@pytest.fixture
def castaway(built):
    _, quests = built
    return Castaway(quests.program())


def test_the_project_passes_its_checks():
    done = subprocess.run([sys.executable, "-m", "geogen.main", "check", "--project", str(EXAMPLE)],
                          capture_output=True, text=True, cwd=EXAMPLE.parent.parent)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_island_has_more_than_one_of_everything_to_find(built):
    scene, quests = built
    by_tag = {}
    for node in scene.iter_nodes():
        for tag in node.tags:
            if tag.startswith("resource."):
                by_tag[tag] = by_tag.get(tag, 0) + 1
    assert by_tag["resource.tree"] >= 30 and by_tag["resource.coconuts"] >= 10 and by_tag["resource.berries"] >= 8
    assert by_tag["resource.sticks"] >= 12 and by_tag["resource.stones"] >= 10 and by_tag["resource.fibre"] >= 20
    assert by_tag["resource.driftwood"] == 5 and by_tag["resource.flint"] == 1 and by_tag["resource.water"] == 1
    program = quests.program()
    periodic = [r["id"] for r in program["rules"] if "every" in r]
    assert periodic == ["hunger_falls", "thirst_falls", "energy_falls", "starving", "cold", "mending"]
    assert all(r["actor"] == "player" for r in program["rules"] if "every" in r)
    assert {i for i in quests.items} >= {"stick", "stone", "fibre", "log", "berries", "coconut", "flint", "raw_fish",
                                        "cooked_fish", "rope", "stone_axe", "fire_drill", "spear"}


def test_gathering_gives_and_the_pile_comes_back_on_the_clock(castaway):
    c = castaway
    c.use("sticks_1#gather")
    assert c.held("stick") == 2 and c.state("sticks_1#gather") == "taken"
    c.use("sticks_1#gather", ok=False)                                   # nothing to gather until it regrows
    c.clock("17:59")
    assert c.state("sticks_1#gather") == "taken"
    c.clock("18:00")                                                     # ten hours on
    assert c.state("sticks_1#gather") == "here"
    c.use("drift_1#take")
    assert c.held("log") == 1 and c.state("drift_1#take") == "taken"    # driftwood is finite
    c.clock("23:59", day=3)
    assert c.state("drift_1#take") == "taken"


def test_the_fire_by_the_drill_route_and_the_night_by_it(castaway):
    c = castaway
    for pile in ("sticks_1", "sticks_2"):
        c.use(f"{pile}#gather")
    c.use("stones_1#gather")
    c.use("grass_1#gather")
    assert (c.held("stick"), c.held("stone"), c.held("fibre")) == (4, 2, 2)
    c.use("workstone#drill", seconds=2.1)
    assert c.held("fire_drill") == 1 and (c.held("stick"), c.held("fibre")) == (2, 1)
    c.use("fire#fire", ok=False)                                         # no ring yet
    c.use("fire#build", ok=False)                                        # 3 sticks needed, 2 held
    c.use("sticks_2#gather", ok=False)                                   # taken: regrows later
    c.clock("18:30")                                                     # sticks_1 (08:00) is back at 18:00
    c.use("sticks_1#gather")
    c.use("fire#build", seconds=1.6)
    assert c.state("fire#build") == "built" and (c.held("stone"), c.held("stick")) == (0, 1)
    c.use("fire#fire", seconds=1.6)
    assert c.state("fire#fire") == "lit" and c.m.get(key("fire#lit"))
    c.use("fire#fire", seconds=1.6)                                      # feed it: a stick, burn restarts
    assert c.state("fire#fire") == "lit" and c.held("stick") == 0
    c.use("fire#fire", ok=False)                                         # no stick left to feed it
    c.clock("22:29")
    assert c.state("fire#fire") == "lit"
    c.clock("22:31")                                                     # four hours after the feed
    assert c.state("fire#fire") == "embers" and c.m.get(key("fire#lit"))
    c.clock("00:31", day=1)
    assert c.state("fire#fire") == "out" and not c.m.get(key("fire#lit"))


def test_the_flint_route_lights_it_too(castaway):
    c = castaway
    c.use("boulder#knap", ok=False)                                      # needs a stone to strike with
    c.give(stone=3, stick=3)
    c.use("boulder#knap", seconds=1.1)
    assert c.held("flint") == 1 and c.held("stone") == 3                 # the stone is kept
    c.use("boulder#knap", ok=False)                                      # an hour before the next flake
    c.use("fire#build", seconds=1.6)
    c.use("fire#fire", seconds=1.6)
    assert c.state("fire#fire") == "lit"


def test_the_axe_route_fells_trees_for_logs(castaway):
    c = castaway
    tree = next(n for n in (f"pines_{i}" for i in range(1, 40)) if c.m.get(key(f"{n}#chop"), None) is not None)
    c.use(f"{tree}#chop", ok=False)                                      # no axe
    c.give(stick=2, stone=1, fibre=1)
    c.use("workstone#axe", seconds=2.1)
    assert c.held("stone_axe") == 1 and c.held("stick") == 0
    c.use(f"{tree}#chop", seconds=1.3)
    assert c.state(f"{tree}#chop") == "notched" and c.held("log") == 0
    c.use(f"{tree}#chop", seconds=1.3)
    assert c.held("log") == 3
    c.m.advance(3.1)
    assert c.state(f"{tree}#chop") == "stump"
    c.use(f"{tree}#chop", ok=False)
    c.clock("08:00", day=2)                                              # two days on, it stands again
    assert c.state(f"{tree}#chop") == "standing"


def test_the_log_hut(castaway):
    c = castaway
    c.use("shelter#frame", ok=False)
    c.give(log=14, fibre=6, stick=3)
    c.use("shelter#frame", seconds=2.1)
    assert c.state("shelter#frame") == "framed" and c.held("log") == 10
    assert c.state("shelter#walls_logs") == "offer" and c.state("shelter#walls_woven") == "offer"
    c.use("shelter#roof_planks", ok=False)                               # pending: no walls yet
    c.use("shelter#walls_logs", seconds=2.6)
    assert c.held("log") == 4 and c.state("shelter#walls_woven") == "gone"
    assert c.state("shelter#roof_planks") == "offer" and c.state("shelter#roof_thatch") == "offer"
    c.use("shelter#roof_planks", seconds=2.6)
    assert c.held("log") == 0 and c.m.get(key("shelter#roofed")) and c.state("shelter#roof_thatch") == "gone"
    c.use("shelter#sleep", ok=False)                                     # no bed yet
    c.use("shelter#bed", seconds=2.1)
    assert (c.held("fibre"), c.held("stick")) == (0, 0) and c.state("shelter#sleep") == "ready"
    c.clock("21:00", night=True)
    c.set("energy", 20)
    c.set("hunger", 100)
    c.set("thirst", 100)
    c.use("shelter#sleep", seconds=0.7)
    assert c.m.clock["time"] == "06:00" and c.m.clock["day"] == 1 and c.get("energy") == 100
    assert c.get("hunger") == 64 and c.get("thirst") == 55            # the night's hours still passed
    c.m.advance(1.1)
    assert not c.get("asleep")
    assert "You sleep until morning" in c.notes()


def test_the_driftwood_and_woven_route_needs_no_axe(castaway):
    c = castaway
    for n in range(1, 6):
        c.use(f"drift_{n}#take")
    assert c.held("log") == 5
    c.use("shelter#frame", seconds=2.1)
    assert c.held("log") == 1
    c.use("shelter#walls_logs", ok=False)                                # 6 logs: the beach has no more
    c.give(fibre=19, stick=11)
    for _ in range(3):
        c.use("workstone#rope", seconds=2.1)
    assert c.held("rope") == 3 and c.held("fibre") == 10
    c.use("shelter#walls_woven", seconds=2.6)
    assert (c.held("stick"), c.held("rope")) == (3, 1) and c.state("shelter#walls_logs") == "gone"
    c.use("shelter#roof_thatch", seconds=2.6)
    assert (c.held("fibre"), c.held("rope")) == (0, 0) and c.m.get(key("shelter#roofed"))
    c.give(fibre=6)
    c.use("shelter#bed", seconds=2.1)
    assert c.state("shelter#sleep") == "ready"


def test_food_and_water_by_every_route(castaway):
    c = castaway
    c.set("hunger", 50)
    c.set("thirst", 50)
    c.use("spring#drink", seconds=1.6)
    assert c.get("thirst") == 100
    bush = next(n for n in (f"bushes_{i}" for i in range(1, 14)) if c.m.get(key(f"{n}#pick"), None) is not None)
    c.use(f"{bush}#eat", seconds=1.3)                                    # straight off the bush
    assert c.get("hunger") == 70 and c.state(f"{bush}#pick") == "picked"
    c.use(f"{bush}#pick", ok=False)                                      # eaten: nothing to pick till it ripens
    c.give(coconut=1)
    c.use("workstone#coconut", ok=False)                                 # needs a stone to crack it
    c.give(stone=1)
    c.set("thirst", 50)
    c.use("workstone#coconut", seconds=1.6)
    assert c.get("hunger") == 95 and c.get("thirst") == 85 and c.held("coconut") == 0
    c.use("fishing#fish", ok=False)                                      # no spear
    c.give(stick=1, flint=1, rope=1)
    c.use("workstone#spear", seconds=2.1)
    c.use("fishing#fish", seconds=4.1)
    assert c.held("raw_fish") == 1
    c.use("fire#cook", ok=False)                                         # the fire isn't lit
    c.give(stone=2, stick=3, fire_drill=1)
    c.use("fire#build", seconds=1.6)
    c.use("fire#fire", seconds=1.6)
    c.use("fire#cook", seconds=3.1)
    assert (c.held("raw_fish"), c.held("cooked_fish")) == (0, 1)
    c.set("hunger", 40)
    c.use("fire#eat", seconds=1.6)
    assert c.get("hunger") == 90 and c.held("cooked_fish") == 0


def test_needs_fall_on_the_clock_and_a_collapse_costs_the_night(castaway):
    c = castaway
    c.use("sticks_1#gather")                                             # the first input starts the periods
    for hour in range(9, 13):
        c.clock(f"{hour:02d}:00")
    assert (c.get("hunger"), c.get("thirst"), c.get("energy"), c.get("health")) == (84, 80, 88, 100)
    c.set("thirst", 0)
    c.clock("13:00")
    assert c.get("health") == 95 and "You are fading: eat, drink, rest" in c.notes()
    c.clock("14:00")
    assert c.get("health") == 90
    c.set("thirst", 60)                                                  # a drink: mending overtakes
    c.clock("15:00")
    assert c.get("health") == 94
    c.set("health", 3)
    c.set("thirst", 0)
    c.clock("16:00")                                                     # the hit takes it to 0: collapse
    assert c.m.clock["time"] == "06:00" and c.m.clock["day"] == 1
    assert "Everything goes dark. You wake at dawn, weak, your food gone" in c.notes()
    assert (c.get("health"), c.get("hunger"), c.get("thirst"), c.get("energy")) == (98, 34, 20, 80)


def test_the_cold_spares_a_fire_or_a_roof(castaway):
    c = castaway
    c.use("sticks_1#gather")
    c.clock("22:00", night=True)
    health = c.get("health")
    c.clock("23:00", night=True)
    assert c.get("health") == health - 3 and "You shiver in the cold" in c.notes()
    c.give(stone=2, stick=3, flint=1)
    c.use("fire#build", seconds=1.6)
    c.use("fire#fire", seconds=1.6)
    c.m.post({"type": "enter", "subject": key("fire/warmth"), "actor": "player"})
    health = c.get("health")
    c.clock("00:00", day=1, night=True)
    assert c.get("health") == health                                     # warm by the fire
    c.m.post({"type": "leave", "subject": key("fire/warmth"), "actor": "player"})
    c.m.post({"type": "enter", "subject": key("shelter/inside"), "actor": "player"})
    c.clock("01:00", day=1, night=True)
    assert c.get("health") == health - 3                                 # inside, but no roof yet
    c.m.set(key("shelter#frame"), "framed")
    c.m.set(key("shelter#walls_logs"), "done")
    c.m.set(key("shelter#roof_thatch"), "done")
    health = c.get("health")
    c.clock("02:00", day=1, night=True)
    assert c.get("health") == health


def test_the_quests_follow_the_player_by_either_route(castaway):
    c = castaway
    status = lambda q: c.get(f"quest_{q}")                               # noqa: E731
    assert status("first_night") == "active" and status("a_roof") == "available"
    c.give(stick=5, stone=2, fibre=2)
    assert c.get("quest_first_night_sticks") and c.get("quest_first_night_fibre")
    c.give(flint=1)                                                      # the flint route counts
    assert c.get("quest_first_night_spark")
    c.use("fire#build", seconds=1.6)
    c.use("fire#fire", seconds=1.6)
    assert c.get("quest_first_night_lit") and status("first_night") == "active"
    c.clock("06:00", day=1)
    assert status("first_night") == "complete" and status("a_roof") == "active"
    c.give(log=4)
    assert c.get("quest_a_roof_logs")
    c.use("shelter#frame", seconds=2.1)
    c.give(stick=8, rope=3, fibre=16, log=0)
    c.use("shelter#walls_woven", seconds=2.6)
    assert c.get("quest_a_roof_walls")
    c.use("shelter#roof_thatch", seconds=2.6)
    assert c.get("quest_a_roof_roof")
    c.give(stick=3)
    c.use("shelter#bed", seconds=2.1)
    c.set("hunger", 100)                                                # fed and watered: a night, not a collapse
    c.set("thirst", 100)
    c.use("shelter#sleep", seconds=0.7)
    assert status("a_roof") == "complete" and status("castaway_life") == "active"
    assert c.m.clock["day"] == 2
    c.use("spring#drink", seconds=1.6)
    assert c.get("quest_castaway_life_drink")
    c.clock("06:00", day=3)
    assert c.get("quest_castaway_life_days") and status("castaway_life") == "active"   # the fish is still to catch
