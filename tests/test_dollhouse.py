"""The dollhouse example (examples/dollhouse, geogen.101.7): a street of four furnished houses
and the nine people who live in them. Nobody is scripted: each has needs and a planner, and
what they do comes from what the furnishing solver happened to put in their house. The night
is checked in the default tier, a whole simulated day in the exhaustive tier. Meals and washing
are planned from needs; going to bed is each person's night routine (`only: [bed], hold: true`)."""

import subprocess
import sys
from pathlib import Path

import pytest

from geogen import roots

EXAMPLE = Path(__file__).parent.parent / "examples" / "dollhouse"
SCENE = "lark_lane"
HOUSEHOLDS = {"no1": ["walter"], "no2": ["grace", "ben", "ada"], "no3": ["priya", "dan"], "no4": ["jo", "sam", "max"]}
PEOPLE = [who for house in HOUSEHOLDS.values() for who in house]

pytestmark = pytest.mark.xdist_group("test_dollhouse")      # one export for the module


@pytest.fixture(scope="module")
def built(project_scene):
    scene = project_scene(EXAMPLE, SCENE)
    try:
        roots.use(EXAMPLE)
        yield scene
    finally:
        roots.use()


def test_the_project_passes_its_checks():
    done = subprocess.run([sys.executable, "-m", "geogen.main", "check", "--project", str(EXAMPLE)],
                          capture_output=True, text=True, cwd=EXAMPLE.parent.parent)
    assert done.returncode == 0, done.stdout + done.stderr


def _assets(node) -> list[str]:
    return [str(n.meta.get("asset", "")) for n in node.iter_nodes()]


def test_every_house_is_furnished_for_the_people_in_it(built):
    houses = {n.name: n for n in built.children if n.name in HOUSEHOLDS}
    assert set(houses) == set(HOUSEHOLDS)
    for name, house in houses.items():
        assert not house.meta.get("furnish_report"), (name, house.meta.get("furnish_report"))
        names = [n.name for n in house.iter_nodes()]
        count = lambda stem: sum(1 for n in names if n == stem or n.startswith(stem + "_") and n[len(stem) + 1:].isdigit())  # noqa: E731
        people = len(HOUSEHOLDS[name])
        assert count("fridge") == 1 and count("sink") == 1 and count("vanity") == 1, name
        assert count("chair") >= min(people, 2), (name, count("chair"))     # three at a table for two take turns
        # A double bed for a couple, a single each otherwise: somewhere for everyone.
        assert count("bed") >= {"no1": 1, "no2": 2, "no3": 1, "no4": 3}[name], name
    # The students have a microwave and no range; everyone else cooks on one.
    cookers = {name: [n.name.split("_")[0] for n in house.iter_nodes() if n.name.split("_")[0] in ("range", "microwave")
                      and n.parent is not None and n.parent.name == "kitchen"] for name, house in houses.items()}
    assert cookers == {"no1": ["range"], "no2": ["range"], "no3": ["range"], "no4": ["microwave"]}


def test_everyone_plans_and_has_a_home_the_size_of_a_house(built):
    npcs = {n.name: n.meta["npc"] for n in built.iter_nodes() if "npc" in n.meta}
    assert set(npcs) == set(PEOPLE)
    for who, npc in npcs.items():                               # a home the size of one house (and Walter's garden)
        xs, zs = zip(*npc["home"]["polygon"])
        assert max(xs) - min(xs) < 11.0 and max(zs) - min(zs) < 13.0, (who, npc["home"])
    for who, npc in npcs.items():
        assert npc["planner"]["depth"] >= 3, who
        assert {"hunger", "rest", "hygiene"} <= set(npc["needs"]), who
        assert any(block.get("only") == ["bed"] and block.get("hold") for block in npc["routine"]), who


@pytest.fixture(scope="module")
def street_export(project_export):
    return project_export(EXAMPLE, SCENE)


def _people(rt) -> dict[str, dict]:
    return {n["npc"].removesuffix("_npc"): n for n in rt.call("npcs.list")}


def _launch(godot_runtime, generated):
    rt = godot_runtime(SCENE, generated)
    rt.call("wait_ready")
    rt.call("time.pause")           # the world runs only inside time.advance
    return rt


@pytest.mark.godot
def test_at_night_everyone_is_in_their_own_bed(godot_runtime, street_export):
    rt = _launch(godot_runtime, street_export)
    rt.call("time.clock", time="22:20")
    for _ in range(5):                                      # to 03:20: the students turn in at 02:00
        rt.call("time.advance", seconds=60, scale=8)
    people = _people(rt)
    beds = {who: n["doing"].split(":")[0] for who, n in people.items() if n["pose"] == "lie"}
    assert set(beds) == set(PEOPLE), {who: (n["pose"], n["doing"]) for who, n in people.items() if who not in beds}
    # A bed each, but for the couples, who share theirs (each owns it: `owns:`).
    thing = lambda who: beds[who].split("#")[0]            # noqa: E731
    assert thing("grace") == thing("ben") and thing("priya") == thing("dan")
    assert len({thing(who) for who in PEOPLE}) == 7 and len(set(beds.values())) == 9


@pytest.mark.godot
@pytest.mark.exhaustive
def test_over_a_day_everyone_cooks_eats_and_washes_by_plans_and_sleeps_at_night(godot_runtime, street_export):
    rt = _launch(godot_runtime, street_export)
    abed: dict[str, int] = {}                               # simulated hours each was seen lying in a bed
    spans = {"no1": (-23.0, -13.0), "no2": (-10.6, -0.4), "no3": (2.0, 12.0), "no4": (14.0, 25.2)}     # x, walls included
    for _ in range(24):                                     # 24 simulated minutes: one day (day_length 1440 s)
        rt.call("time.advance", seconds=60, scale=8)
        now = _people(rt)
        for house, residents in HOUSEHOLDS.items():         # nobody strays next door
            for who in residents:
                x, _, z = now[who]["position"]
                assert spans[house][0] < x < spans[house][1] and -9.0 < z < 4.3, (who, x, z)
                if now[who]["pose"] == "lie" and now[who]["doing"].startswith("bed"):
                    abed[who] = abed.get(who, 0) + 1
    people = _people(rt)
    assert set(people) == set(PEOPLE)
    for who, n in people.items():
        used = n["used"]
        did = lambda part: sum(count for id, count in used.items() if part in id)      # noqa: E731
        assert did(".eat") >= 2, (who, used)                # at a table, a meal they cooked in hand
        assert did("fridge") >= 2 and did("range") + did("microwave") >= 2, (who, used)
        assert abed.get(who, 0) >= 5, (who, abed)           # at night, by their routine (not a plan)
        assert did("vanity") + did("shower") >= 1, (who, used)
        assert n["plans_done"] >= 10 and n["ran_out"] == 0, (who, n["plans_done"], n["ran_out"])
        assert n["plans_done"] >= 0.8 * n["plans"], (who, n["plans"], n["plans_done"], n["failures"][-5:])
    # What is in the house decides how: the students never see a range.
    for who in HOUSEHOLDS["no4"]:
        assert not any("range" in id for id in people[who]["used"]), who
    # Walter potters in his front garden; nobody else's home reaches it.
    garden = lambda n: sum(c for id, c in n["used"].items() if "flowers" in id or "bench" in id)   # noqa: E731
    assert garden(people["walter"]) >= 1 and all(garden(n) == 0 for who, n in people.items() if who != "walter")
