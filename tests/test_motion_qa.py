"""Motion QA (geogen/motion_qa.py, geogen.103.9): every motion scenario played by each archetype's
shortest, tallest and heaviest person, with clipping sampled through the motion.

The baseline (tests/data/motion_qa_baseline.json, regenerate with ``python tests/test_motion_qa.py``)
records what every run shows today; the guard fails when a number gets worse than both its limit
(``motion_qa.LIMITS``) and the baseline. What already holds for everyone is strict: no pops, no
bare heels. Runs over a limit today are listed by ``python -m geogen.motion_qa`` (and the
baseline): they are tasks, not accepted.

Default tier: the affordance scenarios (press a button, sit, take and set down) with the shortest
and tallest townsfolk; ``--all``: every scenario and every archetype's extremes.
"""

import json
from pathlib import Path

import pytest

BASELINE = Path(__file__).parent / "data" / "motion_qa_baseline.json"
CORE_SCENARIOS = ("press_tv", "sit_on_chair", "take_carry_set_down")
CORE_PEOPLE = (("townsfolk", "short"), ("townsfolk", "tall"))
SLACK_MM = 2.0


def _cases():
    from geogen.motion_qa import archetype_names, extremes
    from geogen.motion_review import scenario_names

    people = [(a, label, person) for a in archetype_names() for label, person in extremes(a).items()]
    out = []
    for s in scenario_names():
        for a, label, person in people:
            core = s in CORE_SCENARIOS and any(a == ca and cl in label.split("+") for ca, cl in CORE_PEOPLE)
            out.append(pytest.param(s, a, label, person, id=f"{s}/{a}/{label}",
                                    marks=() if core else pytest.mark.exhaustive))
    return out


CASES = _cases()


def _run(scenario, person):
    from geogen.motion_qa import run

    return run(scenario, person)


def _worse(now: dict, then: dict) -> list[str]:
    """Numbers past their limit that got worse than the baseline."""
    from geogen.motion_qa import LIMITS

    out = []
    for key, value in now.items():
        base = then.get(key)
        if key in ("person", "peak_joint_speed", "clipping_frames") or base is None:
            continue
        if isinstance(value, dict) and "max_mm" in value:
            limit = max(LIMITS.get(key, 2.0), base["max_mm"] + SLACK_MM)
            if value["max_mm"] > limit or (value["max_mm"] > LIMITS.get(key, 2.0)
                                           and value["count"] > base["count"] * 1.15 + 5):
                out.append(f"{key}: {value} (baseline {base})")
        elif isinstance(value, (int, float)):
            slack = 0 if key == "pops" else SLACK_MM
            if value > max(LIMITS.get(key, 0), base + slack):
                out.append(f"{key}: {value} (baseline {base})")
    return out


@pytest.mark.parametrize("scenario,archetype,label,person", CASES)
def test_motion_qa_no_worse_than_baseline(scenario, archetype, label, person):
    key = f"{scenario}/{archetype}/{label}"
    baseline = json.loads(BASELINE.read_text())
    assert key in baseline, f"{key} isn't in the baseline: run python tests/test_motion_qa.py"
    now = _run(scenario, person)
    assert _worse(now, baseline[key]) == []
    # What holds for everyone today: no pops, no bare heels in shoes.
    assert now["pops"] == 0
    assert not (now.get("feet_skin") or {}).get("count"), now["feet_skin"]


def test_extremes_are_the_archetypes_smallest_and_largest():
    from geogen.motion_qa import extremes

    women = extremes("older_woman")
    assert women["short"]["preset"] == "feminine+petite" and women["short"]["height"] == pytest.approx(1.53)
    assert women["heavy"]["build"] == pytest.approx(1.3)
    young = extremes("young_man")
    assert young["tall"]["preset"] == "masculine+tall"
    # one person who is both shortest and heaviest is listed once, under both labels
    assert "short+heavy" in young and len(young) == 2


def test_measure_actor_is_measure_on_a_posed_actor():
    from geogen.clipping import measure, measure_actor, posed, rest_meshes
    from geogen.layout.loader import LayoutLoader
    from geogen.roots import CORE

    character = LayoutLoader().load(CORE / "characters" / "humanoid.yaml",
                                    params={"preset": ["feminine"], "outfit": "dress"})
    assert measure_actor(posed(character, "sit"), rest_meshes(character)) == measure(character, "sit")


def test_props_finds_a_body_inside_furniture():
    """The detector: a person stood half into the table reports how deep."""
    from geogen.motion_qa import _Props
    from geogen.motion_review import _actor, _scene
    from geogen.roots import CORE

    scene = _scene({"slots": {"t": {"position": [0, 0, 0]}},
                    "place": {"table": {"asset": "table.yaml", "slot": "t"}}}, CORE)
    actor = _actor({"archetype": "characters/archetypes/young_woman.yaml", "seed": 1}, scene, CORE)

    class Rig:
        attached = {}

    Rig.actor, Rig.scene = actor, scene
    found = _Props(Rig).depth(Rig)["props"]
    assert found["count"] > 20 and found["max_mm"] > 2, found     # (a thin top: depth tops out at half its thickness)
    assert "in" in found["where"]


if __name__ == "__main__":        # regenerate the baseline
    from concurrent.futures import ProcessPoolExecutor

    cases = [c.values for c in CASES]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_run, [c[0] for c in cases], [c[3] for c in cases]))
    data = {f"{s}/{a}/{label}": r for (s, a, label, _), r in zip(cases, results)}
    BASELINE.write_text(json.dumps(data, indent=1, sort_keys=True, default=float) + "\n")
    print(f"wrote {BASELINE}")
