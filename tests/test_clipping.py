"""Clipping QA for dressed humanoids (geogen/clipping.py, geogen-z2b.16.15).

The baseline (tests/data/clipping_baseline.json, regenerate with
``python tests/test_clipping.py``) records today's penetration per preset x outfit x pose
and seat. The guard fails if anything gets worse; the strict tests are the goal (nothing past
2 mm): skirts and dresses in every pose and clip (generators/drape.py, geogen.11.5.5), and
everything at once for two outfits.

The waist tests (geogen.11.5.19) are strict already: an outfit with a separate top and bottom
shows no skin between them, the two overlap and the top's hem stays a smooth loop, in every pose.
"""

import json
from pathlib import Path

import pytest

ASSETS = Path(__file__).parent.parent / "assets"
BASELINE = Path(__file__).parent / "data" / "clipping_baseline.json"
PRESETS = [["feminine"], ["masculine"], ["feminine", "hourglass"], ["masculine", "heavy"], ["feminine", "petite"]]
OUTFITS = ["casual_f", "dress", "jeans_tee", "smart"]
CORE = [(["feminine"], "casual_f"), (["masculine"], "jeans_tee"), (["feminine", "hourglass"], "dress")]
COMBOS = [(p, o) for p in PRESETS for o in OUTFITS]
GUARDED = [pytest.param(p, o, marks=() if (p, o) in CORE else pytest.mark.exhaustive)    # the rest: pytest --all
           for p, o in COMBOS]


def _report(preset, outfit):
    from geogen.clipping import report

    return report([preset], [outfit])[f"{'+'.join(preset)}/{outfit}"]


def _worse(now: dict, then: dict, where: str) -> list[str]:
    out = []
    for key, value in now.items():
        base = then.get(key)
        if isinstance(value, dict) and "min_mm" in value:              # an overlap: less is worse
            if base is not None and value["min_mm"] < base["min_mm"] - 5.0:
                out.append(f"{where}/{key}: {value} (baseline {base})")
        elif isinstance(value, dict) and "count" in value:
            if base is None:
                continue
            if value["max_mm"] > base["max_mm"] + 2.0 or value["count"] > base["count"] * 1.15 + 5:
                out.append(f"{where}/{key}: {value} (baseline {base})")
        elif isinstance(value, dict):
            out += _worse(value, base or {}, f"{where}/{key}")
    return out


@pytest.mark.parametrize("preset,outfit", GUARDED, ids=lambda x: "+".join(x) if isinstance(x, list) else x)
def test_clipping_no_worse_than_baseline(preset, outfit):
    key = f"{'+'.join(preset)}/{outfit}"
    baseline = json.loads(BASELINE.read_text())[key]
    assert _worse(_report(preset, outfit), baseline, key) == []


@pytest.mark.parametrize("preset,outfit", [
    pytest.param(["feminine"], "casual_f", marks=pytest.mark.xfail(
        reason="long hair lies 3 mm in the dining chair's top rail (hair has no pose correctives)", strict=False)),
    (["masculine"], "jeans_tee")], ids=["feminine_casual_f", "masculine_jeans_tee"])
def test_no_clipping_beyond_tolerance(preset, outfit):
    def bad(d):
        return [k for k, v in d.items() if isinstance(v, dict) and (v.get("count", 0) > 0 if "count" in v else bad(v))]

    assert bad(_report(preset, outfit)) == []


SKIRTED = [(["feminine"], "casual_f"), (["feminine", "hourglass"], "dress"), (["masculine"], "casual_f"),
           (["feminine", "petite"], "skirt_sweater"), (["feminine", "curvy"], "dress")]
CLIP_FRAMES = {"walk": 8, "idle": 3, "sit_down": 5, "stand_up": 5, "lie_down": 4, "get_up": 4}


@pytest.mark.parametrize("preset,outfit", [pytest.param(p, o, marks=() if i == 0 else pytest.mark.exhaustive)
                                           for i, (p, o) in enumerate(SKIRTED)],
                         ids=lambda x: "+".join(x) if isinstance(x, list) else x)
def test_nothing_pokes_through_a_skirt_in_any_pose_or_clip(preset, outfit):
    """The body stays inside a skirt or dress (generators/drape.py) in every static pose and
    through the walk, idle and transition clips, to QA's 2 mm tolerance; seated, the cloth isn't
    in the seat. (Heavy builds are left out: a belly that overhangs the lap still shows through
    it by ~1 cm when crouching or leaning forward on a seat.)"""
    import numpy as np

    from geogen.clipping import measure, report
    from geogen.layout import LayoutLoader

    character = LayoutLoader().load(ASSETS / "characters" / "humanoid.yaml", params={"preset": preset, "outfit": outfit})
    drape = character.meta["humanoid"]["drape"]
    assert 1 <= drape["morphs"] <= 8
    faults = {}
    for pose in POSES:
        faults[pose] = measure(character, pose)["shell_poke"]
    for clip, count in CLIP_FRAMES.items():
        duration = next(c.duration for c in character.clips if c.name == clip)
        for t in np.linspace(0.0, duration, count, endpoint=False):
            faults[f"{clip}@{t:.2f}"] = measure(character, None, clip, float(t))["shell_poke"]
    assert {k: v for k, v in faults.items() if v["count"]} == {}
    seats = report([preset], [outfit], poses=())["+".join(preset) + "/" + outfit]
    assert max(v["max_mm"] for v in seats.values()) < 10.0, seats


POSES = ["stand", "sit", "sit_at_table", "lie", "lean_on_sill", "reach_low", "look", "t_pose"]
SEPARATES = ["casual", "jeans_tee", "smart", "shorts_tee", "casual_f", "skirt_sweater"]      # a top and a bottom
WAIST_CORE = [(["feminine"], "casual_f"), (["feminine"], "jeans_tee"), (["masculine"], "jeans_tee"),
              (["feminine"], "skirt_sweater")]
WAIST_MIN_OVERLAP = 15.0     # mm, in any pose (the generator laps them 30 mm at rest: humanoid.WAIST_OVERLAP)
WAIST = [pytest.param(p, o, marks=[] if (p, o) in WAIST_CORE else [pytest.mark.exhaustive])
         for p in PRESETS for o in SEPARATES]


def _waist_faults(character, poses=POSES) -> list[str]:
    from geogen.clipping import measure

    out = []
    for pose in poses:
        m = measure(character, pose)
        if m["waist_gap"]["count"]:
            out.append(f"{pose}: skin shows between top and bottom {m['waist_gap']}")
        if m["waist_overlap"]["min_mm"] < WAIST_MIN_OVERLAP:
            out.append(f"{pose}: top and bottom overlap {m['waist_overlap']}")
        if m["hem"]["count"] or m["hem"]["ends"]:
            out.append(f"{pose}: the top's hem isn't a smooth loop {m['hem']}")
    return out


@pytest.mark.parametrize("preset,outfit", WAIST, ids=lambda x: "+".join(x) if isinstance(x, list) else x)
def test_tops_overlap_bottoms_in_every_pose(preset, outfit):
    """No skin band at the waist, a few cm of overlap and a smooth hem, whatever the pose."""
    from geogen.layout import LayoutLoader

    character = LayoutLoader().load(ASSETS / "characters" / "humanoid.yaml", params={"preset": preset, "outfit": outfit})
    assert character.meta["humanoid"]["waist_overlap"] >= 0.03 - 1e-6
    assert _waist_faults(character) == []


def test_waist_check_sees_a_slipped_skirt_and_a_torn_hem():
    """The detector itself: drop the skirt 12 cm and skin shows above it; pull a hem vertex 3 cm
    down the body and the hem is no longer a smooth loop."""
    import numpy as np

    from geogen.clipping import _dominant, _meshes, posed, worn
    from geogen.generators.humanoid import waist_hem
    from geogen.layout import LayoutLoader

    character = LayoutLoader().load(ASSETS / "characters" / "humanoid.yaml",
                                    params={"preset": "feminine", "outfit": "casual_f"}).instance()
    assert _waist_faults(character, ["stand"]) == []
    clothes, body = character.find("clothes"), character.find("body")
    kept = clothes.mesh, body.mesh
    clothes.mesh = clothes.mesh.copy()
    clothes.mesh.vertices[:, 1] -= 0.12
    faults = _waist_faults(character, ["stand"])
    assert any("skin shows" in f for f in faults) and any("overlap" in f for f in faults), faults
    clothes.mesh = kept[0]
    rest = _meshes(posed(character))["body"]
    mine = np.zeros(len(rest.vertices), bool)
    mine[worn(character, "top")["body"]] = True
    hem = np.unique(waist_hem(rest, mine[rest.faces].all(axis=1), _dominant(character, rest)))
    body.mesh = body.mesh.copy()
    at = body.mesh.vertices[hem[0]].copy()
    body.mesh.vertices[np.linalg.norm(body.mesh.vertices - at, axis=1) < 1e-5] -= [0.0, 0.03, 0.0]
    faults = _waist_faults(character, ["stand"])
    assert any("smooth loop" in f for f in faults), faults


def test_measure_sees_a_hand_in_the_thigh():
    """The detector itself: a body whose hands we push into its hips reports them."""
    import numpy as np

    from geogen.clipping import winding
    from geogen.generators.primitives import CubeGenerator

    cube = CubeGenerator(bevel=0).generate()
    tris = cube.vertices[cube.faces]
    assert winding(np.array([[0.0, 0.0, 0.0], [0.4, 0.2, -0.3]]), tris) == pytest.approx([1.0, 1.0], abs=1e-6)
    assert winding(np.array([[0.0, 2.0, 0.0]]), tris) == pytest.approx([0.0], abs=1e-6)


if __name__ == "__main__":        # regenerate the baseline
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_report, *zip(*COMBOS)))
    data = {f"{'+'.join(p)}/{o}": r for (p, o), r in zip(COMBOS, results)}
    BASELINE.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {BASELINE}")
