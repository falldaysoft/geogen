"""Clipping QA for dressed humanoids (geogen/clipping.py, geogen-z2b.16.15).

The baseline (tests/data/clipping_baseline.json, regenerate with
``python tests/test_clipping.py``) records today's penetration per preset x outfit x pose
and seat. The guard fails if anything gets worse; the strict test is the goal (nothing past
2 mm) and flips to passing as the clothing fixes land (z2b.16.16-.21).
"""

import json
from pathlib import Path

import pytest

ASSETS = Path(__file__).parent.parent / "assets"
BASELINE = Path(__file__).parent / "data" / "clipping_baseline.json"
PRESETS = [["feminine"], ["masculine"], ["feminine", "hourglass"], ["masculine", "heavy"], ["feminine", "petite"]]
OUTFITS = ["casual_f", "dress", "jeans_tee", "smart"]
COMBOS = [(p, o) for p in PRESETS for o in OUTFITS]


def _report(preset, outfit):
    from geogen.clipping import report

    return report([preset], [outfit])[f"{'+'.join(preset)}/{outfit}"]


def _worse(now: dict, then: dict, where: str) -> list[str]:
    out = []
    for key, value in now.items():
        base = then.get(key)
        if isinstance(value, dict) and "count" in value:
            if base is None:
                continue
            if value["max_mm"] > base["max_mm"] + 2.0 or value["count"] > base["count"] * 1.15 + 5:
                out.append(f"{where}/{key}: {value} (baseline {base})")
        elif isinstance(value, dict):
            out += _worse(value, base or {}, f"{where}/{key}")
    return out


@pytest.mark.parametrize("preset,outfit", COMBOS, ids=lambda x: "+".join(x) if isinstance(x, list) else x)
def test_clipping_no_worse_than_baseline(preset, outfit):
    key = f"{'+'.join(preset)}/{outfit}"
    baseline = json.loads(BASELINE.read_text())[key]
    assert _worse(_report(preset, outfit), baseline, key) == []


@pytest.mark.xfail(reason="seated clothing and seat fit: geogen-z2b.16.16-.21", strict=False)
@pytest.mark.parametrize("preset,outfit", [(["feminine"], "casual_f"), (["masculine"], "jeans_tee")],
                         ids=["feminine_casual_f", "masculine_jeans_tee"])
def test_no_clipping_beyond_tolerance(preset, outfit):
    def bad(d):
        return [k for k, v in d.items() if isinstance(v, dict) and (v.get("count", 0) > 0 if "count" in v else bad(v))]

    assert bad(_report(preset, outfit)) == []


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
