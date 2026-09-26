"""Character archetypes: seeded, declarative people (geogen/characters.py)."""

from pathlib import Path

import numpy as np
import pytest

from geogen.characters import draw, load_archetype, resolve_character
from geogen.layout.composer import SceneComposer
from geogen.layout.yaml_utils import safe_load

ASSETS = Path(__file__).parent.parent / "assets"
ARCHETYPES = ASSETS / "characters" / "archetypes"


@pytest.mark.parametrize("path", sorted(ARCHETYPES.glob("*.yaml")), ids=lambda p: p.stem)
def test_archetype_resolves_to_valid_humanoid_params(path):
    humanoid = safe_load((ASSETS / "characters" / "humanoid.yaml").read_text())
    declared = set(humanoid["params"]) | {"preset"}
    for seed in range(12):
        body, params = resolve_character(path, seed)
        assert body == "characters/humanoid.yaml"
        assert set(params) <= declared, set(params) - declared
        for key, value in params.items():
            choices = humanoid["params"].get(key, {}).get("choices")
            if choices:
                assert value in choices, (key, value)
        for p in params.get("preset", []):
            assert p in humanoid["presets"]


def test_deterministic_and_varied():
    a = [resolve_character(ARCHETYPES / "young_woman.yaml", s) for s in range(8)]
    b = [resolve_character(ARCHETYPES / "young_woman.yaml", s) for s in range(8)]
    assert a == b
    assert len({str(p) for _, p in a}) == 8
    # Different archetypes with one seed aren't the same draw.
    assert resolve_character(ARCHETYPES / "young_man.yaml", 0) != a[0]


def test_variants_keep_correlated_params_together():
    for seed in range(40):
        _, params = resolve_character(ARCHETYPES / "hotel_staff.yaml", seed)
        assert (params["top_color"], params["bottom_color"], params["outfit"]) == ("white", "black", "smart")
        if params["preset"][0] == "feminine":
            assert params["hair_style"] in {"bun", "ponytail", "pixie"}
        else:
            assert params["hair_style"] in {"crop", "buzz"}


def test_mix_archetype_delegates():
    from geogen.characters import _person

    names = set()
    for seed in range(60):
        person = _person(ARCHETYPES / "townsfolk.yaml", seed, None)
        assert person["body"] == "characters/humanoid.yaml" and "preset" in person["params"]
        names.add(person["names"][1])
    assert names == {"young_woman", "young_man", "office_worker", "older_woman", "older_man"}


def test_overrides_and_extends():
    _, params = resolve_character(ARCHETYPES / "young_man.yaml", 3, {"outfit": "smart"})
    assert params["outfit"] == "smart"
    arch = load_archetype(ARCHETYPES / "older_man.yaml")
    assert "skin_tone" in arch["params"]          # inherited from adult.yaml
    assert "character" in arch["tags"] and "character.older" in arch["tags"]


def test_draw_forms():
    rng = np.random.default_rng(0)
    assert draw(5, rng) == 5
    assert 1 <= draw({"random": [1, 2]}, rng) <= 2
    assert draw({"normal": [0, 10], "clamp": [-1, 1]}, rng) in (-1.0, 1.0) or -1 <= draw({"normal": [0, 1]}, rng)
    picks = [draw({"choice": ["a", "b"], "weights": [1, 0]}, rng) for _ in range(20)]
    assert set(picks) == {"a"}
    with pytest.raises(ValueError):
        draw({"choice": ["a", "b"], "weights": [1]}, rng)


def test_scene_places_archetype_and_npc_body(tmp_path):
    scene = tmp_path / "people.yaml"
    scene.write_text(
        "name: people\n"
        "place:\n"
        "  ann: { archetype: characters/archetypes/young_woman.yaml, seed: 3 }\n"
        "  ann_again: { archetype: characters/archetypes/young_woman.yaml, seed: 3 }\n"
        "  resident: { npc: npcs/resident.yaml, seed: 3 }\n"
    )
    root = SceneComposer(assets_dir=ASSETS).compose(scene)
    ann, again, resident = (root.find(n) for n in ("ann", "ann_again", "resident"))
    assert ann.meta["character"] == {"archetype": "young_woman", "seed": 3}
    assert "character.woman" in ann.tags
    # Same archetype and seed: one shared body prototype.
    shared = {id(n.mesh) for n in ann.iter_nodes() if n.mesh is not None}
    assert shared == {id(n.mesh) for n in again.iter_nodes() if n.mesh is not None}
    # The NPC's body is the same person (the placement seed picks it) and seeds the brain.
    npc = resident.meta["npc"]
    assert npc["seed"] == 3 and npc["body"]["character"]["archetype"] == "young_woman"
    body = resident.find("body")
    assert shared == {id(n.mesh) for n in body.iter_nodes() if n.mesh is not None}
