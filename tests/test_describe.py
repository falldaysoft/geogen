"""--describe: the vocabulary catalogue (text, JSON, docs/catalogue markdown)."""

import json
from pathlib import Path

import pytest

from geogen import roots

from geogen import describe as d
from geogen.layout import SceneComposer
from geogen.layout.schema import PRIMITIVES, part_known_keys
from geogen.layout.loader import _BUILDERS, PRIMITIVE_REGISTRY
from geogen.layout.yaml_utils import safe_load, safe_load_path

DOCS = Path(__file__).parent.parent / "docs" / "catalogue"
CATALOGUE = d.catalogue()
REGENERATE = "run docs/make_catalogue.sh"


def test_schema_covers_every_primitive_and_key_the_assets_use():
    assert set(PRIMITIVES) == set(PRIMITIVE_REGISTRY) == set(_BUILDERS)
    known = part_known_keys()
    for entry in CATALOGUE["assets"].values():
        data = safe_load_path(d.ASSETS_DIR / entry["file"])
        for part, spec in (data.get("parts") or {}).items():
            assert set(spec) <= known, f"{entry['file']} part {part}: {set(spec) - known}"


@pytest.mark.parametrize("name", sorted(PRIMITIVES))
def test_primitive_examples_build(name):
    node = d.primitive_node(name)
    assert sum(len(m.faces) for _, m in node.iter_meshes()) > 0
    used = set(safe_load(PRIMITIVES[name].example)) - set(d.PART_KEYS)
    assert used <= set(PRIMITIVES[name].keys), used - set(PRIMITIVES[name].keys)


def test_listing_names_everything():
    text = d.list_text(CATALOGUE)
    for name in PRIMITIVES:
        assert f"  {name}: " in text
    for entry in [*CATALOGUE["assets"].values(), *CATALOGUE["scenes"].values()]:
        assert f"  {entry['file']}: " in text
    for kind in ("npc", "archetype", "traffic", "train", "garment"):
        assert f"  {kind}: " in text
    assert "state " in text and "wood" in text
    # The core's part stays pasteable into a prompt; a project's assets add to it, not to this budget.
    core = {name for name, (_, path) in d.asset_files().items() if roots.root_of(path) == roots.CORE}
    only_core = {**CATALOGUE, **{k: {n: e for n, e in CATALOGUE[k].items() if n in core} for k in ("assets", "scenes")}}
    assert len(d.list_text(only_core)) < 48_000
    data = json.loads(d.describe(as_json=True))
    assert data["format"] == "geogen-describe" and set(data["assets"]) == set(CATALOGUE["assets"])


def test_door_and_car_can_be_placed_from_the_description_alone():
    """The acceptance: width, swing state and a car preset, without opening the YAML."""
    door = json.loads(d.describe(["door"], as_json=True))
    assert door["params"]["width"]["default"] == 0.95
    assert door["opening"] and door["interactions"]["swing"]["initial"] == "closed"
    assert set(door["interactions"]["swing"]["states"]) == {"closed", "open"}
    car = json.loads(d.describe(["vehicles/car.yaml"], as_json=True))
    assert {"sedan", "hatchback", "estate", "taxi"} <= set(car["presets"])
    assert "car_red" in car["params"]["paint"]["choices"] and car["vehicle"]["class"] == "car"
    text = d.describe(["door", "car"])
    assert "state: open" in text and "preset: hatchback" in text

    scene = SceneComposer().compose_string(f"""
name: garage
size: [14, 5, 14]
place:
  house: {{asset: house_peaked.yaml}}
  door: {{asset: {door['file']}, on: house.front_wall, at: {{u: 0.5, v: {{abs: 0}}}}, params: {{width: 1.1}}, state: open}}
  car: {{asset: {car['file']}, params: {{preset: estate, paint: car_red}}}}
""")
    placed = scene.find("door")
    swing = placed.interactions[0]
    assert swing.initial == "open" and swing.motions[0].applied == 100.0
    assert abs(placed.size[0] - 1.1) < 1e-9


def test_placement_state_rejects_unknown_states():
    for state in ("ajar", {"swing": "ajar"}, {"lid": "open"}):
        with pytest.raises(ValueError, match="state of door.yaml"):
            SceneComposer().compose_string(
                f"name: t\nsize: [2, 3, 2]\nplace:\n  door: {{asset: door.yaml, state: {json.dumps(state)}}}\n")


def test_describe_other_names():
    assert d.describe(["cube"]).startswith("cube (primitive")
    assert "npcs/resident.yaml" in d.describe(["npc"])
    assert "kind: npc" in d.describe(["npcs/resident"])
    assert "--inspect" in d.describe(["cottage"])
    with pytest.raises(KeyError, match="Did you mean"):
        d.describe(["dor"])


@pytest.mark.parametrize("page", ["README.md", "primitives.md", "vocabulary.md", "scenes.md"])
def test_committed_pages_are_current(page):
    assert (DOCS / page).read_text() == d.static_pages(CATALOGUE)[page], REGENERATE


@pytest.mark.parametrize("name", sorted(CATALOGUE["assets"]))
def test_committed_asset_page_is_current(name, built_scene):
    entry = CATALOGUE["assets"][name]
    node = built_scene(name).children[0]
    page = DOCS / d.asset_page_path(name)
    assert page.is_file(), REGENERATE
    assert page.read_text() == d.asset_page(d.describe_asset(entry, node)), REGENERATE


def test_no_stale_pages_or_missing_thumbnails():
    assert {p.stem for p in (DOCS / "assets").glob("*.md")} == set(CATALOGUE["assets"]), REGENERATE
    for name in CATALOGUE["assets"]:
        assert (DOCS / "img" / f"{name}.png").is_file() and (DOCS / "img" / f"{name}_views.png").is_file(), REGENERATE
    for name in PRIMITIVES:
        assert (DOCS / "img" / "primitives" / f"{name}.png").is_file(), REGENERATE
