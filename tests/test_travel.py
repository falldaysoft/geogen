"""Travel points (geogen/travel.py): parsing, export, catalogue validation, and the Godot runtime."""

import json

import pytest

from geogen.catalogue import Catalogue, CatalogueEntry, export_catalogue, write_index
from geogen.export import export_scene, gameplay_summary, node_extras
from geogen.layout.composer import SceneComposer
from geogen.travel import apply_travel, check_travel, parse_travel

# A freestanding arch: two posts, a lintel, and a walk-in trigger between them.
PORTAL = """
name: portal_test
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
travel: {on: enter, prompt: Step through}
parts:
  left_post:  {primitive: cube, size: [0.2, 2.4, 0.3], anchor: bottom_center, offset: [-0.7, 0, 0]}
  right_post: {primitive: cube, size: [0.2, 2.4, 0.3], anchor: bottom_center, offset: [0.7, 0, 0]}
  lintel:     {primitive: cube, size: [1.6, 0.2, 0.3], anchor: bottom_center, offset: [0, 2.4, 0]}
  travel_volume: {primitive: cube, size: [1.2, 2.2, 0.3], anchor: bottom_center}
"""


def scene_a(portal: str) -> str:
    return f"""
name: travel_a
size: [12, 3, 12]
spawns:
  start: {{position: [0, 0, 1.1], facing: south}}
  side:  {{position: [4, 0, 2.5], facing: south}}
slots:
  shelf_spot:  {{position: [0, 0, 0]}}
  portal_spot: {{position: [4, 0, 0]}}
place:
  shelf:
    asset: bookshelf.yaml
    slot: shelf_spot
    travel: {{scene: travel_b, spawn: arrival, prompt: Go to B}}
  portal:
    asset: {portal}
    slot: portal_spot
    travel: {{scene: travel_b, spawn: arrival}}
"""


SCENE_B = """
name: travel_b
size: [12, 3, 12]
spawns:
  arrival: {position: [-3, 0, 4], facing: south}
  far:     {position: [5, 0, -4], facing: north}
slots:
  box_spot: {position: [0, 0, 0]}
place:
  mailbox:
    asset: mailbox.yaml
    slot: box_spot
    travel: {spawn: far}
"""


@pytest.fixture(scope="module")
def portal_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("assets") / "portal_test.yaml"
    path.write_text(PORTAL)
    return path


@pytest.fixture(scope="module")
def scenes(portal_path):
    composer = SceneComposer()
    return {"travel_a": composer.compose_string(scene_a(str(portal_path))),
            "travel_b": composer.compose_string(SCENE_B)}


def test_parse_travel_rejects_bad_blocks():
    assert parse_travel({"scene": "hotel", "on": "enter"}) == {"scene": "hotel", "on": "enter"}
    with pytest.raises(ValueError, match="unknown travel keys"):
        parse_travel({"scen": "hotel"})
    with pytest.raises(ValueError, match="on must be"):
        parse_travel({"on": "touch"})


def test_use_travel_becomes_an_interaction(scenes):
    shelf = scenes["travel_a"].find("shelf")
    assert shelf.meta["travel"] == {"on": "use", "prompt": "Go to B", "scene": "travel_b", "spawn": "arrival"}
    travel = next(i for i in shelf.interactions if i.name == "travel")
    assert travel.targets == [shelf]
    assert travel.states["ready"].next == "going" and travel.states["going"].emit == "travel"
    extras = node_extras(shelf)["geogen"]
    assert extras["travel"]["scene"] == "travel_b" and "travel_spec" not in extras


def test_enter_travel_takes_its_volume_part_out_of_the_geometry(scenes):
    portal = scenes["travel_a"].find("portal")
    travel = portal.meta["travel"]
    # The asset's own prompt survives the placement override, which adds the target.
    assert travel["on"] == "enter" and travel["prompt"] == "Step through" and travel["scene"] == "travel_b"
    assert travel["volume"]["size"] == pytest.approx([1.2, 2.2, 0.3])
    assert travel["volume"]["center"] == pytest.approx([0, 1.1, 0])
    assert portal.find("travel_volume") is None
    assert not any(i.name == "travel" for i in portal.interactions)


def test_same_scene_travel_and_default_prompt(scenes):
    mailbox = scenes["travel_b"].find("mailbox")
    assert mailbox.meta["travel"] == {"on": "use", "prompt": "Go", "spawn": "far"}


def test_overriding_travel_replaces_the_interaction(scenes):
    shelf = scenes["travel_a"].find("shelf").instance()
    apply_travel(shelf, {"on": "enter"})
    assert shelf.meta["travel"]["on"] == "enter" and "volume" in shelf.meta["travel"]
    assert not any(i.name == "travel" for i in shelf.interactions)
    assert shelf.meta["travel"]["scene"] == "travel_b"     # earlier keys are kept


def test_manifest_lists_travel_points(scenes):
    summary = gameplay_summary(scenes["travel_a"])
    by_node = {t["node"]: t for t in summary["travel"]}
    assert by_node["shelf"] == {"node": "shelf", "scene": "travel_b", "spawn": "arrival", "on": "use"}
    assert by_node["portal"]["on"] == "enter"


def _catalogue(*names):
    return Catalogue(names[0], {n: CatalogueEntry(n, "showcase") for n in names})


def _manifest(out, name, spawns, travel):
    (out / f"{name}.manifest.json").write_text(json.dumps(
        {"spawns": [{"name": s} for s in spawns], "travel": travel}))


def test_check_travel_finds_unknown_scenes_and_spawns(tmp_path):
    _manifest(tmp_path, "a", ["start"], [{"node": "door", "scene": "b", "spawn": "arrival"},
                                         {"node": "gate", "scene": "nowhere"},
                                         {"node": "stairs", "spawn": "start"},
                                         {"node": "hatch", "spawn": "roof"}])
    _manifest(tmp_path, "b", ["lobby"], [])
    problems = check_travel(_catalogue("a", "b"), tmp_path)
    assert len(problems) == 3
    assert any("'nowhere'" in p and "catalogue" in p for p in problems)
    assert any("'arrival'" in p and "'b'" in p for p in problems)
    assert any("'roof'" in p and "'a'" in p for p in problems)      # same-scene target
    # A target that isn't exported yet can only be checked for being in the catalogue.
    (tmp_path / "b.manifest.json").unlink()
    assert len(check_travel(_catalogue("a", "b"), tmp_path)) == 2


def test_export_catalogue_fails_on_a_bad_target(scenes, tmp_path):
    with pytest.raises(ValueError, match="travel_b"):
        export_catalogue(_catalogue("travel_a"), lambda name: scenes[name].instance(), tmp_path,
                         log=lambda *_: None)


@pytest.fixture(scope="module")
def travel_dir(tmp_path_factory, scenes):
    out = tmp_path_factory.mktemp("generated")
    export_catalogue(_catalogue("travel_a", "travel_b"), lambda name: scenes[name].instance(), out,
                     log=lambda *_: None)
    return out


def _json_lines(out: str, prefix: str) -> list[dict]:
    return [json.loads(line.removeprefix(prefix)) for line in out.splitlines() if line.startswith(prefix)]


def test_use_a_travel_point_to_reach_another_scene(run_godot, travel_dir):
    out = run_godot("--scene=travel_a", f"--generated={travel_dir}", "--use=@aim", "--simulate=2", "--status")
    assert "used: shelf (travel)" in out
    travelled = _json_lines(out, "travelled: ")
    assert len(travelled) == 1 and travelled[0]["ok"] and travelled[0]["via"] == "use"
    status = _json_lines(out, "status: ")[0]
    assert status["scene"] == "travel_b"
    assert (status["x"], status["z"]) == pytest.approx((-3, 4), abs=0.05)     # the 'arrival' spawn


def test_walk_into_a_portal_to_reach_another_scene(run_godot, travel_dir):
    out = run_godot("--scene=travel_a", f"--generated={travel_dir}", "--spawn=4,0,2.5", "--walk=3")
    travelled = _json_lines(out, "travelled: ")
    assert len(travelled) == 1 and travelled[0]["via"] == "enter" and travelled[0]["scene"] == "travel_b"
    result = _json_lines(out, "walk result: ")[0]
    assert result["scene"] == "travel_b" and result["on_floor"]
    assert (result["x"], result["z"]) == pytest.approx((-3, 4), abs=0.05)   # arrived and stopped


def test_same_scene_travel_moves_the_player(run_godot, travel_dir):
    out = run_godot("--scene=travel_b", f"--generated={travel_dir}", "--use=mailbox", "--simulate=2", "--status")
    travelled = _json_lines(out, "travelled: ")[0]
    assert travelled["scene"] == "travel_b" and travelled["spawn"] == "far"
    status = _json_lines(out, "status: ")[0]
    assert (status["x"], status["z"]) == pytest.approx((5, -4), abs=0.05)


GATES = """
name: gates
size: [12, 4, 12]
spawns:
  at_door:   {position: [-2, 0, 1.2], facing: south}
  at_portal: {position: [3, 0, 3], facing: south}
slots:
  door_spot:   {position: [-2, 0, 0]}
  portal_spot: {position: [3, 0, 0]}
place:
  door:
    asset: travel_door.yaml
    slot: door_spot
    params: { label: Travel B }
    travel: { scene: travel_b, spawn: arrival }
  arch:
    asset: portal.yaml
    slot: portal_spot
    params: { label: Travel B }
    travel: { scene: travel_b, spawn: far }
"""


@pytest.fixture(scope="module")
def gates_dir(tmp_path_factory, scenes):
    out = tmp_path_factory.mktemp("generated")
    built = {**scenes, "gates": SceneComposer().compose_string(GATES)}
    export_catalogue(_catalogue("gates", "travel_a", "travel_b"), lambda name: built[name].instance(), out,
                     log=lambda *_: None)
    return out


def test_travel_door_and_portal_assets(scenes):
    gates = SceneComposer().compose_string(GATES)
    door, arch = gates.find("door"), gates.find("arch")
    assert door.meta["travel"]["on"] == "use" and door.meta["travel"]["target"] == "leaf"
    assert next(i for i in door.interactions if i.name == "travel").targets == [door.find("leaf")]
    assert arch.meta["travel"]["on"] == "enter" and arch.meta["travel"]["prompt"] == "Step through"
    assert arch.find("travel_volume") is None and arch.meta["travel"]["volume"]["size"][1] == pytest.approx(2.2)


def test_use_the_travel_door_in_godot(run_godot, gates_dir):
    out = run_godot("--scene=gates", f"--generated={gates_dir}", "--spawn=-2,0,1.2", "--use=@aim",
                    "--simulate=2", "--status")
    assert "used: door (travel)" in out
    status = _json_lines(out, "status: ")[0]
    assert status["scene"] == "travel_b" and (status["x"], status["z"]) == pytest.approx((-3, 4), abs=0.05)


def test_walk_through_the_portal_asset_in_godot(run_godot, gates_dir):
    out = run_godot("--scene=gates", f"--generated={gates_dir}", "--spawn=3,0,3", "--walk=3")
    travelled = _json_lines(out, "travelled: ")
    assert len(travelled) == 1 and travelled[0]["via"] == "enter" and travelled[0]["spawn"] == "far"
    result = _json_lines(out, "walk result: ")[0]
    assert result["scene"] == "travel_b" and (result["x"], result["z"]) == pytest.approx((5, -4), abs=0.05)
