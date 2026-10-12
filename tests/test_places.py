"""Named places (geogen/places.py): trigger volumes scenes and assets declare."""

import json
from pathlib import Path

import numpy as np
import pytest

from geogen import refs
from geogen.export import export_scene, manifest_path
from geogen.inspection import build_tree, format_tree
from geogen.layout.composer import SceneComposer
from geogen.layout.loader import LayoutLoader
from geogen.places import places_summary
from geogen.state import StateError, collect

ASSETS = Path(__file__).parent.parent / "assets"
SCHEMA = Path(__file__).parent.parent / "docs/schema/geogen-extras.v1.schema.json"
FAST = ("--fixed-fps", "60")

# A yard on the runtime's ground: a bench, a resident who wanders the west and sits
# on the bench in the east, and three places (one per shape).
YARD = """
name: yard
size: [16, 3, 16]
spawns:
  start: {position: [0, 0, 7], facing: south}
slots:
  seat: {position: [5, 0, -5], facing: west}
  stand: {position: [-3, 0, -2], facing: north}
place:
  bench: {asset: bench.yaml, slot: seat}
  walker:
    npc: npcs/resident.yaml
    slot: stand
    home: {rect: [-6, -6, 6, 2], y: 0}
    seed: 3
places:
  east_half: {polygon: [[0, -7], [7, -7], [7, 3], [0, 3]], height: 2.2, tags: [outside]}
  strip: {box: [16, 2.2, 1.5], position: [0, 0, 4]}
  corner: {box: [2, 2, 2], anchor: bottom_front_left, offset: [1, 0, -1], yaw: 30}
  by_bench: {around: bench, margin: 1}
"""

PLATE = """
name: plate
origin: bottom_center
size: [1, 1, 1]
bounds: geometry
parts:
  slab: {primitive: cube, size: [1.2, 0.05, 1.2], anchor: bottom_center, bevel: 0}
places:
  top: {around: slab, headroom: 2}
signals:
  occupied: {enter: top}
  vacated:  {leave: top}
"""


def compose(text: str = YARD):
    return SceneComposer(ASSETS).compose_string(text)


def test_scene_places_become_nodes_with_stable_ids():
    root = compose()
    refs.assign_ids(root, "yard")
    assert [n.meta["uid"] for n in refs.select(root, "[type=place]")] == [
        "yard:east_half", "yard:strip", "yard:corner", "yard:by_bench"]
    strip = refs.resolve(root, "strip")
    assert strip.tags == ["place"] and strip.meta["place"] == {"id": "strip", "size": [16.0, 2.2, 1.5]}
    assert refs.resolve(root, "[tag=outside]").name == "east_half"
    half = refs.resolve(root, "east_half").meta["place"]
    assert len(half["polygon"]) == 4 and half["height"] == 2.2


def test_summary_is_in_world_space():
    root = compose()
    places = {p["id"]: p for p in places_summary(root)}
    assert places["strip"]["center"] == [0, 1.1, 4] and places["strip"]["shape"] == "box"
    corner = places["corner"]                      # the box's bottom is on the anchor + offset
    assert corner["center"] == [-7, 1, 7] and corner["yaw_deg"] == pytest.approx(30)
    assert places["east_half"]["shape"] == "polygon" and places["east_half"]["tags"] == ["outside"]
    assert sorted(map(tuple, places["east_half"]["polygon"])) == [(0, -7), (0, 3), (7, -7), (7, 3)]
    bench = places["by_bench"]                     # the bench's box (at 5, -5) plus a metre all round
    assert bench["center"][0] == pytest.approx(5, abs=0.1) and bench["center"][2] == pytest.approx(-5, abs=0.1)
    assert min(bench["size"]) > 2


def test_a_moved_scene_takes_its_places_with_it():
    outer = SceneComposer(ASSETS)
    inner = compose()
    inner.transform.translation = np.array([100.0, 2.0, 0.0])
    inner.transform.rotation = np.array([0.0, np.pi / 2, 0.0])
    parent = outer.compose_string("name: world\nsize: [1, 1, 1]\nplace: {}\n")
    parent.add_child(inner)
    places = {p["id"]: p for p in places_summary(parent)}
    assert places["strip"]["center"] == pytest.approx([104, 3.1, 0])
    corners = sorted(tuple(round(v) for v in p) for p in places["east_half"]["polygon"])
    assert places["east_half"]["y"] == 2 and corners == [(93, -7), (93, 0), (103, -7), (103, 0)]


def test_asset_places_sit_round_parts_and_feed_signals():
    plate = LayoutLoader().load_string(PLATE)
    top = refs.resolve(plate, "top")
    assert top.meta["place"]["size"] == pytest.approx([1.2, 2.05, 1.2])
    assert top.transform.translation == pytest.approx([0, 0, 0])
    plate.meta["source"] = {"asset": "plate.yaml", "params": {}}
    from geogen.state import declare_asset
    from geogen.layout.yaml_utils import safe_load

    declare_asset(plate, safe_load(PLATE), "plate.yaml")
    root = compose("name: hall\nsize: [4, 3, 4]\nplace: {}\n")
    plate.name = "plate"
    root.add_child(plate)
    signals = collect(root, "hall").signals
    assert signals["hall:plate#occupied"]["from"] == {"type": "enter", "subject": "hall:plate/top"}
    assert signals["hall:plate#vacated"]["from"]["type"] == "leave"


def test_signals_only_enter_places_and_rooms():
    text = PLATE.replace("enter: top", "enter: slab").replace("slab: {", "slab: {tags: [decor.slab], ")
    plate = LayoutLoader().load_string(text)
    from geogen.state import declare_asset
    from geogen.layout.yaml_utils import safe_load

    plate.meta["source"] = {"asset": "plate.yaml", "params": {}}
    declare_asset(plate, safe_load(text), "plate.yaml")
    with pytest.raises(StateError, match="'slab' is not a place or a room. Places here: top"):
        collect(plate, "hall")


@pytest.mark.parametrize("places, message", [
    ("{a: {box: [1, 1, 1], polygon: [[0, 0], [1, 0], [1, 1]]}}", "needs exactly one of box, polygon, around"),
    ("{a: {position: [0, 0, 0]}}", "needs exactly one of"),
    ("{a: {box: [1, 1]}}", "box must be 3 numbers"),
    ("{a: {box: [1, 0, 1]}}", "must have a size"),
    ("{a: {polygon: [[0, 0], [1, 0]]}}", "at least 3 points"),
    ("{a: {polygon: [[0, 0], [1, 0], [2, 0]]}}", "no area"),
    ("{a: {box: [1, 1, 1], radius: 2}}", "unknown keys \\['radius'\\]"),
    ("{bench: {box: [1, 1, 1]}}", "already called 'bench'"),
    ("{a: {around: banch}}", "places.a.around.*bench"),
])
def test_bad_places_say_what_is_wrong(places, message):
    with pytest.raises(ValueError, match=message):
        compose(f"name: yard\nsize: [8, 3, 8]\nplace:\n  bench: {{asset: bench.yaml}}\nplaces: {places}\n")


def test_asset_places_have_no_scene_anchor():
    with pytest.raises(ValueError, match="plate: place 'top': unknown keys \\['anchor'\\]"):
        LayoutLoader().load_string(PLATE.replace("around: slab", "box: [1, 1, 1], anchor: center"))
    with pytest.raises(ValueError, match="place 'top': around: .*slab"):
        LayoutLoader().load_string(PLATE.replace("around: slab", "around: slob"))


def test_export_writes_extras_and_manifest(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    from test_export import _gltf_json

    model = export_scene(compose(), tmp_path / "yard.glb")
    schema = json.loads(SCHEMA.read_text())
    extras = {n["name"]: n["extras"] for n in _gltf_json(model)["nodes"] if "extras" in n}
    for e in extras.values():
        jsonschema.validate(e, schema)
    strip = extras["strip"]["geogen"]
    assert strip["type"] == "place" and strip["uid"] == "yard:strip" and strip["place"]["size"] == [16.0, 2.2, 1.5]
    manifest = json.loads(manifest_path(model).read_text())
    assert [(p["id"], p["uid"], p["shape"]) for p in manifest["places"]] == [
        ("east_half", "yard:east_half", "polygon"), ("strip", "yard:strip", "box"),
        ("corner", "yard:corner", "box"), ("by_bench", "yard:by_bench", "box")]
    assert refs.resolve(refs.load_index(manifest_path(model)), "strip") is not None    # other packages can name it


def test_inspect_lists_places():
    text = "\n".join(format_tree(build_tree(compose())))
    assert "place box 16×2.2×1.5 m" in text and "place polygon ×4, 2.2 m high" in text


def test_the_island_names_its_gallery_and_jetty(built_scene):
    island = built_scene("island")
    places = {p["id"]: p for p in places_summary(island)}
    gallery, deck = places["gallery"], places["on_deck"]
    assert gallery["center"][1] > 14 and gallery["size"][1] == 2.4         # up at the lighthouse's deck
    assert deck["size"] == pytest.approx([2.4, 2.1, 18]) and deck["center"][2] < -47
    assert refs.resolve(island, "lighthouse/gallery").meta["type"] == "place"


# -- in the Godot runtime ------------------------------------------------------

def _events(out: str) -> list[dict]:
    return [json.loads(l.removeprefix("place event: ")) for l in out.splitlines() if l.startswith("place event: ")]


@pytest.fixture(scope="module")
def yard_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("generated_places")
    export_scene(compose(), out / "yard.glb")
    return out


def test_player_walking_through_places_raises_enter_and_leave(run_godot, yard_dir):
    # South from z = 7 at x = 3: across the strip (a box), then through the east half (a polygon).
    out = run_godot("--scene=yard", f"--generated={yard_dir}", "--spawn=3,0.1,7", "--yaw=0", "--walk=4",
                    engine_args=FAST)
    events = [(e["type"], e["subject"]) for e in _events(out) if e["actor"] == "player"]
    assert events == [("enter", "yard:strip"), ("enter", "yard:east_half"),
                      ("leave", "yard:strip"), ("leave", "yard:east_half")]
    assert all(e["kind"] == "place" for e in _events(out))


def test_npcs_raise_them_too_and_stay_in_a_place_while_seated(run_godot, yard_dir):
    out = run_godot("--scene=yard", f"--generated={yard_dir}", "--timescale=8", "--simulate=60", engine_args=FAST)
    events = [e for e in _events(out) if e["actor"] == "yard:walker" and e["subject"] == "yard:east_half"]
    assert events and events[0]["type"] == "enter"
    assert [e["type"] for e in events] == ["enter", "leave"] * (len(events) // 2) + ["enter"] * (len(events) % 2)
    (report,) = json.loads(next(l for l in out.splitlines() if l.startswith("npc summary: "))
                           .removeprefix("npc summary: "))
    # A minute in she is sitting on the bench: her collision shape is off, but she hasn't left.
    assert report["pose"] == "sit" and events[-1]["type"] == "enter"
    assert not [e for e in _events(out) if e["actor"] == "player"]          # the player stood outside them all


def test_rooms_are_places_for_events(run_godot, built_scene, tmp_path):
    export_scene(built_scene("hotel_suite"), tmp_path / "hotel_suite.glb")
    manifest = json.loads((tmp_path / "hotel_suite.manifest.json").read_text())
    room = manifest["rooms"][0]
    spawn = ",".join(str(v) for v in (room["center"][0], room["center"][1] - room["size"][1] / 2 + 0.1,
                                      room["center"][2]))
    out = run_godot("--scene=hotel_suite", f"--generated={tmp_path}", f"--spawn={spawn}", "--walk=0.5",
                    engine_args=FAST)
    assert {"type": "enter", "kind": "room", "place": room["id"], "actor": "player",
            "subject": room["uid"].rsplit("/", 1)[0]} in _events(out)      # the room, not its volume
