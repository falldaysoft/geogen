"""NPC data: definitions, actions, affordance/portal resolution and scene placement."""

import numpy as np
import pytest

from geogen.layout import LayoutLoader, SceneComposer
from geogen.npc import ASSETS_DIR, approach_point, load_actions, load_definition, parse_actions, parse_poses

COTTAGE_NPC = """
name: t
place:
  house: { asset: house_peaked.yaml }
  door: { asset: door.yaml, on: house.front_wall, at: { u: 0.5, v: { abs: 0 } } }
  resident:
    npc: npcs/resident.yaml
    on: house.floor
    at: { u: 0.6, v: 0.6 }
    home: house.floor
    seed: 7
"""


def _npc(root):
    return next(n for n in root.iter_nodes() if n.meta.get("type") == "npc")


def test_actions_file_parses():
    actions = load_actions()
    assert {"sit", "lie", "look", "stand", "use", "pass", "wander", "idle"} <= set(actions)
    assert actions["pass"]["steps"][0] == {"go_to": "near"}
    assert actions["sit"]["duration"] == [15.0, 40.0]


@pytest.mark.parametrize("step, message", [
    ({"go_to": "moon"}, "go_to must be"),
    ({"wait": 2, "face": "anchor"}, "exactly one"),
    ({"dance": 1}, "exactly one"),
    ({"pose": "sit", "at": "ceiling"}, "at must be"),
    ({"wait": [5, 1]}, "lo, hi"),
])
def test_bad_steps_are_rejected(step, message):
    with pytest.raises(ValueError, match=message):
        parse_actions({"kind": "npc_actions", "version": 1, "actions": {"x": {"steps": [step]}}})


def test_definition_resolves():
    d = load_definition(ASSETS_DIR / "npcs" / "resident.yaml")
    assert d["definition"] == "resident"
    assert set(d["needs"]) == {"rest", "curiosity", "fresh_air"}
    assert d["activities"]["wander"]["action"] == "wander"
    assert d["scoring"]["retry"] == 30.0
    assert d["flags"]["closes_doors"] is True


def test_definition_rejects_unknown_need(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("kind: npc\nversion: 1\nneeds: {rest: {}}\n"
                    "activities: {idle: {action: idle, advertises: {hunger: 1}}}\n")
    with pytest.raises(ValueError, match="unknown needs"):
        load_definition(path)


def test_poses_need_stand():
    with pytest.raises(ValueError, match="stand"):
        parse_poses({"sit": {"scale": [1, 0.5, 1]}})
    assert parse_poses({"stand": {}})["stand"]["scale"] == [1.0, 1.0, 1.0]


def test_approach_points_follow_the_anchor_heading():
    # sit: 0.5 m in front of the seat; a seat facing +X puts it at +X.
    assert approach_point("sit", np.array([1.0, 0.45, 0.0]), 90.0) == pytest.approx([1.5, 0.0, 0.0])
    # An explicit [x, y, z] is in the anchor frame (x right of the facing).
    assert approach_point("sit", np.zeros(3), 0.0, [0.6, 0, 0.15]) == pytest.approx([0.6, 0.0, 0.15])
    assert approach_point("look", np.zeros(3), 180.0, 0.4) == pytest.approx([0.0, 0.0, -0.4])


def test_affordances_carry_npc_fields():
    chair = LayoutLoader().load(ASSETS_DIR / "chair.yaml")
    (a,) = chair.meta["affordances"]
    assert a["action"] == "sit_at_table" and a["tags"] == ["seat", "table"] and a["slots"] == 1
    alone = LayoutLoader().load(ASSETS_DIR / "chair.yaml", params={"at_table": 0})
    assert alone.meta["affordances"][0]["action"] == "sit"                 # hands in the lap
    assert a["approach"] == pytest.approx([0.6, 0.0, 0.18])
    window = LayoutLoader().load(ASSETS_DIR / "window.yaml")
    assert window.meta["affordances"][0]["action"] == "look"


def test_door_portal_references_its_interaction():
    door = LayoutLoader().load(ASSETS_DIR / "door.yaml")
    portal = door.meta["portal"]
    assert portal["interaction"] == "swing" and portal["open"] == "open"
    assert portal["normal"] == [0.0, 0.0, 1.0]
    assert portal["width"] == pytest.approx(0.95 - 2 * 0.07)


def test_npc_placement_builds_body_and_home():
    root = SceneComposer().compose_string(COTTAGE_NPC)
    npc = _npc(root)
    data = npc.meta["npc"]
    assert npc.name == "resident" and npc.tags == ["npc"]
    assert [c.name for c in npc.children] == ["body"]
    assert data["seed"] == 7
    # The resident is the feminine humanoid preset (1.66 m); her collider is as wide as she is deep.
    assert 1.66 <= data["height"] < 1.71                   # 1.66 m plus her hair
    assert 0.15 < data["radius"] < 0.25
    assert set(data["body"]["poses"]) == {"stand", "sit", "lie"}
    assert "pass" in data["actions"]
    # The home polygon (house floor, 7.4 x 5.4 m) is in the NPC's frame: it contains the NPC.
    poly = np.array(data["home"]["polygon"])
    assert poly.shape == (4, 2)
    assert poly[:, 0].min() < 0 < poly[:, 0].max() and poly[:, 1].min() < 0 < poly[:, 1].max()
    assert np.ptp(poly[:, 0]) * np.ptp(poly[:, 1]) == pytest.approx(7.4 * 5.4, rel=0.01)
    # Body parts don't collide (the runtime gives the NPC its own capsule).
    assert all(n.meta.get("collider") == "none" for n in npc.iter_nodes() if n.mesh is not None)


def test_home_must_be_a_horizontal_surface():
    with pytest.raises(ValueError, match="horizontal"):
        SceneComposer().compose_string(COTTAGE_NPC.replace("home: house.floor", "home: house.front_wall"))


def test_npc_exports_extras(tmp_path):
    import json
    import struct

    from geogen.export import export_scene

    path = tmp_path / "t.glb"
    export_scene(SceneComposer().compose_string(COTTAGE_NPC), path)
    data = path.read_bytes()
    gltf = json.loads(data[20:20 + struct.unpack("<I", data[12:16])[0]])
    npc = next(n for n in gltf["nodes"] if n.get("extras", {}).get("geogen", {}).get("type") == "npc")
    data = npc["extras"]["geogen"]["npc"]
    assert data["definition"] == "resident"
    names = {gltf["nodes"][i]["name"] for i in npc["children"]}
    assert "body" in names


def test_routine_blocks_parse(tmp_path):
    from geogen.npc import load_definition, parse_time

    assert parse_time("22:30", "t") == 22.5 and parse_time("07:00", "t") == 7.0
    d = load_definition(ASSETS_DIR / "npcs" / "pedestrian.yaml")
    night = next(b for b in d["routine"] if b["name"] == "night")
    assert night == {"name": "night", "from": 22.5, "to": 7.0, "away": True, "activities": {}, "preferences": {}}
    bad = tmp_path / "bad.yaml"
    bad.write_text("kind: npc\nversion: 1\nactivities: { idle: { action: idle } }\n"
                   "routine: [{ from: '09:00', to: '10:00', activities: { dance: 2 } }]\n")
    with pytest.raises(ValueError, match="unknown activities"):
        load_definition(bad)


def test_attention_defaults_and_overrides(tmp_path):
    from geogen.npc import ATTENTION_DEFAULTS

    d = load_definition(ASSETS_DIR / "npcs" / "resident.yaml")
    assert d["attention"] == ATTENTION_DEFAULTS           # on by default, every part
    greet = d["attention"]["greet"]                       # the signal is only debounced; the wave has the cooldown
    assert 0 < greet["debounce"] < greet["acknowledge"] < greet["cooldown"]
    actions = load_actions()
    assert [next(iter(s)) for s in actions["greet"]["steps"]] == ["face", "play", "wait"]
    assert actions["step_aside"]["steps"][0] == {"go_to": "aside"}
    path = tmp_path / "shy.yaml"
    path.write_text("kind: npc\nversion: 1\nactivities: { idle: { action: idle } }\n"
                    "attention: { look: { range: 2 }, greet: false }\n")
    shy = load_definition(path)["attention"]
    assert shy["look"]["range"] == 2.0 and shy["look"]["yaw"] == 70.0
    assert "greet" not in shy and "yield" in shy
    path.write_text("kind: npc\nversion: 1\nactivities: { idle: { action: idle } }\nattention: false\n")
    assert load_definition(path)["attention"] == {}
    path.write_text("kind: npc\nversion: 1\nactivities: { idle: { action: idle } }\n"
                    "attention: { greet: { action: dance } }\n")
    with pytest.raises(ValueError, match="unknown action"):
        load_definition(path)


def test_wave_clip_raises_the_right_hand():
    from geogen.core.skin import pose_clips

    body = LayoutLoader().load(ASSETS_DIR / "characters" / "humanoid.yaml")
    wave = next(c for c in body.clips if c.name == "wave")
    assert not wave.loop and wave.duration == pytest.approx(2.4)
    rest = body.instance()
    pose_clips(rest, "wave", 0.0)
    start = rest.find("RightHand").world_transform()[1, 3]
    mid = body.instance()
    pose_clips(mid, "wave", 1.2)
    hand = mid.find("RightHand").world_transform()[1, 3]
    head = mid.find("Head").world_transform()[1, 3]
    assert hand > head - 0.1 > start                      # up by the head, from the hip


def test_routine_blocks_can_keep_an_npc_at_work():
    from geogen.npc import parse_routine
    (block,) = parse_routine([{"from": "09:00", "to": "17:00", "at": "work"}], {}, "routine")
    assert block["at"] == "work"
    with pytest.raises(ValueError, match="at is one of home, work"):
        parse_routine([{"from": "09:00", "to": "17:00", "at": "school"}], {}, "routine")


def test_a_night_block_keeps_an_npc_to_its_bed():
    from geogen.npc import parse_routine
    (block,) = parse_routine([{"from": "23:00", "to": "07:00", "only": ["bed", "owns"], "hold": True}], {}, "r")
    assert block["only"] == ["bed", "owns"] and block["hold"] is True
    (plain,) = parse_routine([{"from": "09:00", "to": "17:00"}], {}, "r")
    assert "only" not in plain and "hold" not in plain
    with pytest.raises(ValueError, match="only is a list"):
        parse_routine([{"from": "23:00", "to": "07:00", "only": "bed"}], {}, "r")


def test_ownership_is_exported_as_stable_ids(tmp_path):
    from geogen import refs
    from geogen.export import exported_extras
    from geogen.layout.composer import SceneComposer

    path = tmp_path / "two.yaml"
    path.write_text("""
name: two
size: [14, 6, 14]
place:
  cottage: { scene: scenes/cottage.yaml, params: { quest: 0 } }
  guest:
    npc: npcs/resident.yaml
    on: cottage/house.floor
    home: cottage/house.floor
    owns: [cottage/armchair]
    prefers: ["cottage/dining/[tag=furniture.chair]"]
    work: cottage/dining
""")
    root = SceneComposer(assets_dir=ASSETS_DIR).compose(path)
    refs.assign_ids(root, "two")
    guest = next(n for n in root.iter_nodes() if n.name == "guest")
    npc = exported_extras(guest)["geogen"]["npc"]
    assert npc["owns"] == ["two:cottage/armchair"] and npc["work"] == "two:cottage/dining"
    assert len(npc["prefers"]) == 4 and all(p.startswith("two:cottage/dining/chairs_") for p in npc["prefers"])
    path.write_text(path.read_text().replace("cottage/armchair]", "cottage/armchiar]"))
    with pytest.raises(refs.RefError, match="place.guest.owns.0: .*Did you mean 'armchair'"):
        SceneComposer(assets_dir=ASSETS_DIR).compose(path)



class _Refs:
    """Stand-in SceneRefs: a root with `n` rooms, each with a floor surface."""

    def __init__(self, n):
        from geogen.core.node import SceneNode

        self.root = SceneNode("scene")
        self.rooms = [SceneNode(f"room_{i}") for i in range(n)]
        for room in self.rooms:
            room.surfaces["floor"] = object()
            self.root.add_child(room)

    def nodes(self, text, keys):
        return self.rooms if text == "[tag=room]" else []

    def surface(self, text, keys):
        if not text.endswith(".floor"):
            raise ValueError(f"no surface in {text}")
        return None, None

    def fail(self, message, keys, ref=""):
        return ValueError(message)


def test_population_puts_people_in_every_place_it_names():
    from geogen.npc import population_placements

    spec = {"guests": {"npc": "npcs/hotel_guest.yaml", "count": [0, 2], "on": "[tag=room].floor"}}
    first = population_placements(spec, _Refs(12), "hotel", ["population"])
    assert first == population_placements(spec, _Refs(12), "hotel", ["population"])     # deterministic
    counts = {name: p["scatter"]["count"] for name, p in first.items()}
    assert set(counts.values()) <= {1, 2} and 3 <= len(counts) <= 12       # rooms with nobody have no entry
    for name, placement in first.items():
        room = name.removeprefix("guests_")
        assert placement["home"] == placement["scatter"]["on"] == f"room_{room}.floor"
        assert placement["npc"] == "npcs/hotel_guest.yaml"
    other = population_placements({**spec, "guests": {**spec["guests"], "seed": 7}}, _Refs(12), "hotel", ["p"])
    assert other != first
    one = population_placements({"guests": {"npc": "x", "count": 3, "on": "[tag=room].floor"}}, _Refs(1), "s", ["p"])
    assert list(one) == ["guests"] and one["guests"]["scatter"]["count"] == 3
    for bad, message in [({"npc": "x", "count": [2, 1], "on": "[tag=room].floor"}, "count"),
                         ({"npc": "x", "on": "[tag=room]"}, "<places>.<surface>"),
                         ({"npc": "x", "on": "[tag=room].floor", "size": 1}, "population entry")]:
        with pytest.raises(ValueError, match=message):
            population_placements({"g": bad}, _Refs(2), "s", ["p"])


def test_activities_take_weights_cooldowns_and_visits(tmp_path):
    path = tmp_path / "npcs" / "visitor.yaml"
    path.parent.mkdir()
    path.write_text("kind: npc\nversion: 1\nneeds: { curiosity: { initial: 0.5, decay: 0.01 } }\n"
                    "activities:\n  look_in: { action: visit, visit: dining, weight: 0.5, cooldown: 120,"
                    " advertises: { curiosity: 0.3 } }\n")
    d = load_definition(path)
    assert d["activities"]["look_in"] == {"action": "visit", "advertises": {"curiosity": 0.3}, "weight": 0.5,
                                          "cooldown": 120.0, "visit": "dining"}
    path.write_text("kind: npc\nversion: 1\nactivities: { idle: { often: 2 } }\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_definition(path)


def test_visits_are_regions_in_the_npcs_frame(tmp_path):
    scene = COTTAGE_NPC.replace("npc: npcs/resident.yaml", "character: people/marco.yaml").replace("seed: 7\n", "")
    scene = scene.replace("home: house.floor", "home: house.floor\n    visits: { lawn: { rect: [-1, 4, 1, 6] } }")
    path = tmp_path / "scene.yaml"
    path.write_text(scene)
    root = SceneComposer(assets_dir=ASSETS_DIR).compose(path)
    npc = next(n for n in root.iter_nodes() if n.meta.get("type") == "npc")
    lawn = npc.meta["npc"]["visits"]["lawn"]
    m = npc.transform.to_matrix()            # back to the scene frame: the rect at y = 0
    corners = [(m @ np.array([x, lawn["y"], z, 1.0]))[:3] for x, z in lawn["polygon"]]
    assert np.allclose(corners, [[-1, 0, 4], [1, 0, 4], [1, 0, 6], [-1, 0, 6]], atol=1e-3)


def test_a_character_is_a_named_person_on_top_of_a_role(tmp_path):
    """A `character:` placement builds the role's NPC with the character's keys on top: its name,
    its look (body params on the role's archetype, its seed) and its own needs; the placement's
    keys override the character's defaults."""
    from geogen.export import exported_extras
    from geogen.npc import load_character

    definition, defaults = load_character(ASSETS_DIR / "people/marco.yaml", ASSETS_DIR)
    role = load_definition(ASSETS_DIR / "npcs/hotel_staff.yaml")
    assert definition["definition"] == "hotel_staff" and definition["character"] == "marco"
    assert definition["name"] == "Marco Bellini" and defaults == {"seed": 5}
    assert definition["body"]["archetype"] == role["body"]["archetype"]
    assert definition["body"]["params"]["hair_style"] == "crop"
    assert definition["needs"]["work"] == {"initial": 0.1, "decay": 0.03}
    assert definition["needs"]["rest"] == role["needs"]["rest"]        # merged key by key

    scene = COTTAGE_NPC.replace("npc: npcs/resident.yaml", "character: people/marco.yaml").replace("seed: 7\n", "")
    npc = _npc(SceneComposer().compose_string(scene))
    data = npc.meta["npc"]
    assert data["seed"] == 5 and data["name"] == "Marco Bellini"
    assert data["body"]["character"]["seed"] == 5
    extras = exported_extras(npc)["geogen"]["npc"]
    assert extras["character"] == "marco" and extras["name"] == "Marco Bellini"
    # Same person in another placement; the placement's seed wins over the character's.
    again = _npc(SceneComposer().compose_string(scene))
    assert again.meta["npc"]["body"]["params"] == data["body"]["params"]
    reseeded = _npc(SceneComposer().compose_string(scene.replace("home: house.floor", "home: house.floor\n    seed: 9")))
    assert reseeded.meta["npc"]["seed"] == 9
    with pytest.raises(ValueError, match="one person"):
        SceneComposer().compose_string(scene.replace("home: house.floor", "home: house.floor\n    npc: npcs/resident.yaml"))


def test_work_steps_loop_a_clip_and_close_in():
    """`play` takes `loop` (how long to loop its clip) and `go_to` takes `close` (step up to the
    spot past the navmesh); the work actions use both, and the kitchen furniture names them."""
    actions = load_actions()
    for name, clip in (("cook", "stir"), ("prep", "chop"), ("wipe", "wipe"), ("write", "write")):
        steps = actions[name]["steps"]
        assert steps[0] == {"go_to": "approach", "close": True}
        assert steps[-1] == {"play": clip, "loop": "duration"}
    for bad in ({"wait": 3, "loop": "duration"}, {"face": "anchor", "close": True}, {"go_to": "approach", "close": 1}):
        with pytest.raises(ValueError):
            parse_actions({"kind": "npc_actions", "version": 1, "actions": {"x": {"steps": [bad]}}})
    parsed = parse_actions({"kind": "npc_actions", "version": 1,
                            "actions": {"x": {"steps": [{"play": "stir", "loop": [5, 9]}]}}})
    assert parsed["x"]["steps"][0]["loop"] == [5.0, 9.0]
    loader = LayoutLoader()
    for asset, action in (("kitchen_range", "cook"), ("prep_counter", "prep"), ("bar_counter", "wipe")):
        assert loader.load(ASSETS_DIR / f"{asset}.yaml").meta["affordances"][0]["action"] == action
    look = load_definition(ASSETS_DIR / "npcs/resident.yaml")["attention"]["look"]
    assert look["busy_every"] == [5.0, 12.0] and look["busy_glance"] == 1.2
