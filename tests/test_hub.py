"""The hub (scenes/hub.yaml): a portal to every showcase scene, and a way back from each."""

import json
import math

import pytest

from geogen.catalogue import export_catalogue, load_catalogue
from geogen.export import gameplay_summary


def _spawns(summary) -> set[str]:
    return {s["name"] for s in summary["spawns"]}


def test_hub_is_the_default_showcase():
    catalogue = load_catalogue()
    assert catalogue.default == "hub" and catalogue.entries["hub"].group == "showcase"


def test_hub_reaches_every_showcase_scene_and_back(built_scene):
    catalogue = load_catalogue()
    hub = gameplay_summary(built_scene("hub"))
    showcases = [e.name for e in catalogue.select("showcase") if e.name != "hub"]
    targets = {t["scene"]: t for t in hub["travel"]}
    assert set(showcases) <= set(targets), f"no hub portal to {set(showcases) - set(targets)}"
    for name in showcases:
        scene = gameplay_summary(built_scene(name))
        # The hub's portal lands on a spawn the scene has...
        assert targets[name]["spawn"] in _spawns(scene), (name, targets[name])
        # ...and the scene has a way back to a hub spawn.
        back = [t for t in scene.get("travel", []) if t.get("scene") == "hub"]
        assert back, f"{name} has no travel point back to the hub"
        assert back[0]["spawn"] in _spawns(hub), (name, back[0])


def test_cottage_inside_other_scenes_has_no_hub_portal(built_scene):
    town = built_scene("town")
    hub_points = [t for t in gameplay_summary(town)["travel"] if t.get("scene") == "hub"]
    assert len(hub_points) == 1            # the town's own park portal, not one per cottage lot


@pytest.fixture(scope="module")
def hub_dir(tmp_path_factory, built_scene):
    """The hub and the cottage exported; the hub's other targets are in the catalogue but not exported."""
    out = tmp_path_factory.mktemp("generated")
    catalogue = load_catalogue()
    export_catalogue(catalogue, built_scene, out, catalogue.select("hub,cottage"), log=lambda *_: None)
    return out


def _json_lines(out: str, prefix: str) -> list[dict]:
    return [json.loads(line.removeprefix(prefix)) for line in out.splitlines() if line.startswith(prefix)]


def _spawn(generated, scene, name) -> dict:
    manifest = json.loads((generated / f"{scene}.manifest.json").read_text())
    return next(s for s in manifest["spawns"] if s["name"] == name)


def _walk_outward(spawn: dict) -> tuple[str, str]:
    """--spawn/--yaw to walk from a hub arrival spawn back out through its portal (away from the centre)."""
    x, y, z = spawn["position"]
    yaw = math.degrees(math.atan2(-x, -z))   # the player looks along (-sin yaw, -cos yaw)
    return f"--spawn={x},{y},{z}", f"--yaw={yaw}"


def test_walk_from_the_hub_to_the_cottage_and_back(run_godot, hub_dir):
    # From in front of the cottage portal, turn round (the arrival spawn faces the plaza) and walk in.
    out = run_godot("--generated=%s" % hub_dir, "--scene=hub",
                    *_walk_outward(_spawn(hub_dir, "hub", "from_cottage")), "--walk=3")
    travelled = _json_lines(out, "travelled: ")
    assert len(travelled) == 1 and travelled[0]["scene"] == "cottage" and travelled[0]["spawn"] == "from_hub"
    arrival = _spawn(hub_dir, "cottage", "from_hub")["position"]
    result = _json_lines(out, "walk result: ")[0]
    assert result["scene"] == "cottage" and (result["x"], result["z"]) == pytest.approx(
        (arrival[0], arrival[2]), abs=0.05)

    # And back: the cottage's hub portal is on the west lawn, behind its arrival spawn.
    out = run_godot("--generated=%s" % hub_dir, "--scene=cottage", "--spawn=-5.4,0,3.5", "--yaw=90", "--walk=3")
    travelled = _json_lines(out, "travelled: ")
    assert len(travelled) == 1 and travelled[0]["scene"] == "hub" and travelled[0]["spawn"] == "from_cottage"
    arrival = _spawn(hub_dir, "hub", "from_cottage")["position"]
    result = _json_lines(out, "walk result: ")[0]
    assert result["scene"] == "hub" and (result["x"], result["z"]) == pytest.approx(
        (arrival[0], arrival[2]), abs=0.05)
