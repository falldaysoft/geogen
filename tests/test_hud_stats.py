"""HUD stats and banners (geogen.hud, runtime hud_stats.gd / hud_banner.gd): numbers a scene
declares over its state, shown while it runs."""

import json

import pytest

from geogen.export import export_scene, manifest_path
from geogen.hud import HudError
from geogen.layout.composer import SceneComposer
from geogen.quests import QuestError, collect

SCENE = """
name: arena
size: [8, 3, 8]
spawns:
  start: {position: [0, 0, 2], facing: north}
state:
  lives: {scope: world, type: number, default: 20}
  wave:  {scope: world, type: number, default: 1}
  phase: {scope: world, type: choice, choices: [ready, wave, won, lost], default: ready}
  score: {scope: actor, type: number, default: 0}
place:
  floor: {asset: floor.yaml}
hud:
  stats:
    - {label: Lives, value: world.lives}
    - {label: Wave, value: world.wave, of: 8}
    - {label: Score, value: "actor.score * 10"}
  banners:
    - {when: 'world.phase == "lost"', text: The core has fallen, tone: bad}
    - {when: 'world.phase == "won"', text: Every wave beaten, tone: good}
"""

FLOOR = """
name: floor
origin: bottom_center
size: [8, 0.1, 8]
parts:
  slab: {primitive: cube, size: [1, 1, 1], anchor: bottom_center, bevel: 0, material: concrete, walkable: true}
"""


@pytest.fixture
def assets(tmp_path):
    (tmp_path / "arena.yaml").write_text(SCENE)
    (tmp_path / "floor.yaml").write_text(FLOOR)
    return tmp_path


def compose(assets):
    return SceneComposer(assets_dir=assets).compose(assets / "arena.yaml")


def test_hud_compiles_to_trees(assets):
    hud = collect(compose(assets), "arena").hud
    assert [s["label"] for s in hud["stats"]] == ["Lives", "Wave", "Score"]
    assert hud["stats"][0]["value"] == {"get": "world", "package": "arena", "name": "lives"}
    assert hud["stats"][1]["of"] == 8
    assert hud["banners"][0]["tone"] == "bad" and hud["banners"][1]["text"] == "Every wave beaten"


def test_hud_is_in_the_manifest(assets, tmp_path):
    out = tmp_path / "out" / "arena.glb"
    out.parent.mkdir()
    export_scene(compose(assets), out)
    manifest = json.loads(manifest_path(out).read_text())
    assert [s["label"] for s in manifest["hud"]["stats"]] == ["Lives", "Wave", "Score"]


@pytest.mark.parametrize("old, new, error, message", [
    ("value: world.lives}", "value: world.livse}", QuestError, r"arena.yaml:\d+: hud.stats.0.value: .*livse"),
    ("tone: bad}", "tone: grim}", HudError, "tone is one of info, good, bad"),
    ("{label: Lives, value: world.lives}", "{label: Lives}", HudError, r"hud.stats.0: .*value required"),
    ("value: world.lives}", "value: event.lives}", QuestError, "the HUD has no event to read"),
])
def test_bad_hud_fails(assets, old, new, error, message):
    path = assets / "arena.yaml"
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(error, match=message):
        collect(compose(assets), "arena")


def test_godot_shows_the_stats_and_the_banner(assets, tmp_path, godot_runtime):
    out = tmp_path / "gen"
    out.mkdir()
    export_scene(compose(assets), out / "arena.glb")
    runtime = godot_runtime("arena", out)
    runtime.call("wait_ready", timeout=60)
    shown = runtime.call("hud.stats")
    assert [s["text"] for s in shown["stats"]["stats"]] == ["20", "1 / 8", "0"], shown
    assert not shown["banner"]["visible"]
    runtime.call("state.set", key="world:arena.lives", value=17)
    runtime.call("state.set", key="actor:arena.score", value=3, actor="player")
    runtime.call("state.set", key="world:arena.phase", value="lost")
    shown = runtime.call("hud.stats")
    assert [s["text"] for s in shown["stats"]["stats"]] == ["17", "1 / 8", "30"], shown
    assert shown["banner"] == {"visible": True, "text": "The core has fallen", "tone": "bad"}
