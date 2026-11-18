"""Emitters (geogen/emitters.py): a part's `emitter:` and `sound:`, their `running` state, and the runtime."""

import json

import numpy as np
import pytest

from geogen import emitters, extras
from geogen.core.node import SceneNode
from geogen.export import export_scene
from geogen.generators.base import box_attachment_points
from geogen.runtime_client import step
from geogen.state import collect, emitter_state

from helpers import gltf_json, schema_validate

VENT = "emitter_yard:vent/nozzle"
POT = "emitter_yard:chimney/stack/cap/pot"      # attached part on part


def _node(block_key="emitter", **spec) -> SceneNode:
    node = SceneNode("spout")
    node.size = np.array([0.2, 0.4, 0.2])
    node.attachments.update(box_attachment_points(node.size))
    emitters.apply_part(node, {block_key: spec}, "spout")
    return node


def test_emitter_is_normalised_in_the_parts_frame():
    e = _node(at="top", rate=10, speed=2, size=[0.1, 0.3], direction=[0, 2, 0]).meta["emitter"]
    assert e["offset"] == [0, 0.2, 0]                  # the part's centre is its origin
    assert e["speed"] == [2.0, 2.0] and e["size"] == [0.1, 0.3] and e["direction"] == [0, 1, 0]
    assert _node(at=[0.1, 0, 0]).meta["emitter"]["offset"] == [0.1, 0, 0]


@pytest.mark.parametrize("spec, message", [
    ({"rates": 3}, "unknown keys"),
    ({"material": "lava"}, "material must be"),
    ({"rate": 9000, "lifetime": 2}, "live particles"),
    ({"at": "nowhere"}, "no attachment point"),
    ({"direction": [0, 0, 0]}, "no length"),
])
def test_bad_emitters_say_what_is_wrong(spec, message):
    with pytest.raises(ValueError, match=message):
        _node(**spec)


def test_a_part_has_one_running_state_for_both_blocks():
    node = _node(running=False)
    assert emitter_state(node) == {"running": {"type": "bool", "default": False}}
    with pytest.raises(ValueError, match="start alike"):
        emitters.apply_part(node, {"sound": {"running": True}}, "spout")


def test_yard_declares_running_and_binds_the_valve(built_scene):
    declared = collect(built_scene("emitter_yard"), "emitter_yard")
    assert declared.variables[f"{VENT}#running"]["default"] is True
    assert declared.variables[f"{POT}#running"]["default"] is True


def test_export_carries_the_blocks_and_the_binding(tmp_path, built_scene):
    path = tmp_path / "emitter_yard.glb"
    export_scene(built_scene("emitter_yard"), path)
    blocks = {}
    for node in gltf_json(path)["nodes"]:
        geogen = node.get("extras", {}).get("geogen", {})
        if "emitter" in geogen or "sound" in geogen:
            schema_validate({"geogen": geogen}, extras.schema())
            blocks[geogen["uid"]] = geogen
    assert {"droplet", "soft", "streak"} == {b["emitter"]["material"] for b in blocks.values()}
    assert "sound" in blocks[VENT] and "sound" not in blocks[POT]
    manifest = json.loads((tmp_path / "emitter_yard.manifest.json").read_text())
    assert f"{VENT}#running" in json.dumps(manifest["rules"])


def test_runtime_runs_and_switches_them(godot_script, scene_export):
    generated = scene_export("emitter_yard")
    run = godot_script([
        step("wait_ready"), step("wait", frames=10), step("world.emitters"),
        step("state.set", key="emitter_yard:vent#valve", value="closed"),
        step("state.set", key=f"{POT}#running", value=False),
        step("wait", frames=5), step("world.emitters"),
    ], f"--generated={generated}", "--scene=emitter_yard")
    before, after = run.all("world.emitters")
    assert len(before) == 4 and all(row["emitting"] for row in before.values())
    assert before[VENT]["sounding"] and before[VENT]["amount"] == 98
    assert not after[VENT]["emitting"] and not after[VENT]["sounding"] and not after[POT]["emitting"]
    assert after["emitter_yard:rain/belly"]["emitting"]              # the others run on
