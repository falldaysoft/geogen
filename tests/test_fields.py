"""The shared readers: authored YAML blocks, mesh attributes and layered NPC actions."""

import numpy as np
import pytest
import yaml

from geogen import fields, npc, roots
from geogen.core.mesh import Mesh
from geogen.core.meshops import boundary_loops, edges, position_ids


def test_unknown_keys_name_the_block_and_the_nearest_key():
    with pytest.raises(KeyError, match=r"crate: unknown keys \['mas'\] \(did you mean 'mass'\?\); known: \['mass'\]"):
        fields.known_keys({"mas": 1}, {"mass"}, "crate", KeyError)


def test_a_block_is_a_mapping():
    with pytest.raises(ValueError, match="crate: expected {mass}, got 3"):
        fields.block(3, ["mass"], "crate")


def test_values_are_reported_where_they_were_written():
    assert fields.number({"mass": "2.5"}, "mass", 1.0, "crate") == 2.5
    assert fields.integer({}, "count", 4, "crate") == 4
    with pytest.raises(ValueError, match="crate: mass is a number, got 'heavy'"):
        fields.number({"mass": "heavy"}, "mass", 1.0, "crate")
    with pytest.raises(ValueError, match="crate: count is a whole number, got None"):
        fields.integer({"count": None}, "count", 1, "crate")
    with pytest.raises(ValueError, match="crate: shape is one of box, ball, got 'cone'"):
        fields.choice({"shape": "cone"}, "shape", "box", ("box", "ball"), "crate")


def test_take_carries_every_vertex_attribute():
    mesh = Mesh(np.arange(12, dtype=np.float64).reshape(4, 3), np.array([[0, 1, 2], [1, 3, 2]]),
                uvs=np.arange(8, dtype=np.float64).reshape(4, 2), colors=np.ones((4, 4)),
                joints=np.zeros((4, 4), dtype=np.int64), weights=np.ones((4, 4)) / 4,
                morphs={"bend": np.arange(12, dtype=np.float64).reshape(4, 3)},
                face_materials=np.array([0, 1]), materials=[None, None])
    part = mesh.take([1, 3], faces=np.empty((0, 3), dtype=np.int64))
    assert part.vertices.tolist() == mesh.vertices[[1, 3]].tolist()
    assert part.morphs["bend"].tolist() == mesh.morphs["bend"][[1, 3]].tolist()
    assert part.uvs.shape == (2, 2) and part.joints.shape == (2, 4) and part.face_materials is None
    kept = mesh.take([0, 1, 2], faces=np.array([[0, 1, 2]]), face_keep=[1])
    assert kept.face_materials.tolist() == [1]


def test_transform_owns_its_arrays():
    mesh = Mesh(np.zeros((3, 3)), np.array([[0, 1, 2]]), colors=np.ones((3, 4)))
    moved = mesh.translated([1, 0, 0])
    moved.colors[0, 0] = 0.0
    assert mesh.colors[0, 0] == 1.0


def test_boundary_loops_of_an_open_quad():
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    assert len(edges(faces)) == 6
    loops = boundary_loops(faces)
    assert len(loops) == 1 and sorted(map(tuple, loops[0].tolist())) == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert position_ids(np.array([[0.0, 0, 0], [0.0, 0, 1e-9], [1.0, 0, 0]])).tolist() == [0, 0, 1]


def test_a_project_adds_npc_actions_to_the_cores(tmp_path, monkeypatch):
    (tmp_path / "npcs").mkdir()
    (tmp_path / "npcs" / "actions.yaml").write_text(yaml.safe_dump({
        "kind": npc.ACTIONS_KIND, "version": npc.VERSION, "actions": {"hum": {"steps": [{"wait": 1.0}]}}}))
    core = set(npc.load_actions())
    monkeypatch.setattr(roots, "roots", lambda: [tmp_path, roots.CORE])
    assert set(npc.load_actions()) == core | {"hum"}
