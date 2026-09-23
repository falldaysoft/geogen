"""Round-trip tests for glTF/GLB and OBJ export."""

import numpy as np
import pytest
import trimesh

from geogen.export import export_scene
from geogen.main import _build_registry
from geogen.render import scene_bounds

REGISTRY = _build_registry()


@pytest.mark.parametrize("scene", ["chair", "dining_set", "house_peaked", "room"])
def test_glb_round_trip_preserves_geometry(scene, tmp_path):
    root = REGISTRY[scene]()
    path = export_scene(root, tmp_path / f"{scene}.glb")
    loaded = trimesh.load(path)
    np.testing.assert_allclose(loaded.bounds, scene_bounds(root), atol=1e-4)
    src_tris = sum(len(m.faces) for _, m in root.iter_meshes())
    assert sum(len(g.faces) for g in loaded.geometry.values()) == src_tris
    # Hierarchy: every scene node with a mesh is its own glTF node.
    assert len(loaded.geometry) == sum(1 for _, m in root.iter_meshes() if len(m.faces))


def test_glb_has_textured_pbr_materials(tmp_path):
    loaded = trimesh.load(export_scene(REGISTRY["table"](), tmp_path / "table.glb"))
    for geom in loaded.geometry.values():
        assert isinstance(geom.visual, trimesh.visual.TextureVisuals)
        mat = geom.visual.material
        assert mat.baseColorTexture is not None
        assert mat.metallicRoughnessTexture is not None
    # Texture-space UVs: a 1.2 m tabletop with a 0.8 m wood tile spans > 1 repeat.
    top = loaded.geometry["top"]
    assert np.ptp(top.visual.uv, axis=0).max() > 1.2


def test_obj_export_writes_material_and_textures(tmp_path):
    path = export_scene(REGISTRY["chair"](), tmp_path / "chair.obj")
    assert path.exists()
    assert any(p.suffix == ".mtl" for p in tmp_path.iterdir())
    assert any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_unknown_format_rejected(tmp_path):
    with pytest.raises(ValueError):
        export_scene(REGISTRY["chair"](), tmp_path / "chair.fbx")
