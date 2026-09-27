"""Clothing (geogen-z2b.16.6): tight garments as ring edits, loose shells, colours, poke-through."""

import numpy as np
import pytest

from geogen.core import meshops
from geogen.core.skin import pose_clips
from geogen.layout import LayoutLoader

HUMANOID = "assets/characters/humanoid.yaml"
OUTFITS = ["none", "casual", "jeans_tee", "smart", "dress", "skirt_sweater", "shorts_tee", "casual_f"]


def _inside(mesh, point) -> bool:
    """Generalized winding number (closed mesh; a hemmed shell as a solid): ~1 inside, ~0 outside."""
    from geogen.clipping import _solid

    tris = _solid(mesh)
    a, b, c = (tris[:, i] - point for i in range(3))
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", b, c) * la \
        + np.einsum("ij,ij->i", c, a) * lb
    return abs(np.arctan2(det, den).sum() * 2 / (4 * np.pi)) > 0.5


@pytest.fixture(scope="module")
def loader():
    return LayoutLoader()


@pytest.mark.parametrize("outfit", OUTFITS)
def test_outfits_build_clean_meshes(loader, outfit):
    body = loader.load(HUMANOID, params={"outfit": outfit})
    mesh = body.find("body").mesh
    assert meshops.validate(mesh).ok, meshops.validate(mesh).issues
    np.testing.assert_allclose(mesh.weights.sum(axis=1), 1.0)
    materials = {m.name for n in body.iter_nodes() if n.mesh is not None
                 for m in (n.mesh.materials or [n.mesh.material])}
    assert len(materials) <= 6                      # skin, face, hair + at most three fabrics
    assert body.meta["humanoid"]["triangles"] <= 4200
    clothes = body.find("clothes")
    if outfit in ("dress", "skirt_sweater", "casual_f"):
        assert clothes is not None and meshops.validate(clothes.mesh).watertight
    else:
        assert clothes is None


def test_garment_colours_are_per_slot_vertex_colours(loader):
    body = loader.load(HUMANOID, params={"outfit": "jeans_tee", "top_color": "red", "bottom_color": "denim",
                                         "skin_tone": 0.5})
    mesh = body.find("body").mesh
    names = [m.name for m in mesh.materials]
    assert names[:4] == ["skin", "cloth", "denim", "leather"]
    for slot, rgb in ((1, (0.72, 0.15, 0.15)), (2, (0.3, 0.42, 0.62))):
        verts = np.unique(mesh.faces[mesh.face_materials == slot])
        np.testing.assert_allclose(mesh.colors[verts, :3], np.tile(rgb, (len(verts), 1)))
    # Garments sit proud of the skin they cover.
    bare = loader.load(HUMANOID, params={"outfit": "none"}).find("body").mesh
    chest = lambda m: np.ptp(m.vertices[np.abs(m.vertices[:, 1] - 1.2) < 0.01][:, 2])
    assert chest(mesh) > chest(bare) + 0.005


def test_tops_cover_the_waistband(loader):
    """Layer 2 tops win the overlap with trousers: nothing denim above the shirt hem at the front."""
    mesh = loader.load(HUMANOID, params={"outfit": "jeans_tee"}).find("body").mesh
    denim = mesh.materials.index(next(m for m in mesh.materials if m.name == "denim"))
    centres = mesh.vertices[mesh.faces[mesh.face_materials == denim]].mean(axis=1)
    front = centres[(np.abs(centres[:, 0]) < 0.12) & (centres[:, 2] > 0.05)]
    assert front[:, 1].max() < 0.96


@pytest.mark.parametrize("pose", ["walk", "pose_sit"])
def test_skirt_follows_the_thighs(loader, pose):
    body = loader.load(HUMANOID, params={"preset": "feminine", "outfit": "casual_f"})
    clip = next(c for c in body.clips if c.name == pose)
    times = np.linspace(0, clip.duration, 7)[:-1] if pose == "walk" else [0.0]
    for t in times:
        clip.apply(body, t)
        skirt = body.find("clothes").world_mesh()
        for side in ("Left", "Right"):
            hip = body.find(f"{side}UpperLeg").world_transform()[:3, 3]
            knee = body.find(f"{side}LowerLeg").world_transform()[:3, 3]
            mid = hip + (knee - hip) * 0.5
            assert _inside(skirt, mid), (pose, t, side)
