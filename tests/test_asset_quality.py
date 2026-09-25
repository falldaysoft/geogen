"""Geometry QA across every registered asset and scene.

Catches regressions in generators and YAML: every mesh must be finite, free
of degenerate triangles and inconsistent winding, textured meshes need
metric UVs, and assets must fit the size they declare.
"""

from pathlib import Path

import numpy as np
import pytest

from geogen.core import meshops, uvmap
from geogen.layout.yaml_utils import safe_load_path
from geogen.main import _build_registry

REGISTRY = _build_registry()
SCENES = REGISTRY.names()


@pytest.fixture(scope="module", params=SCENES)
def scene(request, built_scene):
    return request.param, built_scene(request.param)


def test_meshes_are_valid(scene):
    name, root = scene
    problems = []
    for node, mesh in root.iter_meshes():
        report = meshops.validate(mesh)
        if report.issues:
            problems.append(f"{node.name}: {', '.join(report.issues)}")
    assert not problems, f"{name}:\n" + "\n".join(problems)


def test_textured_meshes_have_metric_uvs(scene):
    name, root = scene
    problems = []
    for node, mesh in root.iter_meshes():
        if mesh.material is None:
            continue
        if mesh.uvs is None:
            problems.append(f"{node.name}: textured but no UVs")
            continue
        density = uvmap.texel_density(mesh)
        # Transforms may scale parts, so allow a generous band around 1.0.
        if not 0.2 < density < 5.0:
            problems.append(f"{node.name}: UV density {density:.2f}")
    assert not problems, f"{name}:\n" + "\n".join(problems)


def test_interiors_pass_layout_qa(scene):
    from geogen.layout.qa import check_layout

    name, root = scene
    assert [str(i) for i in check_layout(root)] == [], name


def test_no_coplanar_z_fighting(scene):
    from geogen.layout.qa import coplanar_overlaps

    name, root = scene
    assert [str(i) for i in coplanar_overlaps(root)] == [], name


def test_scene_is_not_empty(scene):
    name, root = scene
    assert sum(len(m.faces) for _, m in root.iter_meshes()) > 0, name


# Assets whose declared size doesn't match their geometry yet (geogen-o3s.26).
KNOWN_SIZE_MISMATCH: set[str] = set()
ASSETS_DIR = Path(__file__).parent.parent / "assets"
ROOT_ASSETS = [n for n in SCENES if "parts" in (safe_load_path(ASSETS_DIR / f"{n}.yaml") if (ASSETS_DIR / f"{n}.yaml").exists() else {})]


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.xfail(reason="geogen-o3s.26", strict=True)) if n in KNOWN_SIZE_MISMATCH else n
     for n in ROOT_ASSETS],
)
def test_asset_fits_declared_size(name, built_scene):
    root = built_scene(name)
    declared = root.children[0].size
    verts = np.vstack([m.vertices for _, m in root.iter_meshes()])
    extent = np.ptp(verts, axis=0)
    # Parts may overhang slightly (eaves, trim) but not by more than 25%.
    assert np.all(extent <= np.asarray(declared) * 1.25 + 0.05), f"{name}: extent {extent} vs size {declared}"
