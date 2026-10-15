"""`geogen check` (geogen/check.py): a project's QA without the core test suite."""

import json

import numpy as np
import pytest

from geogen import roots
from geogen.check import main, material_findings, own_scenes, seam_ratio
from geogen.main import _build_registry

# A two-scene project on core assets: one fine, one whose rule waits for a signal the lever never raises.
FINE = """
name: fine
size: [6, 3, 6]
place:
  chair: {asset: chair.yaml}
"""
BROKEN = """
name: broken
size: [6, 3, 6]
place:
  lever: {asset: lever.yaml}
rules:
  - on: {signal: lever.yanked}
    do: [{notify: "It moved"}]
"""
@pytest.fixture
def project(tmp_path):
    (tmp_path / "geogen.yaml").write_text("kind: project\nname: checked\n")
    scenes = tmp_path / "assets" / "scenes"
    scenes.mkdir(parents=True)
    (scenes / "fine.yaml").write_text(FINE)
    (scenes / "broken.yaml").write_text(BROKEN)
    yield tmp_path
    roots.use()


def test_a_project_is_checked_scene_by_scene(project, capsys):
    assert main(["--project", str(project), "--json", "--checks", "mesh,state,extras"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert set(out["findings"]) == {"fine", "broken"}                   # its own scenes only
    assert out["findings"]["fine"] == []
    (finding,) = out["findings"]["broken"]
    assert (finding["check"], finding["kind"]) == ("state", "compile") and "never raises 'yanked'" in finding["message"]
    assert out["total"] == 1


def test_named_scenes_and_a_clean_exit(project, capsys):
    assert main(["--project", str(project), "fine", "--checks", "mesh,uvs,layout,state"]) == 0
    assert capsys.readouterr().out.strip().endswith("geogen check: 1 scenes: no findings")
    with pytest.raises(SystemExit):
        main(["--project", str(project), "nowhere"])
    with pytest.raises(SystemExit):
        main(["--project", str(project), "--checks", "vibes"])


def test_without_a_project_every_scene_is_the_projects(project):
    roots.use()
    registry = _build_registry()
    assert own_scenes(registry) == registry.names()
    roots.use(project)
    assert sorted(own_scenes(_build_registry())) == ["broken", "fine"]


def test_seams_are_measured_against_the_sharpest_inner_step():
    ramp = np.tile(np.linspace(0, 255, 64), (64, 1))     # wraps from white straight back to black
    assert seam_ratio(ramp, 1) > 10 and seam_ratio(ramp, 0) == 0
    assert material_findings("brick", size=64) == []
