"""Shared fixtures: locating and running the Godot runtime (runtime/godot)."""

import fcntl
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

GODOT_PROJECT = Path(__file__).parent.parent / "runtime" / "godot"

# Keep test processes out of the macOS Dock (one bouncing icon per xdist
# worker otherwise): Qt tests never show a window, and pyglet (pyrender's
# offscreen GL context) makes its app "regular"; an accessory app can still
# create GL contexts but has no Dock icon.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "darwin":
    from pyglet.libs.darwin import cocoapy
    cocoapy.NSApplicationActivationPolicyRegular = cocoapy.NSApplicationActivationPolicyAccessory


def _godot_binary() -> str | None:
    for candidate in (os.environ.get("GODOT"), shutil.which("godot"), shutil.which("godot4"),
                      "/Applications/Godot.app/Contents/MacOS/Godot"):
        if candidate and Path(candidate).exists():
            return candidate
    return None


@pytest.fixture(scope="session")
def run_godot():
    """Run the runtime headless with user args; returns stdout. Skips without Godot."""
    godot = _godot_binary()
    if godot is None:
        pytest.skip("Godot not installed (set $GODOT)")
    # class_name scripts need the import cache; building it is idempotent, but
    # parallel (xdist) workers must not write it at the same time.
    with open(Path(tempfile.gettempdir()) / "geogen-godot-import.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run([godot, "--headless", "--path", str(GODOT_PROJECT), "--import"],
                       capture_output=True, timeout=180)

    def run(*user_args: str) -> str:
        result = subprocess.run(
            [godot, "--headless", "--path", str(GODOT_PROJECT), "--", *user_args],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "SCRIPT ERROR" not in result.stdout + result.stderr, result.stdout + result.stderr
        return result.stdout

    return run


@pytest.fixture(scope="session")
def scene_registry():
    from geogen.main import _build_registry
    return _build_registry()


@pytest.fixture(scope="session")
def built_scene(scene_registry):
    """``built_scene(name)``: a registry scene, built once per session.

    Returns a fresh ``instance()`` each call (nodes copied, meshes shared), so
    tests may move or re-pose it but must not edit mesh data in place.
    """
    cache = {}

    def get(name: str):
        if name not in cache:
            cache[name] = scene_registry[name]()
        return cache[name].instance()

    return get


# Module fixtures that build or export something expensive, mapped to the
# xdist group their tests share (default: one group per module + fixture).
# The town district joins the town scene's QA tests so town is built once.
SHARED_FIXTURE_GROUPS = {"district": "scene:town"}
SHARED_FIXTURES = ("cottage_dir", "auto_room_dir", "suite_dir", "small_building_dir", "strip_dir", "district")
SCENE_PARAM_MODULES = ("test_asset_quality.py", "test_scenes.py")


@pytest.hookimpl(tryfirst=True)   # before xdist reads the xdist_group marks
def pytest_collection_modifyitems(config, items):
    """Mark Godot tests and pin tests sharing a build to one xdist worker.

    ``--dist loadgroup`` sends each ``xdist_group`` to a single worker, so a
    scene (``built_scene``) or module export fixture is built once per run
    rather than once per worker that happens to get one of its tests.
    """
    for item in items:
        if "run_godot" in item.fixturenames:
            item.add_marker(pytest.mark.godot)
        group = None
        params = getattr(item, "callspec", None)
        if item.path.name in SCENE_PARAM_MODULES and params is not None:
            name = params.params.get("scene", params.params.get("scene_name", params.params.get("name")))
            if isinstance(name, str):
                group = f"scene:{name}"
        for fixture in SHARED_FIXTURES:
            if fixture in item.fixturenames:
                group = SHARED_FIXTURE_GROUPS.get(fixture, f"{item.path.stem}:{fixture}")
                break
        if group:
            item.add_marker(pytest.mark.xdist_group(group))
