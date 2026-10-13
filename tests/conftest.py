"""Shared fixtures: locating and running the Godot runtime (runtime/godot), built scenes,
and the fast tier (``pytest --fast``)."""

import fcntl
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.pop("GEOGEN_PROJECT", None)         # the tests are of the core: no project (geogen.roots)

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

    def run(*user_args: str, engine_args: tuple[str, ...] = ()) -> str:
        """``engine_args`` go before ``--`` (e.g. ("--fixed-fps", "60") to run unthrottled)."""
        result = subprocess.run(
            [godot, "--headless", *engine_args, "--path", str(GODOT_PROJECT), "--", *user_args],
            capture_output=True, text=True, timeout=240,   # generous: parallel test runs share the CPU
        )
        assert result.returncode == 0, result.stderr
        assert "SCRIPT ERROR" not in result.stdout + result.stderr, result.stdout + result.stderr
        return result.stdout

    return run


@pytest.fixture(scope="session")
def godot_runtime(run_godot):
    """``godot_runtime(scene, generated, *args)``: a long-lived headless runtime driven over its
    control API (geogen.runtime_client.RuntimeClient); closed at the end of the session.
    Tests typically hold one per module and ask it many questions."""
    from geogen.runtime_client import RuntimeClient

    clients = []

    def launch(scene: str, generated, *args: str):
        client = RuntimeClient.launch(scene, generated=generated, args=args,
                                      engine_args=("--headless", "--fixed-fps", "60"))
        clients.append(client)
        return client

    yield launch
    for client in clients:
        client.close()


@pytest.fixture(scope="session")
def scene_registry():
    from geogen.main import _build_registry
    return _build_registry()


# ---- built scenes and exports, shared between workers and runs ----------------------------------

BUILT_DIR = Path(__file__).parent.parent / ".pytest_cache" / "geogen-built"
KEEP_BUILDS = 2                 # fingerprints kept on disk (each is every scene a run built)


def _fingerprint() -> str:
    """What built scenes depend on: the assets and geogen's source."""
    import hashlib

    from geogen.layout.cache import source_fingerprint

    root = Path(__file__).parent.parent
    digest = hashlib.sha1(source_fingerprint().encode())
    for path in sorted(p for p in (root / "assets").rglob("*") if p.is_file() and p.suffix != ".yaml"):
        stat = path.stat()
        digest.update(f"{path.relative_to(root)}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def _shared(kind: str, name: str):
    """Lock and path for one shared build: ``with lock: if not path.exists(): make it``.

    Every xdist worker, and every later run until a source file changes, uses the one
    copy. ``GEOGEN_TEST_CACHE=off`` gives each session a private directory instead.
    """
    base = BUILT_DIR
    if os.environ.get("GEOGEN_TEST_CACHE", "").lower() in ("off", "0", "no"):
        base = Path(tempfile.gettempdir()) / f"geogen-built-{os.environ.get('PYTEST_XDIST_TESTRUNUID', os.getpid())}"
    folder = base / _fingerprint()
    folder.mkdir(parents=True, exist_ok=True)
    return _Lock(folder / f".{kind}-{name}.lock"), folder / f"{kind}-{name}"


class _Lock:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self):
        self.file = open(self.path, "w")
        fcntl.flock(self.file, fcntl.LOCK_EX)

    def __exit__(self, *exc) -> None:
        self.file.close()


def _prune_builds() -> None:
    """Drop all but the newest few fingerprints (on the controller, once a run)."""
    if not BUILT_DIR.is_dir():
        return
    folders = sorted((p for p in BUILT_DIR.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in folders[KEEP_BUILDS:]:
        shutil.rmtree(stale, ignore_errors=True)


@pytest.fixture(scope="session")
def built_scene(scene_registry):
    """``built_scene(name)``: a registry scene, built once.

    Built by whichever worker asks first and pickled (``.pytest_cache/geogen-built``);
    everyone else, and later runs with the same sources, load that. Returns a fresh
    ``instance()`` each call (nodes copied, meshes shared), so tests may move or
    re-pose it but must not edit mesh data in place.
    """
    import pickle

    cache = {}

    def get(name: str):
        if name in EXHAUSTIVE_SCENES and not _exhaustive:
            pytest.skip(f"builds '{name}', an exhaustive-tier scene: run pytest --all")
        if name not in cache:
            lock, path = _shared("scene", name)
            with lock:
                if path.exists():
                    cache[name] = pickle.loads(path.read_bytes())
                else:
                    cache[name] = scene_registry[name]()
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(pickle.dumps(cache[name], protocol=pickle.HIGHEST_PROTOCOL))
                    tmp.replace(path)
        return cache[name].instance()

    return get


@pytest.fixture(scope="session")
def scene_export(built_scene):
    """``scene_export(name)``: a directory holding ``<name>.glb`` and its manifest, exported
    once and shared like ``built_scene`` (pass it to the runtime as ``--generated``). The
    runtime's navmesh cache lives in it too (keyed by the export and the baking scripts), so
    a scene is baked once, and editing GDScript reuses the exports. Read-only: tests that
    change an export make their own."""
    from geogen.export import export_scene

    def get(name: str) -> Path:
        lock, path = _shared("export", name)
        with lock:
            if not (path / f"{name}.manifest.json").exists():
                shutil.rmtree(path, ignore_errors=True)
                path.mkdir(parents=True)
                export_scene(built_scene(name), path / f"{name}.glb")
        return path

    return get


# Module fixtures that build or export something expensive, mapped to the
# xdist group their tests share (default: one group per module + fixture).
# The town district joins the town scene's QA tests so town is built once.
SHARED_FIXTURE_GROUPS = {"district": "scene:town"}
SHARED_FIXTURES = ("cottage_dir", "auto_room_dir", "suite_dir", "small_building_dir", "strip_dir", "district")
SCENE_PARAM_MODULES = ("test_asset_quality.py", "test_scenes.py", "test_describe.py")


# The exhaustive tier (``pytest --all``): scenes that are compositions of things the default
# run already checks piece by piece (the districts, the dressed showcases, the galleries of
# people). They cost most of a full run: building one takes up to 90 s, and every per-scene
# check and runtime test on it scales with its size. Without ``--all``, per-scene tests of
# these are deselected and any other test that builds one is skipped.
EXHAUSTIVE_SCENES = {"city", "town", "main_street", "street", "street_side", "hotel_showcase", "restaurant",
                     "archetype_gallery", "people_gallery"}

DURATIONS_KEY = "geogen/durations"     # {test id: seconds}, in .pytest_cache (per machine)
FAST_BUDGET = 1.0
_XDIST_GROUP = re.compile(r"@[\w.:\-]+$")      # what --dist loadgroup appends to a test id


def pytest_addoption(parser):
    parser.addoption(
        "--fast", nargs="?", const=FAST_BUDGET, default=None, type=float, metavar="SECONDS",
        help="the inner loop: skip Godot runs and every test (or group of tests sharing a build) that "
             f"took SECONDS or longer when it last ran (default {FAST_BUDGET:g}). Timings are recorded by "
             "every run; tests never timed are kept.")


    parser.addoption(
        "--all", action="store_true", dest="exhaustive",
        help="also run the exhaustive tier: tests marked `exhaustive` and everything that builds one of "
             "the big composite scenes (conftest.EXHAUSTIVE_SCENES). Slow; for before a release.")


def pytest_xdist_auto_num_workers(config):
    """Workers for ``-n auto`` (pyproject's default): bounded by memory, not cores."""
    return 4 if config.getoption("exhaustive") else min(8, os.cpu_count() or 1)


def pytest_configure(config):
    global _exhaustive
    _exhaustive = bool(config.getoption("exhaustive"))


_exhaustive = False


def _durations(config) -> dict:
    cache = getattr(config, "cache", None)
    return dict(cache.get(DURATIONS_KEY, {})) if cache is not None else {}


def _group_of(item) -> str:
    mark = item.get_closest_marker("xdist_group")
    return mark.args[0] if mark and mark.args else item.nodeid


def _deselect_slow(config, items) -> None:
    budget = config.getoption("fast")
    if budget is None:
        return
    durations = _durations(config)
    cost: dict[str, float] = {}
    for item in items:
        cost[_group_of(item)] = cost.get(_group_of(item), 0.0) + durations.get(item.nodeid, 0.0)
    slow = [i for i in items if cost[_group_of(i)] >= budget or i.get_closest_marker("godot")]
    if slow:
        dropped = set(map(id, slow))
        items[:] = [i for i in items if id(i) not in dropped]
        config.hook.pytest_deselected(items=slow)


@pytest.hookimpl(tryfirst=True)   # before xdist reads the xdist_group marks
def pytest_collection_modifyitems(config, items):
    """Mark Godot tests, pin tests sharing a build to one xdist worker, and drop
    the slow ones under ``--fast``.

    ``--dist loadgroup`` sends each ``xdist_group`` to a single worker, so a
    scene (``built_scene``) or module export fixture is built once per run
    rather than once per worker that happens to get one of its tests. A group's
    cost is what its tests took together: the first one pays for the build.
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
        if group and group.removeprefix("scene:") in EXHAUSTIVE_SCENES:
            item.add_marker(pytest.mark.exhaustive)
    if not config.getoption("exhaustive"):
        extra = [i for i in items if i.get_closest_marker("exhaustive")]
        if extra:
            dropped = set(map(id, extra))
            items[:] = [i for i in items if id(i) not in dropped]
            config.hook.pytest_deselected(items=extra)
    _deselect_slow(config, items)


# ---- timings for --fast -------------------------------------------------------------------------

_timed: dict[str, float] = {}


def pytest_runtest_logreport(report):
    """Add up each test's setup, call and teardown (on the xdist controller, or in a serial run)."""
    nodeid = _XDIST_GROUP.sub("", report.nodeid)
    _timed[nodeid] = _timed.get(nodeid, 0.0) + report.duration


def pytest_sessionfinish(session):
    config = session.config
    if not hasattr(config, "workerinput"):
        _prune_builds()
        for private in Path(tempfile.gettempdir()).glob("geogen-built-*"):     # GEOGEN_TEST_CACHE=off runs
            if private.name.endswith(str(os.environ.get("PYTEST_XDIST_TESTRUNUID", os.getpid()))):
                shutil.rmtree(private, ignore_errors=True)
    if hasattr(config, "workerinput") or not _timed or getattr(config, "cache", None) is None:
        return
    # A run of everything (--all) replaces the timings, and a run of the default tier those of
    # the tests it ran. A partial run only raises one that has clearly grown (not jitter): a
    # test that was quick because another had built its scene is slow when it has to build it itself.
    tier = (config.args_source == pytest.Config.ArgsSource.TESTPATHS and config.getoption("fast") is None
            and not config.getoption("keyword") and not config.getoption("markexpr"))
    durations = {} if tier and config.getoption("exhaustive") else _durations(config)
    for nodeid, seconds in _timed.items():
        known = durations.get(nodeid)
        if tier or known is None or seconds > max(2 * known, known + 0.5):
            durations[nodeid] = round(seconds, 3)
    config.cache.set(DURATIONS_KEY, durations)
