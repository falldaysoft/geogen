"""Shared fixtures: locating and running the Godot runtime (runtime/godot), built scenes,
and the fast tier (``pytest --fast``)."""

import fcntl
import json
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

# Engine flags for a test launch unless it passes its own: unthrottled, so frames run back to
# back with a fixed 1/60 s step instead of waiting for the wall clock. A walk or a wait of game
# time then costs what its frames cost, not its length in real seconds.
UNTHROTTLED = ("--fixed-fps", "60")

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

    def run(*user_args: str, engine_args: tuple[str, ...] | None = None) -> str:
        """``engine_args`` go before ``--`` (default ``UNTHROTTLED``; ``()`` for real time)."""
        engine_args = UNTHROTTLED if engine_args is None else engine_args
        result = subprocess.run(
            [godot, "--headless", *engine_args, "--path", str(GODOT_PROJECT), "--", *user_args],
            capture_output=True, text=True, timeout=240,   # generous: parallel test runs share the CPU
        )
        assert result.returncode == 0, result.stderr
        assert "SCRIPT ERROR" not in result.stdout + result.stderr, result.stdout + result.stderr
        return result.stdout

    return run


@pytest.fixture(scope="session")
def control_validator():
    """Validate a script document against docs/schema/geogen-control.v1.schema.json (raises on a mistake)."""
    import jsonschema
    from referencing import Registry, Resource

    schema_dir = Path(__file__).parent.parent / "docs" / "schema"
    schema = json.loads((schema_dir / "geogen-control.v1.schema.json").read_text())
    save = json.loads((schema_dir / "geogen-save.v1.schema.json").read_text())
    registry = Registry().with_resources([(save["$id"], Resource.from_contents(save))])
    validator = jsonschema.Draft202012Validator(schema, registry=registry)

    def validate(document: dict) -> None:
        errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
        if errors:
            raise jsonschema.ValidationError("; ".join(f"{list(e.absolute_path)}: {e.message}" for e in errors))

    return validate


@pytest.fixture(scope="session")
def godot_script(run_godot, control_validator):
    """``godot_script(steps, *user_args, start=None, engine_args=None)``: a headless launch that runs
    control API ``steps`` from the ``start`` conditions (and any flags in ``user_args``), checked:
    every step succeeded, no script errors. Returns the ``ScriptRun`` (``run["player.status"]``)."""
    from geogen.runtime_client import run_script

    def run(steps=(), *user_args: str, start: dict | None = None, engine_args: tuple[str, ...] | None = None,
            check: bool = True):
        """``check=False``: a step may fail (the run goes on); look at ``run.entry(...)["ok"]``.
        ``engine_args`` default to ``UNTHROTTLED``; pass ``()`` to run in real time."""
        engine_args = UNTHROTTLED if engine_args is None else engine_args
        control_validator({"start": start or {}, "steps": list(steps)})   # what any runtime could run
        result = run_script(steps, start=start, args=user_args, engine_args=("--headless", *engine_args),
                            godot=_godot_binary(), project=GODOT_PROJECT, stop_on_error=check)
        assert "SCRIPT ERROR" not in result.output, result.output
        return result.check() if check else result

    return run


@pytest.fixture(scope="module")
def godot_runtime(run_godot):
    """``godot_runtime(scene, generated, *args)``: a long-lived headless runtime driven over its
    control API (geogen.runtime_client.RuntimeClient); closed when the module's tests are done.
    Tests typically hold one per module and ask it many questions.

    Module scope, not session: an unthrottled runtime (``--fixed-fps``) keeps stepping its
    world between calls, so one kept to the end of the session burns a core for nothing while
    every later test on that worker runs."""
    from geogen.runtime_client import RuntimeClient

    clients = []

    def launch(scene: str, generated, *args: str):
        client = RuntimeClient.launch(scene, generated=generated, args=args,
                                      engine_args=("--headless", *UNTHROTTLED))
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
                from geogen.motion import write_motions

                write_motions(path)          # as --export-godot writes it beside every export
        return path

    return get


@pytest.fixture(scope="session")
def plain_cottage():
    """A directory holding ``cottage.glb`` built with ``lived_in: 0``: one room and the one resident of
    npcs/resident.yaml, which the NPC, quest and save tests are written around (the showcase's
    cottage has a bathroom, a kitchen and two people who plan). Shared like ``scene_export``."""
    from geogen.export import export_scene
    from geogen.layout import SceneComposer

    assets = Path(__file__).parent.parent / "assets"
    lock, path = _shared("export", "cottage_plain")
    with lock:
        if not (path / "cottage.manifest.json").exists():
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True)
            scene = SceneComposer(assets_dir=assets).compose(assets / "scenes" / "cottage.yaml", params={"lived_in": 0})
            export_scene(scene, path / "cottage.glb")
            from geogen.motion import write_motions

            write_motions(path)
    return path


def _project_key(project: Path, scene: str) -> str:
    """A shared build's name for a project's scene: it changes with any of the project's files."""
    import hashlib

    digest = hashlib.sha1()
    for path in sorted(p for p in Path(project).rglob("*") if p.is_file()):
        stat = path.stat()
        digest.update(f"{path.relative_to(project)}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return f"{Path(project).name}-{scene}-{digest.hexdigest()[:10]}"


@pytest.fixture(scope="session")
def project_scene():
    """``project_scene(project, name)``: a scene of a project (``examples/<project>``, a content
    root), built once and shared like ``built_scene``; a fresh ``instance()`` each call. The
    content roots are back to the core's afterwards: a test that goes on to read the project's
    files (``collect``, an export of its own) wraps that in ``roots.use(project)`` itself."""
    import pickle

    from geogen import roots
    from geogen.registry import SceneRegistry

    cache = {}

    def get(project: Path, name: str):
        key = _project_key(project, name)
        if key not in cache:
            lock, path = _shared("project", key)
            with lock:
                if path.exists():
                    cache[key] = pickle.loads(path.read_bytes())
                else:
                    try:
                        roots.use(project)
                        registry = SceneRegistry()
                        registry.discover()
                        cache[key] = registry[name]()
                    finally:
                        roots.use()
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(pickle.dumps(cache[key], protocol=pickle.HIGHEST_PROTOCOL))
                    tmp.replace(path)
        return cache[key].instance()

    return get


@pytest.fixture(scope="session")
def project_export(project_scene):
    """``project_export(project, name)``: a directory holding the scene's GLB and manifest,
    exported once and shared like ``scene_export`` (read-only; the runtime's navmesh cache
    lives in it too)."""
    from geogen import roots
    from geogen.export import export_scene

    def get(project: Path, name: str) -> Path:
        lock, path = _shared("project-export", _project_key(project, name))
        with lock:
            if not (path / f"{name}.manifest.json").exists():
                shutil.rmtree(path, ignore_errors=True)
                path.mkdir(parents=True)
                scene = project_scene(project, name)
                try:
                    roots.use(project)
                    export_scene(scene, path / f"{name}.glb")
                    from geogen.motion import write_motions

                    write_motions(path)
                finally:
                    roots.use()
        return path

    return get


# What makes a test module's own fixture worth pinning its tests to one worker: it depends on one
# of these, so it builds, exports or launches something (see _shared_fixture_groups).
BUILD_FIXTURES = {"tmp_path_factory", "built_scene", "scene_export", "plain_cottage", "project_scene",
                  "project_export", "run_godot", "godot_runtime", "godot_script"}
# A group a fixture joins instead of its own: the town district shares the town scene's build.
SHARED_FIXTURE_GROUPS = {"district": "scene:town"}
SCENE_PARAMS = ("scene", "scene_name", "name")      # what a test parametrized over scenes calls its scene


# The exhaustive tier (``pytest --all``): scenes that are compositions of things the default
# run already checks piece by piece (the districts, the dressed showcases, the galleries of
# people). They cost most of a full run: building one takes up to 90 s, and every per-scene
# check and runtime test on it scales with its size. Without ``--all``, per-scene tests of
# these are deselected and any other test that builds one is skipped.
EXHAUSTIVE_SCENES = {"city", "town", "main_street", "street", "street_side", "hotel_showcase", "restaurant",
                     "archetype_gallery", "people_gallery", "crowd_test"}

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


def _scene_group(item) -> str | None:
    """``scene:<name>`` for a test parametrized over the scenes it builds."""
    params = getattr(item, "callspec", None)
    if params is None or "built_scene" not in item.fixturenames:
        return None
    name = next((params.params[p] for p in SCENE_PARAMS if p in params.params), None)
    return f"scene:{name}" if isinstance(name, str) else None


def _builds(name: str, defs: dict, seen: set) -> bool:
    """Whether the fixture ``name`` reaches one of BUILD_FIXTURES."""
    if name in BUILD_FIXTURES:
        return True
    if name in seen or name not in defs:
        return False
    seen.add(name)
    return any(_builds(arg, defs, seen) for arg in defs[name][-1].argnames)


def _shared_fixture_groups(items) -> dict:
    """``{(module path, fixture): xdist group}`` for the fixtures a module's tests share a build through.

    A module or session fixture defined in a test module that builds something is made once per
    worker, so the tests using it belong on one worker. Fixtures used together by a test join one
    group, since a test can only be on one worker. A parametrized fixture is left out: its tests
    build different things."""
    parent: dict = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for item in items:
        defs = item._fixtureinfo.name2fixturedefs
        shared = [(item.path, name) for name in item.fixturenames if name in defs
                  and defs[name][-1].scope in ("module", "session") and defs[name][-1].params is None
                  and getattr(defs[name][-1].func, "__module__", None) == item.module.__name__
                  and _builds(name, defs, set())]
        for key in shared:
            parent[find(key)] = find(shared[0])
    members: dict = {}
    for key in parent:
        members.setdefault(find(key), []).append(key)
    groups = {}
    for keys in members.values():
        names = sorted(name for _, name in keys)
        chosen = next((SHARED_FIXTURE_GROUPS[n] for n in names if n in SHARED_FIXTURE_GROUPS), None)
        groups.update({key: chosen or f"{key[0].stem}:{names[0]}" for key in keys})
    return groups


@pytest.hookimpl(tryfirst=True)   # before xdist reads the xdist_group marks
def pytest_collection_modifyitems(config, items):
    """Mark Godot tests, pin tests sharing a build to one xdist worker, and drop
    the slow ones under ``--fast``.

    ``--dist loadgroup`` sends each ``xdist_group`` to a single worker, so a
    scene (``built_scene``) or module export fixture is built once per run
    rather than once per worker that happens to get one of its tests. A group's
    cost is what its tests took together: the first one pays for the build.
    """
    fixture_groups = _shared_fixture_groups(items)
    for item in items:
        if "run_godot" in item.fixturenames:
            item.add_marker(pytest.mark.godot)
        group = _scene_group(item)
        for fixture in item.fixturenames:
            group = fixture_groups.get((item.path, fixture), group)
        if group and not item.get_closest_marker("xdist_group"):    # a module may have chosen its own
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
