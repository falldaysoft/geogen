"""Client for the Godot runtime's control API (``--api``, runtime/godot/scripts/api.gd).

The runtime speaks newline-delimited JSON over a localhost TCP socket: requests
``{"id", "cmd", "args"}``, replies ``{"id", "ok", "result" | "error"}``, plus
``{"event", "data"}`` lines for subscribed streams. One long-lived Godot process
can then answer many queries (paths, rooms, interaction states, NPC reports)
instead of one launch per check::

    with RuntimeClient.launch("cottage", generated="runtime/godot/generated") as rt:
        rt.call("wait_ready")
        rt.call("player.walk_to", pos=[0, 0, 2])
        print(rt.call("player.status")["room"])

Command line: start a runtime in the background once, then talk to it (the port is kept in
``STATE_FILE``; ``--port`` talks to one started by hand with ``godot ... -- --api``)::

    python -m geogen.runtime_client --launch cottage            # windowed, so screenshots work
    python -m geogen.runtime_client player.walk_to pos=0,0.3,0.5
    python -m geogen.runtime_client capture.views node=door path=/tmp/door.png
    python -m geogen.runtime_client interactions.set '{"asset": "door", "state": "open", "wait": true}'
    python -m geogen.runtime_client --watch npc,interaction
    python -m geogen.runtime_client --stop

Arguments are one JSON object or ``key=value`` words: values parse as JSON, then as a
comma-separated list of numbers (``pos=1,0,2``), else stay strings.

The same commands run as a script, in a one-shot launch or on a live runtime, with the same
results (docs/schema/geogen-control.v1.schema.json)::

    steps = [step("player.teleport", pos=[0, 0, 6], yaw=0), step("player.move", seconds=1),
             step("player.status")]
    run = run_script(steps, "cottage", generated="runtime/godot/generated")   # --script=...
    run = client.run_script(steps)                                          # over the socket
    run["player.status"]["room"]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

GODOT_PROJECT = Path(__file__).resolve().parents[2] / "runtime" / "godot"
DEFAULT_PORT = 7878
STATE_FILE = Path(tempfile.gettempdir()) / "geogen-runtime.json"


class ApiError(RuntimeError):
    """The runtime answered a command with ``ok: false``."""


def godot_binary() -> str | None:
    """$GODOT, ``godot``/``godot4`` on PATH, or the macOS app bundle."""
    for candidate in (os.environ.get("GODOT"), shutil.which("godot"), shutil.which("godot4"),
                      "/Applications/Godot.app/Contents/MacOS/Godot"):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def step(cmd: str, *, retry: float | None = None, id: str | None = None, **args: Any) -> dict:
    """One script step: ``{cmd, args}``, with ``retry`` (s of world time to keep trying a step
    that fails) and an ``id`` to find its result by."""
    out: dict[str, Any] = {"cmd": cmd}
    if args:
        out["args"] = args
    if retry is not None:
        out["retry"] = retry
    if id is not None:
        out["id"] = id
    return out


class ScriptError(AssertionError):
    """A script step failed (or the run didn't finish)."""


@dataclass
class ScriptRun:
    """What a script did: one entry per step run (``{step, cmd, ok, result | error, id?}``),
    the events it subscribed to, and the runtime's output."""

    steps: list[dict]
    events: list[dict] = field(default_factory=list)
    output: str = ""
    returncode: int = 0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and all(s["ok"] for s in self.steps)

    def entry(self, key: str | int) -> dict:
        """The step with this ``id``, else the last step running this command, else step ``key``."""
        if isinstance(key, int):
            return self.steps[key]
        found = [s for s in self.steps if s.get("id") == key] or [s for s in self.steps if s["cmd"] == key]
        if not found:
            raise KeyError(f"no step {key!r} ran (ran: {[s['cmd'] for s in self.steps]})")
        return found[-1]

    def __getitem__(self, key: str | int) -> Any:
        """A step's result (by id, command or index); raises ScriptError if it failed."""
        entry = self.entry(key)
        if not entry["ok"]:
            raise ScriptError(f"{entry['cmd']}: {entry['error']}")
        return entry["result"]

    def all(self, cmd: str) -> list[Any]:
        """Every result of ``cmd``, in order."""
        return [s.get("result") for s in self.steps if s["cmd"] == cmd]

    def check(self) -> ScriptRun:
        """Raise ScriptError unless every step succeeded; returns self."""
        bad = [s for s in self.steps if not s["ok"]]
        if bad:
            raise ScriptError("; ".join(f"step {s['step']} {s['cmd']}: {s['error']}" for s in bad))
        if self.returncode != 0:
            raise ScriptError(f"the runtime exited {self.returncode}:\n{self.output[-4000:]}")
        return self


def parse_script_output(output: str) -> tuple[list[dict], list[dict], dict | None]:
    """(steps, events, done) from a scripted runtime's output lines."""
    steps, events, done = [], [], None
    for line in output.splitlines():
        if line.startswith("script: "):
            steps.append(json.loads(line.removeprefix("script: ")))
        elif line.startswith("script event: "):
            events.append(json.loads(line.removeprefix("script event: ")))
        elif line.startswith("script done: "):
            done = json.loads(line.removeprefix("script done: "))
    return steps, events, done


def run_script(steps: Iterable[dict], scene: str | None = None, *, generated: str | os.PathLike | None = None,
               start: dict | None = None, args: Iterable[str] = (), engine_args: Iterable[str] = ("--headless",),
               godot: str | None = None, project: str | os.PathLike = GODOT_PROJECT, stop_on_error: bool = True,
               timeout: float = 240.0) -> ScriptRun:
    """Launch the runtime, run ``steps`` (``--script``) and any action flags in ``args``, and
    collect what it reported. ``start`` is how the world starts (the schema's start object: spawn,
    yaw, pitch, time, day_length, timescale, keys, load, set, give, ...). Doesn't raise for a failed
    step: see ``ScriptRun.check``."""
    godot = godot or godot_binary()
    if godot is None:
        raise FileNotFoundError("Godot not found (set $GODOT)")
    user_args = [f"--scene={scene}"] if scene else []
    if generated:
        user_args.append(f"--generated={generated}")
    steps = list(steps)
    with tempfile.NamedTemporaryFile("w", prefix="geogen-script-", suffix=".json", delete=False) as f:
        json.dump({"start": start or {}, "steps": steps, "stop_on_error": stop_on_error}, f)
    try:
        if steps or start:
            user_args.append(f"--script={f.name}")
        process = subprocess.Popen([godot, *engine_args, "--path", str(project), "--", *user_args, *args],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines: list[str] = []
        timer = threading.Timer(timeout, process.kill)
        timer.start()
        try:
            for line in process.stdout:
                lines.append(line)
                if "Parse Error" in line:       # a GDScript parse error leaves Godot running: stop it
                    process.kill()
            returncode = process.wait()
        finally:
            timer.cancel()
    finally:
        os.unlink(f.name)
    output = "".join(lines)
    ran, events, _ = parse_script_output(output)
    return ScriptRun(ran, events, output, returncode)


class RuntimeClient:
    """A connection to a running runtime. ``call`` blocks for the reply; events queue up."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, *, connect_timeout: float = 10.0,
                 process: subprocess.Popen | None = None, log_path: Path | None = None):
        self.host, self.port = host, port
        self.process = process      # set when launch() started the runtime
        self.log_path = log_path
        self._ids = itertools.count(1)
        self._replies: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self.events: queue.Queue = queue.Queue()
        self._sock = self._connect(connect_timeout)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _connect(self, timeout: float) -> socket.socket:
        deadline = time.monotonic() + timeout
        while True:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5)
                sock.settimeout(None)     # replies to slow commands (walks, time.advance) take a while
                return sock
            except OSError:
                if time.monotonic() > deadline:
                    raise
                if self.process is not None and self.process.poll() is not None:
                    raise
                time.sleep(0.1)

    @classmethod
    def launch(cls, scene: str | None = None, *, generated: str | os.PathLike | None = None,
               args: tuple[str, ...] = (), engine_args: tuple[str, ...] = ("--headless",),
               godot: str | None = None, project: str | os.PathLike = GODOT_PROJECT,
               startup_timeout: float = 120.0, detach: bool = False) -> RuntimeClient:
        """Start Godot with ``--api`` on a free port and connect. Output goes to ``log_path``.

        ``engine_args`` without ``--headless`` opens a window (needed for screenshots).
        ``detach`` starts it in its own session, to outlive this process (the CLI's --launch)."""
        godot = godot or godot_binary()
        if godot is None:
            raise FileNotFoundError("Godot not found (set $GODOT)")
        port = _free_port()
        user_args = [f"--api={port}"]
        if scene:
            user_args.append(f"--scene={scene}")
        if generated:
            user_args.append(f"--generated={generated}")
        user_args += list(args)
        log = tempfile.NamedTemporaryFile("w+", prefix="geogen-runtime-", suffix=".log", delete=False)
        process = subprocess.Popen([godot, *engine_args, "--path", str(project), "--", *user_args],
                                   stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=detach)
        deadline = time.monotonic() + startup_timeout
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Godot exited ({process.returncode}) before the API came up:\n"
                                       + Path(log.name).read_text()[-4000:])
                text = Path(log.name).read_text()
                if "api listening:" in text:
                    break
                if "SCRIPT ERROR" in text or "Parse Error" in text:
                    raise RuntimeError("Godot script error:\n" + text[-4000:])
                if time.monotonic() > deadline:
                    raise TimeoutError("the runtime API didn't come up:\n" + text[-4000:])
                time.sleep(0.1)
            return cls(port=port, process=process, log_path=Path(log.name))
        except BaseException:
            process.kill()
            raise

    # -- protocol -----------------------------------------------------------

    def _read(self) -> None:
        buffer = b""
        while True:
            try:
                data = self._sock.recv(65536)
            except OSError:
                data = b""
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                if "event" in msg:
                    self.events.put(msg)
                    continue
                with self._lock:
                    waiter = self._replies.pop(msg.get("id"), None)
                if waiter is not None:
                    waiter.put(msg)
        with self._lock:     # wake every pending call: the connection is gone
            for waiter in self._replies.values():
                waiter.put({"ok": False, "error": "connection closed"})
            self._replies.clear()

    def send(self, cmd: str, **args: Any) -> queue.Queue:
        """Send a command without waiting; the returned queue gets the reply message."""
        msg_id = next(self._ids)
        waiter: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            self._replies[msg_id] = waiter
        self._sock.sendall((json.dumps({"id": msg_id, "cmd": cmd, "args": args}) + "\n").encode())
        return waiter

    def call(self, cmd: str, timeout: float = 120.0, **args: Any) -> Any:
        """Run a command and return its result; raises ApiError when the runtime refuses it."""
        try:
            reply = self.send(cmd, **args).get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"{cmd}: no reply in {timeout} s") from None
        if not reply.get("ok"):
            raise ApiError(f"{cmd}: {reply.get('error')}")
        return reply.get("result")

    def subscribe(self, *streams: str, **filters: Any) -> None:
        """Start receiving ``streams`` (events, interaction, npc, room, place, travel, traffic) from
        now on: events already queued are dropped. ``events`` is the whole event bus; ``types=[...]``
        and ``subjects=[<stable id>, ...]`` narrow it."""
        self.call("events.subscribe", streams=list(streams), **filters)
        self.clear_events()

    def unsubscribe(self, *streams: str) -> None:
        """Stop ``streams`` (default: all) and drop what's queued."""
        self.call("events.unsubscribe", **({"streams": list(streams)} if streams else {}))
        self.clear_events()

    def clear_events(self) -> list[dict]:
        """Take every queued event."""
        taken = []
        while True:
            try:
                taken.append(self.events.get_nowait())
            except queue.Empty:
                return taken

    def wait_event(self, predicate: Callable[[dict], bool] = lambda e: True, timeout: float = 30.0) -> dict:
        """The next event (``{"event", "data"}``) matching ``predicate``; earlier ones are dropped."""
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("no matching event")
            try:
                event = self.events.get(timeout=left)
            except queue.Empty:
                raise TimeoutError("no matching event") from None
            if predicate(event):
                return event

    def run_script(self, steps: Iterable[dict], stop_on_error: bool = True) -> ScriptRun:
        """Run script steps on this runtime, one command at a time, as ``--script`` would."""
        ran = []
        for index, entry in enumerate(steps):
            deadline = float(entry.get("retry", 0.0))
            waited = 0.0
            while True:
                reply = self.send(entry["cmd"], **entry.get("args", {})).get()
                if reply.get("ok") or waited >= deadline:
                    break
                waited += self.call("wait", frames=1)["waited"]
            line = {"step": index, "cmd": entry["cmd"], "ok": bool(reply.get("ok"))}
            if "id" in entry:
                line["id"] = entry["id"]
            line.update({"result": reply.get("result")} if reply.get("ok") else {"error": reply.get("error")})
            ran.append(line)
            if not line["ok"] and stop_on_error:
                break
        return ScriptRun(ran)

    def log(self) -> str:
        """What a launched runtime printed so far."""
        return self.log_path.read_text() if self.log_path else ""

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Disconnect; a launched runtime is asked to quit (killed if it doesn't)."""
        if self.process is not None and self.process.poll() is None:
            try:
                self.call("quit", timeout=5)
            except (OSError, TimeoutError, RuntimeError):
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> RuntimeClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def parse_args(words: list[str]) -> dict:
    """``['{"a": 1}']`` or ``['pos=1,0,2', 'asset=door', 'wait=true']`` -> a dict of arguments."""
    if len(words) == 1 and words[0].lstrip().startswith("{"):
        return json.loads(words[0])
    args = {}
    for word in words:
        key, sep, value = word.partition("=")
        if not sep:
            raise ValueError(f"expected key=value, got {word!r}")
        try:
            args[key] = json.loads(value)
        except json.JSONDecodeError:
            try:
                args[key] = [float(v) for v in value.split(",")] if "," in value else value
            except ValueError:
                args[key] = value
    return args


def _save_png(result: Any) -> Any:
    """Replace base64 PNGs in a CLI result with a file in the temp dir."""
    if isinstance(result, dict) and "png_base64" in result:
        import base64
        path = Path(tempfile.mkstemp(prefix="geogen-shot-", suffix=".png")[1])
        path.write_bytes(base64.b64decode(result.pop("png_base64")))
        result["path"] = str(path)
    return result


def _launch(opts, extra: list[str]) -> int:
    engine_args = ["--headless"] if opts.headless else ["--resolution", opts.resolution]
    client = RuntimeClient.launch(opts.launch, generated=opts.generated, args=tuple(extra),
                                  engine_args=tuple(engine_args), detach=True)
    STATE_FILE.write_text(json.dumps({"port": client.port, "pid": client.process.pid,
                                      "log": str(client.log_path), "scene": opts.launch}))
    info = client.call("wait_ready", timeout=300)
    client.process = None        # leave it running
    client.close()
    print(json.dumps({"port": client.port, "log": str(client.log_path),
                      "scene": info["scene"], "spawns": info["spawns"], "npcs": info["npcs"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description="Talk to a geogen Godot runtime's control API (--api).")
    parser.add_argument("cmd", nargs="?", default="help", help="command, e.g. player.status (default: help)")
    parser.add_argument("args", nargs="*", help="a JSON object, or key=value words")
    parser.add_argument("--port", type=int, help=f"default: the --launch'ed runtime, else {DEFAULT_PORT}")
    parser.add_argument("--launch", metavar="SCENE", help="start a runtime in the background (runtime args after --)")
    parser.add_argument("--generated", help="with --launch: exports directory (default runtime/godot/generated)")
    parser.add_argument("--headless", action="store_true", help="with --launch: no window (no screenshots)")
    parser.add_argument("--resolution", default="1280x720", help="with --launch: window size")
    parser.add_argument("--stop", action="store_true", help="quit the --launch'ed runtime")
    parser.add_argument("--log", action="store_true", help="print the tail of the --launch'ed runtime's output")
    parser.add_argument("--pretty", action="store_true", help="indent the JSON result")
    parser.add_argument("--watch", metavar="STREAMS",
                        help="subscribe to comma-separated streams and print events until Ctrl+C")
    opts = parser.parse_args(argv)
    if opts.launch:
        return _launch(opts, extra)
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    if opts.log:
        print(Path(state["log"]).read_text()[-6000:] if state.get("log") else "no launched runtime")
        return 0
    port = opts.port or state.get("port") or DEFAULT_PORT
    try:
        client = RuntimeClient(port=port, connect_timeout=2)
    except OSError:
        print(f"no runtime API on port {port} (start one with --launch SCENE)", file=sys.stderr)
        return 1
    try:
        if opts.stop:
            client.call("quit", timeout=5)
            STATE_FILE.unlink(missing_ok=True)
            return 0
        if opts.watch:
            client.call("events.subscribe", streams=opts.watch.split(","))
            while True:
                print(json.dumps(client.events.get()), flush=True)
        result = client.call(opts.cmd, timeout=600, **parse_args(opts.args))
        print(json.dumps(_save_png(result), indent=1 if opts.pretty else None))
    except ApiError as e:
        print(e, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
