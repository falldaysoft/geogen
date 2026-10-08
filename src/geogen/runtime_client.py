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
from pathlib import Path
from typing import Any, Callable

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

    def subscribe(self, *streams: str) -> None:
        """Start receiving ``streams`` (interaction, npc, room, travel, traffic) from now on:
        events already queued are dropped."""
        self.call("events.subscribe", streams=list(streams))
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
