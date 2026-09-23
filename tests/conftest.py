"""Shared fixtures: locating and running the Godot runtime (runtime/godot)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

GODOT_PROJECT = Path(__file__).parent.parent / "runtime" / "godot"


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
    # class_name scripts need the import cache; building it is idempotent.
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
