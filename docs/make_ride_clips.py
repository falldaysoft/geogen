"""Record a short clip of each amusement park ride running in the Godot runtime, and
optionally have a video-capable model critique the motion (geogen.88).

    python docs/make_ride_clips.py OUTDIR [--rides carousel,wheel] [--critique [--model M]]

Each ride is one runtime launch with Godot's movie writer (``--write-movie``, a fixed 30 fps,
so every frame is rendered whatever the machine's speed): the player is put at a viewpoint
looking at the ride with the debug overlay off, the ride's lever is pulled and the world runs
for the ride's cycle. ffmpeg trims the loading frames and writes an H.264 mp4 beside the AVI.
``--critique`` sends each clip to the ``agy`` CLI (Gemini by default: it watches video) with
a prompt about the ride's motion, and writes the answers to ``OUTDIR/critique.md``.

Needs the park exported first:
    python -m geogen.main --project examples/amusement_park --export-catalogue
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / "examples" / "amusement_park" / "generated"
GODOT = os.environ.get("GODOT", "/Applications/Godot.app/Contents/MacOS/Godot")

# Where the player stands and looks (world metres; the park's slots are in its scene), the
# lever to pull, and how long the ride needs to show its cycle.
RIDES = {
    "carousel": {"pos": [-16.5, 0.1, 1.5], "at": [-25.0, 2.2, 0.0], "asset": "carousel", "interaction": "run",
                 "seconds": 32, "what": "a carousel: a round platform with horses on poles under a canopy, "
                 "turning about its centre (it ramps up over 6 s to 3 rpm)"},
    "wheel": {"pos": [0.0, 0.1, 9.0], "at": [0.0, 9.0, 26.0], "asset": "wheel", "interaction": "run",
              "seconds": 28, "what": "a Ferris wheel: eight gondolas hanging from a wheel that turns on an axle "
              "between two A-frames; the gondolas hang free on their own hinges (it ramps up over 8 s to 2 rpm)"},
    "tower": {"pos": [16.0, 0.1, -7.0], "at": [25.0, 10.0, 0.0], "asset": "tower", "interaction": "ride",
              "seconds": 30, "what": "a drop tower: a ring of seats on a car that a motor lifts up a mast, "
              "holds at the top, lets fall free under gravity, then brakes with the motor and lowers to the ground"},
    "swings": {"pos": [11.0, 0.1, -13.0], "at": [19.0, 4.5, -19.0], "asset": "swings", "interaction": "run",
               "seconds": 32, "what": "a swing ride (chair-o-plane): chairs on chains hanging from a rotating top "
               "on a mast; as it spins up (10 s to 7.5 rpm) the chairs should fly outward"},
}
SKIP = 2.0          # s of loading frames trimmed from the front of every clip
FREEZE_CHANGE = 0.03   # mean grey-level change (0-255) below which two frames half a second apart count as the same
FREEZE_SECONDS = 2.0   # that long unchanged is a frozen recording

CRITIQUE = """You are reviewing a short video clip from a game being built with procedural assets and a
physics runtime (Godot with Jolt). The clip shows {what}. The ride is simulated as real rigid bodies on
motorised joints, not a hand-made animation, so what you see is what the physics does.

Watch the whole clip carefully and critique the MOTION of the ride, as a theme-park engineer and an
animator would: speed and acceleration (ramp-up, steady state, slowing), smoothness (jitter, stutter,
sudden jumps, parts lagging behind others), physical plausibility (do hanging parts swing as they should,
does anything clip through anything, float, or move rigidly where it should be loose), timing of the
cycle, direction (do riders and seats face the direction of travel? a carousel's horses go head-first,
swing chairs trail their chains outward and back), and anything that looks wrong or lifeless. Then say
what would make it look more like the real
ride. Be specific about what you saw and when (seconds into the clip). If the ride does not move at all,
say so first. Keep it under 250 words, as a bulleted list under two headings: Observed, Suggestions.
Watch the file directly with your own video understanding; do not run shell commands or external tools on it."""


def frozen(clip: Path) -> str:
    """Where the picture stops changing ("" when it never does): on macOS a recording
    occasionally keeps writing the same frame while the world runs on, so a clip that
    freezes is recorded again. Two frames a second are compared; a run of FREEZE_SECONDS
    with less than FREEZE_CHANGE mean grey-level change between them is a freeze (a slow
    ride far away still changes more than that; identical frames re-encoded change less)."""
    import tempfile

    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf", "fps=2", f"{tmp}/f_%04d.png"], check=True)
        frames = sorted(Path(tmp).glob("f_*.png"))
        prev, still = None, 0
        for i, f in enumerate(frames):
            a = np.asarray(Image.open(f).convert("L"), dtype=np.int16)
            if prev is not None:
                still = still + 1 if float(np.abs(a - prev).mean()) < FREEZE_CHANGE else 0
                if still >= FREEZE_SECONDS * 2:
                    return f"{(i - still) / 2:.1f}"
            prev = a
    return ""


def record(name: str, outdir: Path) -> Path:
    for attempt in range(3):
        mp4 = _record(name, outdir)
        at = frozen(mp4)
        if not at:
            return mp4
        print(f"  {name}: the picture froze at {at} s (attempt {attempt + 1}); recording again", file=sys.stderr)
        time.sleep(2)
    raise SystemExit(f"{name}: every recording froze")


def _record(name: str, outdir: Path) -> Path:
    from geogen.runtime_client import run_script, step

    r = RIDES[name]
    avi = outdir / f"{name}.avi"
    steps = [step("hud.overlay", visible=False), step("player.teleport", pos=r["pos"]), step("player.look", at=r["at"]),
             step("interactions.use", asset=r["asset"], interaction=r["interaction"]), step("wait", seconds=r["seconds"])]
    run = run_script(steps, scene="amusement_park", generated=str(GENERATED),
                     engine_args=("--write-movie", str(avi), "--fixed-fps", "30"), godot=GODOT,
                     stop_on_error=True, timeout=900)
    failed = [l for l in run.output.splitlines() if l.startswith("script:") and '"ok":false' in l]
    if failed or not avi.exists():
        raise SystemExit(f"{name}: recording failed: {failed or 'no movie written'}\n{run.output[-2000:]}")
    mp4 = outdir / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(SKIP), "-i", str(avi), "-t", str(r["seconds"] - SKIP),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-an", str(mp4)], check=True)
    avi.unlink()
    return mp4


def critique(name: str, clip: Path, model: str) -> str:
    prompt = (f"Open and watch the video file {clip} ({clip.stat().st_size // 1000} kB, about "
              f"{RIDES[name]['seconds'] - SKIP:.0f} s).\n\n" + CRITIQUE.format(what=RIDES[name]["what"]))
    problem = ""
    for attempt in range(3):            # an empty answer happens when the agent reached for a tool it may not run
        done = subprocess.run(["agy", "--model", model, "--add-dir", str(clip.parent), f"-p={prompt}"],
                              capture_output=True, text=True, timeout=900)
        if done.returncode == 0 and done.stdout.strip():
            return done.stdout.strip()
        problem = (done.stderr or done.stdout).strip()[-600:]
        print(f"  {name}: no critique (attempt {attempt + 1}): {problem}", file=sys.stderr)
    return f"_No critique: {problem}_"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--rides", default=",".join(RIDES), help="comma-separated ride names (default: all)")
    ap.add_argument("--critique", action="store_true", help="ask a video model (agy) to critique each clip")
    ap.add_argument("--model", default="gemini-4-argon-high", help="agy model for --critique")
    ap.add_argument("--no-record", action="store_true", help="critique the clips already in OUTDIR")
    args = ap.parse_args(argv)
    if not (GENERATED / "amusement_park.glb").exists():
        raise SystemExit(f"export the park first: {GENERATED}")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found")
    args.outdir.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.rides.split(",") if n.strip()]
    unknown = [n for n in names if n not in RIDES]
    if unknown:
        raise SystemExit(f"unknown rides {unknown}; rides: {', '.join(RIDES)}")
    for name in names:
        clip = args.outdir / f"{name}.mp4"
        if not args.no_record:
            clip = record(name, args.outdir)
            print(f"recorded {clip}")
        if args.critique:
            if not shutil.which("agy"):
                raise SystemExit("agy not found")
            (args.outdir / f"critique_{name}.md").write_text(f"## {name}\n\n{critique(name, clip, args.model)}\n")
            print(f"critiqued {name}")
    if args.critique:                        # every ride critiqued so far, in the rides' order
        notes = [(args.outdir / f"critique_{n}.md").read_text() for n in RIDES if (args.outdir / f"critique_{n}.md").exists()]
        out = args.outdir / "critique.md"
        out.write_text(f"# Ride motion critique ({args.model})\n\n" + "\n".join(notes))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
