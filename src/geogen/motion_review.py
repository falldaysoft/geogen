"""Motion review (geogen.103.1): scripted scenarios recorded for people to look at.

A scenario (``kind: motion_scenario``, ``assets/motions/scenarios/*.yaml``) is a small scene, one
person placed in it, and a script of what happens when (motions started, interrupted). Running it
steps ``motion.MotionRig`` at a fixed rate, so it plays the same every time and can be watched at
any speed. Each run writes, under ``<out>/<scenario>/``:

- ``video.mp4``: every frame rendered offline, with markers on the targets (hands orange, gaze
  blue) and the time, beats and channel owners printed in the corner;
- ``filmstrip.png``: a frame at every beat change, interruption and emit, labelled;
- ``timeline.svg``: one lane per channel (which motion held it, in which beat) and the events;
- ``metrics.json`` and ``trace.json``: the numbers below and the rig's trace.

``<out>/index.html`` gathers every scenario run into one page: video with speed control, its
timeline with a playhead that follows the video, the filmstrip and the numbers, and the previous
run of the same scenario beside the new one (moved to ``<out>/_previous/`` before a re-run).

Metrics (``metrics``): ``contact_error_mm`` (hands and feet from their targets once a beat has
arrived), ``foot_slide_mm`` (planted feet), ``peak_joint_speed`` (deg/s, with bone and time: a pop
from a bad blend is a spike), ``pops`` (frames over ``POP_SPEED``), ``settle_s`` (after each
interruption, until no joint turns faster than ``SETTLE_SPEED``), ``peak_joint_accel`` (deg/s²:
a joint whose speed swings frame to frame reads as jerky), ``cloth_mm`` (how deep the forearms
and hands go inside a draped skirt; negative is clear).

    python -m geogen.motion_review sit_on_chair sit_interrupted -o /tmp/motion_review
    python -m geogen.motion_review all --no-video        # numbers only (tests use run_scenario)
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import roots
from .core.node import SceneNode
from .core.transform import Transform
from .motion import MotionError, MotionRig, _placement, _yaw_of, contact, load_motion

SCENARIO_KIND = "motion_scenario"
SCENARIOS_DIR = "motions/scenarios"
SCENARIO_KEYS = {"kind", "name", "description", "scene", "actor", "script", "duration", "until", "camera", "fps"}
POP_SPEED = 900.0        # deg/s a joint turning faster than reads as a pop
SETTLE_SPEED = 60.0      # deg/s: settled
COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948", "#9c755f"]


@dataclass
class Result:
    name: str
    rig: MotionRig
    metrics: dict
    frames: list = field(default_factory=list)      # rendered PIL images (when recording)
    scenario: dict = field(default_factory=dict)


# --- scenarios -----------------------------------------------------------------------------------

def scenario_names(assets_dir: Path | None = None) -> list[str]:
    return sorted(p.stem for p in roots.glob(f"{SCENARIOS_DIR}/*.yaml", assets_dir or roots.CORE))


def load_scenario(name: str | Path, assets_dir: Path | None = None) -> dict:
    from .layout.yaml_utils import safe_load_path

    path = Path(name)
    if path.suffix != ".yaml":
        path = roots.find(f"{SCENARIOS_DIR}/{name}.yaml", assets_dir or roots.CORE)
    spec = safe_load_path(path)
    if not isinstance(spec, dict) or spec.get("kind") != SCENARIO_KIND:
        raise MotionError(f"{path}: expected kind: {SCENARIO_KIND}")
    unknown = set(spec) - SCENARIO_KEYS
    if unknown:
        raise MotionError(f"{path}: unknown keys {sorted(unknown)} (known: {sorted(SCENARIO_KEYS)})")
    spec.setdefault("name", path.stem)
    return spec


def _scene(spec, assets_dir: Path) -> SceneNode:
    from .layout.composer import SceneComposer
    import yaml

    composer = SceneComposer()
    if isinstance(spec, dict):
        return composer.compose_string(yaml.safe_dump({"name": "stage", **spec}, sort_keys=False))
    return composer.compose(roots.find(f"scenes/{spec}.yaml", assets_dir))


def _actor(spec: dict, scene: SceneNode, assets_dir: Path) -> SceneNode:
    from .layout.affordance_qa import _load_body
    from .layout.loader import LayoutLoader

    body = _load_body({k: v for k, v in spec.items() if k in ("archetype", "seed", "asset", "params")}
                      or None, LayoutLoader(), assets_dir)
    actor = body.instance(share_clips=True)
    actor.name = spec.get("name", "actor")
    m = np.eye(4)
    if "at" in spec:
        from . import refs

        m = contact(refs.resolve(scene, spec["at"]), spec.get("grip", "sit")).world()
        m = _placement(m[:3, 3], _yaw_of(m))
    m = m @ Transform(translation=np.asarray(spec.get("offset", [0, 0, 0]), dtype=float)).to_matrix()
    facing = spec.get("facing", "away")
    turn = 0.0 if facing == "away" else math.pi if facing == "toward" else math.radians(float(facing))
    floor = float(spec.get("floor", 0.0))      # the person stands on the floor, not at the contact's height
    actor.transform = Transform.from_matrix(_placement([m[0, 3], floor, m[2, 3]], _yaw_of(m) + turn))
    scene.add_child(actor)
    return actor


def run_scenario(name: str | Path | dict, assets_dir: Path | None = None, record: bool = False,
                 resolution: tuple[int, int] = (800, 600)) -> Result:
    """Run a scenario to its end; with ``record`` render every frame."""
    assets_dir = Path(assets_dir or roots.CORE)
    spec = name if isinstance(name, dict) else load_scenario(name, assets_dir)
    scene = _scene(spec["scene"], assets_dir)
    actor = _actor(spec.get("actor") or {}, scene, assets_dir)
    rig = MotionRig(actor, scene, fps=int(spec.get("fps", 30)), assets_dir=assets_dir)
    script = sorted((dict(e) for e in spec.get("script") or []), key=lambda e: float(e.get("at", 0)))
    limit = float(spec.get("duration", 20.0))
    until = spec.get("until", "idle")
    recorder = _Recorder(scene, actor, spec.get("camera") or {}, resolution,
                         _carried(scene, script, assets_dir)) if record else None
    cloth = _ClothProbe.of(actor)
    frames = []
    if recorder:
        frames.append(recorder.frame(rig))
    while rig.time < limit - 1e-9:
        while script and float(script[0].get("at", 0)) <= rig.time + 1e-9:
            _do(rig, script.pop(0))
        rig.step()
        if cloth is not None:
            rig.samples[-1]["cloth"] = cloth.depth()
        if recorder:
            frames.append(recorder.frame(rig))
        if until == "idle" and not script and not rig.running():
            break
    return Result(spec["name"], rig, metrics(rig), frames, spec)


def _carried(scene: SceneNode, script: list[dict], assets_dir: Path) -> list[SceneNode]:
    """The things the script's motions take in hand (a role some beat ``attach``es): they move, so
    a recording draws them every frame instead of with the static scene."""
    from . import refs

    found = []
    for entry in script:
        if "start" not in entry:
            continue
        motion = load_motion(entry["start"], assets_dir)
        roles = {r for beat in motion["beats"].values() for r in ((beat or {}).get("attach") or {})}
        for role in roles:
            if isinstance(entry.get(role), str):
                node = refs.resolve(scene, entry[role])
                if node not in found:
                    found.append(node)
    return found


class _ClothProbe:
    """How deep the forearms and hands are inside the clothes' draped shells (a skirt), per frame,
    as the build's arm clearance (``arms.clear_arms``) and clipping QA's ``hands`` measure it."""

    def __init__(self, body: SceneNode, cloth: SceneNode, shells: list[dict]) -> None:
        from .generators.arms import KEPT_CLEAR
        from .generators.drape import solid_rings

        self.body, self.cloth = body, cloth
        self.rings = [r for r in (solid_rings(cloth.mesh, s["vertices"], s["s_range"]) for s in shells) if len(r) >= 3]
        names = np.array([j.name.removeprefix("Left").removeprefix("Right").startswith(KEPT_CLEAR)
                          for j in body.skin.joints])
        m = body.mesh
        self.rows = np.flatnonzero(np.where(names[m.joints], m.weights, 0.0).sum(axis=1) >= 0.5)

    @classmethod
    def of(cls, actor: SceneNode) -> "_ClothProbe | None":
        nodes = {n.name: n for n in actor.iter_nodes()}
        body, cloth = nodes.get("body"), nodes.get("clothes")
        shells = (actor.meta.get("humanoid") or {}).get("shells") or []
        if body is None or body.skin is None or cloth is None or cloth.skin is None or not shells:
            return None
        probe = cls(body, cloth, shells)
        return probe if probe.rings and len(probe.rows) else None

    def depth(self) -> float:
        from .generators.arms import _Tube

        points = self.body.world_mesh().vertices[self.rows]
        x = self.cloth.world_mesh().vertices
        return float(max(_Tube(rings, x).probe(points)[0].max() for rings in self.rings))


TRACE_EVERY = 5          # frames between the positions a conformance trace holds
TRACE_TOLERANCE = 0.002  # m a runtime rig's bones may be off the reference's in a trace


def motion_trace(name: str, assets_dir: Path | None = None) -> dict:
    """A conformance trace (tests/data/motion/, ``kind: motion_trace``) from a body-only scenario:
    the body's ``rig`` extras, the motions its script starts, the script, and what the reference
    rig did with them: every bone's position in the body's frame every ``TRACE_EVERY`` frames and
    the event sequence. A runtime rig replays the script on the same data and must match
    (runtime/godot/scripts/motion_rig.gd ``run_trace``)."""
    from .motion import load_motion, rig_extras

    assets_dir = Path(assets_dir or roots.CORE)
    spec = load_scenario(name, assets_dir)
    script = sorted((dict(e) for e in spec.get("script") or []), key=lambda e: float(e.get("at", 0)))
    motions = {}
    for entry in script:
        name_ = entry.get("start") or (entry["break"] if isinstance(entry.get("break"), str) else None)
        if name_:
            motion = load_motion(name_, assets_dir)
            motions[motion["name"]] = motion
    scene = _scene(spec["scene"], assets_dir)
    actor = _actor(spec.get("actor") or {}, scene, assets_dir)
    fps = int(spec.get("fps", 30))
    rig = MotionRig(actor, scene, fps=fps, assets_dir=assets_dir)
    limit, pending, frames = float(spec.get("duration", 20.0)), list(script), []
    s = rig.skeleton
    while rig.time < limit - 1e-9:
        while pending and float(pending[0].get("at", 0)) <= rig.time + 1e-9:
            _do(rig, pending.pop(0))
        rig.step()
        if len(rig.samples) % TRACE_EVERY == 0:
            fk = s.fk(rig.out)
            frames.append({"frame": len(rig.samples), "positions": {b: [round(float(v), 5) for v in fk[b][:3, 3]]
                                                                     for b in s.names}})
        if not pending and not rig.running():
            break
    return {"kind": "motion_trace", "name": name, "fps": fps, "frames": len(rig.samples),
            "tolerance": TRACE_TOLERANCE, "rig": rig_extras(actor), "motions": motions, "script": script,
            "expect": {"positions": frames,
                       "events": [[e["event"], e["motion"], e.get("beat")] for e in rig.trace
                                  if e["event"] in ("start", "beat", "loop", "emit", "interrupt", "end",
                                                                 "break", "pause", "resume")]}}


def _do(rig: MotionRig, entry: dict) -> None:
    entry = dict(entry)
    entry.pop("at", None)
    if "start" in entry:
        motion = entry.pop("start")
        rig.start(motion, **entry)
    elif "interrupt" in entry:
        rig.interrupt(entry["interrupt"], entry.get("how", "yield"))
    elif "break" in entry:
        reaction = entry.pop("break")
        rig.break_(reaction if isinstance(reaction, str) else None, **entry)
    else:
        raise MotionError(f"script entry {entry!r}: expected start: or interrupt:")


# --- metrics -------------------------------------------------------------------------------------

def metrics(rig: MotionRig) -> dict:
    samples = rig.samples
    out: dict = {"duration_s": round(rig.time, 3), "frames": len(samples)}
    contact = 0.0
    for s in samples:
        for key, goal in s["aims"].items():
            if key.endswith("@arrived"):
                end = key.split("@")[0]
                contact = max(contact, float(np.linalg.norm(s["ends"][end] - goal)))
    out["contact_error_mm"] = round(contact * 1000, 1)
    slide, anchor = 0.0, {}
    for s in samples:
        for foot in ("LeftFoot", "RightFoot"):
            if foot in s["planted"]:
                anchor.setdefault(foot, s["ends"][foot])
                slide = max(slide, float(np.linalg.norm(s["ends"][foot] - anchor[foot])))
            else:
                anchor.pop(foot, None)
    out["foot_slide_mm"] = round(slide * 1000, 1)
    peak, pops, speeds = (0.0, None, 0.0), 0, []
    accel, last = (0.0, None, 0.0), {}
    for a, b in zip(samples, samples[1:]):
        dt = b["t"] - a["t"]
        fastest, bone = 0.0, None
        for name, q in b["rotations"].items():
            p = a["rotations"].get(name)
            if p is None:
                continue
            ang = math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(p, q)))))) / dt
            if ang > fastest:
                fastest, bone = ang, name
            if name in last and abs(ang - last[name]) / dt > accel[0]:
                accel = (abs(ang - last[name]) / dt, name, b["t"])
            last[name] = ang
        speeds.append((b["t"], fastest))
        if fastest > peak[0]:
            peak = (fastest, bone, b["t"])
        if fastest > POP_SPEED:
            pops += 1
    out["peak_joint_speed"] = {"deg_s": round(peak[0], 1), "bone": peak[1], "t": peak[2]}
    out["pops"] = pops
    # A joint whose speed swings from frame to frame reads as jerky even when it is never fast.
    out["peak_joint_accel"] = {"deg_s2": round(accel[0]), "bone": accel[1], "t": accel[2]}
    depths = [(s["cloth"], s["t"]) for s in samples if "cloth" in s]
    if depths:
        deepest = max(depths)
        out["cloth_mm"] = {"mm": round(deepest[0] * 1000, 1), "t": deepest[1]}
    settle = []
    for e in rig.trace:
        if e["event"] != "interrupt":
            continue
        after = [(t, v) for t, v in speeds if t > e["t"]]
        calm = next((t for t, v in after if v < SETTLE_SPEED and t > e["t"] + 0.2), None)
        settle.append(round((calm or rig.time) - e["t"], 3))
    out["settle_s"] = settle
    out["beats"] = [f"{e['motion']}.{e['beat']}" for e in rig.trace if e["event"] == "beat"]
    out["events"] = [{k: e[k] for k in e if k in ("t", "event", "motion", "beat", "emit", "policy")}
                     for e in rig.trace if e["event"] in ("interrupt", "emit", "end", "start")]
    return out


# --- recording -----------------------------------------------------------------------------------

class _Recorder:
    """Renders the static scene once and the person, what they carry (and markers) per frame, from a
    fixed camera."""

    def __init__(self, scene: SceneNode, actor: SceneNode, camera: dict, resolution: tuple[int, int],
                 carried: list[SceneNode] = ()) -> None:
        from .render import RenderOptions, SceneRenderer, _MaterialCache, frame_bounds, scene_bounds

        self.actor = actor
        self.moving = [actor, *carried]
        parents = [(node, node.parent) for node in self.moving]
        for node, parent in parents:
            parent.remove_child(node)
        options = RenderOptions(width=resolution[0], height=resolution[1], zoom=float(camera.get("zoom", 1.0)),
                                fov=float(camera.get("fov", 35.0)))
        self.renderer = SceneRenderer(scene, options)
        for node, parent in parents:
            parent.add_child(node)
        # Frame what the camera names (things and the person where they start), else the whole scene.
        boxes = [scene_bounds(actor)]
        for ref in camera.get("frame") or []:
            from . import refs

            boxes.append(scene_bounds(refs.resolve(scene, ref)) if ref != "actor" else scene_bounds(actor))
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
        margin = float(camera.get("margin", 0.3))
        self.renderer.framed = np.array([lo - margin, hi + margin])
        self.view = camera.get("view", "iso")
        self.pose = self.renderer.camera_pose(self.view)
        self.materials = _MaterialCache()
        self._frame_bounds = frame_bounds

    def frame(self, rig: MotionRig):
        import pyrender
        from PIL import ImageDraw

        from .render import _pyrender_mesh

        scene = self.renderer.scene
        added = []
        for node in (n for moving in self.moving for n in moving.iter_visible()):
            if node.mesh is None or not len(node.mesh.faces):
                continue
            mesh = node.world_mesh()
            added.append(scene.add(_pyrender_mesh(mesh, self.materials)))
        last = rig.samples[-1] if rig.samples else None
        if last:
            for key, point in last["aims"].items():
                if "@" in key:
                    continue
                color = [0.2, 0.5, 1.0, 1.0] if key == "gaze" else [1.0, 0.55, 0.1, 1.0]
                sphere = pyrender.Mesh.from_trimesh(_sphere(0.025), smooth=True,
                                                    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color))
                added.append(scene.add(sphere, pose=Transform(translation=np.asarray(point)).to_matrix()))
        image = self.renderer.render_from(self.pose) if hasattr(self.renderer, "render_from") else self._render()
        for node in added:
            scene.remove_node(node)
        draw = ImageDraw.Draw(image)
        lines = [f"t {rig.time:5.2f} s"]
        lines += [f"{name}: {beat}" for name, beat in (last["beats"].items() if last else [])]
        owners = {}
        for channel, owner in (last["owners"].items() if last else []):
            owners.setdefault(owner or "-", []).append(channel)
        lines += [f"  {owner}: {', '.join(chs)}" for owner, chs in owners.items()]
        font = _font(15)
        draw.rectangle([6, 6, 12 + max(draw.textlength(line, font=font) for line in lines), 12 + 19 * len(lines)],
                       fill=(255, 255, 255))
        for i, line in enumerate(lines):
            draw.text((12, 9 + 19 * i), line, fill=(20, 20, 20), font=font)
        return image

    def _render(self):
        # SceneRenderer.render picks the camera from a view name; keep ours fixed.
        renderer = self.renderer
        original = renderer.camera_pose
        renderer.camera_pose = lambda view="iso": self.pose
        try:
            return renderer.render(self.view)
        finally:
            renderer.camera_pose = original


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _sphere(radius: float):
    import trimesh

    return trimesh.creation.icosphere(subdivisions=2, radius=radius)


# --- writing -------------------------------------------------------------------------------------

def write_run(result: Result, out: Path, video: bool = True) -> Path:
    """Write one scenario's files under ``out/<name>/`` (the last run moved to ``out/_previous``)."""
    folder = out / result.name
    if folder.exists():
        previous = out / "_previous" / result.name
        if previous.exists():
            shutil.rmtree(previous)
        previous.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(previous))
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(json.dumps(result.metrics, indent=1))
    (folder / "trace.json").write_text(json.dumps(result.rig.trace, indent=1, default=str))
    (folder / "timeline.svg").write_text(timeline_svg(result.rig))
    if result.frames:
        filmstrip(result).save(folder / "filmstrip.png")
        if video:
            _video(result.frames, int(result.scenario.get("fps", 30)), folder / "video.mp4")
    return folder


def _video(frames, fps: int, path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for i, image in enumerate(frames):
            image.save(Path(tmp) / f"{i:05d}.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", f"{tmp}/%05d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart",
                        str(path)], check=True)


def filmstrip(result: Result, columns: int = 4, width: int = 400):
    from PIL import Image, ImageDraw

    rig, frames = result.rig, result.frames
    marks = [(0.0, "start")]
    for e in rig.trace:
        if e["event"] == "beat":
            marks.append((e["t"], f"{e['motion']}.{e['beat']}"))
        elif e["event"] in ("interrupt", "emit"):
            marks.append((e["t"], f"{e['event']} {e.get('emit') or e['motion']}"))
    marks.append((rig.time, "end"))
    marks.sort(key=lambda m: m[0])
    seen, picks = set(), []
    for k, (t, label) in enumerate(marks):
        shown = t
        # A beat shows best halfway through (an instant event, or the start and end, where it happens).
        if label not in ("start", "end") and not label.startswith(("interrupt", "emit")):
            shown = (t + marks[k + 1][0]) / 2 if k + 1 < len(marks) else t
        i = min(int(round(shown * (len(frames) - 1) / max(rig.time, 1e-6))), len(frames) - 1)
        if i not in seen:
            seen.add(i)
            picks.append((i, f"{t:.2f}s {label}"))
    w, h = frames[0].size
    tw, th = width, int(width * h / w)
    rows = (len(picks) + columns - 1) // columns
    sheet = Image.new("RGB", (tw * columns, (th + 24) * rows), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    for k, (i, label) in enumerate(picks):
        x, y = (k % columns) * tw, (k // columns) * (th + 24)
        sheet.paste(frames[i].resize((tw, th)), (x, y + 24))
        draw.text((x + 6, y + 4), label, fill=(20, 20, 20), font=font)
    return sheet


def timeline_svg(rig: MotionRig, width: int = 760) -> str:
    """One lane per channel: who held it in which beat, over time; events on a lane above."""
    samples = rig.samples
    channels = list(rig.skeleton.channels)
    total = max(rig.time, 1e-6)
    left, lane, top = 92, 24, 34
    plot = width - left - 12
    x = lambda t: left + plot * t / total                                      # noqa: E731
    motions = sorted({o for s in samples for o in s["owners"].values() if o})
    color = {m: COLORS[i % len(COLORS)] for i, m in enumerate(motions)}
    height = top + lane * len(channels) + 44
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" class="timeline" '
             f'data-duration="{total:.4f}" data-left="{left}" data-plot="{plot}" font-family="sans-serif" font-size="11">']
    parts.append(f'<rect width="{width}" height="{height}" fill="#fff"/>')
    for e in rig.trace:
        if e["event"] in ("interrupt", "emit", "attach", "release"):
            c = "#e15759" if e["event"] == "interrupt" else "#59a14f"
            label = html.escape(e.get("emit") or f"{e['event']} {e['motion']}" + (f" ({e['policy']})" if e.get("policy") else ""))
            parts.append(f'<line x1="{x(e["t"]):.1f}" x2="{x(e["t"]):.1f}" y1="14" y2="{height - 22}" stroke="{c}" '
                         f'stroke-dasharray="3 3"/><text x="{x(e["t"]) + 3:.1f}" y="12" fill="{c}">{label}</text>')
    for row, channel in enumerate(channels):
        y = top + row * lane
        parts.append(f'<text x="6" y="{y + 15}" fill="#333">{channel}</text>')
        parts.append(f'<rect x="{left}" y="{y + 2}" width="{plot}" height="{lane - 6}" fill="#f2f2f2"/>')
        start, key = 0.0, None
        for s in samples + [None]:
            owner = s["owners"].get(channel) if s else None
            k = (owner, s["beats"].get(owner)) if (s and owner) else None
            if s is None or k != key:
                t = s["t"] if s else total
                if key is not None:
                    w = max(x(t) - x(start), 1.0)
                    parts.append(f'<rect x="{x(start):.1f}" y="{y + 2}" width="{w:.1f}" height="{lane - 6}" '
                                 f'fill="{color[key[0]]}" fill-opacity="0.85"><title>{html.escape(key[0])}.{html.escape(str(key[1]))}'
                                 f' {start:.2f}-{t:.2f}s</title></rect>')
                    if w > 34:
                        parts.append(f'<text x="{x(start) + 3:.1f}" y="{y + 14}" fill="#fff">{html.escape(str(key[1]))}</text>')
                start, key = (s["t"] if s else total), k
    axis = top + lane * len(channels) + 4
    step = 0.5 if total <= 6 else 1.0 if total <= 15 else 2.0
    t = 0.0
    while t <= total + 1e-6:
        parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{axis}" y2="{axis + 4}" stroke="#999"/>'
                     f'<text x="{x(t):.1f}" y="{axis + 15}" fill="#666" text-anchor="middle">{t:g}s</text>')
        t += step
    legend = left
    for m in motions:
        parts.append(f'<rect x="{legend}" y="{height - 9}" width="10" height="8" fill="{color[m]}"/>'
                     f'<text x="{legend + 13}" y="{height - 2}" fill="#333">{html.escape(m)}</text>')
        legend += 20 + 7 * len(m)
    parts.append(f'<line class="playhead" x1="{left}" x2="{left}" y1="14" y2="{axis}" stroke="#000" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def write_index(out: Path) -> Path:
    """The review page over every scenario folder under ``out``."""
    cards = []
    for folder in sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("_")):
        m = json.loads((folder / "metrics.json").read_text()) if (folder / "metrics.json").exists() else {}
        desc = html.escape((m.get("description") or ""))
        previous = out / "_previous" / folder.name
        prev_m = json.loads((previous / "metrics.json").read_text()) if (previous / "metrics.json").exists() else None
        video = f'<video src="{folder.name}/video.mp4" controls muted loop playsinline></video>' \
            if (folder / "video.mp4").exists() else ""
        prev_video = f'<figure><figcaption>previous run</figcaption><video src="_previous/{folder.name}/video.mp4" ' \
                     f'controls muted loop playsinline></video></figure>' if (previous / "video.mp4").exists() else ""
        rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(_fmt(v))}</td>"
                       f"<td class='prev'>{html.escape(_fmt(prev_m.get(k))) if prev_m else ''}</td></tr>"
                       for k, v in m.items() if k not in ("events", "beats", "description", "title"))
        svg = (folder / "timeline.svg").read_text() if (folder / "timeline.svg").exists() else ""
        strip = f'<img src="{folder.name}/filmstrip.png" alt="filmstrip">' if (folder / "filmstrip.png").exists() else ""
        cards.append(f"""<section class="card" id="{folder.name}">
<h2>{html.escape(m.get('title') or folder.name)}</h2><p class="desc">{desc}</p>
<div class="videos"><figure>{video}<div class="speeds">speed <button>0.25</button><button>0.5</button><button class="on">1</button></div></figure>{prev_video}</div>
<div class="tl">{svg}</div>
<details><summary>Filmstrip</summary>{strip}</details>
<table class="metrics"><tr><th></th><th>this run</th><th class='prev'>{'previous' if prev_m else ''}</th></tr>{rows}</table>
<p class="files"><a href="{folder.name}/trace.json">trace.json</a> · <a href="{folder.name}/metrics.json">metrics.json</a></p>
</section>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Motion review</title><style>
:root {{ --bg:#fafafa; --fg:#1d1d1f; --muted:#666; --card:#fff; --line:#ddd; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#161616; --fg:#eee; --muted:#aaa; --card:#222; --line:#444; }} }}
:root[data-theme="dark"] {{ --bg:#161616; --fg:#eee; --muted:#aaa; --card:#222; --line:#444; }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; }}
main {{ max-width:1100px; margin:0 auto; }} h1 {{ font-size:20px; }} h2 {{ font-size:16px; margin:0 0 4px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px; margin:0 0 18px; }}
.desc {{ color:var(--muted); margin:0 0 10px; }} .videos {{ display:flex; gap:10px; flex-wrap:wrap; }}
figure {{ margin:0; flex:1 1 360px; min-width:0; }} video {{ width:100%; border-radius:6px; background:#000; }}
figcaption {{ color:var(--muted); font-size:12px; }}
.speeds button {{ border:1px solid var(--line); background:transparent; color:var(--fg); border-radius:4px; padding:1px 7px; margin-left:3px; cursor:pointer; }}
.speeds button.on {{ background:var(--fg); color:var(--bg); }}
.tl {{ margin:10px 0; overflow-x:auto; }} .tl svg {{ width:100%; min-width:560px; height:auto; border-radius:4px; }}
details img {{ max-width:100%; margin-top:6px; }} table.metrics {{ border-collapse:collapse; margin-top:8px; font-size:13px; }}
.metrics th, .metrics td {{ text-align:left; padding:2px 10px 2px 0; border-bottom:1px solid var(--line); vertical-align:top; }}
.prev {{ color:var(--muted); }} .files {{ font-size:12px; color:var(--muted); }} a {{ color:inherit; }}
</style></head><body><main><h1>Motion review</h1>
<p class="desc">Each card is one scripted scenario (assets/motions/scenarios), stepped by the reference rig at a fixed rate and rendered offline. Orange markers are hand targets, blue the gaze target. The timeline shows which motion held each channel, in which beat; its playhead follows the video.</p>
{''.join(cards)}</main><script>
for (const card of document.querySelectorAll('.card')) {{
  const video = card.querySelector('video'); const svg = card.querySelector('.tl svg');
  for (const b of card.querySelectorAll('.speeds button')) b.onclick = () => {{
    for (const v of card.querySelectorAll('video')) v.playbackRate = parseFloat(b.textContent);
    card.querySelectorAll('.speeds button').forEach(x => x.classList.toggle('on', x === b)); }};
  if (!video || !svg) continue;
  const head = svg.querySelector('.playhead'); const d = +svg.dataset.duration, l = +svg.dataset.left, p = +svg.dataset.plot;
  const move = () => {{ const x = l + p * Math.min(video.currentTime / d, 1); head.setAttribute('x1', x); head.setAttribute('x2', x); requestAnimationFrame(move); }};
  requestAnimationFrame(move);
  svg.addEventListener('click', e => {{ const r = svg.getBoundingClientRect(); const x = (e.clientX - r.left) * svg.viewBox.baseVal.width / r.width;
    video.currentTime = Math.max(0, (x - l) / p * d); }});
}}
</script></body></html>"""
    path = out / "index.html"
    path.write_text(page)
    return path


def _fmt(v) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k} {w}" for k, w in v.items())
    if isinstance(v, list):
        return ", ".join(map(str, v)) or "-"
    return "-" if v is None else str(v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m geogen.motion_review", description=__doc__.split("\n\n")[0])
    parser.add_argument("scenarios", nargs="*", default=["all"], help="scenario names, or all")
    parser.add_argument("-o", "--out", default="motion_review", help="output folder (default: ./motion_review)")
    parser.add_argument("--resolution", default="800x600")
    parser.add_argument("--no-video", action="store_true", help="numbers, trace and timeline only")
    args = parser.parse_args(argv)
    names = scenario_names() if args.scenarios in (["all"], []) else args.scenarios
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = (int(v) for v in args.resolution.lower().split("x"))
    for name in names:
        t0 = time.time()
        result = run_scenario(name, record=not args.no_video, resolution=(w, h))
        spec = result.scenario
        result.metrics = {"title": spec.get("name"), "description": spec.get("description", ""), **result.metrics}
        folder = write_run(result, out, video=not args.no_video)
        m = result.metrics
        cloth = f", cloth {m['cloth_mm']['mm']} mm" if "cloth_mm" in m else ""
        print(f"{name}: {m['duration_s']} s, contact {m['contact_error_mm']} mm, slide {m['foot_slide_mm']} mm, "
              f"peak {m['peak_joint_speed']['deg_s']} deg/s ({m['peak_joint_speed']['bone']}), "
              f"accel {m['peak_joint_accel']['deg_s2']} deg/s2, pops {m['pops']}{cloth}"
              f" -> {folder} ({time.time() - t0:.1f} s)")
    print(write_index(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
