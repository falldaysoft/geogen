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

``--runtime`` also records the scenario in the Godot runtime (``runtime.mp4``, beside the
reference's video on the page): the scene and the person exported, the script played through
the control API (``motion.start`` / ``motion.stop`` / ``motion.break``) from the same viewpoint,
recorded with Godot's movie writer at a fixed 30 fps (``record_runtime``; needs Godot and ffmpeg
and a display, not ``--headless``).

    python -m geogen.motion_review sit_on_chair sit_interrupted -o /tmp/motion_review
    python -m geogen.motion_review all --no-video        # numbers only (tests use run_scenario)
    python -m geogen.motion_review sit_on_chair drink --runtime -o /tmp/motion_review
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
from .fields import known_keys

SCENARIO_KIND = "motion_scenario"
SCENARIOS_DIR = "motions/scenarios"
SCENARIO_KEYS = {"kind", "name", "description", "scene", "actor", "partners", "script", "duration", "until", "camera", "fps"}
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
    partners: dict = field(default_factory=dict)    # name -> the MotionRig of each other person (partners:)


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
    known_keys(spec, SCENARIO_KEYS, f"{path}", MotionError)
    spec.setdefault("name", path.stem)
    return spec


DEMO_ARCHETYPE = "young_woman"


def demo_scenario(motion: str, archetype: str = DEMO_ARCHETYPE, assets_dir: Path | None = None,
                  seed: int = 1) -> dict:
    """A scenario that shows ``motion`` on its own: a body-only one on a rug, one with ``props``
    at a table in front of the person with its props on it at their right hand (a prop with
    ``behind``, a seat, stands that far behind the person instead), each role bound to its prop (the gallery's stage; ``python -m geogen.motion_review --motion NAME``)."""
    spec = load_motion(motion, assets_dir)
    props = spec.get("props") or {}
    roles = spec.get("roles") or {}
    bodies = {r: (e or {}) for r, e in roles.items() if (e or {}).get("body")}
    if bodies:
        return _pair_demo(spec, archetype, seed, bodies)
    missing = [r for r, e in roles.items() if r not in props and not (e or {}).get("optional")]
    if missing and not spec.get("demo"):
        raise MotionError(f"motion '{spec['name']}': no props to stage for its roles {missing}")
    actor = {"archetype": f"characters/archetypes/{archetype}.yaml", "seed": seed, "facing": 0}
    on_table = {r: p for r, p in props.items() if "behind" not in p}
    behind = {r: p for r, p in props.items() if "behind" in p}
    place: dict = {"rug": {"asset": "rug.yaml"}} if not on_table else {
        "table": {"asset": "table.yaml", "params": {"width": 0.9, "depth": 0.5}}}
    slots: dict = {}
    start: dict = {"at": 0.2, "start": spec["name"]}
    for role, prop in behind.items():
        # A seat behind the person, facing the way they do: the person stands in front of it.
        name = f"prop_{role}"
        slots[name] = {"position": [0, 0, 0], "facing": "north"}
        place[name] = {"asset": prop["asset"], "slot": name, **({"params": prop["params"]} if "params" in prop else {})}
        actor.update({"at": name, "offset": [0, 0, float(prop["behind"])], "facing": "away"})
        start[role] = name
    for i, (role, prop) in enumerate(on_table.items()):
        name = f"prop_{role}"
        place[name] = {"asset": prop["asset"], "on": prop.get("on", "table.top"),
                       "at": prop.get("at", {"u": 0.4 - 0.15 * i, "v": 0.75})}
        if "params" in prop:
            place[name]["params"] = prop["params"]
        start[role] = name
    script = [start]
    if spec.get("demo"):
        # Its demonstration's script, roles naming the props staged for them.
        def staged(value):
            base = value.removesuffix("@start") if isinstance(value, str) else None
            return f"prop_{base}{value[len(base):]}" if base in props else value
        script = [{**{k: staged(v) for k, v in e.items() if k != "at"}, "at": 0.2 + float(e.get("at", 0))}
                  for e in spec["demo"]]
    if on_table:
        actor.update({"at": "table", "offset": [0, 0, 0.55], "facing": "toward"})
    if "duration" in (spec.get("params") or {}) or any((b or {}).get("after") == "duration" for b in spec["beats"].values()):
        start.setdefault("duration", 2.5)
    return {"kind": "motion_scenario", "name": f"{spec['name']}__{archetype}" if archetype != DEMO_ARCHETYPE else spec["name"],
            "description": spec.get("summary", ""), "scene": {**({"slots": slots} if slots else {}), "place": place},
            "actor": actor, "script": script, "duration": 40.0,
            "camera": {"view": "iso_back" if on_table else "side" if behind else "iso",
                       "frame": ["table", "actor"] if on_table else [*(f"prop_{r}" for r in behind), "actor"],
                       "margin": 0.25}}


PARTNER_ARCHETYPES = ("young_man", "young_woman")   # who the other person in a two-person demonstration is


def _pair_demo(spec: dict, archetype: str, seed: int, bodies: dict) -> dict:
    """A two-person motion's demonstration: the actor at the origin, each other person (a body
    role, named as the role) where the role's `place` stands them, a side table at the right
    hand of whoever a prop is `by` (the actor unless it names a body role) with the prop on it;
    the motion's `demo` script (entries `by: <role>` for the others), else the motion on the
    actor and on each other person the motion its role `runs`, with this body as theirs."""
    props = spec.get("props") or {}
    partner_archetype = next(a for a in PARTNER_ARCHETYPES if a != archetype)
    actor = {"archetype": f"characters/archetypes/{archetype}.yaml", "seed": seed, "facing": 0}
    partners, frames = {}, {"actor": (np.zeros(3), 0.0)}
    for role, entry in bodies.items():
        place = entry.get("place") or {}
        offset = [float(v) for v in place.get("offset", [0, 0, 1.0])]
        facing = place.get("facing", "toward")
        turn = 0.0 if facing == "away" else math.pi if facing == "toward" else math.radians(float(facing))
        partners[role] = {"archetype": f"characters/archetypes/{partner_archetype}.yaml", "seed": 2,
                          "at": "actor", "offset": offset, "facing": facing}
        frames[role] = (np.array(offset), turn)
    slots, place_spec, tables = {}, {}, {}
    for role, prop in props.items():
        holder = prop.get("by", "actor")
        if holder not in frames:
            raise MotionError(f"motion '{spec['name']}': props.{role}: by '{holder}' is not a body role")
        if holder not in tables:
            # At the holder's right hand, a little ahead (x is to their left).
            origin, yaw = frames[holder]
            c, s_ = math.cos(yaw), math.sin(yaw)
            local = np.array([-0.45, 0.0, 0.5])
            at = origin + np.array([c * local[0] + s_ * local[2], 0.0, -s_ * local[0] + c * local[2]])
            tables[holder] = name = "table" if holder == "actor" else f"table_{holder}"
            # Turned so its things' grips (a plate's rim) face the holder, as they do the actor's.
            from .layout.attachments import CARDINAL_YAW

            want = CARDINAL_YAW["east"] + yaw           # (east: the actor's table, its plate's rim toward them)
            facing = min(CARDINAL_YAW, key=lambda k: abs(math.remainder(CARDINAL_YAW[k] - want, 2 * math.pi)))
            slots[name] = {"position": [round(float(v), 4) for v in at], "facing": facing}
            place_spec[name] = {"asset": "table.yaml", "slot": name, "params": {"width": 0.6, "depth": 0.45}}
        place_spec[f"prop_{role}"] = {"asset": prop["asset"], "on": f"{tables[holder]}.top",
                                      "at": prop.get("at", {"u": 0.5, "v": 0.5}),
                                      **({"params": prop["params"]} if "params" in prop else {})}
    if not place_spec:
        place_spec["rug"] = {"asset": "rug.yaml"}

    def staged(value):
        if not isinstance(value, str):
            return value
        base = value.removesuffix("@start")
        return f"prop_{base}{value[len(base):]}" if base in props else value

    if spec.get("demo"):
        script = [{**{k: staged(v) for k, v in e.items() if k != "at"}, "at": 0.2 + float(e.get("at", 0))}
                  for e in spec["demo"]]
    else:
        start: dict = {"at": 0.2, "start": spec["name"], **{r: r for r in bodies}, **{r: staged(r) for r in props}}
        if "duration" in (spec.get("params") or {}):
            start.setdefault("duration", 2.0)
        script = [start]
        for role, entry in bodies.items():
            if not entry.get("runs"):
                continue
            theirs = load_motion(entry["runs"])
            values = {r: ("actor" if (e or {}).get("body") else staged(r)) for r, e in (theirs.get("roles") or {}).items()
                      if (e or {}).get("body") or r in props}
            script.append({"at": 0.2, "by": role, "start": theirs["name"], **values,
                           **({"duration": start["duration"]} if "duration" in start and "duration" in (theirs.get("params") or {}) else {})})
    name = f"{spec['name']}__{archetype}" if archetype != DEMO_ARCHETYPE else spec["name"]
    return {"kind": "motion_scenario", "name": name, "description": spec.get("summary", ""),
            "scene": {**({"slots": slots} if slots else {}), "place": place_spec},
            "actor": actor, "partners": partners, "script": script, "duration": 40.0,
            "camera": {"view": "side", "frame": ["actor", *partners], "margin": 0.25}}


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
                 resolution: tuple[int, int] = (800, 600), actor: dict | None = None,
                 probe=None) -> Result:
    """Run a scenario to its end; with ``record`` render every frame. ``actor`` replaces keys of
    the scenario's actor (another archetype or seed: ``motion_qa``); ``probe(rig)`` is called
    before the first step and after every step (``rig.samples`` is empty the first time)."""
    assets_dir = Path(assets_dir or roots.CORE)
    spec = name if isinstance(name, dict) else load_scenario(name, assets_dir)
    actor_spec = {**(spec.get("actor") or {}), **(actor or {})}
    if actor and "archetype" in actor and "params" not in actor:
        actor_spec.pop("params", None)       # the scenario's params were for its own person
    scene = _scene(spec["scene"], assets_dir)
    actor = _actor(actor_spec, scene, assets_dir)
    rig = MotionRig(actor, scene, fps=int(spec.get("fps", 30)), assets_dir=assets_dir)
    # Other people (two-person motions): each a body of its own, placed after the actor (so one
    # can stand `at: actor`), its script entries marked `by: <name>`.
    partners = {}
    for pname, pspec in (spec.get("partners") or {}).items():
        body = _actor({**pspec, "name": pname}, scene, assets_dir)
        partners[pname] = MotionRig(body, scene, fps=int(spec.get("fps", 30)), assets_dir=assets_dir)
    rigs = {"actor": rig, **partners}
    script = sorted((dict(e) for e in spec.get("script") or []), key=lambda e: float(e.get("at", 0)))
    limit = float(spec.get("duration", 20.0))
    until = spec.get("until", "idle")
    recorder = _Recorder(scene, actor, spec.get("camera") or {}, resolution,
                         _carried(scene, script, assets_dir),
                         [r.actor for r in partners.values()]) if record else None
    cloth = _ClothProbe.of(actor)
    origins = _origins(scene, script)
    frames = []
    if recorder:
        frames.append(recorder.frame(rig, partners))
    if probe is not None:
        probe(rig)
    while rig.time < limit - 1e-9:
        while script and float(script[0].get("at", 0)) <= rig.time + 1e-9:
            entry = script.pop(0)
            by = entry.pop("by", "actor")
            if by not in rigs:
                raise MotionError(f"script entry {entry!r}: by '{by}' is not the actor or one of the partners")
            _do(rigs[by], {k: rigs.get(v, v) if isinstance(v, str) else v for k, v in entry.items()}, origins)
        for each in rigs.values():
            each.step()
        if cloth is not None:
            rig.samples[-1]["cloth"] = cloth.depth()
        if probe is not None:
            probe(rig)
        if recorder:
            frames.append(recorder.frame(rig, partners))
        if until == "idle" and not script and not any(r.running() for r in rigs.values()):
            break
    result = Result(spec["name"], rig, metrics(rig), frames, spec, partners)
    for pname, prig in partners.items():
        result.metrics.setdefault("partners", {})[pname] = metrics(prig)
    return result


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
        roles -= {r for r, e in (motion.get("roles") or {}).items() if (e or {}).get("body")}
        for role in roles:
            if isinstance(entry.get(role), str) and not entry[role].endswith("@start"):
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
    """A conformance trace (tests/data/motion/, ``kind: motion_trace``) from a scenario: the body's
    ``rig`` extras, its transition clips and pose correctives (``motion.transition_data``, when a
    motion changes posture), the motions its script starts, the script, the things its roles bind (``things``: each one's
    transform in the body's frame and its ``grips``, as the export writes them) and what the
    reference rig did with them: every ``TRACE_EVERY`` frames every bone's position in the body's
    frame, where the body is (``root``, against where it started), its drape correctives
    (``morphs``, with transitions) and where the things are, and the event sequence. A runtime rig
    replays the script on the same data and must match (runtime/godot/scripts/motion_rig.gd
    ``run_trace``). A two-person scenario's other bodies (``partners``) come as ``partners``: each
    one's rig extras, where it stands (``root``, in the actor's frame) and its own ``expect``; its
    script entries keep their ``by``, and a role's value naming a body (``actor``, a partner) binds
    that body's rig."""
    from .motion import grip_points, load_motion, rig_extras, transition_data

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
    partners = {}
    for pname, pspec in (spec.get("partners") or {}).items():
        partners[pname] = MotionRig(_actor({**pspec, "name": pname}, scene, assets_dir), scene, fps=fps, assets_dir=assets_dir)
    rigs = {"actor": rig, **partners}
    things = _trace_things(scene, script, motions)
    body = np.linalg.inv(rig.root)

    def frame_of(node: SceneNode) -> list[float]:
        m = body @ node.world_transform()
        return [round(float(v), 6) for v in m[:3, :].ravel()]

    initial = {key: {"world": frame_of(node), "grips": grip_points(node),
                     **({"affordances": [{"type": a["type"], "position": [round(float(v), 6) for v in a["position"]],
                                          "yaw": round(float(a["yaw"]), 6)} for a in node.meta["affordances"]]}
                        if node.meta.get("affordances") else {})}
               for key, node in things.items()}
    # A posture change plays the body's transition clips: only then does the trace carry them.
    postures = {p for m in motions.values() for p in (m.get("posture") or {}).values()}
    origins = _origins(scene, script)
    limit, pending, frames = float(spec.get("duration", 20.0)), list(script), []
    partner_frames: dict[str, list] = {n: [] for n in partners}
    partner_roots = {n: [round(float(v), 6) for v in (body @ r.root)[:3, :].ravel()] for n, r in partners.items()}
    s = rig.skeleton
    while rig.time < limit - 1e-9:
        while pending and float(pending[0].get("at", 0)) <= rig.time + 1e-9:
            entry = dict(pending.pop(0))
            by = entry.pop("by", "actor")
            _do(rigs[by], {k: rigs.get(v, v) if isinstance(v, str) else v for k, v in entry.items()}, origins)
        for each in rigs.values():
            each.step()
        if len(rig.samples) % TRACE_EVERY == 0:
            for pname, prig in partners.items():
                pfk = prig.skeleton.fk(prig.out)
                partner_frames[pname].append({
                    "frame": len(rig.samples),
                    "positions": {b: [round(float(v), 5) for v in pfk[b][:3, 3]] for b in prig.skeleton.names},
                    "root": [round(float(v), 6) for v in (body @ prig.root)[:3, :].ravel()]})
            fk = s.fk(rig.out)
            frame = {"frame": len(rig.samples), "positions": {b: [round(float(v), 5) for v in fk[b][:3, 3]]
                                                              for b in s.names}}
            frame["root"] = [round(float(v), 6) for v in (body @ rig.root)[:3, :].ravel()]
            if postures and rig.morphs:
                frame["morphs"] = {f"{n}:{k}": round(float(w), 5) for (n, k), w in rig.morphs.items()}
            if things:
                frame["things"] = {key: frame_of(node) for key, node in things.items()}
            frames.append(frame)
        if not pending and not any(r.running() for r in rigs.values()):
            break
    extra = {"partners": {pname: {"rig": rig_extras(prig.actor),
                                  "root": partner_roots[pname],
                                  "expect": {"positions": partner_frames[pname],
                                             "events": [[e["event"], e["motion"], e.get("beat")] for e in prig.trace
                                                        if e["event"] in TRACE_EVENTS]}}
                         for pname, prig in partners.items()}} if partners else {}
    return {"kind": "motion_trace", "name": name, "fps": fps, "frames": len(rig.samples), **extra,
            "tolerance": TRACE_TOLERANCE, "rig": rig_extras(actor),
            **(transition_data(actor, postures) if postures else {}),
            "motions": motions, "script": script,
            **({"things": initial} if things else {}),
            "expect": {"positions": frames,
                       "events": [[e["event"], e["motion"], e.get("beat")] for e in rig.trace
                                  if e["event"] in TRACE_EVENTS]}}


# The events a conformance trace holds a runtime rig to, in order.
TRACE_EVENTS = ("start", "beat", "loop", "emit", "interrupt", "end", "break", "pause", "resume", "attach", "release")


def _trace_things(scene: SceneNode, script: list[dict], motions: dict) -> dict[str, SceneNode]:
    """The scene things a script's role bindings name, by the name a runtime looks them up by: the
    whole value when it is a thing (``cup``), else its head (``table`` of ``table.top``)."""
    from . import refs

    found: dict[str, SceneNode] = {}
    for entry in script:
        motion = motions.get(entry.get("start") or entry.get("break") or "")
        for role, spec in ((motion or {}).get("roles") or {}).items():
            if (spec or {}).get("body"):
                continue                     # another person: a body, not a thing
            value = entry.get(role)
            if not isinstance(value, str):
                continue
            value = value.removesuffix("@start")
            try:
                found.setdefault(value, refs.resolve(scene, value))
            except Exception:
                head = value.rpartition(".")[0]
                found.setdefault(head, refs.resolve(scene, head))
    return found


def _origins(scene: SceneNode, script: list[dict]) -> dict:
    """What a script's ``<thing>@start`` values name: where that thing stood when the script
    began (put a mug back where it stood after another motion carried it), as fixed targets."""
    from . import refs
    from .motion import Target

    out = {}
    for entry in script:
        for value in entry.values():
            if isinstance(value, str) and value.endswith("@start") and value not in out:
                out[value] = Target(None, refs.resolve(scene, value[: -len("@start")]).world_transform().copy(), value)
    return out


def _do(rig: MotionRig, entry: dict, origins: dict | None = None) -> None:
    entry = {k: (origins or {}).get(v, v) if isinstance(v, str) else v for k, v in entry.items()}
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
                 carried: list[SceneNode] = (), others: list[SceneNode] = ()) -> None:
        from .render import RenderOptions, SceneRenderer, _MaterialCache, frame_bounds, scene_bounds

        self.actor = actor
        self.moving = [actor, *others, *carried]
        parents = [(node, node.parent) for node in self.moving]
        for node, parent in parents:
            parent.remove_child(node)
        options = RenderOptions(width=resolution[0], height=resolution[1], zoom=float(camera.get("zoom", 1.0)),
                                fov=float(camera.get("fov", 35.0)))
        self.renderer = SceneRenderer(scene, options)
        for node, parent in parents:
            parent.add_child(node)
        # Frame what the camera names (things and the person where they start), else the whole scene.
        boxes = [scene_bounds(actor), *(scene_bounds(o) for o in others)]
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

    def frame(self, rig: MotionRig, others: dict | None = None):
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
        aims = [(k, v) for r in [rig, *(others or {}).values()] if r.samples for k, v in r.samples[-1]["aims"].items()]
        if last:
            for key, point in aims:
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
        for pname, prig in (others or {}).items():
            plast = prig.samples[-1] if prig.samples else None
            lines += [f"{pname}  {name}: {beat}" for name, beat in (plast["beats"].items() if plast else [])]
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


RUNTIME_LEAD = 0.5      # s the runtime recording shows the scene before the script starts


def record_runtime(result: Result, folder: Path, assets_dir: Path | None = None,
                   resolution: tuple[int, int] = (800, 600), godot: str | None = None) -> Path:
    """Record ``result``'s scenario in the Godot runtime as ``folder/runtime.mp4``: the scene and
    the person exported to a scratch directory (``export_scene`` and ``motions.json``), the
    camera held where the reference's video looks from, the script played through the control
    API at its times (``<thing>@start`` as a fixed place where the thing stood), recorded with
    ``--write-movie`` at a fixed 30 fps and cut to the scenario's length with ffmpeg."""
    from .export import export_scene
    from .motion import write_motions
    from .render import RenderOptions, SceneRenderer, scene_bounds
    from .runtime_client import run_script, step

    assets_dir = Path(assets_dir or roots.CORE)
    spec = result.scenario
    scene = _scene(spec["scene"], assets_dir)
    actor = _actor(spec.get("actor") or {}, scene, assets_dir)
    script = sorted((dict(e) for e in spec.get("script") or []), key=lambda e: float(e.get("at", 0)))
    origins = _origins(scene, script)
    # The camera the reference's video looks from (its frame, view and field of view).
    camera = spec.get("camera") or {}
    actor.parent.remove_child(actor)
    renderer = SceneRenderer(scene, RenderOptions(width=resolution[0], height=resolution[1],
                                                  zoom=float(camera.get("zoom", 1.0)), fov=float(camera.get("fov", 35.0))))
    scene.add_child(actor)
    from . import refs

    boxes = [scene_bounds(actor)] + [scene_bounds(refs.resolve(scene, r) if r != "actor" else actor)
                                     for r in camera.get("frame") or []]
    margin = float(camera.get("margin", 0.3))
    renderer.framed = np.array([np.min([b[0] for b in boxes], axis=0) - margin,
                                np.max([b[1] for b in boxes], axis=0) + margin])
    pose = renderer.camera_pose(camera.get("view", "iso"))
    eye, ahead = pose[:3, 3], -pose[:3, 2]
    motions = {}

    def roles_and_params(entry: dict, motion: dict) -> tuple[dict, dict]:
        roles, params = {}, {}
        for k, v in entry.items():
            if k in ("at", "start", "break", "interrupt", "how"):
                continue
            if isinstance(v, str) and v in origins:
                m = origins[v].world()
                v = {"transform": [round(float(x), 6) for x in m[:3, :].ravel()]}
            (roles if k in (motion.get("roles") or {}) else params)[k] = v
        return roles, params

    steps = [step("hud.overlay", visible=False), step("player.teleport", pos=[60.0, 0.1, 60.0]),
             step("capture.camera", mode="fixed", **{"from": [round(float(v), 4) for v in eye],
                                                      "at": [round(float(v), 4) for v in eye + ahead]},
                  fov=float(camera.get("fov", 35.0))),
             step("wait", seconds=RUNTIME_LEAD)]
    clock = 0.0
    for entry in script:
        at = float(entry.get("at", 0.0))
        if at > clock + 1e-6:
            steps.append(step("wait", seconds=round(at - clock, 4)))
            clock = at
        if "start" in entry:
            motion = motions.setdefault(entry["start"], load_motion(entry["start"], assets_dir))
            roles, params = roles_and_params(entry, motion)
            steps.append(step("motion.start", node=actor.name, motion=entry["start"], roles=roles, params=params))
        elif "interrupt" in entry:
            steps.append(step("motion.stop", node=actor.name, motion=entry["interrupt"], how=entry.get("how", "yield")))
        elif "break" in entry:
            reaction = entry["break"] if isinstance(entry["break"], str) else None
            motion = load_motion(reaction, assets_dir) if reaction else {}
            _, params = roles_and_params(entry, motion)
            steps.append(step("motion.break", node=actor.name, params=params, **({"reaction": reaction} if reaction else {})))
    length = float(result.metrics.get("duration_s") or spec.get("duration", 10.0))
    steps.append(step("wait", seconds=round(max(length - clock, 0.0) + 0.5, 4)))
    name = f"review_{spec['name']}".replace("-", "_")
    with tempfile.TemporaryDirectory(prefix="geogen-review-") as tmp:
        generated = Path(tmp)
        scene.name = name
        export_scene(scene, generated / f"{name}.glb")
        write_motions(generated, assets_dir)
        avi = generated / "runtime.avi"
        run = run_script(steps, scene=name, generated=generated, godot=godot, timeout=600, stop_on_error=False,
                         engine_args=("--write-movie", str(avi), "--fixed-fps", "30",
                                      "--resolution", f"{resolution[0]}x{resolution[1]}"))
        failed = [s for s in run.steps if not s.get("ok")]
        if failed or not avi.exists():
            raise MotionError(f"{spec['name']}: runtime recording failed: {failed or 'no movie written'}")
        mp4 = folder / "runtime.mp4"
        # The movie starts at launch: keep the script's stretch, the last `length` s and a bit.
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", f"-{length + 0.5:.3f}", "-i", str(avi),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-an", str(mp4)], check=True)
    return mp4


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
        if (folder / "runtime.mp4").exists():   # the Godot runtime's recording (--runtime)
            prev_video = f'<figure><figcaption>Godot runtime</figcaption><video src="{folder.name}/runtime.mp4" ' \
                         f'controls muted loop playsinline></video></figure>' + prev_video
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
    parser.add_argument("--list", action="store_true", help="list every motion (the catalogue) and quit")
    parser.add_argument("--json", action="store_true", help="with --list: as JSON")
    parser.add_argument("--motion", action="append", default=[],
                        help="review a motion on its demonstration stage (demo_scenario); repeatable, or all")
    parser.add_argument("--archetype", default=DEMO_ARCHETYPE, help="with --motion: who performs it")
    parser.add_argument("--runtime", action="store_true",
                        help="also record each scenario in the Godot runtime (runtime.mp4; needs Godot and ffmpeg)")
    args = parser.parse_args(argv)
    if args.list:
        from .motion import motion_catalogue

        entries = motion_catalogue()
        if args.json:
            print(json.dumps(entries, indent=1))
        else:
            for e in entries:
                need = f" (with {', '.join(e['roles'])})" if e["roles"] else ""
                print(f"{e['category']:9} {e['name']:18} {e['summary']}{need}")
        return 0
    if args.motion:
        from .motion import motion_names

        motions = motion_names() if args.motion == ["all"] else args.motion
        names = [demo_scenario(m, args.archetype) for m in motions]
    else:
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
        if args.runtime:
            print(f"{name}: runtime recording {record_runtime(result, folder, resolution=(w, h))}")
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
