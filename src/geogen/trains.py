"""Trains as data: consists, timetables, and their precomputed runs (see epic geogen-r65, M4).

``kind: train`` (``assets/trains/*.yaml``)::

    kind: train
    version: 1
    consist:                                # front to back
      - vehicles/loco.yaml
      - { asset: vehicles/carriage.yaml, params: { paint: cream } }
    speed: 20                               # m/s, capped by the line's speed
    accel: 0.6
    decel: 0.9
    timetable: { headway: 240, dwell: 20, offset: 0 }   # seconds of world time
    warning: 20                             # s before a crossing that its barriers go down

Placed on a railway (``place: {commuter: {train: trains/commuter.yaml, railway: main}}``),
the run is simulated once here: the head's position along the line every
``DT`` seconds from departure to arrival (open lines run end to end; loops do
one lap), stopping at every station (the train's middle at the station's
centre) for ``dwell``, plus each level crossing's closed windows (relative to
departure). The runtime (``runtime/godot/scripts/train.gd``) only looks these
up: departure k leaves at ``offset + k * headway`` of world time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.node import SceneNode
from .core.transform import Transform
from .layout.yaml_utils import safe_load_path

TRAIN_KIND = "train"
VERSION = 1
TRAIN_KEYS = {"kind", "version", "consist", "speed", "accel", "decel", "timetable", "warning", "description"}
DT = 0.5
COUPLING = 0.6             # gap between cars (couplers)


def load_train(path) -> dict[str, Any]:
    data = safe_load_path(path) or {}
    if data.get("kind") != TRAIN_KIND or int(data.get("version", 0)) != VERSION:
        raise ValueError(f"{path}: expected kind '{TRAIN_KIND}' version {VERSION}")
    unknown = set(data) - TRAIN_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; known: {sorted(TRAIN_KEYS)}")
    consist = [{"asset": c} if isinstance(c, str) else dict(c) for c in data.get("consist") or []]
    if not consist:
        raise ValueError(f"{path}: consist is empty")
    tt = data.get("timetable") or {}
    return {"definition": Path(path).stem, "consist": consist, "speed": float(data.get("speed", 20.0)),
            "accel": float(data.get("accel", 0.6)), "decel": float(data.get("decel", 0.9)),
            "timetable": {"headway": float(tt.get("headway", 300.0)), "dwell": float(tt.get("dwell", 20.0)),
                          "offset": float(tt.get("offset", 0.0))},
            "warning": float(data.get("warning", 20.0))}


def simulate_run(line_length: float, loop: bool, train_length: float, stations: list[dict], speed: float,
                 accel: float, decel: float, dwell: float) -> tuple[np.ndarray, list[dict]]:
    """Head position every DT from departure to arrival, and the station stops (times).

    Open lines run from the train fully on the line to its head at the end; loops run one
    lap. The train stops with its middle at each station's centre.
    """
    start = 0.0 if loop else train_length
    end = start + line_length if loop else line_length
    stops = []
    for st in stations:
        at = st["s"] + train_length / 2
        if loop and at < start:
            at += line_length
        if start < at < end:
            stops.append((at, st["name"]))
    stops.sort()
    targets = stops + ([] if loop else [(end, "")])
    s, v, t = start, 0.0, 0.0
    samples = [s]
    events = []
    for k, (target, station) in enumerate(targets):
        while target - s > 0.05 or v > 0.3:
            vmax = min(speed, float(np.sqrt(max(2.0 * decel * (target - s), 0.0))))
            v = min(v + accel * DT, vmax) if v < vmax else max(v - decel * DT, vmax)
            s = min(s + v * DT, target)
            t += DT
            samples.append(s)
            if len(samples) > 200_000:
                raise RuntimeError("train run did not finish")
        s, v = target, 0.0
        if k < len(stops):
            events.append({"station": station, "arrive": round(t, 2), "depart": round(t + dwell, 2)})
            for _ in range(int(round(dwell / DT))):
                t += DT
                samples.append(s)
    while loop and s < end - 0.01:     # finish the lap
        v = min(v + accel * DT, speed)
        s = min(s + v * DT, end)
        samples.append(s)
    return np.asarray(samples), events


def closures(run: np.ndarray, train_length: float, crossings: list[dict], warning: float,
             speed: float, line_length: float, loop: bool) -> dict[str, list[list[float]]]:
    """Per crossing, the [t0, t1] windows (from departure) its barriers are down: from ``warning``
    seconds before the head reaches it until the tail has cleared it."""
    out = {}
    for c in crossings:
        s0, s1 = c["s"]
        windows = []
        for lap in ([0.0, line_length] if loop else [0.0]):
            a, b = s0 + lap, s1 + lap
            occupied = (run >= a) & (run - train_length <= b)
            if not occupied.any():
                continue
            idx = np.where(occupied)[0]
            t_in, t_out = idx[0] * DT, idx[-1] * DT
            windows.append([round(max(t_in - warning, 0.0), 2), round(t_out + 2.0, 2)])
        out[c["id"]] = windows
    return out


def place_train(root: SceneNode, name: str, spec: dict[str, Any], load, assets_dir) -> SceneNode:
    """The train placement: ``meta.train`` (definition, railway, run, stops, closures, cars) and
    the consist as children, standing on the line where departure 0 starts."""
    train = load_train(Path(assets_dir) / spec["train"])
    rail_id = spec.get("railway")
    railway = next((n.meta["railway"] for n in root.iter_nodes()
                    if isinstance(n.meta.get("railway"), dict) and (rail_id is None or n.meta["railway"]["id"] == rail_id)),
                   None)
    if railway is None:
        raise ValueError(f"train '{name}': no railway {rail_id!r} in the scene")
    cars, offsets, head = [], [], 0.0
    for k, entry in enumerate(train["consist"]):
        car = load({"asset": entry["asset"], "params": entry.get("params")})
        v = car.meta.get("vehicle") or {}
        if "bogie_centers" not in v:
            raise ValueError(f"train '{name}': {entry['asset']} has no bogies")
        length = float(v["front"] - v["rear"])
        centre = head + float(v["front"])           # distance from the train's front to this car's origin
        offsets.append(round(centre, 3))
        car.name = f"{name}_car_{k + 1}"
        cars.append((car, v))
        head += length + COUPLING
    train_length = head - COUPLING
    speed = min(train["speed"], float(railway.get("speed", train["speed"])))
    run, stops = simulate_run(railway["length"], railway["loop"], train_length, railway["stations"], speed,
                              train["accel"], train["decel"], train["timetable"]["dwell"])
    node = SceneNode(name, tags=["train"])
    node.meta["type"] = "train"
    node.meta["train"] = {
        "definition": train["definition"], "railway": railway["id"], "length": round(train_length, 3),
        "speed": speed, "timetable": train["timetable"], "warning": train["warning"], "dt": DT,
        "run": [round(float(x), 3) for x in run], "duration": round(len(run) * DT, 2), "stops": stops,
        "closures": closures(run, train_length, railway.get("crossings", []), train["warning"], speed,
                             railway["length"], railway["loop"]),
        "cars": [{"node": car.name, "offset": off, "bogies": v["bogie_centers"]}
                 for (car, v), off in zip(cars, offsets)],
    }
    # Stand departure 0 at its start (renders show it there; the runtime moves it by the timetable).
    pts = np.asarray(railway["points"], dtype=np.float64)
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]

    def at(d: float) -> np.ndarray:
        d = d % cum[-1] if railway["loop"] else min(max(d, 0.0), cum[-1])
        return np.array([np.interp(d, cum, pts[:, j]) for j in range(3)])

    s_head = float(run[0])
    for (car, v), off in zip(cars, offsets):
        centre = s_head - off
        b0, b1 = v["bogie_centers"][0], v["bogie_centers"][-1]
        front, rear = at(centre + b0), at(centre + b1)
        d = front - rear
        car.transform = Transform(translation=(front + rear) / 2,
                                  rotation=np.array([0.0, float(np.arctan2(d[0], d[2])), 0.0]))
        car.meta["train_car"] = {"train": name}
        node.add_child(car)
    return node
