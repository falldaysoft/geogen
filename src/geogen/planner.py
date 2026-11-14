"""The reference planner: chains of affordances that fill a need (geogen.101.3).

An NPC that is hungry isn't told to go to the fridge, then the stove, then the table: the
fridge's affordance `gives` raw food, the stove's `takes` it and gives a meal, the table's
takes the meal and `advertises` hunger, and the chain is found by search through what the
things in reach say they need and do. Nothing here names a behaviour.

This is the reference; ``runtime/godot/scripts/npc_planner.gd`` is its port, and the traces
in ``tests/data/planner/`` (written by ``make_traces.py`` there) are what both must agree
on, as ``rules.py`` / ``machine.gd`` do over ``tests/data/rules/``.

The input is a plain snapshot, the same structure in every runtime::

    {"affordances": [{"id": "kitchen/stove#0",
                      "at": [x, y, z] | null,          # where one stands to do it (null: anywhere)
                      "duration": 20.0,                # seconds
                      "advertises": {"hunger": 0.6},   # what doing it adds to the needs
                      "worth": 0.0,                    # what doing it is worth besides (a want put right)
                      "requires": {"holding": [ids],   # carrying one of these (kept)
                                   "state": {key: state}},
                      "takes": [ids],                  # carrying one of these (used up)
                      "gives": {id: count},
                      "sets": {key: state},            # what it leaves a thing's state as
                      "weight": 1.0,                   # the asker's liking for it
                      "reserved": false}],             # someone else has it
     "states": {key: state},                           # e.g. "kitchen/stove#burner": "off"
     "actor": {"at": [x, y, z], "speed": 1.2, "holding": {id: count},
               "needs": {need: 0..1}, "urgency": {need: weight}},   # urgency defaults to 1 - need
     "limits": {"depth": 5, "patience": 60.0, "nodes": 2000}}

Search, per affordance that would fill a need (a goal): backward from the goal, A* over
(what is still required, where the chain starts), a step being an affordance that provides
something still required. A chain is complete when the rest holds as the world stands; its
cost is the walking (straight lines between the places, at the actor's speed) plus the
durations. Then the goal is chosen: what the chain adds to the needs, each weighted by its
urgency and the liking for the step, over the chain's time plus ``patience`` (so a short
errand doesn't always beat a meal).

Deterministic by construction, so a port can match it exactly: costs are whole
milliseconds, ties go to the chain whose step ids sort first, and nothing is random.
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any

LIMITS = {"depth": 5, "patience": 60.0, "nodes": 2000}
DEFAULT_SPEED = 1.2
DEFAULT_DURATION = 5.0


class PlanError(ValueError):
    pass


def urgency(level: float, curve: dict[str, Any] | None = None) -> float:
    """How much a need at ``level`` (0 desperate .. 1 satisfied) matters.

    With no curve it is ``1 - level``. ``{"urgent": t, "boost": b}`` makes it steeper below
    the threshold ``t``: the shortfall under ``t`` counts ``b`` times more (default 4), so
    a need that has got low wins over ones that are merely not full.
    """
    level = min(max(float(level), 0.0), 1.0)
    value = 1.0 - level
    if curve and "urgent" in curve:
        threshold = float(curve["urgent"])
        if level < threshold:
            value += (float(curve.get("boost", 4.0)) - 1.0) * (threshold - level)
    return value


def _ms(seconds: float) -> int:
    """Seconds as whole milliseconds, halves up (the same in every language)."""
    return int(math.floor(float(seconds) * 1000.0 + 0.5))


def _walk_ms(a: Any, b: Any, speed: float) -> int:
    if a is None or b is None:
        return 0
    dx, dy, dz = float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])
    return _ms(math.sqrt(dx * dx + dy * dy + dz * dz) / speed)


def _conditions(a: dict[str, Any]) -> tuple[list[tuple[str, ...]], dict[str, str]]:
    """What doing ``a`` needs: item conditions (each "one of these ids", one unit each) and
    states. ``takes`` and ``requires.holding`` naming the same items are one condition."""
    requires = a.get("requires") or {}
    items: list[tuple[str, ...]] = []
    for spec in (a.get("takes"), requires.get("holding")):
        if spec:
            cond = tuple(sorted(str(i) for i in spec))
            if cond not in items:
                items.append(cond)
    return items, {str(k): str(v) for k, v in (requires.get("state") or {}).items()}


def _held_covers(conds: tuple[tuple[str, ...], ...], holding: dict[str, int]) -> bool:
    """Whether the actor carries a unit for every item condition (first fit, in order)."""
    left = {str(k): int(v) for k, v in holding.items()}
    for cond in conds:
        for item in cond:
            if left.get(item, 0) > 0:
                left[item] -= 1
                break
        else:
            return False
    return True


def _chain(goal: dict[str, Any], snapshot: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any] | None:
    """The cheapest chain of affordances that ends with ``goal``; None when there is none."""
    actor = snapshot["actor"]
    speed = float(actor.get("speed", DEFAULT_SPEED))
    start = actor.get("at")
    holding = actor.get("holding") or {}
    states = snapshot.get("states") or {}
    usable = sorted((a for a in snapshot["affordances"] if not a.get("reserved")), key=lambda a: a["id"])
    by_id = {a["id"]: a for a in usable}

    def duration(a: dict[str, Any]) -> int:
        return _ms(a.get("duration", DEFAULT_DURATION))

    def node(steps: tuple[str, ...], items, wanted: dict[str, str], cost: int):
        first = by_id[steps[0]]
        # (f, key): the walk from where the actor stands to the first step is the least still to pay.
        return (cost + _walk_ms(start, first.get("at"), speed), ">".join(steps), steps,
                tuple(sorted(items)), tuple(sorted(wanted.items())), cost)

    items, wanted = _conditions(goal)
    queue = [node((goal["id"],), items, wanted, duration(goal))]
    expanded = 0
    while queue and expanded < int(limits["nodes"]):
        total, _key, steps, items, wanted_items, cost = heapq.heappop(queue)
        wanted = dict(wanted_items)
        if _held_covers(items, holding) and all(states.get(k) == v for k, v in wanted.items()):
            return {"steps": list(steps), "cost_ms": total}
        expanded += 1
        if len(steps) >= int(limits["depth"]):
            continue
        first = by_id[steps[0]]
        for a in usable:
            if a["id"] in steps:
                continue
            sets = {str(k): str(v) for k, v in (a.get("sets") or {}).items()}
            if any(k in wanted and wanted[k] != v for k, v in sets.items()):
                continue                    # it would undo a state a later step counts on
            left = list(items)
            provided = False
            for item in sorted(a.get("gives") or {}):
                for _ in range(int(a["gives"][item])):
                    match = next((c for c in left if item in c), None)
                    if match is None:
                        break
                    left.remove(match)
                    provided = True
            still = {k: v for k, v in wanted.items() if sets.get(k) != v}
            if not provided and len(still) == len(wanted):
                continue                    # nothing a later step requires
            more_items, more_states = _conditions(a)
            if any(k in still and still[k] != v for k, v in more_states.items()):
                continue                    # it needs a state another step needs otherwise
            still.update(more_states)
            heapq.heappush(queue, node((a["id"],) + steps, left + more_items, still,
                                       cost + duration(a) + _walk_ms(a.get("at"), first.get("at"), speed)))
    return None


def _worth(steps: list[str], snapshot: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """What a chain adds to the needs (in order, each need full at 1), weighted by urgency
    and the liking for each step; and the amounts by need."""
    actor = snapshot["actor"]
    levels = {str(k): float(v) for k, v in (actor.get("needs") or {}).items()}
    weights = actor.get("urgency") or {}
    by_id = {a["id"]: a for a in snapshot["affordances"]}
    worth = 0.0
    fills: dict[str, float] = {}
    for step in steps:
        a = by_id[step]
        worth += float(a.get("worth", 0.0)) * float(a.get("weight", 1.0))
        for need in sorted(a.get("advertises") or {}):
            if need not in levels:
                continue
            after = min(max(levels[need] + float(a["advertises"][need]), 0.0), 1.0)
            gain = after - levels[need]
            levels[need] = after
            weight = float(weights[need]) if need in weights else urgency(actor["needs"][need])
            worth += gain * weight * float(a.get("weight", 1.0))
            fills[need] = fills.get(need, 0.0) + gain
    return worth, fills


def plan(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The best thing for the snapshot's actor to do now, as a chain of affordance ids.

    ``{"goal", "steps", "cost" (s), "score", "fills" {need: amount}, "goals" (how many were
    weighed)}``, or ``{}`` when nothing would fill any need. ``steps`` ends with ``goal``.
    """
    limits = {**LIMITS, **(snapshot.get("limits") or {})}
    for a in snapshot["affordances"]:
        if "id" not in a:
            raise PlanError(f"an affordance has no id: {a!r}")
    best: dict[str, Any] = {}
    goals = 0
    for goal in sorted(snapshot["affordances"], key=lambda a: a["id"]):
        if goal.get("reserved") or not (goal.get("advertises") or float(goal.get("worth", 0.0)) > 0.0):
            continue
        found = _chain(goal, snapshot, limits)
        if found is None:
            continue
        worth, fills = _worth(found["steps"], snapshot)
        if worth <= 0.0:
            continue
        goals += 1
        score = worth / (found["cost_ms"] / 1000.0 + float(limits["patience"]))
        # The best score; of equals the quicker chain, then the goal whose id sorts first.
        if not best or (score, -found["cost_ms"]) > (best["score"], -best["cost_ms"]):
            best = {"goal": goal["id"], "steps": found["steps"], "cost_ms": found["cost_ms"],
                    "score": score, "fills": fills}
    if not best:
        return {}
    return {"goal": best["goal"], "steps": best["steps"], "cost": best.pop("cost_ms") / 1000.0,
            "score": round(best["score"], 6), "fills": {k: round(v, 6) for k, v in best["fills"].items()},
            "goals": goals}


def run_trace(trace: dict[str, Any] | str | Path) -> list[str]:
    """Replay a planner trace; the mismatches ([] when it conforms).

    A trace is ``{world: {affordances, states}, actors: {name: actor}, limits, steps}``. A
    step is ``{"plan": actor, "expect": {...}}`` (any of ``goal``, ``steps``, ``cost``,
    ``score``, ``fills``; ``{"steps": []}`` for no plan): a plan found claims its affordances
    for that actor, in place of what the actor claimed before, and the others then plan
    without them. ``{"release": actor}`` gives its claims up. ``{"set": {...}}`` changes
    what there is: ``holding`` / ``needs`` / ``urgency`` / ``at`` ({actor: value}),
    ``states`` ({key: state}), ``remove`` ([affordance ids]) and ``add`` ([affordances]).
    """
    if not isinstance(trace, dict):
        trace = json.loads(Path(trace).read_text())
    trace = json.loads(json.dumps(trace))
    affordances: list[dict[str, Any]] = trace["world"]["affordances"]
    states: dict[str, str] = trace["world"].get("states", {})
    actors: dict[str, dict[str, Any]] = trace["actors"]
    claims: dict[str, list[str]] = {}
    problems: list[str] = []
    for index, step in enumerate(trace["steps"]):
        where = f"step {index + 1}"
        if "plan" in step:
            who = step["plan"]
            taken = {i for other, ids in claims.items() if other != who for i in ids}
            found = plan({"affordances": [{**a, "reserved": a["id"] in taken} for a in affordances],
                          "states": states, "actor": actors[who], "limits": trace.get("limits", {})})
            claims[who] = list(found.get("steps", []))
            for key, want in step.get("expect", {}).items():
                got = found.get(key, [] if key == "steps" else None)
                if got != want:
                    problems.append(f"{where}: {who}'s plan has {key} {got!r}, expected {want!r}")
        elif "release" in step:
            claims.pop(step["release"], None)
        elif "set" in step:
            change = step["set"]
            for key in ("holding", "needs", "urgency", "at"):
                for who, value in change.get(key, {}).items():
                    actors[who][key] = value
            states.update(change.get("states", {}))
            affordances[:] = [a for a in affordances if a["id"] not in change.get("remove", [])]
            affordances.extend(change.get("add", []))
        else:
            problems.append(f"{where}: not a step: {step!r}")
    return problems
