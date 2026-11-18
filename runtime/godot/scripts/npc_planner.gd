class_name GeogenPlanner
extends RefCounted
## Chains of affordances that fill a need (geogen.101.4): the GDScript port of the reference
## planner, geogen/planner.py, which documents the snapshot it takes and the search. The two
## must agree on the traces in tests/data/planner/ (`--trace=FILE`, tests/test_planner_godot.py).
##
## Deterministic like the reference: costs are whole milliseconds, ties go to the chain whose
## step ids sort first, nothing is random.

const LIMITS := {"depth": 5, "patience": 60.0, "nodes": 2000}
const DEFAULT_SPEED := 1.2
const DEFAULT_DURATION := 5.0


## How much a need at ``level`` matters: 1 - level, and with a curve {urgent, boost} the
## shortfall under `urgent` counts `boost` times more.
static func urgency(level: float, curve: Dictionary = {}) -> float:
    level = clampf(level, 0.0, 1.0)
    var value := 1.0 - level
    if curve.has("urgent"):
        var threshold := float(curve["urgent"])
        if level < threshold:
            value += (float(curve.get("boost", 4.0)) - 1.0) * (threshold - level)
    return value


static func _ms(seconds: float) -> int:
    return int(floor(seconds * 1000.0 + 0.5))


static func _walk_ms(a, b, speed: float) -> int:
    if a == null or b == null:
        return 0
    var dx := float(a[0]) - float(b[0])
    var dy := float(a[1]) - float(b[1])
    var dz := float(a[2]) - float(b[2])
    return _ms(sqrt(dx * dx + dy * dy + dz * dz) / speed)


static func _sorted_strings(values) -> Array:
    var out := []
    for v in values:
        out.append(str(v))
    out.sort_custom(func(x, y): return _less(x, y))
    return out


## Strings in the order Python sorts them (by code point).
static func _less(x: String, y: String) -> bool:
    return x.casecmp_to(y) < 0


## What doing ``a`` needs: [item conditions (each a sorted list of ids: one of these, one unit
## each), states {key: state}].
static func _conditions(a: Dictionary) -> Array:
    var requires: Dictionary = a["requires"] if a.get("requires") is Dictionary else {}
    var items := []
    for spec in [a.get("takes"), requires.get("holding")]:
        if spec is Array and not spec.is_empty():
            var cond := _sorted_strings(spec)
            if not items.has(cond):
                items.append(cond)
    var states := {}
    if requires.get("state") is Dictionary:
        for k in requires["state"]:
            states[str(k)] = str(requires["state"][k])
    return [items, states]


static func _held_covers(conds: Array, holding: Dictionary) -> bool:
    var left := {}
    for k in holding:
        left[str(k)] = int(holding[k])
    for cond in conds:
        var found := false
        for item in cond:
            if int(left.get(item, 0)) > 0:
                left[item] -= 1
                found = true
                break
        if not found:
            return false
    return true


## Whether the actor can do ``steps`` without carrying more than its `hands`.
static func _fits_hands(steps: Array, by_id: Dictionary, actor: Dictionary) -> bool:
    if actor.get("hands") == null:
        return true
    var held := _carried(actor)
    for step in steps:
        held += _hands_change(by_id[step])
        if held > int(actor["hands"]):
            return false
    return true


static func _carried(actor: Dictionary) -> int:
    var held := 0
    if actor.get("holding") is Dictionary:
        for k in actor["holding"]:
            held += int(actor["holding"][k])
    return held


## How many more things the hands hold after doing ``a``.
static func _hands_change(a: Dictionary) -> int:
    var change := 0
    if a.get("gives") is Dictionary:
        for k in a["gives"]:
            change += int(a["gives"][k])
    if a.get("takes") is Array and not (a["takes"] as Array).is_empty():
        change -= 1
    return change


## A plan of an actor with `hands`, with what to commit to now: `run`, the steps up to the one
## that leaves the hands empty, and `waits`, the states it needs that aren't so yet.
static func _commit(found: Dictionary, snapshot: Dictionary) -> Dictionary:
    var actor: Dictionary = snapshot["actor"]
    if found.is_empty() or actor.get("hands") == null:
        return found
    var by_id := {}
    for a in snapshot["affordances"]:
        by_id[str(a["id"])] = a
    var states: Dictionary = snapshot["states"] if snapshot.get("states") is Dictionary else {}
    var held := _carried(actor)
    var run := []
    var made := {}
    var waits := {}
    for step in found["steps"]:
        var a: Dictionary = by_id[step]
        run.append(step)
        var requires: Dictionary = a["requires"] if a.get("requires") is Dictionary else {}
        if requires.get("state") is Dictionary:
            for key in requires["state"]:
                if str(made.get(key, states.get(key, ""))) != str(requires["state"][key]) or not (made.has(key) or states.has(key)):
                    waits[str(key)] = true
        if a.get("sets") is Dictionary:
            for key in a["sets"]:
                made[str(key)] = str(a["sets"][key])
        held += _hands_change(a)
        if held <= 0:
            break
    var out := found.duplicate()
    out["run"] = run
    out["waits"] = _sorted_strings(waits.keys())
    return out


## Item conditions in a fixed order (as the reference sorts tuples of strings).
static func _sort_conditions(conds: Array) -> Array:
    var out := conds.duplicate()
    out.sort_custom(func(x: Array, y: Array):
        for i in mini(x.size(), y.size()):
            if x[i] != y[i]:
                return _less(x[i], y[i])
        return x.size() < y.size())
    return out


static func _node(steps: Array, items: Array, wanted: Dictionary, cost: int, by_id: Dictionary, start, speed: float) -> Dictionary:
    var first: Dictionary = by_id[steps[0]]
    return {"f": cost + _walk_ms(start, first.get("at"), speed), "key": ">".join(PackedStringArray(steps)),
        "steps": steps, "items": _sort_conditions(items), "wanted": wanted, "cost": cost}


## The cheapest chain of affordances that ends with ``goal``: {steps, cost_ms}, or {}.
static func _chain(goal: Dictionary, snapshot: Dictionary, limits: Dictionary) -> Dictionary:
    var actor: Dictionary = snapshot["actor"]
    var speed := float(actor.get("speed", DEFAULT_SPEED))
    var start = actor.get("at")
    var holding: Dictionary = actor["holding"] if actor.get("holding") is Dictionary else {}
    var states: Dictionary = snapshot["states"] if snapshot.get("states") is Dictionary else {}
    var promised: Dictionary = snapshot["promised"] if snapshot.get("promised") is Dictionary else {}
    var usable: Array = (snapshot["affordances"] as Array).filter(func(a): return not a.get("reserved", false))
    usable.sort_custom(func(x, y): return _less(str(x["id"]), str(y["id"])))
    var by_id := {}
    for a in usable:
        by_id[str(a["id"])] = a
    var conds := _conditions(goal)
    var queue: Array = [_node([str(goal["id"])], conds[0], conds[1], _ms(float(goal.get("duration", DEFAULT_DURATION))),
        by_id, start, speed)]
    var expanded := 0
    while not queue.is_empty() and expanded < int(limits["nodes"]):
        var at := 0
        for i in range(1, queue.size()):
            var q: Dictionary = queue[i]
            var b: Dictionary = queue[at]
            if q["f"] < b["f"] or (q["f"] == b["f"] and _less(q["key"], b["key"])):
                at = i
        var node: Dictionary = queue[at]
        queue.remove_at(at)
        var steps: Array = node["steps"]
        var items: Array = node["items"]
        var wanted: Dictionary = node["wanted"]
        var met := _held_covers(items, holding)
        for k in wanted:
            var holds: bool = states.has(k) and str(states[k]) == wanted[k]
            if not holds and not (promised.has(k) and str(promised[k]) == wanted[k]):
                met = false
        if met and _fits_hands(steps, by_id, actor):
            return {"steps": steps, "cost_ms": node["f"]}
        expanded += 1
        if steps.size() >= int(limits["depth"]):
            continue
        var first: Dictionary = by_id[steps[0]]
        for a in usable:
            var id := str(a["id"])
            if steps.has(id):
                continue
            var sets := {}
            if a.get("sets") is Dictionary:
                for k in a["sets"]:
                    sets[str(k)] = str(a["sets"][k])
            var undoes := false
            for k in sets:
                if wanted.has(k) and wanted[k] != sets[k]:
                    undoes = true
            if undoes:
                continue                    # it would undo a state a later step counts on
            var left := items.duplicate()
            var provided := false
            var gives: Dictionary = a["gives"] if a.get("gives") is Dictionary else {}
            for item in _sorted_strings(gives.keys()):
                for _n in int(gives[item]):
                    var match_at := -1
                    for i in left.size():
                        if (left[i] as Array).has(item):
                            match_at = i
                            break
                    if match_at < 0:
                        break
                    left.remove_at(match_at)
                    provided = true
            var still := {}
            for k in wanted:
                if str(sets.get(k, "")) != wanted[k] or not sets.has(k):
                    still[k] = wanted[k]
            if not provided and still.size() == wanted.size():
                continue                    # nothing a later step requires
            var more := _conditions(a)
            var clash := false
            for k in more[1]:
                if still.has(k) and still[k] != more[1][k]:
                    clash = true
            if clash:
                continue                    # it needs a state another step needs otherwise
            still.merge(more[1], true)
            left.append_array(more[0])
            queue.append(_node([id] + steps, left, still, int(node["cost"]) + _ms(float(a.get("duration", DEFAULT_DURATION)))
                + _walk_ms(a.get("at"), first.get("at"), speed), by_id, start, speed))
    return {}


## [what a chain is worth, {need: amount it adds}].
static func _worth(steps: Array, snapshot: Dictionary) -> Array:
    var actor: Dictionary = snapshot["actor"]
    var needs: Dictionary = actor["needs"] if actor.get("needs") is Dictionary else {}
    var weights: Dictionary = actor["urgency"] if actor.get("urgency") is Dictionary else {}
    var levels := {}
    for k in needs:
        levels[str(k)] = float(needs[k])
    var by_id := {}
    for a in snapshot["affordances"]:
        by_id[str(a["id"])] = a
    var worth := 0.0
    var fills := {}
    for step in steps:
        var a: Dictionary = by_id[step]
        var liking := float(a.get("weight", 1.0))
        worth += float(a.get("worth", 0.0)) * liking
        var advertises: Dictionary = a["advertises"] if a.get("advertises") is Dictionary else {}
        for need in _sorted_strings(advertises.keys()):
            if not levels.has(need):
                continue
            var after := clampf(float(levels[need]) + float(advertises[need]), 0.0, 1.0)
            var gain: float = after - float(levels[need])
            levels[need] = after
            var weight: float = float(weights[need]) if weights.has(need) else urgency(float(needs[need]))
            worth += gain * weight * liking
            fills[need] = float(fills.get(need, 0.0)) + gain
    return [worth, fills]


## The best thing for the snapshot's actor to do now: {goal, steps, cost (s), score, fills,
## goals}, or {} when nothing is worth doing.
static func plan(snapshot: Dictionary) -> Dictionary:
    var limits := LIMITS.duplicate()
    if snapshot.get("limits") is Dictionary:
        limits.merge(snapshot["limits"], true)
    var all: Array = (snapshot["affordances"] as Array).duplicate()
    all.sort_custom(func(x, y): return _less(str(x["id"]), str(y["id"])))
    var best := {}
    var goals := 0
    for goal in all:
        var advertises: Dictionary = goal["advertises"] if goal.get("advertises") is Dictionary else {}
        if goal.get("reserved", false) or (advertises.is_empty() and float(goal.get("worth", 0.0)) <= 0.0):
            continue
        var found := _chain(goal, snapshot, limits)
        if found.is_empty():
            continue
        var w := _worth(found["steps"], snapshot)
        if float(w[0]) <= 0.0:
            continue
        goals += 1
        var score: float = float(w[0]) / (int(found["cost_ms"]) / 1000.0 + float(limits["patience"]))
        if best.is_empty() or score > float(best["score"]) \
                or (score == float(best["score"]) and int(found["cost_ms"]) < int(best["cost_ms"])):
            best = {"goal": str(goal["id"]), "steps": found["steps"], "cost_ms": found["cost_ms"],
                "score": score, "fills": w[1]}
    if best.is_empty():
        return {}
    var fills := {}
    for k in best["fills"]:
        fills[k] = _round6(float(best["fills"][k]))
    return _commit({"goal": best["goal"], "steps": best["steps"], "cost": int(best["cost_ms"]) / 1000.0,
        "score": _round6(float(best["score"])), "fills": fills, "goals": goals}, snapshot)


## The chain that ends with affordance ``goal``, whether or not it is worth doing (someone was
## told to): as plan() gives it (score 0, goals 1), or {} when there is no such affordance,
## someone else has it, or nothing leads to it.
static func plan_to(snapshot: Dictionary, goal: String) -> Dictionary:
    var limits := LIMITS.duplicate()
    if snapshot.get("limits") is Dictionary:
        limits.merge(snapshot["limits"], true)
    for a in snapshot["affordances"]:
        if str(a.get("id", "")) != goal:
            continue
        if a.get("reserved", false):
            return {}
        var found := _chain(a, snapshot, limits)
        if found.is_empty():
            return {}
        var fills := {}
        var w := _worth(found["steps"], snapshot)
        for k in w[1]:
            fills[k] = _round6(float(w[1][k]))
        return _commit({"goal": goal, "steps": found["steps"], "cost": int(found["cost_ms"]) / 1000.0,
            "score": 0.0, "fills": fills, "goals": 1}, snapshot)
    return {}


## Six decimal places, halves to even, as Python's round() gives.
static func _round6(value: float) -> float:
    return float("%.6f" % value)


## Replay a planner trace (geogen/planner.py run_trace); the mismatches.
static func run_trace(trace: Dictionary) -> Array[String]:
    var problems: Array[String] = []
    var affordances: Array = (trace["world"]["affordances"] as Array).duplicate(true)
    var states: Dictionary = (trace["world"].get("states", {}) as Dictionary).duplicate(true)
    var actors: Dictionary = (trace["actors"] as Dictionary).duplicate(true)
    var claims := {}
    var steps: Array = trace.get("steps", [])
    for index in steps.size():
        var step: Dictionary = steps[index]
        var where := "step %d" % (index + 1)
        if step.has("plan"):
            var who := str(step["plan"])
            var taken := {}
            for other in claims:
                if other != who:
                    for id in claims[other]:
                        taken[id] = true
            var listed := []
            var promised := {}
            for a in affordances:
                var copy: Dictionary = (a as Dictionary).duplicate()
                copy["reserved"] = taken.has(str(a["id"]))
                listed.append(copy)
                if copy["reserved"] and a.get("sets") is Dictionary:
                    for k in a["sets"]:
                        promised[str(k)] = str(a["sets"][k])
            var asked := {"affordances": listed, "states": states, "promised": promised, "actor": actors[who],
                "limits": trace.get("limits", {})}
            var found := plan_to(asked, str(step["to"])) if step.has("to") else plan(asked)
            claims[who] = found.get("run", found.get("steps", []))
            var expect: Dictionary = step.get("expect", {})
            for key in expect:
                var got = found.get(key, [] if key == "steps" else null)
                if not _same(got, expect[key]):
                    problems.append("%s: %s's plan has %s %s, expected %s" % [where, who, key,
                        JSON.stringify(got), JSON.stringify(expect[key])])
        elif step.has("release"):
            claims.erase(str(step["release"]))
        elif step.has("set"):
            var change: Dictionary = step["set"]
            for key in ["holding", "needs", "urgency", "at"]:
                var values: Dictionary = change.get(key, {})
                for who in values:
                    actors[who][key] = values[who]
            states.merge(change.get("states", {}), true)
            var remove: Array = change.get("remove", [])
            affordances = affordances.filter(func(a): return not remove.has(a["id"]))
            affordances.append_array(change.get("add", []))
        else:
            problems.append("%s: not a step: %s" % [where, JSON.stringify(step)])
    return problems


static func _same(got, want) -> bool:
    if got is Array and want is Array:
        if got.size() != want.size():
            return false
        for i in got.size():
            if not _same(got[i], want[i]):
                return false
        return true
    if got is Dictionary and want is Dictionary:
        if got.size() != want.size():
            return false
        for k in want:
            if not got.has(k) or not _same(got[k], want[k]):
                return false
        return true
    if (got is float or got is int) and (want is float or want is int):
        return float(got) == float(want)
    return typeof(got) == typeof(want) and got == want
