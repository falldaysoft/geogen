"""The quest bot (geogen.15.9): plays a scene's quests with the player's real body, over the
runtime-neutral control protocol (docs/schema/geogen-control.v1.schema.json), and reports any
quest it can't finish. So every quest is a regression test of the scenes it walks through.

For each quest (or those asked for), in the order the runtime lists them:

- **Start.** One that is already active (``start: auto``, or given at load) is played as it is.
  One that is only available is started by the bot (its status set to active, as the
  ``start_quest`` effect would): how a quest is given (a greeting, a dialogue choice, a place)
  is the scene's business, and the bot checks what comes after.
- **Objectives.** The first open, required objective is worked on through its ``goals`` (the
  same the export gives NPC holders, ``quests.goals``): ``enter`` walks into the place;
  ``interaction`` walks up in front of the thing, looks at it and presses E (L first if it is
  locked and a held key fits), again until it shows the goal's ``state``; ``greet`` walks up to
  the NPC, looks at them and presses E, then answers any conversation that opens (an
  ``accept``-like choice first). ``player.walk_to`` opens doors on the way.
- **Report.** ``{ok, quests: {id: {status, started, steps, problem}}}``. A quest ends complete,
  failed (its own rules failed it: a timed dash the bot was too slow for, say), or stopped with a
  ``problem``: an objective with no goals the bot can work on, a goal that didn't take after a
  few tries, a subject in another scene, or a timeout.

Run it on an export::

    python -m geogen.quest_bot quest_yard                  # every quest; exit 1 if one isn't complete
    python -m geogen.quest_bot cottage --quest lights_out --generated DIR

or from Python on a live runtime: ``play(RuntimeClient.launch(...))``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any

STAND_OFF = 0.9          # m in front of a thing to stand to use it (facing it)
GREET_DISTANCE = 1.1     # m from an NPC to greet them from
GREET_TRIES = 3          # goes after a walking NPC this many times
REACH = 1.45             # m from the eyes to a head to greet it (the player's reach, a little less)
TRIES = 3                # attempts at one goal before calling it a problem
NEAR_MARKER = 2.5        # m: a usable thing this near a marked spot is what the objective is about
MAX_ASKED = 6            # people the bot greets to be given an available quest
MARKER_WAIT = 180.0      # s of world time to wait at a marked spot for the objective (a train due)
ACCEPT = ("accept", "yes", "agree", "ok", "sure", "help")


def play(rt, quests: list[str] | None = None, timeout: float = 180.0) -> dict[str, Any]:
    """Play the loaded scene's quests (``quests``: those ids only) on runtime client ``rt``;
    ``timeout`` real seconds a quest. The report is described in the module docstring."""
    out: dict[str, Any] = {"ok": True, "quests": {}}
    for quest in rt.call("quests.list"):
        if quests and quest["id"] not in quests:
            continue
        result = _play_quest(rt, quest, timeout)
        out["quests"][quest["id"]] = result
        out["ok"] = out["ok"] and result["status"] == "complete"
    return out


def _quest(rt, quest_id: str) -> dict[str, Any]:
    """The quest's report now; across a scene switch, once the next scene's rules are running."""
    for _ in range(40):
        found = next((q for q in rt.call("quests.list") if q["id"] == quest_id), None)
        if found is not None:
            return found
        rt.call("wait_ready")
        time.sleep(0.25)
    raise RuntimeError(f"quest '{quest_id}' isn't running here any more")


def _play_quest(rt, quest: dict[str, Any], timeout: float) -> dict[str, Any]:
    result: dict[str, Any] = {"status": quest["status"], "started": quest["status"], "steps": [], "problem": None}
    if quest["status"] == "unavailable":
        result["problem"] = "unavailable"
        return result
    if quest["status"] == "available":
        # As a player would: talk to the people about (a giver may hand something over with it,
        # a key), and only then start it by hand.
        result["started"] = _ask_around(rt, quest["id"], result["steps"])
        if result["started"] is None:
            rt.call("state.set", key=f"actor:{quest['package']}.quest_{quest['id']}", value="active", actor="player")
            result["started"] = "by the bot"
    deadline = time.monotonic() + timeout
    tries: dict[tuple[str, str], int] = {}
    while True:
        scene = rt.call("player.status").get("scene", "")        # it may have travelled
        quest = _quest(rt, quest["id"])
        if quest["status"] in ("complete", "failed"):
            break
        if time.monotonic() > deadline:
            result["problem"] = f"timed out after {timeout:.0f} s"
            break
        objective = next((o for o in quest["objectives"] if o["state"] == "open" and not o["optional"]), None)
        if objective is None:
            result["problem"] = "no open objective, but the quest isn't complete"
            break
        goals = [g for g in objective.get("goals", []) if not _met(rt, g)]
        if not goals and objective.get("markers"):
            # No goal (a condition, say), but the quest marks where to go: a place, or the
            # travel point toward the scene it continues in.
            goals = [{"subject": m["subject"], "marker": m["position"]} for m in objective["markers"]]
        if not goals:
            result["problem"] = f"objective '{objective['id']}' ({objective['text']}): no goal the bot can work on"
            break
        goal = goals[0]
        if str(goal["subject"]).partition(":")[0] != scene and "marker" not in goal:
            # In another scene: the markers (or a travel point here going there) lead toward it.
            here = [m for m in objective.get("markers", []) if str(m["subject"]).partition(":")[0] == scene]
            here += _travel_toward(rt, str(goal["subject"]).partition(":")[0])
            if not here:
                result["problem"] = f"objective '{objective['id']}': {goal['subject']} is in another scene"
                break
            goal = {"subject": here[0]["subject"], "marker": here[0]["position"]}
        key = (objective["id"], str(goal["subject"]))
        tries[key] = tries.get(key, 0) + 1
        if tries[key] > TRIES:
            result["problem"] = f"objective '{objective['id']}': {_describe(goal)} didn't take ({result['steps'][-1]})"
            break
        result["steps"].append(_work(rt, goal, quest["id"], objective["id"]))
        rt.call("time.advance", seconds=0.5)
    result["status"] = quest["status"]
    result["objectives"] = {o["id"]: o["state"] for o in quest["objectives"]}
    return result


def _ask_around(rt, quest_id: str, steps: list) -> str | None:
    """Greet the NPCs here, nearest first (MAX_ASKED of them), answering what they say, until
    the quest is active; who gave it (``by greeting <id>``), or None."""
    asked: set[str] = set()
    for _ in range(MAX_ASKED):
        me = rt.call("player.status")["position"]
        people = sorted((n for n in rt.call("npcs.list") if n.get("uid") and not n.get("away") and n["uid"] not in asked),
                        key=lambda n: math.dist(n["position"], me))
        if not people:
            break
        npc = people[0]
        asked.add(npc["uid"])
        steps.append(_greet(rt, {"subject": npc["uid"], "greet": True}, {"position": npc["position"]}))
        if _quest(rt, quest_id)["status"] == "active":
            return f"by greeting {npc['uid']}"
    return None


def _travel_toward(rt, scene: str) -> list[dict[str, Any]]:
    """This scene's travel points that go to ``scene``, as markers."""
    return [{"subject": n["uid"], "position": n["position"]} for n in rt.call("world.query")
            if (n["extras"].get("travel") or {}).get("scene") == scene and n.get("uid")]


def _describe(goal: dict[str, Any]) -> str:
    if "marker" in goal:
        return f"going to the marker at {goal['subject']}"
    if goal.get("enter"):
        return f"entering {goal['subject']}"
    if goal.get("greet"):
        return f"greeting {goal['subject']}"
    return f"using {goal['subject']}.{goal['interaction']}" + (f" until {goal['state']}" if "state" in goal else "")


def _met(rt, goal: dict[str, Any]) -> bool:
    if "state" not in goal:
        return False
    it = _interaction(rt, goal)
    return it is not None and it["state"] == goal["state"] and it["target"] == goal["state"]


def _interaction(rt, goal: dict[str, Any]) -> dict[str, Any] | None:
    node = rt.call("world.node", uid=goal["subject"])
    return next((i for i in node.get("interactions", []) if i["interaction"] == goal["interaction"]), None)


def _work(rt, goal: dict[str, Any], quest_id: str = "", objective_id: str = "") -> dict[str, Any]:
    """One attempt at a goal; what happened."""
    node = rt.call("world.node", uid=goal["subject"])
    if "marker" in goal:
        return _go_to_marker(rt, goal, node, quest_id, objective_id)
    if goal.get("enter"):
        walked = _walk_up(rt, _inside(node))
        return {"goal": _describe(goal), "walk": walked["status"], "doors": walked.get("doors", [])}
    if goal.get("greet"):
        return _greet(rt, goal, node)
    return _use(rt, goal, node)


def _open(rt, quest_id: str, objective_id: str) -> bool:
    quest = _quest(rt, quest_id)
    return quest["status"] == "active" and any(o["id"] == objective_id and o["state"] == "open"
                                               for o in quest["objectives"])


def _go_to_marker(rt, goal: dict[str, Any], node: dict[str, Any], quest_id: str, objective_id: str) -> dict[str, Any]:
    """Walk to a marked spot. A travel point there is taken (walked into, or used with E), and
    the bot waits for the next scene; otherwise, as a player would, it presses the nearest
    usable thing there (a lever, a button), then waits a while for the objective (a train due)."""
    scene = rt.call("player.status").get("scene", "")
    travel = node.get("extras", {}).get("travel")
    spot = goal["marker"]
    if travel:
        spot = node["position"]
    elif node.get("extras", {}).get("type") == "place":       # a place: into the middle of it
        spot = _inside(node)
    walked = _walk_up(rt, spot)
    step = {"goal": _describe(goal), "walk": walked["status"], "doors": walked.get("doors", [])}
    if travel and travel.get("on") == "use" and rt.call("player.status").get("scene", "") == scene:
        rt.call("player.look", at=[node["position"][0], node["position"][1] + 1.2, node["position"][2]])
        step["pressed"] = rt.call("player.use").get("kind", "") if rt.call("hud.aim")["active"] else "nothing in view"
    if travel:
        for _ in range(60):                               # the fade and the next scene's load
            if rt.call("player.status").get("scene", "") != scene:
                rt.call("wait_ready")
                break
            time.sleep(0.25)
        step["scene"] = rt.call("player.status").get("scene", "")
        return step
    if quest_id and _open(rt, quest_id, objective_id):
        mark = goal["marker"]
        usable = [i for i in rt.call("interactions.list") if i["prompt"] and not i["moving"]
                  and math.dist([i["position"][0], i["position"][2]], [mark[0], mark[2]]) < NEAR_MARKER]
        usable.sort(key=lambda i: math.dist(i["position"], mark))
        for it in usable[:1]:
            thing = rt.call("world.query", name=it["asset"])
            if thing and thing[0].get("uid"):
                step["used"] = _use(rt, {"subject": thing[0]["uid"], "interaction": it["interaction"]},
                                    rt.call("world.node", uid=thing[0]["uid"]))
        waited = 0.0
        while waited < MARKER_WAIT and _open(rt, quest_id, objective_id):
            rt.call("time.advance", seconds=5.0)
            waited += 5.0
        step["waited"] = waited
    return step


def _inside(node: dict[str, Any]) -> list[float]:
    """A floor point inside a place or room: the middle of its box, or of its footprint polygon
    (in the node's frame; a platform along a curved track)."""
    polygon = (node.get("extras", {}).get("place") or {}).get("polygon")
    if polygon:
        from shapely.geometry import Polygon

        p = Polygon(polygon).representative_point()
        yaw = math.radians(node["rotation_deg"][1])
        x, z = p.x * math.cos(yaw) + p.y * math.sin(yaw), -p.x * math.sin(yaw) + p.y * math.cos(yaw)
        return [node["position"][0] + x, node["position"][1], node["position"][2] + z]
    box = node["bounds"]
    return [box["center"][0], box["min"][1], box["center"][2]]


def _walk_up(rt, stand: list[float], seconds: float = 120.0) -> dict[str, Any]:
    """player.walk_to ``stand``, giving up after ``seconds`` of world time. (Sent, not ``call``ed:
    the client's own ``timeout`` is how long it waits for the reply.)"""
    reply = rt.send("player.walk_to", pos=stand, timeout=seconds).get(timeout=seconds * 4 + 60)
    if not reply.get("ok"):
        return {"status": "error", "error": reply.get("error"), "doors": []}
    return reply["result"]


def _use(rt, goal: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    it = next((i for i in node["interactions"] if i["interaction"] == goal["interaction"]), None)
    if it is None:
        return {"goal": _describe(goal), "error": "no such interaction here"}
    target = (it.get("targets") or [{"bounds": node["bounds"]}])[0]["bounds"]["center"]
    # Things face +Z (a switch out of its wall, a door's outside): stand in front of it.
    yaw = math.radians(node["rotation_deg"][1])
    stand = [target[0] + math.sin(yaw) * STAND_OFF, node["position"][1], target[2] + math.cos(yaw) * STAND_OFF]
    walked = _walk_up(rt, stand)
    rt.call("player.look", at=target)
    if it["locked"]:
        rt.call("input.key", key="L")                    # unlock it, if a key the player holds fits
    used = []
    for _ in range(max(len(it["states"]), 1)):
        pressed = rt.call("player.use") if rt.call("hud.aim")["active"] else {}
        used.append(pressed.get("kind", "nothing in view") if isinstance(pressed, dict) else "nothing in view")
        rt.call("time.advance", seconds=1.0)
        now = _interaction(rt, goal)
        if "state" not in goal or (now and now["state"] == goal["state"]):
            break
    return {"goal": _describe(goal), "walk": walked["status"], "doors": walked.get("doors", []), "pressed": used}


def _npc_at(rt, uid: str, what: str = "position") -> list[float] | None:
    """Where the NPC with stable ID ``uid`` is now (its placement node stays where it was put),
    or its ``head``."""
    return next((n.get(what) for n in rt.call("npcs.list") if n.get("uid") == uid), None)


def _greet(rt, goal: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    """Walk up to an NPC and greet them, then answer what they say. A walking NPC may have
    moved on by the time the bot looks: it goes after them again, GREET_TRIES times."""
    walked: dict[str, Any] = {"status": "", "doors": []}
    doors: list = []
    pressed: Any = {}
    for _ in range(GREET_TRIES):
        npc = _npc_at(rt, goal["subject"]) or node["position"]
        walked = _walk_up(rt, _greeting_spot(rt, npc, _npc_at(rt, goal["subject"], "head")))
        doors += walked.get("doors", [])
        npc = _npc_at(rt, goal["subject"]) or npc                     # they may have moved
        head = _npc_at(rt, goal["subject"], "head") or [npc[0], npc[1] + 1.5, npc[2]]
        for drop in (0.1, 0.3, 0.6):                                  # face, chest, middle
            rt.call("player.look", at=[head[0], head[1] - drop, head[2]])
            if rt.call("hud.aim")["focus"].get("kind") == "npc":
                pressed = rt.call("player.use")
                break
        if pressed:
            break
    rt.call("time.advance", seconds=0.5)
    answers = _converse(rt)
    return {"goal": _describe(goal), "walk": walked["status"], "doors": doors,
            "pressed": pressed.get("kind", "no one in view") if isinstance(pressed, dict) else "no one in view",
            "answers": answers}


def _greeting_spot(rt, npc: list[float], head: list[float] | None = None) -> list[float]:
    """Where to stand to greet an NPC at ``npc``: of the spots round their head (a lying NPC's
    is a metre from their hips) on the navmesh, one within reach of it with a clear line from
    the eyes (not the far side of a counter), the shortest walk away."""
    me = rt.call("player.status")["position"]
    head = head or [npc[0], npc[1] + 1.4, npc[2]]
    best, best_length = None, math.inf
    for radius in (0.8, GREET_DISTANCE, 1.4):
        for k in range(8):
            a = k * math.pi / 4
            spot = [head[0] + math.sin(a) * radius, npc[1], head[2] + math.cos(a) * radius]
            path = rt.call("world.nav_path", **{"from": me, "to": spot})
            if not path.get("reached") or not path["points"]:
                continue
            end = path["points"][-1]
            if math.dist([end[0], end[1] + 1.65, end[2]], head) > REACH:
                continue                                  # the navmesh ends out of reach
            hit = rt.call("world.raycast", **{"from": [end[0], end[1] + 1.6, end[2]], "to": head})
            if hit.get("hit") and not hit.get("npc") and math.dist(hit["position"], head) > 0.4:
                continue                                  # something in between
            if path["length"] < best_length:
                best, best_length = end, path["length"]
        if best is not None:
            break
    if best is not None:
        return best
    d = math.hypot(me[0] - npc[0], me[2] - npc[2]) or 1.0
    return [npc[0] + (me[0] - npc[0]) / d * GREET_DISTANCE, npc[1], npc[2] + (me[2] - npc[2]) / d * GREET_DISTANCE]


def _converse(rt) -> list[str]:
    """Answer an open conversation to its end: an accept-like choice where there is one, else
    the first; the choices made."""
    made = []
    for _ in range(12):
        shown = rt.call("dialogue.show")
        if not shown.get("open"):
            break
        choices = shown.get("choices") or []
        if not choices:
            rt.call("dialogue.choose", close=True)
            break
        pick = next((c for c in choices if any(w in (c["id"] + " " + c["text"]).lower() for w in ACCEPT)), choices[0])
        made.append(pick["id"])
        rt.call("dialogue.choose", choice=pick["id"])
    return made


def main(argv: list[str] | None = None) -> int:
    from .runtime_client import RuntimeClient

    parser = argparse.ArgumentParser(prog="python -m geogen.quest_bot", description=__doc__.split("\n\n")[0])
    parser.add_argument("scene", help="the exported scene to load")
    parser.add_argument("--generated", help="exports directory (default runtime/godot/generated)")
    parser.add_argument("--quest", action="append", help="play only this quest (repeatable)")
    parser.add_argument("--timeout", type=float, default=180.0, help="real seconds a quest (default 180)")
    args = parser.parse_args(argv)
    rt = RuntimeClient.launch(args.scene, generated=args.generated, engine_args=("--headless", "--fixed-fps", "60"))
    try:
        rt.call("wait_ready")
        report = play(rt, args.quest, args.timeout)
    finally:
        rt.close()
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
