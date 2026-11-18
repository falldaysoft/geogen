"""Writes the planner's conformance traces beside this file. Run after editing a trace
here: python tests/data/planner/make_traces.py

Each trace is replayed by geogen.planner.run_trace (tests/test_planner.py) and by the Godot
runtime's port (runtime/godot/scripts/npc_planner.gd): both must agree with what the steps
expect. The expectations are written by hand: a trace that fails is a question about the
planner, not something to regenerate.
"""
import json, sys
from pathlib import Path


def aff(id, at, duration, **more):
    return {"id": id, "at": at, "duration": duration, **more}


RAW = ["raw_food"]
# A kitchen along a wall (speed 1 m/s below, so metres are seconds): fridge, microwave, stove;
# a table 3 m out from the stove; a sofa at the far end.
FRIDGE = aff("kitchen/fridge#0", [0, 0, 0], 4, gives={"raw_food": 1})
MICROWAVE = aff("kitchen/microwave#0", [1, 0, 0], 8, takes=RAW, gives={"meal": 1})
STOVE = aff("kitchen/stove#0", [2, 0, 0], 30, takes=RAW, gives={"meal": 1})
TABLE = aff("kitchen/table/chair#0", [2, 0, 3], 60, takes=["meal"], gives={"dirty_dish": 1},
            advertises={"hunger": 0.7})
SINK = aff("kitchen/sink#0", [3, 0, 0], 10, takes=["dirty_dish"], advertises={"order": 0.3})
SOFA = aff("lounge/sofa#0", [8, 0, 3], 40, advertises={"rest": 0.3})
FRUIT = aff("kitchen/fruit_bowl#0", [2, 0, 4], 5, advertises={"hunger": 0.15})


def person(at, hunger=0.2, rest=0.9, **more):
    return {"at": at, "speed": 1.0, "holding": {}, "needs": {"hunger": hunger, "rest": rest}, **more}


traces = {}

traces["kitchen"] = {
    "about": "A hungry person and a kitchen: nothing says cook, the chain fridge > stove > table is found "
             "from what each gives and takes. A microwave added to the room is quicker, so the plan changes; "
             "taken away again, the plan is the stove's.",
    "world": {"affordances": [FRIDGE, STOVE, TABLE, SOFA]},
    "actors": {"ana": person([4, 0, 0])},
    "steps": [
        # walk 4 + fridge 4 + walk 2 + stove 30 + walk 3 + eat 60
        {"plan": "ana", "expect": {"goal": TABLE["id"], "steps": [FRIDGE["id"], STOVE["id"], TABLE["id"]],
                                   "cost": 103.0, "fills": {"hunger": 0.7}}},
        {"set": {"add": [MICROWAVE]}},
        # walk 4 + 4 + walk 1 + 8 + walk sqrt(10) = 3.162 + 60
        {"plan": "ana", "expect": {"steps": [FRIDGE["id"], MICROWAVE["id"], TABLE["id"]], "cost": 80.162}},
        {"set": {"remove": [MICROWAVE["id"]]}},
        {"plan": "ana", "expect": {"steps": [FRIDGE["id"], STOVE["id"], TABLE["id"]]}},
        # Not hungry, a little tired: the sofa.
        {"set": {"needs": {"ana": {"hunger": 1.0, "rest": 0.6}}}},
        {"plan": "ana", "expect": {"goal": SOFA["id"], "steps": [SOFA["id"]], "fills": {"rest": 0.3}}},
        # Wanting nothing, there is nothing to do.
        {"set": {"needs": {"ana": {"hunger": 1.0, "rest": 1.0}}}},
        {"plan": "ana", "expect": {"steps": []}},
    ],
}

traces["directed"] = {
    "about": "Told what to do (plan_to): the chain to a named affordance, wanted or not. Full and rested, "
             "nothing is worth doing, but told to eat she still fetches and cooks first; told to wash up with "
             "no dish, she makes one. What someone else has, or what isn't there, gives no chain.",
    "world": {"affordances": [FRIDGE, STOVE, TABLE, SINK, SOFA]},
    "actors": {"ana": person([4, 0, 0], hunger=1.0, rest=1.0), "ben": person([4, 0, 0])},
    "steps": [
        {"plan": "ana", "expect": {"steps": []}},
        {"plan": "ana", "to": TABLE["id"], "expect": {"goal": TABLE["id"], "steps": [FRIDGE["id"], STOVE["id"], TABLE["id"]],
                                                     "cost": 103.0, "score": 0.0, "fills": {"hunger": 0.0}}},     # full: eating adds nothing
        {"release": "ana"},
        {"plan": "ana", "to": SOFA["id"], "expect": {"steps": [SOFA["id"]]}},
        {"plan": "ana", "to": SINK["id"], "expect": {"steps": [FRIDGE["id"], STOVE["id"], TABLE["id"], SINK["id"]]}},
        {"plan": "ana", "to": "kitchen/kettle#0", "expect": {"steps": []}},
        # Ben is hungry and plans the meal: the stove is his, so Ana, told to eat, can't.
        {"release": "ana"},
        {"plan": "ben", "expect": {"steps": [FRIDGE["id"], STOVE["id"], TABLE["id"]]}},
        {"plan": "ana", "to": TABLE["id"], "expect": {"steps": []}},
    ],
}

traces["missing_ingredient"] = {
    "about": "What is carried shortens the chain, and a chain that can't start isn't a plan: with no fridge "
             "nothing gives raw food, so the meal is out of reach and the fruit bowl is what there is; "
             "carrying raw food the chain starts at the stove; carrying a meal it is the table alone.",
    "world": {"affordances": [STOVE, TABLE, FRUIT]},
    "actors": {"ana": person([4, 0, 0])},
    "steps": [
        {"plan": "ana", "expect": {"goal": FRUIT["id"], "steps": [FRUIT["id"]], "fills": {"hunger": 0.15}}},
        {"set": {"holding": {"ana": {"raw_food": 1}}}},
        {"plan": "ana", "expect": {"steps": [STOVE["id"], TABLE["id"]], "cost": 95.0}},    # 2 + 30 + 3 + 60
        {"set": {"holding": {"ana": {"meal": 1}}}},
        {"plan": "ana", "expect": {"steps": [TABLE["id"]]}},
        # A dirty dish in hand and a sink: an untidy mind washes up (the dish is used up by it).
        {"set": {"holding": {"ana": {"dirty_dish": 1}}, "add": [SINK],
                 "needs": {"ana": {"hunger": 1.0, "rest": 1.0, "order": 0.5}}}},
        {"plan": "ana", "expect": {"goal": SINK["id"], "steps": [SINK["id"]], "cost": 11.0}},
        # Without the dish nothing gives one but eating, which needs a meal: fridge > stove > table > sink.
        {"set": {"holding": {"ana": {}}, "add": [FRIDGE]}},
        {"plan": "ana", "expect": {"goal": SINK["id"],
                                   "steps": [FRIDGE["id"], STOVE["id"], TABLE["id"], SINK["id"]]}},
    ],
}

traces["depth"] = {
    "about": "The depth limit is the longest chain: four steps are found at the default depth and not at "
             "depth 3.",
    "world": {"affordances": [FRIDGE, STOVE, TABLE, SINK]},
    "actors": {"ana": {"at": [4, 0, 0], "speed": 1.0, "holding": {}, "needs": {"order": 0.5}}},
    "limits": {"depth": 3},
    "steps": [
        {"plan": "ana", "expect": {"steps": []}},
        {"set": {"holding": {"ana": {"raw_food": 1}}}},
        {"plan": "ana", "expect": {"steps": [STOVE["id"], TABLE["id"], SINK["id"]]}},
    ],
}

traces["one_stove"] = {
    "about": "Two hungry people and one stove: a plan claims every thing in it, so the second plans with "
             "what is left (the fruit bowl), and takes the kitchen when the first lets go. With a microwave "
             "as well, both cook.",
    "world": {"affordances": [FRIDGE, STOVE, TABLE, FRUIT,
                              {**TABLE, "id": "kitchen/table/chair2#0", "at": [2, 0, 5]},
                              {**FRIDGE, "id": "kitchen/fridge#1"}]},
    "actors": {"ana": person([4, 0, 0]), "ben": person([5, 0, 0])},
    "steps": [
        {"plan": "ana", "expect": {"steps": [FRIDGE["id"], STOVE["id"], TABLE["id"]]}},
        {"plan": "ben", "expect": {"steps": [FRUIT["id"]]}},
        {"plan": "ana", "expect": {"steps": [FRIDGE["id"], STOVE["id"], TABLE["id"]]}},     # her own claims don't stop her
        {"release": "ana"},
        {"release": "ben"},
        {"plan": "ben", "expect": {"steps": [FRIDGE["id"], STOVE["id"], TABLE["id"]], "cost": 104.0}},
        {"plan": "ana", "expect": {"steps": [FRUIT["id"]]}},
        {"set": {"add": [MICROWAVE]}},
        {"plan": "ana", "expect": {"steps": ["kitchen/fridge#1", MICROWAVE["id"], "kitchen/table/chair2#0"]}},
    ],
}

TV = "lounge/tv#power"
traces["states"] = {
    "about": "States of things: watching from the sofa requires the TV on, and the TV's switch sets it, so "
             "the plan is switch > watch; with the TV on already it is the sofa alone. A step isn't put "
             "before one whose state it would undo.",
    "world": {"affordances": [
        aff("lounge/tv#0", [8, 0, 0], 2, sets={TV: "on"}),
        aff("lounge/tv#1", [8, 0, 0], 2, sets={TV: "off"}, advertises={"quiet": 0.2}),
        aff("lounge/sofa#watch", [8, 0, 3], 100, requires={"state": {TV: "on"}}, advertises={"fun": 0.5}),
        aff("lounge/armchair#read", [6, 0, 3], 100, requires={"state": {TV: "off"}, "holding": ["book"]},
            advertises={"fun": 0.4}),
        aff("lounge/bookcase#0", [6, 0, 0], 3, gives={"book": 1})],
        "states": {TV: "off"}},
    "actors": {"ana": {"at": [4, 0, 0], "speed": 1.0, "holding": {}, "needs": {"fun": 0.2, "quiet": 1.0}}},
    "steps": [
        # walk 4 + 2 + walk 3 + 100
        {"plan": "ana", "expect": {"steps": ["lounge/tv#0", "lounge/sofa#watch"], "cost": 109.0}},
        {"set": {"states": {TV: "on"}}},
        {"plan": "ana", "expect": {"steps": ["lounge/sofa#watch"], "cost": 105.0}},
        # The sofa taken, reading: a book, and the TV off (in either order: the cheaper one).
        {"set": {"remove": ["lounge/sofa#watch"]}},
        {"plan": "ana", "expect": {"goal": "lounge/armchair#read",
                                   "steps": ["lounge/bookcase#0", "lounge/tv#1", "lounge/armchair#read"]}},
    ],
}

traces["urgency"] = {
    "about": "Choosing the goal: what a chain adds to the needs, weighted by their urgency, over its time "
             "plus the patience. Peckish and tired, the sofa (near, quick) wins; given an urgency for "
             "hunger (a curve the runtime works out) the meal does; a liking (`weight`) for the other sofa "
             "beats that; and a need that is nearly full is only filled to 1.",
    "world": {"affordances": [FRIDGE, MICROWAVE, TABLE, SOFA, FRUIT]},
    "actors": {"ana": person([8, 0, 2], hunger=0.5, rest=0.3)},
    "limits": {"patience": 60.0},
    "steps": [
        {"plan": "ana", "expect": {"goal": SOFA["id"], "score": 0.002079}},      # 0.3 x 0.7 / (1 + 40 + 60)
        {"set": {"urgency": {"ana": {"hunger": 2.0, "rest": 0.7}}}},
        {"plan": "ana", "expect": {"goal": TABLE["id"], "steps": [FRIDGE["id"], MICROWAVE["id"], TABLE["id"]]}},
        {"set": {"add": [{**SOFA, "id": "lounge/sofa2#0", "weight": 4.0}]}},
        {"plan": "ana", "expect": {"goal": "lounge/sofa2#0", "score": 0.008317}},
        {"set": {"needs": {"ana": {"hunger": 0.9, "rest": 1.0}}, "urgency": {"ana": {}}}},
        {"plan": "ana", "expect": {"goal": FRUIT["id"], "fills": {"hunger": 0.1}}},
        # Something worth doing for itself (a want put right: a can on the floor), no need filled.
        {"set": {"add": [aff("want/tidy/can", [7, 0, 2], 15, worth=0.5)]}},
        {"plan": "ana", "expect": {"goal": "want/tidy/can", "fills": {}, "score": 0.006579}},     # 0.5 / (1 + 15 + 60)
    ],
}

# Two people and one job (geogen.110.1): a tree to fetch and set up, a box to fetch and put out,
# ornaments to hang. Speed 1 m/s, everything in a line.
FETCH_TREE = aff("yard/truck#0", [0, 0, 0], 2, gives={"tree": 1})
SET_UP = aff("room/tree#0", [4, 0, 0], 2, takes=["tree"], sets={"room/tree#tree": "up"})
FETCH_BOX = aff("hall/cupboard#0", [6, 0, 0], 2, gives={"box": 1})
PUT_BOX = aff("room/tree#1", [5, 0, 0], 2, takes=["box"], sets={"room/tree#box": "out"})
TAKE = aff("room/tree#2", [5, 0, 1], 1, gives={"ornament": 1}, requires={"state": {"room/tree#box": "out"}})
HANG = aff("room/tree#3", [4, 0, 1], 1, takes=["ornament"], advertises={"festive": 0.5},
           requires={"state": {"room/tree#tree": "up"}})
TREE_STATES = {"room/tree#tree": "absent", "room/tree#box": "away"}


def helper(at, **more):
    return {"at": at, "speed": 1.0, "holding": {}, "needs": {"festive": 0.0}, "hands": 1, **more}


CHAIN = [FETCH_TREE["id"], SET_UP["id"], FETCH_BOX["id"], PUT_BOX["id"], TAKE["id"], HANG["id"]]

traces["hands"] = {
    "about": "One pair of hands: the chain never has the tree and the box carried together, though carrying "
             "both would be the shorter way round. The plan commits to its first run, up to the step that "
             "leaves the hands empty.",
    "world": {"affordances": [FETCH_TREE, SET_UP, FETCH_BOX, PUT_BOX, TAKE, HANG], "states": TREE_STATES},
    "actors": {"nell": helper([0, 0, 0])},
    "limits": {"depth": 6},
    "steps": [
        {"plan": "nell", "expect": {"goal": HANG["id"], "steps": CHAIN, "run": CHAIN[:2], "waits": []}},
        {"set": {"holding": {"nell": {"tree": 1}}}},
        {"plan": "nell", "expect": {"steps": CHAIN[1:], "run": [SET_UP["id"]], "waits": []}},
    ],
}

traces["shared_job"] = {
    "about": "Two people share the job. Nell claims the tree; Tom counts on it being up and sees to the box. "
             "With the box out and the tree still promised, the run to hang an ornament has to wait for it. "
             "Once it is up, both can hang.",
    "world": {"affordances": [FETCH_TREE, SET_UP, FETCH_BOX, PUT_BOX, TAKE, HANG,
                              {**TAKE, "id": "room/tree#4"}, {**HANG, "id": "room/tree#5"}],
              "states": TREE_STATES},
    "actors": {"nell": helper([0, 0, 0]), "tom": helper([6, 0, 0])},
    "limits": {"depth": 6},
    "steps": [
        {"plan": "nell", "expect": {"run": CHAIN[:2], "waits": []}},
        {"plan": "tom", "expect": {"steps": [FETCH_BOX["id"], PUT_BOX["id"], TAKE["id"], HANG["id"]],
                                   "run": [FETCH_BOX["id"], PUT_BOX["id"]], "waits": []}},
        {"set": {"states": {"room/tree#box": "out"}, "remove": [FETCH_BOX["id"], PUT_BOX["id"]]}},
        {"plan": "tom", "expect": {"steps": [TAKE["id"], HANG["id"]], "run": [TAKE["id"], HANG["id"]],
                                   "waits": ["room/tree#tree"]}},
        {"set": {"states": {"room/tree#tree": "up"}, "remove": [FETCH_TREE["id"], SET_UP["id"]]}},
        {"release": "nell"},
        {"plan": "tom", "expect": {"steps": [TAKE["id"], HANG["id"]], "waits": []}},
        {"plan": "nell", "expect": {"steps": ["room/tree#4", "room/tree#5"], "waits": []}},
    ],
}

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
for name, trace in traces.items():
    (out / f"{name}.json").write_text(json.dumps({"name": name, **trace}, indent=1) + "\n")
print(len(traces))
