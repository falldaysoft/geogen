"""Conditions over live state: compiling, resolving to stable IDs, the reference evaluator."""

import json
from pathlib import Path

import pytest

from geogen import conditions, refs
from geogen.conditions import ConditionError, compile_condition, evaluate, holds, resolve, validate

SCHEMA = Path(__file__).parent.parent / conditions.SCHEMA


def lookup_in(values):
    """A lookup over {(kind, id or package or None, name): value}."""
    def lookup(term):
        return values[(term["get"], term.get("id", term.get("package")), term["name"])]
    return lookup


def test_literals_and_operators_compile_to_a_plain_tree():
    assert compile_condition("1 + 2 * 3 > 6") == {
        "op": ">", "args": [{"op": "+", "args": [1, {"op": "*", "args": [2, 3]}]}, 6]}
    assert compile_condition("not (true or false)") == {"op": "not", "args": [{"op": "or", "args": [True, False]}]}
    assert compile_condition("-2") == -2
    assert compile_condition(True) is True
    assert compile_condition("{world.done}") == {"get": "world", "name": "done"}    # braces are optional


def test_terms():
    assert compile_condition('door.swing == "open"') == {
        "op": "==", "args": [{"get": "ref", "ref": "door", "name": "swing"}, "open"]}
    assert compile_condition('ref("plot/house/door").swing')["ref"] == "plot/house/door"
    assert compile_condition("self.power") == {"get": "self", "name": "power"}
    assert compile_condition("actor.has_ticket") == {"get": "actor", "name": "has_ticket"}
    assert compile_condition('clock.time >= "22:30" or clock.night')["args"][1] == {"get": "clock", "name": "night"}


def test_chained_comparison_and_in():
    tree = compile_condition("1 <= world.n < 4")
    assert tree["op"] == "and" and [a["op"] for a in tree["args"]] == ["<=", "<"]
    tree = compile_condition('actor.room not in ("lobby", "bar")')
    assert tree == {"op": "not", "args": [{"op": "in", "args": [{"get": "actor", "name": "room"}, "lobby", "bar"]}]}


def test_params_become_values():
    assert compile_condition("world.n >= need", {"need": 3})["args"][1] == 3
    with pytest.raises(ConditionError, match="isn't a param here"):
        compile_condition("world.n >= need")


@pytest.mark.parametrize("bad, message", [
    ("door.swing.x == 1", "longer reference"),
    ("len(world.items) > 1", "unsupported call"),
    ("world.n ** 2", "unsupported operator"),
    ("it.power == 'on'", "only means something inside"),
    ("all('[tag=x]', any('[tag=y]', it.a))", "can't be nested"),
    ("all(doors, it.swing)", "selector"),
    ("clock.month > 3", "the clock has"),
    ("[x for x in world.n]", "unsupported syntax"),
    ("__import__('os')", "unsupported call"),
    ("world.n ==", "can't read condition"),
    ("", "expected a condition"),
])
def test_rejects(bad, message):
    with pytest.raises(ConditionError, match=message):
        compile_condition(bad)


def test_resolve_names_things_by_stable_id(built_scene):
    room = built_scene("hotel_room_auto")
    tree = resolve(compile_condition('ref("suite/entry").swing == "open"'), room, "hotel_room_auto")
    assert tree["args"][0] == {"get": "instance", "id": "hotel_room_auto:suite/entry", "name": "swing"}

    tree = resolve(compile_condition('count("[tag=switch.light]", it.switch == "off") >= 2 and world.tidy'),
                   room, "hotel_room_auto")
    quantifier = tree["args"][0]["args"][0]
    assert quantifier["op"] == "count" and len(quantifier["over"]) == 5
    assert "hotel_room_auto:suite/bedroom/bedroom_switch_1" in quantifier["over"]
    assert tree["args"][1] == {"get": "world", "package": "hotel_room_auto", "name": "tidy"}
    validate(tree)

    # Stable IDs are references: each resolves back to the node it came from.
    for uid in quantifier["over"]:
        assert [i.name for i in refs.resolve(room, uid, "hotel_room_auto").interactions] == ["switch"]

    entry = refs.resolve(room, "suite/entry", "hotel_room_auto")
    tree = resolve(compile_condition('self.swing == "closed"'), room, "hotel_room_auto", self_node=entry)
    assert tree["args"][0]["id"] == "hotel_room_auto:suite/entry"


def test_resolve_diagnostics(built_scene):
    room = built_scene("hotel_room_auto")

    def fails(text, message):
        with pytest.raises(ConditionError, match=message):
            resolve(compile_condition(text), room, "hotel_room_auto")

    fails('ref("suite/entri").swing == "open"', "entry")                # a near name
    fails('ref("suite/entry").swign == "open"', "Did you mean 'swing'")
    fails('all("[tag=furniture]", it.drawer == "closed")', "has no state 'drawer'")    # not all have one
    fails('any("[tag=no.such]", it.swing)', "no.such")
    fails("self.swing", "no 'self' here")


def test_evaluate():
    values = {("instance", "a:door", "swing"): "open", ("world", "a", "n"): 3, ("actor", "a", "ticket"): False,
              ("actor", None, "room"): "lobby", ("clock", None, "time"): "23:10", ("clock", None, "hour"): 23.17}
    lookup = lookup_in(values)

    def run(text):
        tree = compile_condition(text)
        for node in conditions.walk(tree):                   # resolve by hand: no scene here
            if node.get("get") == "ref":
                node.update(get="instance", id="a:" + node.pop("ref"))
            elif node.get("get") in ("world", "actor") and node["name"] not in conditions.ACTOR_BUILTINS:
                node["package"] = "a"
        validate(tree)
        return evaluate(json.loads(json.dumps(tree)), lookup)        # what a runtime would be handed

    assert run('door.swing == "open"') is True
    assert run('door.swing == 1') is False
    assert run("world.n * 2 + 1") == 7
    assert run("world.n / 2 >= 1.5 and not actor.ticket") is True
    assert run("world.n and 'x'") is True                    # and / or give booleans, not operands
    assert run('"yes" if actor.room in ("lobby", "bar") else "no"') == "yes"
    assert run('clock.time >= "22:30" or clock.time < "07:00"') is True
    assert run("max(world.n, 5) - abs(-2) + round(2.5) + floor(clock.hour)") == 5 - 2 + 3 + 23
    assert run("false and world.n / 0") is False             # short-circuits
    with pytest.raises(ConditionError, match="division by zero"):
        run("world.n / 0")
    with pytest.raises(ConditionError, match="Can't order"):
        run("door.swing > 2")
    with pytest.raises(ConditionError, match="needs numbers"):
        run("door.swing + 1")


def test_quantifiers_evaluate_over_ids():
    state = {"a:l1": "off", "a:l2": "on", "a:l3": "off"}
    lookup = lambda term: state[term["id"]]
    where = {"op": "==", "args": [{"get": "it", "name": "power"}, "off"]}
    tree = lambda op: {"op": op, "over": list(state), "where": where}
    assert evaluate(tree("count"), lookup) == 2
    assert evaluate(tree("any"), lookup) is True
    assert evaluate(tree("all"), lookup) is False
    state["a:l2"] = "off"
    assert holds({"op": "==", "args": [tree("count"), 3]}, lookup) and evaluate(tree("all"), lookup) is True
    assert evaluate({"op": "all", "over": [], "where": where}, lookup) is True     # nothing to fail


def test_validate_rejects_unresolved_and_malformed():
    for bad in (compile_condition("door.swing"), compile_condition("world.n"), compile_condition("all('[tag=x]', it.a)"),
                {"op": "xor", "args": [1, 2]}, {"op": "not", "args": [1, 2]}, {"get": "clock", "name": "month"},
                {"get": "it", "name": "a"}, [1], None):
        with pytest.raises(ConditionError):
            validate(bad)


def test_compiled_trees_match_the_schema(built_scene):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text())
    room = built_scene("hotel_room_auto")
    for text in ('ref("suite/entry").swing == "open"', "world.n + 1 > min(actor.m, 3) or not clock.night",
                 'count("[tag=switch.light]", it.switch == "off") == 5', 'actor.room in ("bedroom", "bathroom")',
                 '"a" if clock.hour < 12 else "b"', "true", "-world.n"):
        tree = json.loads(json.dumps(resolve(compile_condition(text), room, "hotel_room_auto")))
        validate(tree)
        jsonschema.validate(tree, schema)
    for bad in (compile_condition("door.swing"), {"op": "xor", "args": [1]}, {"get": "clock", "name": "month"},
                {"op": "all", "select": "[tag=x]", "where": True}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
