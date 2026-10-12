class_name GeogenConditions
extends RefCounted
## Evaluates compiled conditions (docs/schema/geogen-condition.v1.schema.json):
## the GDScript twin of geogen/conditions.py `evaluate`. A runtime never parses
## expression text; it is handed these trees.
##
## A literal is a number, text or boolean. Everything else is a Dictionary:
##   {"op": "==", "args": [a, b]}       also != < <= > >= + - * / neg not and or
##                                      if (condition, then, else), in (value, candidates...)
##                                      min max abs round floor ceil
##   {"op": "all", "over": [id, ...], "where": tree}    also any, count
##   {"get": "instance" | "world" | "actor" | "clock" | "it", ...}
##
## Rules (shared with the reference): false, 0 and "" are false; and / or / not
## give a boolean; text never equals a number; ordering needs two numbers or two
## texts ("HH:MM" times order as text); arithmetic needs numbers and gives
## floats; a boolean counts as 0 or 1 among numbers.
##
## GDScript has no exceptions, so a failure (ordering text against a number,
## arithmetic on text, division by zero, a term nothing answers) makes
## `evaluate` return null and put one message in the `errors` array it was given.

const QUANTIFIERS := ["all", "any", "count"]
const COMPARISONS := ["==", "!=", "<", "<=", ">", ">="]
const FUNCTIONS := ["min", "max", "abs", "round", "floor", "ceil"]


## The value of a resolved tree. `lookup(term: Dictionary)` gives the current
## value of a `get` node (`it` terms arrive as the `instance` term of each
## thing); null from it means it has no value, which fails the evaluation.
static func evaluate(tree: Variant, lookup: Callable, errors: Array = []) -> Variant:
    return _run(tree, "", false, lookup, errors)


## Whether a condition is true (false when it can't be evaluated).
static func holds(tree: Variant, lookup: Callable, errors: Array = []) -> bool:
    return truthy(evaluate(tree, lookup, errors))


## Whether a value counts as true: not false, 0 or "".
static func truthy(value: Variant) -> bool:
    match typeof(value):
        TYPE_NIL:
            return false
        TYPE_BOOL:
            return value
        TYPE_INT, TYPE_FLOAT:
            return value != 0
        TYPE_STRING, TYPE_STRING_NAME:
            return str(value) != ""
        TYPE_ARRAY, TYPE_DICTIONARY:
            return not value.is_empty()
    return true


static func is_text(value: Variant) -> bool:
    return typeof(value) == TYPE_STRING or typeof(value) == TYPE_STRING_NAME


## A number that isn't a boolean (JSON numbers arrive as floats).
static func is_number(value: Variant) -> bool:
    return typeof(value) == TYPE_INT or typeof(value) == TYPE_FLOAT


## Equality as conditions see it: 1 equals 1.0, true equals 1, text never
## equals a number. (GDScript's own == raises on mixed types.)
static func equal(a: Variant, b: Variant) -> bool:
    if is_text(a) or is_text(b):
        return is_text(a) and is_text(b) and str(a) == str(b)
    if _numeric(a) and _numeric(b):
        return float(a) == float(b)
    if typeof(a) != typeof(b):
        return false
    return a == b


## Equal and of one kind: true is not the number 1 here (what a variable holds).
static func same(a: Variant, b: Variant) -> bool:
    return (typeof(a) == TYPE_BOOL) == (typeof(b) == TYPE_BOOL) and equal(a, b)


## One comparison; null with a message in `errors` when the two can't be ordered.
static func compare(op: String, left: Variant, right: Variant, errors: Array) -> Variant:
    if op == "==" or op == "!=":
        return equal(left, right) == (op == "==")
    if is_text(left) != is_text(right):
        errors.append("Can't order %s and %s: one is text, the other a number" % [shown(left), shown(right)])
        return null
    if is_text(left):
        var a := str(left)
        var b := str(right)
        match op:
            "<": return a < b
            "<=": return a <= b
            ">": return a > b
        return a >= b
    if not (_numeric(left) and _numeric(right)):
        errors.append("Can't order %s and %s" % [shown(left), shown(right)])
        return null
    var x := float(left)
    var y := float(right)
    match op:
        "<": return x < y
        "<=": return x <= y
        ">": return x > y
    return x >= y


## A value as messages show it.
static func shown(value: Variant) -> String:
    if is_number(value) and float(value) == floorf(float(value)) and absf(float(value)) < 1e15:
        return str(int(value))
    return JSON.stringify(value)


static func _numeric(value: Variant) -> bool:
    return typeof(value) == TYPE_BOOL or typeof(value) == TYPE_INT or typeof(value) == TYPE_FLOAT


static func _run(t: Variant, it: String, has_it: bool, lookup: Callable, errors: Array) -> Variant:
    if typeof(t) != TYPE_DICTIONARY:
        return t
    var node: Dictionary = t
    if node.has("get"):
        var term := node
        if node["get"] == "it":
            if not has_it:
                errors.append("'it' outside all / any / count")
                return null
            term = {"get": "instance", "id": it, "name": node["name"]}
        elif node["get"] == "ref" or node["get"] == "self":
            errors.append("unresolved term %s: resolve the condition first" % JSON.stringify(node))
            return null
        var found: Variant = lookup.call(term)
        if found == null:
            errors.append("nothing gives a value for %s" % JSON.stringify(term))
        return found
    var op: String = node["op"]
    if op in QUANTIFIERS:
        if not node.has("over"):
            errors.append("unresolved selector '%s': resolve the condition first" % str(node.get("select")))
            return null
        var hits := 0
        for thing in node["over"]:
            var hit := truthy(_run(node["where"], str(thing), true, lookup, errors))
            if not errors.is_empty():
                return null
            if hit:
                hits += 1
                if op == "any":
                    return true
            elif op == "all":
                return false
        return float(hits) if op == "count" else op == "all"
    var args: Array = node["args"]
    if op == "and" or op == "or":
        for arg in args:
            var held := truthy(_run(arg, it, has_it, lookup, errors))
            if not errors.is_empty():
                return null
            if held != (op == "and"):       # the first false ends an and, the first true an or
                return held
        return op == "and"
    if op == "not":
        var negated := not truthy(_run(args[0], it, has_it, lookup, errors))
        return negated if errors.is_empty() else null
    if op == "if":
        var chosen := truthy(_run(args[0], it, has_it, lookup, errors))
        if not errors.is_empty():
            return null
        return _run(args[1] if chosen else args[2], it, has_it, lookup, errors)
    var values := []
    for arg in args:
        values.append(_run(arg, it, has_it, lookup, errors))
        if not errors.is_empty():
            return null
    if op == "in":
        for i in range(1, values.size()):
            if equal(values[0], values[i]):
                return true
        return false
    if op in COMPARISONS:
        return compare(op, values[0], values[1], errors)
    var numbers: Array[float] = []
    for v in values:
        if not _numeric(v):
            errors.append("'%s' needs numbers, got %s" % [op, ("text " + shown(v)) if is_text(v) else shown(v)])
            return null
        numbers.append(float(v))
    match op:
        "neg":
            return -numbers[0]
        "+":
            return numbers[0] + numbers[1]
        "-":
            return numbers[0] - numbers[1]
        "*":
            return numbers[0] * numbers[1]
        "/":
            if numbers[1] == 0.0:
                errors.append("division by zero")
                return null
            return numbers[0] / numbers[1]
        "min":
            return numbers.min()
        "max":
            return numbers.max()
        "abs":
            return absf(numbers[0])
        "round":
            return floorf(numbers[0] + 0.5)
        "floor":
            return floorf(numbers[0])
        "ceil":
            return ceilf(numbers[0])
    errors.append("unknown op '%s'" % op)
    return null
