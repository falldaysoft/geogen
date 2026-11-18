// The part of the rules contract the viewer runs: compiled conditions (docs/schema/geogen-condition.v1.schema.json,
// the twin of geogen/conditions.py and runtime/godot/scripts/conditions.gd) and the manifest's bindings, which are what
// joins a light switch to its lamp. Rules, quests, items and the clock are the game runtime's: a term that needs one of
// them is Unknown here, so a binding that uses it is left alone and a guard that uses it doesn't stop anybody.

export class Unknown extends Error {}

export function truthy(value) {
  return !(value === false || value === 0 || value === '' || value === null || value === undefined);
}

const isText = v => typeof v === 'string';
const number = v => {
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (typeof v === 'number') return v;
  throw new Unknown(`not a number: ${JSON.stringify(v)}`);
};

function equal(a, b) {
  if (isText(a) || isText(b)) return a === b;    // text never equals a number
  return number(a) === number(b);
}

function order(op, a, b) {
  if (isText(a) !== isText(b)) throw new Unknown(`can't order ${a} and ${b}`);
  const [x, y] = isText(a) ? [a, b] : [number(a), number(b)];
  return op === '<' ? x < y : op === '<=' ? x <= y : op === '>' ? x > y : x >= y;
}

// The value of a tree. `lookup(term)` answers `get` terms; `it` is the thing an all / any / count is looking at.
export function evaluate(tree, lookup, it = null) {
  if (tree === null || typeof tree !== 'object') return tree;
  if ('get' in tree) {
    if (tree.get === 'it') {
      if (it === null) throw new Unknown("'it' outside all / any / count");
      return lookup({ get: 'instance', id: it, name: tree.name });
    }
    return lookup(tree);
  }
  const { op, args = [] } = tree;
  const run = arg => evaluate(arg, lookup, it);
  switch (op) {
    case 'all': case 'any': case 'count': {
      if (!tree.over) throw new Unknown('unresolved selector');
      const hits = tree.over.filter(thing => truthy(evaluate(tree.where, lookup, thing))).length;
      return op === 'count' ? hits : op === 'any' ? hits > 0 : hits === tree.over.length;
    }
    case 'and': return args.every(a => truthy(run(a)));
    case 'or': return args.some(a => truthy(run(a)));
    case 'not': return !truthy(run(args[0]));
    case 'if': return truthy(run(args[0])) ? run(args[1]) : run(args[2]);
  }
  const v = args.map(run);
  switch (op) {
    case '==': return equal(v[0], v[1]);
    case '!=': return !equal(v[0], v[1]);
    case '<': case '<=': case '>': case '>=': return order(op, v[0], v[1]);
    case 'in': return v.slice(1).some(c => equal(v[0], c));
    case 'neg': return -number(v[0]);
    case '+': return number(v[0]) + number(v[1]);
    case '-': return number(v[0]) - number(v[1]);
    case '*': return number(v[0]) * number(v[1]);
    case '/':
      if (number(v[1]) === 0) throw new Unknown('division by zero');
      return number(v[0]) / number(v[1]);
    case 'min': return Math.min(...v.map(number));
    case 'max': return Math.max(...v.map(number));
    case 'abs': return Math.abs(number(v[0]));
    case 'round': return Math.floor(number(v[0]) + 0.5);
    case 'floor': return Math.floor(number(v[0]));
    case 'ceil': return Math.ceil(number(v[0]));
  }
  throw new Unknown(`unknown op '${op}'`);
}

// The instance terms a tree reads, as [id, name]: which switches a binding listens to.
export function instanceTerms(tree, out = []) {
  if (tree === null || typeof tree !== 'object') return out;
  if (tree.get === 'instance') out.push([tree.id, tree.name]);
  for (const arg of tree.args ?? []) instanceTerms(arg, out);
  return out;
}

// `<thing>#<name>` values: the manifest's defaults, what interactions declare, and its bindings kept true.
export class WorldState {
  constructor(manifest) {
    this.values = new Map();
    this.guards = new Map();
    this.bindings = new Map();
    this.listeners = new Set();
    for (const [id, variable] of Object.entries(manifest?.state?.variables ?? {})) {
      if ('default' in variable) this.values.set(id, variable.default);
      if ('when' in variable) this.guards.set(id, variable.when);
    }
    for (const binding of manifest?.rules?.bindings ?? []) this.bindings.set(binding.set, binding.to);
  }

  lookup = term => {
    if (term.get !== 'instance') throw new Unknown(`a ${term.get} term`);
    const value = this.values.get(`${term.id}#${term.name}`);
    if (value === undefined) throw new Unknown(`no value for ${term.id}#${term.name}`);
    return value;
  };

  get(id) { return this.values.get(id); }
  isBound(id) { return this.bindings.has(id); }

  // Whether a guard lets the thing be used: only what it can't work out here is let through.
  allows(id) {
    if (!this.guards.has(id)) return true;
    try { return truthy(evaluate(this.guards.get(id), this.lookup)); } catch (err) {
      if (err instanceof Unknown) return true;
      throw err;
    }
  }

  set(id, value) {
    if (this.values.get(id) === value) return;
    this.values.set(id, value);
    for (const listener of this.listeners) listener(id, value);
    this.settle();
  }

  // Bindings until nothing changes (a binding can read what another one sets).
  settle() {
    for (let pass = 0; pass < 8; pass++) {
      let changed = false;
      for (const [id, tree] of this.bindings) {
        let value;
        try { value = evaluate(tree, this.lookup); } catch (err) {
          if (err instanceof Unknown) continue;
          throw err;
        }
        if (this.values.get(id) !== value) {
          this.values.set(id, value);
          for (const listener of this.listeners) listener(id, value);
          changed = true;
        }
      }
      if (!changed) return;
    }
  }
}
