// extras.geogen.interactions as state machines that pose their parts: the viewer's counterpart of
// runtime/godot/scripts/interaction.gd. Exported transforms show the `initial` state, so a part's rest pose is
// M(initial value)^-1 * (asset^-1 * part), and its pose for a value v is asset * M(v) * rest. `drive` motions (the motors
// of mechanisms) need a physics engine and are left out.

import * as THREE from 'three';

export const geogen = object => object.userData?.geogen;
export const nodeName = object => object.userData?.name ?? object.name;   // GLTFLoader makes names unique; extras don't know

function findNode(root, name) {
  if (nodeName(root) === name) return root;
  for (const child of root.children) {
    const found = findNode(child, name);
    if (found) return found;
  }
  return null;
}

const axis = new THREE.Vector3(), pivot = new THREE.Vector3();
const toPivot = new THREE.Matrix4(), fromPivot = new THREE.Matrix4(), turn = new THREE.Matrix4();

function motionMatrix(motion, value, out) {
  axis.fromArray(motion.axis ?? [0, 1, 0]).normalize();
  if (motion.type === 'translate') return out.makeTranslation(axis.x * value, axis.y * value, axis.z * value);
  pivot.fromArray(motion.pivot ?? [0, 0, 0]);
  toPivot.makeTranslation(-pivot.x, -pivot.y, -pivot.z);
  fromPivot.makeTranslation(pivot.x, pivot.y, pivot.z);
  turn.makeRotationAxis(axis, THREE.MathUtils.degToRad(value));
  return out.copy(fromPivot).multiply(turn).multiply(toPivot);
}

const pose = new THREE.Matrix4(), world = new THREE.Matrix4(), local = new THREE.Matrix4();

export class Interaction {
  constructor(asset, name, data, state) {
    this.asset = asset;
    this.name = name;
    this.data = data;
    this.state = state;
    this.states = data.states ?? {};
    this.current = this.target = data.initial;
    this.duration = Math.max(data.duration ?? 0.9, 0.01);
    this.locked = !!data.lock?.locked;
    this.reach = data.reach ?? 0;
    this.dwell = 0;
    const uid = geogen(asset)?.uid;
    this.id = uid ? `${uid}#${name}` : `${asset.uuid}#${name}`;
    this.label = nodeName(asset);
    this.targets = (data.targets ?? []).map(n => findNode(asset, n)).filter(Boolean);
    this.motions = [];
    this.shows = [];
    this.parts = new Set();

    const assetInverse = asset.matrixWorld.clone().invert();
    let widest = 0;
    for (const motion of data.motions ?? []) {
      if (motion.type === 'drive') continue;
      const parts = (motion.nodes ?? []).map(n => findNode(asset, n)).filter(Boolean);
      parts.forEach(p => this.parts.add(p));
      if (motion.type === 'visible') { this.shows.push({ parts, values: motion.values ?? {} }); continue; }
      const values = motion.values ?? {};
      const value = values[this.current] ?? 0;
      const initialInverse = motionMatrix(motion, value, new THREE.Matrix4()).invert();
      const rest = parts.map(p => initialInverse.clone().multiply(assetInverse).multiply(p.matrixWorld));
      const numbers = Object.values(values);
      if (numbers.length) widest = Math.max(widest, Math.max(...numbers) - Math.min(...numbers));
      this.motions.push({ spec: motion, parts, rest, values, value });
    }
    this.speed = Math.max(widest, 1e-6) / this.duration;

    this.meshes = [];       // what the player aims at
    for (const target of this.targets.length ? this.targets : [asset]) {
      target.traverse(o => { if (o.isMesh && geogen(o)?.type !== 'collider') this.meshes.push(o); });
    }
    state.values.set(this.id, this.current);
    this.applyVisibility();
  }

  get usable() { return 'next' in (this.states[this.target] ?? {}) && this.state.allows(this.id); }

  // What using it would do now ('' when nothing). The viewer has every key: a lock only adds a word.
  get prompt() {
    const here = this.states[this.target] ?? {};
    if (!('next' in here)) return '';
    if (!this.state.allows(this.id)) return this.data.refused ?? '';
    const text = here.prompt ?? 'Use';
    return this.locked ? `Unlock and ${text.toLowerCase()}` : text;
  }

  use() {
    if (!this.usable) return false;
    this.locked = false;
    this.headTo(this.states[this.target].next);
    return true;
  }

  headTo(state) {
    if (state === this.target) return;
    this.target = state;
    this.applyVisibility();
    this.state.set(this.id, state);
  }

  applyVisibility() {
    for (const { parts, values } of this.shows) {
      const on = (values[this.target] ?? 0) >= 0.5 || (values[this.current] ?? 0) >= 0.5;
      for (const part of parts) part.visible = on;
    }
  }

  apply(motion) {
    motionMatrix(motion.spec, motion.value, pose);
    motion.parts.forEach((part, i) => {
      part.parent.updateWorldMatrix(true, false);
      world.copy(this.asset.matrixWorld).multiply(pose).multiply(motion.rest[i]);
      local.copy(part.parent.matrixWorld).invert().multiply(world);
      local.decompose(part.position, part.quaternion, part.scale);
      part.updateWorldMatrix(false, true);
    });
  }

  // One frame. True while a part moved, so colliders that ride on it know to follow.
  update(dt) {
    let arrived = true, moved = false;
    for (const motion of this.motions) {
      const goal = motion.values[this.target] ?? 0;
      if (motion.value !== goal) {
        const step = this.speed * dt, gap = goal - motion.value;
        motion.value = Math.abs(gap) <= step ? goal : motion.value + Math.sign(gap) * step;
        this.apply(motion);
        moved = true;
      }
      if (motion.value !== goal) arrived = false;
    }
    if (this.current === this.target) {
      const here = this.states[this.current] ?? {};     // a door that closes itself after a while
      if (arrived && typeof here.after === 'number' && 'then' in here) {
        this.dwell += dt;
        if (this.dwell >= here.after) { this.dwell = 0; this.headTo(here.then); }
      }
      return moved;
    }
    this.dwell = 0;
    if (arrived) {
      this.current = this.target;
      this.applyVisibility();
      const here = this.states[this.current] ?? {};
      if ('then' in here && !('after' in here)) this.headTo(here.then);
      return true;
    }
    return moved;
  }
}

// Every interaction under `model`, in the order the export wrote them.
export function collectInteractions(model, state) {
  model.updateMatrixWorld(true);
  const out = [];
  model.traverseVisible(node => {     // not the hidden item templates and icons
    for (const [name, data] of Object.entries(geogen(node)?.interactions ?? {})) {
      out.push(new Interaction(node, name, data, state));
    }
  });
  return out;
}
