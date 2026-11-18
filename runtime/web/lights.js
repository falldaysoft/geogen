// Light fixtures: the KHR_lights_punctual lights of an export, lit by the state `<uid>#lit` as in the Godot runtime
// (world_lights.gd), or by the dark for `auto: night` ones. Three.js recompiles every material when the number of lights
// changes and pays per light per pixel, so a fixed pool of lights goes to the lit fixtures nearest the camera.

import * as THREE from 'three';
import { geogen, nodeName } from './interactions.js';
import { instanceTerms } from './rules.js';

const POOL = 8;
const BLACK = new THREE.Color(0x000000);

export class Lights {
  constructor(scene) {
    this.pool = Array.from({ length: POOL }, () => {
      const light = new THREE.PointLight(0xffffff, 0, 1, 2);
      scene.add(light);
      return light;
    });
    this.fixtures = [];
    this.night = false;
    this.state = null;
    this.interactions = new Map();
    this.wait = 0;
  }

  load(model, state, interactions) {
    this.state = state;
    this.interactions = new Map(interactions.map(it => [it.id, it]));
    this.fixtures = [];
    model.updateMatrixWorld(true);
    const found = [];
    model.traverseVisible(o => { if (o.isPointLight) found.push(o); });
    for (const light of found) {
      const node = light.parent;
      const spec = geogen(node)?.light ?? {};
      const uid = geogen(node)?.uid;
      const id = uid ? `${uid}#lit` : `${node.uuid}#lit`;
      const glow = [];
      node.traverse(o => {
        if (!o.isMesh || geogen(o)?.type === 'collider' || !o.material?.emissive) return;
        if (o.material.emissive.equals(BLACK) || o.material.emissiveIntensity === 0) return;
        o.material = o.material.clone();     // materials are shared with fixtures that stay lit
        glow.push({ material: o.material, color: o.material.emissive.clone() });
      });
      if (!spec.auto && state.get(id) === undefined) state.values.set(id, true);
      this.fixtures.push({
        id, node, glow, auto: spec.auto ?? null, name: nodeName(node),
        position: light.getWorldPosition(new THREE.Vector3()),
        color: light.color.clone(), intensity: light.intensity, distance: light.distance || spec.range || 8,
      });
      light.visible = false;
    }
    this.sync();
  }

  isLit(fixture) {
    return fixture.auto === 'night' ? this.night : this.state.get(fixture.id) !== false;
  }

  // A switch that drives the fixture, when one does: the panel works the lamp the way a hand would.
  switchFor(fixture) {
    const tree = this.state.bindings.get(fixture.id);
    if (!tree) return null;
    for (const [id, name] of instanceTerms(tree)) {
      const it = this.interactions.get(`${id}#${name}`);
      if (it) return it;
    }
    return null;
  }

  toggle(fixture) {
    if (fixture.auto) return;
    const sw = this.switchFor(fixture);
    if (sw) sw.use();
    else if (!this.state.isBound(fixture.id)) this.state.set(fixture.id, !this.isLit(fixture));
    this.sync();
  }

  sync() {
    for (const fixture of this.fixtures) {
      const lit = this.isLit(fixture);
      for (const { material, color } of fixture.glow) material.emissive.copy(lit ? color : BLACK);
    }
    this.wait = 0;
  }

  // The pool, nearest lit fixtures first; every quarter of a second is plenty for a walking pace.
  update(dt, camera) {
    this.wait -= dt;
    if (this.wait > 0) return;
    this.wait = 0.25;
    const lit = this.fixtures.filter(f => this.isLit(f))
      .sort((a, b) => a.position.distanceToSquared(camera.position) - b.position.distanceToSquared(camera.position));
    this.pool.forEach((light, i) => {
      const fixture = lit[i];
      light.intensity = fixture ? fixture.intensity : 0;
      if (!fixture) return;
      light.color.copy(fixture.color);
      light.distance = fixture.distance;
      light.position.copy(fixture.position);
    });
  }
}
