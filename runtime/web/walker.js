// Walking: a body of the manifest's player size against the export's collider nodes (extras.geogen.type == "collider").
// The body is a stack of spheres from step height to head height, pushed sideways out of walls, standing on whatever is
// under it within a step's height. Colliders below a moving part (a door leaf) are re-read while it moves.

import * as THREE from 'three';
import { geogen } from './interactions.js';

const CELL = 1;     // metres: the triangle grid over x, z
const key = (i, j) => (i + 32768) * 65536 + (j + 32768);

function visibleAbove(object) {
  for (let o = object.parent; o; o = o.parent) if (!o.visible) return false;
  return true;
}

function readTriangles(mesh, out) {
  const position = mesh.geometry.attributes.position, index = mesh.geometry.index;
  const count = index ? index.count : position.count;
  const v = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    v.fromBufferAttribute(position, index ? index.getX(i) : i).applyMatrix4(mesh.matrixWorld);
    out.push(v.x, v.y, v.z);
  }
}

export class CollisionWorld {
  // `moving`: the nodes interactions move; colliders under them are kept apart and refreshed.
  constructor(model, moving) {
    model.updateMatrixWorld(true);
    this.minUp = 0;      // the cosine of the steepest slope that can be stood on (Walker sets it)
    const fixed = [];
    this.dynamic = [];
    model.traverse(o => {
      if (!o.isMesh || geogen(o)?.type !== 'collider') return;
      let rides = false;
      for (let p = o.parent; p && !rides; p = p.parent) rides = moving.has(p);
      if (rides) this.dynamic.push({ mesh: o, tris: [], enabled: true });
      else if (visibleAbove(o)) readTriangles(o, fixed);
    });
    this.tris = Float32Array.from(fixed);
    this.count = this.tris.length / 9;
    this.grid = new Map();
    this.stamp = new Uint32Array(this.count);
    this.tick = 0;
    for (let t = 0; t < this.count; t++) {
      const p = t * 9, d = this.tris;
      const [x0, x1] = [Math.min(d[p], d[p + 3], d[p + 6]), Math.max(d[p], d[p + 3], d[p + 6])];
      const [z0, z1] = [Math.min(d[p + 2], d[p + 5], d[p + 8]), Math.max(d[p + 2], d[p + 5], d[p + 8])];
      for (let i = Math.floor(x0 / CELL); i <= Math.floor(x1 / CELL); i++) {
        for (let j = Math.floor(z0 / CELL); j <= Math.floor(z1 / CELL); j++) {
          const k = key(i, j);
          let cell = this.grid.get(k);
          if (!cell) this.grid.set(k, cell = []);
          cell.push(t);
        }
      }
    }
    this.refresh();
  }

  refresh() {
    for (const entry of this.dynamic) {
      entry.enabled = visibleAbove(entry.mesh);
      entry.mesh.updateWorldMatrix(true, false);
      entry.tris.length = 0;
      readTriangles(entry.mesh, entry.tris);
    }
  }

  // Calls fn(data, offset) for every triangle whose cell the x, z rectangle touches.
  each(x0, z0, x1, z1, fn) {
    this.tick++;
    for (let i = Math.floor(x0 / CELL); i <= Math.floor(x1 / CELL); i++) {
      for (let j = Math.floor(z0 / CELL); j <= Math.floor(z1 / CELL); j++) {
        for (const t of this.grid.get(key(i, j)) ?? []) {
          if (this.stamp[t] === this.tick) continue;
          this.stamp[t] = this.tick;
          fn(this.tris, t * 9);
        }
      }
    }
    for (const { tris, enabled } of this.dynamic) {
      if (enabled) for (let p = 0; p < tris.length; p += 9) fn(tris, p);
    }
  }

  // The highest walkable surface at x, z no higher than `top`, or -Infinity.
  groundAt(x, z, top) {
    let best = -Infinity;
    this.each(x, z, x, z, (d, p) => {
      const ax = d[p], ay = d[p + 1], az = d[p + 2];
      const ux = d[p + 3] - ax, uy = d[p + 4] - ay, uz = d[p + 5] - az;
      const vx = d[p + 6] - ax, vy = d[p + 7] - ay, vz = d[p + 8] - az;
      const det = ux * vz - vx * uz;
      if (Math.abs(det) < 1e-9) return;                    // a vertical face
      const nx = uy * vz - uz * vy, nz = ux * vy - uy * vx;
      if (Math.abs(det) < this.minUp * Math.sqrt(nx * nx + det * det + nz * nz)) return;   // too steep to stand on
      const wx = x - ax, wz = z - az;
      const s = (wx * vz - vx * wz) / det, t = (ux * wz - wx * uz) / det;
      if (s < 0 || t < 0 || s + t > 1) return;
      const y = ay + s * uy + t * vy;
      if (y <= top && y > best) best = y;
    });
    return best;
  }

  // Distance along the ray to the first collider within `far`, or Infinity.
  raycast(origin, dir, far) {
    const ex = origin.x + dir.x * far, ez = origin.z + dir.z * far;
    let best = Infinity;
    this.each(Math.min(origin.x, ex), Math.min(origin.z, ez), Math.max(origin.x, ex), Math.max(origin.z, ez), (d, p) => {
      const ax = d[p], ay = d[p + 1], az = d[p + 2];
      const e1x = d[p + 3] - ax, e1y = d[p + 4] - ay, e1z = d[p + 5] - az;
      const e2x = d[p + 6] - ax, e2y = d[p + 7] - ay, e2z = d[p + 8] - az;
      const px = dir.y * e2z - dir.z * e2y, py = dir.z * e2x - dir.x * e2z, pz = dir.x * e2y - dir.y * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (Math.abs(det) < 1e-9) return;
      const inv = 1 / det, tx = origin.x - ax, ty = origin.y - ay, tz = origin.z - az;
      const u = (tx * px + ty * py + tz * pz) * inv;
      if (u < 0 || u > 1) return;
      const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
      const v = (dir.x * qx + dir.y * qy + dir.z * qz) * inv;
      if (v < 0 || u + v > 1) return;
      const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
      if (t >= 0 && t <= far && t < best) best = t;
    });
    return best;
  }

  // Pushes the sphere at (x, y, z) sideways out of every triangle it overlaps; the push, as [dx, dz].
  pushOut(x, y, z, radius) {
    let dx = 0, dz = 0;
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    const tri = new THREE.Triangle(a, b, c), centre = new THREE.Vector3(x, y, z), near = new THREE.Vector3();
    this.each(x - radius, z - radius, x + radius, z + radius, (d, p) => {
      if (Math.min(d[p + 1], d[p + 4], d[p + 7]) > y + radius || Math.max(d[p + 1], d[p + 4], d[p + 7]) < y - radius) return;
      a.set(d[p], d[p + 1], d[p + 2]); b.set(d[p + 3], d[p + 4], d[p + 5]); c.set(d[p + 6], d[p + 7], d[p + 8]);
      centre.set(x + dx, y, z + dz);
      tri.closestPointToPoint(centre, near);
      const gap = near.distanceTo(centre);
      if (gap >= radius) return;
      let nx = centre.x - near.x, nz = centre.z - near.z, length = Math.hypot(nx, nz);
      if (length < 1e-5) {                                 // the centre is on the face: out along its normal
        tri.getNormal(near);
        nx = near.x; nz = near.z; length = Math.hypot(nx, nz);
        if (length < 1e-5) return;                         // a floor or a ceiling: not a wall
      }
      const push = radius - gap;
      dx += nx / length * push; dz += nz / length * push;
    });
    return [dx, dz];
  }
}

const WALK = 4, SPRINT = 7, GRAVITY = 9.8;     // m/s, as the Godot runtime's player

export class Walker {
  constructor(world, player, hasGround) {
    this.world = world;
    this.radius = player?.radius ?? 0.3;
    this.height = player?.height ?? 1.8;
    this.eye = player?.eye_height ?? 1.65;
    this.step = player?.step_height ?? 0.3;
    this.reach = player?.reach ?? 1.5;
    this.hasGround = hasGround;
    world.minUp = Math.cos(THREE.MathUtils.degToRad(player?.max_slope_deg ?? 40));
    this.feet = new THREE.Vector3();
    this.start = new THREE.Vector3();
    this.yaw = 0;
    this.pitch = 0;
    this.fall = 0;
    this.eyeY = 0;
  }

  // Yaw 0 looks along -Z, as a camera does.
  place(position, yaw = this.yaw) {
    this.feet.copy(position);
    this.start.copy(position);
    this.yaw = yaw;
    this.pitch = 0;
    this.fall = 0;
    this.feet.y = Math.max(this.groundUnder(position.y + this.step), position.y - 0.5);
    this.eyeY = this.feet.y + this.eye;
  }

  groundUnder(top) {
    const r = this.radius * 0.7, { x, z } = this.feet;
    let best = Math.max(this.hasGround && top >= 0 ? 0 : -Infinity, this.world.groundAt(x, z, top));
    // The rim keeps the body from dropping off an edge; only its middle climbs, or it would scale furniture.
    const rim = Math.min(top, this.feet.y + 0.03);
    for (const [dx, dz] of [[r, 0], [-r, 0], [0, r], [0, -r]]) {
      best = Math.max(best, this.world.groundAt(x + dx, z + dz, rim));
    }
    return best;
  }

  // One frame of movement: forward and strafe in -1..1.
  move(forward, strafe, sprint, dt) {
    const speed = sprint ? SPRINT : WALK;
    const length = Math.hypot(forward, strafe) || 1;
    const sin = Math.sin(this.yaw), cos = Math.cos(this.yaw);
    const wishX = (-sin * forward + cos * strafe) / length * speed;
    const wishZ = (-cos * forward - sin * strafe) / length * speed;
    const steps = Math.max(1, Math.ceil(speed * dt / (this.radius * 0.5)));
    for (let s = 0; s < steps; s++) this.advance(wishX * dt / steps, wishZ * dt / steps, dt / steps);
    this.eyeY += (this.feet.y + this.eye - this.eyeY) * Math.min(1, dt * 14);   // steps don't jolt the view
    if (this.feet.y < -60) this.place(this.start);
  }

  advance(dx, dz, dt) {
    this.feet.x += dx; this.feet.z += dz;
    const low = this.step + this.radius, high = Math.max(low, this.height - this.radius);
    const rows = Math.max(1, Math.ceil((high - low) / this.radius));
    for (let pass = 0; pass < 3; pass++) {
      let pushed = false;
      for (let row = 0; row <= rows; row++) {
        const y = this.feet.y + low + (high - low) * row / rows;
        const [px, pz] = this.world.pushOut(this.feet.x, y, this.feet.z, this.radius);
        if (px || pz) { this.feet.x += px; this.feet.z += pz; pushed = true; }
      }
      if (!pushed) break;
    }
    const ground = this.groundUnder(this.feet.y + this.step);
    if (ground >= this.feet.y - 0.03) {
      this.feet.y = ground;
      this.fall = 0;
    } else {
      this.fall += GRAVITY * dt;
      this.feet.y = Math.max(ground, this.feet.y - this.fall * dt);
    }
  }

  applyTo(camera) {
    camera.position.set(this.feet.x, this.eyeY, this.feet.z);
    camera.rotation.set(this.pitch, this.yaw, 0, 'YXZ');
  }
}
