// Geogen web viewer: a geogen GLB in Three.js. Orbit it or walk through it, use its doors and switches
// (extras.geogen.interactions), light its fixtures from the manifest's state, and look things up by room and tag.
//
//   index.html?s=cottage             models/cottage.scene.glb and its manifest (docs/make_models.py)
//   index.html?m=armchair            models/armchair.glb beside the viewer's folder (../models/)
//   index.html?src=URL               any GLB (its .manifest.json is read when there is one)
//   &walk=1                          start walking; &night=1 start after dark
//   &embed=1                         just the model (the catalogue's asset pages)

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { WorldState } from './rules.js';
import { collectInteractions, geogen, nodeName } from './interactions.js';
import { Lights } from './lights.js';
import { CollisionWorld, Walker } from './walker.js';

const params = new URLSearchParams(location.search);
const MODELS = params.get('models') || '../models/';
const embed = params.has('embed');
document.body.classList.toggle('embed', embed);

const $ = id => document.getElementById(id);
const canvas = $('view'), status = $('status'), select = $('model'), actions = $('actions'), info = $('info');
const walkButton = $('walk'), nightButton = $('night'), cutawayButton = $('cutaway');
const exploreButton = $('explore-toggle'), explore = $('explore');
const reticle = $('reticle'), promptLine = $('prompt'), hint = $('hint');

const ORBIT_HINT = 'Drag to orbit · right-drag to pan · scroll to zoom · click a part to inspect it';
const WALK_HINT = 'WASD to walk · Shift to run · E or click to use · N for night · V to stop walking';
const LOOK_HINT = 'Click the view to look around · WASD to walk · V to stop walking';
hint.textContent = ORBIT_HINT;

// -- renderer, scene, lights ---------------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.AgXToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const DAY_SKY = new THREE.Color(dark ? 0x1e2125 : 0xdfe4ea), NIGHT_SKY = new THREE.Color(0x0a0d14);
scene.background = DAY_SKY.clone();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.6;

const sun = new THREE.DirectionalLight(0xfff4e6, 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.02;
scene.add(sun, sun.target);

const ground = new THREE.Mesh(new THREE.CircleGeometry(1, 64).rotateX(-Math.PI / 2),
                              new THREE.ShadowMaterial({ opacity: dark ? 0.45 : 0.22 }));
ground.receiveShadow = true;
scene.add(ground);
let grid = null;   // 1 m squares, sized to the model

const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

const lights = new Lights(scene);

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

// -- loading -------------------------------------------------------------------------------

const loader = new GLTFLoader();
const clock = new THREE.Clock();
let model = null, mixer = null, manifest = null, state = new WorldState(null);
let interactions = [], rooms = [], world = null, walker = null, label = '', triangles = 0, loading = 0;

const CUTAWAY_NAMES = new Set(['ceiling', 'roof', 'gables', 'chimney', 'chimney_cap', 'cornice']);
let cutawayMeshes = [];

function isHiddenOnLoad(node) {
  const extras = geogen(node);
  // Physics shapes, an inventory's icons and the templates flows copy aren't part of the world.
  return extras?.type === 'collider' || extras?.type === 'lod' || extras?.item_icon || extras?.item;
}

async function readManifest(url) {
  try {
    const response = await fetch(url.replace(/\.glb(\?.*)?$/, '.manifest.json'));
    return response.ok ? await response.json() : null;
  } catch { return null; }
}

function dispose(root) {
  root.traverse(o => {
    o.geometry?.dispose();
    for (const material of [o.material].flat()) material?.dispose();
  });
}

async function load(url, name) {
  const ticket = ++loading;
  status.textContent = `Loading ${name}…`;
  actions.hidden = info.hidden = true;
  let gltf, nextManifest;
  try {
    [gltf, nextManifest] = await Promise.all([
      loader.loadAsync(url, e => {
        if (ticket === loading && e.total) status.textContent = `Loading ${name}… ${Math.round(e.loaded / e.total * 100)}%`;
      }),
      readManifest(url),
    ]);
  } catch (err) {
    if (ticket === loading) status.textContent = `Couldn't load ${name}: ${err.message || err}`;
    return;
  }
  if (ticket !== loading) { dispose(gltf.scene); return; }    // another model was picked meanwhile

  stopWalking();
  clearHighlights();
  inspected = null;
  if (model) { scene.remove(model); dispose(model); }
  model = gltf.scene;
  manifest = nextManifest;
  label = name;
  model.traverse(o => {
    if (isHiddenOnLoad(o)) o.visible = false;
    if (o.isMesh) o.castShadow = o.receiveShadow = true;
  });
  scene.add(model);
  frame(model);

  state = new WorldState(manifest);
  interactions = collectInteractions(model, state);
  lights.load(model, state, interactions);
  state.listeners.add(() => { lights.sync(); refreshUses(); });
  state.settle();
  lights.sync();
  setupAnimations(gltf.animations);
  poseCharacters(gltf.animations);

  const moving = new Set(interactions.flatMap(it => [...it.parts]));
  world = new CollisionWorld(model, moving);
  walker = new Walker(world, manifest?.player, manifest?.ground !== false);
  collectRooms();
  cutawayMeshes = findCutaway();
  setCutaway(false);
  buildExplore();

  triangles = 0;
  model.traverse(o => { if (o.isMesh && o.visible) triangles += (o.geometry.index?.count ?? o.geometry.attributes.position.count) / 3; });
  showStatus();
  walkButton.hidden = embed;
  nightButton.hidden = embed || !lights.fixtures.length;
  if (params.has('walk') && !embed) startWalking();
}

function showStatus(place = '') {
  status.textContent = [label, place || `${Math.round(triangles).toLocaleString()} triangles`].join(' · ');
}

function frame(object) {
  const box = new THREE.Box3();
  object.traverse(o => { if (o.isMesh && isShown(o)) box.expandByObject(o); });
  const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.length() / 2, 0.05);
  // The catalogue's iso view: in front (+Z), to the right and above.
  const dir = new THREE.Vector3(1, 0.8, 1.4).normalize();
  const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.05;
  camera.position.copy(centre).addScaledVector(dir, distance);
  camera.near = distance / 100; camera.far = distance * 100;
  camera.updateProjectionMatrix();
  controls.target.copy(centre);
  controls.update();

  const floor = Math.min(box.min.y, 0);
  const extent = Math.max(size.x, size.z, 1) * 3;
  ground.scale.setScalar(extent); ground.position.y = floor;
  if (grid) { scene.remove(grid); grid.dispose(); }
  const cells = 2 * Math.ceil(extent / 2);
  grid = new THREE.GridHelper(cells, cells, 0x8a8f96, 0x8a8f96);
  grid.position.set(Math.round(centre.x), floor + 0.001, Math.round(centre.z));
  grid.material.opacity = 0.3; grid.material.transparent = true;
  scene.add(grid);
  sun.position.copy(centre).add(new THREE.Vector3(0.5, 1.2, 0.8).multiplyScalar(radius * 4));
  sun.target.position.copy(centre);
  const cam = sun.shadow.camera;
  cam.left = cam.bottom = -radius * 1.5; cam.right = cam.top = radius * 1.5;
  cam.near = radius; cam.far = radius * 8;
  cam.updateProjectionMatrix();
}

// People stand: the export's bind pose is a T, and the viewer doesn't run their behaviours.
function poseCharacters(list) {
  const npcs = [];
  model.traverse(o => { if (geogen(o)?.type === 'npc') npcs.push(o); });
  for (const npc of npcs) {
    npc.traverse(o => {
      const pose = geogen(o)?.clips?.find(c => c.name === 'pose_stand');
      const clip = pose && list.find(c => c.name === pose.animation);
      if (!clip) return;
      mixer ??= new THREE.AnimationMixer(model);
      const action = mixer.clipAction(clip);
      action.setLoop(THREE.LoopOnce, 1);
      action.clampWhenFinished = true;
      action.play();
    });
  }
}

// -- what can be done here: the interactions, or an export's plain animations ----------------

let clips = [];

function setupAnimations(list) {
  clips = list;
  mixer = null;
  actions.replaceChildren();
  if (interactions.length) { buildUses(); return; }
  clips = list.filter(c => c.name.includes('/'));      // <node>/<interaction>/<from>-><to>, for a GLB without extras
  if (clips.length) mixer = new THREE.AnimationMixer(model);
  if (clips.length > 6) {     // a building's worth of doors: pick one from a list
    const picker = document.createElement('select');
    picker.setAttribute('aria-label', 'Animation');
    for (const clip of clips) picker.add(new Option(clip.name));
    const button = document.createElement('button');
    button.textContent = 'Play';
    button.onclick = () => play(clips[picker.selectedIndex]);
    actions.append(picker, button);
  } else {
    for (const clip of clips) {
      const button = document.createElement('button');
      button.textContent = clip.name;     // e.g. door_2/swing/closed->open
      button.title = 'Play';
      button.onclick = () => play(clip);
      actions.append(button);
    }
  }
  actions.hidden = !clips.length;
}

function play(clip) {
  // Clips are named <node>/<interaction>/<from>-><to>: stop the node's other transitions, leave other nodes be.
  const node = clip.name.split('/')[0];
  for (const other of clips) if (other.name.split('/')[0] === node) mixer.existingAction(other)?.stop();
  const action = mixer.clipAction(clip);
  action.setLoop(THREE.LoopOnce, 1);
  action.clampWhenFinished = true;
  action.reset().play();
}

const usable = () => interactions.filter(it => it.prompt);
const useLabel = it => `${it.label}: ${it.prompt}`;
let usePicker = null, useButtons = [];

function buildUses() {
  actions.replaceChildren();
  usePicker = null; useButtons = [];
  const list = interactions.filter(it => Object.values(it.states).some(s => 'next' in s));
  if (list.length > 6) {
    usePicker = document.createElement('select');
    usePicker.setAttribute('aria-label', 'Thing to use');
    usePicker.list = list;
    const button = document.createElement('button');
    button.textContent = 'Use';
    button.onclick = () => list[usePicker.selectedIndex]?.use();
    actions.append(usePicker, button);
  } else {
    for (const it of list) {
      const button = document.createElement('button');
      button.onclick = () => it.use();
      button.interaction = it;
      useButtons.push(button);
      actions.append(button);
    }
  }
  actions.hidden = !list.length;
  refreshUses();
}

function refreshUses() {
  if (usePicker) {
    const at = usePicker.selectedIndex;
    usePicker.replaceChildren(...usePicker.list.map(it => new Option(useLabel(it))));
    usePicker.selectedIndex = Math.max(at, 0);
  }
  for (const button of useButtons) {
    button.textContent = useLabel(button.interaction);
    button.disabled = !button.interaction.usable;
  }
  refreshLightRows();
  if (!info.hidden && inspected) showUses(inspected);
}

// -- walking -------------------------------------------------------------------------------

const keys = new Set();
let walking = false, saved = null, aimed = null;
const aimRay = new THREE.Raycaster();

function spawnPoint() {
  const spawn = manifest?.spawns?.[0];
  if (spawn) {
    const [fx, , fz] = spawn.forward ?? [0, 0, -1];
    return { position: new THREE.Vector3().fromArray(spawn.position), yaw: Math.atan2(-fx, -fz) };
  }
  const marker = [];
  model.traverse(o => { if (geogen(o)?.type === 'spawn') marker.push(o); });
  if (marker.length) {
    // A spawn faces its node's +Z.
    const forward = new THREE.Vector3(0, 0, 1).transformDirection(marker[0].matrixWorld);
    return { position: marker[0].getWorldPosition(new THREE.Vector3()), yaw: Math.atan2(-forward.x, -forward.z) };
  }
  const box = new THREE.Box3().setFromObject(model);     // 3 m in front of the model, facing it, like the Godot runtime
  return { position: new THREE.Vector3((box.min.x + box.max.x) / 2, Math.max(box.min.y, 0), box.max.z + 3), yaw: 0 };
}

function startWalking() {
  if (walking || !walker) return;
  walking = true;
  saved = { position: camera.position.clone(), target: controls.target.clone(), near: camera.near, far: camera.far, fov: camera.fov };
  controls.enabled = false;
  const { position, yaw } = spawnPoint();
  walker.place(position, yaw);
  camera.fov = 70; camera.near = 0.05; camera.far = Math.max(saved.far, 500);
  camera.updateProjectionMatrix();
  ground.visible = false;
  if (grid) grid.visible = false;
  document.body.classList.add('walking');
  walkButton.setAttribute('aria-pressed', 'true');
  reticle.hidden = false;
  setExplore(false);      // the view is the whole window now; the Explore button brings the panel back
  hint.textContent = LOOK_HINT;
  info.hidden = true;
  clearHighlights();
}

function stopWalking() {
  if (!walking) return;
  walking = false;
  if (document.pointerLockElement) document.exitPointerLock();
  camera.fov = saved.fov; camera.near = saved.near; camera.far = saved.far;
  camera.position.copy(saved.position);
  camera.rotation.set(0, 0, 0);
  camera.updateProjectionMatrix();
  controls.target.copy(saved.target);
  controls.enabled = true;
  controls.update();
  ground.visible = true;
  if (grid) grid.visible = true;
  document.body.classList.remove('walking');
  walkButton.setAttribute('aria-pressed', 'false');
  reticle.hidden = promptLine.hidden = true;
  hint.textContent = ORBIT_HINT;
  keys.clear();
  aimed = null;
  showStatus();
}

walkButton.onclick = () => { walking ? stopWalking() : startWalking(); walkButton.blur(); };

document.addEventListener('pointerlockchange', () => {
  if (walking) hint.textContent = document.pointerLockElement === canvas ? WALK_HINT : LOOK_HINT;
});

function look(dx, dy) {
  walker.yaw -= dx * 0.0024;
  walker.pitch = THREE.MathUtils.clamp(walker.pitch - dy * 0.0024, -1.45, 1.45);
}

let dragging = false;
canvas.addEventListener('pointermove', e => {
  if (!walking) return;
  if (document.pointerLockElement === canvas || dragging) look(e.movementX, e.movementY);
});

addEventListener('keydown', e => {
  if (e.target instanceof HTMLSelectElement || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.code === 'KeyV' && !e.repeat && walker && !embed) { walking ? stopWalking() : startWalking(); return; }
  if (e.code === 'KeyN' && !e.repeat && !nightButton.hidden) { setNight(!lights.night); return; }
  if (!walking) return;
  keys.add(e.code);
  if (e.code === 'KeyE' && !e.repeat) aimed?.use();
  if (e.code.startsWith('Arrow') || e.code === 'Space') e.preventDefault();
});
addEventListener('keyup', e => keys.delete(e.code));
addEventListener('blur', () => keys.clear());

const toCentre = new THREE.Vector3(), sphere = new THREE.Sphere();

// What the player is aiming at: the nearest usable part in reach with nothing solid in front of it. A part the aim
// only just misses counts too, so a light switch doesn't take pixel-exact aiming.
function findAim() {
  const eye = camera.position, dir = camera.getWorldDirection(new THREE.Vector3());
  aimRay.set(eye, dir);
  let best = null, bestScore = Infinity;
  for (const it of interactions) {
    if (!it.prompt) continue;
    const reach = it.reach || walker.reach;
    aimRay.far = reach;
    for (const mesh of it.meshes) {
      if (!isShown(mesh)) continue;
      if (!mesh.geometry.boundingSphere) mesh.geometry.computeBoundingSphere();
      sphere.copy(mesh.geometry.boundingSphere).applyMatrix4(mesh.matrixWorld);
      const distance = eye.distanceTo(sphere.center);
      if (distance > reach + sphere.radius) continue;
      const hit = aimRay.intersectObject(mesh, false)[0];
      if (hit) {
        if (hit.distance < bestScore && world.raycast(eye, dir, hit.distance) >= hit.distance - 0.12) {
          best = it; bestScore = hit.distance;
        }
        continue;
      }
      toCentre.copy(sphere.center).sub(eye).normalize();
      if (distance <= reach && toCentre.dot(dir) > Math.cos(THREE.MathUtils.degToRad(5))) {
        const score = distance + 0.5;       // a direct hit wins
        if (score < bestScore && world.raycast(eye, toCentre, distance) >= distance - 0.15) { best = it; bestScore = score; }
      }
    }
  }
  return best;
}

function updateWalk(dt) {
  const axis = (plus, minus) => (keys.has(plus[0]) || keys.has(plus[1]) ? 1 : 0) - (keys.has(minus[0]) || keys.has(minus[1]) ? 1 : 0);
  const forward = axis(['KeyW', 'ArrowUp'], ['KeyS', 'ArrowDown']), strafe = axis(['KeyD', 'ArrowRight'], ['KeyA', 'ArrowLeft']);
  walker.move(forward, strafe, keys.has('ShiftLeft') || keys.has('ShiftRight'), dt);
  walker.applyTo(camera);
  camera.updateMatrixWorld();
  aimed = findAim();
  reticle.classList.toggle('on', !!aimed);
  promptLine.hidden = !aimed;
  if (aimed) {
    const text = aimed.prompt;
    if (promptLine.dataset.text !== text) {
      promptLine.dataset.text = text;
      const key = document.createElement('kbd');
      key.textContent = 'E';
      promptLine.replaceChildren(key, text);
    }
  }
  const room = roomAt(walker.feet.clone().setY(walker.feet.y + 1));
  showStatus(room ? roomName(room) : 'outside');
}

// -- night and cutaway ---------------------------------------------------------------------

function setNight(on) {
  lights.night = on;
  sun.intensity = on ? 0.08 : 2.2;
  sun.color.set(on ? 0x8fa6d8 : 0xfff4e6);
  scene.environmentIntensity = on ? 0.05 : 0.6;
  scene.background.copy(on ? NIGHT_SKY : DAY_SKY);
  nightButton.setAttribute('aria-pressed', String(on));
  lights.sync();
  refreshLightRows();
}
nightButton.onclick = () => { setNight(!lights.night); nightButton.blur(); };

// Ceilings, roofs and what hangs from them, as render.py picks them; exports number repeats (ceiling_2). Only the
// meshes go: a hidden pendant still lights the room.
function findCutaway() {
  const out = [];
  const visit = node => {
    const tags = geogen(node)?.tags ?? [];
    if (CUTAWAY_NAMES.has(nodeName(node).replace(/_\d+$/, '')) || tags.includes('ceiling') || tags.includes('light.ceiling')) {
      node.traverse(o => { if (o.isMesh && geogen(o)?.type !== 'collider') out.push(o); });
      return;
    }
    node.children.forEach(visit);
  };
  visit(model);
  return out;
}

function setCutaway(on) {
  for (const mesh of cutawayMeshes) mesh.visible = !on;
  cutawayButton.setAttribute('aria-pressed', String(on));
  cutawayButton.hidden = embed || !cutawayMeshes.length;
}
cutawayButton.onclick = () => { setCutaway(cutawayButton.getAttribute('aria-pressed') !== 'true'); cutawayButton.blur(); };

// -- explore: rooms, lights, tags ----------------------------------------------------------

const roomName = room => room.id.replaceAll('_', ' ');

function collectRooms() {
  rooms = [];
  model.traverse(o => {
    const extras = geogen(o);
    if (extras?.type === 'room_volume' && extras.size) rooms.push({ id: extras.room?.id ?? nodeName(o), type: extras.room?.type ?? '', node: o, size: extras.size });
  });
}

function roomAt(point) {
  const local = new THREE.Vector3();
  return rooms.find(room => {
    room.node.worldToLocal(local.copy(point));
    return Math.abs(local.x) <= room.size[0] / 2 && Math.abs(local.y) <= room.size[1] / 2 && Math.abs(local.z) <= room.size[2] / 2;
  }) ?? null;
}

function row(text, note, onclick) {
  const item = document.createElement('li'), button = document.createElement('button');
  button.type = 'button';
  const name = document.createElement('span'), detail = document.createElement('span');
  name.className = 'name'; name.textContent = text; name.title = text;
  detail.className = 'note'; detail.textContent = note;
  button.append(name, detail);
  button.onclick = () => onclick(button);
  item.append(button);
  return item;
}

function pressOnly(list, button) {
  for (const other of list.querySelectorAll('button')) if (other !== button) other.setAttribute('aria-pressed', 'false');
  const on = button.getAttribute('aria-pressed') !== 'true';
  button.setAttribute('aria-pressed', String(on));
  return on;
}

let lightRows = [];

function buildExplore() {
  $('rooms').replaceChildren(...rooms.map(room => row(roomName(room), `${(room.size[0] * room.size[2]).toFixed(0)} m²`, button => {
    if (pressOnly($('rooms'), button)) goToRoom(room); else clearHighlights();
  })));
  $('rooms-section').hidden = !rooms.length;

  lightRows = lights.fixtures.map(fixture => {
    const item = row(fixture.name.replaceAll('_', ' '), '', () => lights.toggle(fixture));
    return { fixture, button: item.firstChild, note: item.querySelector('.note'), item };
  });
  $('lights').replaceChildren(...lightRows.map(r => r.item));
  $('lights-section').hidden = !lightRows.length;
  refreshLightRows();

  const tagged = new Map();     // furniture.bed also counts as furniture
  model.traverse(o => {
    for (const tag of geogen(o)?.tags ?? []) {
      const parts = tag.split('.');
      parts.forEach((_, i) => {
        const prefix = parts.slice(0, i + 1).join('.');
        if (!tagged.has(prefix)) tagged.set(prefix, new Set());
        tagged.get(prefix).add(o);
      });
    }
  });
  const names = [...tagged.keys()].sort();
  $('tags').replaceChildren(...names.map(tag => row(tag, String(tagged.get(tag).size), button => {
    if (pressOnly($('tags'), button)) highlightNodes([...tagged.get(tag)]); else clearHighlights();
  })));
  $('tags-section').hidden = !names.length;

  const any = rooms.length || lightRows.length || names.length;
  exploreButton.hidden = embed || !any;
  setExplore(any && !embed && innerWidth > 640);
}

function refreshLightRows() {
  for (const { fixture, button, note } of lightRows) {
    const lit = lights.isLit(fixture);
    note.textContent = fixture.auto ? (lit ? 'on after dark' : 'off by day') : lit ? 'on' : 'off';
    button.setAttribute('aria-pressed', String(lit && !fixture.auto));
    button.disabled = !!fixture.auto;
  }
}

function setExplore(open) {
  explore.hidden = !open;
  exploreButton.setAttribute('aria-pressed', String(!!open));
}
exploreButton.onclick = () => { setExplore(explore.hidden); exploreButton.blur(); };

function goToRoom(room) {
  const centre = room.node.getWorldPosition(new THREE.Vector3());
  if (walking) {
    walker.place(centre.clone().setY(centre.y - room.size[1] / 2 + 0.05));
    return;
  }
  setCutaway(true);
  highlightRoom(room);
  const reach = Math.hypot(room.size[0], room.size[2]) / 2;
  const distance = reach / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.25;
  camera.position.copy(centre).addScaledVector(new THREE.Vector3(0.45, 1, 0.65).normalize(), distance);
  controls.target.copy(centre);
  controls.update();
}

// -- highlights ----------------------------------------------------------------------------

const HIGHLIGHT = new THREE.Color(0xe58a5e), MAX_BOXES = 200;
let highlights = [];

function clearHighlights() {
  for (const h of highlights) { scene.remove(h); h.geometry?.dispose(); h.material?.dispose(); }
  highlights = [];
}

function highlightNodes(nodes) {
  clearHighlights();
  for (const node of nodes.slice(0, MAX_BOXES)) {
    const box = new THREE.BoxHelper(node, HIGHLIGHT);
    scene.add(box);
    highlights.push(box);
  }
}

function highlightRoom(room) {
  clearHighlights();
  const size = new THREE.Vector3().fromArray(room.size);
  const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x, size.y, size.z)),
                                     new THREE.LineBasicMaterial({ color: HIGHLIGHT }));
  box.matrixAutoUpdate = false;
  box.matrix.copy(room.node.matrixWorld);
  scene.add(box);
  highlights.push(box);
}

// -- inspector -----------------------------------------------------------------------------

const raycaster = new THREE.Raycaster();
let down = null, inspected = null;
canvas.addEventListener('pointerdown', e => {
  down = [e.clientX, e.clientY];
  if (!walking) return;
  dragging = e.pointerType === 'touch';
  if (document.pointerLockElement === canvas) { aimed?.use(); return; }
  if (e.pointerType !== 'touch') canvas.requestPointerLock?.()?.catch?.(() => { dragging = true; });
});
addEventListener('pointerup', () => { dragging = false; });
canvas.addEventListener('pointerup', e => {
  if (walking || !down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || embed || !model) return;
  const rect = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2((e.clientX - rect.left) / rect.width * 2 - 1, -(e.clientY - rect.top) / rect.height * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  raycaster.far = Infinity;
  const hit = raycaster.intersectObject(model, true).find(h => isShown(h.object));
  if (hit) inspect(hit.object, hit.point); else { clearHighlights(); info.hidden = true; inspected = null; }
});
info.querySelector('.close').onclick = () => { info.hidden = true; inspected = null; clearHighlights(); };

function isShown(object) {
  for (let o = object; o; o = o.parent) if (!o.visible) return false;
  return true;
}

function ancestorWith(node, test) {
  for (let o = node; o && o !== model.parent; o = o.parent) if (test(o)) return o;
  return null;
}

function inspect(mesh, point) {
  // The nearest node the export gave extras to: the part the mesh belongs to.
  let node = mesh;
  while (node.parent && node.parent !== model && !geogen(node)) node = node.parent;
  const path = [];
  for (let o = node; o && o !== model; o = o.parent) path.unshift(nodeName(o));
  $('info-name').textContent = nodeName(node) || '(unnamed)';
  $('info-path').textContent = path.join(' / ');

  const thing = ancestorWith(node, o => geogen(o)?.uid);
  const tagged = ancestorWith(node, o => geogen(o)?.tags?.length);
  const room = roomAt(point);
  const facts = [['ID', geogen(thing ?? node)?.uid], ['Tags', geogen(tagged ?? node)?.tags?.join(', ')],
                 ['Room', room && `${roomName(room)}${room.type ? ` (${room.type.replaceAll('_', ' ')})` : ''}`]];
  $('info-facts').replaceChildren(...facts.filter(([, value]) => value).flatMap(([name, value]) => {
    const dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = name; dd.textContent = value;
    return [dt, dd];
  }));

  const extras = geogen(node);
  $('info-extras').textContent = extras ? JSON.stringify(extras, null, 1) : 'no extras.geogen';
  inspected = node;
  showUses(node);
  info.hidden = false;
  highlightNodes([node]);
}

// What can be done to the inspected part: the interactions it belongs to and the fixture it is part of.
function showUses(node) {
  const uses = $('info-uses');
  uses.replaceChildren();
  const chain = new Set();
  for (let o = node; o; o = o.parent) chain.add(o);
  for (const it of interactions) {
    if (!chain.has(it.asset) && !it.targets.some(t => chain.has(t))) continue;
    if (!it.prompt) continue;
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = `${it.prompt} (${it.name}: ${it.target})`;
    button.onclick = () => it.use();
    uses.append(button);
  }
  for (const fixture of lights.fixtures) {
    if (!chain.has(fixture.node) || fixture.auto) continue;
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = lights.isLit(fixture) ? 'Turn the light off' : 'Turn the light on';
    button.onclick = () => { lights.toggle(fixture); showUses(node); };
    uses.append(button);
  }
}

// -- model list ----------------------------------------------------------------------------

function pick(kind, name) {
  const query = new URLSearchParams(location.search);
  query.delete('m'); query.delete('s');
  query.set(kind, name);
  history.replaceState(null, '', `?${query}`);
  load(kind === 's' ? `${MODELS}${name}.scene.glb` : `${MODELS}${name}.glb`, name);
}

async function start() {
  resize();
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.05);
    if (mixer) mixer.update(dt);
    let moved = false;
    for (const it of interactions) moved = it.update(dt) || moved;
    if (moved) { world?.refresh(); refreshUses(); }
    for (const h of highlights) h.update?.();
    if (walking) updateWalk(dt); else controls.update();
    lights.update(dt, camera);
    renderer.render(scene, camera);
  });
  if (params.has('night')) setNight(true);
  const src = params.get('src');
  if (src) { select.hidden = true; return load(src, src.split('/').pop()); }
  let assets = [], scenes = [];
  try {
    const index = await (await fetch(`${MODELS}index.json`)).json();
    assets = index.assets ?? [];
    scenes = index.scenes ?? [];
  } catch { /* no index: just the one asked for */ }
  const wantScene = params.get('s'), wantAsset = params.get('m');
  if (wantScene && !scenes.includes(wantScene)) scenes.unshift(wantScene);
  if (wantAsset && !assets.includes(wantAsset)) assets.unshift(wantAsset);
  const first = wantScene ? ['s', wantScene] : wantAsset ? ['m', wantAsset]
    : scenes.length ? ['s', scenes[0]] : assets.length ? ['m', assets[0]] : null;
  if (!first) { status.textContent = 'No models (run docs/make_models.py)'; return; }
  for (const [kind, title, names] of [['s', 'Scenes', scenes], ['m', 'Assets', assets]]) {
    if (!names.length) continue;
    const group = document.createElement('optgroup');
    group.label = title;
    for (const name of names) group.append(new Option(name, `${kind}:${name}`, false, kind === first[0] && name === first[1]));
    select.append(group);
  }
  select.onchange = () => {
    const [kind, name] = select.value.split(/:(.*)/s);
    select.blur();
    pick(kind, name);
  };
  load(first[0] === 's' ? `${MODELS}${first[1]}.scene.glb` : `${MODELS}${first[1]}.glb`, first[1]);
}

// For automated checks of the page (a browser test can't hold a key down for a measured time).
window.geogenViewer = {
  get walker() { return walker; }, get world() { return world; }, get interactions() { return interactions; },
  get lights() { return lights; }, get state() { return state; }, get rooms() { return rooms; },
  get aimed() { return aimed; }, get camera() { return camera; }, get walking() { return walking; },
  get ready() { return !!walker; }, startWalking, stopWalking, setNight, setCutaway, usable, findAim,
};

start();
