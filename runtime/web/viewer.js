// Geogen web viewer: a geogen GLB in Three.js, orbit camera, part inspector (extras.geogen)
// and the interaction animations the export writes (one glTF animation per transition).
//
//   index.html?m=armchair            models/armchair.glb beside the viewer's folder (../models/)
//   index.html?src=URL               any GLB
//   &embed=1                         just the model (the catalogue's asset pages)

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const params = new URLSearchParams(location.search);
const MODELS = params.get('models') || '../models/';
const embed = params.has('embed');
document.body.classList.toggle('embed', embed);

const canvas = document.getElementById('view');
const status = document.getElementById('status');
const select = document.getElementById('model');
const actions = document.getElementById('actions');
const info = document.getElementById('info');

// -- renderer, scene, lights ---------------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.AgXToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
scene.background = new THREE.Color(dark ? 0x1e2125 : 0xdfe4ea);
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
let model = null, mixer = null, highlight = null;

function geogen(object) { return object.userData?.geogen; }

async function load(url, label) {
  status.textContent = `Loading ${label}…`;
  actions.hidden = info.hidden = true;
  let gltf;
  try {
    gltf = await loader.loadAsync(url);
  } catch (err) {
    status.textContent = `Couldn't load ${label}: ${err.message || err}`;
    return;
  }
  if (model) scene.remove(model);
  clearHighlight();
  model = gltf.scene;
  model.traverse(o => {
    if (geogen(o)?.type === 'collider') o.visible = false;   // physics shapes, not for show
    if (o.isMesh) { o.castShadow = o.receiveShadow = true; }
  });
  scene.add(model);
  frame(model);
  setupAnimations(gltf.animations);
  let tris = 0;
  model.traverse(o => { if (o.isMesh && o.visible) tris += (o.geometry.index?.count ?? o.geometry.attributes.position.count) / 3; });
  status.textContent = `${label} · ${Math.round(tris).toLocaleString()} triangles`;
}

function frame(object) {
  const box = new THREE.Box3();
  object.traverse(o => { if (o.isMesh && o.visible) box.expandByObject(o); });
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

// -- interaction animations ----------------------------------------------------------------

let clips = [];

function setupAnimations(list) {
  clips = list;
  mixer = clips.length ? new THREE.AnimationMixer(model) : null;
  actions.replaceChildren();
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

// -- inspector -----------------------------------------------------------------------------

const raycaster = new THREE.Raycaster();
let down = null;
canvas.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', e => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || embed || !model) return;
  const rect = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2((e.clientX - rect.left) / rect.width * 2 - 1, -(e.clientY - rect.top) / rect.height * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObject(model, true).find(h => isShown(h.object));
  if (hit) inspect(hit.object); else { clearHighlight(); info.hidden = true; }
});
info.querySelector('.close').onclick = () => { info.hidden = true; clearHighlight(); };

function isShown(object) {
  for (let o = object; o; o = o.parent) if (!o.visible) return false;
  return true;
}

function inspect(mesh) {
  // The nearest node the export gave extras to: the part the mesh belongs to.
  let node = mesh;
  while (node.parent && node.parent !== model && !geogen(node)) node = node.parent;
  const path = [];
  for (let o = node; o && o !== model; o = o.parent) path.unshift(o.name);
  document.getElementById('info-name').textContent = node.name || '(unnamed)';
  document.getElementById('info-path').textContent = path.join(' / ');
  const extras = geogen(node);
  document.getElementById('info-extras').textContent = extras ? JSON.stringify(extras, null, 1) : 'no extras.geogen';
  info.hidden = false;
  clearHighlight();
  highlight = new THREE.BoxHelper(node, new THREE.Color(0xe58a5e));
  scene.add(highlight);
}

function clearHighlight() {
  if (highlight) { scene.remove(highlight); highlight.dispose(); highlight = null; }
}

// -- model list ----------------------------------------------------------------------------

async function start() {
  resize();
  renderer.setAnimationLoop(() => {
    const dt = clock.getDelta();
    if (mixer) mixer.update(dt);
    if (highlight) highlight.update();
    controls.update();
    renderer.render(scene, camera);
  });
  const src = params.get('src');
  if (src) { select.hidden = true; return load(src, src.split('/').pop()); }
  let names = [];
  try {
    names = (await (await fetch(`${MODELS}index.json`)).json()).assets;
  } catch { /* no index: just the one asked for */ }
  const first = params.get('m') || names[0];
  if (!first) { status.textContent = 'No models (run docs/make_models.py)'; return; }
  if (!names.includes(first)) names.unshift(first);
  for (const name of names) select.add(new Option(name, name, false, name === first));
  select.onchange = () => {
    history.replaceState(null, '', `?m=${encodeURIComponent(select.value)}`);
    load(`${MODELS}${select.value}.glb`, select.value);
  };
  load(`${MODELS}${first}.glb`, first);
}

start();
