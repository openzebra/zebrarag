import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const canvas = document.getElementById('canvas');
const host = canvas.parentElement;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setPixelRatio(1);
renderer.setClearColor(0xffffff, 1);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(6, 4, 8);
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.25;
controls.target.set(0, 0, 0);
controls.minDistance = 5;
controls.maxDistance = 18;

const root = new THREE.Group();
scene.add(root);

const black = new THREE.LineBasicMaterial({ color: 0x000000 });
const gray = new THREE.LineBasicMaterial({ color: 0x777777 });
const faint = new THREE.LineBasicMaterial({ color: 0xb0b0b0 });
const fillWhite = new THREE.MeshBasicMaterial({ color: 0xffffff });
const fillBlack = new THREE.MeshBasicMaterial({ color: 0x000000 });

function line(a, b, material = black) {
  const geometry = new THREE.BufferGeometry().setFromPoints([a, b]);
  const object = new THREE.Line(geometry, material);
  root.add(object);
  return object;
}

function wireBox(width, height, depth, material = black) {
  const geometry = new THREE.BoxGeometry(width, height, depth);
  const edges = new THREE.EdgesGeometry(geometry);
  return new THREE.LineSegments(edges, material);
}

function node(x, y, z, radius = 0.08, filled = false) {
  const group = new THREE.Group();
  const geometry = new THREE.SphereGeometry(radius, 10, 6);
  if (filled) {
    group.add(new THREE.Mesh(geometry, fillBlack));
  } else {
    group.add(new THREE.LineSegments(new THREE.EdgesGeometry(geometry), black));
  }
  group.position.set(x, y, z);
  root.add(group);
  return group;
}

function textSprite(text) {
  const canvas2d = document.createElement('canvas');
  canvas2d.width = 256;
  canvas2d.height = 64;
  const ctx = canvas2d.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas2d.width, canvas2d.height);
  ctx.strokeStyle = '#000000';
  ctx.strokeRect(0, 0, canvas2d.width, canvas2d.height);
  ctx.fillStyle = '#000000';
  ctx.font = '20px monospace';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 12, 34);
  const texture = new THREE.CanvasTexture(canvas2d);
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture }));
  sprite.scale.set(1.55, 0.38, 1);
  return sprite;
}

// floor grid: deliberately boring, CAD/UNIX style.
for (let i = -6; i <= 6; i++) {
  line(new THREE.Vector3(i, -2.2, -4), new THREE.Vector3(i, -2.2, 4), i === 0 ? gray : faint);
  line(new THREE.Vector3(-6, -2.2, i), new THREE.Vector3(6, -2.2, i), i === 0 ? gray : faint);
}

// vertical pipeline spine.
line(new THREE.Vector3(0, -2.0, 0), new THREE.Vector3(0, 2.4, 0), black);

const stages = [
  { y: -1.8, label: 'src' },
  { y: -1.05, label: 'ast' },
  { y: -0.3, label: 'chunks' },
  { y: 0.45, label: 'embed' },
  { y: 1.2, label: 'store' },
  { y: 1.95, label: 'query' },
];

const stageNodes = stages.map((stage, index) => {
  const n = node(0, stage.y, 0, index === 5 ? 0.14 : 0.1, index === 5);
  const label = textSprite(`${index + 1}:${stage.label}`);
  label.position.set(1.35, stage.y, 0);
  root.add(label);
  return n;
});

for (let i = 0; i < stageNodes.length - 1; i++) {
  line(stageNodes[i].position, stageNodes[i + 1].position, black);
}

// source files as plain boxes.
const sourceBoxes = [];
['rs', 'ts', 'py', 'go'].forEach((name, i) => {
  const angle = (i / 4) * Math.PI * 2;
  const x = Math.cos(angle) * 1.25;
  const z = Math.sin(angle) * 1.25;
  const box = wireBox(0.35, 0.48, 0.08, black);
  box.position.set(x, stages[0].y, z);
  box.rotation.y = -angle;
  root.add(box);
  sourceBoxes.push(box);
  line(box.position, stageNodes[0].position, gray);
  const tag = textSprite(`.${name}`);
  tag.scale.set(0.62, 0.16, 1);
  tag.position.set(x, stages[0].y + 0.42, z);
  root.add(tag);
});

// AST tree.
const astRoot = stageNodes[1];
const astChildren = [];
for (let i = 0; i < 5; i++) {
  const angle = (i / 5) * Math.PI * 2;
  const child = node(Math.cos(angle) * 1.15, stages[1].y + 0.05, Math.sin(angle) * 1.15, 0.07);
  astChildren.push(child);
  line(astRoot.position, child.position, black);
  for (let j = 0; j < 2; j++) {
    const leafAngle = angle + (j ? 0.22 : -0.22);
    const leaf = node(Math.cos(leafAngle) * 1.75, stages[1].y - 0.08, Math.sin(leafAngle) * 1.75, 0.045);
    line(child.position, leaf.position, gray);
  }
}

// chunks as stacked slabs.
const slabs = [];
for (let i = 0; i < 6; i++) {
  const slab = wireBox(1.25 - i * 0.08, 0.05, 0.42, i % 2 ? gray : black);
  slab.position.set((i - 2.5) * 0.04, stages[2].y - 0.22 + i * 0.09, 0);
  root.add(slab);
  slabs.push(slab);
}

// embedding points.
const embedPositions = new Float32Array(96 * 3);
for (let i = 0; i < 96; i++) {
  embedPositions[i * 3] = (Math.random() - 0.5) * 2.0;
  embedPositions[i * 3 + 1] = stages[3].y + (Math.random() - 0.5) * 0.55;
  embedPositions[i * 3 + 2] = (Math.random() - 0.5) * 2.0;
}
const embedGeometry = new THREE.BufferGeometry();
embedGeometry.setAttribute('position', new THREE.BufferAttribute(embedPositions, 3));
const embedPoints = new THREE.Points(embedGeometry, new THREE.PointsMaterial({ color: 0x000000, size: 0.035 }));
root.add(embedPoints);

// vector store box + graph.
const db = wireBox(1.15, 0.48, 0.72, black);
db.position.set(0, stages[4].y, 0);
root.add(db);
const graphNodes = [];
for (let i = 0; i < 10; i++) {
  const angle = (i / 10) * Math.PI * 2;
  const n = node(Math.cos(angle) * 1.65, stages[4].y + 0.58, Math.sin(angle) * 1.65, 0.045);
  graphNodes.push(n);
}
for (let i = 0; i < graphNodes.length; i++) {
  line(graphNodes[i].position, graphNodes[(i + 1) % graphNodes.length].position, gray);
  line(graphNodes[i].position, graphNodes[(i + 3) % graphNodes.length].position, faint);
}

// query results.
const results = [];
for (let i = 0; i < 5; i++) {
  const box = wireBox(0.34, 0.2, 0.1, i === 0 ? black : gray);
  box.position.set(-0.85 + i * 0.42, stages[5].y + 0.45, 0);
  root.add(box);
  results.push(box);
  line(stageNodes[5].position, box.position, i === 0 ? black : gray);
}

// zebra stripes wrapping the object. Simple black arcs, no glow.
const stripes = new THREE.Group();
for (let i = 0; i < 9; i++) {
  const arc = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(
      Array.from({ length: 40 }, (_, p) => {
        const a = -Math.PI * 0.72 + (p / 39) * Math.PI * 1.44;
        const r = 2.25 + (i % 2) * 0.18;
        return new THREE.Vector3(Math.cos(a) * r, -1.8 + i * 0.48, Math.sin(a) * r);
      })
    ),
    i % 2 ? gray : black
  );
  arc.rotation.y = i * 0.42;
  stripes.add(arc);
}
root.add(stripes);

let lastStep = 0;
let active = 0;
function pulse() {
  const object = stageNodes[active % stageNodes.length];
  object.scale.set(1.65, 1.65, 1.65);
  setTimeout(() => object.scale.set(1, 1, 1), 180);
  active += 1;
}

const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();
  const dt = clock.getDelta();

  root.rotation.y += dt * 0.08;
  stripes.rotation.y -= dt * 0.1;
  sourceBoxes.forEach((box, i) => { box.rotation.y += dt * (0.25 + i * 0.04); });
  astChildren.forEach((child, i) => { child.position.y = stages[1].y + 0.05 + Math.sin(t * 1.4 + i) * 0.035; });
  slabs.forEach((slab, i) => { slab.position.x = (i - 2.5) * 0.04 + Math.sin(t * 1.2 + i) * 0.025; });
  graphNodes.forEach((n, i) => { n.scale.setScalar(1 + Math.sin(t * 1.7 + i) * 0.12); });
  results.forEach((r, i) => { r.position.y = stages[5].y + 0.45 + Math.sin(t * 1.8 + i) * 0.035; });

  if (t - lastStep > 0.75) {
    lastStep = t;
    pulse();
  }

  controls.update();
  renderer.render(scene, camera);
}

function resize() {
  const rect = host.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

resize();
window.addEventListener('resize', resize);
if ('ResizeObserver' in window) {
  new ResizeObserver(resize).observe(host);
}
animate();
