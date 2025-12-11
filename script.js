// Runtime config
ort.env.wasm.numThreads = 2;
ort.env.wasm.simd = true;

const MODEL_URL = 'model.onnx';
const DRAW_SIZE = 320;
const PEN_SIZE = 18;
const PEN_COLOR = '#000';
const CANVAS_BG = '#fff';
const BBOX_THRESHOLD = 250; // pixel < 250 counted as ink
const BIN_THRESHOLD = 0.0;  // binarize: value > threshold -> 1

// DOM elements
const canvas = document.getElementById('draw');
const ctx = canvas.getContext('2d');
const preview = document.getElementById('preview');
const pctx = preview.getContext('2d');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const clearBtn = document.getElementById('clear');
const predictBtn = document.getElementById('predict');

// Helper canvases
const workCanvas = document.createElement('canvas');
const workCtx = workCanvas.getContext('2d');
workCanvas.width = DRAW_SIZE;
workCanvas.height = DRAW_SIZE;

canvas.width = DRAW_SIZE;
canvas.height = DRAW_SIZE;
resetCanvas();

// Model variables
let session = null;
let inputName = null;
let outputName = null;
let layout = 'NCHW'; // or 'NHWC'
let channels = 1;
let targetH = 28;
let targetW = 28;

function setStatus(msg) { statusEl.textContent = msg; }

function resetCanvas() {
  ctx.fillStyle = CANVAS_BG;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  pctx.clearRect(0, 0, preview.width, preview.height);
  resultsEl.textContent = '';
}

// Load model and detect input shape
async function loadModel() {
  setStatus('Chargement du modele...');
  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  inputName = session.inputNames[0];
  outputName = session.outputNames[0];
  const meta = session.inputMetadata[inputName];
  const dims = meta.dimensions.map(d => (typeof d === 'number' && d > 0) ? d : 1);

  // Detect layout
  if (dims.length === 4) {
    if (dims[1] <= 4) {
      layout = 'NCHW';
      channels = dims[1];
      targetH = dims[2];
      targetW = dims[3];
    } else {
      layout = 'NHWC';
      channels = dims[3];
      targetH = dims[1];
      targetW = dims[2];
    }
  } else {
    layout = 'NCHW';
    channels = 1;
    targetH = 28;
    targetW = 28;
  }

  preview.width = targetW;
  preview.height = targetH;
  setStatus(`Modele pret (input ${layout} ${channels}x${targetH}x${targetW}). Dessine puis clique sur Predire.`);
  console.log('Inputs:', session.inputNames, 'Outputs:', session.outputNames, 'Dims:', dims, 'Layout:', layout);
}
loadModel().catch(err => setStatus('Erreur modele: ' + err));

// Drawing
let drawing = false;
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDraw);
canvas.addEventListener('mouseleave', stopDraw);
canvas.addEventListener('touchstart', e => startDraw(e.touches[0]));
canvas.addEventListener('touchmove', e => { draw(e.touches[0]); e.preventDefault(); });
canvas.addEventListener('touchend', stopDraw);

function getPos(evt) {
  const rect = canvas.getBoundingClientRect();
  return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
}
function startDraw(evt) {
  drawing = true;
  const { x, y } = getPos(evt);
  ctx.beginPath();
  ctx.moveTo(x, y);
}
function draw(evt) {
  if (!drawing) return;
  const { x, y } = getPos(evt);
  ctx.lineWidth = PEN_SIZE;
  ctx.lineCap = 'round';
  ctx.strokeStyle = PEN_COLOR;
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}
function stopDraw() {
  drawing = false;
  ctx.beginPath();
}

clearBtn.onclick = () => {
  resetCanvas();
  setStatus('Canvas nettoye.');
};

// Preprocess: crop + pad + resize to model shape, then binarize
function preprocess() {
  workCtx.fillStyle = CANVAS_BG;
  workCtx.fillRect(0, 0, DRAW_SIZE, DRAW_SIZE);
  workCtx.drawImage(canvas, 0, 0, DRAW_SIZE, DRAW_SIZE);

  const img = workCtx.getImageData(0, 0, DRAW_SIZE, DRAW_SIZE);
  const bbox = findBBox(img, BBOX_THRESHOLD);

  const bw = bbox.w || DRAW_SIZE;
  const bh = bbox.h || DRAW_SIZE;
  const cx = bbox.x + bw / 2;
  const cy = bbox.y + bh / 2;

  const maxSide = Math.max(bw, bh);
  const paddedSide = maxSide * 1.2; // keep margin around digit
  pctx.fillStyle = CANVAS_BG;
  pctx.fillRect(0, 0, targetW, targetH);
  pctx.save();
  pctx.translate(targetW / 2, targetH / 2);
  pctx.drawImage(
    workCanvas,
    cx - paddedSide / 2,
    cy - paddedSide / 2,
    paddedSide,
    paddedSide,
    -targetW / 2,
    -targetH / 2,
    targetW,
    targetH
  );
  pctx.restore();

  const { data } = pctx.getImageData(0, 0, targetW, targetH);

  const floatData = new Float32Array(channels * targetH * targetW);
  for (let i = 0; i < targetH * targetW; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    const norm = 1 - gray / 255; // white background -> 0, black ink -> 1
    const val = norm > BIN_THRESHOLD ? 1 : 0; // hard binarization
    if (layout === 'NCHW') {
      for (let c = 0; c < channels; c++) {
        const offset = c * targetH * targetW + i;
        floatData[offset] = val;
      }
    } else {
      const base = i * channels;
      for (let c = 0; c < channels; c++) floatData[base + c] = val;
    }
  }

  const dims = layout === 'NCHW'
    ? [1, channels, targetH, targetW]
    : [1, targetH, targetW, channels];

  return new ort.Tensor('float32', floatData, dims);
}

function findBBox(imageData, threshold) {
  const { data, width, height } = imageData;
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = data[(y * width + x) * 4];
      if (v < threshold) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX === -1 || maxY === -1) {
    return { x: 0, y: 0, w: width, h: height };
  }
  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

predictBtn.onclick = () => {
  predict().catch(err => setStatus('Erreur: ' + err));
};

async function predict() {
  if (!session) { setStatus('Modele non pret.'); return; }
  setStatus('Pretraitement...');
  const inputTensor = preprocess();
  const feeds = { [inputName]: inputTensor };

  setStatus('Inference en cours...');
  const outputs = await session.run(feeds);
  const out = outputs[outputName];
  const data = Array.from(out.data);
  const probs = softmax(data);
  const top = probs
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 3);

  resultsEl.textContent = top
    .map(t => `Classe ${t.i}: ${(t.v * 100).toFixed(2)}%`)
    .join('\n');
  setStatus(`Prediction: ${top[0].i} (${(top[0].v * 100).toFixed(1)}%)`);
}
