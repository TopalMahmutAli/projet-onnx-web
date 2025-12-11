const MODEL_ONNX = "simple_cnn.onnx";
const MODEL_DATA = "simple_cnn.onnx.data";

const CANVAS_SIZE = 320;      
const TARGET_SIZE = 28;       
const MNIST_DIGIT_SIZE = 20;  

const MEAN = 0.1307;
const STD = 0.3081;

const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d");

canvas.width = CANVAS_SIZE;
canvas.height = CANVAS_SIZE;

ctx.fillStyle = "#000000"; 
ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "#FFFFFF";

let drawing = false;

const tmp = document.createElement("canvas");
tmp.width = TARGET_SIZE;
tmp.height = TARGET_SIZE;
const tctx = tmp.getContext("2d");

let session = null;

async function loadModel() {
    try {
        const dataResp = await fetch(MODEL_DATA);
        const dataBuffer = await dataResp.arrayBuffer();

        const modelResp = await fetch(MODEL_ONNX);
        const modelBuffer = await modelResp.arrayBuffer();

        session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all",
            externalData: [
                { data: new Uint8Array(dataBuffer), path: MODEL_DATA }
            ]
        });

        document.getElementById("status").textContent = "Modèle chargé ✔️";
    } catch (err) {
        console.error(err);
        document.getElementById("status").textContent = "Erreur chargement modèle ❌";
    }
}
loadModel();

function getPos(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
}

canvas.addEventListener("mousedown", e => {
    drawing = true;
    const { x, y } = getPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
});

canvas.addEventListener("mousemove", e => {
    if (!drawing) return;
    const { x, y } = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
});

canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseleave", () => drawing = false);

canvas.addEventListener("touchstart", e => {
    drawing = true;
    const t = e.touches[0];
    const { x, y } = getPos(t);
    ctx.beginPath();
    ctx.moveTo(x, y);
});

canvas.addEventListener("touchmove", e => {
    e.preventDefault();
    if (!drawing) return;
    const t = e.touches[0];
    const { x, y } = getPos(t);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
});
canvas.addEventListener("touchend", () => drawing = false);

document.getElementById("clear").onclick = () => {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    document.getElementById("results").textContent = "";
    document.getElementById("status").textContent = "Effacé";
};

function getBoundingBox(imageData) {
    const { data, width, height } = imageData;

    let minX = width, minY = height, maxX = -1, maxY = -1;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            const sum = data[i] + data[i+1] + data[i+2];

            if (sum > 50) { 
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    return maxX === -1 ? null : { minX, minY, maxX, maxY };
}

function preprocess() {
    const full = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const box = getBoundingBox(full);

    if (!box) {
        return new ort.Tensor("float32", new Float32Array(784), [1,1,28,28]);
    }

    const w = box.maxX - box.minX + 1;
    const h = box.maxY - box.minY + 1;

    const crop = document.createElement("canvas");
    crop.width = w;
    crop.height = h;
    const cropCtx = crop.getContext("2d");

    cropCtx.drawImage(canvas, box.minX, box.minY, w, h, 0, 0, w, h);

    const resized = document.createElement("canvas");
    resized.width = MNIST_DIGIT_SIZE;
    resized.height = MNIST_DIGIT_SIZE;
    const rctx = resized.getContext("2d");

    rctx.drawImage(crop, 0, 0, w, h, 0, 0, MNIST_DIGIT_SIZE, MNIST_DIGIT_SIZE);

    // center in 28×28
    tctx.fillStyle = "#000000";
    tctx.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);

    const dx = Math.floor((TARGET_SIZE - MNIST_DIGIT_SIZE) / 2);
    const dy = Math.floor((TARGET_SIZE - MNIST_DIGIT_SIZE) / 2);

    tctx.drawImage(resized, dx, dy);

    const img = tctx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE);
    const buf = new Float32Array(784);

    for (let i = 0; i < 784; i++) {
        const r = img.data[i * 4] / 255; 
        buf[i] = (r - MEAN) / STD;
    }

    return new ort.Tensor("float32", buf, [1,1,28,28]);
}

async function predict() {
    if (!session) {
        document.getElementById("status").textContent = "Modèle pas prêt...";
        return;
    }

    document.getElementById("status").textContent = "Prétraitement...";
    const tensor = preprocess();

    const output = await session.run({ input: tensor });

    const logits = output.output.data;
    const max = Math.max(...logits);
    const exp = logits.map(v => Math.exp(v - max));
    const sum = exp.reduce((a,b)=>a+b);
    const probs = exp.map(v => v / sum);

    const digit = probs.indexOf(Math.max(...probs));

    document.getElementById("results").textContent =
        `Chiffre : ${digit} (${(probs[digit]*100).toFixed(1)}%)`;

    document.getElementById("status").textContent = "Terminé ✔️";
}

document.getElementById("predict").onclick = predict;