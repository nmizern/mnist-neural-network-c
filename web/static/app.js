const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let drawing = false;

ctx.fillStyle = "black";
ctx.fillRect(0, 0, 280, 280);
ctx.strokeStyle = "white";
ctx.lineWidth = 22;
ctx.lineCap = "round";
ctx.lineJoin = "round";

canvas.addEventListener("mousedown", (e) => {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
});

canvas.addEventListener("mouseup", () => { drawing = false; });
canvas.addEventListener("mouseleave", () => { drawing = false; });

canvas.addEventListener("touchstart", (e) => {
    e.preventDefault();
    drawing = true;
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    ctx.beginPath();
    ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
});

canvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
    ctx.stroke();
});

canvas.addEventListener("touchend", () => { drawing = false; });

function getPixels() {
    const srcData = ctx.getImageData(0, 0, 280, 280).data;

    // Calcul de la boite englobante et du centre de masse
    let minX = 280, minY = 280, maxX = 0, maxY = 0;
    let massX = 0, massY = 0, totalMass = 0;

    for (let y = 0; y < 280; y++) {
        for (let x = 0; x < 280; x++) {
            const val = srcData[(y * 280 + x) * 4];
            if (val > 10) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                massX += x * val;
                massY += y * val;
                totalMass += val;
            }
        }
    }

    if (totalMass === 0) {
        return new Array(784).fill(0);
    }

    // Mise a l'echelle du contenu pour tenir dans une zone 20x20
    const cropW = maxX - minX + 1;
    const cropH = maxY - minY + 1;
    const cropSize = Math.max(cropW, cropH);
    const scale = 20.0 / cropSize;

    // Centre de masse dans la region recadree, puis decalage pour le placer a (14, 14) dans 28x28
    const cmX = (massX / totalMass - minX) * scale;
    const cmY = (massY / totalMass - minY) * scale;
    const dx = 14 - cmX;
    const dy = 14 - cmY;

    const tmp = document.createElement("canvas");
    tmp.width = 28;
    tmp.height = 28;
    const tctx = tmp.getContext("2d");
    tctx.fillStyle = "black";
    tctx.fillRect(0, 0, 28, 28);
    tctx.imageSmoothingEnabled = true;
    tctx.imageSmoothingQuality = "high";
    tctx.drawImage(canvas, minX, minY, cropW, cropH, dx, dy, cropW * scale, cropH * scale);

    const imgData = tctx.getImageData(0, 0, 28, 28);
    const data = imgData.data;
    const pixels = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
        pixels[i] = data[i * 4] / 255.0;
    }

    // Afficher l'apercu
    const preview = document.getElementById("preview");
    const pctx = preview.getContext("2d");
    pctx.putImageData(imgData, 0, 0);

    return Array.from(pixels);
}

async function predict() {
    const pixels = getPixels();
    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pixels }),
        });
        const result = await res.json();
        displayResult(result);
    } catch (err) {
        console.error("Echec de la prediction :", err);
    }
}

function displayResult(result) {
    document.getElementById("prediction").textContent = result.digit;

    const barsDiv = document.getElementById("bars");
    barsDiv.innerHTML = "";

    for (let i = 0; i < result.confidence.length; i++) {
        const pct = (result.confidence[i] * 100).toFixed(1);
        const isActive = i === result.digit;

        barsDiv.innerHTML += `
            <div class="bar-container">
                <span class="bar-label">${i}</span>
                <div class="bar-track">
                    <div class="bar-fill ${isActive ? "active" : ""}" style="width: ${pct}%"></div>
                </div>
                <span class="bar-value">${pct}%</span>
            </div>`;
    }
}

function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 280, 280);
    document.getElementById("prediction").textContent = "-";
    document.getElementById("bars").innerHTML = "";
}
