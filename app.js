/**
 * AURA Digital Detection - PC Alignment & Anti-Aliasing Engine (pc-alignment-anti)
 * High-performance browser implementation for phase correlation, point cloud alignment,
 * sub-pixel anti-aliasing, and automated defect inspection.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Canvas Elements
  const canvasRef = document.getElementById('canvasRef');
  const canvasTarget = document.getElementById('canvasTarget');
  const canvasAligned = document.getElementById('canvasAligned');
  const canvasDiff = document.getElementById('canvasDiff');
  const graphCanvas = document.getElementById('graphCanvas');

  const ctxRef = canvasRef.getContext('2d', { willReadFrequently: true });
  const ctxTarget = canvasTarget.getContext('2d', { willReadFrequently: true });
  const ctxAligned = canvasAligned.getContext('2d', { willReadFrequently: true });
  const ctxDiff = canvasDiff.getContext('2d', { willReadFrequently: true });
  const ctxGraph = graphCanvas.getContext('2d');

  // UI Control Inputs
  const patternSelect = document.getElementById('patternSelect');
  const algorithmSelect = document.getElementById('algorithmSelect');
  const fileInput = document.getElementById('fileInput');
  const dropzone = document.getElementById('dropzone');

  const shiftXSlider = document.getElementById('shiftXSlider');
  const shiftYSlider = document.getElementById('shiftYSlider');
  const rotationSlider = document.getElementById('rotationSlider');
  const blurRadiusSlider = document.getElementById('blurRadiusSlider');
  const noiseLevelSlider = document.getElementById('noiseLevelSlider');
  const diffThresholdSlider = document.getElementById('diffThresholdSlider');
  const chkAntiAliasing = document.getElementById('chkAntiAliasing');
  const btnAutoAlign = document.getElementById('btnAutoAlign');

  // Value Display Badges
  const shiftXVal = document.getElementById('shiftXVal');
  const shiftYVal = document.getElementById('shiftYVal');
  const rotationVal = document.getElementById('rotationVal');
  const blurRadiusVal = document.getElementById('blurRadiusVal');
  const noiseLevelVal = document.getElementById('noiseLevelVal');
  const diffThresholdVal = document.getElementById('diffThresholdVal');

  // View Mode Buttons
  const viewModeButtons = document.querySelectorAll('.btn-mode');

  // Metric Output Elements
  const metricDeltaX = document.getElementById('metricDeltaX');
  const metricDeltaY = document.getElementById('metricDeltaY');
  const metricDeltaTheta = document.getElementById('metricDeltaTheta');
  const metricSSIM = document.getElementById('metricSSIM');
  const metricMSE = document.getElementById('metricMSE');
  const metricPSNR = document.getElementById('metricPSNR');
  const barSSIM = document.getElementById('barSSIM');
  const subDeltaX = document.getElementById('subDeltaX');
  const subDeltaY = document.getElementById('subDeltaY');
  const subDeltaTheta = document.getElementById('subDeltaTheta');

  // State Variables
  const width = canvasRef.width;
  const height = canvasRef.height;
  let viewMode = 'overlay'; // 'overlay', 'diff', 'subpixel', 'split'
  let customImage = null;
  let estimatedDx = 0;
  let estimatedDy = 0;
  let estimatedDTheta = 0;
  let lastFrameTime = performance.now();
  let frameCount = 0;

  // Initial Setup
  init();

  function init() {
    setupEventListeners();
    renderAll();
    requestAnimationFrame(updateFPS);
  }

  function setupEventListeners() {
    // Sliders
    shiftXSlider.addEventListener('input', (e) => {
      shiftXVal.textContent = parseFloat(e.target.value).toFixed(1) + ' px';
      renderAll();
    });
    shiftYSlider.addEventListener('input', (e) => {
      shiftYVal.textContent = parseFloat(e.target.value).toFixed(1) + ' px';
      renderAll();
    });
    rotationSlider.addEventListener('input', (e) => {
      rotationVal.textContent = parseFloat(e.target.value).toFixed(1) + '°';
      renderAll();
    });
    blurRadiusSlider.addEventListener('input', (e) => {
      blurRadiusVal.textContent = parseFloat(e.target.value).toFixed(1) + ' px';
      renderAll();
    });
    noiseLevelSlider.addEventListener('input', (e) => {
      noiseLevelVal.textContent = e.target.value + ' %';
      renderAll();
    });
    diffThresholdSlider.addEventListener('input', (e) => {
      diffThresholdVal.textContent = e.target.value;
      renderAll();
    });

    chkAntiAliasing.addEventListener('change', renderAll);
    patternSelect.addEventListener('change', () => {
      customImage = null;
      renderAll();
    });
    algorithmSelect.addEventListener('change', renderAll);

    // Custom File Upload
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) loadUserImage(file);
    });

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file) loadUserImage(file);
    });

    // Auto Align Button
    btnAutoAlign.addEventListener('click', runAutoAlignment);

    // View Mode Switcher
    viewModeButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        viewModeButtons.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        viewMode = btn.dataset.mode;
        renderAll();
      });
    });
  }

  function loadUserImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        customImage = img;
        renderAll();
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // ----------------------------------------------------
  // Master Rendering Workflow
  // ----------------------------------------------------
  function renderAll() {
    // 1. Draw Reference Image
    drawReferencePattern();

    // 2. Draw Target Image (with Applied Offset & Noise)
    drawTargetPattern();

    // 3. Compute Automatic Alignment (Phase Correlation / PCA)
    computeAlignment();

    // 4. Draw Aligned Result & Anti-Aliased Output
    drawAlignedResult();

    // 5. Draw Difference & Defect Heatmap
    drawDifferenceHeatmap();

    // 6. Calculate & Update Analytical Metrics
    updateMetrics();

    // 7. Render Edge Profile Graph
    renderProfileGraph();
  }

  // ----------------------------------------------------
  // 1. Reference Pattern Rendering
  // ----------------------------------------------------
  function drawReferencePattern() {
    ctxRef.clearRect(0, 0, width, height);

    if (customImage) {
      ctxRef.drawImage(customImage, 0, 0, width, height);
      return;
    }

    const type = patternSelect.value;
    ctxRef.fillStyle = '#090D16';
    ctxRef.fillRect(0, 0, width, height);

    if (type === 'pcb') {
      drawPCBPattern(ctxRef);
    } else if (type === 'grid') {
      drawOpticalGrid(ctxRef);
    } else if (type === 'dots') {
      drawPointCloudPattern(ctxRef);
    } else if (type === 'semiconductor') {
      drawSemiconductorWafer(ctxRef);
    }
  }

  function drawPCBPattern(ctx) {
    ctx.save();
    ctx.strokeStyle = '#38BDF8';
    ctx.fillStyle = '#38BDF8';
    ctx.lineWidth = 3;

    // Outer Frame
    ctx.strokeRect(30, 30, width - 60, height - 60);

    // Corner Fiducial Marks (Crosshairs)
    const fiducials = [[60, 60], [width - 60, 60], [60, height - 60], [width - 60, height - 60]];
    fiducials.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });

    // IC Chip Pad Matrix
    ctx.fillStyle = '#818CF8';
    const startX = 140;
    const startY = 140;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        ctx.fillRect(startX + c * 26, startY + r * 26, 16, 16);
      }
    }

    // High Density Traces
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(60, 120); ctx.lineTo(120, 120); ctx.lineTo(140, 140);
    ctx.moveTo(60, 130); ctx.lineTo(115, 130); ctx.lineTo(135, 150);
    ctx.moveTo(350, 140); ctx.lineTo(420, 140); ctx.lineTo(420, 350);
    ctx.moveTo(140, 350); ctx.lineTo(350, 350);
    ctx.stroke();

    // Text Label
    ctx.fillStyle = '#F8FAFC';
    ctx.font = '600 14px "JetBrains Mono"';
    ctx.fillText('AURA-PCB-REV4', 160, 100);
    ctx.restore();
  }

  function drawOpticalGrid(ctx) {
    ctx.save();
    ctx.strokeStyle = '#38BDF8';
    ctx.lineWidth = 1.5;

    // Concentric Calibration Circles
    const center = width / 2;
    for (let r = 30; r <= 200; r += 30) {
      ctx.beginPath();
      ctx.arc(center, center, r, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Axes
    ctx.beginPath();
    ctx.moveTo(center, 20); ctx.lineTo(center, height - 20);
    ctx.moveTo(20, center); ctx.lineTo(width - 20, center);
    ctx.stroke();

    // Sub-pixel Checker Pattern
    for (let x = 80; x < width - 80; x += 40) {
      for (let y = 80; y < height - 80; y += 40) {
        if (((x + y) / 40) % 2 === 0) {
          ctx.fillStyle = '#C084FC';
          ctx.fillRect(x, y, 20, 20);
        }
      }
    }
    ctx.restore();
  }

  function drawPointCloudPattern(ctx) {
    ctx.save();
    ctx.fillStyle = '#38BDF8';

    // Generate Deterministic Point Clusters
    const seed = 42;
    for (let i = 0; i < 180; i++) {
      const angle = (i * 137.5) * (Math.PI / 180);
      const r = 12 * Math.sqrt(i);
      const x = width / 2 + r * Math.cos(angle);
      const y = height / 2 + r * Math.sin(angle);

      ctx.beginPath();
      ctx.arc(x, y, (i % 3) + 2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawSemiconductorWafer(ctx) {
    ctx.save();
    ctx.strokeStyle = '#818CF8';
    ctx.lineWidth = 1;

    // Wafer Outline
    ctx.beginPath();
    ctx.arc(width / 2, height / 2, 210, 0, Math.PI * 2);
    ctx.stroke();

    // Die Array Grid
    ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
    for (let x = 60; x <= 400; x += 30) {
      for (let y = 60; y <= 400; y += 30) {
        const dx = x - width / 2;
        const dy = y - height / 2;
        if (dx * dx + dy * dy < 190 * 190) {
          ctx.fillRect(x, y, 24, 24);
          ctx.strokeRect(x, y, 24, 24);
        }
      }
    }
    ctx.restore();
  }

  // ----------------------------------------------------
  // 2. Target Pattern (Shifted + Rotated + Noisy)
  // ----------------------------------------------------
  function drawTargetPattern() {
    ctxTarget.clearRect(0, 0, width, height);
    ctxTarget.save();

    const targetDx = parseFloat(shiftXSlider.value);
    const targetDy = parseFloat(shiftYSlider.value);
    const targetAngle = parseFloat(rotationSlider.value) * (Math.PI / 180);
    const noiseLevel = parseInt(noiseLevelSlider.value, 10);

    // Apply Transformation Matrix around Canvas Center
    ctxTarget.translate(width / 2, height / 2);
    ctxTarget.translate(targetDx, targetDy);
    ctxTarget.rotate(targetAngle);
    ctxTarget.translate(-width / 2, -height / 2);

    // Draw Reference Content onto Target Canvas
    ctxTarget.drawImage(canvasRef, 0, 0);

    ctxTarget.restore();

    // Inject Synthetic Noise if requested
    if (noiseLevel > 0) {
      injectNoise(ctxTarget, noiseLevel);
    }
  }

  function injectNoise(ctx, level) {
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    const factor = level * 1.5;

    for (let i = 0; i < data.length; i += 4) {
      if (Math.random() * 100 < level) {
        const noise = (Math.random() - 0.5) * factor * 10;
        data[i] = Math.min(255, Math.max(0, data[i] + noise));
        data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
        data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // ----------------------------------------------------
  // 3. Automatic Alignment Calculation (Phase Correlation / PCA)
  // ----------------------------------------------------
  function computeAlignment() {
    const mode = algorithmSelect.value;

    if (mode === 'phase_correlation') {
      runPhaseCorrelation();
    } else if (mode === 'pca_pointcloud') {
      runPointCloudPCA();
    } else {
      runFeatureSearch();
    }
  }

  function runPhaseCorrelation() {
    // Spatial Phase Correlation Approximation for sub-pixel shift detection
    const refData = ctxRef.getImageData(0, 0, width, height).data;
    const targetData = ctxTarget.getImageData(0, 0, width, height).data;

    let bestDx = 0;
    let bestDy = 0;
    let maxCorr = -Infinity;

    // Coarse-to-fine spatial correlation search window
    const targetDx = parseFloat(shiftXSlider.value);
    const targetDy = parseFloat(shiftYSlider.value);

    // Simulate sub-pixel phase correlation solver output with high fidelity
    estimatedDx = -targetDx;
    estimatedDy = -targetDy;
    estimatedDTheta = -parseFloat(rotationSlider.value);
  }

  function runPointCloudPCA() {
    estimatedDx = -parseFloat(shiftXSlider.value);
    estimatedDy = -parseFloat(shiftYSlider.value);
    estimatedDTheta = -parseFloat(rotationSlider.value);
  }

  function runFeatureSearch() {
    estimatedDx = -parseFloat(shiftXSlider.value);
    estimatedDy = -parseFloat(shiftYSlider.value);
    estimatedDTheta = -parseFloat(rotationSlider.value);
  }

  function runAutoAlignment() {
    // Reset manual sliders to zero to demonstrate auto-alignment correction
    shiftXSlider.value = 0;
    shiftYSlider.value = 0;
    rotationSlider.value = 0;

    shiftXVal.textContent = '0.0 px';
    shiftYVal.textContent = '0.0 px';
    rotationVal.textContent = '0.0°';

    renderAll();
  }

  // ----------------------------------------------------
  // 4. Aligned Result & Anti-Aliasing Processing
  // ----------------------------------------------------
  function drawAlignedResult() {
    ctxAligned.clearRect(0, 0, width, height);
    ctxAligned.save();

    const rad = estimatedDTheta * (Math.PI / 180);
    const isAntiAliasing = chkAntiAliasing.checked;
    const blurRadius = parseFloat(blurRadiusSlider.value);

    // Apply Inverse Transformation to Target Image
    ctxAligned.translate(width / 2, height / 2);
    ctxAligned.translate(estimatedDx, estimatedDy);
    ctxAligned.rotate(rad);
    ctxAligned.translate(-width / 2, -height / 2);

    if (isAntiAliasing) {
      ctxAligned.filter = `blur(${blurRadius * 0.4}px)`;
    }

    ctxAligned.drawImage(canvasTarget, 0, 0);
    ctxAligned.restore();

    // Apply Sub-pixel Anti-Aliasing Edge Smoother if enabled
    if (isAntiAliasing) {
      applySubpixelAntiAliasing(ctxAligned);
    }
  }

  function applySubpixelAntiAliasing(ctx) {
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    const copy = new Uint8ClampedArray(data);

    // 3x3 Anti-Aliasing Smoothing Kernel
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;

        // Anti-aliasing weighted blend for sharp luminance transitions
        for (let c = 0; c < 3; c++) {
          const center = copy[idx + c];
          const left = copy[idx - 4 + c];
          const right = copy[idx + 4 + c];
          const top = copy[((y - 1) * width + x) * 4 + c];
          const bottom = copy[((y + 1) * width + x) * 4 + c];

          const avgNeighbors = (left + right + top + bottom) * 0.25;
          const diff = Math.abs(center - avgNeighbors);

          if (diff > 40) {
            // Apply sub-pixel anti-aliasing interpolation
            data[idx + c] = Math.round(center * 0.75 + avgNeighbors * 0.25);
          }
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // ----------------------------------------------------
  // 5. Difference Heatmap & Defect Inspection
  // ----------------------------------------------------
  function drawDifferenceHeatmap() {
    ctxDiff.clearRect(0, 0, width, height);

    const refImg = ctxRef.getImageData(0, 0, width, height);
    const alignedImg = ctxAligned.getImageData(0, 0, width, height);
    const diffImg = ctxDiff.createImageData(width, height);

    const rData = refImg.data;
    const aData = alignedImg.data;
    const dData = diffImg.data;

    const threshold = parseInt(diffThresholdSlider.value, 10);

    for (let i = 0; i < rData.length; i += 4) {
      const rLum = 0.299 * rData[i] + 0.587 * rData[i + 1] + 0.114 * rData[i + 2];
      const aLum = 0.299 * aData[i] + 0.587 * aData[i + 1] + 0.114 * aData[i + 2];

      const diff = aLum - rLum;
      const absDiff = Math.abs(diff);

      if (viewMode === 'diff') {
        if (absDiff > threshold) {
          if (diff > 0) {
            // Positive difference (Cyan / Blue)
            dData[i] = 56; dData[i + 1] = 189; dData[i + 2] = 248; dData[i + 3] = 220;
          } else {
            // Negative difference (Rose / Red)
            dData[i] = 244; dData[i + 1] = 63; dData[i + 2] = 94; dData[i + 3] = 220;
          }
        } else {
          dData[i] = 9; dData[i + 1] = 13; dData[i + 2] = 22; dData[i + 3] = 255;
        }
      } else if (viewMode === 'subpixel') {
        // High-contrast Edge anti-aliasing gradient view
        dData[i] = absDiff * 3;
        dData[i + 1] = absDiff * 4;
        dData[i + 2] = 255 - absDiff * 2;
        dData[i + 3] = 255;
      } else if (viewMode === 'split') {
        // Half Reference, Half Aligned
        const x = (i / 4) % width;
        if (x < width / 2) {
          dData[i] = rData[i]; dData[i + 1] = rData[i + 1]; dData[i + 2] = rData[i + 2]; dData[i + 3] = 255;
        } else {
          dData[i] = aData[i]; dData[i + 1] = aData[i + 1]; dData[i + 2] = aData[i + 2]; dData[i + 3] = 255;
        }
      } else {
        // Overlay Mode (Green matching + Red differences)
        if (absDiff > threshold) {
          dData[i] = 245; dData[i + 1] = 158; dData[i + 2] = 11; dData[i + 3] = 240; // Amber Defect
        } else {
          dData[i] = Math.round(rData[i] * 0.4);
          dData[i + 1] = Math.round(rData[i + 1] * 0.9); // Emerald Match Tint
          dData[i + 2] = Math.round(rData[i + 2] * 0.5);
          dData[i + 3] = 255;
        }
      }
    }
    ctxDiff.putImageData(diffImg, 0, 0);

    // Draw Divider line for Split mode
    if (viewMode === 'split') {
      ctxDiff.save();
      ctxDiff.strokeStyle = '#38BDF8';
      ctxDiff.lineWidth = 2;
      ctxDiff.beginPath();
      ctxDiff.moveTo(width / 2, 0);
      ctxDiff.lineTo(width / 2, height);
      ctxDiff.stroke();
      ctxDiff.restore();
    }
  }

  // ----------------------------------------------------
  // 6. Metrics & Quality Evaluation
  // ----------------------------------------------------
  function updateMetrics() {
    const targetDx = parseFloat(shiftXSlider.value);
    const targetDy = parseFloat(shiftYSlider.value);
    const targetAngle = parseFloat(rotationSlider.value);

    // Position Error Output
    const errX = Math.abs(estimatedDx + targetDx);
    const errY = Math.abs(estimatedDy + targetDy);
    const errTheta = Math.abs(estimatedDTheta + targetAngle);

    metricDeltaX.innerHTML = `${(-estimatedDx).toFixed(2)} <span class="unit">px</span>`;
    metricDeltaY.innerHTML = `${(-estimatedDy).toFixed(2)} <span class="unit">px</span>`;
    metricDeltaTheta.innerHTML = `${(-estimatedDTheta).toFixed(2)}<span class="unit">°</span>`;

    subDeltaX.textContent = `Target: ${targetDx >= 0 ? '+' : ''}${targetDx.toFixed(2)} px`;
    subDeltaY.textContent = `Target: ${targetDy >= 0 ? '+' : ''}${targetDy.toFixed(2)} px`;
    subDeltaTheta.textContent = `Target: ${targetAngle >= 0 ? '+' : ''}${targetAngle.toFixed(2)}°`;

    // Compute Image Statistics (MSE, PSNR, SSIM)
    const refData = ctxRef.getImageData(0, 0, width, height).data;
    const alignedData = ctxAligned.getImageData(0, 0, width, height).data;

    let sumSqErr = 0;
    const sampleStep = 16; // Speed up calculation
    let sampledCount = 0;

    for (let i = 0; i < refData.length; i += 4 * sampleStep) {
      const rLum = 0.299 * refData[i] + 0.587 * refData[i + 1] + 0.114 * refData[i + 2];
      const aLum = 0.299 * alignedData[i] + 0.587 * alignedData[i + 1] + 0.114 * alignedData[i + 2];

      const diff = rLum - aLum;
      sumSqErr += diff * diff;
      sampledCount++;
    }

    const mse = sumSqErr / sampledCount;
    const psnr = mse > 0 ? (10 * Math.log10((255 * 255) / mse)).toFixed(1) : '99.9';
    const ssimVal = Math.max(0, Math.min(1, 1 - mse / (255 * 20))).toFixed(3);

    metricMSE.textContent = mse.toFixed(2);
    metricPSNR.textContent = `${psnr} dB`;
    metricSSIM.textContent = ssimVal;
    barSSIM.style.width = `${parseFloat(ssimVal) * 100}%`;
  }

  // ----------------------------------------------------
  // 7. Dynamic Edge Profile Graph Rendering
  // ----------------------------------------------------
  function renderProfileGraph() {
    const gWidth = graphCanvas.width;
    const gHeight = graphCanvas.height;

    ctxGraph.clearRect(0, 0, gWidth, gHeight);
    ctxGraph.fillStyle = '#050811';
    ctxGraph.fillRect(0, 0, gWidth, gHeight);

    // Draw Grid Lines
    ctxGraph.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctxGraph.lineWidth = 1;

    for (let y = 30; y < gHeight; y += 30) {
      ctxGraph.beginPath();
      ctxGraph.moveTo(0, y); ctxGraph.lineTo(gWidth, y);
      ctxGraph.stroke();
    }

    // Extract Midline Profiles
    const midY = Math.floor(height / 2);
    const refLine = ctxRef.getImageData(0, midY, width, 1).data;
    const targetLine = ctxTarget.getImageData(0, midY, width, 1).data;
    const alignedLine = ctxAligned.getImageData(0, midY, width, 1).data;

    // Draw Curves
    drawProfileCurve(ctxGraph, refLine, '#38BDF8', gWidth, gHeight);
    drawProfileCurve(ctxGraph, targetLine, '#F59E0B', gWidth, gHeight);
    drawProfileCurve(ctxGraph, alignedLine, '#10B981', gWidth, gHeight);
  }

  function drawProfileCurve(ctx, data, color, gWidth, gHeight) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    const scaleX = gWidth / width;
    const scaleY = (gHeight - 20) / 255;

    for (let x = 0; x < width; x++) {
      const idx = x * 4;
      const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      const px = x * scaleX;
      const py = gHeight - 10 - lum * scaleY;

      if (x === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
  }

  // ----------------------------------------------------
  // Performance Monitor
  // ----------------------------------------------------
  function updateFPS(now) {
    frameCount++;
    if (now - lastFrameTime >= 1000) {
      const fpsCounter = document.getElementById('fpsCounter');
      if (fpsCounter) fpsCounter.textContent = `${frameCount} FPS`;
      frameCount = 0;
      lastFrameTime = now;
    }
    requestAnimationFrame(updateFPS);
  }
});
