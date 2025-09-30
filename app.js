/* Logistic Regression Practice App (front-end only) */

// Utility: simple CSV parser for two-column numeric data with optional header.
function parseCsvTwoCols(text, { expectHeader = false, colNames = ["x", "y"] } = {}) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.trim().length > 0);
  if (lines.length === 0) return [];
  let startIdx = 0;
  if (expectHeader) {
    startIdx = 1;
  } else if (/^[a-zA-Z]/.test(lines[0])) {
    // auto-detect header if first line starts with a letter
    startIdx = 1;
  }
  const out = [];
  for (let i = startIdx; i < lines.length; i++) {
    const parts = lines[i].split(/[;,\t]/).map(s => s.trim());
    if (parts.length < 2) continue;
    const a = Number(parts[0]);
    const b = Number(parts[1]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    out.push({ [colNames[0]]: a, [colNames[1]]: b });
  }
  return out;
}

// Sigmoid and helper
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
function sigmoidAt(x, a, b) { return sigmoid(a * x + b); }

// Synthetic dataset generator: 1D x in [0,10], y ~ Bernoulli(sigmoid(ax+b))
function generateSyntheticData(n = 60, a = 1.6, b = -0.2, seed = 42) {
  const rnd = mulberry32(seed);
  const xs = [...Array(n)].map(() => 10 * rnd());
  const data = xs.map(x => {
    const p = sigmoidAt(x, a, b);
    const y = rnd() < p ? 1 : 0;
    return { x, y };
  });
  // Sort by x for stable visuals
  data.sort((p, q) => p.x - q.x);
  return data;
}

// Small PRNG for reproducibility
function mulberry32(a) {
  return function() {
    var t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
}

// Chart.js setup
let chart;
let curveDatasetIndex = null; // current sigmoid curve dataset index
let halfLineDatasetIndex = null; // y=0.5 reference line index
let thresholdDatasetIndex = null; // vertical line where y=0.5 (x = -b/a)
let thresholdMarkerDatasetIndex = null; // moving circle at threshold on x-axis
let pluginRegistered = false;

// State for overlays
let currentDataPoints = []; // array of { x, y, cls }
let thresholdX0 = null;     // current threshold x = -b/a (if available)
let currentView = 'sigmoid';

// Plugin to draw quadrant counts (sigmoid view only)
const quadrantCountsPlugin = {
  id: 'quadrantCounts',
  afterDatasetsDraw(c, args, opts) {
    if (currentView !== 'sigmoid') return;
    if (!Number.isFinite(thresholdX0)) return;
    if (!currentDataPoints || currentDataPoints.length === 0) return;
    const { ctx, scales } = c;
    const xScale = scales.x;
    const yScale = scales.y;

    // Compute counts relative to threshold line at x0
    const x0 = thresholdX0;
    let tn = 0, fp = 0, tp = 0, fn = 0;
    for (const p of currentDataPoints) {
      if (!Number.isFinite(p.x)) continue;
      const predicted = (p.x >= x0) ? 1 : 0; // threshold rule (right => 1)
      if (p.cls === 1 && predicted === 0) fn++;
      else if (p.cls === 0 && predicted === 0) tn++;
      else if (p.cls === 1 && predicted === 1) tp++;
      else if (p.cls === 0 && predicted === 1) fp++;
    }

    // Place labels at quadrant centers
    const xMin = xScale.min ?? (xScale.getValueForPixel(xScale.left));
    const xMax = xScale.max ?? (xScale.getValueForPixel(xScale.right));
    const leftCenterX = xScale.getPixelForValue((xMin + x0) / 2);
    const rightCenterX = xScale.getPixelForValue((x0 + xMax) / 2);
    const upperY = yScale.getPixelForValue(0.8);
    const lowerY = yScale.getPixelForValue(0.2);

    ctx.save();
    ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    ctx.fillStyle = '#e5e7eb';

    // Upper-left: FN
    ctx.strokeText(`FN: ${fn}`, leftCenterX, upperY);
    ctx.fillText(`FN: ${fn}`, leftCenterX, upperY);
    // Lower-left: TN
    ctx.strokeText(`TN: ${tn}`, leftCenterX, lowerY);
    ctx.fillText(`TN: ${tn}`, leftCenterX, lowerY);
    // Upper-right: TP
    ctx.strokeText(`TP: ${tp}`, rightCenterX, upperY);
    ctx.fillText(`TP: ${tp}`, rightCenterX, upperY);
    // Lower-right: FP
    ctx.strokeText(`FP: ${fp}`, rightCenterX, lowerY);
    ctx.fillText(`FP: ${fp}`, rightCenterX, lowerY);

    ctx.restore();
  }
};

function initChart() {
  const ctx = document.getElementById('chart');
  if (chart) { chart.destroy(); }
  if (!pluginRegistered && typeof Chart !== 'undefined') {
    Chart.register(quadrantCountsPlugin);
    pluginRegistered = true;
  }

  chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Class 0 (y=0)',
          data: [],
          pointBackgroundColor: '#3b82f6',
          pointRadius: 4,
          showLine: false,
        },
        {
          label: 'Class 1 (y=1)',
          data: [],
          pointBackgroundColor: '#ef4444',
          pointRadius: 4,
          showLine: false,
        },
        {
          label: 'y = 0.5',
          data: [],
          borderColor: 'rgba(148,163,184,0.8)',
          borderDash: [6,6],
          showLine: true,
          pointRadius: 0,
        },
        {
          label: 'Sigmoid',
          data: [],
          borderColor: '#22d3ee',
          showLine: true,
          pointRadius: 0,
        },
        {
          label: 'x at y=0.5',
          data: [],
          borderColor: '#f59e0b', // amber
          borderDash: [4,4],
          showLine: true,
          pointRadius: 0,
        }
        ,
        {
          label: 'Threshold marker',
          data: [],
          pointBackgroundColor: '#facc15', // yellow
          borderColor: '#facc15',
          showLine: false,
          pointRadius: 7,
        }
      ]
    },
    options: {
      animation: false,
      responsive: true,
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'x' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          min: -0.05,
          max: 1.05,
          title: { display: true, text: 'y' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        }
      },
      plugins: {
        legend: { labels: { color: '#e5e7eb' } },
        tooltip: { enabled: true }
      }
    }
  });

  curveDatasetIndex = 3;
  halfLineDatasetIndex = 2;
  thresholdDatasetIndex = 4;
  thresholdMarkerDatasetIndex = 5;
}

function updateHalfLine(xmin, xmax) {
  chart.data.datasets[halfLineDatasetIndex].data = [
    { x: xmin, y: 0.5 },
    { x: xmax, y: 0.5 },
  ];
}

function setDataPoints(data) {
  // data items expected: { x, y, cls } where cls in {0,1}
  const class0 = data.filter(p => p.cls === 0).map(p => ({ x: p.x, y: p.y }));
  const class1 = data.filter(p => p.cls === 1).map(p => ({ x: p.x, y: p.y }));
  chart.data.datasets[0].data = class0;
  chart.data.datasets[1].data = class1;
  currentDataPoints = data;
}

function computeXRangeFromData(data) {
  if (!data.length) return { min: 0, max: 10 };
  const xs = data.map(d => d.x);
  const min = Math.min(...xs), max = Math.max(...xs);
  const pad = 0.1 * (max - min || 1);
  return { min: min - pad, max: max + pad };
}

function linearlySpaced(xmin, xmax, n = 200) {
  const out = [];
  const step = (xmax - xmin) / (n - 1);
  for (let i = 0; i < n; i++) out.push(xmin + i * step);
  return out;
}

function setSigmoidCurve(a, b, xmin, xmax) {
  const xs = linearlySpaced(xmin, xmax, 300);
  const pts = xs.map(x => ({ x, y: sigmoidAt(x, a, b) }));
  chart.data.datasets[curveDatasetIndex].data = pts;
}

function updateThresholdLine(a, b) {
  // x0 = -b/a only if a != 0
  if (!Number.isFinite(a) || Math.abs(a) < 1e-8 || !Number.isFinite(b)) {
    chart.data.datasets[thresholdDatasetIndex].data = [];
    return;
  }
  const x0 = -b / a;
  // y range from chart options (fixed in this app)
  const yMin = -0.05;
  const yMax = 1.05;
  chart.data.datasets[thresholdDatasetIndex].data = [
    { x: x0, y: yMin },
    { x: x0, y: yMax },
  ];
  thresholdX0 = x0;
}

function updateThresholdMarker(a, b) {
  if (!Number.isFinite(a) || Math.abs(a) < 1e-8 || !Number.isFinite(b)) {
    chart.data.datasets[thresholdMarkerDatasetIndex].data = [];
    if (!(Number.isFinite(a) && Number.isFinite(b))) thresholdX0 = null;
    return;
  }
  const x0 = -b / a;
  chart.data.datasets[thresholdMarkerDatasetIndex].data = [ { x: x0, y: 0 } ];
  thresholdX0 = x0;
}

// Animation state
let paramsSequence = [];
let animTimer = null;
let animIdx = 0;
let animDelay = 700; // ms
let lastXRange = { min: 0, max: 10 };

function stepAnimation() {
  if (animIdx >= paramsSequence.length) {
    stopAnimation();
    return;
  }
  const { a, b } = paramsSequence[animIdx];
  setSigmoidCurve(a, b, lastXRange.min, lastXRange.max);
  updateThresholdLine(a, b);
  updateThresholdMarker(a, b);
  chart.update();
  animIdx += 1;
}

function startAnimation() {
  if (!paramsSequence.length) return;
  stopAnimation();
  stepAnimation(); // show first immediately
  animTimer = setInterval(stepAnimation, animDelay);
}

function pauseAnimation() {
  if (animTimer) {
    clearInterval(animTimer);
    animTimer = null;
  }
}

function stopAnimation() {
  if (animTimer) clearInterval(animTimer);
  animTimer = null;
}

function resetAnimation() {
  stopAnimation();
  animIdx = 0;
  chart.data.datasets[curveDatasetIndex].data = [];
  chart.data.datasets[thresholdDatasetIndex].data = [];
  chart.data.datasets[thresholdMarkerDatasetIndex].data = [];
  thresholdX0 = null;
  chart.update();
}

// DOM wiring
window.addEventListener('DOMContentLoaded', () => {
  initChart();

  const datasetTA = document.getElementById('dataset-csv');
  const paramsTA = document.getElementById('params-csv');
  const conventionSel = document.getElementById('param-convention');
  const viewSel = document.getElementById('view-mode');

  document.getElementById('btn-load-sample-data').addEventListener('click', () => {
    // Generate a fresh sample each time using a varying seed
    const seed = (Date.now() ^ Math.floor(Math.random() * 1e9)) >>> 0;
    // Parameters tuned for x in [0,10]; threshold near mid-range
    const data = generateSyntheticData(70, 1.2, -6.0, seed);
    const csv = 'x,y\n' + data.map(d => `${d.x.toFixed(4)},${d.y}`).join('\n');
    datasetTA.value = csv;
    // Auto-plot after loading so the change is visible immediately
    plotCurrentDataset();
    // Clear any existing sigmoid curve to avoid confusion when data changes
    chart.data.datasets[curveDatasetIndex].data = [];
    chart.data.datasets[thresholdDatasetIndex].data = [];
    chart.data.datasets[thresholdMarkerDatasetIndex].data = [];
    thresholdX0 = null;
    chart.update();
  });

  document.getElementById('btn-copy-sample-data').addEventListener('click', async () => {
    if (!datasetTA.value.trim()) {
      const data = generateSyntheticData(70, 1.2, -6.0, 13);
      const csv = 'x,y\n' + data.map(d => `${d.x.toFixed(4)},${d.y}`).join('\n');
      datasetTA.value = csv;
    }
    await navigator.clipboard.writeText(datasetTA.value);
    flash(datasetTA);
  });

  // Sample params buttons removed per request; students will paste their own results.

  function plotCurrentDataset() {
    const parsed = parseCsvTwoCols(datasetTA.value, { colNames: ['x','y'] });
    if (!parsed.length) return;
    const isThresholdView = viewSel && viewSel.value === 'threshold';
    const data = parsed.map(({ x, y }) => {
      // If y is not strictly 0/1 (e.g., probabilities), binarize at 0.5
      const yNum = Number(y);
      const cls = Number.isFinite(yNum) && yNum >= 0.5 ? 1 : 0;
      const yPos = isThresholdView ? 0 : cls;
      return { x: Number(x), y: yPos, cls };
    });
    setDataPoints(data);
    lastXRange = computeXRangeFromData(data);
    updateHalfLine(lastXRange.min, lastXRange.max);
    applyViewMode();
    chart.update();
  }

  // Auto-plot when the dataset text changes (debounced) and clear the curve
  let plotDebounce;
  datasetTA.addEventListener('input', () => {
    clearTimeout(plotDebounce);
    plotDebounce = setTimeout(() => {
      plotCurrentDataset();
      chart.data.datasets[curveDatasetIndex].data = [];
      chart.data.datasets[thresholdDatasetIndex].data = [];
      chart.data.datasets[thresholdMarkerDatasetIndex].data = [];
      thresholdX0 = null;
      chart.update();
    }, 200);
  });

  function applyViewMode() {
    const mode = (viewSel && viewSel.value) || 'sigmoid';
    const isThreshold = mode === 'threshold';
    // Show/hide sigmoid curve and y=0.5 line
    chart.data.datasets[curveDatasetIndex].hidden = isThreshold; // hide sigmoid in threshold view
    chart.data.datasets[halfLineDatasetIndex].hidden = isThreshold; // hide y=0.5 in threshold view
    // Threshold visuals: in sigmoid view, show vertical line; in threshold view, show marker dot
    chart.data.datasets[thresholdDatasetIndex].hidden = isThreshold; // hide line in threshold view
    chart.data.datasets[thresholdMarkerDatasetIndex].hidden = !isThreshold; // show marker only in threshold view
    // In threshold view, points are on x-axis (already handled in plotting), keep threshold line visible
    // In sigmoid view, ensure class points reflect 0/1 positions; re-plot to update Y positions
    if (!isThreshold) {
      const class0 = chart.data.datasets[0].data.map(p => ({ x: p.x, y: 0 }));
      const class1 = chart.data.datasets[1].data.map(p => ({ x: p.x, y: 1 }));
      chart.data.datasets[0].data = class0;
      chart.data.datasets[1].data = class1;
    } else {
      // Ensure class points sit at y=0 in threshold view
      const class0 = chart.data.datasets[0].data.map(p => ({ x: p.x, y: 0 }));
      const class1 = chart.data.datasets[1].data.map(p => ({ x: p.x, y: 0 }));
      chart.data.datasets[0].data = class0;
      chart.data.datasets[1].data = class1;
    }
  }

  // React to view-mode changes
  if (viewSel) {
    viewSel.addEventListener('change', () => {
      plotCurrentDataset();
      // In threshold view we typically only want the threshold line during animation.
      // Clear existing sigmoid curve/marker when switching modes for clarity.
      chart.data.datasets[curveDatasetIndex].data = [];
      chart.data.datasets[thresholdMarkerDatasetIndex].data = [];
      currentView = (viewSel && viewSel.value) || 'sigmoid';
      chart.update();
    });
  }
  // Set initial view state
  currentView = (viewSel && viewSel.value) || 'sigmoid';

  document.getElementById('btn-start').addEventListener('click', () => {
    const raw = parseCsvTwoCols(paramsTA.value, { colNames: ['b','a'] });
    if (!raw.length) return;
    // Normalize to internal form sigmoid(a*x + b): a=slope, b=intercept.
    // We always expect input order to be (b, a): first column is intercept b, second is slope a.
    // Therefore no swapping is needed regardless of the selected convention.
    paramsSequence = raw.map(({ a, b }) => ({ a, b }));
    startAnimation();
  });

  document.getElementById('btn-pause').addEventListener('click', () => {
    if (animTimer) pauseAnimation(); else startAnimation();
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    resetAnimation();
  });

  const speedRange = document.getElementById('speed');
  const speedLabel = document.getElementById('speed-label');
  speedRange.addEventListener('input', () => {
    animDelay = Number(speedRange.value);
    speedLabel.textContent = `${animDelay}ms`;
    if (animTimer) { // apply new speed
      startAnimation(); // restarts with new interval
    }
  });

  // Initialize with a sample dataset for convenience
  document.getElementById('btn-load-sample-data').click();
});

function flash(el) {
  el.style.outline = '2px solid #22d3ee';
  setTimeout(() => { el.style.outline = 'none'; }, 400);
}
