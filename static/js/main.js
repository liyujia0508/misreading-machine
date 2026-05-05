/* ═══════════════════════════════════════════════════════════
   Emotion Recognition – main.js
   Affective Computing Homework
   ═══════════════════════════════════════════════════════════ */

'use strict';

// ── Emotion Config ────────────────────────────────────────────────────────────
const EMOTION_CONFIG = {
  sadness:  { emoji: '😢', color: '#60A5FA', valence: -0.6, arousal: -0.5 },
  joy:      { emoji: '😊', color: '#FBBF24', valence:  0.8, arousal:  0.6 },
  love:     { emoji: '❤️', color: '#F472B6', valence:  0.7, arousal:  0.2 },
  anger:    { emoji: '😠', color: '#F87171', valence: -0.7, arousal:  0.8 },
  fear:     { emoji: '😨', color: '#A78BFA', valence: -0.5, arousal:  0.7 },
  surprise: { emoji: '😲', color: '#34D399', valence:  0.3, arousal:  0.9 },
};

const EMOTION_ORDER = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'];

const GROUND_TRUTH_DOUBT_MESSAGES = [
  'Are you sure this is your real emotion?',
  'Self-report inconsistency detected.',
  'Re-evaluating your input…',
];

const SYSTEM_STATEMENTS = [
  'Confidence does not equal truth.',
  'Emotion reduced to classification.',
  'This is a statistical interpretation, not an understanding.',
  'Your experience has been translated into probabilities.',
  'System certainty increases as ambiguity is removed.',
];

const HIGH_MISREAD_PHRASES = [
  'System failure or human complexity?',
  'Irreducible emotional state detected.',
  'Classification breakdown.',
];

// API endpoints fallback (supports running page outside same-origin server)
const API_ENDPOINTS = (() => {
  const endpoints = [];
  const isHttp = window.location.protocol === 'http:' || window.location.protocol === 'https:';

  if (isHttp) {
    endpoints.push(`${window.location.origin}/api/predict`);
  }

  endpoints.push(
    'http://127.0.0.1:5001/api/predict',
    'http://localhost:5001/api/predict',
    'http://127.0.0.1:5000/api/predict',
    'http://localhost:5000/api/predict',
  );

  return [...new Set(endpoints)];
})();

// ── State ─────────────────────────────────────────────────────────────────────
let history = [];
let chartInstance = null;
let vaCtx = null;
let lastResult = null;
let analysisCount = 0;
let emotionCounts = {};
let groundTruthEmotion = null;
let sortHistoryByMisreading = true;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const textInput    = document.getElementById('text-input');
const charNum      = document.getElementById('char-num');
const analyzeBtn   = document.getElementById('analyze-btn');
const btnIcon      = document.getElementById('btn-icon');
const btnLabel     = document.getElementById('btn-label');
const placeholder  = document.getElementById('placeholder');
const resultsBody  = document.getElementById('results-body');
const loadingOverlay = document.getElementById('loading-overlay');
const errorToast   = document.getElementById('error-toast');
const gtInputs     = document.querySelectorAll('input[name="ground-truth"]');
const sortHistoryBtn = document.getElementById('sort-history-btn');
const groundTruthDoubt = document.getElementById('gt-doubt');

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initVACanvas();
  textInput.addEventListener('input', onInput);
  textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyzeEmotion();
  });

  gtInputs.forEach((input) => {
    input.addEventListener('change', () => {
      groundTruthEmotion = input.value;
      updateGroundTruthStyles();
      maybeTriggerGroundTruthDoubt();
    });
  });

  if (sortHistoryBtn) {
    sortHistoryBtn.textContent = 'Sort by Misreading: ON';
  }
});

function onInput() {
  charNum.textContent = textInput.value.length;
}

// ── Main analysis ─────────────────────────────────────────────────────────────
async function analyzeEmotion() {
  const text = textInput.value.trim();
  if (!text) { showToast('Please enter some text first.'); return; }
  if (!groundTruthEmotion) { showToast('Please select your felt emotion (Ground Truth).'); return; }

  setLoading(true);

  try {
    let response = null;
    let lastError = null;

    for (const endpoint of API_ENDPOINTS) {
      try {
        response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });
        if (response) break;
      } catch (e) {
        lastError = e;
      }
    }

    if (!response) {
      if (lastError?.message?.includes('expected pattern')) {
        throw new Error('Page origin is not valid for API requests. Please open the app from http://localhost:5001 (or the printed localhost port).');
      }
      throw new Error('Cannot connect to backend API. Please start the server with python run.py and open the localhost page.');
    }

    if (!response.ok) {
      let errMsg = `HTTP ${response.status}`;
      try {
        const err = await response.json();
        errMsg = err.error || errMsg;
      } catch (_) {
        // keep default message when response is not JSON
      }
      throw new Error(errMsg);
    }

    const data = await response.json();
    const misreading = computeMisreading(data.probabilities, groundTruthEmotion);
    const emotionalGap = computeEmotionalGap(data.predicted_emotion, groundTruthEmotion);

    const enriched = {
      ...data,
      ground_truth: groundTruthEmotion,
      ...misreading,
      emotional_gap: emotionalGap,
      timestamp: Date.now(),
    };

    lastResult = enriched;
    renderResults(enriched);
    addToHistory(enriched);
    updateStats(enriched.predicted_emotion);

  } catch (err) {
    showToast(err.message);
  } finally {
    setLoading(false);
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  const {
    predicted_emotion,
    confidence,
    emoji,
    color,
    probabilities,
    top_words,
  } = data;
  const cfg = EMOTION_CONFIG[predicted_emotion];

  // Show results panel
  placeholder.style.display = 'none';
  resultsBody.style.display = '';

  // Primary emotion
  document.getElementById('emotion-emoji').textContent = emoji;
  const nameEl = document.getElementById('emotion-name');
  nameEl.textContent = `System Interpretation: ${capitalize(predicted_emotion)}`;
  nameEl.style.color = color;

  const confBar = document.getElementById('conf-bar');
  confBar.style.background = color;
  // Animate width on next frame
  requestAnimationFrame(() => {
    confBar.style.width = `${(confidence * 100).toFixed(1)}%`;
  });
  document.getElementById('conf-label').textContent =
    `System Confidence: ${(confidence * 100).toFixed(1)} %`;

  // Misreading panel + critical layer
  renderMisreadingPanel(data);
  renderCriticalLayer(data);

  // Bar chart
  renderBarChart(probabilities);

  // Keywords
  renderKeywords(top_words, color);

  // Valence-Arousal circumplex
  const circCard = document.getElementById('circumplex-card');
  circCard.style.display = '';
  drawCircumplex(data);

  // Scroll to results (mobile)
  if (window.innerWidth < 960) {
    document.getElementById('results-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function computeMisreading(probabilities, groundTruth) {
  // This interface does not aim to understand emotion.
  // It exposes the gap between lived experience and algorithmic classification.
  // Misreading is not a bug, but a structural condition.
  const gtProbability = probabilities?.[groundTruth]?.probability ?? 0;
  const misreadingScore = clamp01(1 - gtProbability);

  let misreadingLabel = 'Accurate';
  if (misreadingScore > 0.7) {
    misreadingLabel = 'Severe Misreading';
  } else if (misreadingScore > 0.3) {
    misreadingLabel = 'Partial Misread';
  }

  return {
    ground_truth_probability: gtProbability,
    misreading_score: misreadingScore,
    misreading_label: misreadingLabel,
    high_misinterpretation: misreadingScore > 0.7,
  };
}

function computeEmotionalGap(predictedEmotion, groundTruth) {
  const algo = EMOTION_CONFIG[predictedEmotion];
  const human = EMOTION_CONFIG[groundTruth];
  if (!algo || !human) return 0;

  const distance = Math.sqrt(
    Math.pow(algo.valence - human.valence, 2) +
    Math.pow(algo.arousal - human.arousal, 2),
  );

  return Number(distance.toFixed(3));
}

function renderMisreadingPanel(data) {
  const scorePct = (data.misreading_score * 100).toFixed(1);
  const bar = document.getElementById('misreading-bar');
  const tag = document.getElementById('misreading-tag');
  const value = document.getElementById('misreading-value');
  const note = document.getElementById('misreading-note');

  bar.style.width = `${scorePct}%`;
  value.textContent = `${scorePct}%`;
  tag.textContent = data.misreading_label;

  if (data.misreading_score > 0.7) {
    tag.className = 'misreading-tag severe';
    note.textContent = 'System strongly diverges from reported human feeling';
  } else if (data.misreading_score > 0.3) {
    tag.className = 'misreading-tag partial';
    note.textContent = 'System partially aligns with your reported emotion';
  } else {
    tag.className = 'misreading-tag accurate';
    note.textContent = 'System interpretation is close to your reported emotion';
  }
}

function renderCriticalLayer(data) {
  const algoCfg = EMOTION_CONFIG[data.predicted_emotion];
  const humanCfg = EMOTION_CONFIG[data.ground_truth];
  const warningEl = document.getElementById('critical-warning');

  document.getElementById('critical-algorithm').innerHTML =
    `Algorithm thinks: ${algoCfg.emoji} ${capitalize(data.predicted_emotion)} (${(data.confidence * 100).toFixed(1)}%)`;
  document.getElementById('critical-human').innerHTML =
    `You feel: ${humanCfg.emoji} ${capitalize(data.ground_truth)}`;
  document.getElementById('critical-score').innerHTML =
    `Misreading Score: ${(data.misreading_score * 100).toFixed(1)}%`;

  if (data.high_misinterpretation) {
    warningEl.textContent = pickRandom(HIGH_MISREAD_PHRASES);
    warningEl.style.display = '';
    pulseSystemVoice(warningEl);
  } else {
    warningEl.style.display = 'none';
  }

  renderSystemStatement();
}

function renderSystemStatement() {
  const box = document.getElementById('system-statement');
  const text = document.getElementById('system-statement-text');
  box.style.display = '';
  text.textContent = pickRandom(SYSTEM_STATEMENTS);
  pulseSystemVoice(text);
}

// ── Bar chart ─────────────────────────────────────────────────────────────────
function renderBarChart(probabilities) {
  const labels = EMOTION_ORDER.map(e => capitalize(e));
  const values = EMOTION_ORDER.map(e => (probabilities[e]?.probability ?? 0) * 100);
  const colors = EMOTION_ORDER.map(e => EMOTION_CONFIG[e].color + 'cc');
  const borders = EMOTION_ORDER.map(e => EMOTION_CONFIG[e].color);
  const emojis = EMOTION_ORDER.map(e => EMOTION_CONFIG[e].emoji);

  if (chartInstance) { chartInstance.destroy(); }

  const ctx = document.getElementById('emotion-chart').getContext('2d');
  chartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels.map((l, i) => `${emojis[i]} ${l}`),
      datasets: [{
        label: 'Probability (%)',
        data: values,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 1.5,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.y.toFixed(2)} %`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#94a3b8', font: { size: 11 } },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          min: 0,
          max: 100,
          ticks: { color: '#94a3b8', font: { size: 11 }, callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.06)' },
        },
      },
    },
  });
}

// ── Keywords ──────────────────────────────────────────────────────────────────
function renderKeywords(topWords, color) {
  const wrap = document.getElementById('keywords-wrap');
  const chips = document.getElementById('keyword-chips');
  const tokenPhilosophy = document.getElementById('token-philosophy');
  chips.innerHTML = '';

  if (!topWords || topWords.length === 0) {
    wrap.style.display = 'none';
    return;
  }

  wrap.style.display = '';
  if (tokenPhilosophy) {
    tokenPhilosophy.title = 'Original sentence → Tokenized fragments';
  }
  const maxScore = topWords[0].score;
  topWords.forEach(({ word, score }) => {
    const alpha = 0.3 + 0.7 * (score / maxScore);
    const chip = document.createElement('span');
    chip.className = 'kwchip';
    chip.textContent = word;
    chip.style.borderColor = color;
    chip.style.color = color;
    chip.style.opacity = alpha.toFixed(2);
    chip.title = 'Original sentence → Tokenized fragments';
    chips.appendChild(chip);
  });
}

// ── Valence-Arousal Circumplex (Canvas) ───────────────────────────────────────
function initVACanvas() {
  vaCtx = document.getElementById('va-canvas').getContext('2d');
}

function drawCircumplex(data) {
  const canvas = document.getElementById('va-canvas');
  const W = canvas.width;
  const H = canvas.height;
  const cx = W / 2;
  const cy = H / 2;
  const R = Math.min(W, H) * 0.43;
  const ctx = vaCtx;

  ctx.clearRect(0, 0, W, H);

  // Background gradient
  const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
  grd.addColorStop(0,   'rgba(124,58,237,0.06)');
  grd.addColorStop(0.5, 'rgba(37,99,235,0.04)');
  grd.addColorStop(1,   'rgba(0,0,0,0)');
  ctx.fillStyle = grd;
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.fill();

  // Concentric circles
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1;
  [0.33, 0.66, 1.0].forEach(f => {
    ctx.beginPath();
    ctx.arc(cx, cy, R * f, 0, Math.PI * 2);
    ctx.stroke();
  });

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(cx - R - 8, cy); ctx.lineTo(cx + R + 8, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, cy - R - 8); ctx.lineTo(cx, cy + R + 8); ctx.stroke();
  ctx.setLineDash([]);

  // Axis labels
  ctx.font = '600 11px Inter, sans-serif';
  ctx.fillStyle = 'rgba(148,163,184,0.85)';
  ctx.textAlign = 'center';
  ctx.fillText('High Arousal', cx, cy - R - 14);
  ctx.fillText('Low Arousal',  cx, cy + R + 22);
  ctx.textAlign = 'left';
  ctx.fillText('Positive Valence', cx + R + 12, cy + 4);
  ctx.textAlign = 'right';
  ctx.fillText('Negative Valence', cx - R - 12, cy + 4);

  // All emotion reference dots
  EMOTION_ORDER.forEach(emotion => {
    const cfg = EMOTION_CONFIG[emotion];
    const ex = cx + cfg.valence * R;
    const ey = cy - cfg.arousal * R;

    const isActive = emotion === data.predicted_emotion;
    const baseAlpha = isActive ? 1 : 0.35;

    ctx.beginPath();
    ctx.arc(ex, ey, isActive ? 10 : 6, 0, Math.PI * 2);
    ctx.fillStyle = cfg.color + Math.round(baseAlpha * 255).toString(16).padStart(2, '0');
    ctx.fill();

    if (isActive) {
      ctx.strokeStyle = cfg.color;
      ctx.lineWidth = 2;
      ctx.stroke();
      // Pulse ring
      ctx.beginPath();
      ctx.arc(ex, ey, 17, 0, Math.PI * 2);
      ctx.strokeStyle = cfg.color + '44';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    ctx.font = isActive ? '600 12px Inter, sans-serif' : '500 10px Inter, sans-serif';
    ctx.fillStyle = isActive ? '#fff' : 'rgba(148,163,184,0.65)';
    ctx.textAlign = 'center';
    const labelY = ey + (cfg.arousal > 0 ? -15 : 20);
    ctx.fillText(capitalize(emotion), ex, labelY);
  });

  // Dual-point critical view: algorithm vs human-reported ground truth
  const algo = EMOTION_CONFIG[data.predicted_emotion];
  const human = EMOTION_CONFIG[data.ground_truth];
  if (algo && human) {
    const ax = cx + algo.valence * R;
    const ay = cy - algo.arousal * R;
    const hx = cx + human.valence * R;
    const hy = cy - human.arousal * R;

    // Gap line (critical mismatch)
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = 'rgba(255,77,79,0.9)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(hx, hy);
    ctx.stroke();
    ctx.setLineDash([]);

    const mx = (ax + hx) / 2;
    const my = (ay + hy) / 2;
    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillStyle = '#ff8a8c';
    ctx.textAlign = 'center';
    ctx.fillText(`Emotional Gap ${(data.emotional_gap ?? 0).toFixed(2)}`, mx, my - 8);

    // Algorithm point (warm yellow)
    ctx.beginPath();
    ctx.arc(ax, ay, 9, 0, Math.PI * 2);
    ctx.fillStyle = '#FBBF24';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(251,191,36,0.95)';
    ctx.stroke();

    // Ground Truth point (cool blue)
    ctx.beginPath();
    ctx.arc(hx, hy, 9, 0, Math.PI * 2);
    ctx.fillStyle = '#60A5FA';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(96,165,250,0.95)';
    ctx.stroke();

    ctx.font = '600 10px Inter, sans-serif';
    ctx.fillStyle = '#fde68a';
    ctx.fillText('Algorithm', ax, ay - 14);
    ctx.fillStyle = '#93c5fd';
    ctx.fillText('Ground Truth', hx, hy - 14);
  }

  // Update legend
  const cfg = EMOTION_CONFIG[data.predicted_emotion];
  const gt = EMOTION_CONFIG[data.ground_truth];
  document.getElementById('va-current').innerHTML = `
    <strong>Algorithm Emotion:</strong> ${cfg.emoji} ${capitalize(data.predicted_emotion)}<br/>
    <strong>Ground Truth Emotion:</strong> ${gt.emoji} ${capitalize(data.ground_truth)}<br/>
    Algorithm (V/A): <strong>${cfg.valence >= 0 ? '+' : ''}${cfg.valence.toFixed(1)} / ${cfg.arousal >= 0 ? '+' : ''}${cfg.arousal.toFixed(1)}</strong><br/>
    Ground Truth (V/A): <strong>${gt.valence >= 0 ? '+' : ''}${gt.valence.toFixed(1)} / ${gt.arousal >= 0 ? '+' : ''}${gt.arousal.toFixed(1)}</strong><br/>
    Emotional Gap: <strong>${(data.emotional_gap ?? 0).toFixed(3)}</strong><br/>
    System Confidence: <strong>${(data.confidence * 100).toFixed(1)} %</strong>
  `;
}

// ── History ───────────────────────────────────────────────────────────────────
function addToHistory(data) {
  history.unshift(data);
  if (history.length > 20) history.pop();

  renderHistory();
}

function renderHistory() {
  const histCard = document.getElementById('history-card');
  const histList = document.getElementById('history-list');

  const items = [...history];
  if (sortHistoryByMisreading) {
    items.sort((a, b) => b.misreading_score - a.misreading_score);
  } else {
    items.sort((a, b) => b.timestamp - a.timestamp);
  }

  histCard.style.display = '';
  histList.innerHTML = '';

  items.forEach(item => {
    const cfg = EMOTION_CONFIG[item.predicted_emotion];
    const gt = EMOTION_CONFIG[item.ground_truth];
    const div = document.createElement('div');
    div.className = 'hist-item';
    div.style.setProperty('--ec', misreadingColor(item.misreading_score));

    const truncated = item.text.length > 100
      ? item.text.slice(0, 100) + '…'
      : item.text;

    div.innerHTML = `
      <div class="hist-emoji">${item.emoji}</div>
      <div class="hist-body">
        <div class="hist-text">"${escHtml(truncated)}"</div>
        <div class="hist-meta">
          <span class="hist-emotion" style="color:${cfg.color}">Algorithm: ${cfg.emoji} ${capitalize(item.predicted_emotion)}</span>
          <span class="hist-human">Human: ${gt.emoji} ${capitalize(item.ground_truth)}</span>
          <span class="hist-misread">Misreading: ${(item.misreading_score * 100).toFixed(1)}%</span>
          <span>${new Date(item.timestamp).toLocaleTimeString()}</span>
        </div>
      </div>
    `;
    histList.appendChild(div);
  });
}

function toggleHistorySort() {
  sortHistoryByMisreading = !sortHistoryByMisreading;
  if (sortHistoryBtn) {
    sortHistoryBtn.textContent = `Sort by Misreading: ${sortHistoryByMisreading ? 'ON' : 'OFF'}`;
  }
  renderHistory();
}

function clearHistory() {
  history = [];
  document.getElementById('history-card').style.display = 'none';
  document.getElementById('history-list').innerHTML = '';
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats(emotion) {
  analysisCount++;
  emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
  document.getElementById('total-count').textContent = analysisCount;

  const topEmotion = Object.entries(emotionCounts)
    .sort((a, b) => b[1] - a[1])[0];
  const cfg = EMOTION_CONFIG[topEmotion[0]];
  document.getElementById('top-emotion-stat').textContent =
    `${cfg.emoji} ${capitalize(topEmotion[0])}`;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function tryExample(btn) {
  textInput.value = btn.textContent.trim();
  charNum.textContent = textInput.value.length;
  analyzeEmotion();
}

function clearAll() {
  textInput.value = '';
  charNum.textContent = '0';
  placeholder.style.display = '';
  resultsBody.style.display = 'none';
  document.getElementById('circumplex-card').style.display = 'none';
  document.getElementById('system-statement').style.display = 'none';
  if (groundTruthDoubt) groundTruthDoubt.style.display = 'none';
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
  textInput.focus();
}

function setLoading(on) {
  loadingOverlay.style.display = on ? '' : 'none';
  analyzeBtn.disabled = on;
  btnIcon.style.display = on ? 'none' : '';
  if (on) {
    const spinner = document.createElement('span');
    spinner.className = 'btn-spinner';
    spinner.id = 'btn-spinner-el';
    btnIcon.parentNode.insertBefore(spinner, btnIcon);
    btnLabel.textContent = 'Submitting…';
  } else {
    const s = document.getElementById('btn-spinner-el');
    if (s) s.remove();
    btnLabel.textContent = 'Submit to Analysis';
    btnIcon.style.display = '';
  }
}

let toastTimer = null;
function showToast(msg) {
  errorToast.textContent = msg;
  errorToast.style.display = '';
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { errorToast.style.display = 'none'; }, 4500);
}

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function misreadingColor(score) {
  if (score > 0.7) return '#FF4D4F';
  if (score > 0.3) return '#fb923c';
  return '#22c55e';
}

function updateGroundTruthStyles() {
  document.querySelectorAll('.gt-option').forEach((el) => {
    const input = el.querySelector('input');
    if (!input) return;
    if (input.checked) el.classList.add('active');
    else el.classList.remove('active');
  });
}

function maybeTriggerGroundTruthDoubt() {
  if (!groundTruthDoubt) return;

  groundTruthDoubt.style.display = 'none';
  if (Math.random() >= 0.2) return;

  const msg = pickRandom(GROUND_TRUTH_DOUBT_MESSAGES);
  const show = () => {
    groundTruthDoubt.textContent = msg;
    groundTruthDoubt.style.display = '';
    pulseSystemVoice(groundTruthDoubt);
    window.setTimeout(() => {
      groundTruthDoubt.style.display = 'none';
    }, 2400);
  };

  if (msg.includes('Re-evaluating')) {
    window.setTimeout(show, 350);
  } else {
    show();
  }
}

function pulseSystemVoice(el) {
  el.classList.remove('voice-fade');
  void el.offsetWidth;
  el.classList.add('voice-fade');
}

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function escHtml(str) {
  return str.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
}
