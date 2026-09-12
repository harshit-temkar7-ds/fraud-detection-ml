const API_BASE = 'http://localhost:8000';

// ── API helpers ────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(API_BASE + path, options);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.error('API error:', e);
    return null;
  }
}

// ── Clock ─────────────────────────────────────────────────
function startClock(el) {
  function tick() {
    const now = new Date();
    el.textContent = now.toLocaleTimeString('en-IN', { hour12: false }) +
      ' IST';
  }
  tick();
  setInterval(tick, 1000);
}

// ── Format helpers ─────────────────────────────────────────
function fmtCurrency(n) {
  if (n >= 1e6) return '₹' + (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return '₹' + (n / 1e3).toFixed(1) + 'K';
  return '₹' + n.toFixed(2);
}

function fmtPct(n) { return (n * 100).toFixed(2) + '%'; }

function riskBadge(level) {
  const map = { HIGH: 'badge-high', MEDIUM: 'badge-medium', LOW: 'badge-low' };
  return `<span class="badge ${map[level] || 'badge-low'}">${level}</span>`;
}

function fraudBadge(isFraud) {
  return isFraud
    ? `<span class="badge badge-fraud">⚠ FRAUD</span>`
    : `<span class="badge badge-legit">✓ LEGIT</span>`;
}

function probBar(prob) {
  const pct = (prob * 100).toFixed(0);
  const color = prob > 0.7 ? '#ff2d55' : prob > 0.4 ? '#ff9500' : '#00ff88';
  return `<span class="prob-bar"><span class="prob-fill" style="width:${pct}%;background:${color}"></span></span>${pct}%`;
}

function timeAgo(ts) {
  const diff = Math.floor((Date.now() - new Date(ts)) / 1000);
  if (diff < 60) return diff + 's ago';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  return Math.floor(diff / 3600) + 'h ago';
}

// ── Sidebar active link ───────────────────────────────────
function setActiveNav() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-item').forEach(a => {
    a.classList.remove('active');
    if (a.getAttribute('href') === path) a.classList.add('active');
  });
}

// ── Mini sparkline canvas ─────────────────────────────────
function drawSparkline(canvas, data, color = '#00d4ff') {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const max = Math.max(...data), min = Math.min(...data);
  const range = max - min || 1;
  ctx.clearRect(0, 0, w, h);
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * (h - 4) - 2;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Fill gradient
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color + '33');
  grad.addColorStop(1, color + '00');
  ctx.fillStyle = grad;
  ctx.fill();
}

// ── Generic Chart.js bar ──────────────────────────────────
function drawBarChart(canvasId, labels, datasets, options = {}) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (ctx._chart) ctx._chart.destroy();
  ctx._chart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, ...options.plugins },
      scales: {
        x: {
          grid: { color: '#1a3a5c55' },
          ticks: { color: '#7fa8c9', font: { family: 'Share Tech Mono', size: 9 } }
        },
        y: {
          grid: { color: '#1a3a5c55' },
          ticks: { color: '#7fa8c9', font: { family: 'Share Tech Mono', size: 9 } }
        }
      },
      ...options
    }
  });
  return ctx._chart;
}

function drawLineChart(canvasId, labels, datasets, options = {}) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (ctx._chart) ctx._chart.destroy();
  ctx._chart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#7fa8c9', font: { family: 'Share Tech Mono', size: 10 } } }, ...options.plugins },
      scales: {
        x: {
          grid: { color: '#1a3a5c44' },
          ticks: { color: '#7fa8c9', font: { family: 'Share Tech Mono', size: 9 } }
        },
        y: {
          grid: { color: '#1a3a5c44' },
          ticks: { color: '#7fa8c9', font: { family: 'Share Tech Mono', size: 9 } }
        }
      },
      ...options
    }
  });
  return ctx._chart;
}

function drawDoughnut(canvasId, labels, data, colors) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (ctx._chart) ctx._chart.destroy();
  ctx._chart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{ data, backgroundColor: colors, borderWidth: 0, hoverOffset: 4 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '72%',
      plugins: {
        legend: { position: 'bottom', labels: { color: '#7fa8c9', font: { family: 'Share Tech Mono', size: 10 }, padding: 12 } }
      }
    }
  });
  return ctx._chart;
}

// ── Sidebar HTML ──────────────────────────────────────────
function renderSidebar(activePage) {
  return `
    <div class="sidebar">
      <div class="sidebar-logo">
        <div class="logo-text">SENTINEL</div>
        <div class="logo-sub">FRAUD DETECTION SYSTEM</div>
        <div class="status-pill"><span class="status-dot"></span>SYSTEM ONLINE</div>
      </div>
      <div class="nav-section">
        <div class="nav-label">NAVIGATION</div>
        <a href="/" class="nav-item ${activePage==='home'?'active':''}"><span class="nav-icon">⊞</span>Overview</a>
        <a href="/monitor" class="nav-item ${activePage==='monitor'?'active':''}"><span class="nav-icon">◉</span>Live Monitor</a>
        <a href="/analyzer" class="nav-item ${activePage==='analyzer'?'active':''}"><span class="nav-icon">⬡</span>Analyzer</a>
        <a href="/performance" class="nav-item ${activePage==='performance'?'active':''}"><span class="nav-icon">▲</span>Performance</a>
        <a href="/shap" class="nav-item ${activePage==='shap'?'active':''}"><span class="nav-icon">◈</span>SHAP Explain</a>
      </div>
      <div class="sidebar-footer">
        <div class="api-status" id="apiStatusText">● CHECKING API...</div>
      </div>
    </div>`;
}

function renderTopbar(title) {
  return `
    <div class="topbar">
      <div class="page-title">${title}</div>
      <div class="topbar-right">
        <div class="live-badge"><span class="live-dot"></span>LIVE</div>
        <div class="clock" id="clock">--:--:-- IST</div>
      </div>
    </div>`;
}

async function checkApiStatus() {
  const data = await apiFetch('/health');
  const el = document.getElementById('apiStatusText');
  if (el) {
    if (data && data.status === 'online') {
      el.innerHTML = '<span style="color:#00ff88">● API ONLINE</span>';
    } else {
      el.innerHTML = '<span style="color:#ff2d55">● API OFFLINE</span>';
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const clockEl = document.getElementById('clock');
  if (clockEl) startClock(clockEl);
  checkApiStatus();
});
