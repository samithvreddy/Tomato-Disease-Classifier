/* ═══════════════════════════════════════════════════════════
   TomatoDoc — app.js
═══════════════════════════════════════════════════════════ */

// ── Tab navigation ─────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => {
      p.classList.remove('active');
      p.classList.add('hidden');
    });
    btn.classList.add('active');
    const panel = document.getElementById('tab-' + tab);
    panel.classList.remove('hidden');
    panel.classList.add('active');
  });
});

// ── Diagnose tab ───────────────────────────────────────────
const dropZone   = document.getElementById('dropZone');
const fileInput  = document.getElementById('fileInput');
const uploadIdle = document.getElementById('uploadIdle');
const uploadPrev = document.getElementById('uploadPreview');
const previewImg = document.getElementById('previewImg');
const clearBtn   = document.getElementById('clearBtn');
const analyseBtn = document.getElementById('analyseBtn');
const resultPanel  = document.getElementById('resultPanel');
const loadingPanel = document.getElementById('loadingPanel');

let selectedFile = null;

// Drag and drop
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) setFile(file);
});
dropZone.addEventListener('click', e => {
  if (e.target === dropZone || e.target.closest('.upload-idle')) {
    fileInput.click();
  }
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});
clearBtn.addEventListener('click', e => { e.stopPropagation(); resetDiagnose(); });

function setFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src = ev.target.result;
    uploadIdle.classList.add('hidden');
    uploadPrev.classList.remove('hidden');
    analyseBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function resetDiagnose() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  uploadIdle.classList.remove('hidden');
  uploadPrev.classList.add('hidden');
  analyseBtn.disabled = true;
  resultPanel.classList.add('hidden');
  loadingPanel.classList.add('hidden');
  document.getElementById('uploadCard').classList.remove('hidden');
}

analyseBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  document.getElementById('uploadCard').classList.add('hidden');
  resultPanel.classList.add('hidden');
  loadingPanel.classList.remove('hidden');

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    loadingPanel.classList.add('hidden');
    if (data.error) {
      showError(data.error);
      document.getElementById('uploadCard').classList.remove('hidden');
    } else {
      renderResult(data);
    }
  } catch (err) {
    loadingPanel.classList.add('hidden');
    document.getElementById('uploadCard').classList.remove('hidden');
    showError('Network error — is the Flask server running?');
  }
});

function renderResult(data) {
  const d = data.disease;

  // Name
  document.getElementById('resultName').textContent = d.display;

  // Confidence ring
  const pct = data.confidence;
  const circumference = 163.4;
  const offset = circumference - (pct / 100) * circumference;
  const fill = document.getElementById('ringFill');
  fill.style.strokeDashoffset = circumference; // reset
  requestAnimationFrame(() => {
    setTimeout(() => { fill.style.strokeDashoffset = offset; }, 50);
  });
  document.getElementById('confPct').textContent = pct + '%';
  // Colour the ring by confidence
  fill.style.stroke = pct >= 70 ? 'var(--green-leaf)' : pct >= 50 ? 'var(--yellow)' : 'var(--orange)';

  // Low confidence warning
  const warn = document.getElementById('lowConfWarning');
  data.low_confidence ? warn.classList.remove('hidden') : warn.classList.add('hidden');

  // Description
  document.getElementById('resultDesc').textContent = d.description;

  // Severity
  const badge = document.getElementById('severityBadge');
  badge.textContent = d.severity === 'none' ? '✅ No disease' :
                      d.severity === 'low'  ? 'Low severity' :
                      d.severity === 'medium' ? 'Medium severity' : 'High severity';
  badge.className = 'severity-badge severity-' + d.severity;

  // Treatment
  const treatSection = document.getElementById('treatmentSection');
  const treatList    = document.getElementById('treatmentList');
  treatList.innerHTML = '';
  if (d.treatment && d.treatment.length > 0) {
    d.treatment.forEach(t => {
      const li = document.createElement('li');
      li.textContent = t;
      treatList.appendChild(li);
    });
    treatSection.classList.remove('hidden');
  } else {
    treatSection.classList.add('hidden');
  }

  // Prevention
  const prevList = document.getElementById('preventionList');
  prevList.innerHTML = '';
  d.prevention.forEach(p => {
    const li = document.createElement('li');
    li.textContent = p;
    prevList.appendChild(li);
  });

  // Seasonal risk
  const seasons = d.seasonal_risk;
  const riskEl = {
    monsoon: document.getElementById('riskMonsoon'),
    summer:  document.getElementById('riskSummer'),
    winter:  document.getElementById('riskWinter'),
  };
  for (const [s, el] of Object.entries(riskEl)) {
    const level = seasons[s] || 'low';
    el.textContent = level.replace('_', ' ');
    el.className   = 'season-risk risk-' + level;
  }

  resultPanel.classList.remove('hidden');
}

function showError(msg) {
  const el = document.createElement('div');
  el.className = 'error-msg';
  el.textContent = '⚠️ ' + msg;
  document.getElementById('tab-diagnose').querySelector('.container').appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

// ── Weather tab ────────────────────────────────────────────
document.getElementById('weatherBtn').addEventListener('click', fetchWeather);
document.getElementById('cityInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') fetchWeather();
});

async function fetchWeather() {
  const city = document.getElementById('cityInput').value.trim();
  if (!city) return;

  const loading = document.getElementById('weatherLoading');
  const result  = document.getElementById('weatherResult');
  loading.classList.remove('hidden');
  result.classList.add('hidden');
  result.innerHTML = '';

  try {
    const res  = await fetch('/weather?city=' + encodeURIComponent(city));
    const data = await res.json();
    loading.classList.add('hidden');
    if (data.error) {
      result.innerHTML = `<div class="error-msg">⚠️ ${data.error}</div>`;
    } else {
      renderWeather(data);
    }
    result.classList.remove('hidden');
  } catch (err) {
    loading.classList.add('hidden');
    result.innerHTML = `<div class="error-msg">⚠️ Network error — is the server running?</div>`;
    result.classList.remove('hidden');
  }
}

function renderWeather(data) {
  const result = document.getElementById('weatherResult');
  let html = '';

  if (data.demo) {
    html += `<div class="demo-banner">⚠️ <strong>Demo mode:</strong> No API key set. Showing simulated weather data for "${data.city}". See README for setup.</div>`;
  }

  html += '<div class="forecast-grid">';
  for (const day of data.forecast) {
    const riskLabel = day.risk.charAt(0).toUpperCase() + day.risk.slice(1) + ' Risk';
    const diseaseList = day.risk_diseases.length
      ? `<div class="risk-diseases">Watch for:<br/>${day.risk_diseases.join(', ')}</div>`
      : `<div class="risk-diseases" style="color:var(--green-leaf)">No major disease risk today.</div>`;

    html += `
      <div class="forecast-card">
        <div class="forecast-day-header">${day.day}</div>
        <div class="forecast-body">
          <div class="forecast-stats">
            <div class="fstat">
              <span class="fstat-val">${day.temp}°C</span>
              <span class="fstat-label">avg temp</span>
            </div>
            <div class="fstat">
              <span class="fstat-val">${day.humidity}%</span>
              <span class="fstat-label">humidity</span>
            </div>
          </div>
          <div class="risk-chip ${day.risk}">${riskLabel}</div>
          ${diseaseList}
        </div>
      </div>`;
  }
  html += '</div>';
  result.innerHTML = html;
}

// ── Soil tab ───────────────────────────────────────────────
const sliders = [
  { id: 'phSlider',       val: 'phVal',       suffix: '' },
  { id: 'moistureSlider', val: 'moistureVal', suffix: '%' },
  { id: 'nSlider',        val: 'nVal',        suffix: '' },
  { id: 'pSlider',        val: 'pVal',        suffix: '' },
  { id: 'kSlider',        val: 'kVal',        suffix: '' },
];
sliders.forEach(({ id, val, suffix }) => {
  const slider = document.getElementById(id);
  const display = document.getElementById(val);
  slider.addEventListener('input', () => {
    display.textContent = slider.value + suffix;
  });
});

document.getElementById('soilBtn').addEventListener('click', checkSoil);

async function checkSoil() {
  const payload = {
    ph:         parseFloat(document.getElementById('phSlider').value),
    moisture:   parseFloat(document.getElementById('moistureSlider').value),
    nitrogen:   parseFloat(document.getElementById('nSlider').value),
    phosphorus: parseFloat(document.getElementById('pSlider').value),
    potassium:  parseFloat(document.getElementById('kSlider').value),
  };

  const result = document.getElementById('soilResult');
  result.innerHTML = '<div class="loading-panel"><div class="spinner"></div><p>Checking…</p></div>';
  result.classList.remove('hidden');

  try {
    const res  = await fetch('/soil', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    renderSoil(data);
  } catch (err) {
    result.innerHTML = `<div class="error-msg">⚠️ Network error.</div>`;
  }
}

function renderSoil(data) {
  const result = document.getElementById('soilResult');
  const overallLabel = data.overall === 'good' ? '✅ Soil conditions look good for tomato.' :
                       data.overall === 'fair' ? '⚠️ Soil has a few issues — see below.' :
                                                  '❌ Soil conditions are poor — action needed.';

  let html = `<div class="soil-overall overall-${data.overall}">${overallLabel}</div>`;

  html += '<div class="soil-params">';
  for (const issue of data.issues) {
    html += `
      <div class="soil-param ${issue.status}">
        <span class="param-name">${issue.param}</span>
        <span style="font-family:var(--font-mono);font-size:0.82rem;color:var(--ink-muted)">
          ${issue.value} &nbsp;/&nbsp; ideal ${issue.ideal}
        </span>
        <span class="param-status ${issue.status}">${issue.status.toUpperCase()}</span>
      </div>`;
  }
  html += '</div>';

  if (data.advice.length > 0) {
    html += '<div style="font-weight:700;font-size:0.88rem;margin-bottom:0.5rem;color:var(--ink)">Recommended Actions</div>';
    html += '<div class="soil-advice-list">';
    data.advice.forEach(a => {
      html += `<div class="soil-advice-item">💡 ${a}</div>`;
    });
    html += '</div>';
  }

  result.innerHTML = html;
}
