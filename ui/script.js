// =========================================================================
// AQI Predictor — UI logic
// =========================================================================

// Pollutant sliders — must match the features fed to the model (POLLUTANTS)
const SLIDERS = [
  { key: "PM2.5",   label: "PM2.5",   min: 0, max: 500, step: 1,  def: 80  },
  { key: "PM10",    label: "PM10",    min: 0, max: 600, step: 1,  def: 150 },
  { key: "NO",      label: "NO",      min: 0, max: 300, step: 1,  def: 10  },
  { key: "NO2",     label: "NO2",     min: 0, max: 200, step: 1,  def: 40  },
  { key: "NOx",     label: "NOx",     min: 0, max: 300, step: 1,  def: 50  },
  { key: "NH3",     label: "NH3",     min: 0, max: 200, step: 1,  def: 20  },
  { key: "CO",      label: "CO",      min: 0, max: 50,  step: 0.1, def: 1.0 },
  { key: "SO2",     label: "SO2",     min: 0, max: 200, step: 1,  def: 15  },
  { key: "O3",      label: "O3",      min: 0, max: 300, step: 1,  def: 50  },
  { key: "Benzene", label: "Benzene", min: 0, max: 50,  step: 0.1, def: 2.0 },
  { key: "Toluene", label: "Toluene", min: 0, max: 100, step: 0.1, def: 10  },
];

// AQI category thresholds (Indian CPCB)
function aqiCategory(aqi) {
  if (aqi <= 50)  return { name: "Good",         cls: "cat-good"         };
  if (aqi <= 100) return { name: "Satisfactory", cls: "cat-satisfactory" };
  if (aqi <= 200) return { name: "Moderate",     cls: "cat-moderate"     };
  if (aqi <= 300) return { name: "Poor",         cls: "cat-poor"         };
  if (aqi <= 400) return { name: "Very Poor",    cls: "cat-verypoor"     };
  return                 { name: "Severe",       cls: "cat-severe"       };
}

// -------------------------------------------------------------------------
// Build slider rows
// -------------------------------------------------------------------------
function buildSliders() {
  const host = document.getElementById("sliders");
  host.innerHTML = "";
  SLIDERS.forEach(s => {
    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML = `
      <label for="slider-${s.key}">${s.label}</label>
      <input type="range" id="slider-${s.key}"
             min="${s.min}" max="${s.max}" step="${s.step}" value="${s.def}" />
      <span class="slider-value" id="val-${s.key}">${s.def}</span>
    `;
    host.appendChild(row);
    const input = row.querySelector("input");
    const disp  = row.querySelector(".slider-value");
    input.addEventListener("input", () => { disp.textContent = input.value; });
  });
}

// -------------------------------------------------------------------------
// Collect inputs
// -------------------------------------------------------------------------
function collectInputs() {
  const out = {
    city:   document.getElementById("city").value,
    season: document.getElementById("season").value,
  };
  SLIDERS.forEach(s => {
    out[s.key] = parseFloat(document.getElementById(`slider-${s.key}`).value);
  });
  return out;
}

// -------------------------------------------------------------------------
// Render output
// -------------------------------------------------------------------------
function renderResult(aqi, shap) {
  document.getElementById("placeholder").classList.add("hidden");
  const box = document.getElementById("resultBox");
  box.classList.remove("hidden");

  // Big number
  document.getElementById("aqiNumber").textContent = aqi.toFixed(1);

  // Category badge
  const cat = aqiCategory(aqi);
  const badge = document.getElementById("aqiCategory");
  badge.textContent = cat.name;
  badge.className = "aqi-category " + cat.cls;

  // Gauge marker (0–500 → 0–100%)
  const pct = Math.min(Math.max(aqi / 500, 0), 1) * 100;
  document.getElementById("gaugeMarker").style.left = pct + "%";

  // SHAP bars
  const shapHost = document.getElementById("shapBars");
  shapHost.innerHTML = "";
  if (shap && Object.keys(shap).length) {
    const max = Math.max(...Object.values(shap).map(Math.abs)) || 1;
    Object.entries(shap)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 5)
      .forEach(([feat, val]) => {
        const row = document.createElement("div");
        row.className = "shap-row";
        const width = (Math.abs(val) / max) * 100;
        row.innerHTML = `
          <span class="shap-label">${feat}</span>
          <div class="shap-bar-track">
            <div class="shap-bar-fill" style="width:${width}%"></div>
          </div>
          <span class="shap-value">${val >= 0 ? "+" : ""}${val.toFixed(1)}</span>
        `;
        shapHost.appendChild(row);
      });
  } else {
    shapHost.innerHTML = `<p class="placeholder">SHAP explanation unavailable.</p>`;
  }
}

// -------------------------------------------------------------------------
// Predict
// -------------------------------------------------------------------------
async function predict() {
  const btn = document.getElementById("predictBtn");
  const status = document.getElementById("status");
  btn.disabled = true;
  status.classList.remove("error");
  status.textContent = "Predicting…";

  try {
    const payload = collectInputs();
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResult(data.aqi, data.shap);
    status.textContent = `Predicted using ${payload.city} model · R² = ${(data.r2 ?? 0).toFixed(2)}`;
  } catch (e) {
    console.error(e);
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
  } finally {
    btn.disabled = false;
  }
}

// -------------------------------------------------------------------------
// Init
// -------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  buildSliders();
  document.getElementById("predictBtn").addEventListener("click", predict);
});