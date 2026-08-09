// ---------------------------------------------------------------------
// Tiny Shakespeare LM — frontend
// ---------------------------------------------------------------------

/* ---------------------------- tab switching --------------------------- */
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

/* ---------------------------- token coloring --------------------------- */
// interpolate brick -> gold -> sage as confidence goes 0 -> 1
const COLOR_STOPS = [
  { p: 0.0, rgb: [177, 71, 63] },   // brick  (unsure)
  { p: 0.5, rgb: [201, 162, 39] },  // gold
  { p: 1.0, rgb: [127, 161, 126] }, // sage   (confident)
];

function lerp(a, b, t) { return a + (b - a) * t; }

function probToColor(prob) {
  if (prob === null || prob === undefined) return null;
  let lo = COLOR_STOPS[0], hi = COLOR_STOPS[COLOR_STOPS.length - 1];
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    if (prob >= COLOR_STOPS[i].p && prob <= COLOR_STOPS[i + 1].p) {
      lo = COLOR_STOPS[i]; hi = COLOR_STOPS[i + 1]; break;
    }
  }
  const span = hi.p - lo.p || 1;
  const t = (prob - lo.p) / span;
  const rgb = [0, 1, 2].map((i) => Math.round(lerp(lo.rgb[i], hi.rgb[i], t)));
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.38)`;
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Render {char, prob}[] token lists into a token-stream container. */
function renderTokens(container, promptTokens, generatedTokens) {
  container.classList.remove("placeholder");
  let html = "";
  (promptTokens || []).forEach((t) => {
    html += `<span class="tok prompt">${escapeHtml(t.char)}</span>`;
  });
  (generatedTokens || []).forEach((t) => {
    const bg = probToColor(t.prob);
    const pct = t.prob !== null && t.prob !== undefined ? Math.round(t.prob * 100) : null;
    const style = bg ? `background:${bg}` : "";
    const title = pct !== null ? ` title="${pct}% confident"` : "";
    html += `<span class="tok" style="${style}"${title}>${escapeHtml(t.char)}</span>`;
  });
  container.innerHTML = html || "<span class='placeholder'>(empty)</span>";
}

/* ------------------------------- status chip ---------------------------- */
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

function setStatus(mode, text) {
  statusDot.className = "dot" + (mode ? ` ${mode}` : "");
  statusText.textContent = text;
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.is_training) {
      setStatus("live", `training · step ${data.current_step}/${data.max_iters}`);
    } else {
      setStatus("", `idle · vocab ${data.vocab_size} · ${data.device}`);
    }
    return data;
  } catch (e) {
    setStatus("error", "backend unreachable");
    return null;
  }
}

/* --------------------------------- talk tab ------------------------------ */
const talkPrompt = document.getElementById("talk-prompt");
const talkLength = document.getElementById("talk-length");
const talkLengthVal = document.getElementById("talk-length-val");
const talkTemp = document.getElementById("talk-temp");
const talkTempVal = document.getElementById("talk-temp-val");
const talkBtn = document.getElementById("talk-generate-btn");
const talkNote = document.getElementById("talk-note");
const talkOutput = document.getElementById("talk-output");

talkLength.addEventListener("input", () => { talkLengthVal.textContent = `${talkLength.value} chars`; });
talkTemp.addEventListener("input", () => { talkTempVal.textContent = talkTemp.value; });

talkBtn.addEventListener("click", async () => {
  talkBtn.disabled = true;
  talkNote.textContent = "generating…";
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: talkPrompt.value || "\n",
        max_new_tokens: parseInt(talkLength.value, 10),
        temperature: parseFloat(talkTemp.value),
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderTokens(talkOutput, data.prompt_tokens, data.generated_tokens);
    talkNote.textContent = "";
  } catch (e) {
    talkNote.textContent = `error: ${e.message}`;
  } finally {
    talkBtn.disabled = false;
  }
});

/* -------------------------------- train tab ------------------------------- */
const startBtn = document.getElementById("train-start-btn");
const stopBtn = document.getElementById("train-stop-btn");
const resetBtn = document.getElementById("train-reset-btn");
const progressFill = document.getElementById("progress-fill");
const progressText = document.getElementById("progress-text");
const banner = document.getElementById("train-banner");
const sampleEl = document.getElementById("train-sample");

function showBanner(kind, msg) {
  banner.className = `banner ${kind}`;
  banner.textContent = msg;
}
function hideBanner() {
  banner.className = "banner hidden";
}

function currentConfig() {
  return {
    max_iters: parseInt(document.getElementById("cfg-max-iters").value, 10),
    batch_size: parseInt(document.getElementById("cfg-batch-size").value, 10),
    block_size: parseInt(document.getElementById("cfg-block-size").value, 10),
    learning_rate: parseFloat(document.getElementById("cfg-lr").value),
    eval_interval: parseInt(document.getElementById("cfg-eval-interval").value, 10),
    warmup_iters: parseInt(document.getElementById("cfg-warmup").value, 10),
  };
}

/* ---- charts ---- */
Chart.defaults.color = "#a7a394";
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 10;
Chart.defaults.borderColor = "#2a2f3b";

function makeChart(canvasId, datasetsMeta) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: datasetsMeta.map((m) => ({
        label: m.label,
        data: [],
        borderColor: m.color,
        backgroundColor: m.color,
        borderWidth: 1.75,
        pointRadius: 0,
        tension: 0.15,
      })),
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { display: false },
        y: { grid: { color: "#242833" }, ticks: { maxTicksLimit: 4 } },
      },
      plugins: {
        legend: {
          display: datasetsMeta.length > 1,
          labels: { boxWidth: 10, padding: 8 },
        },
        tooltip: { enabled: true },
      },
    },
  });
}

const charts = {
  loss: makeChart("chart-loss", [
    { label: "train", color: "#e0c473" },
    { label: "val", color: "#7fa17e" },
  ]),
  ppl: makeChart("chart-ppl", [
    { label: "train", color: "#e0c473" },
    { label: "val", color: "#7fa17e" },
  ]),
  lr: makeChart("chart-lr", [{ label: "lr", color: "#c9a227" }]),
  gradnorm: makeChart("chart-gradnorm", [{ label: "grad norm", color: "#b1473f" }]),
  paramnorm: makeChart("chart-paramnorm", [{ label: "param norm", color: "#8aa8c9" }]),
  tokens: makeChart("chart-tokens", [{ label: "tokens", color: "#c9a227" }]),
};

function pushPoint(chart, label, values) {
  chart.data.labels.push(label);
  chart.data.datasets.forEach((ds, i) => {
    const v = values[i];
    ds.data.push(v === undefined ? null : v);
  });
  // keep charts light: cap history shown
  const CAP = 400;
  if (chart.data.labels.length > CAP) {
    chart.data.labels.shift();
    chart.data.datasets.forEach((ds) => ds.data.shift());
  }
  chart.update("none");
}

function clearCharts() {
  Object.values(charts).forEach((c) => {
    c.data.labels = [];
    c.data.datasets.forEach((ds) => (ds.data = []));
    c.update("none");
  });
}

/* ---- overfitting / spike heuristics ---- */
let recentValLoss = [];
let recentTrainLoss = [];
let recentGradNorms = [];

function checkWarnings(ev) {
  if (ev.grad_norm !== undefined) {
    recentGradNorms.push(ev.grad_norm);
    if (recentGradNorms.length > 20) recentGradNorms.shift();
    if (recentGradNorms.length >= 6) {
      const median = [...recentGradNorms].sort((a, b) => a - b)[Math.floor(recentGradNorms.length / 2)];
      if (median > 0 && ev.grad_norm > median * 6) {
        showBanner("warn", `⚠️ Gradient spike at step ${ev.step}: norm ${ev.grad_norm.toFixed(2)} (recent median ${median.toFixed(2)})`);
        return;
      }
    }
  }
  if (ev.type === "eval") {
    recentValLoss.push(ev.val_loss);
    recentTrainLoss.push(ev.train_loss);
    if (recentValLoss.length > 5) { recentValLoss.shift(); recentTrainLoss.shift(); }
    if (recentValLoss.length >= 3) {
      const n = recentValLoss.length;
      const valRising = recentValLoss[n - 1] > recentValLoss[n - 2] && recentValLoss[n - 2] > recentValLoss[n - 3];
      const trainFalling = recentTrainLoss[n - 1] < recentTrainLoss[n - 3];
      if (valRising && trainFalling) {
        showBanner("warn", `⚠️ Validation loss rising while training loss falls near step ${ev.step} — possible overfitting.`);
        return;
      }
    }
  }
  hideBanner();
}

/* ---- SSE stream ---- */
let evtSource = null;

function connectTrainStream() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource("/api/train/stream");

  evtSource.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);

    if (ev.type === "step" || ev.type === "eval") {
      progressFill.style.width = `${Math.min(100, (ev.step / parseInt(document.getElementById("cfg-max-iters").value, 10)) * 100)}%`;
      progressText.textContent = `step ${ev.step}`;

      pushPoint(charts.lr, ev.step, [ev.lr]);
      pushPoint(charts.gradnorm, ev.step, [ev.grad_norm]);
      pushPoint(charts.paramnorm, ev.step, [ev.param_norm]);
      pushPoint(charts.tokens, ev.step, [ev.tokens_processed]);

      if (ev.type === "eval") {
        pushPoint(charts.loss, ev.step, [ev.train_loss, ev.val_loss]);
        pushPoint(charts.ppl, ev.step, [ev.train_ppl, ev.val_ppl]);
        if (ev.sample) renderTokens(sampleEl, [{ char: "", prob: null }], ev.sample.tokens);
      } else {
        pushPoint(charts.loss, ev.step, [ev.train_loss, undefined]);
        pushPoint(charts.ppl, ev.step, [ev.train_ppl, undefined]);
      }

      checkWarnings(ev);
      setStatus("live", `training · step ${ev.step}`);
    } else if (ev.type === "error") {
      showBanner("warn", `⚠️ ${ev.message}`);
      setStatus("error", "training stopped: error");
    } else if (ev.type === "done") {
      startBtn.disabled = false;
      stopBtn.disabled = true;
      resetBtn.disabled = false;
      setStatus("", `idle · finished at step ${ev.step}`);
      evtSource.close();
      evtSource = null;
    }
  };

  evtSource.onerror = () => {
    // server closes the stream intentionally when idle/finished;
    // avoid noisy auto-reconnect loops.
    if (evtSource) { evtSource.close(); evtSource = null; }
  };
}

startBtn.addEventListener("click", async () => {
  hideBanner();
  startBtn.disabled = true;
  resetBtn.disabled = true;
  try {
    const res = await fetch("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentConfig()),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    stopBtn.disabled = false;
    connectTrainStream();
  } catch (e) {
    showBanner("warn", `⚠️ Could not start training: ${e.message}`);
    startBtn.disabled = false;
    resetBtn.disabled = false;
  }
});

stopBtn.addEventListener("click", async () => {
  stopBtn.disabled = true;
  await fetch("/api/train/stop", { method: "POST" });
});

resetBtn.addEventListener("click", async () => {
  resetBtn.disabled = true;
  try {
    const res = await fetch("/api/train/reset", { method: "POST" });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    clearCharts();
    recentValLoss = []; recentTrainLoss = []; recentGradNorms = [];
    progressFill.style.width = "0%";
    progressText.textContent = "step 0 / 0";
    sampleEl.className = "token-stream placeholder";
    sampleEl.textContent = "A sample generation will appear here after the first evaluation pass, so you can watch the prose sharpen up as training proceeds.";
    hideBanner();
  } catch (e) {
    showBanner("warn", `⚠️ Could not reset: ${e.message}`);
  } finally {
    resetBtn.disabled = false;
  }
});

/* ---------------------------------- init ---------------------------------- */
(async function init() {
  const status = await refreshStatus();
  progressText.textContent = `step ${status?.current_step ?? 0} / ${status?.max_iters ?? 0}`;
  if (status?.max_iters) {
    document.getElementById("cfg-max-iters").value = status.max_iters;
  }
  if (status?.is_training) {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    resetBtn.disabled = true;
    connectTrainStream();
  }
})();
