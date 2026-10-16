// Index page, "Best by trip time" tab: each model's combined Pareto front over
// every published sequence (docs/data/<model>/fronts.json, built by fronts.py),
// or every model's front on one chart. Plotly loads only when the tab is first
// opened, so the gallery stays light.

const ALL = '*';
const HELP = {
  model: 'Step line: for each trip time, the least propellant any sequence needs (the combined ' +
         'Pareto front of this model). Coloured points: the sequences that win the widest time ' +
         'ranges; grey: other sequences\' own fronts. Click a point to open the trajectory.',
  all: 'One step line per model: for each trip time, the least propellant that model needs over ' +
       'every published sequence. Same launch and end orbit for all, so the lines compare directly. ' +
       'Click a point to open the trajectory.',
};

const fronts = { models: [], loaded: {}, plotlyReady: null, bound: false };

function showTab(name) {
  document.querySelectorAll('.tabs a').forEach(a => a.classList.toggle('on', a.dataset.tab === name));
  document.querySelectorAll('section[data-panel]').forEach(s => { s.hidden = s.dataset.panel !== name; });
  if (name === 'fronts') openFronts();
}

const currentTab = () => (location.hash === '#fronts' ? 'fronts' : 'runs');
window.addEventListener('hashchange', () => showTab(currentTab()));
showTab(currentTab());

function loadPlotly() {
  if (!fronts.plotlyReady) {
    fronts.plotlyReady = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'vendor/plotly.min.js';
      script.onload = resolve;
      script.onerror = () => reject(new Error('could not load vendor/plotly.min.js'));
      document.head.appendChild(script);
    });
  }
  return fronts.plotlyReady;
}

async function openFronts() {
  const plot = document.getElementById('frontPlot');
  try {
    const [runs] = await Promise.all([loadJSON('data/index.json'), loadPlotly()]);
    if (!fronts.bound) {
      fronts.models = [...new Set(liveRuns(runs).map(run => run.model))]
        .sort((a, b) => modelRank(a) - modelRank(b) || a.localeCompare(b));   // gallery.js order
      const select = document.getElementById('frontModel');
      if (fronts.models.length > 1) select.add(new Option('all models', ALL));
      fronts.models.forEach(model => select.add(new Option(model, model)));
      const saved = sessionStorageGet('frontModel');
      if (saved && [ALL, ...fronts.models].includes(saved)) select.value = saved;
      select.onchange = () => { sessionStorageSet('frontModel', select.value); show(select.value); };
      fronts.bound = true;
    }
    await show(document.getElementById('frontModel').value);
  } catch (error) {
    plot.innerHTML = `<p class="placeholder">Could not load the fronts (${error.message}).</p>`;
  }
}

function sessionStorageGet(key) { try { return sessionStorage.getItem(key); } catch { return null; } }
function sessionStorageSet(key, value) { try { sessionStorage.setItem(key, value); } catch { /* private mode */ } }

const years = d => d / 365.25;
const openUrl = (model, seq, id) => `sequence.html?model=${encodeURIComponent(model)}` +
  `&seq=${encodeURIComponent(seq)}&pt=${encodeURIComponent(id)}`;

async function loadFront(model) {
  if (!(model in fronts.loaded)) {
    fronts.loaded[model] = await loadJSON(`data/${model}/fronts.json`).catch(() => null);
  }
  return fronts.loaded[model];
}

// Clear a placeholder message, but never the live chart: Plotly keeps state on
// the element and would "update" a chart whose DOM had been removed.
function preparePlot() {
  const plot = document.getElementById('frontPlot');
  if (plot.querySelector('.placeholder')) plot.innerHTML = '';
  return plot;
}

// Plotly adds .on() to the element on its first plot, so bind after drawing.
function bindClicks(plot) {
  if (!plot.dataset.clickBound) {
    plot.on('plotly_click', event => {
      const [model, seq, id] = event.points[0].customdata || [];
      if (seq) location.href = openUrl(model, seq, id);
    });
    plot.dataset.clickBound = '1';
  }
}

function layoutFor(t, yTitle, extra = {}) {
  return {
    paper_bgcolor: t.surface, plot_bgcolor: t.surface, font: { color: t.ink2 },
    margin: { l: 70, r: 10, t: 10, b: 50 }, hovermode: 'closest',
    xaxis: { title: 'Time of flight, launch → in orbit (years)', gridcolor: t.grid, zeroline: false },
    yaxis: { title: yTitle, gridcolor: t.grid, zeroline: false },
    legend: { orientation: 'h', y: 1.08, x: 0 }, ...extra,
  };
}

async function show(choice) {
  document.getElementById('frontHelp').textContent = choice === ALL ? HELP.all : HELP.model;
  return choice === ALL ? drawAll() : drawFronts(choice);
}

// Every model's combined front on one chart (all costs are propellant, kg).
async function drawAll() {
  const t = theme();
  const loaded = (await Promise.all(fronts.models.map(loadFront)))
    .map((data, i) => [fronts.models[i], data]).filter(([, data]) => data);
  const plot = preparePlot();
  const traces = [], rows = [];
  loaded.forEach(([model, data], i) => {
    const best = data.points.filter(p => p.best).sort((a, b) => a.tof_days - b.tof_days);
    if (!best.length) return;
    const color = i < 3 ? t.cat[i] : t.catOther;
    traces.push({
      type: 'scatter', mode: 'lines+markers', name: model,
      x: best.map(p => years(p.tof_days)), y: best.map(p => p.cost),
      line: { shape: 'hv', color, width: 2 }, marker: { size: 6, color },
      customdata: best.map(p => [model, p.seq, p.id]), hoverinfo: 'text',
      text: best.map(p => `<b>${model}</b> · ${p.seq}<br>${fmtCost(p.cost, data.cost_units)}<br>` +
                          `${years(p.tof_days).toFixed(2)} yr`),
    });
    const cheapest = best.reduce((a, b) => (b.cost < a.cost ? b : a));
    const fastest = best[0];
    const cell = p => `${fmtCost(p.cost, data.cost_units)} · ${years(p.tof_days).toFixed(1)} yr · ` +
                      `<a href="${openUrl(model, p.seq, p.id)}">${p.seq}</a>`;
    rows.push(`<tr><td><span class="swatch" style="background:${color}"></span>${model}</td>
      <td>${data.counts || ''}</td><td class="num">${cell(cheapest)}</td><td class="num">${cell(fastest)}</td></tr>`);
  });
  Plotly.react(plot, traces, layoutFor(t, 'Propellant (kg)'), { displayModeBar: false, responsive: true });
  bindClicks(plot);
  document.getElementById('frontNote').textContent = `${traces.length} models`;
  document.getElementById('frontTable').innerHTML =
    `<tr><th>model</th><th>cost counts</th><th class="num">least propellant</th><th class="num">fastest</th></tr>${rows.join('')}`;
}

async function drawFronts(model) {
  const plot = document.getElementById('frontPlot');
  const table = document.getElementById('frontTable');
  const data = await loadFront(model);
  if (!data) {
    Plotly.purge(plot);
    plot.innerHTML = `<p class="placeholder">No combined front for <b>${model}</b> yet: run
      <code>python orbit_planning/fronts.py</code> (or publish.py).</p>`;
    table.innerHTML = '';
    document.getElementById('frontNote').textContent = '';
    return;
  }
  preparePlot();
  const t = theme();
  const best = data.points.filter(p => p.best).sort((a, b) => a.tof_days - b.tof_days);
  const rest = data.points.filter(p => !p.best);

  // Which sequence wins where: contiguous stretches of the step line.
  const spans = [];
  best.forEach((p, i) => {
    const next = best[i + 1];
    const end = next ? next.tof_days : p.tof_days;
    const last = spans[spans.length - 1];
    if (last && last.seq === p.seq) { last.to = end; last.points.push(p); }
    else spans.push({ seq: p.seq, from: p.tof_days, to: end, points: [p] });
  });
  const won = {};
  spans.forEach(s => { won[s.seq] = (won[s.seq] || 0) + Math.max(s.to - s.from, 1); });
  // Colour only the three sequences that win the widest time ranges (the
  // all-pairs-safe slots for a scatter); every other winner is neutral and
  // named by its direct label and in the table, so identity never rests on hue.
  const ranked = Object.keys(won).sort((a, b) => won[b] - won[a]);
  const colorOf = seq => (ranked.indexOf(seq) < 3 ? t.cat[ranked.indexOf(seq)] : t.catOther);
  const units = data.cost_units;
  const hover = p => `<b>${p.seq}</b><br>${fmtCost(p.cost, units)}<br>${years(p.tof_days).toFixed(2)} yr` +
                     `<br>launch ${isoDay(mjd2000ToDate(p.launch_mjd2000))}`;

  const traces = [{
    type: 'scattergl', mode: 'markers', name: "other sequences' fronts",
    x: rest.map(p => years(p.tof_days)), y: rest.map(p => p.cost),
    customdata: rest.map(p => [model, p.seq, p.id]), text: rest.map(hover), hoverinfo: 'text',
    marker: { size: 5, color: t.ink2, opacity: 0.22 },
  }, {
    type: 'scatter', mode: 'lines', name: 'best by trip time', line: { shape: 'hv', color: t.ink2, width: 1.5 },
    x: best.map(p => years(p.tof_days)), y: best.map(p => p.cost), hoverinfo: 'skip',
  }];
  const legendOrder = ranked.slice(0, 3);
  if (ranked.length > 3) legendOrder.push('other winners');
  legendOrder.forEach(group => {
    const pts = best.filter(p => (group === 'other winners' ? ranked.indexOf(p.seq) >= 3 : p.seq === group));
    traces.push({
      type: 'scatter', mode: 'markers', name: group,
      x: pts.map(p => years(p.tof_days)), y: pts.map(p => p.cost),
      customdata: pts.map(p => [model, p.seq, p.id]), text: pts.map(hover), hoverinfo: 'text',
      marker: { size: 10, color: group === 'other winners' ? t.catOther : colorOf(group),
                line: { width: 2, color: t.surface } },
    });
  });
  // Direct labels: one per winning stretch (the longest stretches if there are many).
  const labelled = spans.length > 12 ? [...spans].sort((a, b) => (b.to - b.from) - (a.to - a.from)).slice(0, 12) : spans;
  const annotations = labelled.map(s => ({
    x: years(s.points[0].tof_days), y: s.points[0].cost, text: s.seq, showarrow: false,
    yshift: 14, xanchor: 'left', font: { size: 11, color: t.ink2 },
  }));

  Plotly.react(plot, traces, layoutFor(t, `${data.cost_label} (${units})`, { annotations }),
               { displayModeBar: false, responsive: true });
  bindClicks(plot);

  document.getElementById('frontNote').textContent =
    `${data.counts ? data.counts + ' · ' : ''}${data.runs.length} sequences · ${best.length} points on the front`;
  const rows = spans.map(s => {
    const cheapest = s.points.reduce((a, b) => (b.cost < a.cost ? b : a));
    const range = s.from === s.to ? `${years(s.from).toFixed(1)} yr`
                                  : `${years(s.from).toFixed(1)}–${years(s.to).toFixed(1)} yr`;
    return `<tr><td><span class="swatch" style="background:${colorOf(s.seq)}"></span>${s.seq}</td>
      <td>${range}</td><td class="num">${fmtCost(cheapest.cost, units)}</td>
      <td><a href="${openUrl(model, s.seq, cheapest.id)}">open</a></td></tr>`;
  }).join('');
  table.innerHTML = `<tr><th>sequence</th><th>best for trip times</th><th class="num">least propellant there</th><th></th></tr>${rows}`;
}
