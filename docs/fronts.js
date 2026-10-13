// Index page, "Best by trip time" tab: each model's combined Pareto front over
// every published sequence (docs/data/<model>/fronts.json, built by fronts.py).
// Plotly loads only when the tab is first opened, so the gallery stays light.

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
      fronts.models = [...new Set(runs.map(run => run.model))]
        .sort((a, b) => modelRank(a) - modelRank(b) || a.localeCompare(b));   // gallery.js order
      const select = document.getElementById('frontModel');
      fronts.models.forEach(model => select.add(new Option(model, model)));
      const saved = sessionStorageGet('frontModel');
      if (saved && fronts.models.includes(saved)) select.value = saved;
      select.onchange = () => { sessionStorageSet('frontModel', select.value); drawFronts(select.value); };
      fronts.bound = true;
    }
    await drawFronts(document.getElementById('frontModel').value);
  } catch (error) {
    plot.innerHTML = `<p class="placeholder">Could not load the fronts (${error.message}).</p>`;
  }
}

function sessionStorageGet(key) { try { return sessionStorage.getItem(key); } catch { return null; } }
function sessionStorageSet(key, value) { try { sessionStorage.setItem(key, value); } catch { /* private mode */ } }

async function drawFronts(model) {
  const plot = document.getElementById('frontPlot');
  const table = document.getElementById('frontTable');
  let data = fronts.loaded[model];
  if (!data) {
    try {
      data = fronts.loaded[model] = await loadJSON(`data/${model}/fronts.json`);
    } catch {
      Plotly.purge(plot);
      plot.innerHTML = `<p class="placeholder">No combined front for <b>${model}</b> yet: run
        <code>python orbit_planning/fronts.py</code> (or publish.py) and commit docs/data.</p>`;
      table.innerHTML = '';
      document.getElementById('frontNote').textContent = '';
      return;
    }
  }
  // Clear a placeholder message, but never the live chart: Plotly keeps state on
  // the element and would "update" a chart whose DOM had been removed.
  if (plot.querySelector('.placeholder')) plot.innerHTML = '';
  const t = theme();
  const years = d => d / 365.25;
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
    customdata: rest.map(p => [p.seq, p.id]), text: rest.map(hover), hoverinfo: 'text',
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
      customdata: pts.map(p => [p.seq, p.id]), text: pts.map(hover), hoverinfo: 'text',
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

  Plotly.react(plot, traces, {
    paper_bgcolor: t.surface, plot_bgcolor: t.surface, font: { color: t.ink2 },
    margin: { l: 70, r: 10, t: 10, b: 50 }, hovermode: 'closest',
    xaxis: { title: 'Time of flight (years)', gridcolor: t.grid, zeroline: false },
    yaxis: { title: `${data.cost_label} (${units})`, gridcolor: t.grid, zeroline: false },
    legend: { orientation: 'h', y: 1.08, x: 0 }, annotations,
  }, { displayModeBar: false, responsive: true });
  if (!plot.dataset.clickBound) {
    plot.on('plotly_click', event => {
      const [seq, id] = event.points[0].customdata || [];
      if (seq) location.href = `sequence.html?model=${encodeURIComponent(fronts.current)}` +
                               `&seq=${encodeURIComponent(seq)}&pt=${encodeURIComponent(id)}`;
    });
    plot.dataset.clickBound = '1';
  }
  fronts.current = model;

  document.getElementById('frontNote').textContent =
    `${data.runs.length} sequences · ${best.length} points on the combined front`;
  const rows = spans.map(s => {
    const cheapest = s.points.reduce((a, b) => (b.cost < a.cost ? b : a));
    const range = s.from === s.to ? `${years(s.from).toFixed(1)} yr`
                                  : `${years(s.from).toFixed(1)}–${years(s.to).toFixed(1)} yr`;
    return `<tr><td><span class="swatch" style="background:${colorOf(s.seq)}"></span>${s.seq}</td>
      <td>${range}</td><td class="num">${fmtCost(cheapest.cost, units)}</td>
      <td><a href="sequence.html?model=${encodeURIComponent(model)}&seq=${encodeURIComponent(s.seq)}` +
      `&pt=${encodeURIComponent(cheapest.id)}">open</a></td></tr>`;
  }).join('');
  table.innerHTML = `<tr><th>sequence</th><th>wins for time of flight</th><th>cheapest there</th><th></th></tr>${rows}`;
}
