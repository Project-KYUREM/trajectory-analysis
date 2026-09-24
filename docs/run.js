// Sequence page: the interactive cloud, one trajectory's detail, 3D view, exports.
// All data is precomputed by publish.py; trajectories are rebuilt in-browser
// from arc states with kepler.js.

const HINT = 'Click a point to inspect. Ringed points are Pareto-optimal (largest rings were ' +
             'locally polished); only detailed points are clickable.';

const state = {
  model: null, seq: null,        // which run
  cloud: null, planets: null,    // loaded payloads
  axis: 'trade',                 // 'trade' (cost vs TOF) or 'porkchop' (launch vs arrival)
  frontOnly: false,
  shown: [],                     // points currently plotted, in trace order
  selected: null,                // id of the inspected point
  clickBound: false,
};

async function main() {
  const params = new URLSearchParams(location.search);
  const index = await loadJSON('data/index.json');
  if (!index.length) throw new Error('no runs published yet');
  state.model = params.get('model') || index[0].model;
  state.seq = params.get('seq') || index[0].sequence;

  const select = document.getElementById('modelSel');
  index.filter(run => run.sequence === state.seq).forEach(run =>
    select.add(new Option(run.model, run.model, run.model === state.model, run.model === state.model)));
  select.onchange = () => { location.search = `?model=${select.value}&seq=${state.seq}`; };

  [state.cloud, state.planets] = await Promise.all([
    loadJSON(`data/${state.model}/${state.seq}/cloud.json`),
    loadJSON('data/planets.json'),
  ]);
  document.getElementById('crumb').textContent =
    `${state.seq} · ${state.cloud.labels.join(' → ')} · ${state.cloud.blurb}`;
  bindControls();
  drawScatter();
}

main().catch(error => {
  document.querySelector('.run-wrap').innerHTML =
    `<p class="placeholder">Could not load this run (${error.message}).</p>`;
});

// --- scatter -----------------------------------------------------------------
function drawScatter() {
  const t = theme();
  const cloud = state.cloud;
  const points = state.shown = state.frontOnly ? cloud.points.filter(p => p.pareto) : cloud.points;
  const trade = state.axis === 'trade';

  const x = points.map(p => trade ? p.tof_days / 365.25 : p.launch_mjd2000);
  const y = points.map(p => trade ? p.cost : p.launch_mjd2000 + p.tof_days);
  const hover = points.map(p => {
    const flags = (p.pareto ? '<br>Pareto-optimal' : '') + (p.polished ? '<br>locally polished' : '') +
                  (p.detail ? '' : '<br>(not detailed)');
    return trade
      ? `${fmtCost(p.cost, cloud.cost_units)}<br>${(p.tof_days / 365.25).toFixed(2)} yr${flags}`
      : `launch ${isoDay(mjd2000ToDate(p.launch_mjd2000))}<br>arrival ` +
        `${isoDay(mjd2000ToDate(p.launch_mjd2000 + p.tof_days))}<br>${fmtCost(p.cost, cloud.cost_units)}${flags}`;
  });
  const selectedIndex = points.findIndex(p => p.id === state.selected);

  const trace = {
    type: 'scattergl', mode: 'markers', x, y, text: hover, hoverinfo: 'text',
    customdata: points.map(p => p.id),
    marker: {
      size: points.map(p => p.polished ? 11 : p.pareto ? 9 : 5),
      color: points.map(p => p.cost), colorscale: 'Blues', reversescale: true, opacity: 0.85,
      colorbar: { title: { text: cloud.cost_units, side: 'right', font: { color: t.ink2 } },
                  thickness: 12, len: 0.9, tickfont: { color: t.ink2 } },
      line: { width: 1.4, color: points.map(p => p.pareto ? t.accent : 'rgba(0,0,0,0)') },
    },
    selectedpoints: selectedIndex >= 0 ? [selectedIndex] : null,
    selected: { marker: { opacity: 1, size: 16 } },
    unselected: { marker: { opacity: selectedIndex >= 0 ? 0.45 : 0.85 } },
  };
  const dateAxis = title => {
    const ticks = yearTicks(x.concat(y));
    return { title, gridcolor: t.grid, zeroline: false, tickvals: ticks.values, ticktext: ticks.labels };
  };
  const layout = {
    paper_bgcolor: t.surface, plot_bgcolor: t.surface, font: { color: t.ink2 },
    margin: { l: 60, r: 10, t: 10, b: 50 }, hovermode: 'closest', clickmode: 'event',
    xaxis: trade ? { title: 'Time of flight (years)', gridcolor: t.grid, zeroline: false }
                 : dateAxis('Launch (year)'),
    yaxis: trade ? { title: `${cloud.cost_label} (${cloud.cost_units})`, gridcolor: t.grid, zeroline: false }
                 : dateAxis('Arrival (year)'),
  };
  Plotly.react('scatter', [trace], layout, { displayModeBar: false, responsive: true });

  if (!state.clickBound) {   // bind once: Plotly.react keeps the div's listeners
    document.getElementById('scatter').on('plotly_click', event => {
      const id = event.points[0].customdata;
      if (state.cloud.detail[id]) showDetail(id);
      else flash('That point was not exported in detail — pick a ringed point or one near it.');
    });
    state.clickBound = true;
  }
}

// Whole-year tick marks for an MJD2000 axis.
function yearTicks(mjds) {
  const y0 = mjd2000ToYear(Math.min(...mjds)), y1 = mjd2000ToYear(Math.max(...mjds));
  const step = Math.max(1, Math.ceil((y1 - y0) / 6));
  const values = [], labels = [];
  for (let year = Math.ceil(y0); year <= y1; year += step) {
    values.push((year - 2000) * 365.25);
    labels.push(String(year));
  }
  return { values, labels };
}

let hintTimer;
function flash(message) {
  const hint = document.getElementById('scatterHint');
  hint.textContent = message;
  hint.style.color = theme().accent;
  clearTimeout(hintTimer);
  hintTimer = setTimeout(() => { hint.style.color = ''; hint.textContent = HINT; }, 4000);
}

// --- detail -------------------------------------------------------------------
function showDetail(id) {
  state.selected = id;
  const d = state.cloud.detail[id];
  const cloud = state.cloud;
  const head = document.getElementById('detailHead');
  const body = document.getElementById('detailBody');
  if (d.error) {
    head.innerHTML = `<p class="placeholder">decode error: ${d.error}</p>`;
    body.innerHTML = '';
    return;
  }
  const arrival = d.arrival, launch = d.launch;
  head.innerHTML = `
    <h2>${cloud.sequence} <span class="muted">· ${cloud.model}</span></h2>
    <div class="muted">${isoDay(launch.date)} → ${isoDay(arrival.date)}</div>
    <div class="kpis">
      ${kpi(fmtCost(d.cost, cloud.cost_units), cloud.cost_label)}
      ${kpi(`${d.totals.tof_years.toFixed(1)} yr`, 'time of flight')}
      ${kpi(`${(arrival.vinf_ms / 1000).toFixed(2)} km/s`, `arrival speed at ${arrival.body} (v∞)`)}
      ${kpi(`${(arrival.entry_speed_ms / 1000).toFixed(1)} km/s`, 'atmospheric entry speed')}
      ${kpi(launch.c3_km2s2.toFixed(1), 'launch energy C3 (km²/s²)')}
    </div>`;
  const legRows = d.legs.map(leg => `<tr><td>${leg.from} → ${leg.to}</td>
      <td class="num">${leg.tof_days.toFixed(0)} d</td>
      <td class="num">${leg.dsm_ms != null ? leg.dsm_ms.toFixed(0) + ' m/s' : '—'}</td>
      <td>${isoDay(leg.arr_date)}</td></tr>`).join('');
  const captureRow = arrival.capture_dv_ms != null
    ? `<tr><th>${arrival.body} capture burn</th><td class="num" colspan="3">${arrival.capture_dv_ms.toFixed(0)} m/s</td></tr>` : '';
  // Ballistic model: the assist cannot always bend the path enough; the leftover is a burn at the flyby.
  const mismatchRows = d.flybys.filter(f => f.mismatch_dv_ms != null).map(f =>
    `<tr><th>burn at ${f.body} flyby (assist shortfall)</th><td class="num" colspan="3">${f.mismatch_dv_ms.toFixed(0)} m/s</td></tr>`).join('');
  body.innerHTML = `
    <table class="mission">
      <tr><th>leg</th><th>time of flight</th><th>deep-space maneuver</th><th>arrival date</th></tr>
      ${legRows}
      ${mismatchRows}
      <tr><th>launch asymptote (right ascension / declination)</th>
          <td class="num" colspan="3">${launch.rla_deg.toFixed(1)}° / ${launch.dla_deg.toFixed(1)}°</td></tr>
      ${captureRow}
    </table>
    <div class="exports">
      <button id="expCsv">state vectors (CSV)</button>
      <button id="expGmat">GMAT seed</button>
      <button id="expJson">full data (JSON)</button>
    </div>
    <p class="muted">The CSV and GMAT seed give heliocentric arc state vectors in two reference frames —
    ecliptic J2000 and Earth-equatorial J2000 (EME2000) — to paste into GMAT or STK and re-propagate.</p>`;
  const stem = `${cloud.model}_${cloud.sequence}_${id}`;
  document.getElementById('expCsv').onclick = () => download(`${stem}_states.csv`, statesCsv(d));
  document.getElementById('expGmat').onclick = () => download(`${stem}_gmat_seed.txt`, gmatSeed(d));
  document.getElementById('expJson').onclick = () => download(`${stem}.json`, JSON.stringify(d, null, 2));

  document.getElementById('view3d').hidden = false;
  draw3d(d);
  const index = state.shown.findIndex(p => p.id === id);   // mark without redrawing
  Plotly.restyle('scatter', { selectedpoints: [index >= 0 ? [index] : null],
                              'unselected.marker.opacity': 0.45 });
}

const kpi = (value, label) => `<div class="kpi"><div class="v">${value}</div><div class="l">${label}</div></div>`;

function draw3d(d) {
  const t = theme();
  const toAU = points => ({ x: points.map(p => p[0] / AU), y: points.map(p => p[1] / AU), z: points.map(p => p[2] / AU) });
  const traces = [...new Set(state.cloud.labels)].map(label => ({
    type: 'scatter3d', mode: 'lines', ...toAU(state.planets[label].xyz),
    line: { color: t.grid, width: 2 }, name: label, hoverinfo: 'name',
  }));
  d.arcs.forEach(arc => traces.push({
    type: 'scatter3d', mode: 'lines', ...toAU(sampleArc(arc, 80)),
    line: { color: t.accent, width: 4 }, showlegend: false, hoverinfo: 'skip',
  }));
  traces.push({ type: 'scatter3d', mode: 'markers', x: [0], y: [0], z: [0],
                marker: { color: '#eda100', size: 4 }, name: 'Sun', hoverinfo: 'name' });
  const axis = title => ({ title, gridcolor: t.grid, color: t.ink2, showbackground: false });
  Plotly.react('view3d', traces, {
    paper_bgcolor: t.surface, font: { color: t.ink2 }, showlegend: false,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: { xaxis: axis('X (AU)'), yaxis: axis('Y (AU)'), zaxis: axis('Z (AU)'),
             aspectmode: 'data', bgcolor: t.surface },
  }, { displayModeBar: false, responsive: true });
}

// --- exports (the site owns these; states arrive in ECLIPJ2000 and are rotated here) ---
function statesCsv(d) {
  const R = state.cloud.frames.eclip_to_eme;
  const header = ['arc', 'epoch_mjd2000', 'date', 'dt_days', 'mu_m3s2',
    'x_eclip_m', 'y_eclip_m', 'z_eclip_m', 'vx_eclip_ms', 'vy_eclip_ms', 'vz_eclip_ms',
    'x_eme_m', 'y_eme_m', 'z_eme_m', 'vx_eme_ms', 'vy_eme_ms', 'vz_eme_ms'];
  const rows = d.arcs.map((arc, i) => [i, arc.epoch_mjd2000, arc.date, arc.dt_days, arc.mu,
    ...arc.r_eclip_m, ...arc.v_eclip_ms, ...rotate(R, arc.r_eclip_m), ...rotate(R, arc.v_eclip_ms)].join(','));
  return [header.join(','), ...rows].join('\n') + '\n';
}

function gmatSeed(d) {
  const R = state.cloud.frames.eclip_to_eme;
  const dep = d.arcs[0], launch = d.launch, arrival = d.arrival;
  const [X, Y, Z] = rotate(R, dep.r_eclip_m), [VX, VY, VZ] = rotate(R, dep.v_eclip_ms);
  const lines = [
    `% GMAT seed -- ${state.cloud.model} ${state.cloud.sequence}`,
    `% cost ${d.cost.toFixed(1)} ${state.cloud.cost_units}   TOF ${d.totals.tof_years.toFixed(2)} yr`,
    `% Launch ${launch.date}   C3 ${launch.c3_km2s2.toFixed(4)} km^2/s^2`,
    `%   launch asymptote RA ${launch.rla_deg.toFixed(3)} deg   dec ${launch.dla_deg.toFixed(3)} deg`,
    '% Heliocentric departure state, EME2000 (J2000), metres and m/s:',
    `%   X  = ${X.toFixed(3)}`, `%   Y  = ${Y.toFixed(3)}`, `%   Z  = ${Z.toFixed(3)}`,
    `%   VX = ${VX.toFixed(6)}`, `%   VY = ${VY.toFixed(6)}`, `%   VZ = ${VZ.toFixed(6)}`,
    '% Encounter sequence and epochs:',
  ];
  d.flybys.forEach(f => lines.push(`%   ${f.body.padEnd(8)} ${f.date}` +
    (f.rp_m ? `   rp = ${(f.rp_m / 1e3).toFixed(1)} km` : '')));
  lines.push(`%   ${arrival.body.padEnd(8)} ${arrival.date}   vinf = ${arrival.vinf_ms.toFixed(1)} m/s`);
  lines.push('% Maneuvers to model as finite / electric-propulsion burns:');
  lines.push(`%   launch            dv = ${launch.vinf_ms.toFixed(1)} m/s (departure vinf)`);
  d.legs.forEach((leg, i) => { if (leg.dsm_ms)
    lines.push(`%   dsm_leg${i + 1}          dv = ${leg.dsm_ms.toFixed(1)} m/s (${leg.from} -> ${leg.to})`); });
  d.flybys.forEach(f => { if (f.mismatch_dv_ms)
    lines.push(`%   flyby_${f.body.toLowerCase().padEnd(11)} dv = ${f.mismatch_dv_ms.toFixed(1)} m/s (assist shortfall at ${f.body})`); });
  if (arrival.capture_dv_ms) lines.push(`%   capture           dv = ${arrival.capture_dv_ms.toFixed(1)} m/s at ${arrival.body}`);
  return lines.join('\n') + '\n';
}

// --- controls -----------------------------------------------------------------
function bindControls() {
  const setAxis = axis => {
    state.axis = axis;
    document.getElementById('axTrade').classList.toggle('on', axis === 'trade');
    document.getElementById('axPork').classList.toggle('on', axis === 'porkchop');
    drawScatter();
  };
  document.getElementById('axTrade').onclick = () => setAxis('trade');
  document.getElementById('axPork').onclick = () => setAxis('porkchop');
  document.getElementById('frontOnly').onclick = event => {
    state.frontOnly = !state.frontOnly;
    event.target.classList.toggle('on', state.frontOnly);
    drawScatter();
  };
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    drawScatter();
    if (state.selected) draw3d(state.cloud.detail[state.selected]);
  });
}
