const AU = 1.495978707e11;
const P = new URLSearchParams(location.search);
let MODEL = P.get('model') || 'capture';
const SEQ = P.get('seq') || 'EVEEJN';
let CLOUD = null, AXIS = 'trade', FRONT_ONLY = false, SELECTED = null;

const isDark = () => matchMedia('(prefers-color-scheme: dark)').matches
  && document.documentElement.dataset.theme !== 'light';
const ink = () => isDark() ? '#f3f2ef' : '#17161c';
const ink2 = () => isDark() ? '#b7b6b0' : '#52514e';
const grid = () => isDark() ? '#34333a' : '#e4e3df';
const surface = () => isDark() ? '#16161a' : '#fcfcfb';
const ACCENT = '#eb6834';
const fmtCost = (v, u) => u === 'm/s' ? `${(v/1000).toFixed(3)} km/s` : `${v.toFixed(1)} ${u}`;
const fmtDate = s => (s || '').slice(0, 10);

// --- model switcher across the same sequence -------------------------------
fetch('data/index.json').then(r => r.json()).then(idx => {
  const models = idx.filter(r => r.sequence === SEQ).map(r => r.model);
  const sel = document.getElementById('modelSel');
  models.forEach(m => {
    const o = document.createElement('option'); o.value = o.textContent = m;
    if (m === MODEL) o.selected = true; sel.appendChild(o);
  });
  sel.onchange = () => { location.search = `?model=${sel.value}&seq=${SEQ}`; };
});

function load() {
  fetch(`data/${MODEL}/${SEQ}/cloud.json`).then(r => r.json()).then(c => {
    CLOUD = c;
    document.getElementById('crumb').textContent =
      `${SEQ} · ${c.labels.join(' → ')} · ${c.blurb}`;
    drawScatter();
  });
}

function drawScatter() {
  const pts = FRONT_ONLY ? CLOUD.points.filter(p => p.pareto) : CLOUD.points;
  const clickable = new Set(Object.keys(CLOUD.detail));
  const x = pts.map(p => AXIS === 'trade' ? p.tof_years : p.launch_mjd2000);
  const y = pts.map(p => AXIS === 'trade' ? p.cost : p.arrival_mjd2000);
  const color = pts.map(p => AXIS === 'trade' ? p.cost : p.cost);
  const size = pts.map(p => p.pareto ? 9 : 5);
  const line = pts.map(p => p.pareto ? ACCENT : 'rgba(0,0,0,0)');

  const mj2date = m => { const d = new Date(Date.UTC(2000,0,1)); d.setUTCDate(d.getUTCDate()+m); return d.toISOString().slice(0,10); };
  const hover = pts.map(p => AXIS === 'trade'
    ? `${fmtCost(p.cost, CLOUD.cost_units)}<br>${p.tof_years.toFixed(2)} yr${p.pareto?'<br>Pareto':''}${clickable.has(p.id)?'':'<br>(not detailed)'}`
    : `launch ${mj2date(p.launch_mjd2000)}<br>arrival ${mj2date(p.arrival_mjd2000)}<br>${fmtCost(p.cost,CLOUD.cost_units)}`);

  const trace = {
    x, y, customdata: pts.map(p => p.id), text: hover, hoverinfo: 'text',
    mode: 'markers', type: 'scattergl',
    marker: {
      size, color, colorscale: 'Blues', reversescale: true,
      colorbar: { title: { text: CLOUD.cost_units, side: 'right' }, thickness: 12, len: 0.9,
        tickfont: { color: ink2() }, titlefont: { color: ink2() } },
      line: { width: 1.4, color: line },
      opacity: 0.85,
    },
  };
  const tickvals = AXIS === 'porkchop' ? niceDateTicks(x.concat(y)) : undefined;
  const layout = {
    paper_bgcolor: surface(), plot_bgcolor: surface(), font: { color: ink2() },
    margin: { l: 60, r: 10, t: 10, b: 50 }, hovermode: 'closest',
    xaxis: { title: AXIS === 'trade' ? 'Time of flight (years)' : 'Launch (year)',
      gridcolor: grid(), zeroline: false,
      tickvals: AXIS==='porkchop'?tickvals.v:undefined, ticktext: AXIS==='porkchop'?tickvals.t:undefined },
    yaxis: { title: AXIS === 'trade' ? `${CLOUD.cost_label} (${CLOUD.cost_units})` : 'Arrival (year)',
      gridcolor: grid(), zeroline: false,
      tickvals: AXIS==='porkchop'?tickvals.v:undefined, ticktext: AXIS==='porkchop'?tickvals.t:undefined },
  };
  Plotly.react('scatter', [trace], layout, { displayModeBar: false, responsive: true });
  document.getElementById('scatter').on('plotly_click', ev => {
    const id = ev.points[0].customdata;
    if (CLOUD.detail[id]) showDetail(id);
    else flash('That point was not exported in detail — pick a ringed (Pareto) point or one near it.');
  });
}

function niceDateTicks(mjds) {
  const lo = Math.min(...mjds), hi = Math.max(...mjds);
  const y0 = 2000 + lo/365.25, y1 = 2000 + hi/365.25;
  const step = Math.max(1, Math.ceil((y1 - y0) / 6));
  const v = [], t = [];
  for (let y = Math.ceil(y0); y <= y1; y += step) {
    v.push((y - 2000) * 365.25); t.push(String(y));
  }
  return { v, t };
}

let hintTimer;
function flash(msg) {
  const h = document.getElementById('scatterHint');
  h.textContent = msg; h.style.color = ACCENT;
  clearTimeout(hintTimer);
  hintTimer = setTimeout(() => { h.style.color = ''; h.textContent =
    'Click a point to inspect. Ringed points are Pareto-optimal; only detailed points are clickable.'; }, 4000);
}

function showDetail(id) {
  SELECTED = id;
  const d = CLOUD.detail[id];
  if (d.error) { document.getElementById('detail').innerHTML = `<p class="placeholder">decode error: ${d.error}</p>`; return; }
  const arr = d.arrival, L = d.launch;
  const capRow = arr.capture_dv_ms != null
    ? `<tr><th>Neptune capture burn</th><td class="num">${arr.capture_dv_ms.toFixed(0)} m/s</td></tr>` : '';
  let legRows = d.legs.map((lg, i) => `<tr><td>${lg.from} → ${lg.to}</td>
      <td class="num">${lg.tof_days.toFixed(0)} d</td>
      <td class="num">${lg.dsm_ms != null ? lg.dsm_ms.toFixed(0) : '—'}</td>
      <td>${fmtDate(lg.arr_date)}</td></tr>`).join('');
  document.getElementById('detail').innerHTML = `
    <h2>${d.sequence} <span class="muted">· ${d.model}</span></h2>
    <div class="muted">${fmtDate(L.date)} → ${fmtDate(arr.date)}</div>
    <div class="kpis">
      <div class="kpi"><div class="v">${fmtCost(d.cost, d.cost_units)}</div><div class="l">${d.cost_label}</div></div>
      <div class="kpi"><div class="v">${d.totals.tof_years.toFixed(1)} yr</div><div class="l">time of flight</div></div>
      <div class="kpi"><div class="v">${(arr.vinf_ms/1000).toFixed(2)} km/s</div><div class="l">v∞ at Neptune</div></div>
      <div class="kpi"><div class="v">${L.c3_km2s2.toFixed(1)}</div><div class="l">launch C3 km²/s²</div></div>
    </div>
    <div id="view3d"></div>
    <table class="mission">
      <tr><th>leg</th><th>TOF</th><th>DSM m/s</th><th>arrive</th></tr>
      ${legRows}
      <tr><th>Launch RLA / DLA</th><td class="num" colspan="3">${L.rla_deg.toFixed(1)}° / ${L.dla_deg.toFixed(1)}°</td></tr>
      ${capRow}
    </table>
    <div class="exports">
      <button id="expCsv">states CSV</button>
      <button id="expGmat">GMAT seed</button>
      <button id="expJson">full JSON</button>
    </div>
    <p class="muted">CSV/GMAT give heliocentric arc states in ECLIPJ2000 <em>and</em> J2000 (EME2000) — paste into GMAT/STK and re-propagate.</p>`;
  draw3d(d);
  document.getElementById('expCsv').onclick = () => dl(`${d.model}_${d.sequence}_${id}_states.csv`, statesCsv(d));
  document.getElementById('expGmat').onclick = () => dl(`${d.model}_${d.sequence}_${id}_gmat_seed.txt`, gmatSeed(d));
  document.getElementById('expJson').onclick = () => dl(`${d.model}_${d.sequence}_${id}.json`, JSON.stringify(d, null, 2));
  drawScatter();  // redraw to mark selection (kept simple)
}

function draw3d(d) {
  const traces = [];
  // planet orbits
  CLOUD.planets.forEach(pl => {
    traces.push({ type: 'scatter3d', mode: 'lines',
      x: pl.xy.map(p => p[0]/AU), y: pl.xy.map(p => p[1]/AU), z: pl.xy.map(p => p[2]/AU),
      line: { color: grid(), width: 2 }, name: pl.label, hoverinfo: 'name' });
  });
  // trajectory arcs, propagated in-browser
  d.arcs.forEach((arc, i) => {
    const pts = sampleArc(arc, 80);
    traces.push({ type: 'scatter3d', mode: 'lines',
      x: pts.map(p => p[0]/AU), y: pts.map(p => p[1]/AU), z: pts.map(p => p[2]/AU),
      line: { color: ACCENT, width: 4 }, showlegend: false, hoverinfo: 'skip' });
  });
  // sun
  traces.push({ type: 'scatter3d', mode: 'markers', x: [0], y: [0], z: [0],
    marker: { color: '#eda100', size: 4 }, name: 'Sun', hoverinfo: 'name' });
  const layout = {
    paper_bgcolor: surface(), font: { color: ink2() }, showlegend: false,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: { xaxis: ax('X (AU)'), yaxis: ax('Y (AU)'), zaxis: ax('Z (AU)'),
      aspectmode: 'data', bgcolor: surface() },
  };
  Plotly.react('view3d', traces, layout, { displayModeBar: false, responsive: true });
}
const ax = t => ({ title: t, gridcolor: grid(), color: ink2(), backgroundcolor: surface(), showbackground: false });

// --- client-side exports ---------------------------------------------------
function statesCsv(d) {
  const H = ['arc','epoch_mjd2000','date','dt_days','mu_m3s2',
    'x_eclip_m','y_eclip_m','z_eclip_m','vx_eclip_ms','vy_eclip_ms','vz_eclip_ms',
    'x_eme_m','y_eme_m','z_eme_m','vx_eme_ms','vy_eme_ms','vz_eme_ms'];
  const rows = d.arcs.map((a, i) => [i, a.epoch_mjd2000, a.date, a.dt_days, a.mu,
    ...a.r_eclip_m, ...a.v_eclip_ms, ...a.r_eme_m, ...a.v_eme_ms].join(','));
  return H.join(',') + '\n' + rows.join('\n') + '\n';
}
function gmatSeed(d) {
  const dep = d.arcs[0], L = d.launch, A = d.arrival;
  let s = [`% GMAT seed -- ${d.model} ${d.sequence}`,
    `% ${fmtCost(d.cost, d.cost_units)}  TOF ${d.totals.tof_years.toFixed(2)} yr`,
    `% Launch ${d.launch.date}  C3 ${L.c3_km2s2.toFixed(4)} km^2/s^2  RLA ${L.rla_deg.toFixed(3)} DLA ${L.dla_deg.toFixed(3)}`,
    `% Heliocentric departure state, EME2000 (m, m/s):`,
    `%   X=${dep.r_eme_m[0].toFixed(3)}  Y=${dep.r_eme_m[1].toFixed(3)}  Z=${dep.r_eme_m[2].toFixed(3)}`,
    `%   VX=${dep.v_eme_ms[0].toFixed(6)}  VY=${dep.v_eme_ms[1].toFixed(6)}  VZ=${dep.v_eme_ms[2].toFixed(6)}`,
    `% Encounters:`];
  d.flybys.forEach(f => s.push(`%   ${f.body}  ${f.date}` + (f.rp_m ? `  rp=${(f.rp_m/1e3).toFixed(1)} km` : '')));
  s.push(`%   Neptune  ${A.date}  vinf=${A.vinf_ms.toFixed(1)} m/s`);
  s.push(`% Maneuvers to model as finite/EP burns:`);
  d.legs.forEach((lg, i) => { if (lg.dsm_ms) s.push(`%   dsm_leg${i+1}  ${lg.dsm_ms.toFixed(1)} m/s  (${lg.from}->${lg.to})`); });
  if (A.capture_dv_ms) s.push(`%   neptune_capture  ${A.capture_dv_ms.toFixed(1)} m/s`);
  return s.join('\n') + '\n';
}
function dl(name, text) {
  const b = new Blob([text], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// --- controls --------------------------------------------------------------
document.getElementById('axTrade').onclick = () => setAxis('trade');
document.getElementById('axPork').onclick = () => setAxis('porkchop');
function setAxis(a) {
  AXIS = a;
  document.getElementById('axTrade').classList.toggle('on', a === 'trade');
  document.getElementById('axPork').classList.toggle('on', a === 'porkchop');
  drawScatter();
}
document.getElementById('frontOnly').onclick = e => {
  FRONT_ONLY = !FRONT_ONLY; e.target.classList.toggle('on', FRONT_ONLY); drawScatter();
};
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  drawScatter(); if (SELECTED) draw3d(CLOUD.detail[SELECTED]);
});
load();
