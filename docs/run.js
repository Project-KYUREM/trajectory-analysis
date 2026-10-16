// Sequence page: the interactive cloud, one trajectory's detail, 3D view, exports.
// All data is precomputed by publish.py; trajectories are rebuilt in-browser
// from arc states with kepler.js.

const HINT = 'Each dot is one trajectory. Ringed: Pareto-optimal (least propellant for its trip ' +
             'time); large ring: locally refined. Click a ringed dot for its details.';

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
    loadCloud(`data/${state.model}/${state.seq}`),
    loadJSON('data/planets.json'),
  ]);
  document.getElementById('crumb').textContent =
    `${state.seq} · ${state.cloud.labels.join(' → ')} · ${state.cloud.blurb}`;
  bindControls();
  drawScatter();
  // Deep link from the "Best by trip time" tab: ?pt=<point id> opens that trajectory.
  const pt = params.get('pt');
  if (pt && state.cloud.detail[pt]) showDetail(pt);
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
    const flags = (p.pareto ? '<br>Pareto-optimal' : '') + (p.polished ? '<br>locally refined' : '') +
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
    xaxis: trade ? { title: 'Time of flight, launch → in orbit (years)', gridcolor: t.grid, zeroline: false }
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
// Propellant budget of a low-thrust trajectory published before describe()
// produced a ``budget`` section (same rows trajectory._lowthrust_budget builds).
function budgetFromLowthrust(d) {
  const lt = d.lowthrust;
  let mass = lt.wet_mass_kg;
  const events = d.legs.map((leg, i) => {
    const end = lt.leg_end_mass_kg[i];
    const e = { label: `thrust, ${leg.from} → ${leg.to} (${leg.tof_days.toFixed(0)} d)`, date: leg.dep_date,
                dv_ms: leg.dsm_ms, propellant_kg: mass - end, mass_after_kg: end, phase: 'to_destination' };
    mass = end;
    return e;
  });
  const cap = lt.capture;
  events.push({ label: `${d.arrival.body} capture spiral (${cap.days.toFixed(0)} d)`, date: d.arrival.date,
                dv_ms: cap.dv_ms, propellant_kg: mass - lt.final_mass_kg, mass_after_kg: lt.final_mass_kg,
                days: cap.days, phase: 'insertion' });
  const total = phase => events.filter(e => e.phase === phase).reduce(
    (acc, e) => ({ dv_ms: acc.dv_ms + e.dv_ms, propellant_kg: acc.propellant_kg + e.propellant_kg }),
    { dv_ms: 0, propellant_kg: 0 });
  return { wet_mass_kg: lt.wet_mass_kg, final_mass_kg: lt.final_mass_kg,
           propellant_kg: lt.wet_mass_kg - lt.final_mass_kg, isp_s: lt.spacecraft.isp_s, events,
           to_destination: total('to_destination'), insertion: total('insertion') };
}

function budgetTable(b, body, lowThrust) {
  const kg = v => `${fmtInt(v)} kg`;
  const row = e => `<tr><td>${e.label}</td><td class="date">${isoDay(e.date)}</td><td class="num">${fmtInt(e.dv_ms)} m/s</td>
    <td class="num">${kg(e.propellant_kg)}</td><td class="num">${kg(e.mass_after_kg)}</td></tr>`;
  const sub = (label, part, cls = 'sub') => `<tr class="${cls}"><th colspan="2">${label}</th>
    <td class="num">${fmtInt(part.dv_ms)} m/s</td><td class="num">${kg(part.propellant_kg)}</td><td></td></tr>`;
  const phase = name => b.events.filter(e => e.phase === name).map(row).join('');
  return `<table class="mission">
    <tr><th>${lowThrust ? 'thrust phase' : 'burn'} (Isp ${fmtInt(b.isp_s)} s)</th><th>${lowThrust ? 'starts' : 'date'}</th><th class="num">Δv</th>
        <th class="num">propellant</th><th class="num">mass after</th></tr>
    <tr><td>launch: launcher delivers to Earth escape (C3 = 0)</td><td></td><td class="num">—</td>
        <td class="num">—</td><td class="num">${kg(b.wet_mass_kg)}</td></tr>
    ${phase('to_destination')}
    ${sub(`subtotal: getting to ${body}`, b.to_destination)}
    ${phase('insertion')}
    ${sub(`subtotal: ${body} orbit insertion`, b.insertion)}
    <tr class="sub total"><th colspan="2">total</th>
      <td class="num">${fmtInt(b.to_destination.dv_ms + b.insertion.dv_ms)} m/s</td>
      <td class="num">${kg(b.propellant_kg)}</td><td class="num">${kg(b.final_mass_kg)}</td></tr>
  </table>`;
}

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
  const lt = d.lowthrust;   // low-thrust (Sims-Flanagan) models only
  const b = d.budget || (lt ? budgetFromLowthrust(d) : null);
  const orbit = arrival.orbit_periapsis_km
    ? `${fmtInt(arrival.orbit_periapsis_km)} × ${fmtInt(arrival.orbit_apoapsis_km)} km orbit` : 'orbit';
  head.innerHTML = `
    <h2>${cloud.sequence} <span class="muted">· ${cloud.model}</span></h2>
    <div class="muted">launch ${isoDay(launch.date)} · arrive ${isoDay(arrival.date)}` +
    `${arrival.orbit_date && isoDay(arrival.orbit_date) !== isoDay(arrival.date) ? ` · in orbit ${isoDay(arrival.orbit_date)}` : ''}</div>
    <div class="kpis">
      ${b ? kpi(`${fmtInt(b.propellant_kg)} kg`, `propellant, of ${fmtInt(b.wet_mass_kg)} kg`)
          : kpi(fmtCost(d.cost, cloud.cost_units), cloud.cost_label)}
      ${b ? kpi(`${fmtInt(b.to_destination.propellant_kg)} + ${fmtInt(b.insertion.propellant_kg)} kg`,
                `to ${arrival.body} + orbit insertion`) : ''}
      ${b ? kpi(`${fmtInt(b.final_mass_kg)} kg`, `in ${orbit}`) : ''}
      ${kpi(`${d.totals.tof_years.toFixed(1)} yr`, 'launch → in orbit')}
      ${kpi(fmtKms(arrival.vinf_ms), `arrival v∞ at ${arrival.body}`)}
    </div>`;
  const legRows = d.legs.map((leg, i) => {
    const flyby = d.flybys[i];   // the flyby at this leg's end, if it is not the last leg
    const rp = flyby && flyby.rp_m ? `${fmtInt(flyby.rp_m / 1e3)} km (${flyby.rp_radii.toFixed(2)} R)` : '—';
    return `<tr><td>${leg.from} → ${leg.to}</td><td class="num">${leg.tof_days.toFixed(0)} d</td>
      <td>${isoDay(leg.dep_date)} → ${isoDay(leg.arr_date)}</td><td class="num">${rp}</td></tr>`;
  }).join('');
  const departure = launch.vinf_ms > 1
    ? `${fmtKms(launch.vinf_ms)} (C3 ${launch.c3_km2s2.toFixed(1)} km²/s²), asymptote RA ${launch.rla_deg.toFixed(1)}° / dec ${launch.dla_deg.toFixed(1)}°`
    : '0 (leaves at the launcher\'s C3 = 0)';
  body.innerHTML = `
    ${b ? budgetTable(b, arrival.body, !!lt) : ''}
    <table class="mission">
      <tr><th>leg</th><th class="num">duration</th><th>dates</th><th class="num">flyby periapsis at end</th></tr>
      ${legRows}
    </table>
    <table class="mission">
      <tr><th>departure v∞ (after any burn)</th><td class="num">${departure}</td></tr>
      <tr><th>arrival v∞ at ${arrival.body}</th><td class="num">${fmtKms(arrival.vinf_ms)}</td></tr>
      <tr><th>probe entry speed at ${fmtInt(arrival.entry_altitude_km)} km altitude</th>
          <td class="num">${(arrival.entry_speed_ms / 1000).toFixed(1)} km/s</td></tr>
    </table>
    ${lt ? `<p class="muted">3D path coloured by throttle, coast to full ${lt.spacecraft.thrust_n} N; hover for
      date and level. The capture spiral is planetocentric and not drawn.</p>` : ''}
    <div class="exports">
      <button id="expCsv">state vectors (CSV)</button>
      <button id="expGmat">GMAT seed</button>
      <button id="expJson">full data (JSON)</button>
    </div>
    <p class="muted">CSV and GMAT seed: heliocentric arc states in ecliptic J2000 and EME2000, for GMAT/STK.</p>`;
  const stem = `${cloud.model}_${cloud.sequence}_${id}`;
  document.getElementById('expCsv').onclick = () => download(`${stem}_states.csv`, statesCsv(d));
  document.getElementById('expGmat').onclick = () => download(`${stem}_gmat_seed.txt`, gmatSeed(d, b));
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
  const throttleAt = d.lowthrust ? throttleLookup(d) : null;
  d.arcs.forEach((arc, i) => {
    const n = 80;
    const trace = { type: 'scatter3d', mode: 'lines', ...toAU(sampleArc(arc, n)), showlegend: false };
    if (!throttleAt) {
      traces.push({ ...trace, line: { color: t.accent, width: 4 }, hoverinfo: 'skip' });
      return;
    }
    // Low-thrust: colour each sample by the engine throttle of the
    // Sims-Flanagan segment it falls in (one blue ramp, pale = coasting).
    const level = Array.from({ length: n + 1 }, (_, k) =>
      throttleAt(arc.epoch_mjd2000 + arc.dt_days * k / n));
    traces.push({
      ...trace,
      line: {
        width: 5, color: level, cmin: 0, cmax: 1,
        colorscale: [[0, t.thrust[0]], [0.5, t.thrust[1]], [1, t.thrust[2]]],
        showscale: i === 0,
        colorbar: { title: { text: 'thrust', side: 'right', font: { color: t.ink2 } },
                    tickvals: [0, 0.5, 1], ticktext: ['coast', '50 %', 'full'],
                    thickness: 12, len: 0.6, tickfont: { color: t.ink2 } },
      },
      text: level.map((v, k) => `${isoDay(mjd2000ToDate(arc.epoch_mjd2000 + arc.dt_days * k / n))}` +
                                `<br>thrust ${(100 * v).toFixed(0)} %`),
      hoverinfo: 'text',
    });
  });
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

// Throttle (fraction of full thrust, 0..1) at an MJD2000 epoch of a low-thrust
// trajectory: Sims-Flanagan segment j of a leg spans its impulse epoch +- dt/2.
function throttleLookup(d) {
  const segments = [];
  d.legs.forEach(leg => {
    const impulses = leg.impulses || [];
    const half = leg.tof_days / impulses.length / 2;
    impulses.forEach(p => segments.push([p.epoch_mjd2000 - half, p.epoch_mjd2000 + half,
                                         Math.min(p.throttle ?? 0, 1)]));
  });
  return epoch => {
    const seg = segments.find(([lo, hi]) => epoch >= lo && epoch <= hi);
    return seg ? seg[2] : 0;
  };
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

function gmatSeed(d, budget) {
  const R = state.cloud.frames.eclip_to_eme;
  const dep = d.arcs[0], launch = d.launch, arrival = d.arrival;
  const [X, Y, Z] = rotate(R, dep.r_eclip_m), [VX, VY, VZ] = rotate(R, dep.v_eclip_ms);
  const lines = [
    `% GMAT seed -- ${state.cloud.model} ${state.cloud.sequence}`,
    `% cost ${d.cost.toFixed(1)} ${state.cloud.cost_units}   TOF ${d.totals.tof_years.toFixed(2)} yr`,
    `% Launch ${launch.date}   C3 ${launch.c3_km2s2.toFixed(4)} km^2/s^2`,
    launch.rla_deg == null ? '%   launch asymptote: none (C3 = 0, Earth escape)'
      : `%   launch asymptote RA ${launch.rla_deg.toFixed(3)} deg   dec ${launch.dla_deg.toFixed(3)} deg`,
    '% Heliocentric departure state, EME2000 (J2000), metres and m/s:',
    `%   X  = ${X.toFixed(3)}`, `%   Y  = ${Y.toFixed(3)}`, `%   Z  = ${Z.toFixed(3)}`,
    `%   VX = ${VX.toFixed(6)}`, `%   VY = ${VY.toFixed(6)}`, `%   VZ = ${VZ.toFixed(6)}`,
    '% Encounter sequence and epochs:',
  ];
  d.flybys.forEach(f => lines.push(`%   ${f.body.padEnd(8)} ${f.date}` +
    (f.rp_m ? `   rp = ${(f.rp_m / 1e3).toFixed(1)} km` : '')));
  lines.push(`%   ${arrival.body.padEnd(8)} ${arrival.date}   vinf = ${arrival.vinf_ms.toFixed(1)} m/s`);
  if (budget) {
    lines.push(`% Burns (Isp ${budget.isp_s} s, ${budget.wet_mass_kg} kg at Earth escape):`);
    budget.events.forEach(e => lines.push(`%   ${e.date.slice(0, 10)}  dv ${e.dv_ms.toFixed(1).padStart(8)} m/s  ` +
      `prop ${e.propellant_kg.toFixed(1).padStart(8)} kg  mass after ${e.mass_after_kg.toFixed(1)} kg  ${e.label}`));
  }
  if (d.lowthrust) {
    const sc = d.lowthrust.spacecraft;
    lines.push(`% Low-thrust: ${sc.engine} ${sc.thrust_n} N @ ${sc.isp_s} s, wet ${sc.wet_mass_kg} kg; ` +
               `each arc ends at a Sims-Flanagan impulse (full list in the JSON export, legs[].impulses)`);
  }
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
