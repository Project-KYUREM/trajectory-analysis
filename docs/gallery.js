// Index page: one card per published (model, sequence) run, grouped by sequence,
// with one column per model in a fixed order so models line up across rows.

// Known models first, in this order; anything else follows alphabetically.
const MODEL_ORDER = ['chem', 'dfd', 'dfd-1mw', 'nep'];
// Models that priced a different mission (free launch C3, other end orbit);
// hidden until publish.py drops them from index.json.
const RETIRED = ['capture', 'ballistic'];
const modelRank = m => (MODEL_ORDER.includes(m) ? MODEL_ORDER.indexOf(m) : MODEL_ORDER.length);
const liveRuns = runs => runs.filter(run => !RETIRED.includes(run.model));

// "4,461 kg in orbit" when the run's cost is propellant out of a known launch mass.
const inOrbit = (run, cost) => (run.wet_mass_kg && run.cost_units === 'kg'
  ? ` → ${fmtInt(run.wet_mass_kg - cost)} kg in orbit` : '');

loadJSON('data/index.json').then(all => {
  const runs = liveRuns(all);
  const gallery = document.getElementById('gallery');
  if (!runs.length) {
    gallery.innerHTML = '<p class="placeholder">No runs published yet. Run batch.py, then publish.py.</p>';
    return;
  }
  const bySequence = {};
  runs.forEach(run => (bySequence[run.sequence] = bySequence[run.sequence] || []).push(run));
  // Simpler sequences (fewer flybys) first, then alphabetical.
  const order = Object.keys(bySequence).sort((a, b) => a.length - b.length || a.localeCompare(b));

  const models = [...new Set(runs.map(run => run.model))]
    .sort((a, b) => modelRank(a) - modelRank(b) || a.localeCompare(b));
  gallery.style.setProperty('--cols', models.length);

  gallery.innerHTML = '';
  const head = document.createElement('div');
  head.className = 'grid colhead';
  models.forEach(model => {
    const run = runs.find(r => r.model === model);
    const cell = document.createElement('div');
    cell.innerHTML = `<b>${model}</b><span class="counts">${run.counts || run.cost_label}</span>`;
    head.appendChild(cell);
  });
  gallery.appendChild(head);

  order.forEach(sequence => {
    const label = document.createElement('div');
    label.className = 'section-label';
    label.textContent = `${sequence}  ·  ${bySequence[sequence][0].labels.join(' → ')}`;
    gallery.appendChild(label);

    const grid = document.createElement('div');
    grid.className = 'grid';
    models.forEach(model => {
      const run = bySequence[sequence].find(r => r.model === model);
      if (!run) {
        const empty = document.createElement('div');
        empty.className = 'card empty';
        empty.textContent = `no ${model} run for ${sequence}`;
        grid.appendChild(empty);
        return;
      }
      const card = document.createElement('a');
      card.className = 'card';
      card.href = `sequence.html?model=${run.model}&seq=${run.sequence}`;
      card.innerHTML = `
        <div class="model">${run.model}</div>
        <div class="seq">${run.sequence}</div>
        <div class="stat">cheapest: <b>${fmtCost(run.best_cost, run.cost_units)}</b>${inOrbit(run, run.best_cost)}</div>
        <div class="stat">in <b>${run.best_tof_years.toFixed(1)} yr</b> · ${run.n_front} on front of ${fmtInt(run.n_points)}</div>`;
      grid.appendChild(card);
    });
    gallery.appendChild(grid);
  });
}).catch(error => {
  document.getElementById('gallery').innerHTML =
    `<p class="placeholder">Could not load <code>data/index.json</code> (${error.message}).<br>` +
    'On GitHub Pages this usually means <code>docs/data/</code> was not committed.</p>';
});
