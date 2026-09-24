const AU = 1.495978707e11;
const fmtCost = (v, u) => u === 'm/s' ? `${(v/1000).toFixed(2)} km/s` : `${v.toFixed(0)} ${u}`;

fetch('data/index.json').then(r => r.json()).then(runs => {
  const g = document.getElementById('gallery');
  if (!runs.length) { g.innerHTML = '<p class="placeholder">No runs published yet. Run batch.py then publish.py.</p>'; return; }

  // group by sequence, cards per model within
  const bySeq = {};
  runs.forEach(r => (bySeq[r.sequence] = bySeq[r.sequence] || []).push(r));
  const order = Object.keys(bySeq).sort((a, b) => a.length - b.length || a.localeCompare(b));

  g.innerHTML = '';
  order.forEach(seq => {
    const lab = document.createElement('div');
    lab.className = 'section-label';
    lab.textContent = `${seq}  ·  ${bySeq[seq][0].labels.join(' → ')}`;
    g.appendChild(lab);
    const grid = document.createElement('div');
    grid.className = 'grid';
    bySeq[seq].forEach(r => {
      const a = document.createElement('a');
      a.className = 'card';
      a.href = `sequence.html?model=${r.model}&seq=${r.sequence}`;
      a.innerHTML = `
        <div class="model">${r.model}</div>
        <div class="seq">${r.sequence}</div>
        <div class="stat">best <b>${fmtCost(r.best_cost, r.cost_units)}</b> · ${r.cost_label}</div>
        <div class="stat">at <b>${r.best_tof_years.toFixed(1)} yr</b> · ${r.n_front} on front / ${r.n_points} pts</div>`;
      grid.appendChild(a);
    });
    g.appendChild(grid);
  });
});
