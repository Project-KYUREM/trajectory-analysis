// Helpers shared by index.html and sequence.html. Loaded before the page scripts.

const AU = 1.495978707e11;
const MJD2000_MS = Date.UTC(2000, 0, 1);

// Colours come from style.css tokens so the plots follow the page theme.
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const theme = () => ({
  surface: cssVar('--surface'), ink: cssVar('--ink'), ink2: cssVar('--ink-2'),
  grid: cssVar('--grid'), accent: cssVar('--accent'), accent2: cssVar('--accent-2'),
});

// One formatting policy for the cost axis, whatever its units.
function fmtCost(value, units) {
  return units === 'm/s' ? `${(value / 1000).toFixed(2)} km/s` : `${value.toFixed(0)} ${units}`;
}

const mjd2000ToDate = mjd => new Date(MJD2000_MS + mjd * 86400e3);
const mjd2000ToYear = mjd => 2000 + mjd / 365.25;
const isoDay = value => (value instanceof Date ? value.toISOString() : String(value || '')).slice(0, 10);

// Apply a 3x3 rotation matrix (rows) to a vector.
const rotate = (M, v) => M.map(row => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);

async function loadJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

function download(name, text) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
