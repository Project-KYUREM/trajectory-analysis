// Helpers shared by index.html and sequence.html. Loaded before the page scripts.

const AU = 1.495978707e11;
const MJD2000_MS = Date.UTC(2000, 0, 1);

// Colours come from style.css tokens so the plots follow the page theme.
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const theme = () => ({
  surface: cssVar('--surface'), ink: cssVar('--ink'), ink2: cssVar('--ink-2'),
  grid: cssVar('--grid'), accent: cssVar('--accent'), accent2: cssVar('--accent-2'),
  thrust: [cssVar('--thrust-lo'), cssVar('--thrust-mid'), cssVar('--thrust-hi')],
  cat: [cssVar('--cat-1'), cssVar('--cat-2'), cssVar('--cat-3')], catOther: cssVar('--cat-other'),
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

// Fetch a file that publish.py gzipped. Servers that already decode it
// (Content-Encoding: gzip) hand back plain bytes, so check the gzip magic
// number before decompressing.
async function loadGz(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes.buffer;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

// One run's payload (see orbit_planning/sitedata.py): metadata in cloud.json,
// points as gzipped binary columns, trajectory descriptions as gzipped JSON.
// Runs still in the old single-file cloud.json are returned as they are.
async function loadCloud(dir) {
  const cloud = await loadJSON(`${dir}/cloud.json`);
  if (Array.isArray(cloud.points)) return cloud;
  const head = cloud.points_file;
  const [buffer, detail] = await Promise.all([
    loadGz(`${dir}/${head.name}`),
    loadGz(`${dir}/${cloud.detail_file}`).then(b => JSON.parse(new TextDecoder().decode(b))),
  ]);
  const types = { '<f8': Float64Array, '<f4': Float32Array, '<u4': Uint32Array, u1: Uint8Array };
  const cols = {};
  let offset = 0;
  head.columns.forEach(([name, dtype]) => {
    const Type = types[dtype];
    cols[name] = new Type(buffer, offset, head.n);
    offset += Type.BYTES_PER_ELEMENT * head.n;
  });
  const flag = head.flags;
  const points = new Array(head.n);
  for (let i = 0; i < head.n; i++) {
    const f = cols.flags[i];
    points[i] = {
      id: 'p' + String(cols.index[i]).padStart(5, '0'),
      cost: cols.cost[i], tof_days: cols.tof_days[i], launch_mjd2000: cols.launch_mjd2000[i],
      pareto: !!(f & flag.pareto), polished: !!(f & flag.polished), detail: !!(f & flag.detail),
    };
  }
  return { ...cloud, points, detail };
}

function download(name, text) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
