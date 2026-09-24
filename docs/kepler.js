// Universal-variable Kepler propagator. Each trajectory arc arrives as an
// initial state (r, v) + duration + mu; propagating it reproduces the whole
// conic in-browser, so the site needs no SPICE and no server. Vallado's
// algorithm (Stumpff functions + Newton iteration on the universal anomaly).

function stumpff(z) {
  if (z > 1e-6) {
    const s = Math.sqrt(z);
    return { C: (1 - Math.cos(s)) / z, S: (s - Math.sin(s)) / (s * s * s) };
  } else if (z < -1e-6) {
    const s = Math.sqrt(-z);
    return { C: (1 - Math.cosh(s)) / z, S: (Math.sinh(s) - s) / (s * s * s) };
  }
  return { C: 0.5, S: 1 / 6 };
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const nrm = (a) => Math.sqrt(dot(a, a));

// Propagate (r0, v0) forward by dt seconds under point mass mu. Returns [r, v].
function propagate(r0, v0, dt, mu) {
  if (dt === 0) return [r0.slice(), v0.slice()];   // hyperbolic guess below would log(0)
  const r0m = nrm(r0), v0m = nrm(v0);
  const sqmu = Math.sqrt(mu);
  const alpha = 2 / r0m - v0m * v0m / mu;   // 1/a
  const rdotv = dot(r0, v0);

  // initial guess for universal anomaly chi
  let chi;
  if (alpha > 1e-9) {                       // ellipse
    chi = sqmu * dt * alpha;
  } else if (alpha < -1e-9) {               // hyperbola
    const a = 1 / alpha;
    chi = Math.sign(dt) * Math.sqrt(-a) *
      Math.log((-2 * mu * alpha * dt) /
        (rdotv + Math.sign(dt) * Math.sqrt(-mu * a) * (1 - r0m * alpha)));
  } else {                                  // near-parabolic
    chi = sqmu * dt / r0m;
  }

  let r = r0m;
  for (let i = 0; i < 200; i++) {
    const psi = chi * chi * alpha;
    const { C, S } = stumpff(psi);
    r = chi * chi * C + (rdotv / sqmu) * chi * (1 - psi * S) + r0m * (1 - psi * C);
    const dchi = (sqmu * dt - chi * chi * chi * S -
      (rdotv / sqmu) * chi * chi * C - r0m * chi * (1 - psi * S)) / r;
    chi += dchi;
    if (Math.abs(dchi) < 1e-8) break;
  }

  const psi = chi * chi * alpha;
  const { C, S } = stumpff(psi);
  const f = 1 - (chi * chi / r0m) * C;
  const g = dt - (chi * chi * chi / sqmu) * S;
  const rv = [f * r0[0] + g * v0[0], f * r0[1] + g * v0[1], f * r0[2] + g * v0[2]];
  const rm = nrm(rv);
  const fdot = (sqmu / (rm * r0m)) * chi * (psi * S - 1);
  const gdot = 1 - (chi * chi / rm) * C;
  const vv = [fdot * r0[0] + gdot * v0[0], fdot * r0[1] + gdot * v0[1],
              fdot * r0[2] + gdot * v0[2]];
  return [rv, vv];
}

// Sample one arc into a polyline of [x,y,z] points (metres).
function sampleArc(arc, n = 60) {
  const dtSec = arc.dt_days * 86400;
  const pts = [];
  for (let k = 0; k <= n; k++) {
    const [r] = propagate(arc.r_eclip_m, arc.v_eclip_ms, dtSec * k / n, arc.mu);
    pts.push(r);
  }
  return pts;
}
