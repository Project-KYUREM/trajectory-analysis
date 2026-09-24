"""Shared mission setup: SPICE kernels, gravitational parameters, bodies, UDP.

Imported by the notebook, the optimisers, the batch runner, and the publish
step. Kernel loading happens at import time on purpose: pygmo's mp_island
ships the pickled problem to a child process, and a pickled pykep.planet.spice
is only ~250 bytes -- it carries the target/observer strings and the mus,
nothing else, and resolves positions against whatever kernel pool the child
happens to have. So every process that touches a problem has to furnsh the
kernels itself, which importing this module guarantees.

Sequences are strings over the letters below, always starting at Earth and
ending at Neptune: "EVEEJN", "EJN", "EVMJN", ...
"""
import datetime as _dt
import math as _math
from pathlib import Path

# spiceypy must be imported before pykep: whichever is imported first owns the
# CSPICE handle, and if pykep wins, spiceypy gets a *separate* kernel pool that
# never sees anything pykep loaded (spice.ktotal() stays 0 and every bodvcd()
# raises SpiceKERNELVARNOTFOUND).
import spiceypy as spice

import pykep as pk

# ---------------------------------------------------------------- kernels ---
HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "data").resolve()
RESULTS_DIR = (HERE / ".." / "results").resolve()
CHAMPION_PATH = HERE / "champion.json"

# de440.bsp carries only barycenters 1-10 plus 199, 299, 301, 399 -- no
# Neptune centre (899) and no Triton (801). nep097.bsp supplies both.
KERNELS = ("de440.bsp", "naif0012.tls", "pck00011.tpc", "gm_de440.tpc", "nep097.bsp")


def _load_kernels():
    for kernel in KERNELS:
        path = DATA_DIR / kernel
        if not path.is_file():
            raise FileNotFoundError(
                f"SPICE kernel missing: {path}\n"
                f"Run scripts/fetch_kernels.sh to download the kernel set.")
        # Feed both loaders rather than relying on the shared-pool coincidence.
        pk.util.load_spice_kernel(str(path))
        spice.furnsh(str(path))


_load_kernels()


def gm(body: str) -> float:
    """GM from the kernel pool, km^3/s^2 -> m^3/s^2."""
    _, values = spice.bodvcd(spice.bodn2c(body), "GM", 1)
    return values[0] * 1e9


# ----------------------------------------------------------- body registry --
# ephem: the SPICE target de440 can actually serve. Mars/Jupiter/Saturn have
#   no planet-centre ephemeris in de440, so their barycenters stand in; the
#   offsets (<=200 km even for Jupiter) are irrelevant at flyby scale.
# gm_of: the body whose GM prices the flyby (always the planet, never the
#   barycenter -- BODY5_GM folds in the Galilean moons, etc.).
# safe_ratio: periapsis floor for fb_prop, in planet radii. Jupiter's 9.0 is
#   pykep's own convention (radiation-belt standoff). Saturn is deliberately
#   NOT pykep's 1.1 -- that would fly through the B ring; 2.4 clears the F
#   ring (~2.32 R_S).
BODIES = {
    'V': dict(label='Venus', ephem='VENUS', gm_of='VENUS',
              radius=6051800.0, safe_ratio=1.1),
    'E': dict(label='Earth', ephem='EARTH', gm_of='EARTH',
              radius=6378137.0, safe_ratio=1.1),
    'M': dict(label='Mars', ephem='MARS BARYCENTER', gm_of='MARS',
              radius=3396200.0, safe_ratio=1.1),
    'J': dict(label='Jupiter', ephem='JUPITER BARYCENTER', gm_of='JUPITER',
              radius=71492000.0, safe_ratio=9.0),
    'S': dict(label='Saturn', ephem='SATURN BARYCENTER', gm_of='SATURN',
              radius=60268000.0, safe_ratio=2.4),
    'N': dict(label='Neptune', ephem='NEPTUNE', gm_of='NEPTUNE',
              radius=24622000.0, safe_ratio=1.1),
}

_planet_cache = {}


def make_planet(letter: str):
    """Heliocentric pykep planet for one sequence letter (cached)."""
    if letter not in _planet_cache:
        b = BODIES[letter]
        mu = pk.MU_EARTH if letter == 'E' else gm(b['gm_of'])
        _planet_cache[letter] = pk.planet.spice(
            b['ephem'], 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, mu,
            b['radius'], b['radius'] * b['safe_ratio'])
    return _planet_cache[letter]


def make_sequence(sequence: str):
    """(planets, labels) for a sequence string like 'EVEEJN'."""
    sequence = sequence.upper().strip()
    if not sequence.startswith('E') or not sequence.endswith('N'):
        raise ValueError(f"sequence must start at Earth and end at Neptune: {sequence!r}")
    unknown = set(sequence) - set(BODIES)
    if unknown:
        raise ValueError(f"unknown bodies {sorted(unknown)} in {sequence!r}")
    return ([make_planet(c) for c in sequence],
            [BODIES[c]['label'] for c in sequence])


DEFAULT_SEQUENCE = 'EVEEJN'

# Backward-compatible module-level default (the notebook uses these).
seq, LABELS = make_sequence(DEFAULT_SEQUENCE)
earth, venus = make_planet('E'), make_planet('V')
jupiter, neptune = make_planet('J'), make_planet('N')

MU_NEPTUNE = gm('NEPTUNE')
MU_TRITON = gm('TRITON')
MU_VENUS = gm('VENUS')
MU_JUPITER = gm('JUPITER')

R_EARTH = BODIES['E']['radius']
R_VENUS = BODIES['V']['radius']
R_JUPITER = BODIES['J']['radius']
R_NEPTUNE = BODIES['N']['radius']
R_TRITON = 1353400.0

# Neptune-centred body for stage two / notebook.
triton = pk.planet.spice('TRITON', 'NEPTUNE', 'ECLIPJ2000', 'NONE',
                         MU_NEPTUNE, MU_TRITON, R_TRITON, R_TRITON * 1.1)

# ------------------------------------------------------------- the problem --
# One bound on the whole cruise; tof_encoding='alpha' takes [min, max] days
# and lets the optimiser split the budget across legs itself.
MISSION_TOF_DAYS = [3000.0, 12000.0]

# vinf is in km/s -- get_bounds() does vinf[i] * 1000 internally.
LAUNCH_VINF_KMS = [0.0, 5.0]

LAUNCH_WINDOW = [pk.epoch(68000, 'mjd'), pk.epoch(77500, 'mjd')]

# ---------------------------------------------------- Neptune capture -------
# Arrival vinf is the speed relative to Neptune while still far away; it is
# NOT the cost of arriving. orbit_insertion=True makes pykep price the real
# periapsis burn into a loose capture ellipse instead of the raw vinf, which
# would overprice arrival ~10x and bias the search towards slow trajectories.
NEPTUNE_CAPTURE_RP = 1.1 * R_NEPTUNE   # periapsis of the capture orbit, m
NEPTUNE_CAPTURE_E = 0.99               # eccentricity of the capture orbit


def build_udp(multi_objective=False, orbit_insertion=True, launch_window=None,
              sequence=None):
    """The heliocentric MGA-1DSM problem for one sequence string.

    multi_objective=True -> fitness is (total DV, total TOF) for Pareto runs.
    orbit_insertion=True (default) -> the final DV term is the Neptune capture
      burn rather than raw arrival vinf. The two objectives are NOT comparable,
      but the decision vector is unaffected (get_bounds is objective-blind), so
      an x from either scores under both.
    launch_window -> [epoch, epoch] override, used by the slice sweep.
    """
    planets, _ = make_sequence(sequence or DEFAULT_SEQUENCE)
    window = launch_window if launch_window is not None else LAUNCH_WINDOW

    extra = {}
    if orbit_insertion:
        # pykep raises if add_vinf_arr is False while orbit_insertion is on.
        extra = dict(orbit_insertion=True,
                     rp_target=NEPTUNE_CAPTURE_RP,
                     e_target=NEPTUNE_CAPTURE_E)

    return pk.trajopt.mga_1dsm(
        seq=planets,
        t0=list(window),
        tof=MISSION_TOF_DAYS,
        tof_encoding='alpha',
        vinf=LAUNCH_VINF_KMS,
        add_vinf_dep=False,   # launcher supplies the departure vinf
        add_vinf_arr=True,    # required by orbit_insertion; see above
        multi_objective=multi_objective,
        **extra,
    )


# ------------------------------------------------- launch-window slicing ----
def orbital_period(planet, epoch=None):
    """Osculating period about the central body, seconds (planet.spice has no
    period(): derive a from the state, then T = 2*pi*sqrt(a^3/mu))."""
    epoch = epoch if epoch is not None else LAUNCH_WINDOW[0]
    r, v = planet.eph(epoch)
    r_mag = _math.sqrt(sum(c * c for c in r))
    v_sq = sum(c * c for c in v)
    mu = planet.mu_central_body
    a = 1.0 / (2.0 / r_mag - v_sq / mu)
    return 2.0 * _math.pi * _math.sqrt(a ** 3 / mu)


def synodic_period(p1, p2, epoch=None):
    """Days between successive p1 -> p2 launch opportunities. For a resonant
    first leg (same body twice, e.g. 'EE...') the synodic period diverges --
    any date works equally -- so fall back to two years per slice."""
    t1 = orbital_period(p1, epoch) / 86400.0
    t2 = orbital_period(p2, epoch) / 86400.0
    diff = abs(1.0 / t1 - 1.0 / t2)
    if diff < 1e-6 or 1.0 / diff > 3000.0:
        return 730.5
    return 1.0 / diff


def launch_slices(window=None, width_days=None, overlap=0.2, sequence=None):
    """Chop the launch window into overlapping slices, one per opportunity.

    First-leg launch opportunities recur at the synodic period of the first
    two bodies, separated by stretches of very poor DV. Handing an optimiser
    a multi-decade window makes it search across many nearly disconnected
    basins at once; one search per slice turns that into many easy problems.
    Slices overlap so a boundary optimum is not split and missed by both.
    """
    planets, _ = make_sequence(sequence or DEFAULT_SEQUENCE)
    window = window if window is not None else LAUNCH_WINDOW
    start_mjd, end_mjd = window[0].mjd, window[1].mjd
    width = width_days if width_days else synodic_period(planets[0], planets[1])
    step = width * (1.0 - overlap)

    out, t = [], start_mjd
    while t < end_mjd:
        hi = min(t + width, end_mjd)
        out.append([pk.epoch(t, 'mjd'), pk.epoch(hi, 'mjd')])
        if hi >= end_mjd:
            break
        t += step
    return out


def objective_config(udp=None):
    """How the final DV term is priced -- recorded into every saved result so a
    champion can never be silently compared against a differently-scored run."""
    udp = udp if udp is not None else build_udp()
    if getattr(udp, '_orbit_insertion', False):
        return {'objective': 'orbit_insertion',
                'capture_rp_m': NEPTUNE_CAPTURE_RP,
                'capture_e': NEPTUNE_CAPTURE_E}
    return {'objective': 'vinf_arr'}


# MJD2000 epoch, for turning decision-vector dates into calendar dates.
MJD2000_EPOCH = _dt.datetime(2000, 1, 1)


def mjd2000_to_datetime(mjd2000):
    return MJD2000_EPOCH + _dt.timedelta(days=float(mjd2000))
