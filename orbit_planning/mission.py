"""Load the SPICE kernels and define the body registry, sequences and bounds.

Kernels load at import time, so every process that imports this module,
including pygmo's spawned workers, has them: a pickled ``pykep.planet.spice``
resolves positions against the kernel pool of the process it lands in.

A sequence is a string of ``BODIES`` letters that starts at Earth and ends at
a body with capture parameters, such as ``"EVEEJN"``. Stored epochs are
MJD2000 (MJD - 51544.5); ``LAUNCH_WINDOW`` is quoted in MJD.
"""
import datetime as _dt
import math as _math
from dataclasses import dataclass

# spiceypy must be imported before pykep: whichever is imported first owns the
# CSPICE handle, and if pykep wins, spiceypy gets a *separate* kernel pool that
# never sees anything pykep loaded (spice.ktotal() stays 0 and every bodvcd()
# raises SpiceKERNELVARNOTFOUND).
import spiceypy as spice

import pykep as pk

import paths

# de440.bsp carries only barycenters 1-10 plus 199, 299, 301, 399 -- no
# Neptune centre (899) and no Triton (801). nep097.bsp supplies both.
KERNELS = ("de440.bsp", "naif0012.tls", "pck00011.tpc", "gm_de440.tpc", "nep097.bsp")


def _load_kernels():
    """Load every kernel in ``KERNELS`` into both pykep's and spiceypy's pool.

    Raises:
        FileNotFoundError: If a kernel is missing from ``paths.DATA_DIR``.
    """
    for kernel in KERNELS:
        path = paths.DATA_DIR / kernel
        if not path.is_file():
            raise FileNotFoundError(
                f"SPICE kernel missing: {path}\n"
                f"Run scripts/fetch_kernels.sh to download the kernel set.")
        # Feed both loaders rather than relying on the shared-pool coincidence.
        pk.util.load_spice_kernel(str(path))
        spice.furnsh(str(path))


_load_kernels()


def gm(body):
    """Return the gravitational parameter of a SPICE body.

    Args:
        body (str): SPICE body name, such as ``"NEPTUNE"``.

    Returns:
        float: GM in m^3/s^2.
    """
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
# capture / entry_altitude_m: present only on bodies that can end a sequence.
#   capture = the loose insertion ellipse the `capture` model prices;
#   entry_altitude_m = height of the atmospheric entry interface above the
#   1-bar radius, used for the reported entry speed.
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
              radius=24622000.0, safe_ratio=1.1,
              # loose first-capture ellipse (cheap to enter, walked down later);
              # 1000 km is the usual giant-planet entry-interface altitude
              capture=dict(rp_ratio=1.1, e=0.99), entry_altitude_m=1000e3),
}

_planet_cache = {}


def make_planet(letter):
    """Return the cached heliocentric pykep planet for one ``BODIES`` letter."""
    if letter not in _planet_cache:
        body = BODIES[letter]
        mu = pk.MU_EARTH if letter == 'E' else gm(body['gm_of'])
        _planet_cache[letter] = pk.planet.spice(
            body['ephem'], 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, mu,
            body['radius'], body['radius'] * body['safe_ratio'])
    return _planet_cache[letter]


def make_sequence(sequence):
    """Validate a sequence string and resolve it to planets and labels.

    Args:
        sequence (str): ``BODIES`` letters, such as ``"EVEEJN"``; case-insensitive.

    Returns:
        tuple[list[pykep.planet.spice], list[str]]: The planets in visiting order
        and their names.

    Raises:
        ValueError: If a letter is unknown, the sequence does not start at Earth,
            or the final body has no capture parameters.
    """
    sequence = sequence.upper().strip()
    unknown = set(sequence) - set(BODIES)
    if unknown:
        raise ValueError(f"unknown bodies {sorted(unknown)} in {sequence!r}")
    if not sequence.startswith('E'):
        raise ValueError(f"sequence must start at Earth: {sequence!r}")
    if 'capture' not in BODIES[sequence[-1]]:
        destinations = [k for k, b in BODIES.items() if 'capture' in b]
        raise ValueError(f"sequence must end at a destination {destinations}: {sequence!r}")
    return [make_planet(c) for c in sequence], [BODIES[c]['label'] for c in sequence]


# ------------------------------------------------------------------ bounds --
# One bound on the whole cruise (8.2 .. 32.9 years); tof_encoding='alpha' takes
# [min, max] days and lets the optimiser split the budget across legs itself.
# Trident-class flyby profiles sit near 12 yr; capture-heavy ones run to ~30.
MISSION_TOF_DAYS = [3000.0, 12000.0]

# Launch hyperbolic excess speed, km/s (pykep multiplies by 1000 internally).
# 5 km/s is C3 = 25 km^2/s^2, roughly what a heavy-lift launcher gives a
# few-tonne probe; the optimiser may use less.
LAUNCH_VINF_KMS = [0.0, 5.0]

# Two time scales are in play. NAIF/JPL tables and this window are quoted in
# MJD; pykep works and every stored value (x[0], launch_mjd2000, ...) is in
# MJD2000 = MJD - 51544.5. This window is Jan 2045 .. Jan 2071.
LAUNCH_WINDOW = [pk.epoch(68000, 'mjd'), pk.epoch(77500, 'mjd')]


@dataclass(frozen=True)
class Setup:
    """Hold everything a model builder needs to construct a problem.

    Args:
        sequence (str): Upper-case sequence string.
        planets (list[pykep.planet.spice]): Planets in visiting order.
        labels (list[str]): Body names in visiting order.
        destination (dict): ``BODIES`` entry of the final body.
        window (list[pykep.epoch]): ``[start, end]`` launch window.
        tof_days (list[float]): ``[min, max]`` total time of flight in days.
    """
    sequence: str
    planets: list
    labels: list
    destination: dict
    window: list
    tof_days: list


def resolve_bounds(sequence, launch_window=None, tof_days=None):
    """Resolve a sequence and optional overrides into a ``Setup``.

    Args:
        sequence (str): Sequence string, see ``make_sequence``.
        launch_window (list[pykep.epoch], optional): ``[start, end]`` launch
            window; ``None`` uses ``LAUNCH_WINDOW``. Defaults to None.
        tof_days (list[float], optional): ``[min, max]`` total time of flight in
            days; ``None`` uses ``MISSION_TOF_DAYS``. Defaults to None.

    Returns:
        Setup: The resolved problem setup.
    """
    planets, labels = make_sequence(sequence)
    return Setup(
        sequence=sequence.upper().strip(),
        planets=planets,
        labels=labels,
        destination=BODIES[sequence.upper().strip()[-1]],
        window=list(launch_window) if launch_window is not None else LAUNCH_WINDOW,
        tof_days=list(tof_days) if tof_days is not None else MISSION_TOF_DAYS,
    )


# ------------------------------------------------- launch-window slicing ----
def orbital_period(planet, epoch=None):
    """Return the osculating period of a planet about its central body.

    Args:
        planet (pykep.planet.spice): Planet to evaluate.
        epoch (pykep.epoch, optional): Epoch of the state; ``None`` uses the
            start of ``LAUNCH_WINDOW``. Defaults to None.

    Returns:
        float: Period in seconds.
    """
    epoch = epoch if epoch is not None else LAUNCH_WINDOW[0]
    r, v = planet.eph(epoch)
    r_mag = _math.sqrt(sum(c * c for c in r))
    v_sq = sum(c * c for c in v)
    mu = planet.mu_central_body
    a = 1.0 / (2.0 / r_mag - v_sq / mu)
    return 2.0 * _math.pi * _math.sqrt(a ** 3 / mu)


def synodic_period(p1, p2, epoch=None):
    """Return the days between successive launch opportunities from p1 to p2.

    Args:
        p1 (pykep.planet.spice): Departure planet.
        p2 (pykep.planet.spice): Arrival planet.
        epoch (pykep.epoch, optional): Epoch of the orbits; ``None`` uses the
            start of ``LAUNCH_WINDOW``. Defaults to None.

    Returns:
        float: Synodic period in days, or 730.5 when it exceeds 3000 days or
        diverges, as for a repeated first body such as ``"EE..."``.
    """
    t1 = orbital_period(p1, epoch) / 86400.0
    t2 = orbital_period(p2, epoch) / 86400.0
    diff = abs(1.0 / t1 - 1.0 / t2)
    return 730.5 if diff < 1e-6 or 1.0 / diff > 3000.0 else 1.0 / diff


def launch_slices(sequence, window=None, width_days=None, overlap=0.2):
    """Split a launch window into overlapping slices, one per launch opportunity.

    Opportunities recur at the synodic period of the first two bodies. Slices
    overlap so an optimum on a boundary lies inside at least one slice.

    Args:
        sequence (str): Sequence string; its first two bodies set the default
            slice width.
        window (list[pykep.epoch], optional): ``[start, end]`` to split; ``None``
            uses ``LAUNCH_WINDOW``. Defaults to None.
        width_days (float, optional): Slice width in days; ``None`` uses the
            synodic period. Defaults to None.
        overlap (float, optional): Fraction of each slice shared with the next.
            Defaults to 0.2.

    Returns:
        list[list[pykep.epoch]]: ``[start, end]`` slices covering the window.
    """
    planets, _ = make_sequence(sequence)
    window = window if window is not None else LAUNCH_WINDOW
    start_mjd, end_mjd = window[0].mjd, window[1].mjd
    width = width_days if width_days else synodic_period(planets[0], planets[1])
    step = width * (1.0 - overlap)

    slices, t = [], start_mjd
    while t < end_mjd:
        hi = min(t + width, end_mjd)
        slices.append([pk.epoch(t, 'mjd'), pk.epoch(hi, 'mjd')])
        if hi >= end_mjd:
            break
        t += step
    return slices


# ------------------------------------------------------------------- dates --
MJD2000_EPOCH = _dt.datetime(2000, 1, 1)


def mjd2000_to_datetime(mjd2000):
    """Return the calendar datetime of a pykep MJD2000 epoch."""
    return MJD2000_EPOCH + _dt.timedelta(days=float(mjd2000))
