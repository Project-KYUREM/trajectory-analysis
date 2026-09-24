"""Decode a decision vector into a tool-agnostic mission description.

``describe(model, sequence, x)`` returns one schema for every decodable model,
so the publish step and the site never branch on model:

    launch    epoch, C3, vinf vector, launch-asymptote RA/dec (EME2000)
    legs      per-leg time of flight, dates, deep-space maneuver (if the
              model has them)
    flybys    body, epoch, periapsis radius and plane angle where the model
              exposes them, and for the ballistic model the velocity
              mismatch a burn must cover because the assist cannot
    arrival   destination epoch, vinf vector and magnitude (the approach speed
              far from the body), capture burn if the model inserts, and the
              atmospheric entry speed once gravity has done its work
    arcs      per ballistic arc: r, v, dt, mu -- a Kepler propagator (the
              browser's ``kepler.js`` or GMAT) rebuilds the whole trajectory
              from these alone
    totals    headline cost, time of flight, maneuver sum

States are heliocentric ECLIPJ2000 (pykep's frame). The constant
ECLIPJ2000 -> J2000/EME2000 rotation is exported by ``publish.py`` so the site
can produce the equatorial frame GMAT and STK default to; the CSV and GMAT
exports themselves are generated client-side from this description.
"""
import math

import numpy as np
import spiceypy as spice

import mission
import models

# Both frames are inertial, so the rotation is time-invariant (23.4393 deg obliquity).
ECLIP_TO_EME = np.array(spice.pxform('ECLIPJ2000', 'J2000', 0.0))


def to_eme(vec):
    """Rotate an ECLIPJ2000 vector into J2000 / EME2000."""
    return [float(c) for c in ECLIP_TO_EME @ np.asarray(vec, dtype=float)]


def _ra_dec(vec):
    """Right ascension and declination (degrees) of a direction vector."""
    x, y, z = vec
    return (math.degrees(math.atan2(y, x)) % 360.0,
            math.degrees(math.asin(z / float(np.linalg.norm(vec)))))


def _date(mjd2000):
    """ISO date-time string for an MJD2000 epoch."""
    return mission.mjd2000_to_datetime(mjd2000).isoformat(timespec='seconds')


def entry_speed(vinf_ms, body, planet):
    """Speed at the atmospheric entry interface after falling into the well.

    ``sqrt(vinf^2 + 2*mu/r)``: even ``vinf = 0`` gives the local escape speed
    (~23 km/s at Neptune) -- one cannot arrive slowly at a giant planet.

    Args:
        vinf_ms: Hyperbolic excess speed on arrival, m/s.
        body: The destination's ``BODIES`` entry (radius, entry altitude).
        planet: The destination's pykep planet (its GM).

    Returns:
        Entry speed in m/s.
    """
    r = body['radius'] + body['entry_altitude_m']
    return math.sqrt(vinf_ms ** 2 + 2.0 * planet.mu_self / r)


def describe(model_name, sequence, x):
    """Decode one trajectory into the schema documented at the top of this module.

    Args:
        model_name: A decodable model (``capture`` or ``ballistic``).
        sequence: Sequence string the vector was optimised for.
        x: The decision vector.

    Returns:
        A JSON-serialisable dict; see the module docstring for its sections.

    Raises:
        NotImplementedError: If the model has no decoder (``nep``).
    """
    model = models.get(model_name)
    if model.decode is None:
        raise NotImplementedError(f"{model_name!r} is not decodable; it is a GMAT-seed model")

    setup = mission.resolve_bounds(sequence)
    udp = model.build(setup, False)
    decoded = model.decode(udp, x)
    cost = model.cost_of(udp.fitness(x))

    epochs = np.cumsum([x[0]] + decoded.leg_tof_days)
    body_v = [planet.eph(float(e))[1] for planet, e in zip(setup.planets, epochs)]
    vinf_dep = [a - b for a, b in zip(decoded.arc_states[0][1], body_v[0])]
    vinf_arr = [a - b for a, b in zip(decoded.lamberts[-1].get_v2()[0], body_v[-1])]
    vinf_dep_mag = float(np.linalg.norm(vinf_dep))
    vinf_arr_mag = float(np.linalg.norm(vinf_arr))
    rla, dla = _ra_dec(to_eme(vinf_dep))
    total_tof = float(sum(decoded.leg_tof_days))

    legs = [dict(**{'from': setup.labels[i], 'to': setup.labels[i + 1]},
                 tof_days=tof, dep_date=_date(epochs[i]), arr_date=_date(epochs[i + 1]),
                 dsm_ms=decoded.leg_dsm_ms[i] if decoded.leg_dsm_ms else None)
            for i, tof in enumerate(decoded.leg_tof_days)]

    flybys = []
    for i in range(1, len(setup.planets) - 1):
        flyby = dict(body=setup.labels[i], epoch_mjd2000=float(epochs[i]), date=_date(epochs[i]))
        if decoded.flyby_mismatch_ms:
            flyby['mismatch_dv_ms'] = decoded.flyby_mismatch_ms[i - 1]
        if decoded.flyby_rp_radii is not None:
            flyby['rp_radii'] = decoded.flyby_rp_radii[i - 1]
            flyby['rp_m'] = flyby['rp_radii'] * setup.planets[i].radius
            flyby['beta_rad'] = decoded.flyby_beta_rad[i - 1]
        flybys.append(flyby)

    arc_ends = list(decoded.arc_epochs[1:]) + [float(epochs[-1])]
    arcs = [dict(epoch_mjd2000=t, date=_date(t), dt_days=t_end - t, mu=mission.pk.MU_SUN,
                 r_eclip_m=[float(c) for c in r], v_eclip_ms=[float(c) for c in v])
            for (r, v), t, t_end in zip(decoded.arc_states, decoded.arc_epochs, arc_ends)]

    return {
        'cost': cost,
        'launch': dict(epoch_mjd2000=float(x[0]), date=_date(x[0]),
                       vinf_ms=vinf_dep_mag, c3_km2s2=(vinf_dep_mag / 1000.0) ** 2,
                       vinf_vec_eclip_ms=[float(c) for c in vinf_dep],
                       rla_deg=rla, dla_deg=dla),
        'legs': legs,
        'flybys': flybys,
        'arrival': dict(body=setup.labels[-1], epoch_mjd2000=float(epochs[-1]),
                        date=_date(epochs[-1]), vinf_ms=vinf_arr_mag,
                        vinf_vec_eclip_ms=[float(c) for c in vinf_arr],
                        capture_dv_ms=decoded.capture_dv_ms,
                        entry_speed_ms=entry_speed(vinf_arr_mag, setup.destination,
                                                   setup.planets[-1]),
                        entry_altitude_km=setup.destination['entry_altitude_m'] / 1e3),
        'totals': dict(cost=cost, tof_days=total_tof, tof_years=total_tof / 365.25,
                       dsm_total_ms=float(sum(decoded.leg_dsm_ms)),
                       flyby_mismatch_total_ms=float(sum(decoded.flyby_mismatch_ms))),
        'arcs': arcs,
    }
