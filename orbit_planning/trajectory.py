"""Decode a decision vector into a model-independent mission description.

States are heliocentric ECLIPJ2000, the frame pykep uses; ``ECLIP_TO_EME``
rotates them into J2000/EME2000. Epochs are MJD2000.
"""
import math

import numpy as np
import spiceypy as spice

import mission
import models

# Both frames are inertial, so the rotation is time-invariant (23.4393 deg obliquity).
ECLIP_TO_EME = np.array(spice.pxform('ECLIPJ2000', 'J2000', 0.0))


def to_eme(vec):
    """Rotate an ECLIPJ2000 vector into J2000/EME2000."""
    return [float(c) for c in ECLIP_TO_EME @ np.asarray(vec, dtype=float)]


def _ra_dec(vec):
    """Return the right ascension and declination, in degrees, of a direction vector.

    A zero vector (a C3 = 0 launch has no asymptote) gives ``(None, None)``.
    """
    x, y, z = vec
    if not float(np.linalg.norm(vec)):
        return None, None
    return (math.degrees(math.atan2(y, x)) % 360.0,
            math.degrees(math.asin(z / float(np.linalg.norm(vec)))))


def _date(mjd2000):
    """Return the ISO date-time string of an MJD2000 epoch."""
    return mission.mjd2000_to_datetime(mjd2000).isoformat(timespec='seconds')


def entry_speed(vinf_ms, body, planet):
    """Return the speed at the destination's atmospheric entry interface.

    The speed is ``sqrt(vinf^2 + 2*mu/r)``, so even ``vinf = 0`` gives the local
    escape speed (about 23 km/s at Neptune).

    Args:
        vinf_ms (float): Hyperbolic excess speed on arrival, in m/s.
        body (dict): The destination's ``mission.BODIES`` entry.
        planet (pykep.planet.spice): The destination planet, which supplies its GM.

    Returns:
        float: Entry speed in m/s.
    """
    r = body['radius'] + body['entry_altitude_m']
    return math.sqrt(vinf_ms ** 2 + 2.0 * planet.mu_self / r)


def describe(model_name, sequence, x):
    """Decode one trajectory into a JSON-serialisable mission description.

    Args:
        model_name (str): Name of a decodable model: ``capture``, ``ballistic``
            or a low-thrust model (``dfd``, ``dfd-1mw``).
        sequence (str): Sequence string the vector was optimised for.
        x (list[float]): Decision vector.

    Returns:
        dict: Sections ``cost``, ``launch`` (epoch, C3, vinf, launch-asymptote
        RA/dec in EME2000), ``legs``, ``flybys``, ``arrival`` (vinf, capture burn,
        entry speed), ``totals`` and ``arcs``. Each ``arcs`` entry holds the
        ``r``, ``v``, ``dt`` and ``mu`` a Kepler propagator needs to rebuild that
        ballistic arc. Low-thrust models add a ``lowthrust`` section (masses,
        spacecraft, capture spiral, constraint residuals), per-leg
        ``impulses``, and in ``legs[i].dsm_ms`` the leg's total thrust delta-v;
        their arcs are the Sims-Flanagan coasts between impulses, and
        ``totals.tof_days`` includes the capture spiral.

    Raises:
        NotImplementedError: If the model has no decoder.
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
    vinf_dep = (decoded.vinf_dep_vec if decoded.vinf_dep_vec is not None
                else [a - b for a, b in zip(decoded.arc_states[0][1], body_v[0])])
    vinf_arr = (decoded.vinf_arr_vec if decoded.vinf_arr_vec is not None
                else [a - b for a, b in zip(decoded.lamberts[-1].get_v2()[0], body_v[-1])])
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

    description = {
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
    if decoded.lowthrust is not None:
        lt = dict(decoded.lowthrust)
        for leg, imps in zip(legs, lt.pop('impulses')):
            leg['impulses'] = [dict(p, date=_date(p['epoch_mjd2000'])) for p in imps]
        for flyby, vinf in zip(flybys, lt['flyby_vinf_in_ms']):
            flyby['vinf_ms'] = vinf
        cap = lt['capture']
        description['arrival'].update(
            capture_dv_ms=cap['dv_ms'], capture_spiral_days=cap['days'],
            orbit_periapsis_km=cap['periapsis_km'], orbit_apoapsis_km=cap['apoapsis_km'],
            final_mass_kg=lt['final_mass_kg'],
            orbit_epoch_mjd2000=float(epochs[-1]) + cap['days'],
            orbit_date=_date(float(epochs[-1]) + cap['days']))
        description['totals'].update(
            thrust_dv_ms=float(sum(decoded.leg_dsm_ms)) + cap['dv_ms'],
            tof_days=total_tof + cap['days'], tof_years=(total_tof + cap['days']) / 365.25,
            cruise_days=total_tof)
        description['lowthrust'] = lt
    return description
