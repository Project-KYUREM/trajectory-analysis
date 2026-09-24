"""Decode a decision vector into a full, tool-agnostic mission description.

One `describe(model, sequence, x)` returns the same schema for every impulsive
model, so the publish step, the exports, and the site never branch on model:

    launch   epoch, C3, vinf vector, RLA/DLA (EME2000) -- launch-vehicle inputs
    legs     per-leg TOF, dates, DSM magnitude
    flybys   body, epoch, periapsis radius (capture) / turn (ballistic)
    arrival  Neptune epoch, vinf vector & magnitude (the "velocity entering
             Neptune"), capture burn if the model does insertion
    arcs     per ballistic arc: r, v, dt, mu  -> a Kepler propagator (browser or
             GMAT) reproduces the whole trajectory from these alone
    totals   headline cost, TOF, DSM sum

States are given in ECLIPJ2000 (pykep's native frame, heliocentric) AND J2000 /
EME2000 (what GMAT and STK default to); the two differ by the 23.4393 deg
obliquity rotation.
"""
import math

import numpy as np
import spiceypy as spice

import mission
import models

# Constant ECLIPJ2000 -> J2000 (EME2000) rotation (both inertial; time-invariant).
_R_ECLIP_TO_EME = np.array(spice.pxform('ECLIPJ2000', 'J2000', 0.0))


def _to_eme(vec):
    return list(_R_ECLIP_TO_EME @ np.asarray(vec, dtype=float))


def _norm(v):
    return float(np.linalg.norm(v))


def _ra_dec(vec):
    """Right ascension / declination (deg) of a vector, for launch asymptote."""
    x, y, z = vec
    ra = math.degrees(math.atan2(y, x)) % 360.0
    dec = math.degrees(math.asin(z / _norm(vec)))
    return ra, dec


def _date(mjd2000):
    return mission.mjd2000_to_datetime(mjd2000).isoformat(timespec='seconds')


def _arc_records(ballistic_legs, ballistic_ep, tof_days, mu):
    """Turn pykep's (r, v) arc starts + epochs into propagatable arc records.

    dt of each arc is the gap to the next arc start; the final arc runs to the
    end of the last leg.
    """
    arcs = []
    eps = list(ballistic_ep) + [ballistic_ep[0] + sum(tof_days)]
    for i, (r, v) in enumerate(ballistic_legs):
        dt = (eps[i + 1] - eps[i]) * 86400.0
        arcs.append({
            'epoch_mjd2000': float(eps[i]),
            'date': _date(eps[i]),
            'dt_days': (eps[i + 1] - eps[i]),
            'mu': mu,
            'r_eclip_m': [float(c) for c in r],
            'v_eclip_ms': [float(c) for c in v],
            'r_eme_m': _to_eme(r),
            'v_eme_ms': _to_eme(v),
        })
    return arcs


def describe(model, sequence, x):
    spec = models.get(model)
    if not spec['impulsive']:
        raise NotImplementedError(
            f"describe() covers impulsive models; {model!r} is a NEP seed "
            f"(see trajectory.nep_seed / export).")

    udp = models.build(model, sequence, multi_objective=False)
    planets, labels = mission.make_sequence(sequence)
    mu = mission.pk.MU_SUN
    t0 = x[0]

    # planet states at each encounter
    ep = np.cumsum([t0] + list(_leg_tofs(udp, x)))
    r_P, v_P = [], []
    for i, pl in enumerate(planets):
        r, v = pl.eph(float(ep[i]))
        r_P.append(r)
        v_P.append(v)

    if model == 'capture':
        DV, lamberts, T, bleg, bep = udp._compute_dvs(x)
        dsm = [float(d) for d in DV[:-1]]
        arrival_term = float(DV[-1])          # capture burn
        v_arr = lamberts[-1].get_v2()[0]
        v_dep = bleg[0][1]
        cost = spec['cost_of'](udp.fitness(x))
        capture_dv = arrival_term
    else:  # ballistic
        DVlaunch, DVfb, DVarrival, lamberts, DVl_tot, T, bleg, bep = udp._compute_dvs(x)
        dsm = [float(d) for d in DVfb]        # flyby mismatch (ideally ~0)
        arrival_term = float(DVarrival)
        v_arr = lamberts[-1].get_v2()[0]
        v_dep = bleg[0][1]
        cost = spec['cost_of'](udp.fitness(x))
        capture_dv = arrival_term if udp.orbit_insertion else None

    # departure
    vinf_dep = [a - b for a, b in zip(v_dep, v_P[0])]
    vinf_dep_eme = _to_eme(vinf_dep)
    rla, dla = _ra_dec(vinf_dep_eme)
    vinf_dep_mag = _norm(vinf_dep)

    # arrival at Neptune
    vinf_arr = [a - b for a, b in zip(v_arr, v_P[-1])]
    vinf_arr_mag = _norm(vinf_arr)

    total_tof = float(sum(T))

    legs = []
    for i in range(len(T)):
        legs.append({
            'from': labels[i], 'to': labels[i + 1],
            'tof_days': float(T[i]),
            'dep_date': _date(ep[i]), 'arr_date': _date(ep[i + 1]),
            'dsm_ms': dsm[i] if i < len(dsm) else None,
        })

    flybys = []
    for i in range(1, len(planets) - 1):
        fb = {'body': labels[i], 'epoch_mjd2000': float(ep[i]), 'date': _date(ep[i])}
        if model == 'capture':
            # x layout: [t0,u,v,Vinf,eta1,a1] + [beta, rp/rP, eta2, a2]*(n-1)
            fb['rp_radii'] = float(x[7 + 4 * (i - 1)])
            fb['rp_m'] = fb['rp_radii'] * planets[i].radius
            fb['beta_rad'] = float(x[6 + 4 * (i - 1)])
        flybys.append(fb)

    return {
        'model': model, 'sequence': sequence, 'labels': labels,
        'cost': cost, 'cost_label': spec['label'], 'cost_units': spec['units'],
        'launch': {
            'epoch_mjd2000': float(t0), 'date': _date(t0),
            'vinf_ms': vinf_dep_mag,
            'c3_km2s2': (vinf_dep_mag / 1000.0) ** 2,
            'vinf_vec_eclip_ms': [float(c) for c in vinf_dep],
            'vinf_vec_eme_ms': vinf_dep_eme,
            'rla_deg': rla, 'dla_deg': dla,
        },
        'legs': legs,
        'flybys': flybys,
        'arrival': {
            'epoch_mjd2000': float(ep[-1]), 'date': _date(ep[-1]),
            'vinf_ms': vinf_arr_mag,
            'vinf_vec_eclip_ms': [float(c) for c in vinf_arr],
            'vinf_vec_eme_ms': _to_eme(vinf_arr),
            'capture_dv_ms': capture_dv,
        },
        'totals': {
            'cost': cost, 'cost_units': spec['units'],
            'tof_days': total_tof, 'tof_years': total_tof / 365.25,
            'dsm_total_ms': float(sum(dsm)),
        },
        'arcs': _arc_records(bleg, bep, T, mu),
    }


def _leg_tofs(udp, x):
    """Per-leg TOF (days) for either impulsive model."""
    if hasattr(udp, '_decode_tofs'):          # mga (ballistic)
        return list(udp._decode_tofs(x))
    return list(udp._compute_dvs(x)[2])        # mga_1dsm (capture)


# ----------------------------------------------------------------- exports --
def maneuvers(desc):
    """Flat maneuver list (epoch, kind, dv) for CSV/GMAT: launch, DSMs, capture."""
    rows = [{'epoch_mjd2000': desc['launch']['epoch_mjd2000'],
             'date': desc['launch']['date'], 'kind': 'launch',
             'dv_ms': desc['launch']['vinf_ms'],
             'note': f"C3={desc['launch']['c3_km2s2']:.3f} km2/s2 "
                     f"RLA={desc['launch']['rla_deg']:.2f} DLA={desc['launch']['dla_deg']:.2f}"}]
    for i, leg in enumerate(desc['legs']):
        if leg.get('dsm_ms'):
            rows.append({'epoch_mjd2000': None, 'date': leg['arr_date'],
                         'kind': f"dsm_leg{i + 1}", 'dv_ms': leg['dsm_ms'],
                         'note': f"{leg['from']}->{leg['to']}"})
    if desc['arrival'].get('capture_dv_ms'):
        rows.append({'epoch_mjd2000': desc['arrival']['epoch_mjd2000'],
                     'date': desc['arrival']['date'], 'kind': 'neptune_capture',
                     'dv_ms': desc['arrival']['capture_dv_ms'],
                     'note': f"arrival vinf={desc['arrival']['vinf_ms']:.1f} m/s"})
    return rows


def to_states_csv(desc):
    """CSV of every ballistic-arc initial state in both frames -- the handoff
    a GMAT/STK user pastes in and re-propagates."""
    lines = ["arc,epoch_mjd2000,date,dt_days,mu_m3s2,"
             "x_eclip_m,y_eclip_m,z_eclip_m,vx_eclip_ms,vy_eclip_ms,vz_eclip_ms,"
             "x_eme_m,y_eme_m,z_eme_m,vx_eme_ms,vy_eme_ms,vz_eme_ms"]
    for i, a in enumerate(desc['arcs']):
        r_e, v_e, r_m, v_m = a['r_eclip_m'], a['v_eclip_ms'], a['r_eme_m'], a['v_eme_ms']
        lines.append(",".join(str(v) for v in
                     [i, a['epoch_mjd2000'], a['date'], a['dt_days'], a['mu'],
                      *r_e, *v_e, *r_m, *v_m]))
    return "\n".join(lines) + "\n"


def to_gmat_seed(desc):
    """A GMAT-oriented seed summary: the departure state (EME2000, what GMAT's
    default coordinate system uses) plus maneuver epochs and the flyby targets.
    This is the starting point for converting to a finite/low-thrust burn: drop
    it into a GMAT mission, replace each impulsive Maneuver with a finite burn
    at the same epoch, and re-target the flybys. See PLAN.md / the NEP note."""
    dep = desc['arcs'][0]
    lines = [
        f"% GMAT seed  --  {desc['model']} {desc['sequence']}",
        f"% cost {desc['cost']:.1f} {desc['cost_units']}  "
        f"TOF {desc['totals']['tof_years']:.2f} yr",
        f"% Launch {desc['launch']['date']}  C3 {desc['launch']['c3_km2s2']:.4f} km^2/s^2",
        f"%   RLA {desc['launch']['rla_deg']:.3f} deg   DLA {desc['launch']['dla_deg']:.3f} deg",
        "% Heliocentric departure state, EME2000 (J2000), metres & m/s:",
        f"%   X  = {dep['r_eme_m'][0]:.3f}",
        f"%   Y  = {dep['r_eme_m'][1]:.3f}",
        f"%   Z  = {dep['r_eme_m'][2]:.3f}",
        f"%   VX = {dep['v_eme_ms'][0]:.6f}",
        f"%   VY = {dep['v_eme_ms'][1]:.6f}",
        f"%   VZ = {dep['v_eme_ms'][2]:.6f}",
        "% Encounter sequence and epochs:",
    ]
    for f in desc['flybys']:
        rp = f.get('rp_m')
        lines.append(f"%   {f['body']:8s} {f['date']}" +
                     (f"   rp = {rp / 1e3:.1f} km" if rp else ""))
    lines.append(f"%   Neptune  {desc['arrival']['date']}   "
                 f"vinf = {desc['arrival']['vinf_ms']:.1f} m/s")
    lines.append("% Maneuvers to model as finite/EP burns:")
    for mv in maneuvers(desc):
        lines.append(f"%   {mv['kind']:16s} dv = {mv['dv_ms']:8.1f} m/s   {mv['note']}")
    return "\n".join(lines) + "\n"
