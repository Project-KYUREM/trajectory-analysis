"""Read a solved ``TritonProblem`` back into plain numbers, check it, save it and tabulate it.

``summarize`` builds the dict written to ``solution.json``: inputs (opt flags
resolved), constants, atmosphere, arrival B-plane, Neptune periapsis with the
ring-plane crossings and the Neptune-GRAM drag estimate, the Triton encounter
with the direct-intercept comparison, ballute and probe passes (every heating
term), capture orbit, burn, final orbit, the ``stability_days`` propagation,
corridor, solver stages, the simulate check, the margins and assumptions in
effect with the constraints active at the solution, and every warning.
``print_summary`` prints it in the section order shared with the Neptune
mission: header, solver, arrival, flyby, entry, loads, heating, final orbit,
checks, assumptions and margins, warnings. ``table_row`` flattens a summary
into one sweep/trade CSV row (columns in ``COLUMNS``). JSON, CSV and table
writers are in ``core.io``.

Checks:
    simulate     each phase re-integrated from its collocated start (Dymos
                 simulate); the final Triton apoapsis must agree within
                 ``sim_tol`` (relative) and the periapsis within ``sim_rp_tol_km``.
    rings        z sign changes of the Neptune-phase trajectory (simulated when
                 available); each crossing radius must lie inside
                 ``ring_keepout_km[0]`` or outside ``ring_keepout_km[1]``.
    stability    scipy propagation of the final state for ``stability_days``
                 with Triton, Neptune point mass + J2 and the differential
                 tide (``frame.accel_tidal``); the distance from Triton
                 must stay within [R_T + rp_final_alt_km, ra_max].
"""
import math
from dataclasses import asdict

import numpy as np

from aerocapture.core import atmosphere
from aerocapture.core import dynamics
from aerocapture.core.guess import propagate
from aerocapture.core import io
from aerocapture.core import system
from aerocapture.triton import ballute
from aerocapture.triton import frame as tf
from aerocapture.triton import guess as guess_mod

STABILITY_SAMPLE_S = 300.0        # sampling step of the stability propagation, s
NOTES = [
    "Triton's ephemeris is a circular fit to nep097 about the epoch; the Dymos Triton-frame "
    "phases use the exactly translated NEQ gravity, the stability propagation the physical "
    "differential tide.",
    "Heating: Sutton-Graves N2 continuum bridged to free-molecular; the Tauber-Sutton air "
    "radiative term is an optically-thin upper bound (linear in R_n) and is constrained only "
    "with rad_in_limits.",
    "Sun tide (~2e-10 of Triton's pull) and Triton J2 are neglected; ballute inflation, "
    "flutter and Triton winds are out of scope.",
]

COLUMNS = {
    "scope": "triton_only (S0) or full mission",
    "vinf_kms": "Neptune hyperbolic excess speed, km/s",
    "encounter": "Triton encounter sense of the guess (co / counter)",
    "ballute": "ballute name",
    "diameter_m": "ballute diameter, m",
    "envelope_mass_kg": "ballute envelope mass, kg",
    "probe_mass_kg": "probe mass, kg",
    "beta_ballute_kg_m2": "continuum ballistic coefficient of ballute + probe, kg/m^2",
    "converged": "IPOPT reported success on the final stage",
    "ipopt_status": "IPOPT exit text (or the failure reason)",
    "ipopt_iterations": "IPOPT iterations of the last stage",
    "neptune_rp_rn": "Neptune periapsis radius, R_N",
    "ring_check": "every ring-plane crossing outside the keep-out band (None: triton_only)",
    "v_rel_inf_kms": "Triton-relative v_inf at the frame switch, km/s",
    "entry_speed_kms": "inertial (Triton-centred) speed at the Triton interface, km/s",
    "entry_speed_rel_kms": "atmosphere-relative speed at the Triton interface, km/s",
    "direct_entry_speed_kms": "entry speed of a direct tangent intercept at the same v_inf, km/s",
    "entry_fpa_rel_deg": "atmosphere-relative flight-path angle at the Triton interface, deg",
    "corridor_width_deg": "guess entry corridor width, deg",
    "release_alt_km": "altitude at ballute release, km",
    "release_t_s": "time from interface to release, s",
    "min_alt_km": "minimum altitude of the pass, km",
    "peak_q_conv_w_cm2": "peak Sutton-Graves N2 continuum heat rate on the ballute, W/cm^2",
    "peak_q_fm_w_cm2": "peak free-molecular heat rate, W/cm^2",
    "peak_q_bridged_w_cm2": "peak bridged convective heat rate, W/cm^2",
    "peak_q_rad_ts_w_cm2": "peak Tauber-Sutton radiative upper bound, W/cm^2",
    "peak_q_heat_w_cm2": "peak heat rate used by limits/objective, W/cm^2",
    "heat_load_j_cm2": "ballute heat load, J/cm^2",
    "peak_t_film_k": "peak film radiative-equilibrium temperature, K",
    "peak_g": "peak sensed deceleration (ballute or probe), Earth g",
    "peak_q_dyn_pa": "peak dynamic pressure, Pa",
    "burn_dv_ms": "periapsis-raise burn, m/s",
    "tcm_ms": "post-flyby TCM left in the solution, m/s",
    "final_rp_alt_km": "final osculating periapsis altitude after the burn, km",
    "final_ra_km": "final osculating apoapsis radius after the burn, km",
    "final_i_t_deg": "final inclination to Triton's orbit plane, deg",
    "stability_min_alt_km": "minimum altitude over the stability propagation, km",
    "stability_max_r_km": "maximum distance from Triton over the stability propagation, km",
    "stability_passed": "stability propagation within the floor and the apoapsis limit",
    "sim_passed": "simulate check within sim_tol and sim_rp_tol_km",
    "active_constraints": "inequality constraints on their bound at the solution, joined by ' | '",
    "active_box_bounds": "numerical box bounds active at the solution (unstated constraints)",
    "warnings": "every summary warning, joined by ' | '",
    "start": "warm start or fresh guess (sweep/trade only)",
}


# ------------------------------------------------------------------ readers
def _states(tp, phase, sim=False):
    """Return (t, y) of a phase from the collocated or simulated timeseries, or (None, None)."""
    get = tp.sim_ts if sim else tp.ts
    names = tp._states(phase)
    cols = [get(phase, n) for n in names]
    if any(c is None for c in cols):
        return None, None
    return get(phase, "time"), np.stack(cols, axis=1)


def _elements(y, tr):
    """Return the osculating Triton elements of one TC state as floats."""
    el = tf.elements(y[:3], y[3:6], tr)
    out = {k: float(v[0]) for k, v in el.items()}
    out["rp_alt_km"] = out["rp"] - system.R_TRITON
    out["period_h"] = (2 * math.pi * math.sqrt(out["a"] ** 3 / system.MU_TRITON) / 3600.0
                       if out["energy"] < 0 else None)
    if not math.isfinite(out["ra"]):
        out["ra"] = None
    return out


def _tc_point(y, t, tp):
    """Return altitude, speeds, flight-path angles and latitude (on Triton's plane) of a TC state."""
    tr = tp.epoch.triton
    r, v = y[:3], y[3:6]
    rm, vm = float(np.linalg.norm(r)), float(np.linalg.norm(v))
    vrel = dynamics.relative_velocity(r, v, tf.omega_vec(tr, tp.cfg.atm_rotation))
    vr = float(np.linalg.norm(vrel))
    return dict(t_s=float(t), utc=tp.epoch.iso(t), altitude_km=rm - system.R_TRITON,
                speed_inertial_kms=vm, speed_rel_atm_kms=vr,
                fpa_inertial_deg=math.degrees(math.asin(r @ v / (rm * vm))),
                fpa_rel_deg=math.degrees(math.asin(r @ vrel / (rm * vr))),
                latitude_deg=math.degrees(math.asin(r @ np.asarray(tr.h_hat) / rm)))


def ring_crossings(tp, sim=True):
    """Return every equatorial-plane crossing of the Neptune phases with its radius and status.

    Args:
        tp (problem.TritonProblem): Solved full-scope problem.
        sim (bool, optional): Use the simulated trajectory when available. Defaults to True.

    Returns:
        dict: ``source``, ``crossings`` (phase, t_s, r_km, clear) and ``passed``.
    """
    r_in, r_out = tp.cfg.ring_keepout_km
    out, source = [], "simulated"
    for ph in _neptune_phases(tp):
        t, y = _states(tp, ph, sim=sim)
        if t is None:
            t, y = _states(tp, ph)
            source = "collocated"
        t, idx = np.unique(t, return_index=True)
        y = y[idx]
        z = y[:, 2]
        for i in np.where(np.sign(z[:-1]) * np.sign(z[1:]) < 0)[0]:
            f = z[i] / (z[i] - z[i + 1])
            pos = y[i, :3] + f * (y[i + 1, :3] - y[i, :3])
            r = float(np.linalg.norm(pos))
            out.append(dict(phase=ph, t_s=float(t[i] + f * (t[i + 1] - t[i])), r_km=r,
                            clear=bool(r <= r_in or r >= r_out)))
    return dict(source=source, keepout_km=[r_in, r_out], crossings=out,
                passed=all(c["clear"] for c in out))


def _neptune_phases(tp):
    """Return the Neptune-frame phase names of a problem."""
    return [p for p in tp.phase_names if p in ("approach", "flyby_in", "flyby_out", "cruise")]


def gram_drag(tp):
    """Return the drag Δv (m/s) of the inflated ballute across the Neptune flyby (GRAM density)."""
    cfg = tp.cfg
    beta = (cfg.ballute.envelope_mass_kg + cfg.probe.mass_kg) / (
        cfg.ballute.cd_fm_effective * cfg.ballute.area_m2)
    atm = atmosphere.default()
    dv, rho_max = 0.0, 0.0
    for ph in ("flyby_in", "flyby_out"):
        t, y = _states(tp, ph, sim=True)
        if t is None:
            t, y = _states(tp, ph)
        t, idx = np.unique(t, return_index=True)
        y = y[idx]
        rho = atm.density(np.linalg.norm(y[:, :3], axis=1) - system.R_NEPTUNE)
        vrel = np.linalg.norm(dynamics.relative_velocity(y[:, :3], y[:, 3:]), axis=1) * 1e3
        dv += float(np.trapezoid(0.5 * rho * vrel ** 2 / beta, t))
        rho_max = max(rho_max, float(rho.max()))
    return dict(beta_free_molecular_kg_m2=beta, dv_ms=dv, peak_density_kg_m3=rho_max)


def stability(tp, y0, t0):
    """Propagate the final state for ``stability_days`` with the differential tide.

    Args:
        tp (problem.TritonProblem): Solved problem (config, ephemeris).
        y0 (np.ndarray): TC state at the end of the final phase (6,).
        t0 (float): Its time, s.

    Returns:
        dict: Extremes of altitude, distance, osculating rp/ra and i_T, the
        sampled series (for plots) and ``passed``.
    """
    cfg, tr = tp.cfg, tp.epoch.triton
    t_end = t0 + cfg.stability_days * 86400.0

    def crash(t, y, *_):
        return math.sqrt(y[:3] @ y[:3]) - system.R_TRITON
    crash.terminal = True
    sol = propagate(guess_mod.f_tc_tidal, t0, y0, t_end, (tr,), events=[crash],
                                 rtol=1e-10, atol=1e-6)
    tt = np.arange(t0, sol.t[-1], STABILITY_SAMPLE_S)
    yy = sol.sol(tt).T
    d = np.linalg.norm(yy[:, :3], axis=1)
    el = tf.elements(yy[:, :3], yy[:, 3:], tr)
    floor, ra_max = cfg.rp_final, cfg.ra_max_km(tr)
    ra = np.where(np.isfinite(el["ra"]), el["ra"], np.nan)
    passed = bool(len(sol.t_events[0]) == 0 and d.min() >= floor - 1e-6 and d.max() <= ra_max)
    return dict(days=cfg.stability_days, impacted=len(sol.t_events[0]) > 0,
                min_alt_km=float(d.min() - system.R_TRITON), max_r_km=float(d.max()),
                min_rp_osc_alt_km=float(el["rp"].min() - system.R_TRITON),
                max_ra_osc_km=float(np.nanmax(ra)),
                i_t_range_deg=[float(el["i_t_deg"].min()), float(el["i_t_deg"].max())],
                e_range=[float(el["e"].min()), float(el["e"].max())],
                floor_km=floor, ra_limit_km=ra_max, passed=passed,
                series=dict(t_days=(tt - t0) / 86400.0, r_km=d, rp_km=el["rp"], ra_km=ra,
                            i_t_deg=el["i_t_deg"]))


def simulation_check(tp):
    """Compare each phase's collocated final state with its re-integration and the final orbit."""
    out = {"available": True, "phases": {}}
    for ph in tp.phase_names:
        _, y = _states(tp, ph)
        _, ys = _states(tp, ph, sim=True)
        if ys is None:
            return {"available": False}
        out["phases"][ph] = dict(dr_km=float(np.linalg.norm(y[-1, :3] - ys[-1, :3])),
                                 dv_kms=float(np.linalg.norm(y[-1, 3:] - ys[-1, 3:])))
    tr = tp.epoch.triton
    oc, os_ = _elements(y[-1], tr), _elements(ys[-1], tr)
    ra_err = abs(os_["ra"] - oc["ra"]) / oc["ra"] if oc["ra"] and os_["ra"] else float("inf")
    rp_err = abs(os_["rp"] - oc["rp"])
    out.update(final_collocated=oc, final_simulated=os_, ra_rel_error=ra_err, rp_error_km=rp_err,
               passed=bool(ra_err <= tp.cfg.sim_tol and rp_err <= tp.cfg.sim_rp_tol_km))
    return out


def _pass(tp, phase):
    """Return the timeseries diagnostics of a ballute or probe phase."""
    names = ("time", "alt", "q_conv", "q_fm", "q_bridged", "q_rad", "q_heat", "t_film", "g_load",
             "q_dyn", "kn", "cd", "beta", "v_rel")
    d = {n: tp.ts(phase, n) for n in names}
    if phase == "ballute":
        d["Q"] = tp.ts(phase, "Q")
    return d


def summarize(tp, guess=None, solve_info=None, atm_record=None, warnings=()):
    """Return the JSON-ready summary of a solved problem.

    Args:
        tp (problem.TritonProblem): Solved (and simulated) problem.
        guess (guess.TritonGuess, optional): Guess it was built from. Defaults to None.
        solve_info (dict, optional): Info returned by ``case.solve``. Defaults to None.
        atm_record (dict, optional): Atmosphere record. Defaults to None.
        warnings (tuple, optional): Extra warnings. Defaults to ().

    Returns:
        dict: The summary written to ``solution.json``; the private key
        ``_stability_series`` (sampled stability propagation, for plots) is not saved.
    """
    cfg, ep, p, tr = tp.cfg, tp.epoch, tp.p, tp.epoch.triton
    solve_info = dict(solve_info or {})
    warns = list(warnings) + cfg.warnings()
    arr = {n: float(p.get_val(f"arrival_dv.{n}")[0])
           for n in ("vinf", "ra", "dec", "b_t", "b_r", "t_peri")}
    free = [n.split(".")[-1] for n in p.model.get_design_vars(get_sizes=False)
            if n.startswith("arrival_dv.")]
    arrival = dict(frame="NEQ (Neptune)" if tp.full else "TC (Triton, at frame_switch_km)",
                   vinf_kms=arr["vinf"], ra_deg=math.degrees(arr["ra"]),
                   dec_deg=math.degrees(arr["dec"]), b_t_km=arr["b_t"], b_r_km=arr["b_r"],
                   b_km=math.hypot(arr["b_t"], arr["b_r"]), t_peri_s=arr["t_peri"],
                   t_peri_utc=ep.iso(arr["t_peri"]), optimised_parameters=free)
    if tp.full:
        s_hat = dynamics.unit_from_radec(arr["ra"], arr["dec"])
        arrival["vinf_vec_eclip_kms"] = list(np.linalg.solve(ep.neq_from_eclip, s_hat)
                                             * arr["vinf"])

    neptune = None
    if tp.full:
        t, y = _states(tp, "flyby_in")
        rp = float(np.linalg.norm(y[-1, :3]))
        rings = ring_crossings(tp)
        neptune = dict(rp_km=rp, rp_rn=rp / system.R_NEPTUNE, altitude_km=rp - system.R_NEPTUNE,
                       t_s=float(t[-1]), utc=ep.iso(t[-1]),
                       latitude_deg=math.degrees(math.asin(y[-1, 2] / rp)),
                       speed_inertial_kms=float(np.linalg.norm(y[-1, 3:])), rings=rings,
                       gram_drag=gram_drag(tp))
        if not rings["passed"]:
            warns.append("ring post-check FAILED: a ring-plane crossing lies in the keep-out band")
        for kind in ("tcm", "neptune_burn"):
            if kind in tp.impulses:
                dv = float(np.linalg.norm(tp.impulse_dv(kind)))
                neptune[f"{kind}_ms"] = dv * 1e3
                if dv > 1e-4:
                    warns.append(f"{kind} of {dv * 1e3:.1f} m/s remains in the solution")

    # Triton encounter
    t, y = _states(tp, "triton_approach")
    bp = dynamics.state_to_bplane(y[0, :3], y[0, 3:6], t[0], mu=system.MU_TRITON)
    tb, yb = _states(tp, "ballute")
    entry = _tc_point(yb[0], tb[0], tp)
    pc = (guess.info.get("patched_conic") if guess else None) or {}
    direct = guess_mod.direct_intercept(cfg.vinf_kms, tr, cfg.r_interface)
    encounter = dict(v_rel_inf_kms=bp["vinf"], b_km=math.hypot(bp["b_t"], bp["b_r"]),
                     sense=cfg.guess_encounter, entry=entry,
                     patched_conic=pc,
                     direct_intercept=dict(direct, penalty_kms=entry["speed_inertial_kms"]
                                           - direct["v_entry"]))

    # passes
    bal = _pass(tp, "ballute")
    i_pk = int(np.argmax(bal["q_heat"]))
    heating = dict(
        conv_model=cfg.conv_model, bridging=cfg.bridging, rad_in_limits=cfg.rad_in_limits,
        peak_q_conv_w_cm2=float(bal["q_conv"].max()), peak_q_fm_w_cm2=float(bal["q_fm"].max()),
        peak_q_bridged_w_cm2=float(bal["q_bridged"].max()),
        peak_q_rad_tauber_sutton_w_cm2=float(bal["q_rad"].max()),
        peak_q_heat_w_cm2=float(bal["q_heat"].max()),
        heat_load_j_cm2=float(bal["Q"][-1]), peak_t_film_k=float(bal["t_film"].max()),
        t_film_limit_k=cfg.ballute.t_film_max_k,
        at_peak=dict(alt_km=float(bal["alt"][i_pk]), kn=float(bal["kn"][i_pk]),
                     cd=float(bal["cd"][i_pk]), v_rel_kms=float(bal["v_rel"][i_pk]),
                     q_conv=float(bal["q_conv"][i_pk]), q_fm=float(bal["q_fm"][i_pk])))
    release = None
    if "probe" in tp.phase_names:
        release = dict(_tc_point(yb[-1], tb[-1], tp), time_from_entry_s=float(tb[-1] - tb[0]))
        release["osculating"] = _elements(yb[-1], tr)
    loads = dict(peak_g_ballute=float(bal["g_load"].max()),
                 peak_q_dyn_pa=float(bal["q_dyn"].max()),
                 min_alt_ballute_km=float(bal["alt"].min()),
                 ballute_duration_s=float(tb[-1] - tb[0]))
    exit_phase = "probe" if "probe" in tp.phase_names else "ballute"
    probe = None
    if exit_phase == "probe":
        pr = _pass(tp, "probe")
        probe = dict(peak_g=float(pr["g_load"].max()), min_alt_km=float(pr["alt"].min()),
                     peak_q_conv_w_cm2=float(pr["q_conv"].max()),
                     peak_q_bridged_w_cm2=float(pr["q_bridged"].max()),
                     duration_s=float(pr["time"][-1] - pr["time"][0]))
        loads["peak_g"] = max(loads["peak_g_ballute"], probe["peak_g"])
        loads["duration_s"] = loads["ballute_duration_s"] + probe["duration_s"]
        loads["min_alt_km"] = min(loads["min_alt_ballute_km"], probe["min_alt_km"])
    else:
        loads["peak_g"], loads["min_alt_km"] = loads["peak_g_ballute"], loads["min_alt_ballute_km"]
        loads["duration_s"] = loads["ballute_duration_s"]
    tx, yx = _states(tp, exit_phase)
    capture_orbit = dict(_tc_point(yx[-1], tx[-1], tp), osculating=_elements(yx[-1], tr))

    tc, yc = _states(tp, "capture_coast")
    dv = tp.impulse_dv("burn")
    burn = dict(dv_kms=list(dv), dv_ms=float(np.linalg.norm(dv)) * 1e3, t_s=float(tc[-1]),
                utc=ep.iso(tc[-1]), altitude_km=float(np.linalg.norm(yc[-1, :3]))
                - system.R_TRITON,
                coast_duration_h=float(tc[-1] - tc[0]) / 3600.0)
    tfn, yf = _states(tp, "final")
    df = np.linalg.norm(yf[:, :3], axis=1)
    final = dict(after_burn=_elements(yf[0], tr), end=_elements(yf[-1], tr),
                 min_alt_km=float(df.min() - system.R_TRITON), max_r_km=float(df.max()),
                 ra_limit_km=cfg.ra_max_km(tr), rp_floor_km=cfg.rp_final,
                 sense=cfg.final_sense, i_t_band_deg=list(cfg.inclination_bounds_deg()),
                 hill_radius_km=tf.hill_radius(tr))
    stab = stability(tp, yf[-1], tfn[-1])
    check = simulation_check(tp)

    solver = io.solver_record(solve_info, p)
    if not stab["passed"]:
        warns.append(f"stability FAILED over {cfg.stability_days:g} d: altitude "
                     f"{stab['min_alt_km']:.0f} km (floor {cfg.rp_final_alt_km:g}), max distance "
                     f"{stab['max_r_km']:.0f} km (limit {stab['ra_limit_km']:.0f})")
    if check.get("available") and not check["passed"]:
        warns.append(f"simulate check FAILED: ra error {check['ra_rel_error']:.2e} (tol "
                     f"{cfg.sim_tol}), rp error {check['rp_error_km']:.2f} km (tol "
                     f"{cfg.sim_rp_tol_km})")
    if not solve_info.get("success"):
        warns.append(f"final stage did not converge: {solve_info.get('ipopt_text')}")
    if solve_info.get("continuation_complete") is False:
        warns.append("continuation to the fixed v-inf direction did not complete")
    if heating["peak_q_rad_tauber_sutton_w_cm2"] > heating["peak_q_bridged_w_cm2"]:
        ratio = heating["peak_q_rad_tauber_sutton_w_cm2"] / heating["peak_q_bridged_w_cm2"]
        warns.append(f"Tauber-Sutton radiative bound is {ratio:.1f}x the convective peak "
                     f"({'constrained' if cfg.rad_in_limits else 'not constrained'}; LAURA "
                     "solutions at Pluto found N2 radiation minimal)")
    if heating["peak_t_film_k"] > cfg.ballute.t_film_max_k * 0.98:
        warns.append(f"film temperature {heating['peak_t_film_k']:.0f} K is at its limit")
    gr = io.guess_record(guess)
    cor = gr["corridor"]
    if cor and cor["width_deg"] < 0.3:
        warns.append(f"entry corridor is only {cor['width_deg']:.2f} deg wide")
    if solver["active_box_bounds"]:
        warns.append(f"{io.BOX_BOUND_TEXT}: {', '.join(solver['active_box_bounds'])}")

    s = dict(
        mission="triton", tag=cfg.tag, scope=tp.scope, config=cfg.record(),
        ballute=asdict(cfg.ballute), probe=asdict(cfg.probe),
        epoch=dict(label=ep.label, et=ep.et_ref), constants=constants_record(tr),
        atmosphere=atm_record, triton_orbit=io.triton_record(ep), arrival=arrival,
        neptune=neptune, triton_encounter=encounter, heating=heating, loads=loads,
        release=release, probe_pass=probe, capture_orbit=capture_orbit, burn=burn,
        final_orbit=final, stability={k: v for k, v in stab.items() if k != "series"},
        objective=dict(name=cfg.objective, value=float(p.get_val("objective.obj")[0])),
        solver=solver, guess=gr, simulation_check=check, assumptions=cfg.assumptions(),
        active_box_bounds=solver["active_box_bounds"], warnings=warns, notes=NOTES,
        _stability_series=stab["series"])
    s["active_constraints"] = active_constraints(s, cfg, ep)
    return s


def active_constraints(s, cfg, epoch):
    """Return the inequality constraints of a summary that sit on their bound.

    Args:
        s (dict): Summary under construction (``loads``, ``heating``,
            ``final_orbit``, ``neptune``, ``burn``, ``arrival``, ``config``).
        cfg (config.Config): Inputs.
        epoch (system.Epoch): Reference epoch (Triton's period and Hill radius).

    Returns:
        list: Text per active constraint (``core.io.active_constraints``).
    """
    tr = epoch.triton
    ld, h, f, n = s["loads"], s["heating"], s["final_orbit"], s["neptune"] or {}
    i_lo, i_hi = cfg.inclination_bounds_deg()
    ab, en = f["after_burn"], f["end"]
    rp_lo, rp_hi = cfg.rp_band_km
    window = cfg.t_peri_window_periods * tr.period
    checks = [
        ("Neptune periapsis", n.get("rp_km"), rp_lo, "min", "km", "rp_min_rn"),
        ("Neptune periapsis", n.get("rp_km"), rp_hi, "max", "km", "rp_max_rn"),
        ("minimum pass altitude", ld["min_alt_km"], cfg.h_min_km, "min", "km",
         "h_min_km [margin]"),
        ("peak g", ld["peak_g"], cfg.g_max, "max", "g", "g_max"),
        ("peak ballute q_heat", h["peak_q_heat_w_cm2"], cfg.q_max_w_cm2, "max", "W/cm^2",
         "q_max_w_cm2"),
        ("peak film T", h["peak_t_film_k"], cfg.ballute.t_film_max_k, "max", "K",
         "ballute.t_film_max_k"),
        ("final osculating rp", min(ab["rp"], en["rp"]), cfg.rp_design, "min", "km",
         "R_T + rp_final_alt_km + final_rp_margin_km [margin]"),
        ("final osculating ra", max(ab["ra"] or 0.0, en["ra"] or 0.0), cfg.ra_design_km(tr),
         "max", "km", "ra_max_hill x (1 - final_ra_margin) [margin]"),
        ("final i_T", ab["i_t_deg"], i_lo if i_lo > 0.0 else None, "min", "deg",
         "i_retro_min_deg [margin]"),
        ("final i_T", ab["i_t_deg"], i_hi if i_hi < 180.0 else None, "max", "deg",
         "i_pro_max_deg [margin]"),
        ("burn", s["burn"]["dv_ms"], cfg.burn_dv_max_kms * 1e3, "max", "m/s", "burn_dv_max_kms"),
        ("Neptune burn", n.get("neptune_burn_ms"),
         cfg.neptune_dv_max_kms * 1e3 if cfg.neptune_dv_max_kms > 0 else None, "max", "m/s",
         "neptune_dv_max_kms"),
        ("TCM", n.get("tcm_ms"), cfg.tcm_max_kms * 1e3 if cfg.tcm_max_kms > 0 else None, "max",
         "m/s", "tcm_max_kms"),
    ]
    if s["config"]["opt_t_peri"] and s["neptune"]:
        t_peri = s["arrival"]["t_peri_s"]
        checks += [("t_peri", t_peri, -window, "min", "s", "t_peri_window_periods"),
                   ("t_peri", t_peri, window, "max", "s", "t_peri_window_periods")]
    return io.active_constraints(checks)


def acceptable(summary):
    """Return True for a converged case that passes every post-solve check.

    Args:
        summary (dict): Summary from ``summarize`` (or its saved ``solution.json``).

    Returns:
        bool: IPOPT success on the final stage, the simulate check passed (or
        not available), the ring-plane check passed (full scope only) and the
        ``stability_days`` propagation passed.
    """
    check = summary["simulation_check"]
    n = summary.get("neptune")
    return (bool(summary["solver"].get("success"))
            and (not check.get("available") or bool(check["passed"]))
            and (n is None or bool(n["rings"]["passed"]))
            and bool(summary["stability"]["passed"]))


def constants_record(triton):
    """Return the constants used, with their sources (Neptune system plus Triton-specific)."""
    d = io.constants_record()
    d.update(r_hill_triton_km=tf.hill_radius(triton), omega_triton_rads=triton.n,
             triton_speed_kms=triton.a * triton.n, n2_molecular_mass_kg=ballute.M_N2,
             n2_diameter_m=ballute.D_N2, tauber_sutton_c=ballute.TS_C)
    return d


def save(summary, out_dir):
    """Write ``solution.json`` (private ``_`` keys left out) into ``out_dir`` and return its path."""
    return io.save(summary, out_dir)


def print_summary(s, stream=print):
    """Print the summary sections, the checks, the assumptions and margins and every warning."""
    cfg, h, ld, f, sv = s["config"], s["heating"], s["loads"], s["final_orbit"], s["solver"]
    b = s["ballute"]
    # header
    stream(f"triton | scope {s['scope']} | v_inf {cfg['vinf_kms']:.2f} km/s | ballute {b['name']} "
           f"(D {b['diameter_m']:g} m, {b['envelope_mass_kg']:g} + {s['probe']['mass_kg']:g} kg) | "
           f"objective {s['objective']['name']} = {s['objective']['value']:.4g}")
    io.print_solver(sv, stream)
    # arrival
    a = s["arrival"]
    stream(f"  arrival [{a['frame']}]: v_inf {a['vinf_kms']:.4f} km/s, RA {a['ra_deg']:.2f} deg, "
           f"Dec {a['dec_deg']:.2f} deg, B {a['b_km']:.0f} km, t_peri {a['t_peri_utc']}; "
           f"optimised {', '.join(a['optimised_parameters'])} (opt_direction="
           f"{cfg['opt_direction']}, opt_t_peri={cfg['opt_t_peri']})")
    # flyby
    n = s["neptune"]
    if n:
        rings = ", ".join(f"{c['r_km']:.0f} km {'ok' if c['clear'] else 'IN BAND'}"
                          for c in n["rings"]["crossings"])
        stream(f"  Neptune flyby: periapsis {n['rp_rn']:.4f} R_N ({n['altitude_km']:.0f} km alt), "
               f"lat {n['latitude_deg']:.2f} deg, {n['utc']}; ring crossings [{rings}] "
               f"({'PASS' if n['rings']['passed'] else 'FAIL'}); GRAM drag dv "
               f"{n['gram_drag']['dv_ms']:.2e} m/s (peak rho "
               f"{n['gram_drag']['peak_density_kg_m3']:.1e} kg/m^3)")
    else:
        stream("  Neptune flyby: none (scope triton_only)")
    # entry
    e = s["triton_encounter"]
    en, di = e["entry"], e["direct_intercept"]
    stream(f"  entry: v_rel_inf {e['v_rel_inf_kms']:.3f} km/s; v {en['speed_inertial_kms']:.3f} "
           f"km/s inertial / {en['speed_rel_atm_kms']:.3f} km/s vs atmosphere, FPA "
           f"{en['fpa_inertial_deg']:.3f} / {en['fpa_rel_deg']:.3f} deg, lat "
           f"{en['latitude_deg']:.1f} deg")
    stream(f"  direct-intercept comparison: a direct tangent intercept (no close Neptune pass) "
           f"would enter at {di['v_entry']:.3f} km/s (v_rel {di['v_rel']:.3f} km/s); the close "
           f"pass costs {di['penalty_kms']:+.3f} km/s of entry speed")
    cor = s["guess"]["corridor"]
    if cor:
        stream(f"  corridor (guess, scipy): {cor['gamma_shallow_deg']:.3f} .. "
               f"{cor['gamma_steep_deg']:.3f} deg ({cor['width_deg']:.3f} deg) at v_E "
               f"{cor['v_entry_kms']:.3f} km/s")
    r = s["release"]
    if r:
        stream(f"  release: {r['time_from_entry_s']:.1f} s after entry, alt {r['altitude_km']:.1f}"
               f" km, v {r['speed_inertial_kms']:.3f} km/s; osculating ra "
               f"{r['osculating']['ra'] or 0:.0f} km, rp alt {r['osculating']['rp_alt_km']:.1f} km")
    # loads
    stream(f"  loads: peak g {ld['peak_g']:.2f}, min alt {ld['min_alt_km']:.1f} km, peak q_dyn "
           f"{ld['peak_q_dyn_pa']:.1f} Pa, pass {ld['duration_s']:.0f} s")
    # heating
    stream(f"  heating [W/cm^2] (limits and objective use q_heat = bridged"
           f"{' + radiative' if h['rad_in_limits'] else ''}): continuum "
           f"{h['peak_q_conv_w_cm2']:.3f}, free-molecular {h['peak_q_fm_w_cm2']:.3f}, bridged "
           f"{h['peak_q_bridged_w_cm2']:.3f}, Tauber-Sutton bound "
           f"{h['peak_q_rad_tauber_sutton_w_cm2']:.3g}; peak q_heat {h['peak_q_heat_w_cm2']:.3f}"
           f" at {h['at_peak']['alt_km']:.1f} km (Kn {h['at_peak']['kn']:.1e})")
    stream(f"  heat load [J/cm^2]: {h['heat_load_j_cm2']:.1f}; film T {h['peak_t_film_k']:.0f} K "
           f"(limit {h['t_film_limit_k']:.0f} K)")
    # final orbit
    co = s["capture_orbit"]["osculating"]
    stream(f"  capture orbit at exit: ra {co['ra'] or float('nan'):.0f} km, rp alt "
           f"{co['rp_alt_km']:.1f} km, i_T {co['i_t_deg']:.2f} deg")
    bu = s["burn"]
    stream(f"  burn: {bu['dv_ms']:.1f} m/s at {bu['altitude_km']:.0f} km alt, "
           f"{bu['coast_duration_h']:.2f} h after exit")
    ab = f["after_burn"]
    stream(f"  final orbit (after the burn): rp alt {ab['rp_alt_km']:.1f} km, ra {ab['ra']:.0f} km "
           f"(limit {f['ra_limit_km']:.0f} km), e {ab['e']:.3f}, i_T {ab['i_t_deg']:.2f} deg, "
           f"period {ab['period_h']:.2f} h; over {cfg['final_coast_h']:g} h min alt "
           f"{f['min_alt_km']:.1f} km, max r {f['max_r_km']:.0f} km")
    st = s["stability"]
    stream(f"  stability {st['days']:g} d: min alt {st['min_alt_km']:.1f} km (floor "
           f"{cfg['rp_final_alt_km']:g} km), max r {st['max_r_km']:.0f} km (limit "
           f"{st['ra_limit_km']:.0f} km), i_T {st['i_t_range_deg'][0]:.1f}-"
           f"{st['i_t_range_deg'][1]:.1f} deg, e {st['e_range'][0]:.3f}-{st['e_range'][1]:.3f}")
    # checks, assumptions and margins, warnings
    c = s["simulation_check"]
    sim = (f"{'PASS' if c['passed'] else 'FAIL'} (ra error {c['ra_rel_error']:.1e}, rp error "
           f"{c['rp_error_km']:.2f} km)" if c.get("available") else "not available")
    io.print_tail(s, {"converged": bool(sv.get("success")), "simulate": sim,
                      "rings": n["rings"]["passed"] if n else "-",
                      "stability": "PASS" if st["passed"] else "FAIL"}, stream)


def table_row(s):
    """Return the flat sweep/trade row of a summary (keys in ``COLUMNS``)."""
    cfg, b, h, ld, f = s["config"], s["ballute"], s["heating"], s["loads"], s["final_orbit"]
    n, e, r, st, c = (s["neptune"] or {}), s["triton_encounter"], s["release"] or {}, \
        s["stability"], s["simulation_check"]
    cor = s["guess"]["corridor"] or {}
    return dict(
        scope=s["scope"], vinf_kms=cfg["vinf_kms"], encounter=cfg["guess_encounter"],
        ballute=b["name"], diameter_m=b["diameter_m"], envelope_mass_kg=b["envelope_mass_kg"],
        probe_mass_kg=s["probe"]["mass_kg"],
        beta_ballute_kg_m2=cfg["ballute_derived"]["beta_continuum_kg_m2"],
        converged=bool(s["solver"].get("success")), ipopt_status=s["solver"].get("ipopt_text"),
        ipopt_iterations=s["solver"].get("ipopt_iterations"),
        neptune_rp_rn=n.get("rp_rn"), ring_check=n["rings"]["passed"] if n else None,
        v_rel_inf_kms=e["v_rel_inf_kms"], entry_speed_kms=e["entry"]["speed_inertial_kms"],
        entry_speed_rel_kms=e["entry"]["speed_rel_atm_kms"],
        direct_entry_speed_kms=e["direct_intercept"]["v_entry"],
        entry_fpa_rel_deg=e["entry"]["fpa_rel_deg"], corridor_width_deg=cor.get("width_deg"),
        release_alt_km=r.get("altitude_km"), release_t_s=r.get("time_from_entry_s"),
        min_alt_km=ld["min_alt_km"], peak_q_conv_w_cm2=h["peak_q_conv_w_cm2"],
        peak_q_fm_w_cm2=h["peak_q_fm_w_cm2"], peak_q_bridged_w_cm2=h["peak_q_bridged_w_cm2"],
        peak_q_rad_ts_w_cm2=h["peak_q_rad_tauber_sutton_w_cm2"],
        peak_q_heat_w_cm2=h["peak_q_heat_w_cm2"], heat_load_j_cm2=h["heat_load_j_cm2"],
        peak_t_film_k=h["peak_t_film_k"], peak_g=ld["peak_g"], peak_q_dyn_pa=ld["peak_q_dyn_pa"],
        burn_dv_ms=s["burn"]["dv_ms"], tcm_ms=n.get("tcm_ms"),
        final_rp_alt_km=f["after_burn"]["rp_alt_km"], final_ra_km=f["after_burn"]["ra"],
        final_i_t_deg=f["after_burn"]["i_t_deg"], stability_min_alt_km=st["min_alt_km"],
        stability_max_r_km=st["max_r_km"], stability_passed=st["passed"],
        sim_passed=c.get("passed"), active_constraints=" | ".join(s["active_constraints"]),
        active_box_bounds=" | ".join(s["active_box_bounds"]), warnings=" | ".join(s["warnings"]))


def failed_row(cfg, reason):
    """Return the table row of a case that produced no solution."""
    b = cfg.ballute
    return dict(scope=cfg.scope, vinf_kms=cfg.vinf_kms, encounter=cfg.guess_encounter,
                ballute=b.name, diameter_m=b.diameter_m, envelope_mass_kg=b.envelope_mass_kg,
                probe_mass_kg=cfg.probe.mass_kg, converged=False, ipopt_status=reason)


def write_csv(rows, path):
    """Write table rows to CSV in ``COLUMNS`` order and return the path."""
    return io.write_csv(rows, path, COLUMNS)


print_table = io.print_table
