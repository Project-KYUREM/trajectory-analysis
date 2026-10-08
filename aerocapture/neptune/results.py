"""Read a solved ``AeroProblem`` back into plain numbers, check it, save it and tabulate it.

``summarize`` builds the dict written to ``solution.json``: inputs (opt flags
resolved), constants, arrival B-plane, Triton encounter, entry/exit
conditions, loads, heating (coupled and uncoupled radiation), final orbit,
solver stages, the simulation check, the margins and assumptions in effect
with the constraints active at the solution, and every warning.
``print_summary`` prints it in the section order shared with the Triton
mission: header, solver, arrival, flyby, entry/exit, loads, heating, final
orbit, checks, assumptions and margins, warnings. ``table_row`` flattens a
summary into one sweep/trade CSV row (columns in ``COLUMNS``). JSON, CSV and
table writers are in ``core.io``.
"""
import math
from dataclasses import asdict

import numpy as np

from aerocapture.core import atmosphere
from aerocapture.core import dynamics
from aerocapture.core import heating
from aerocapture.core import io
from aerocapture.core import system

STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz")
TRITON_USED_FACTOR = 10.0   # a flyby counts as used within this x the minimum altitude
NOTES = [
    "Solar perturbations are not modelled; over the first (long) orbit to an apoapsis near "
    "0.1 R_Hill they are ~0.1% of Neptune's pull and grow as ra^3.",
    "Heating is cold-wall, non-ablating stagnation-point engineering heating (Vision Neptune "
    "fits); no reradiation, hot-wall or blowing credit (conservative by ~10-20 %).",
    "Triton's ephemeris is a circular fit to nep097 about the epoch.",
    "Vehicle presets hold placeholder numbers, not sourced designs.",
]

COLUMNS = {
    "mode": "Triton mode",
    "vinf_kms": "hyperbolic excess speed, km/s",
    "vehicle": "vehicle name",
    "beta_kg_m2": "ballistic coefficient, kg/m^2",
    "ld": "lift-to-drag ratio",
    "rn_m": "nose radius, m",
    "converged": "IPOPT reported success",
    "ipopt_status": "IPOPT exit text (or the failure reason)",
    "ipopt_iterations": "IPOPT iterations of the last stage",
    "triton_used": "Triton closest approach within 10x the minimum altitude (None: mode none)",
    "triton_alt_km": "Triton closest-approach altitude, km",
    "triton_dv_ms": "Triton flyby velocity change, m/s",
    "triton_dE_km2s2": "Neptune-frame energy change across the Triton phase, km^2/s^2",
    "entry_speed_kms": "inertial speed at the entry interface, km/s",
    "entry_speed_rel_kms": "atmosphere-relative speed at the entry interface, km/s",
    "entry_fpa_rel_deg": "atmosphere-relative flight-path angle at entry, deg",
    "peak_g": "peak sensed deceleration, Earth g",
    "min_alt_km": "minimum altitude, km",
    "peak_q_conv_w_cm2": "peak convective heat rate, W/cm^2",
    "peak_q_rad_coupled_w_cm2": "peak radiative heat rate, Goulard-coupled, W/cm^2",
    "peak_q_rad_uncoupled_w_cm2": "peak radiative heat rate, uncoupled, W/cm^2",
    "peak_q_total_w_cm2": "peak total heat rate with the selected coupling, W/cm^2",
    "peak_q_total_coupled_w_cm2": "peak total heat rate, coupled radiation, W/cm^2",
    "peak_q_total_uncoupled_w_cm2": "peak total heat rate, uncoupled radiation, W/cm^2",
    "heat_load_j_cm2": "heat load with the selected coupling, J/cm^2",
    "peak_t_radeq_k": "peak radiative-equilibrium wall temperature, K",
    "final_ra_km": "final osculating apoapsis radius, km",
    "final_rp_km": "final osculating periapsis radius, km",
    "final_inc_deg": "final inclination to Neptune's equator, deg",
    "capture_margin_km2s2": "energy below the target-apoapsis bound, km^2/s^2 (>= 0: captured)",
    "dv_raise_ms": "two-body periapsis-raise burn at apoapsis to rp_raise_km, m/s",
    "sim_ra_rel_error": "relative final-apoapsis error, simulated vs collocated",
    "sim_passed": "simulation check within sim_tol",
    "active_constraints": "inequality constraints on their bound at the solution, joined by ' | '",
    "active_box_bounds": "numerical box bounds active at the solution (unstated constraints)",
    "warnings": "every summary warning, joined by ' | '",
    "start": "warm start or fresh guess (sweep/trade only)",
}


# ------------------------------------------------------------------ readers
def _states(ap, phase, sim=False):
    """Return (t, y) of a phase from the collocated or simulated timeseries, or (None, None)."""
    get = ap.sim_ts if sim else ap.ts
    cols = [get(phase, n) for n in STATE_NAMES]
    if any(c is None for c in cols):
        return None, None
    return get(phase, "time"), np.stack(cols, axis=1)


def _orbit(y, cfg):
    """Return the osculating Neptune two-body orbit of a state against the config's target."""
    orb = dynamics.final_orbit(y[:3], y[3:6], cfg.ra_target, cfg.rp_raise_km)
    energy = float(orb["energy"])
    bound = energy < 0
    a = -system.MU_NEPTUNE / (2 * energy)
    period = 2 * math.pi * math.sqrt(a ** 3 / system.MU_NEPTUNE) if bound else None
    return dict(energy_km2s2=energy, ra_km=float(orb["ra"]) if bound else None,
                rp_km=float(orb["rp"]),
                inclination_deg=math.degrees(math.acos(float(orb["cos_inc"]))),
                capture_margin_km2s2=-float(orb["capture"]),
                period_days=period / 86400 if bound else None,
                dv_raise_kms=float(orb["dv_raise"]) if bound else None)


def _point(y, t, epoch):
    """Return position, speed and flight-path angle of one state (inertial and vs atmosphere)."""
    r, v = y[:3], y[3:6]
    rm, vm = float(np.linalg.norm(r)), float(np.linalg.norm(v))
    vrel = dynamics.relative_velocity(r, v)
    vr = float(np.linalg.norm(vrel))
    return dict(t_s=float(t), utc=epoch.iso(t), altitude_km=rm - system.R_NEPTUNE,
                latitude_deg=math.degrees(math.asin(r[2] / rm)),
                speed_inertial_kms=vm, speed_rel_atm_kms=vr,
                fpa_inertial_deg=math.degrees(math.asin(r @ v / (rm * vm))),
                fpa_rel_deg=math.degrees(math.asin(r @ vrel / (rm * vr))),
                energy_km2s2=0.5 * vm ** 2 - system.MU_NEPTUNE / rm)


def triton_encounter(ap, phase):
    """Return closest approach, flyby turn, Δv and Neptune-energy change of a Triton phase.

    The closest approach is the periapsis of the Triton-centred osculating
    conic at the node nearest Triton (the quantity the flyby-altitude
    constraint bounds), so it does not depend on the node spacing.
    """
    tr = ap.epoch.triton
    t, y = _states(ap, phase)
    i = int(np.argmin(np.linalg.norm(y[:, :3] - tr.position(t), axis=1)))
    rel_r = y[i, :3] - tr.position(t[i])
    rel_v = y[i, 3:] - tr.velocity(t[i])
    mu = system.MU_TRITON
    h2 = float(np.sum(np.cross(rel_r, rel_v) ** 2))
    energy = 0.5 * rel_v @ rel_v - mu / np.linalg.norm(rel_r)
    e = math.sqrt(1.0 + 2.0 * energy * h2 / mu ** 2)
    rp = h2 / (mu * (1.0 + e))
    v_rel = math.sqrt(2.0 * energy) if energy > 0 else float(np.linalg.norm(rel_v))
    turn = 2 * math.asin(1 / e) if e > 1 else math.pi
    t_ca = dynamics.state_to_bplane(rel_r, rel_v, t[i], mu=mu)["t_peri"] if e > 1 else float(t[i])
    e0 = 0.5 * y[0, 3:] @ y[0, 3:] - system.MU_NEPTUNE / np.linalg.norm(y[0, :3])
    e1 = 0.5 * y[-1, 3:] @ y[-1, 3:] - system.MU_NEPTUNE / np.linalg.norm(y[-1, :3])
    return dict(phase=phase, t_ca_s=t_ca, utc=ap.epoch.iso(t_ca),
                altitude_km=rp - system.R_TRITON, v_rel_kms=v_rel, turn_deg=math.degrees(turn),
                dv_kms=2 * v_rel * math.sin(turn / 2),
                neptune_energy_change_km2s2=float(e1 - e0))


def simulation_check(ap):
    """Compare each phase's collocated final state with its scipy re-integration.

    Returns:
        dict: Per-phase final position/velocity errors, the final osculating
        orbit from both, the relative apoapsis error ``ra_rel_error`` and
        ``passed`` (error <= ``sim_tol``); ``{"available": False}`` without a simulation.
    """
    out = {"available": True, "phases": {}}
    for ph in ap.phase_names:
        _, y = _states(ap, ph)
        _, ys = _states(ap, ph, sim=True)
        if ys is None:
            return {"available": False}
        out["phases"][ph] = dict(dr_km=float(np.linalg.norm(y[-1, :3] - ys[-1, :3])),
                                 dv_kms=float(np.linalg.norm(y[-1, 3:] - ys[-1, 3:])))
    oc, os_ = _orbit(y[-1], ap.cfg), _orbit(ys[-1], ap.cfg)
    err = abs(os_["ra_km"] - oc["ra_km"]) / oc["ra_km"] if oc["ra_km"] and os_["ra_km"] \
        else float("inf")
    out.update(final_collocated=oc, final_simulated=os_, ra_rel_error=err,
               passed=bool(err <= ap.cfg.sim_tol))
    return out


def summarize(ap, guess=None, solve_info=None, warnings=()):
    """Return the JSON-ready summary of a solved problem.

    Args:
        ap (problem.AeroProblem): Solved (and simulated) problem.
        guess (guess.Guess, optional): Guess it was built from. Defaults to None.
        solve_info (dict, optional): Info returned by ``case.solve``. Defaults to None.
        warnings (tuple, optional): Extra warnings to record. Defaults to ().

    Returns:
        dict: The summary written to ``solution.json``.
    """
    cfg, ep, p = ap.cfg, ap.epoch, ap.p
    solve_info = dict(solve_info or {})
    arr = {n: float(p.get_val(f"arrival_dv.{n}")[0])
           for n in ("vinf", "ra", "dec", "b_t", "b_r", "t_peri")}
    s_hat = dynamics.unit_from_radec(arr["ra"], arr["dec"])
    free = [n.split(".")[-1] for n in p.model.get_design_vars(get_sizes=False)
            if n.startswith("arrival_dv.")]
    arrival = dict(frame="NEQ (Neptune)", vinf_kms=arr["vinf"], ra_deg=math.degrees(arr["ra"]),
                   dec_deg=math.degrees(arr["dec"]), b_t_km=arr["b_t"], b_r_km=arr["b_r"],
                   b_km=math.hypot(arr["b_t"], arr["b_r"]), t_peri_s=arr["t_peri"],
                   t_peri_utc=ep.iso(arr["t_peri"]),
                   vinf_vec_eclip_kms=list(np.linalg.solve(ep.neq_from_eclip, s_hat)
                                           * arr["vinf"]),
                   optimised_parameters=free)

    t, y = _states(ap, "atmosphere")
    atm = {n: ap.ts("atmosphere", n) for n in ("q_conv", "q_rad_unc", "q_total", "q_total_unc",
                                                "g_load", "t_radeq", "alt", "q_dyn", "Q",
                                                "sigma", "v_rel")}
    rho = atmosphere.default().density(atm["alt"])
    q_rad_cpl = heating.q_radiative_coupled(atm["q_rad_unc"], rho, atm["v_rel"] * 1e3,
                                            cfg.rad_kappa)
    q_tot_cpl = atm["q_conv"] + q_rad_cpl
    heat = dict(
        rad_coupling=cfg.rad_coupling, rad_kappa=cfg.rad_kappa,
        peak_q_conv_w_cm2=float(atm["q_conv"].max()),
        peak_q_rad_coupled_w_cm2=float(q_rad_cpl.max()),
        peak_q_rad_uncoupled_w_cm2=float(atm["q_rad_unc"].max()),
        peak_q_total_coupled_w_cm2=float(q_tot_cpl.max()),
        peak_q_total_uncoupled_w_cm2=float(atm["q_total_unc"].max()),
        peak_q_total_selected_w_cm2=float(atm["q_total"].max()),
        heat_load_selected_j_cm2=float(atm["Q"][-1]),
        heat_load_coupled_j_cm2=float(np.trapezoid(q_tot_cpl, t)),
        heat_load_uncoupled_j_cm2=float(np.trapezoid(atm["q_total_unc"], t)),
        peak_t_radeq_k=float(atm["t_radeq"].max()))
    loads = dict(peak_g=float(atm["g_load"].max()), min_alt_km=float(atm["alt"].min()),
                 peak_q_dyn_pa=float(atm["q_dyn"].max()) * 1e3, duration_s=float(t[-1] - t[0]),
                 bank_min_deg=float(np.degrees(atm["sigma"].min())),
                 bank_max_deg=float(np.degrees(atm["sigma"].max())))

    last = ap.phase_names[-1]
    _, yl = _states(ap, last)
    final = dict(_orbit(yl[-1], cfg), read_at=f"end of {last}")
    tri = [triton_encounter(ap, ph) for ph in ap.phase_names if ph.startswith("triton")]
    triton = tri[0] if tri else None
    check = simulation_check(ap)
    solver = io.solver_record(solve_info, p)

    warns = list(warnings) + cfg.warnings()
    if triton and triton["altitude_km"] > TRITON_USED_FACTOR * cfg.triton_alt_min_km:
        warns.append(f"Triton closest approach {triton['altitude_km']:.0f} km: the flyby is "
                     "essentially unused (a local optimum of the Triton mode)")
    if final["rp_km"] < system.R_NEPTUNE:
        warns.append(f"final osculating periapsis {final['rp_km']:.0f} km is below Neptune's "
                     f"surface ({system.R_NEPTUNE:.0f} km); the periapsis-raise burn must fix it")
    if check.get("available") and not check["passed"]:
        warns.append(f"simulate check FAILED: final apoapsis error {check['ra_rel_error']:.2e} "
                     f"> sim_tol {cfg.sim_tol}")
    if solve_info.get("continuation_complete") is False:
        warns.append("continuation to the fixed v-inf direction did not complete")
    if solver["active_box_bounds"]:
        warns.append(f"{io.BOX_BOUND_TEXT}: {', '.join(solver['active_box_bounds'])}")

    s = dict(
        mission="neptune", tag=cfg.tag, mode=cfg.triton_mode, config=cfg.record(),
        vehicle=asdict(cfg.vehicle), epoch=dict(label=ep.label, et=ep.et_ref),
        constants=io.constants_record(), triton_orbit=io.triton_record(ep), arrival=arrival,
        triton=triton, entry=_point(y[0], t[0], ep), exit=_point(y[-1], t[-1], ep),
        loads=loads, heating=heat, final_orbit=final,
        objective=dict(name=cfg.objective, value=float(p.get_val("objective.obj")[0])),
        solver=solver, guess=io.guess_record(guess), simulation_check=check,
        assumptions=cfg.assumptions(), active_box_bounds=solver["active_box_bounds"],
        warnings=warns, notes=NOTES)
    s["active_constraints"] = active_constraints(s, cfg, ep)
    return s


def active_constraints(s, cfg, epoch):
    """Return the inequality constraints of a summary that sit on their bound.

    Args:
        s (dict): Summary under construction (``loads``, ``heating``,
            ``final_orbit``, ``triton``, ``arrival``, ``config``).
        cfg (config.Config): Inputs.
        epoch (system.Epoch): Reference epoch (Triton's period for the t_peri window).

    Returns:
        list: Text per active constraint (``core.io.active_constraints``).
    """
    v, ld, h, f = cfg.vehicle, s["loads"], s["heating"], s["final_orbit"]
    window = cfg.t_peri_window_periods * epoch.triton.period
    checks = [
        ("Triton flyby altitude", (s["triton"] or {}).get("altitude_km"), cfg.triton_alt_min_km,
         "min", "km", "triton_alt_min_km [margin]"),
        ("minimum altitude", ld["min_alt_km"], 0.0, "min", "km", "surface"),
        ("peak g", ld["peak_g"], v.g_max, "max", "g", "vehicle.g_max"),
        ("peak q_total", h["peak_q_total_selected_w_cm2"], v.q_max_w_cm2, "max", "W/cm^2",
         "vehicle.q_max_w_cm2"),
        ("peak T_radeq", h["peak_t_radeq_k"],
         v.t_wall_max_k if v.tps == "non_ablative" else None, "max", "K", "vehicle.t_wall_max_k"),
        ("bank angle", ld["bank_min_deg"], cfg.bank_min_deg, "min", "deg", "bank_min_deg"),
        ("bank angle", ld["bank_max_deg"], cfg.bank_max_deg, "max", "deg", "bank_max_deg"),
        ("final apoapsis", f["ra_km"], cfg.ra_target if cfg.ra_mode == "max" else None, "max",
         "km", "ra_target_km, else ra_target_hill [margin]"),
        ("final inclination", f["inclination_deg"],
         None if cfg.match_triton_plane else cfg.min_inclination_deg, "min", "deg",
         "min_inclination_deg [margin]"),
    ]
    if s["config"]["opt_t_peri"]:
        t_peri = s["arrival"]["t_peri_s"]
        checks += [("t_peri", t_peri, -window, "min", "s", "t_peri_window_periods"),
                   ("t_peri", t_peri, window, "max", "s", "t_peri_window_periods")]
    return io.active_constraints(checks)


def timeseries(ap, sim=False):
    """Return ``{phase: {name: array}}`` of the plotted outputs (collocated or simulated)."""
    get = ap.sim_ts if sim else ap.ts
    out = {}
    for ph in ap.phase_names:
        names = list(STATE_NAMES) + ["time", "r"]
        if ph == "atmosphere":
            names += ["sigma", "g_load", "q_conv", "q_rad", "q_rad_unc", "q_total", "Q", "alt",
                      "v_rel", "fpa_rel"]
        elif ph.startswith("triton"):
            names += ["d_triton"]
        d = {}
        for n in names:
            try:
                v = get(ph, n)
            except Exception:    # output not in this phase's timeseries
                v = None
            if v is not None:
                d[n] = v
        out[ph] = d
    return out


# ------------------------------------------------------------------- checks
def triton_used(summary):
    """Return True if the Triton flyby passes within 10x the minimum flyby altitude."""
    t = summary.get("triton")
    return bool(t) and t["altitude_km"] <= TRITON_USED_FACTOR * summary["config"]["triton_alt_min_km"]


def acceptable(summary):
    """Return True for a converged case whose Triton mode actually uses Triton."""
    ok = bool(summary["solver"].get("success"))
    return ok and (summary["mode"] == "none" or triton_used(summary))


# ------------------------------------------------------------------- output
def save(summary, out_dir):
    """Write ``solution.json`` into ``out_dir`` and return its path."""
    return io.save(summary, out_dir)


def print_summary(s, stream=print):
    """Print the summary sections, the checks, the assumptions and margins and every warning."""
    h, ld, f, a, sv = s["heating"], s["loads"], s["final_orbit"], s["arrival"], s["solver"]
    cfg = s["config"]
    # header
    stream(f"neptune | mode {s['mode']} | v_inf {a['vinf_kms']:.2f} km/s | vehicle "
           f"{s['vehicle']['name']} | objective {s['objective']['name']} = "
           f"{s['objective']['value']:.4g}")
    io.print_solver(sv, stream)
    # arrival
    stream(f"  arrival [{a['frame']}]: v_inf {a['vinf_kms']:.4f} km/s, RA {a['ra_deg']:.2f} deg, "
           f"Dec {a['dec_deg']:.2f} deg, B {a['b_km']:.0f} km, t_peri {a['t_peri_utc']}; "
           f"optimised {', '.join(a['optimised_parameters'])} (opt_direction="
           f"{cfg['opt_direction']}, opt_t_peri={cfg['opt_t_peri']})")
    # flyby
    if s["triton"]:
        t = s["triton"]
        stream(f"  Triton flyby: CA alt {t['altitude_km']:.1f} km, v_rel {t['v_rel_kms']:.3f} km/s,"
               f" turn {t['turn_deg']:.3f} deg, dv {t['dv_kms'] * 1e3:.1f} m/s, dE_Neptune "
               f"{t['neptune_energy_change_km2s2']:+.4f} km^2/s^2")
    else:
        stream("  Triton flyby: none (mode none)")
    # entry / exit
    e, x = s["entry"], s["exit"]
    stream(f"  entry: v {e['speed_inertial_kms']:.3f} km/s inertial / {e['speed_rel_atm_kms']:.3f}"
           f" km/s vs atmosphere, FPA {e['fpa_inertial_deg']:.3f} / {e['fpa_rel_deg']:.3f} deg,"
           f" lat {e['latitude_deg']:.1f} deg")
    stream(f"  exit: v {x['speed_inertial_kms']:.3f} km/s inertial, energy "
           f"{x['energy_km2s2']:+.4f} km^2/s^2")
    # loads
    stream(f"  loads: peak g {ld['peak_g']:.2f}, min alt {ld['min_alt_km']:.1f} km, peak q_dyn "
           f"{ld['peak_q_dyn_pa']:.0f} Pa, pass {ld['duration_s']:.0f} s")
    # heating
    stream(f"  heating [W/cm^2] (limits and objective use rad_coupling={h['rad_coupling']}): "
           f"q_conv {h['peak_q_conv_w_cm2']:.1f}; q_rad {h['peak_q_rad_coupled_w_cm2']:.1f} "
           f"coupled / {h['peak_q_rad_uncoupled_w_cm2']:.1f} uncoupled; q_total "
           f"{h['peak_q_total_coupled_w_cm2']:.1f} coupled / "
           f"{h['peak_q_total_uncoupled_w_cm2']:.1f} uncoupled")
    stream(f"  heat load [J/cm^2]: {h['heat_load_coupled_j_cm2']:.0f} coupled / "
           f"{h['heat_load_uncoupled_j_cm2']:.0f} uncoupled; wall T_radeq "
           f"{h['peak_t_radeq_k']:.0f} K")
    # final orbit
    ra = f"{f['ra_km']:.4g} km" if f["ra_km"] else "unbound"
    per = f"{f['period_days']:.1f} d" if f["period_days"] else "-"
    dv = f"{f['dv_raise_kms'] * 1e3:.1f} m/s" if f["dv_raise_kms"] is not None else "-"
    stream(f"  final orbit ({f['read_at']}): ra {ra}, rp {f['rp_km']:.0f} km, i "
           f"{f['inclination_deg']:.2f} deg, period {per}, capture margin "
           f"{f['capture_margin_km2s2']:.2e} km^2/s^2, periapsis-raise dv {dv}")
    # checks, assumptions and margins, warnings
    c = s["simulation_check"]
    sim = (f"{'PASS' if c['passed'] else 'FAIL'} (ra error {c['ra_rel_error']:.2e})"
           if c.get("available") else "not available")
    io.print_tail(s, {"converged": bool(sv.get("success")),
                      "triton_used": "-" if s["mode"] == "none" else triton_used(s),
                      "simulate": sim}, stream)


# -------------------------------------------------------------------- table
def table_row(summary):
    """Return the flat sweep/trade row of a summary (keys in ``COLUMNS``)."""
    s = summary
    h, ld, f, e, v = s["heating"], s["loads"], s["final_orbit"], s["entry"], s["vehicle"]
    t, sv, chk = s["triton"] or {}, s["solver"], s["simulation_check"]
    return dict(
        mode=s["mode"], vinf_kms=s["arrival"]["vinf_kms"], vehicle=v["name"],
        beta_kg_m2=v["beta_kg_m2"], ld=v["lift_to_drag"], rn_m=v["nose_radius_m"],
        converged=bool(sv.get("success")), ipopt_status=sv.get("ipopt_text"),
        ipopt_iterations=sv.get("ipopt_iterations"),
        triton_used=None if s["mode"] == "none" else triton_used(s),
        triton_alt_km=t.get("altitude_km"),
        triton_dv_ms=t["dv_kms"] * 1e3 if t else None,
        triton_dE_km2s2=t.get("neptune_energy_change_km2s2"),
        entry_speed_kms=e["speed_inertial_kms"], entry_speed_rel_kms=e["speed_rel_atm_kms"],
        entry_fpa_rel_deg=e["fpa_rel_deg"], peak_g=ld["peak_g"], min_alt_km=ld["min_alt_km"],
        peak_q_conv_w_cm2=h["peak_q_conv_w_cm2"],
        peak_q_rad_coupled_w_cm2=h["peak_q_rad_coupled_w_cm2"],
        peak_q_rad_uncoupled_w_cm2=h["peak_q_rad_uncoupled_w_cm2"],
        peak_q_total_w_cm2=h["peak_q_total_selected_w_cm2"],
        peak_q_total_coupled_w_cm2=h["peak_q_total_coupled_w_cm2"],
        peak_q_total_uncoupled_w_cm2=h["peak_q_total_uncoupled_w_cm2"],
        heat_load_j_cm2=h["heat_load_selected_j_cm2"], peak_t_radeq_k=h["peak_t_radeq_k"],
        final_ra_km=f["ra_km"], final_rp_km=f["rp_km"], final_inc_deg=f["inclination_deg"],
        capture_margin_km2s2=f["capture_margin_km2s2"],
        dv_raise_ms=f["dv_raise_kms"] * 1e3 if f["dv_raise_kms"] is not None else None,
        sim_ra_rel_error=chk.get("ra_rel_error"), sim_passed=chk.get("passed"),
        active_constraints=" | ".join(s["active_constraints"]),
        active_box_bounds=" | ".join(s["active_box_bounds"]),
        warnings=" | ".join(s["warnings"]))


def failed_row(cfg, reason):
    """Return the table row of a case that produced no solution."""
    v = cfg.vehicle
    return dict(mode=cfg.triton_mode, vinf_kms=cfg.vinf_kms, vehicle=v.name,
                beta_kg_m2=v.beta_kg_m2, ld=v.lift_to_drag, rn_m=v.nose_radius_m,
                converged=False, ipopt_status=reason)


def write_csv(rows, path):
    """Write table rows to CSV in ``COLUMNS`` order and return the path."""
    return io.write_csv(rows, path, COLUMNS)


print_table = io.print_table
