"""The Dymos multi-phase optimal-control problem, built only from a ``config.Config``.

The whole NLP, with the ``Config`` field behind every number in brackets:

Phases (Radau, compressed, order [order], [segments] per phase)::

    none      approach -> coast_in -> atmosphere
    inbound   approach -> triton_in -> coast_in -> atmosphere
    outbound  approach -> coast_in -> atmosphere -> coast_mid -> triton_out

    Every phase has free initial time and duration and the states x y z vx vy vz
    (km, km/s, Neptune-equator frame). The atmosphere phase adds the heat load Q
    (J/cm^2, starts at 0), the bank-angle control sigma (rad, 0 = lift up) and
    the vehicle parameters beta, L/D, nose radius [vehicle] as fixed inputs.

Design variables outside the phases:
    b_t, b_r          B-plane aim point of the arrival hyperbola, km.
    ra, dec           v-inf direction, if [opt_direction]; else held at the guess
                      (or at the fixed asymptote).
    t_peri            Periapsis time, within +-[t_peri_window_periods] Triton
                      periods, if [opt_t_peri]; else held at the guess.
    q_peak / g_peak   Slack bounding the peak heat rate / g-load ([objective]
                      "qpeak" / "g" only).
    |v-inf| [vinf_kms] is always fixed.

Objective (minimised) [objective] + [heat_weight] * Q(t_final):
    qpeak  q_peak, with q_total(t) - q_peak <= 0 at every node: min max q_total.
    g      g_peak, with g_load(t) - g_peak <= 0 at every node: min max g_load.
    heat   Q at atmosphere exit.
    dv     two-body periapsis-raise burn to [rp_raise_km] at the final node.

Constraints:
    arrival match  hyperbola state at r = [r0_km] and its time equal the first
                   approach node (6 states + time).
    linkage        time and the 6 states are continuous between neighbours.
    approach       d_T >= [triton_clear_km]; final r = [r_split_km] (none,
                   outbound) or final d_T = [triton_window_km] (inbound).
    triton_in/out  d_T >= R_Triton + [triton_alt_min_km] at every node and for
                   the periapsis of the Triton-centred osculating conic at the
                   node nearest the guess's closest approach; d_T' <= 0 at
                   start, >= 0 at end; final d_T = [triton_window_km] (in) or
                   [triton_exit_km] (out).
    coast_in       d_T >= [triton_clear_km]; final r = R_N + [interface_alt_km],
                   r.v <= 0.
    coast_mid      d_T >= [triton_clear_km]; final d_T = [triton_window_km].
    atmosphere     final r = R_N + [interface_alt_km], r.v >= 0; altitude >= 0;
                   |sigma'| <= [bank_rate_max_degs]; sigma within
                   [bank_min_deg, bank_max_deg]; g_load <= vehicle.g_max;
                   q_total <= vehicle.q_max_w_cm2 (if set); T_radeq <=
                   vehicle.t_wall_max_k (non-ablative TPS with a limit).
    final orbit    on the last node of the last phase, for the osculating
                   Neptune two-body orbit: capture = E + mu/(ra_target + rp)
                   <= 0 ([ra_mode] "max": bound with ra <= ra_target) or = 0
                   ("equal"), ra_target from [ra_target_km] / [ra_target_hill];
                   cos(i) <= cos([min_inclination_deg]); h.p_T = h.q_T = 0
                   ([match_triton_plane]: final plane = Triton's plane).

The conic-periapsis bound makes the minimum-altitude constraint hold between
nodes too: node values alone let the true closest approach dip ~2 km below the
bound. Near closest approach Neptune's tide is ~0.2 % of Triton's pull, so the
conic periapsis is the true one to well under 1 km.

The capture metric implies E < 0. The final orbit is read at atmosphere exit
(none, inbound) or at d_T = [triton_exit_km] (outbound). At that distance
Triton still perturbs the osculating orbit (mu_T/d = 0.014 km^2/s^2 at 1e5
km): propagating on shows the true apoapsis about 1.5 % below the value read,
so the ``max`` bound is met with margin and ``equal`` overshoots by ~1.5 %.
The outbound final periapsis is not constrained and may lie inside Neptune.

Scaling: every constraint, design variable and objective gets an OpenMDAO
``ref`` equal to the size of a change that matters for it (``SCALE`` below,
or the largest value along the guess for states and times), so IPOPT sees
O(1) quantities. Linkage, arrival-match and capture refs are deliberately tight:
the problem is nearly invariant to rotating the whole trajectory, and loose
scaling there lets IPOPT's quasi-Newton steps run away along those directions.

Box bounds: ``BOX`` below bounds phase start times and durations, every state,
the heat load, the min-max slacks, the B-plane aim point and the declination.
They only keep IPOPT near the guess and are not part of the formulation;
after every solve ``info["active_box_bounds"]`` lists any that is active
(``core.solve.active_box_bounds``), which would make it an unstated constraint.

Solver plumbing (IPOPT driver, solve, readback, design-vector copies) is
``core.solve.DymosProblem``; the arrival hyperbola and its match constraints
are ``core.arrival.add_arrival_match``.
"""
import math

import dymos as dm
import numpy as np
import openmdao.api as om

from aerocapture.core import arrival
from aerocapture.core import guess as cg
from aerocapture.core import paths
from aerocapture.core import solve
from aerocapture.core import system
from aerocapture.neptune import guess as guess_mod
from aerocapture.neptune import ode

PHASES = {
    "none": ["approach", "coast_in", "atmosphere"],
    "inbound": ["approach", "triton_in", "coast_in", "atmosphere"],
    "outbound": ["approach", "coast_in", "atmosphere", "coast_mid", "triton_out"],
}
SEGMENT_WEIGHT = {"approach": "radius", "coast_in": "radius", "coast_mid": "radius",
                  "triton_in": "triton", "triton_out": "triton", "atmosphere": "atmosphere"}
DEFAULT_WORK_DIR = paths.work_dir("neptune")

SCALE = dict(                       # OpenMDAO ref of each quantity: the size of a change that matters
    radius_km=1.0e4,                # r = r_split at the approach end
    interface_km=1.0e3,             # r = interface radius at the coast_in and atmosphere ends
    rdotv_km2s=1.0e4,               # sign of r.v at the coast_in and atmosphere ends
    d_triton_km=1.0e4,              # Triton clearance and window distances
    triton_alt_km=1.0e3,            # Triton minimum flyby distance (path and conic periapsis)
    d_triton_dot_kms=1.0,           # sign of the Triton range rate at the window ends
    alt_km=100.0,                   # atmosphere altitude >= 0
    g=10.0,                         # g-load limit, g_peak slack and "g" objective
    q_w_cm2=1.0e3,                  # heat-rate limit, q_peak slack and "qpeak" objective
    q_excess_w_cm2=100.0,           # q_total - q_peak <= 0
    g_excess=1.0,                   # g_load - g_peak <= 0
    t_radeq_k=1.0e3,                # radiative-equilibrium wall temperature limit
    bank_rad=1.0,                   # bank-angle control
    b_km=1.0e4,                     # B-plane aim point design variables
    t_peri_s=1.0e4,                 # periapsis-time design variable
    radec_rad=1.0,                  # v-inf RA/Dec design variables
    heat_j_cm2=1.0e4,               # "heat" objective
    dv_kms=0.1,                     # "dv" objective
    capture_km2s2=0.1,              # final capture metric
    cos_inc=1.0,                    # final inclination bound
    plane=0.01,                     # final h.p_T = h.q_T = 0 (Triton plane)
    link_time_s=10.0,               # time linkage and arrival-match time
    link_pos_km=100.0,              # position linkage and arrival match
    link_vel_kms=0.01,              # velocity linkage and arrival match
)
BOX = dict(                         # bounds that only keep IPOPT near the guess (see module docstring)
    start_time_triton_periods=0.5,  # phase start within this many Triton periods of the guess
    duration_factor=(0.3, 3.0),     # phase duration within these multiples of the guess
    state_factor=2.0,               # each state within +- this multiple of the guess max |r| or |v|
    heat_factor=100.0,              # heat load Q below this multiple of the guess heat load
    b_factor=5.0,                   # |b_t|, |b_r| within this multiple of the guess |B|
    dec_margin_rad=1e-3,            # dec kept this far from the poles
    q_peak_w_cm2=(1.0, 1.0e5),      # q_peak slack range, W/cm^2
    g_peak_min=0.01,                # g_peak slack floor
)
SLACK_SEED_FACTOR = 1.02            # q_peak / g_peak start this far above the guess peaks


class AeroProblem(solve.DymosProblem):
    """Build, seed, solve and read back one aerocapture problem.

    Args:
        cfg (config.Config): Inputs.
        epoch (system.Epoch): Reference epoch (frame and Triton ephemeris).
        guess (guess.Guess): Seed; also sets segment clustering and scaling.
        fixed_radec (tuple, optional): v-inf (ra, dec) in rad to hold fixed.
            Defaults to None (direction free).
        name (str, optional): OpenMDAO problem name. Defaults to "aerocapture".
        work_dir (str or Path, optional): OpenMDAO output directory. Defaults to
            the ``OPENMDAO_WORKDIR`` environment variable if set, else
            ``results/aerocapture/neptune/_openmdao``.
    """

    def __init__(self, cfg, epoch, guess, fixed_radec=None, name="aerocapture", work_dir=None):
        cfg.validate()
        self.cfg, self.epoch, self.guess = cfg, epoch, guess
        self.fixed_radec = fixed_radec
        self.phase_names = PHASES[cfg.triton_mode]
        self.name = name
        self.work_dir = solve.resolve_work_dir(work_dir, DEFAULT_WORK_DIR)
        self.free_arrival = arrival.free_arrival_parameters(cfg, fixed_radec is not None)
        with paths.workdir(self.work_dir):
            self._build()

    # ------------------------------------------------------------------ phases
    def _transcription(self, name, arc):
        """Return the Radau transcription for a phase, segments clustered on its guess arc."""
        nseg = self.cfg.segments[name]
        weight = {"radius": cg.radius_weight, "atmosphere": guess_mod.atmosphere_weight,
                  "triton": guess_mod.triton_weight(self.epoch.triton)}[SEGMENT_WEIGHT[name]]
        return dm.Radau(num_segments=nseg, order=self.cfg.order, compressed=True,
                        segment_ends=cg.segment_ends(arc, nseg, weight))

    def _time_and_states(self, ph, arc, vacuum):
        """Add free initial time, duration and the six Cartesian states, scaled on the guess."""
        t0, dur = float(arc.t[0]), float(arc.t[-1] - arc.t[0])
        span = BOX["start_time_triton_periods"] * self.epoch.triton.period
        lo, hi = BOX["duration_factor"]
        ph.set_time_options(units="s", targets=["time"] if vacuum else [],
                            initial_ref0=t0, initial_ref=t0 + max(dur, 100.0),
                            initial_bounds=(t0 - span, t0 + span),
                            duration_ref=dur, duration_bounds=(lo * dur, hi * dur))
        k = BOX["state_factor"]
        for names, cols, units in ((ode.STATES[:3], slice(0, 3), "km"),
                                   (ode.STATES[3:], slice(3, 6), "km/s")):
            ref = float(np.max(np.linalg.norm(arc.y[:, cols], axis=1)))
            for n in names:
                ph.add_state(n, rate_source=f"{n}_dot", units=units, ref=ref,
                             lower=-k * ref, upper=k * ref)

    def _orbit_options(self):
        """Return the ODE options of the final-orbit outputs (target ra, rp_raise, Triton plane)."""
        tr = self.epoch.triton
        return dict(ra_max=float(self.cfg.ra_target), rp_target=float(self.cfg.rp_raise_km),
                    triton_plane=(np.asarray(tr.p), np.asarray(tr.q)))

    def _vacuum_phase(self, name, arc):
        """Return a vacuum phase (Neptune + J2 + Triton) with its distance constraints."""
        cfg, S = self.cfg, SCALE
        triton_phase = name.startswith("triton")
        ph = dm.Phase(ode_class=ode.VacuumODE, transcription=self._transcription(name, arc),
                      ode_init_kwargs=dict(triton=self.epoch.triton, **self._orbit_options()))
        self._time_and_states(ph, arc, vacuum=True)
        ph.add_timeseries_output("*")
        final = lambda out, ref, **kw: ph.add_boundary_constraint(out, loc="final", ref=ref, **kw)
        if triton_phase:
            ph.add_path_constraint("d_triton", lower=system.R_TRITON + cfg.triton_alt_min_km,
                                   ref=S["triton_alt_km"])
            ph.add_boundary_constraint("d_triton_dot", loc="initial", upper=0.0,
                                       ref=S["d_triton_dot_kms"])
            final("d_triton_dot", S["d_triton_dot_kms"], lower=0.0)
            exit_d = cfg.triton_window_km if name == "triton_in" else cfg.triton_exit_km
            final("d_triton", S["d_triton_km"], equals=exit_d)
        else:
            ph.add_path_constraint("d_triton", lower=cfg.triton_clear_km, ref=S["d_triton_km"])
        if name == "approach" and cfg.triton_mode == "inbound":
            final("d_triton", S["d_triton_km"], equals=cfg.triton_window_km)
        elif name == "approach":
            final("r", S["radius_km"], equals=cfg.r_split_km)
        elif name == "coast_in":
            final("r", S["interface_km"], equals=cfg.r_interface)
            final("rdotv", S["rdotv_km2s"], upper=0.0)
        elif name == "coast_mid":
            final("d_triton", S["d_triton_km"], equals=cfg.triton_window_km)
        return ph

    def _atmosphere_phase(self, arc):
        """Return the atmosphere phase: bank control, heat load, loads and exit constraints."""
        cfg, veh, S = self.cfg, self.cfg.vehicle, SCALE
        aero = dict(conv_model=cfg.conv_model, coupled=cfg.coupled, kappa=float(cfg.rad_kappa),
                    rn_exp=float(cfg.rad_rn_exp), emissivity=float(veh.emissivity),
                    **self._orbit_options())
        ph = dm.Phase(ode_class=ode.AtmosphereODE, transcription=self._transcription(
            "atmosphere", arc), ode_init_kwargs=dict(aero_options=aero))
        self._time_and_states(ph, arc, vacuum=False)
        q_ref = max(float(arc.Q[-1]), 100.0)
        ph.add_state("Q", rate_source="Q_dot", units="J/cm**2", fix_initial=True, lower=0.0,
                     upper=BOX["heat_factor"] * q_ref, ref=q_ref)
        ph.add_control("sigma", units="rad", lower=math.radians(cfg.bank_min_deg),
                       upper=math.radians(cfg.bank_max_deg), continuity=True,
                       rate_continuity=True, targets=["sigma"], ref=S["bank_rad"])
        for n, units, val in (("beta", "kg/m**2", veh.beta_kg_m2), ("ld", None, veh.lift_to_drag),
                              ("rn", "m", veh.nose_radius_m)):
            ph.add_parameter(n, units=units, opt=False, static_target=True, val=val)
        q_lo, q_hi = BOX["q_peak_w_cm2"]
        ph.add_parameter("q_peak", units="W/cm**2", opt=cfg.objective == "qpeak",
                         static_target=True, val=1000.0, lower=q_lo, upper=q_hi,
                         ref=S["q_w_cm2"])
        ph.add_parameter("g_peak", opt=cfg.objective == "g", static_target=True,
                         val=veh.g_max, lower=BOX["g_peak_min"], ref=S["g"])
        rate = math.radians(cfg.bank_rate_max_degs)
        ph.add_path_constraint("sigma_rate", lower=-rate, upper=rate, units="rad/s", ref=rate)
        ph.add_path_constraint("alt", lower=0.0, ref=S["alt_km"])
        ph.add_path_constraint("g_load", upper=veh.g_max, ref=S["g"])
        if veh.q_max_w_cm2 is not None:
            ph.add_path_constraint("q_total", upper=veh.q_max_w_cm2, ref=S["q_w_cm2"])
        if veh.tps == "non_ablative" and veh.t_wall_max_k is not None:
            ph.add_path_constraint("t_radeq", upper=veh.t_wall_max_k, ref=S["t_radeq_k"])
        if cfg.objective == "qpeak":
            ph.add_path_constraint("q_excess", upper=0.0, ref=S["q_excess_w_cm2"])
        if cfg.objective == "g":
            ph.add_path_constraint("g_excess", upper=0.0, ref=S["g_excess"])
        ph.add_boundary_constraint("r", loc="final", equals=cfg.r_interface,
                                   ref=S["interface_km"])
        ph.add_boundary_constraint("rdotv", loc="final", lower=0.0, ref=S["rdotv_km2s"])
        ph.add_timeseries_output("*")
        return ph

    def _final_orbit_constraints(self, ph):
        """Constrain the osculating orbit at the last node: capture, inclination, plane."""
        cfg, S = self.cfg, SCALE
        bound = dict(upper=0.0) if cfg.ra_mode == "max" else dict(equals=0.0)
        ph.add_boundary_constraint("capture", loc="final", ref=S["capture_km2s2"], **bound)
        ph.add_boundary_constraint("cos_inc", loc="final",
                                   upper=math.cos(math.radians(cfg.min_inclination_deg)),
                                   ref=S["cos_inc"])
        if cfg.match_triton_plane:
            ph.add_boundary_constraint("h_p", loc="final", equals=0.0, ref=S["plane"])
            ph.add_boundary_constraint("h_q", loc="final", equals=0.0, ref=S["plane"])

    def _closest_approach_constraint(self, p, name):
        """Bound the Triton-centred conic periapsis at the node nearest the guess CA."""
        ph, arc, tr = self.phases[name], self.guess.arcs[name], self.epoch.triton
        t_ca = guess_mod.closest_approach_time(arc, tr)
        tau = 2.0 * (t_ca - arc.t[0]) / (arc.t[-1] - arc.t[0]) - 1.0
        node = int(np.argmin(np.abs(ph.options["transcription"].grid_data.node_ptau - tau)))
        comp = f"{name}_periapsis"
        p.model.add_subsystem(comp, ode.TritonPeriapsisComp(triton=tr))
        for n in ode.STATES + ("time",):
            p.model.connect(f"traj.{name}.timeseries.{n}", f"{comp}.{n}", src_indices=[node])
        p.model.add_constraint(f"{comp}.rp", lower=system.R_TRITON + self.cfg.triton_alt_min_km,
                               ref=SCALE["triton_alt_km"])

    # ------------------------------------------------------------------- build
    def _build(self):
        """Assemble phases, linkages, arrival, objective and driver, then set up the problem."""
        cfg, arcs, S = self.cfg, self.guess.arcs, SCALE
        p = om.Problem(name=self.name, reports=False, work_dir=str(self.work_dir))
        traj = dm.Trajectory()
        self.phases = {}
        for name in self.phase_names:
            arc = arcs[name]
            ph = self._atmosphere_phase(arc) if name == "atmosphere" else \
                self._vacuum_phase(name, arc)
            traj.add_phase(name, ph)
            self.phases[name] = ph
        self._final_orbit_constraints(self.phases[self.phase_names[-1]])
        traj.link_phases(self.phase_names, vars=["time"], ref=S["link_time_s"])
        traj.link_phases(self.phase_names, vars=["x", "y", "z"], ref=S["link_pos_km"])
        traj.link_phases(self.phase_names, vars=["vx", "vy", "vz"], ref=S["link_vel_kms"])
        p.model.add_subsystem("traj", traj)
        for name in self.phase_names:
            if name.startswith("triton"):
                self._closest_approach_constraint(p, name)
        self._add_arrival(p)
        self._add_objective(p)

        solve.add_ipopt_driver(p, cfg)
        p.setup(force_alloc_complex=True)
        self.p, self.traj = p, traj

    def _add_arrival(self, p):
        """Add the arrival hyperbola, its match to the first approach node and its design vars."""
        cfg, a, S = self.cfg, self.guess.arrival, SCALE
        b_lim = BOX["b_factor"] * math.hypot(a["b_t"], a["b_r"])
        dec_lim = math.pi / 2 - BOX["dec_margin_rad"]
        half = cfg.t_peri_window_periods * self.epoch.triton.period
        bounds = dict(ra=dict(ref=S["radec_rad"]),
                      dec=dict(lower=-dec_lim, upper=dec_lim, ref=S["radec_rad"]),
                      b_t=dict(lower=-b_lim, upper=b_lim, ref=S["b_km"]),
                      b_r=dict(lower=-b_lim, upper=b_lim, ref=S["b_km"]),
                      t_peri=dict(lower=-half, upper=half, ref=S["t_peri_s"]))
        refs = dict(pos=S["link_pos_km"], vel=S["link_vel_kms"], time=S["link_time_s"])
        arrival.add_arrival_match(p, a, cfg.vinf_kms, cfg.r0_km, system.MU_NEPTUNE, "approach",
                                  ode.STATES, refs, bounds, self.free_arrival)

    def _add_objective(self, p):
        """Add the objective ``main + heat_weight * Q(t_final)``, main set by ``cfg.objective``."""
        cfg, S = self.cfg, SCALE
        p.model.add_subsystem("objective", om.ExecComp("obj = main + w * heat",
                                                       w={"val": cfg.heat_weight}))
        atm = "traj.atmosphere"
        src, ref = {"qpeak": (f"{atm}.parameter_vals:q_peak", S["q_w_cm2"]),
                    "g": (f"{atm}.parameter_vals:g_peak", S["g"]),
                    "heat": (f"{atm}.timeseries.Q", S["heat_j_cm2"]),
                    "dv": (f"traj.{self.phase_names[-1]}.timeseries.dv_raise", S["dv_kms"])}[
                        cfg.objective]
        p.model.connect(src, "objective.main", src_indices=[-1])
        p.model.connect(f"{atm}.timeseries.Q", "objective.heat", src_indices=[-1])
        p.model.add_objective("objective.obj", ref=ref)

    # -------------------------------------------------------------------- seed
    def set_guess(self, guess=None):
        """Seed every phase (time, states, bank, heat load) and the arrival parameters.

        Args:
            guess (guess.Guess, optional): Seed. Defaults to the guess the problem was built on.
        """
        g = guess or self.guess
        p = self.p
        for name, ph in self.phases.items():
            arc = g.arcs[name]
            ph.set_time_val(initial=float(arc.t[0]), duration=float(arc.t[-1] - arc.t[0]))
            for i, n in enumerate(ode.STATES):
                ph.set_state_val(n, arc.y[:, i], time_vals=arc.t, interpolation_kind="cubic")
            if name == "atmosphere":
                ph.set_state_val("Q", arc.Q, time_vals=arc.t, interpolation_kind="cubic")
                ph.set_control_val("sigma", arc.sigma, time_vals=arc.t,
                                   interpolation_kind="cubic")
                diag = guess_mod.atmosphere_diagnostics(arc.t, arc.y, arc.sigma,
                                                        self.cfg.vehicle_params())
                ph.set_parameter_val("q_peak", float(np.max(diag["q_total"])) * SLACK_SEED_FACTOR)
                ph.set_parameter_val("g_peak", float(np.max(diag["g_load"])) * SLACK_SEED_FACTOR)
        for n in arrival.ARRIVAL_PARAMETERS:
            p.set_val(f"arrival_dv.{n}", g.arrival[n])
        if self.fixed_radec is not None:
            p.set_val("arrival_dv.ra", self.fixed_radec[0])
            p.set_val("arrival_dv.dec", self.fixed_radec[1])
        p.set_val("arrival_dv.vinf", self.cfg.vinf_kms)

    def set_vehicle(self, vehicle):
        """Set beta, L/D and nose radius in place, for warm-started vehicle trades."""
        atm = self.phases["atmosphere"]
        atm.set_parameter_val("beta", vehicle.beta_kg_m2)
        atm.set_parameter_val("ld", vehicle.lift_to_drag)
        atm.set_parameter_val("rn", vehicle.nose_radius_m)
