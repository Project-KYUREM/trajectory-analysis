"""The Dymos multi-phase problem of the Triton ballute capture, built only from a Config.

Phases (Radau, compressed, order [order], [segments] per phase)::

    full         approach -> flyby_in -> flyby_out -> cruise        (NEQ, VacuumODE)
                 -> triton_approach -> ballute -> probe -> capture_coast -> final   (TC)
    triton_only  triton_approach -> ballute -> probe -> capture_coast -> final

    [release_mode] "none" drops ``probe``: the ballute flies to the exit.

Design variables outside the phases:
    full: arrival b_t, b_r, and ra/dec or t_peri per the resolved opt flags
          (``Config.resolved``); |v_inf| fixed.
    triton_only: b_t, b_r of the Triton-frame hyperbola at [frame_switch_km];
          its v_inf vector and periapsis time are fixed at the guess.
    burn dv (3, or 1 along v for [burn_mode] "tangential"), |dv| <= [burn_dv_max_kms].
    optional impulses with |dv| <= cap: TCM (flyby_out -> cruise; cap
          [tcm_max_kms], or the S1 homotopy value) and Neptune periapsis burn
          (flyby_in -> flyby_out; cap [neptune_dv_max_kms]). The cap is an
          input, so the staged solve tightens it without rebuilding.
    q_peak / g_peak slack ([objective] "qpeak" / "g").

Objective: main + [heat_weight] Q_ballute(t_f) + [dv_weight] |dv_burn| (m/s), main =
    qpeak  q_peak with q_heat - q_peak <= 0 on the ballute phase
    heat   ballute heat load Q (J/cm^2)
    g      g_peak with g_load - g_peak <= 0 on the ballute phase
    dv     |dv_burn| (km/s)

Constraints:
    arrival match   hyperbola state and time = first node (6 states + time).
    linkage         time and states continuous; capture_coast -> final links
                    time and position, velocity through the burn; cruise ->
                    triton_approach links time, states through FrameMatchComp.
    approach        final r = [r_split_km].
    flyby_in        final r.v = 0 and r within [rp_min_rn, rp_max_rn] R_N;
                    ring nodes: near node u >= 1/[ring_keepout_km][0], far node
                    u <= 1/[ring_keepout_km][1].
    flyby_out       final r = [r_split_km].
    cruise          final d_Triton = [frame_switch_km], closing.
    triton_approach final r = R_T + [interface_alt_km], r.v <= 0.
    ballute         alt >= [h_min_km], g <= [g_max], T_film <= ballute.t_film_max_k,
                    q_heat <= [q_max_w_cm2] (if set); slack path constraint.
    probe           alt >= [h_min_km], g <= [g_max]; final r = interface, r.v >= 0.
    capture_coast   final osculating Triton apoapsis <= ra_max ([ra_max_hill] R_H).
    final           duration [final_coast_h]; at every node osculating rp >=
                    R_T + [rp_final_alt_km] + [final_rp_margin_km] and ra <=
                    (1 - [final_ra_margin]) ra_max ([ra_mode] "equal": equal to it
                    right after the burn); i_T in the [final_sense] band at the
                    start. The margins absorb the tidal oscillation of rp and ra
                    over [stability_days]; the stability check uses the bare limits.

Scaling follows ``aerocapture/neptune/problem.py``: every constraint, design variable
and objective gets a ``ref`` the size of a change that matters (``SCALE``);
states and times are scaled on the guess.

Box bounds: ``BOX`` bounds phase start times and durations, every state, the
heat load, the min-max slacks and the B-plane aim point (and, full scope, the
declination). They only keep IPOPT near the guess; after every solve
``info["active_box_bounds"]`` lists any that is active
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
from aerocapture.neptune import ode as aero_ode
from aerocapture.triton import guess as guess_mod
from aerocapture.triton import ode

NEPTUNE_PHASES = ["approach", "flyby_in", "flyby_out", "cruise"]
TRITON_PHASES = ["triton_approach", "ballute", "probe", "capture_coast", "final"]
ATM_PHASES = ("ballute", "probe")
DEFAULT_WORK_DIR = paths.work_dir("triton")

SCALE = dict(                       # OpenMDAO ref of each quantity: the size of a change that matters
    radius_km=1.0e4,                # r = r_split at the approach and flyby_out ends
    rp_km=1.0e3,                    # Neptune periapsis band
    rdotv_km2s=1.0e4,               # r.v = 0 at Neptune periapsis
    d_triton_km=1.0e3,              # Triton distance at the frame switch
    d_triton_dot_kms=1.0,           # closing Triton range rate at the frame switch
    interface_km=10.0,              # r = Triton interface radius
    rdotv_t_km2s=1.0e3,             # sign of r.v at the Triton interface
    alt_km=10.0,                    # altitude floors (h_min, capture coast >= 0)
    g=10.0,                         # g-load limit, g_peak slack and "g" objective
    q_w_cm2=1.0,                    # heat-rate limit, q_peak slack and "qpeak" objective
    t_film_k=100.0,                 # film temperature limit
    q_excess_w_cm2=0.1,             # q_heat - q_peak <= 0
    g_excess=1.0,                   # g_load - g_peak <= 0
    capture_km2s2=0.01,             # Triton capture metric (coast end, final orbit)
    rp_final_km=100.0,              # final periapsis floor
    cos_inc=0.1,                    # final i_T band
    ring_inv_km=1.0e-5,             # ring-node inverse radii
    b_km=1.0e3,                     # Neptune B-plane aim point design variables
    b_t_km=1.0e2,                   # Triton-frame B-plane aim point (triton_only)
    t_peri_s=1.0e4,                 # periapsis-time design variable
    radec_rad=1.0,                  # v-inf RA/Dec design variables
    heat_j_cm2=10.0,                # "heat" objective
    dv_kms=0.01,                    # impulse residuals, caps and design variables
    dv_obj_kms=0.1,                 # "dv" objective
    link_time_s=10.0,               # NEQ time linkage and arrival match
    link_pos_km=100.0,              # NEQ position linkage, arrival and frame match (/10)
    link_vel_kms=0.01,              # NEQ velocity linkage, arrival and frame match (/10)
    link_t_time_s=1.0,              # TC time linkage and Triton-frame arrival match
    link_t_pos_km=1.0,              # TC position linkage and arrival match
    link_t_vel_kms=1.0e-3,          # TC velocity linkage, burn residuals and arrival match
)
BOX = dict(                         # bounds that only keep IPOPT near the guess (see module docstring)
    start_time_triton_periods=0.5,  # Neptune-phase start within this many Triton periods of the guess
    triton_start_window_s=86400.0,  # Triton-frame phase start within this many s of the guess
    duration_factor=(0.3, 3.0),     # phase duration within these multiples of the guess
    state_factor=2.0,               # each state within +- this multiple of its guess max |component|
    heat_factor=100.0,              # heat load Q below this multiple of the guess heat load
    b_factor=5.0,                   # Neptune |b_t|, |b_r| within this multiple of the guess |B|
    b_triton_km=2.0e4,              # Triton-frame |b_t|, |b_r| bound, km
    dec_margin_rad=1e-3,            # dec kept this far from the poles
    q_peak_w_cm2=(1e-3, 1e3),       # q_peak slack range, W/cm^2
    g_peak_min=0.01,                # g_peak slack floor
)
SLACK_SEED_FACTOR = 1.02            # q_peak / g_peak start this far above the guess peaks


def phase_names(cfg, scope=None):
    """Return the ordered phase names of a scope (default ``cfg.scope``) and release mode."""
    scope = scope or cfg.scope
    names = (NEPTUNE_PHASES if scope == "full" else []) + TRITON_PHASES
    if cfg.release_mode == "none":
        names.remove("probe")
    return names


def unique_arc(t, y, Q=None):
    """Return a ``core.guess.Arc`` with duplicate times (segment ends) removed."""
    t, idx = np.unique(t, return_index=True)
    return cg.Arc(t, y[idx], None, None if Q is None else Q[idx])


class TritonProblem(solve.DymosProblem):
    """Build, seed, solve and read back one Triton capture problem.

    Args:
        cfg (config.Config): Inputs.
        epoch (system.Epoch): Reference epoch.
        atm (atmosphere.Atmosphere): Triton atmosphere.
        guess (guess.TritonGuess): Seed; also sets segment clustering and scaling.
        scope (str, optional): ``"full"`` or ``"triton_only"``. Defaults to ``cfg.scope``.
        tcm_cap (float, optional): Build the TCM impulse with this initial cap,
            km/s (None: no TCM unless ``cfg.tcm_max_kms`` > 0). Defaults to None.
        fixed_radec (tuple, optional): v-inf (ra, dec), rad, held fixed. Defaults to None.
        name (str, optional): OpenMDAO problem name. Defaults to "triton".
        work_dir (str or Path, optional): OpenMDAO output directory. Defaults to
            ``OPENMDAO_WORKDIR`` if set, else ``results/aerocapture/triton/_openmdao``.
    """

    def __init__(self, cfg, epoch, atm, guess, scope=None, tcm_cap=None, fixed_radec=None,
                 name="triton", work_dir=None):
        cfg.validate()
        self.cfg, self.epoch, self.atm, self.guess = cfg, epoch, atm, guess
        self.scope = scope or cfg.scope
        self.tr = epoch.triton
        self.ra_max = cfg.ra_max_km(self.tr)
        self.phase_names = phase_names(cfg, self.scope)
        self.full = self.scope == "full"
        if self.full and tcm_cap is None and cfg.tcm_max_kms > 0.0:
            tcm_cap = cfg.tcm_max_kms
        self.tcm_cap = tcm_cap if self.full else None
        self.neptune_burn = self.full and cfg.neptune_dv_max_kms > 0.0
        self.fixed_radec = fixed_radec
        self.name = name
        self.work_dir = solve.resolve_work_dir(work_dir, DEFAULT_WORK_DIR)
        self.free_arrival = (arrival.free_arrival_parameters(cfg, fixed_radec is not None)
                             if self.full else ["b_t", "b_r"])
        with paths.workdir(self.work_dir):
            self._build()

    # ------------------------------------------------------------------ phases
    def _states(self, name):
        """Return the state names of a phase (NEQ or TC)."""
        return ode.STATES_T if name in TRITON_PHASES else ode.STATES_N

    def _transcription(self, name, arc):
        """Return the Radau transcription, segments clustered on the guess arc."""
        nseg = self.cfg.segments[name]
        if name in ATM_PHASES:
            weight = cg.density_weight(self.atm, system.R_TRITON)
        elif name == "final":
            weight = guess_mod.orbit_weight
        else:
            weight = cg.radius_weight
        return dm.Radau(num_segments=nseg, order=self.cfg.order, compressed=True,
                        segment_ends=cg.segment_ends(arc, nseg, weight))

    def _time_and_states(self, ph, name, arc):
        """Add time options and the six Cartesian states, scaled on the guess."""
        t0, dur = float(arc.t[0]), float(arc.t[-1] - arc.t[0])
        span = (BOX["start_time_triton_periods"] * self.tr.period if name in NEPTUNE_PHASES
                else BOX["triton_start_window_s"])
        lo, hi = BOX["duration_factor"]
        opts = dict(units="s", targets=["time"], initial_ref0=t0, initial_ref=t0 + max(dur, 100.0),
                    initial_bounds=(t0 - span, t0 + span), duration_ref=dur)
        if name == "final":
            opts.update(fix_duration=True)
        else:
            opts.update(duration_bounds=(lo * dur, hi * dur))
        ph.set_time_options(**opts)
        k = BOX["state_factor"]
        names = self._states(name)
        for group, cols, units in ((names[:3], slice(0, 3), "km"), (names[3:], slice(3, 6), "km/s")):
            ref = float(np.max(np.abs(arc.y[:, cols])))
            for n in group:
                ph.add_state(n, rate_source=f"{n}_dot", units=units, ref=ref,
                             lower=-k * ref, upper=k * ref)

    def _neptune_phase(self, name, arc):
        """Return a NEQ vacuum phase (Neptune + J2 + Triton) with its boundary constraints."""
        cfg, S = self.cfg, SCALE
        ph = dm.Phase(ode_class=aero_ode.VacuumODE, transcription=self._transcription(name, arc),
                      ode_init_kwargs=dict(triton=self.tr))
        self._time_and_states(ph, name, arc)
        ph.add_timeseries_output("*")
        final = lambda out, ref, **kw: ph.add_boundary_constraint(out, loc="final", ref=ref, **kw)
        if name in ("approach", "flyby_out"):
            final("r", S["radius_km"], equals=cfg.r_split_km)
        elif name == "flyby_in":
            lo, hi = cfg.rp_band_km
            final("rdotv", S["rdotv_km2s"], equals=0.0)
            final("r", S["rp_km"], lower=lo, upper=hi)
        elif name == "cruise":
            final("d_triton", S["d_triton_km"], equals=cfg.frame_switch_km)
            final("d_triton_dot", S["d_triton_dot_kms"], upper=0.0)
        return ph

    def _triton_vacuum_phase(self, name, arc):
        """Return a TC vacuum phase (approach, capture coast or final orbit)."""
        cfg, S = self.cfg, SCALE
        ra = cfg.ra_design_km(self.tr) if name == "final" else self.ra_max
        ph = dm.Phase(ode_class=ode.TritonVacuumODE,
                      transcription=self._transcription(name, arc),
                      ode_init_kwargs=dict(triton=self.tr, ra_max=float(ra),
                                           rp_target=float(cfg.rp_design)))
        self._time_and_states(ph, name, arc)
        ph.add_timeseries_output("*")
        if name == "triton_approach":
            ph.add_boundary_constraint("r", loc="final", equals=cfg.r_interface,
                                       ref=S["interface_km"])
            ph.add_boundary_constraint("rdotv", loc="final", upper=0.0, ref=S["rdotv_t_km2s"])
        elif name == "capture_coast":
            ph.add_path_constraint("alt", lower=0.0, ref=S["alt_km"])
            ph.add_boundary_constraint("capture", loc="final", upper=0.0,
                                       ref=S["capture_km2s2"])
        elif name == "final":
            ph.add_path_constraint("rp", lower=cfg.rp_design, ref=S["rp_final_km"])
            if cfg.ra_mode == "equal":
                ph.add_boundary_constraint("capture", loc="initial", equals=0.0,
                                           ref=S["capture_km2s2"])
            else:
                ph.add_path_constraint("capture", upper=0.0, ref=S["capture_km2s2"])
            lo, hi = cfg.inclination_bounds_deg()
            bounds = {}
            if hi < 180.0:
                bounds["lower"] = math.cos(math.radians(hi))
            if lo > 0.0:
                bounds["upper"] = math.cos(math.radians(lo))
            if bounds:
                ph.add_boundary_constraint("cos_inc", loc="initial", ref=S["cos_inc"], **bounds)
        return ph

    def _atmosphere_phase(self, name, arc):
        """Return the ballute or probe phase with its parameters and path constraints."""
        cfg, S = self.cfg, SCALE
        eom = dict(triton=self.tr, ra_max=float(self.ra_max), rp_target=float(cfg.rp_final),
                   **cfg.eom_options(name))
        ph = dm.Phase(ode_class=ode.TritonAtmosphereODE,
                      transcription=self._transcription(name, arc),
                      ode_init_kwargs=dict(atmosphere=self.atm, eom_options=eom))
        self._time_and_states(ph, name, arc)
        units = dict(mass="kg", area="m**2", length="m", rn="m", cd_c=None, cd_fm=None)
        for n, val in cfg.pass_params(name).items():
            ph.add_parameter(n, units=units[n], opt=False, static_target=True, val=val)
        ph.add_path_constraint("alt", lower=cfg.h_min_km, ref=S["alt_km"])
        ph.add_path_constraint("g_load", upper=cfg.g_max, ref=S["g"])
        exits = name == "probe" or cfg.release_mode == "none"
        if name == "ballute":
            q_ref = max(float(arc.Q[-1]), 1.0)
            ph.add_state("Q", rate_source="Q_dot", units="J/cm**2", fix_initial=True, lower=0.0,
                         upper=BOX["heat_factor"] * q_ref, ref=q_ref)
            q_lo, q_hi = BOX["q_peak_w_cm2"]
            ph.add_parameter("q_peak", units="W/cm**2", opt=cfg.objective == "qpeak",
                             static_target=True, val=1.0, lower=q_lo, upper=q_hi,
                             ref=S["q_w_cm2"])
            ph.add_parameter("g_peak", opt=cfg.objective == "g", static_target=True,
                             val=cfg.g_max, lower=BOX["g_peak_min"], ref=S["g"])
            ph.add_path_constraint("t_film", upper=cfg.ballute.t_film_max_k, ref=S["t_film_k"])
            if cfg.q_max_w_cm2 is not None:
                ph.add_path_constraint("q_heat", upper=cfg.q_max_w_cm2, ref=S["q_w_cm2"])
            if cfg.objective == "qpeak":
                ph.add_path_constraint("q_excess", upper=0.0, ref=S["q_excess_w_cm2"])
            if cfg.objective == "g":
                ph.add_path_constraint("g_excess", upper=0.0, ref=S["g_excess"])
        if exits:
            ph.add_boundary_constraint("r", loc="final", equals=cfg.r_interface,
                                       ref=S["interface_km"])
            ph.add_boundary_constraint("rdotv", loc="final", lower=0.0, ref=S["rdotv_t_km2s"])
        ph.add_timeseries_output("*")
        return ph

    # ------------------------------------------------------------------- build
    def _build(self):
        """Assemble phases, linkages, boundary components, objective and driver."""
        cfg, arcs, S = self.cfg, self.guess.arcs, SCALE
        p = om.Problem(name=self.name, reports=False, work_dir=str(self.work_dir))
        traj = dm.Trajectory()
        self.phases = {}
        for name in self.phase_names:
            arc = arcs[name]
            if name in NEPTUNE_PHASES:
                ph = self._neptune_phase(name, arc)
            elif name in ATM_PHASES:
                ph = self._atmosphere_phase(name, arc)
            else:
                ph = self._triton_vacuum_phase(name, arc)
            traj.add_phase(name, ph)
            self.phases[name] = ph
        self.impulses = {}
        if self.full:
            nep = list(NEPTUNE_PHASES)
            for a, b in zip(nep[:-1], nep[1:]):
                kind = {("flyby_in", "flyby_out"): "neptune_burn" if self.neptune_burn else None,
                        ("flyby_out", "cruise"): "tcm" if self.tcm_cap is not None else None}
                self._link(traj, a, b, kind.get((a, b)), triton=False)
            traj.link_phases(["cruise", "triton_approach"], vars=["time"], ref=S["link_time_s"])
        tri = [n for n in self.phase_names if n in TRITON_PHASES]
        for a, b in zip(tri[:-1], tri[1:]):
            self._link(traj, a, b, "burn" if b == "final" else None, triton=True)
        p.model.add_subsystem("traj", traj)
        self._add_impulses(p)
        if self.full:
            self._add_frame_match(p)
            self._add_ring(p)
        self._add_arrival(p)
        self._add_objective(p)
        solve.add_ipopt_driver(p, cfg)
        p.setup(force_alloc_complex=True)
        self.p, self.traj = p, traj

    def _link(self, traj, a, b, impulse, triton):
        """Link time and states of two phases; an impulse joins the velocities instead."""
        S = SCALE
        k = "link_t_" if triton else "link_"
        names = ode.STATES_T if triton else ode.STATES_N
        traj.link_phases([a, b], vars=["time"], ref=S[k + "time_s"])
        traj.link_phases([a, b], vars=list(names[:3]), ref=S[k + "pos_km"])
        if impulse is None:
            traj.link_phases([a, b], vars=list(names[3:]), ref=S[k + "vel_kms"])
        else:
            self.impulses[impulse] = (a, b, names, S[k + "vel_kms"])

    def _add_impulses(self, p):
        """Add the burn / TCM / Neptune-burn impulses, their design variables and caps."""
        cfg = self.cfg
        caps = dict(burn=cfg.burn_dv_max_kms, tcm=self.tcm_cap, neptune_burn=cfg.neptune_dv_max_kms)
        for kind, (a, b, names, vref) in self.impulses.items():
            mode = cfg.burn_mode if kind == "burn" else "free"
            n_dv = 3 if mode == "free" else 1
            cap = float(caps[kind])
            ivc = om.IndepVarComp()
            ivc.add_output("dv", val=np.zeros(n_dv), units="km/s")
            ivc.add_output("cap", val=cap, units="km/s")
            p.model.add_subsystem(f"{kind}_dv", ivc)
            p.model.add_subsystem(kind, ode.ImpulseComp(mode=mode))
            p.model.connect(f"{kind}_dv.dv", f"{kind}.dv")
            for c, n in zip("xyz", names[3:]):
                p.model.connect(f"traj.{a}.timeseries.{n}", f"{kind}.v{c}_b", src_indices=[-1])
                p.model.connect(f"traj.{b}.timeseries.{n}", f"{kind}.v{c}_a", src_indices=[0])
                p.model.add_constraint(f"{kind}.res_{c}", equals=0.0, ref=vref)
            p.model.add_subsystem(f"{kind}_cap", om.ExecComp(
                "excess = dv_mag - cap", excess={"units": "km/s"}, dv_mag={"units": "km/s"},
                cap={"units": "km/s"}))
            p.model.connect(f"{kind}.dv_mag", f"{kind}_cap.dv_mag")
            p.model.connect(f"{kind}_dv.cap", f"{kind}_cap.cap")
            p.model.add_constraint(f"{kind}_cap.excess", upper=0.0, ref=SCALE["dv_kms"])
            bound = max(cap, cfg.tcm_homotopy_kms if kind == "tcm" else cap)
            p.model.add_design_var(f"{kind}_dv.dv", lower=-bound, upper=bound, ref=SCALE["dv_kms"])

    def _add_frame_match(self, p):
        """Join the cruise end (NEQ) to the Triton-approach start (TC) at the same time."""
        S = SCALE
        p.model.add_subsystem("frame_match", ode.FrameMatchComp(triton=self.tr))
        for n in ode.STATES_N:
            p.model.connect(f"traj.cruise.timeseries.{n}", f"frame_match.{n}", src_indices=[-1])
        for n in ode.STATES_T:
            p.model.connect(f"traj.triton_approach.timeseries.{n}", f"frame_match.{n}",
                            src_indices=[0])
        p.model.connect("traj.cruise.timeseries.time", "frame_match.time", src_indices=[-1])
        for n in ("dx", "dy", "dz"):
            p.model.add_constraint(f"frame_match.{n}", equals=0.0, ref=S["link_pos_km"] / 10.0)
        for n in ("dvx", "dvy", "dvz"):
            p.model.add_constraint(f"frame_match.{n}", equals=0.0, ref=S["link_vel_kms"] / 10.0)

    def _add_ring(self, p):
        """Bound the equatorial-node radii of the osculating conic at Neptune periapsis."""
        r_in, r_out = self.cfg.ring_keepout_km
        p.model.add_subsystem("ring", ode.RingNodeComp(sign=float(self.guess.ring_sign)))
        for n in ode.STATES_N:
            p.model.connect(f"traj.flyby_in.timeseries.{n}", f"ring.{n}", src_indices=[-1])
        p.model.add_constraint("ring.u_near", lower=1.0 / r_in, ref=SCALE["ring_inv_km"])
        p.model.add_constraint("ring.u_far", upper=1.0 / r_out, ref=SCALE["ring_inv_km"])

    def _add_arrival(self, p):
        """Add the arrival hyperbola (Neptune or Triton frame), its match and design variables."""
        cfg, S = self.cfg, SCALE
        if self.full:
            a, mu, r0, phase = self.guess.arrival, system.MU_NEPTUNE, cfg.r0_km, "approach"
            states, vinf, k = ode.STATES_N, cfg.vinf_kms, "link_"
            b_lim, b_ref = BOX["b_factor"] * math.hypot(a["b_t"], a["b_r"]), S["b_km"]
        else:
            a, mu, r0, phase = (self.guess.triton_arrival, system.MU_TRITON, cfg.frame_switch_km,
                                "triton_approach")
            states, vinf, k = ode.STATES_T, a["vinf"], "link_t_"
            b_lim, b_ref = BOX["b_triton_km"], S["b_t_km"]
        dec_lim = math.pi / 2 - BOX["dec_margin_rad"]
        half = cfg.t_peri_window_periods * self.tr.period
        bounds = dict(ra=dict(ref=S["radec_rad"]),
                      dec=dict(lower=-dec_lim, upper=dec_lim, ref=S["radec_rad"]),
                      b_t=dict(lower=-b_lim, upper=b_lim, ref=b_ref),
                      b_r=dict(lower=-b_lim, upper=b_lim, ref=b_ref),
                      t_peri=dict(lower=-half, upper=half, ref=S["t_peri_s"]))
        refs = dict(pos=S[k + "pos_km"], vel=S[k + "vel_kms"], time=S[k + "time_s"])
        arrival.add_arrival_match(p, a, vinf, r0, mu, phase, states, refs, bounds,
                                  self.free_arrival)

    def _add_objective(self, p):
        """Add ``main + heat_weight * Q + dv_weight * |dv_burn| (m/s)``."""
        cfg, S = self.cfg, SCALE
        p.model.add_subsystem("objective", om.ExecComp(
            "obj = main + wq * heat + wv * dv * 1000.0", wq={"val": cfg.heat_weight},
            wv={"val": cfg.dv_weight}))
        bal = "traj.ballute"
        src, ref = {"qpeak": (f"{bal}.parameter_vals:q_peak", S["q_w_cm2"]),
                    "g": (f"{bal}.parameter_vals:g_peak", S["g"]),
                    "heat": (f"{bal}.timeseries.Q", S["heat_j_cm2"]),
                    "dv": ("burn.dv_mag", S["dv_obj_kms"])}[cfg.objective]
        if cfg.objective == "dv":
            p.model.connect(src, "objective.main")
        else:
            p.model.connect(src, "objective.main", src_indices=[-1])
        p.model.connect(f"{bal}.timeseries.Q", "objective.heat", src_indices=[-1])
        p.model.connect("burn.dv_mag", "objective.dv")
        p.model.add_objective("objective.obj", ref=ref)

    # -------------------------------------------------------------------- seed
    def set_guess(self, guess=None):
        """Seed every phase, the slacks, the impulses and the arrival parameters.

        Args:
            guess (guess.TritonGuess, optional): Seed. Defaults to the build guess.
        """
        g = guess or self.guess
        p = self.p
        for name, ph in self.phases.items():
            arc = g.arcs[name]
            dur = (self.cfg.final_coast_h * 3600.0 if name == "final"
                   else float(arc.t[-1] - arc.t[0]))
            ph.set_time_val(initial=float(arc.t[0]), duration=dur)
            for i, n in enumerate(self._states(name)):
                ph.set_state_val(n, arc.y[:, i], time_vals=arc.t, interpolation_kind="cubic")
            if name == "ballute":
                ph.set_state_val("Q", arc.Q, time_vals=arc.t, interpolation_kind="cubic")
                diag = guess_mod.pass_diagnostics(arc, self.cfg, self.tr, self.atm, "ballute")
                ph.set_parameter_val("q_peak", float(np.max(diag["q_heat"])) * SLACK_SEED_FACTOR)
                ph.set_parameter_val("g_peak", float(np.max(diag["g_load"])) * SLACK_SEED_FACTOR)
        params = g.arrival if self.full else g.triton_arrival
        for n in arrival.ARRIVAL_PARAMETERS:
            p.set_val(f"arrival_dv.{n}", params[n])
        p.set_val("arrival_dv.vinf", self.cfg.vinf_kms if self.full else params["vinf"])
        if self.full and self.fixed_radec is not None:
            p.set_val("arrival_dv.ra", self.fixed_radec[0])
            p.set_val("arrival_dv.dec", self.fixed_radec[1])
        dvs = getattr(g, "impulses", None) or {}
        for kind in self.impulses:
            if kind == "burn":
                dv = np.asarray(dvs.get("burn", g.burn_dv), float)
                if self.cfg.burn_mode == "tangential":
                    dv = np.array([np.linalg.norm(dv)])
            else:
                dv = np.asarray(dvs.get(kind, np.zeros(3)), float)
            p.set_val(f"{kind}_dv.dv", dv)

    def set_cap(self, kind, cap):
        """Set the |dv| cap of an impulse (``"tcm"``, ``"neptune_burn"``) in place."""
        self.p.set_val(f"{kind}_dv.cap", cap)

    def set_vehicle(self, cfg):
        """Set the ballute and probe parameters of ``cfg`` in place, for warm-started trades."""
        for name in ATM_PHASES:
            if name in self.phases:
                for n, val in cfg.pass_params(name).items():
                    self.phases[name].set_parameter_val(n, val)

    def solution_guess(self):
        """Return a ``TritonGuess`` holding this problem's current trajectory and impulses."""
        arcs = {}
        for name in self.phase_names:
            t = self.ts(name, "time")
            y = np.stack([self.ts(name, n) for n in self._states(name)], axis=1)
            Q = self.ts(name, "Q") if name == "ballute" else None
            arcs[name] = unique_arc(t, y, Q)
        g = self.guess
        arr = {n: float(self.p.get_val(f"arrival_dv.{n}")[0])
               for n in arrival.ARRIVAL_PARAMETERS}
        arr["vinf"] = float(self.p.get_val("arrival_dv.vinf")[0])
        new = guess_mod.TritonGuess(
            arcs=arcs, arrival=arr if self.full else g.arrival,
            triton_arrival=g.triton_arrival if self.full else arr,
            burn_dv=self.impulse_dv("burn"), ring_sign=g.ring_sign, info=dict(g.info))
        new.impulses = {k: self.impulse_dv(k) for k in self.impulses}
        return new

    def impulse_dv(self, kind):
        """Return the impulse vector of ``kind`` in km/s (zeros if absent)."""
        if kind not in self.impulses:
            return np.zeros(3)
        dv = self.p.get_val(f"{kind}_dv.dv").copy()
        if kind == "burn" and self.cfg.burn_mode == "tangential":
            vb = np.array([self.ts(self.impulses[kind][0], n)[-1]
                           for n in ode.STATES_T[3:]])
            return dv[0] * vb / np.linalg.norm(vb)
        return dv
