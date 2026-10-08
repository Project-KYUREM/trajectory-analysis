# %% [markdown]
# # Triton ballute aerocapture after a close Neptune flyby (Dymos)
#
# Run the cells top to bottom in an interactive window, or the whole file with
# `python aerocapture/triton/run_triton.py`. Every input is in the CONFIG cell, tagged
# [physics] (changes the problem), [numerics] (changes how it is solved), [guess] (only
# shapes the starting point) or [margin] (a margin, assumed value, objective weight or check
# tolerance; each is listed with its reason in `config.ASSUMPTIONS` and printed with the
# summary). Outputs go to `results/aerocapture/triton/<tag>/`. Design and findings:
# `aerocapture/triton/PLAN.md`.

# %% imports
import sys

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aerocapture.triton import case  # noqa: E402
from aerocapture.triton import config  # noqa: E402
from aerocapture.triton import plots  # noqa: E402
from aerocapture.triton import results  # noqa: E402
from aerocapture.triton.ballute import Ballute, Probe  # noqa: E402

# %% CONFIG -- every input of the run, with units
# Ballute [physics]: enveloping sphere around the probe (NIAC "Pluto Hop, Skip, and Jump").
BALLUTE = Ballute(
    name="ead-70",                  # label used in result paths and tables
    diameter_m=70.0,                # envelope diameter, m (drag area, Kn length, nose radius D/2)
    envelope_mass_kg=330.0,         # envelope + inflation-system mass, kg
    cd_continuum=0.9,               # continuum drag coefficient
    cd_free_molecular=2.0,          # free-molecular drag coefficient
    cd_model="wilmoth",             # "wilmoth" (sine^2 Kn bridge, 1e-3..10) | "constant" (cd_continuum)
    emissivity=0.85,                # film emissivity
    radiating_sides=2,              # faces reradiating the stagnation heat (2 for a film)
    t_film_max_k=773.0,             # film radiative-equilibrium temperature limit, K (Kapton ~673 K continuous)
)
# Probe [physics]: the orbiter after release.
PROBE = Probe(
    mass_kg=200.0,                  # mass after release, kg
    cd_area_m2=2.0,                 # drag area C_D A, m^2 (beta 100 kg/m^2)
    nose_radius_m=0.5,              # stagnation-point nose radius, m (probe heating report)
)

CONFIG = config.Config(
    # --- arrival [physics] -----------------------------------------------------------------
    vinf_kms=5.0,                       # hyperbolic excess speed at Neptune, km/s
    direction="free",                   # v-inf direction: "free" | "radec" | "from_run"
    ra_deg=0.0,                         # v-inf right ascension in NEQ, deg (direction="radec")
    dec_deg=0.0,                        # v-inf declination in NEQ, deg (direction="radec")
    from_run="capture/EJN",             # published run docs/data/<model>/<sequence> (direction="from_run")
    point="p13116",                     # point id in that run's cloud.json (from_run; v_inf 5.0 km/s)
    epoch="2060-01-01 00:00:00 TDB",    # time zero; from_run uses the run's arrival epoch
    # opt flags, None = automatic: a free direction optimises RA/Dec, a fixed one t_peri.
    opt_direction=None,                 # optimise the v-inf RA/Dec
    opt_t_peri=None,                    # optimise the Neptune periapsis time (sets Triton's phase)
    t_peri_window_periods=0.5,          # t_peri bound when optimised, +- Triton periods
    r0_km=1.0e6,                        # start radius on the incoming hyperbola, km
    # --- Neptune flyby [physics] -----------------------------------------------------------
    rp_min_rn=1.10,                     # lowest Neptune periapsis radius, R_N (24 622 km)
    rp_max_rn=1.25,                     # highest Neptune periapsis radius, R_N
    ring_keepout_km=(38000.0, 85000.0), # ring-plane crossings inside [0] or outside [1], km
    neptune_dv_max_kms=0.0,             # Neptune periapsis (Oberth) burn cap, km/s (0: none)
    # --- Triton approach [physics] ---------------------------------------------------------
    frame_switch_km=30000.0,            # [numerics] Triton distance of the NEQ -> Triton-frame switch, km
    interface_alt_km=1000.0,            # Triton entry/exit interface altitude, km
    tcm_max_kms=0.0,                    # post-flyby TCM allowed in the final answer, km/s
    # --- Triton atmosphere [physics] -------------------------------------------------------
    atm_model="bates",                  # "bates" (hydrostatic N2) | "table" (atm_table_path)
    atm_table_path=None,                # two-column table: altitude km, density kg/m^3
    p_surface_ubar=16.0,                # Bates surface pressure, microbar
    t_surface_k=38.0,                   # Bates surface temperature T_0, K
    t_inf_k=92.0,                       # Bates thermospheric temperature T_inf, K
    t_scale_km=161.6,                   # Bates scale L, km (p(1026 km) = 2.8e-9 Pa)
    atm_rotation=True,                  # synchronous co-rotation with Triton (16.7 m/s at the surface)
    # (density multiplier: margins and assumptions below)
    # --- ballute and probe [physics] -------------------------------------------------------
    ballute=BALLUTE,                    # see BALLUTE above
    probe=PROBE,                        # see PROBE above
    # --- release and limits [physics] ------------------------------------------------------
    release_mode="free",                # "free": ballute released at a free time | "none": kept to exit
    g_max=25.0,                         # peak sensed-deceleration limit, Earth g (NIAC Pluto 16-36 g)
    q_max_w_cm2=None,                   # peak ballute heat-rate limit, W/cm^2 (None: none)
    # --- heating [physics] -----------------------------------------------------------------
    conv_model="sutton_graves_n2",      # continuum correlation (heating.CONVECTIVE_SETS key)
    bridging="smooth_min",              # "smooth_min" (q_c^-2 + q_fm^-2)^-1/2 | "continuum"
    rad_model="tauber_sutton_air",      # radiative upper bound, always reported
    # --- final Triton orbit [physics] ------------------------------------------------------
    ra_mode="max",                      # "max": ra <= limit | "equal": ra = design limit after the burn
    final_sense="retrograde",           # "retrograde" (i_T >= i_retro_min_deg) | "prograde" | "any"
    burn_mode="free",                   # periapsis-raise burn: "free" 3-vector | "tangential"
    burn_dv_max_kms=0.3,                # burn magnitude cap, km/s
    final_coast_h=24.0,                 # final-orbit phase length, h
    # (apoapsis limit, periapsis floor, i_T bands, design margins: margins group below)
    # --- objective [physics] ---------------------------------------------------------------
    objective="qpeak",                  # "qpeak" (peak ballute q_heat) | "heat" | "dv" (burn) | "g" (peak g)
    # --- margins and assumptions [margin]: value, unit; why this value / what it costs ----
    h_min_km=10.0,                      # km; altitude floor through the ballute and probe passes
                                        #   Triton's relief is ~1 km; active only on steep passes
    rp_final_alt_km=1000.0,             # km; final periapsis floor, enforced by the stability check
                                        #   50-150 km above the exobase (850-950 km): drag does not decay the orbit
    final_rp_margin_km=50.0,            # km; the optimiser holds rp >= floor + this
                                        #   the tide moves rp ~-10 km over 120 d; costs a slightly larger burn
    ra_max_hill=0.3,                    # R_H; final apoapsis limit (0.3: 4379 km), stability-checked
                                        #   inside the circular-orbit limits 0.93 R_H retro / 0.49 R_H prograde
    final_ra_margin=0.05,               # fraction; the optimiser holds ra <= (1 - this) * limit
                                        #   the tide adds ~+60 km over 120 d; costs a slightly hotter capture
    i_retro_min_deg=141.0,              # deg; retrograde i_T floor, 0.2 deg outside the Kozai band (140.8)
                                        #   circular-orbit theory; not sufficient at e >~ 0.25
    i_pro_max_deg=39.0,                 # deg; prograde i_T ceiling, 0.2 deg inside the Kozai band (39.2)
    atm_density_scale=1.0,              # Bates N2 density multiplier (1: nominal; dispersions 0.5-1.5)
                                        #   the entry corridor moves with it
    fm_accommodation=1.0,               # free-molecular energy accommodation (1: all energy absorbed)
                                        #   conservative
    rad_in_limits=False,                # constrain the Tauber-Sutton radiative bound (in q_heat)
                                        #   False: reported only (thin-gas upper bound; LAURA: N2 radiation minimal)
    heat_weight=0.0,                    # per J/cm^2; heat-load term added to the objective (0: none)
    dv_weight=1.0e-3,                   # per m/s; burn term pinning the burn and capture-orbit plane
                                        #   0: IPOPT drifts to a plane change onto the Kozai edge; moves q < 2 %
    stability_days=120.0,               # d; post-solve tidal propagation horizon (check only)
                                        #   spans the 60-180 d Kozai growth time
    sim_tol=0.01,                       # simulate check: largest relative final-apoapsis error
    sim_rp_tol_km=5.0,                  # km; simulate check: largest final-periapsis error
    # --- run [physics] ---------------------------------------------------------------------
    scope="full",                       # "full" mission | "triton_only" (stage S0 only)
    # --- transcription and solver [numerics] -----------------------------------------------
    order=3,                            # Radau polynomial order per segment
    segments=dict(approach=20, flyby_in=15, flyby_out=15, cruise=20, triton_approach=15,
                  ballute=40, probe=25, capture_coast=15, final=40),   # Radau segments per phase
    r_split_km=2.0e5,                   # approach / flyby_in and flyby_out / cruise boundary radius, km
    tcm_homotopy_kms=0.2,               # TCM cap of stage S1 (removed in S2), km/s
    max_iter=500,                       # IPOPT iteration limit per solve
    tol=1e-5,                           # IPOPT overall NLP tolerance
    constr_viol_tol=1e-6,               # IPOPT constraint-violation tolerance (scaled)
    print_level=0,                      # IPOPT print level (0: quiet)
    linear_solver="mumps",              # IPOPT linear solver
    # --- initial guess [guess] (starting point only) ---------------------------------------
    guess_corridor_fraction=0.5,        # entry angle within the scipy corridor (0 shallow, 1 steep)
    guess_plane="triton",               # Neptune-leg plane of the guess: Triton's orbit plane
    guess_encounter="co",               # encounter sense: "co" (with Triton) | "counter"
    guess_ra_fraction=0.92,             # guess releases at this fraction of the apoapsis limit
    guess_rp_margin_km=50.0,            # guess burn targets the design periapsis plus this, km (guess only)
    # --- output ----------------------------------------------------------------------------
    tag=None,                           # results/aerocapture/triton/<tag>/ (None: from scope, v_inf, ballute)
)

# %% initial guess (prints every input, [margin] fields marked, and the input warnings first)
cfg, epoch, target_radec, atm, atm_record = case.resolve(CONFIG)
guess = case.make_guess(cfg, epoch, atm)
out_dir = case.output_dir(cfg)
plots.save_all(plots.guess_figures(guess, epoch, cfg, atm), out_dir)

# %% solve (staged S0 -> S1 -> S2 [-> S3]; see case.solve) and simulate check
tp, info = case.solve(cfg, epoch, atm, guess, target_radec)

# %% summary: sections, checks, assumptions and margins in effect, active constraints, warnings
summary = results.summarize(tp, guess, info, atm_record)
results.print_summary(summary)

# %% plots (collocated solid, simulated dashed, initial guess dotted)
figs = plots.solution_figures(tp, summary, guess=guess)
if "ipykernel" in sys.modules:
    plt.show()
plots.save_all(figs, out_dir)

# %% save solution.json (inputs, assumptions and margins, results, checks, warnings)
path = results.save(summary, out_dir)
print(f"wrote {path}")
