# %% [markdown]
# # Neptune trade: vehicle preset x beta x L/D x nose radius x v-inf x Triton mode
#
# BASE holds every input; the TRADE cell lists the traded values (None keeps the preset's
# own value) and the CASES cell shows exactly which fields each case overrides. Cases are
# solved in order; a case that differs from the last accepted one only in v-inf or the
# vehicle's beta, L/D or nose radius is warm-started from it (they are Dymos parameters),
# with a fresh guess as fallback (`case.solve_chain`). Failed cases are recorded, not
# dropped. Run the cells top to bottom, or `python aerocapture/neptune/trade_neptune.py`.
# Outputs: one `solution.json` per case under
# `results/aerocapture/neptune/trade/<TRADE_NAME>/`, plus `<TRADE_NAME>.csv` (columns in
# `results.COLUMNS`) and `<TRADE_NAME>.png`.

# %% imports
import itertools
import sys
from dataclasses import replace

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aerocapture.core import system  # noqa: E402
from aerocapture.neptune import case  # noqa: E402
from aerocapture.neptune import config  # noqa: E402
from aerocapture.neptune import plots  # noqa: E402
from aerocapture.neptune import results  # noqa: E402
from aerocapture.neptune import vehicles  # noqa: E402
from aerocapture.neptune.vehicles import PRESETS  # noqa: E402

# %% BASE -- every input of each case, with units (field meanings and tags: the CONFIG cell
# of run_neptune.py); fields marked SET PER CASE are overridden in CASES
# Vehicle [physics]: a preset (vehicles.PRESETS: "ellipsled", "heritage-70deg", "loftid";
# placeholder numbers, not sourced designs) with the tuned fields overridden here.
VEHICLE = replace(
    PRESETS["heritage-70deg"],
    beta_kg_m2=200.0,           # ballistic coefficient m/(C_D S), kg/m^2
    lift_to_drag=0.4,           # trim L/D (the bank angle points it)
    nose_radius_m=0.5,          # effective stagnation-point nose radius, m (radiative fit anchored at 0.291 m)
    mass_kg=2000.0,             # entry mass, kg (report-only)
    tps="ablative",             # "ablative" | "non_ablative" (only non_ablative enforces t_wall_max_k)
    emissivity=0.85,            # surface emissivity of the radiative-equilibrium temperature
    t_wall_max_k=None,          # radiative-equilibrium temperature limit, K (None: none)
    q_max_w_cm2=None,           # peak total heat-rate limit, W/cm^2 (None: none)
    g_max=20.0,                 # peak sensed-deceleration limit, Earth g
)

BASE = config.Config(
    # --- arrival [physics] -----------------------------------------------------------------
    vinf_kms=5.0,                       # SET PER CASE; hyperbolic excess speed at Neptune, km/s
    direction="free",                   # v-inf direction: "free" | "radec" | "from_run"
    ra_deg=0.0,                         # v-inf right ascension in NEQ, deg (direction="radec")
    dec_deg=0.0,                        # v-inf declination in NEQ, deg (direction="radec")
    from_run="capture/EJN",             # published run docs/data/<model>/<sequence> (direction="from_run")
    point="p13116",                     # point id in that run's cloud.json (from_run; v_inf 5.0 km/s)
    epoch="2060-01-01 00:00:00 TDB",    # time zero; from_run uses the run's arrival epoch
    # opt flags, None = automatic: with Triton (mode != "none"), a free direction optimises
    # RA/Dec and a fixed direction optimises t_peri; in mode "none" both stay at the guess.
    opt_direction=None,                 # optimise the v-inf RA/Dec
    opt_t_peri=None,                    # optimise the periapsis time (sets Triton's phase)
    t_peri_window_periods=0.5,          # t_peri bound when optimised, +- Triton periods
    r0_km=1.0e6,                        # start radius on the incoming hyperbola, km
    # --- Triton [physics] ------------------------------------------------------------------
    triton_mode="inbound",              # SET PER CASE; "none" | "inbound" (before entry) | "outbound" (after exit)
    triton_window_km=30000.0,           # [numerics] Triton-phase boundary distance, km
    # (flyby floor, clearance and outbound exit distance: margins and assumptions below)
    # --- target orbit [physics] ------------------------------------------------------------
    ra_target_km=None,                  # apoapsis target, km (None: ra_target_hill below)
    ra_mode="max",                      # "max": ra <= target (minimal capture) | "equal": ra = target
    match_triton_plane=False,           # pin the final orbit plane to Triton's
    # --- vehicle [physics] -----------------------------------------------------------------
    vehicle=VEHICLE,                    # SET PER CASE; beta, L/D, nose radius, TPS and limits: see VEHICLE above
    # --- atmosphere and heating [physics] --------------------------------------------------
    interface_alt_km=1000.0,            # entry/exit interface altitude above 24 622 km, km
    conv_model="vision_neptune",        # convective fit: "vision_neptune" | "sutton_graves"
    # --- objective and control limits [physics] --------------------------------------------
    objective="qpeak",                  # "qpeak" (peak q_total) | "heat" | "dv" (raise burn) | "g" (peak g)
    bank_min_deg=-180.0,                # bank-angle lower bound, deg (0 = lift up)
    bank_max_deg=180.0,                 # bank-angle upper bound, deg
    bank_rate_max_degs=15.0,            # bank-rate limit, deg/s
    # --- margins and assumptions [margin]: value, unit; why this value / what it costs ----
    triton_alt_min_km=300.0,            # km; Triton flyby floor; binds at the inbound optimum
                                        #   Triton's air is not modelled (drag < 1 m/s at 300 km)
    triton_clear_km=15000.0,            # km; non-Triton phases stay this far from Triton
                                        #   beyond its SOI (11 970 km): the Triton phase is the only close pass
    triton_exit_km=100000.0,            # km; outbound reads the final orbit this far from Triton
                                        #   Triton's residual pull biases ra ~1.5 % high (ra_mode 'equal' overshoots)
    ra_target_hill=0.1,                 # R_Hill; capture target if ra_target_km is None (0.1: 1.16e7 km)
                                        #   solar tide (not modelled) ~0.1 % of Neptune's pull there, ~8 % at 0.5
    min_inclination_deg=90.0,           # deg; final inclination lower bound (polar or retrograde)
                                        #   the free optimum sits on it (polar minimises airspeed): normally active
    rp_raise_km=system.R_NEPTUNE + 3000.0,# km; target of the reported raise dv and objective "dv"
                                        #   report-only, not a constraint
    rad_coupling="goulard",             # Goulard cooling of the radiative fit | "none" (conservative)
                                        #   the fit may already include it; both are reported, limits use this one
    rad_kappa=3.0,                      # Goulard kappa, the air value (no H2-He value is established)
                                        #   cuts q_total ~1 % at Rn 0.3 m, 22-67 % at Rn 1-3 m and 30-33 km/s
    rad_rn_exp=1.0,                     # radiative-fit nose-radius exponent (optically thin, 0.291 m anchor)
                                        #   q_rad grows linearly with Rn: an upper bound for large noses
    heat_weight=0.0,                    # per J/cm^2; heat-load term added to the objective (0: none)
    sim_tol=0.01,                       # simulate check: largest relative final-apoapsis error
    # --- transcription and solver [numerics] -----------------------------------------------
    order=3,                            # Radau polynomial order per segment
    segments=dict(approach=20, triton_in=20, coast_in=25, atmosphere=40, coast_mid=20,
                  triton_out=24),       # Radau segments per phase
    r_split_km=2.0e5,                   # approach / coast_in boundary radius (none, outbound), km
    max_iter=500,                       # IPOPT iteration limit per solve
    tol=1e-5,                           # IPOPT overall NLP tolerance
    constr_viol_tol=1e-6,               # IPOPT constraint-violation tolerance (scaled)
    print_level=0,                      # IPOPT print level (0: quiet for trades)
    linear_solver="mumps",              # IPOPT linear solver
    # --- initial guess [guess] (starting point only) ---------------------------------------
    guess_inclination_deg=92.0,         # guess plane inclination, deg (None or match_triton_plane: Triton's plane)
    guess_turn_fraction=0.7,            # inbound Triton turn, fraction of the maximum at triton_alt_min_km
    guess_banks_deg=(0.0, 45.0, 75.0, 90.0, 105.0, 135.0),  # constant-bank candidates, deg
    guess_capture_fraction=1.0,         # guess exit energy = fraction * capture bound
    # --- output ----------------------------------------------------------------------------
    tag=None,                           # SET PER CASE; results/aerocapture/neptune/<tag>/ (None: from mode, v_inf, vehicle)
)

# %% TRADE -- traded values (None = the preset's own value)
TRADE_NAME = "vehicle_trade"                 # results/aerocapture/neptune/trade/<TRADE_NAME>/
MODES = ["inbound"]                          # Triton modes
VINF_LIST_KMS = [5.0]                        # v-inf values, km/s
VEHICLES = ["heritage-70deg", "loftid"]      # vehicles.PRESETS keys
BETA_LIST_KG_M2 = [None]                     # ballistic coefficients, kg/m^2
LD_LIST = [None]                             # lift-to-drag ratios
RN_LIST_M = [None, 1.0]                      # nose radii, m

# %% CASES -- the only overrides of BASE
def variant_case(mode, vinf, preset, beta, ld, rn):
    """Return BASE with the traded mode, v-inf and vehicle variant (None keeps the preset value)."""
    veh = vehicles.get(preset, beta_kg_m2=beta, lift_to_drag=ld, nose_radius_m=rn)
    vtag = f"b{veh.beta_kg_m2:g}_ld{veh.lift_to_drag:g}_rn{veh.nose_radius_m:g}"
    return replace(BASE, triton_mode=mode, vinf_kms=vinf, vehicle=veh,
                   tag=f"trade/{TRADE_NAME}/{mode}_vinf{vinf:g}_{preset}_{vtag}")


CASES = [variant_case(*c) for c in itertools.product(MODES, VINF_LIST_KMS, VEHICLES,
                                                     BETA_LIST_KG_M2, LD_LIST, RN_LIST_M)]

# %% run (each case prints its summary and writes its solution.json)
OUT = case.RESULTS_DIR / "trade" / TRADE_NAME
rows = case.solve_chain(CASES)

# %% table (CSV columns documented in results.COLUMNS)
results.write_csv(rows, OUT / f"{TRADE_NAME}.csv")
results.print_table(rows, ["mode", "vinf_kms", "vehicle", "beta_kg_m2", "ld", "rn_m", "converged",
                           "start", "triton_used", "peak_g", "peak_q_conv_w_cm2",
                           "peak_q_rad_coupled_w_cm2", "peak_q_rad_uncoupled_w_cm2",
                           "peak_q_total_w_cm2", "heat_load_j_cm2", "peak_t_radeq_k",
                           "final_ra_km", "capture_margin_km2s2", "sim_passed"])

# %% plots
fig = plots.trade_figure(rows, f"vehicle trade (objective {BASE.objective})")
if "ipykernel" in sys.modules:
    plt.show()
plots.save_all({TRADE_NAME: fig}, OUT)
print(f"wrote {OUT}")
