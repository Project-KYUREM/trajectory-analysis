# Neptune aerocapture with a Triton gravity assist — Dymos design

Sister of `aerocapture/triton/` (Triton ballute capture); both share
`aerocapture/core/` and the same workflow, headings and input conventions.
Decisions are in §8, every input in §9 (margins and assumptions included),
what the solver taught us in §10.

## 1. Mission

A spacecraft arrives at Neptune on a hyperbola with a chosen `v∞`. It may fly
past Triton before entry (`inbound`), after exit (`outbound`) or not at all
(`none`), dips into Neptune's atmosphere once, and uses bank-angle modulation
to bleed off just enough energy to be captured (minimal capture: apoapsis at
most 0.1 Neptune Hill radii by default) on a retrograde-or-polar orbit
(inclination ≥ `min_inclination_deg`, default 90°).

Dymos solves this as one multi-phase optimal-control problem. The B-plane aim
point, the bank-angle history and, depending on the opt flags (§4), the v∞
direction or the periapsis time are optimised together to minimise the peak
stagnation heat rate (default) subject to g-load, heating, Triton-altitude and
capture constraints.

## 2. Physics sizing (patched conics, numbers from the repo's SPICE kernels)

| quantity | value |
|---|---|
| GM Neptune / GM Triton | 6 835 103 / 1 428.5 km³/s² |
| Neptune radius used for altitude | 24 622 km (mean; matches the GRAM table) |
| Neptune rotation period | 15.966 h → 2.71 km/s at the equator |
| Neptune J2 | 3408.428e-6 at its reference radius 25 225 km (Jacobson 2009; not in the PCK) |
| Triton orbit | a = 354 760 km, e = 2e-4, P = 5.88 d, v = 4.39 km/s |
| Triton inclination | 157.3° to Neptune's equator (retrograde), 129° to Neptune's orbit |
| Triton radius / SOI / Hill | 1 352.6 / 11 970 / 14 600 km |

One Triton flyby at 300 km altitude, best-case geometry:

| v∞ (km/s) | speed rel. Triton | max Δv from Triton | entry speed, inertial | entry speed vs retrograde atmosphere |
|---|---|---|---|---|
| 3  | 7.2  | 235 m/s | 23.3 | 26.1 |
| 5  | 8.3  | 207 m/s | 23.6 | 26.4 |
| 10 | 11.9 | 144 m/s | 25.2 | 28.0 |
| 15 | 16.3 | 106 m/s | 27.5 | 30.3 |
| 20 | 21.0 |  82 m/s | 30.6 | 33.4 |

Kyurem's published fronts arrive at 3–23 km/s, so the whole range is accepted.

Consequences:

1. **Triton cannot capture the spacecraft by itself.** It supplies 0.1–0.25 km/s;
   capture needs ~1.5 km/s (v∞ = 5) to ~8 km/s (v∞ = 20). The atmosphere does the
   capture and Triton trims it, so the Triton phase must be modelled accurately.
2. **Retrograde costs heating.** Flying against Neptune's rotation adds up to
   ~2.7 km/s of airspeed; convective heating scales roughly as v³. Aerodynamics
   always use the atmosphere-relative velocity `v − ω × r`.
3. **Retrograde is what makes Triton useful.** Triton orbits retrograde, so a
   retrograde spacecraft meets it at lower relative speed.
4. **Time scales span five orders of magnitude** (days on approach, ~20 min near
   Triton, ~10 min in the atmosphere), which forces separate phases and makes the
   initial guess the main convergence risk (§5).

## 3. Model

**Frame.** Neptune-centred inertial, axes on Neptune's IAU equator and prime
meridian frozen at the arrival epoch (pole = z). ECLIPJ2000 inputs are rotated in
once at setup.

**States** (every phase, Cartesian, km and km/s): `x y z vx vy vz`, free of the
singularities spherical states hit in retrograde and polar geometry. The
atmosphere phase adds the heat load `Q` (J/cm²).

**Gravity.** Neptune point mass + J2 everywhere; Triton as a third body (direct
and indirect terms) in every vacuum phase, on an analytic circular orbit fitted
to `nep097.bsp` at the epoch. The Sun is neglected (~1e-9 of Neptune's pull at
Triton's orbit; see §4 for the apoapsis target).

**Atmosphere.** Neptune-GRAM mean density table from AMAT
(`neptune-gram-avg.dat`, 5 km steps), downloaded into `data/` by
`scripts/fetch_kernels.sh` because it is GPL. log ρ is interpolated with an
Akima spline over the full table (−85 to 4000 km) and continued log-linearly
(exponentially) beyond it (`DensityComp`, analytic derivatives). The atmosphere co-rotates.

**Aerodynamics.** Bank angle σ (0 = lift up). Drag `½ρ v_rel² S C_D` along
−v_rel; lift `(L/D)·D` normal to v_rel, rotated by σ from local up.

**Vehicle** (`vehicles.Vehicle`). The ODE sees only β = m/(C_D S), L/D and the
nose radius R_n. The TPS type, emissivity, `t_wall_max_k`, `q_max_w_cm2` and
`g_max` set the path limits; mass is report-only. Presets (`ellipsled`,
`heritage-70deg`, `loftid`) hold placeholder numbers; the CONFIG cell shows a
preset with every field overridden through `dataclasses.replace`.

**Heating** (stagnation point, ρ in kg/m³, airspeed v in m/s, R_n in m):

```
q_conv = C_c · ρ^a · v^b · (R_n_ref / R_n)^0.5          [W/cm²]
q_rad  = C_r · ρ^c · (v / 10 km/s)^d · (R_n / R_n_ref)^e  [W/cm²]
```

- Convective: Vision Neptune fit (2005, as in AMAT), C_c = 2.24008e-7,
  a = 0.4521, b = 2.6918, R_n_ref = 0.291 m (`conv_model="vision_neptune"`); or
  Sutton–Graves with the H₂–He constant (`"sutton_graves"`), within ~25 %.
- Radiative: Vision Neptune fit, C_r = 8.1258e-3, c = 0.4981, d = 15.113,
  e = `rad_rn_exp` (1). Its steep velocity dependence matters at the 26–33 km/s
  retrograde airspeeds, and it is what makes the nose-radius trade two-sided
  (convective falls as 1/√R_n, radiative grows with R_n).
- Validity: the radiative fit is anchored at R_n = 0.291 m and assumes an
  optically thin shock layer, so for large noses it is an upper bound; the
  run warns when R_n is outside 0.097–0.873 m.
- Goulard coupling (`rad_coupling="goulard"`, κ = `rad_kappa` = 3, the air
  value; no H₂–He value is established):
  `q_rad,coupled = q_rad,unc / (1 + κ Γ^0.7)`, `Γ = 2 q_rad,unc / (½ ρ V³)`.
  It cuts total heat rate by ~1 % at 27 km/s, R_n = 0.3 m and by 22–67 % at
  30–33 km/s, R_n = 1–3 m. Whether the Vision Neptune fit is already coupled is
  unconfirmed, so `rad_coupling="none"` exists and every run reports both.
  Constraints and the objective use the selected one.
- Radiative-equilibrium wall temperature `T_radeq = (q_total / (ε σ))^¼` is
  always reported and constrained by `t_wall_max_k` only for
  `tps="non_ablative"`.
- Out of scope (conservative by ~10–20 %): reradiation, hot-wall correction,
  ablation blowing; incident radiation is fully absorbed.

Components are vectorised `ExplicitComponent`s with complex-step partials and
partial coloring; `DensityComp` and `core.ode.AltitudeComp` are analytic.

## 4. Dymos problem

### Phases (Radau, `compressed=True`, order 3)

| mode | phases | what Triton does |
|---|---|---|
| `none`     | approach → coast_in → atmosphere | baseline |
| `inbound`  | approach → triton_in → coast_in → atmosphere | lowers the entry energy |
| `outbound` | approach → coast_in → atmosphere → coast_mid → triton_out | finishes the capture after exit; atmosphere-exit energy left free |

| phase | from → to | boundary conditions | path constraints |
|---|---|---|---|
| `approach`   | r₀ on the incoming conic → r_split (none, outbound) or d_T = `triton_window_km` (inbound) | initial state = arrival conic | d_T ≥ `triton_clear_km` |
| `triton_in` / `triton_out` | d_T = window → window (in) or `triton_exit_km` (out) | ḋ_T ≤ 0 at start, ≥ 0 at end | d_T ≥ R_T + `triton_alt_min_km` |
| `coast_in`   | → entry interface | r = R_I, r·v ≤ 0 | d_T ≥ `triton_clear_km` |
| `atmosphere` | interface → interface | r = R_I, r·v ≥ 0 at exit | alt ≥ 0, g ≤ g_max, bank rate, optional q_total ≤ q_max and T_radeq ≤ T_wall,max, q_total ≤ q_peak (objective slack) |
| `coast_mid`  | exit interface → Triton window (outbound) | d_T = window | d_T ≥ `triton_clear_km` |

All phases link on time and the six states. Segment ends are clustered from the
guess (∫dt/r for coasts, ∫dt/d_T near Triton, density-weighted in the
atmosphere).

**`both` is rejected (geometrically impossible).** Between the inbound and
outbound crossings of Triton's orbit the spacecraft sweeps ~260–325° around
Neptune while Triton advances only ~40–100° in the same sense, a mismatch of
≥ ~205° for v∞ 3–20 km/s; tilting the plane cannot help because both
encounters would have to lie on the line of nodes. `triton_mode="both"` raises a
`ValueError` with this explanation.

### Arrival parametrisation and opt flags

`ArrivalComp` maps (v∞, RA, Dec, B·T, B·R, t_peri) to the conic state at r₀;
equality constraints tie it to the first approach node. |v∞| is fixed, B·T and
B·R are always optimised. Two flags pick the rest (None = automatic rule,
resolved by `Config.resolved()` and shown in the echo and `solution.json`):

| flag | automatic value | why |
|---|---|---|
| `opt_direction` | True when `direction="free"` and `triton_mode != "none"` | the free optimum is the plane that minimises airspeed (§10) |
| `opt_t_peri` | True when `direction != "free"` and `triton_mode != "none"` | with a fixed asymptote, t_peri sets Triton's phase; bounded by ±`t_peri_window_periods` Triton periods |

In `none` mode both stay at the guess, also with a fixed direction: the problem is then nearly invariant to
rotations about the pole and to shifts along the orbit, and those null
directions stall IPOPT (§10).

A fixed direction comes from `radec` or `from_run` (the v∞ vector and arrival
epoch of `docs/data/<model>/<sequence>/cloud.json`). The guess is built around
the fixed asymptote and solved directly; if that fails, a free-direction solve
is continued to the requested asymptote in ≤ 10° steps (`case.solve`). An
inbound flyby with a fixed asymptote exists only if the asymptote is within
~10–20° of Triton's orbit plane; outbound has no such restriction.

### Target orbit

- **Capture:** the final osculating two-body orbit (atmosphere exit for
  none/inbound, Triton-phase exit for outbound) satisfies
  `energy + μ/(ra_target + rp) ≤ 0` (`ra_mode="max"`) or `= 0` (`"equal"`), i.e.
  ra ≤ ra_target written smoothly through parabolic. Default ra_target =
  0.1 R_Hill ≈ 1.16e7 km: the exit is then ~25 m/s below escape (~5 m/s at
  0.5 R_Hill), while the unmodelled solar tide grows from ~0.1 % to ~8 % of
  Neptune's pull; a target beyond 0.1 R_Hill is warned about.
- **Inclination:** final i ≥ `min_inclination_deg` (90° default); optional
  `match_triton_plane` pins the final plane to Triton's.
- **Reported only:** final ra, rp, period, and the two-body periapsis-raise Δv at
  apoapsis to `rp_raise_km` (R_N + 3000 km).

### Objective

`qpeak` (default) minimises the peak total heat rate through a slack parameter
q_peak with q_total − q_peak ≤ 0; `heat` the heat load; `dv` the reported
periapsis-raise Δv; `g` the peak g (slack). `heat_weight` adds a heat-load term.

### Optimizer

IPOPT via pyOptSparse with total-derivative coloring, L-BFGS Hessian, adaptive μ,
gradient-based NLP scaling (`tol`, `constr_viol_tol`, `max_iter` from the Config).

## 5. Initial guess and staged solve

1. The guess plane has inclination `guess_inclination_deg` (92°, just above the
   bound where the optimum sits, §10) and shares Triton's ascending node; the
   entry point is rotated in-plane until the Triton crossing lies on Triton's
   circle.
2. Entry speed from |v∞|; inbound adds a patched-conic flyby with
   `guess_turn_fraction` of the maximum turn at the minimum altitude.
3. The constant bank in `guess_banks_deg` with the lowest peak heat rate inside
   the g limit is flown with scipy; the entry flight-path angle is root-found so
   the exit energy is `guess_capture_fraction` × the capture bound (outbound: the
   pre-flyby energy the patched conic needs). Lift-up and lift-down root-finds
   give the reported entry corridor.
4. Full-dynamics shooting (the same dynamics as the ODEs) closes v∞ at r₀ and the
   flyby geometry; outbound raises the plane inclination until the post-flyby
   orbit is retrograde.
5. The entry time is shifted by whole Triton periods so t_peri is within half a
   period of zero; arcs are cut at the phase boundaries.

Staged solve (`case.solve`): a free direction is solved from the guess, with
warm restarts from the last iterate; a fixed direction (`radec`, `from_run`)
is solved directly from a guess built around the asymptote, or reached by
rotating a free solution's asymptote in ≤ 10° steps. Every solve ends with the
Dymos simulate check. `case.solve_chain` (sweeps, trades) warm-starts a case
from the last accepted one when only v∞ or the vehicle's β, L/D, R_n change.

## 6. Code layout and workflow

Shared modules live in `aerocapture/core/`, mission modules in `aerocapture/neptune/`;
imports are package-absolute (`from aerocapture.core import system`).

| file | role |
|---|---|
| `core/config_base.py` | `ConfigBase`: validation, opt-flag resolution, margins-and-assumptions registry, record, echo |
| `neptune/config.py` | `Config`: every input in CONFIG-cell groups, `ASSUMPTIONS`, automatic opt-flag rule, warnings |
| `neptune/vehicles.py` | `Vehicle` record, placeholder presets, `get(preset, **overrides)` |
| `core/system.py`, `core/paths.py` | kernels, constants, NEQ frame, Triton circular fit, Hill radius |
| `core/atmosphere.py`, `core/heating.py` | GRAM spline and `DensityComp`; heating fits, Goulard, T_radeq |
| `core/dynamics.py`, `core/ode.py`, `neptune/ode.py` | shared dynamics and orbit metrics, ODE building blocks; `VacuumODE`, `AtmosphereODE` |
| `core/arrival.py` | `ArrivalComp`, published-run v∞ reader |
| `core/guess.py`, `neptune/guess.py` | shared propagation and shooting numerics; patched-conic design, scipy shooting, segment clustering |
| `neptune/problem.py` | `AeroProblem`: builds, seeds and solves the trajectory from a `Config` |
| `core/solve.py`, `neptune/case.py` | IPOPT driver and stage helpers; Config → epoch/arrival (`resolve_arrival`) → guess → staged `solve`; `warm_start`, `solve_chain` |
| `core/io.py`, `neptune/results.py` | JSON/CSV/table writers and the printed assumptions block; summary, simulation check, active constraints, warnings, CSV rows (`COLUMNS`) |
| `core/plotting.py`, `neptune/plots.py` | plot primitives; system view, atmosphere panels, corridor, Triton close-up, guess, sweep and trade figures |
| `neptune/run_neptune.py` | `# %%` cells: one case |
| `neptune/sweep_neptune.py` | `# %%` cells: v∞ list × Triton modes, warm-started in v∞ |
| `neptune/trade_neptune.py` | `# %%` cells: presets × β × L/D × R_n × v∞ × mode, warm-started in β, L/D, R_n |
| `tests/core/`, `tests/neptune/` | pytest checks (§7) |

Workflow: edit the CONFIG cell of `run_neptune.py` (every field with value,
unit and meaning, tagged [physics], [numerics], [guess] or [margin]) and run
the cells: imports → CONFIG → initial guess (+ figures) → solve → summary →
figures → save. The sweep and trade scripts list the same fields as BASE
(per-case fields marked SET PER CASE), then the SWEEP / TRADE lists, a CASES
cell with the only overrides, run, table and figure. The Triton mission's
scripts have the same cells.

Outputs, `results/aerocapture/neptune/<tag>/`: `solution.json`, `guess_system.png`,
`guess_atmosphere.png`, `system.png`, `atmosphere.png`, `corridor.png`, `triton.png`, and the
Dymos solution / simulation `.db`. Sweeps and trades write per-case directories
under `sweep/<name>/` or `trade/<name>/` plus `<name>.csv` and `<name>.png`.
OpenMDAO reports go under `results/aerocapture/neptune/_openmdao/`.

The printed summary follows the section order shared with the Triton
mission: header, solver (IPOPT status, iterations, wall time, stages),
arrival (optimised parameters, resolved opt flags), Triton flyby, entry/exit,
loads, heating with coupled **and** uncoupled radiation, final orbit, a checks
line (converged, triton_used, simulate), the "Assumptions & margins in effect"
block (§9: every [margin] field with value and reason, the fixed model
assumptions, the constraints active at the solution, numerical box bounds
hit) and every warning: target beyond 0.1 R_Hill, R_n far from 0.291 m, guess
inclination below the bound, unused Triton flyby, final periapsis below
Neptune's surface, failed simulate check, incomplete continuation, active box
bound.

## 7. Verification

- `tests/neptune/test_components.py`, `tests/core/test_components.py`, `tests/core/test_solve.py`: `check_partials` on the ODEs, `core.ode.AltitudeComp`,
  `DensityComp`, `ArrivalComp`; a J2-off, Triton-off Dymos coast reproduces the
  analytic Kepler orbit and conserves energy.
- `tests/core/test_physics.py`, `tests/neptune/test_guess_consistency.py`: Triton fit vs nep097; GRAM spline; hyperbola ↔ B-plane
  round trip; Goulard and T_radeq formulas; guess dynamics = ODE dynamics;
  retrograde airspeed; capture metric; `both` rejected; full-dynamics flyby Δv
  within 5 % of the patched conic.
- Triton closest approach (summary and constraint) is the periapsis of the
  Triton-centred osculating conic at the node nearest Triton.
- Every solve: Dymos `simulate` re-integrates each phase from its collocated
  initial state; the final apoapsis must agree within `sim_tol` (1 %).
- Not done: an AMAT cross-check case.

## 8. Decisions

| # | topic | decision |
|---|---|---|
| 1 | Triton placement | `none`, `inbound`, `outbound`; `both` rejected as geometrically impossible |
| 2 | Target orbit | minimal capture, ra ≤ 0.1 R_Hill by default, `ra_mode` max/equal; i ≥ `min_inclination_deg` (90°); periapsis-raise Δv to R_N + 3000 km reported only |
| 3 | Objective | minimise peak stagnation heat rate (`qpeak`); `heat`, `dv`, `g` available |
| 4 | Vehicle | user-set β, L/D, R_n, TPS, limits; placeholder presets; `trade_neptune.py` sweeps them |
| 5 | g-load limit | 20 g default, per vehicle |
| 6 | Heating | Vision Neptune convective + radiative fits (Sutton–Graves alternative), Goulard coupling by default, coupled and uncoupled always reported; T_radeq constrained only for non-ablative TPS |
| 7 | v∞ | 5 km/s default; direction free, fixed RA/Dec, or from a published kyurem run |
| 8 | Arrival freedom | `opt_direction` / `opt_t_peri` with an automatic rule (§4), overridable |
| 9 | Inputs | one `Config` dataclass, every field spelled out in the scripts' CONFIG cell, echoed resolved and saved with each run; no CLI |
| 10 | Margins | every margin, assumed value, weight and check tolerance in one [margin] group, printed with each summary next to the active constraints |

## 9. Inputs (`config.Config`; every field spelled out in the CONFIG cell)

Groups in CONFIG-cell order. Tags: [physics], [numerics], [guess], [margin].

| field | default | meaning |
|---|---|---|
| **arrival** [physics] | | |
| `vinf_kms` | 5.0 | hyperbolic excess speed, km/s |
| `direction` | `free` | `free` / `radec` / `from_run` |
| `ra_deg`, `dec_deg` | 0, 0 | fixed v∞ direction in NEQ, deg (`radec`) |
| `from_run`, `point` | `capture/EJN`, `p13116` | published run and point (`from_run`) |
| `epoch` | 2060-01-01 TDB | time zero (`from_run`: the arrival epoch) |
| `opt_direction`, `opt_t_peri` | None, None | optimise v∞ RA/Dec / periapsis time; None: automatic (§4) |
| `t_peri_window_periods` | 0.5 | t_peri bound, ± Triton periods |
| `r0_km` | 1e6 | start radius, km |
| **Triton** [physics] | | |
| `triton_mode` | `inbound` | `none` / `inbound` / `outbound` |
| `triton_window_km` | 30 000 | [numerics] Triton-phase boundary distance, km |
| **target orbit** [physics] | | |
| `ra_target_km` | None | apoapsis target, km (None: `ra_target_hill`) |
| `ra_mode` | `max` | `max` (≤) or `equal` |
| `match_triton_plane` | False | pin the final plane to Triton's |
| **vehicle** [physics] | heritage-70deg | `Vehicle(name, beta_kg_m2, lift_to_drag, nose_radius_m, mass_kg, tps, emissivity, t_wall_max_k, q_max_w_cm2, g_max)` |
| **atmosphere and heating** [physics] | | |
| `interface_alt_km` | 1000 | entry/exit interface altitude, km |
| `conv_model` | `vision_neptune` | or `sutton_graves` |
| **objective and control limits** [physics] | | |
| `objective` | `qpeak` | `qpeak` / `heat` / `dv` / `g` |
| `bank_min_deg`, `bank_max_deg`, `bank_rate_max_degs` | −180, 180, 15 | bank bounds, deg, and rate limit, deg/s |
| **margins and assumptions** [margin] | | (table below) |
| **transcription and solver** [numerics] | | |
| `order`, `segments`, `r_split_km` | 3, per phase, 2e5 | Radau order, segments per phase, approach/coast_in split radius, km |
| `max_iter`, `tol`, `constr_viol_tol`, `print_level`, `linear_solver` | 500, 1e-5, 1e-6, 5, mumps | IPOPT |
| **initial guess** [guess] | | |
| `guess_inclination_deg`, `guess_turn_fraction`, `guess_banks_deg`, `guess_capture_fraction` | 92, 0.7, 0–135°, 1.0 | guess plane, Triton turn, bank candidates, exit-energy fraction |
| **output** | | |
| `tag` | None | results sub-directory |

Margins and assumptions (`config.ASSUMPTIONS`; printed with every summary):

| field | default | kind | why this value / what it costs |
|---|---|---|---|
| `triton_alt_min_km` | 300 km | margin | flyby floor; Triton's atmosphere is not modelled (ρ ≈ 3e-9 kg/m³ at 300 km, drag Δv < 1 m/s); binds at the inbound optimum, so a lower floor buys a little more Triton Δv |
| `triton_clear_km` | 15 000 km | margin | non-Triton vacuum phases stay beyond Triton's SOI (11 970 km), so the Triton phase is the only close pass; normally inactive |
| `triton_exit_km` | 100 000 km | model | outbound reads the final orbit here; Triton's residual pull biases the Neptune-frame apoapsis ~1.5 % high (`ra_mode="equal"` overshoots by that much); larger shrinks the bias |
| `ra_target_hill` | 0.1 R_Hill | model | two-body capture target (1.16e7 km); the unmodelled solar tide is ~0.1 % of Neptune's pull there (~8 % at 0.5); exit ~25 m/s below escape |
| `min_inclination_deg` | 90° | requirement | polar-or-retrograde final orbit; the free optimum sits on it (§10), so it is normally active |
| `rp_raise_km` | R_N + 3000 km | requirement | target of the reported periapsis-raise Δv and of objective `dv`; not a constraint |
| `rad_coupling` | `goulard` | model | Goulard cooling of the Vision Neptune radiative fit (whether the fit already includes it is unconfirmed); `none` is conservative; both reported |
| `rad_kappa` | 3 | model | Goulard κ, the air value (no H₂–He value established); cuts q_total ~1 % at R_n 0.3 m, 22–67 % at R_n 1–3 m |
| `rad_rn_exp` | 1 | model | radiative-fit nose-radius exponent (optically thin, anchored at 0.291 m): an upper bound for large noses |
| `heat_weight` | 0 per J/cm² | weight | heat-load term in the objective; 0 leaves the objective pure |
| `sim_tol` | 0.01 | check | largest relative final-apoapsis error of the simulate re-integration |

Fixed model assumptions (`results.NOTES`): no solar perturbations; cold-wall,
non-ablating stagnation heating (conservative by ~10–20 %); Triton on a
circular fit to nep097; placeholder vehicle presets.

## 10. Findings

- **The heating optimum is a polar orbit.** Rotation adds airspeed for
  retrograde orbits (~+2.5 km/s at i = 157°) but almost none for a polar one
  (v_rel ≈ √(v² + (ωR)²)). With i ≥ 90° every mode converges onto i = 90°, so the
  free-direction optimum sits on `min_inclination_deg`; guesses in Triton's plane
  lose ~25 % in peak heating and IPOPT crawls toward the polar plane.
- **Near-null directions stall IPOPT's quasi-Newton.** Rotations of the whole
  trajectory are nearly free; with an L-BFGS Hessian IPOPT then creeps or, with
  loose constraint scaling, diverges. Tight linkage/match scaling, a
  near-optimal guess plane and the opt-flag rule give converged solves.
- **Minimal capture is gentle at v∞ = 5 km/s:** ~0.55 km/s of drag Δv, peak load
  ~0.35 g, minimum altitude ~345 km for the heritage preset.
- **Triton buys ~3 % in peak heat rate (290 vs 299 W/cm²).** The inbound flyby
  sits on the 300 km bound with the maximum turn, but changes the Neptune-frame
  energy by only V_T·Δv ≤ 0.86 km²/s² (0.78 achieved), applied far out where the
  spacecraft is slow: ~6 % of the energy the atmosphere removes, ~6 % thinner
  peak density, and with q_conv ∝ ρ^0.45 ~3 %. Outbound is weaker at the free
  optimum (ΔE ≈ −0.2 km²/s²) because the inclination bound is active.
- **Plane matters more than Triton.** `match_triton_plane` (157°) raises the
  inbound peak heat rate from 290 to 365 W/cm²; capturing to ra = 360 000 km
  (`ra_mode="equal"`) raises the `none` case from 299 to 411 W/cm².
- Outbound reads the final orbit `triton_exit_km` (1e5 km) from Triton, where
  Triton's residual potential biases the Neptune-frame apoapsis ~1.2–1.7 % high;
  `ra_mode="max"` holds with margin but `"equal"` overshoots by ~1.5 % (the run
  warns; a larger `triton_exit_km` shrinks it).
- An outbound flyby can leave the final osculating periapsis below the surface;
  the run warns, and the reported periapsis-raise Δv (~10 m/s) covers it.

## References

- AMAT, Girija et al. 2021, JOSS 10.21105/joss.03710; Neptune GRAM table and
  heating fits: https://github.com/athulpg007/AMAT
- Girija et al., "Feasibility and Performance Analysis of Neptune Aerocapture
  Using Heritage Blunt-Body Aeroshells", J. Spacecraft & Rockets.
- Lockwood et al., Neptune Aerocapture Systems Analysis, NASA (2004–2006), NTRS
  20040111217 / 20040111219.
- Sutton & Graves (1971), H₂/He stagnation heating constants.
- Goulard radiative-cooling parameter and the Tauber–Wakefield correlation
  q_rad,c/q_rad,u = 1/(1+κΓ^0.7) (κ = 3 for air); see Anderson, *Hypersonic and
  High-Temperature Gas Dynamics*.
- NASA Vision Neptune Orbiter with Probes, Contract NNH04CC41C final report
  (2005): Neptune convective and radiative heating fits (as in AMAT `vehicle.py`).
- SPICE: de440, nep097, pck00011, gm_de440.
