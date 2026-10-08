# Triton ballute aerocapture after a close Neptune flyby — Dymos design

Sister of `aerocapture/neptune/` (Neptune aerocapture with a Triton flyby);
both share `aerocapture/core/` (kernels, frame, Triton fit, gravity, conics,
Dymos/IPOPT plumbing, reporting) and the same workflow, headings and input
conventions. Sizing numbers come from the repo constants
(`aerocapture/core/system.py`) and scipy prototypes. Decisions are in §8,
every input in §9 (margins and assumptions included), findings in §10.

## 1. Mission

Hyperbolic arrival at Neptune (v∞ user-set, default 5 km/s) → close Neptune
flyby with periapsis 1.10–1.25 R_N (user band, outside the atmosphere) →
~10 h cruise to Triton → **ballute aerocapture** in Triton's N₂ atmosphere
(drag modulation by a free-time ballute release, after the NIAC "Pluto Hop, Skip,
and Jump" enveloping-decelerator concept) → coast to apoapsis → periapsis-raise
burn → **stable Triton orbit** (configurable). Default objective: minimise the
peak stagnation heat rate on the ballute.

## 2. Physics sizing

R_N = 24 622 km (repo altitude reference); a_T = 354 759 km, V_T = 4.390 km/s,
i_T = 157.3°, Triton fit residual vs nep097 = 17.9 km over ±6 d; μ_T = 1428.5
km³/s², R_T = 1352.6 km, Triton Hill radius R_H = 14 597 km; escape speed at the
1000 km interface 1.10 km/s.

### 2.1 Triton encounter speed (outbound crossing of r = a_T, in-plane)

| v∞ | rp/R_N | Neptune turn | true anomaly at a_T | v_rel co-moving / polar / counter (km/s) | entry speed at 1000 km, co / polar / counter |
|---|---|---|---|---|---|
| 3 | 1.10–1.25 | 150–148° | 145–142° | 7.18–7.11 / 8.17 / 9.06–9.11 | 7.27–7.20 / 8.25 / 9.12–9.18 |
| 5 | 1.10–1.25 | 131–128° | 140–137° | 8.21–8.14 / 9.10 / 9.91–9.97 | 8.28–8.22 / 9.17 / 9.97–10.03 |
| 7 | 1.10–1.25 | 114–110° | 134–131° | 9.54–9.48 / 10.33 / 11.07–11.12 | 9.60–9.54 / 10.39 / 11.13–11.18 |

- The close pass makes the Triton crossing near-radial (flight-path angle ≈ 77°
  at a_T), so the periapsis band changes the encounter speed by only ±0.05 km/s;
  the encounter sense matters (co-moving = trajectory in Triton's retrograde
  plane, i ≈ 157°).
- A direct tangent intercept (no close pass) gives 3.75 km/s at v∞ = 5: **the
  close flyby costs ~4.5 km/s of Triton entry speed.** It buys re-pointing
  (~130° turn), Voyager-like science and phasing freedom, not energy (Neptune-frame
  energy is unchanged). Every run reports this direct-intercept comparison in its
  printed summary. An optional Oberth burn at Neptune periapsis is available
  (0.2 km/s takes v∞ 5 → 4.0), default off.
- Check against Brisby & Lyne 2024 (v∞ = 5.44, θi = 0–90°): the same formula gives
  4.01–9.41 km/s entry, identical to their published range (unit test).
- The Neptune flyby comes first, so the Triton encounter is on the outbound leg.
- Neptune periapsis → Triton ≈ 9.9 h (v∞ 5); Triton moves ~25° meanwhile.

### 2.2 Capture requirement

Exit speed at the interface for ra = 0.3 R_H with periapsis in the atmosphere
≈ 0.97 km/s ⇒ remove **7.3 km/s (v∞ 5, co-moving) to 9 km/s (counter)**, i.e.
over 98.6 % of the kinetic energy. Required column mass along the chord
≈ β ln(v_E/v_X) ≈ 2.2 β; a grazing chord to the surface offers only ≈ 48 kg/m²
⇒ **β ≲ 2–3 kg/m²**. Prototype (planar, release on energy, h_min 10 km):
β 0.5 / 1 / 2 → corridor 0.7 / 0.4 / 0.2°; β ≥ 5 → no capture. **A LOFTID-class
aeroshell (β ≈ 23) cannot capture at Triton**, consistent with Brisby & Lyne
using it only for an aerogravity assist into Neptune orbit.

### 2.3 Ballute pass (prototype; enveloping sphere, C_D 1, probe β 60–100 after release, ra ≤ 5000 km)

| v_E (km/s) | D (m) / mass (kg) | β (kg/m²) | corridor γ_E at 1000 km | peak q, Sutton–Graves N₂ | peak g | min alt | film T, 2-sided, ε 0.85 |
|---|---|---|---|---|---|---|---|
| 8.3 | 30 / 400 | 0.57 | −53.0…−53.5° (0.50°) | 2.3 W/cm² | 12 | 46 km | ~710 K |
| 8.3 | 70 / 530 | 0.14 | −52.0…−53.0° (1.00°) | 0.73 | 13 | 70 | ~535 K |
| 8.3 | 100 / 880 | 0.11 | −51.75…−52.75° (1.00°) | 0.54 | 12 | 78 | ~495 K |
| 10.0 | 30 / 400 | 0.57 | 0.25° | 3.9 | 17 | 40 | ~800 K |
| 10.0 | 70 / 530 | 0.14 | 0.75° | 1.26 | 18 | 65 | ~610 K |

- **Release is what makes it flyable:** a retained ballute has a ~0.1° corridor;
  a free-time release widens it ~10×.
- Peak heating occurs at ρ ≈ 2–8e-6 kg/m³, Kn(D) ≈ 3e-4–1e-3 ⇒ continuum at the
  peak; the upper pass is transitional/free-molecular and carries negligible
  heat. Dynamic pressure 16–60 Pa (NIAC Pluto: 15–42 Pa).
- Calibration: NIAC Pluto 3.5 W/cm² → 520–550 °C film. Two-sided radiative
  equilibrium with ε = 0.85 gives 776 K (503 °C) ⇒ default `radiating_sides = 2`.
- Radiative heating: the Tauber–Sutton air correlation (optically thin, linear in
  R_n) gives ~19 W/cm² at 10 km/s, ρ 5e-6, R_n 35 m — an order of magnitude above
  convective. The NIAC LAURA solutions at Pluto (N₂, 14 km/s, 70 m envelope) found
  radiation minimal, so the correlation is treated as a non-binding **upper
  bound**: computed and reported every run, constrained only if the user opts in.
- Post-release periapsis is 20–80 km altitude ⇒ **a periapsis-raise burn is
  required**: 41 m/s (→ 500 km) / 79 m/s (→ 1000 km) at ra = 0.3 R_H. Neptune's
  tide (2.7 % of Triton's pull at 0.3 R_H) contributes ≲ 20 m/s per half
  revolution; burn time and vector are free so the optimiser can use it.

### 2.4 Neptune flyby leg

- 1.10 R_N = 27 084 km = 2 462 km altitude; Neptune-GRAM ρ ≈ 1.4e-13 kg/m³ ⇒ drag
  Δv ≲ 0.1 m/s even with the ballute inflated. **Flyby phases are vacuum**; the
  GRAM drag integral is reported after the solve (reuses `aerocapture/core/atmosphere.py`).
- Voyager 2 template: closest approach ≈ 29 300 km from the centre (≈ 1.19 R_N),
  Triton ~40 000 km five hours later; ring-plane crossings at 85 400 km inbound and
  105 500 km outbound; dust halo 1.6–6.5 R_N, dense layer within ~700 km of the
  plane.
- Rings: Galle ~41 900, Le Verrier ~53 200, Lassell/Arago 53 200–57 200, Adams
  ~62 930 km. **Ring-crossing constraint**: every equatorial-plane crossing lies
  inside `ring_keepout_km[0]` = 38 000 km or outside `ring_keepout_km[1]` =
  85 000 km (Voyager-demonstrated). For v∞ 5, rp 1.10 R_N this holds whenever the
  node is within ±63° of periapsis — compatible with an in-Triton-plane trajectory.
- Phasing/plane: free asymptote ⇒ trajectory plane = Triton's plane (co-moving),
  periapsis argument free within the ring window, periapsis time sets Triton's
  phase. Fixed asymptote ⇒ the plane rotates about the asymptote until the
  outbound crossing lies in Triton's plane; the encounter sense then follows and
  the ring constraint may be infeasible — reported by the guess generator.

### 2.5 Stable Triton orbit

| criterion | default | source |
|---|---|---|
| apoapsis | ≤ `ra_max_hill`·R_H, default 0.3 (4 379 km); prograde stability limit ≈ 0.49 R_H, retrograde ≈ 0.93 R_H (circular) | Domingos et al. 2006; Hamilton & Burns 1991 |
| inclination to Triton's orbit plane | Kozai–Lidov: 39.2° < i_T < 140.8° unstable, τ ≈ 60–180 d; default `final_sense = "retrograde"`, i_T ≥ 141° | Kozai 1962; Scheeres et al. 2001 |
| periapsis | ≥ `rp_final_alt_km` = 1000 km (exobase ~850–950 km) | this design |
| proof | post-solve scipy propagation for `stability_days` = 120 with Triton + Neptune J2 + tide; min rp, max ra, i_T history reported | — |

Sun tide on the orbiter ~2e-10 of Triton's pull: neglected. Triton J2: neglected.

## 3. Model

| item | approach / flyby / cruise (Neptune frame) | Triton approach, ballute, probe, coasts (Triton frame) |
|---|---|---|
| frame | NEQ (`aerocapture/core/system.py`), states `x y z vx vy vz` | Triton-centred, NEQ-parallel, non-rotating; states `xt yt zt vxt vyt vzt` |
| gravity | Neptune point mass + J2 + Triton third body (`aerocapture` `VacuumODE`) | a = −μ_T r/r³ + a_N(r + r_T(t)) + n² r_T − μ_T r_T/a_T³ with a_N = Neptune point mass + J2 and r_T, n from the circular fit (the NEQ dynamics translated) |
| atmosphere | none (GRAM drag reported only) | Triton N₂ (§3.1), log-ρ Akima; co-rotating with ω = n_T ĥ_T (synchronous, 16.7 m/s at the surface) |
| aero | — | drag only, β(Kn) = m/(C_D(Kn) A); ballute phase: envelope + probe; probe phase: probe only |
| heating | — | stagnation convective Sutton–Graves N₂ (k = 1.7407e-4 SI, R_n = D/2) bridged to free-molecular q_fm = α ½ρv³ by q = (q_c⁻² + q_fm⁻²)^-½; Tauber–Sutton radiative upper bound reported every run (`rad_in_limits = False` by default) |
| thermal limit | — | film radiative equilibrium T = (q/(n_sides ε σ))^¼ ≤ `t_film_max_k` (default 773 K) |
| loads | — | sensed g = D/(m g0), dynamic pressure |

### 3.1 Triton atmosphere

No machine-readable published profile was found (Strobel & Zhu 2017 give figures
only; the Benne et al. supplement is not openly hosted). **Decision: a built-in,
reproducible hydrostatic N₂ model; no data file.** `triton/atmosphere.py`
integrates dp/dz = −m g(r) p/(kT) with the Bates profile
T(z) = T∞ − (T∞ − T₀) e^(−z/L) and tabulates 0–1100 km at 0.5 km:

| parameter | default | anchor |
|---|---|---|
| p₀ | 16 μbar | Strobel & Zhu 2017 / Benne 2022; occultations 14–16 μbar 1989–2022 (Sicardy 2024) |
| T₀ | 38 K | same |
| T∞ | 92 K | Triton-3 thermosphere 90–102 K |
| L | 161.6 km | fitted so p(1026 km) = 2.8e-9 Pa (Benne 2022 model top) |

Independent checks: n_N₂(575 km) = 3.97e8 cm⁻³ (Voyager UVS 4.0 ± 0.4e8);
p(47 km) = 1.11 μbar (2017 occultation 1.18 ± 0.03). Densities: 1.4e-4 (0),
6.6e-7 (100), 2.6e-8 (200), 3.1e-10 (400), 1.3e-11 (600), 1.0e-12 (800),
1.0e-13 kg/m³ (1026 km). This is 2.6× (200 km) to ~10× (1030 km) denser than the
Brisby & Lyne Fig. 1 curve; the Voyager anchor favours this model. Knobs:
`atm_density_scale` (dispersions), `p_surface_ubar`, `t_inf_k`, `t_scale_km`,
and `atm_table_path` for a user-supplied table.

### 3.2 Ballute aerodynamics

- Kn = λ/D, λ = m_N₂ / (√2 π d² ρ), d = 3.7e-10 m.
- C_D(Kn) = C_c + (C_fm − C_c) sin²[π(3/8 + ⅛ log₁₀ Kn)] for 1e-3 < Kn < 10, with
  smooth clamps outside (Wilmoth sine² bridging); defaults C_c = 0.9, C_fm = 2.0;
  `cd_model = "constant"` gives C_c.
- Release: one stage (`release_mode = "free"`); `"none"` keeps the ballute through
  exit. Inflation happens before the interface (ρ ≲ 1e-13 there), so it has no
  dynamic effect and is not modelled.

## 4. Dymos problem

### 4.1 Phases (Radau, compressed, order 3)

| # | phase | frame / ODE | from → to | boundary constraints | path constraints |
|---|---|---|---|---|---|
| 1 | `approach` | NEQ / `VacuumODE` | r₀ = 1e6 km (arrival conic match) → r = 2e5 km | r final = r_split | — |
| 2 | `flyby_in` | NEQ / `VacuumODE` (+ `RingNodeComp` at periapsis) | 2e5 km → periapsis | r·v = 0 final; r final ∈ [rp_min, rp_max]·R_N; ring-node inverse radii | — |
| 3 | `flyby_out` | NEQ / VacuumODE | periapsis → 2e5 km | — | — |
| 4 | `cruise` | NEQ / VacuumODE | 2e5 km → Triton distance `frame_switch_km` (30 000) | Triton distance final; closing | — |
| 5 | `triton_approach` | Triton / `TritonVacuumODE` | frame match → r = R_T + `interface_alt_km` | r·v ≤ 0 final | — |
| 6 | `ballute` | Triton / `TritonAtmosphereODE` (β_ballute) | interface → release (free duration) | — | alt ≥ h_min; g ≤ g_max; q ≤ q_peak (objective slack); T_film ≤ T_max; optional q ≤ q_max |
| 7 | `probe` | Triton / `TritonAtmosphereODE` (β_probe) | release → exit interface | r final = interface, r·v ≥ 0 | alt ≥ h_min |
| 8 | `capture_coast` | Triton / TritonVacuumODE | exit → burn (free time) | osculating Triton ra ≤ ra_max | alt ≥ 0 |
| 9 | `final` | Triton / TritonVacuumODE | burn → burn + `final_coast_h` (24 h) | i_T band per `final_sense` | osculating rp ≥ R_T + `rp_final_alt_km` + `final_rp_margin_km`; osculating ra ≤ (1 − `final_ra_margin`) · ra_max |

Links:
- Phases 1–4 and 5–9 use `link_phases` on time and states.
- 4→5 links time only; states are joined by a `FrameMatchComp` equality
  r_TC − (r_NEQ − r_T(t)) = 0, v_TC − (v_NEQ − v_T(t)) = 0 (the arrival-match
  pattern of the Neptune sim).
- 8→9 links time and position; velocity carries an impulsive burn
  v₉,₀ − v₈,f − Δv = 0, Δv a 3-component design variable bounded by
  `burn_dv_max_kms`.
- Optional impulses (default off): Neptune periapsis burn (2→3,
  `neptune_dv_max_kms`) and a post-flyby trajectory-correction burn (3→4,
  `tcm_max_kms`, used only as a solve homotopy).

Ring constraint: at the periapsis node, the two equatorial-node radii come from
the osculating conic, u_k = (1 + e cos f_k)/p (complex-step safe); require the
near node u ≥ 1/r_in and the far node u ≤ 1/r_out (u < 0 means the node is beyond
the asymptote and passes). A post-solve check locates the actual z sign changes
on the simulated trajectory.

### 4.2 Design variables

| variable | notes |
|---|---|
| arrival RA, Dec | free (default) or fixed (`radec` / `from_run`); same Config fields and `opt_direction` rule as the Neptune sim |
| B·T, B·R | set plane and periapsis (rp band enforced at the end of phase 2) |
| periapsis time | free within ± `t_peri_window_periods`·P_T (selects Triton's phase); `opt_t_peri` as in the Neptune sim |
| collocation states / phase times | the Triton aim point and entry FPA are implicit |
| ballute phase duration | = release time |
| Δv_burn (3) | periapsis raise; optional Δv_Neptune, Δv_TCM |
| q_peak / g_peak slacks | per objective |

### 4.3 Objective (`objective`)

`qpeak` (default; slack on peak bridged convective heat rate on the ballute, plus
radiative only if `rad_in_limits`), `heat` (heat load), `g` (peak g), `dv`
(|Δv_burn| [+ |Δv_N|]); `heat_weight` and `dv_weight` add regularisation terms
([margin] inputs, §9; `dv_weight` must be non-zero, §10).

### 4.4 Scaling and grids

- Triton-frame position refs: 30 000 km (phase 5), 3 000 km (6–7), 5 000 km (8–9);
  velocity refs 10, 10, 1, 0.5 km/s; ballute-phase time ref ≈ 300 s.
- Segments: approach 20, flyby_in 15, flyby_out 15, cruise 20, triton_approach 15,
  ballute 40 (density-weighted), probe 25, capture_coast 15, final 40 (2.7 eccentric
  orbits in 24 h: defects ~20 km at 20 segments, < 1 m/s at 40).
- IPOPT settings shared with the Neptune mission (`core/solve.py`); segments, order and
  IPOPT limits are Config fields.

### 4.5 Implementation notes

- The Triton-frame phases use the exactly translated NEQ gravity (residual
  ~1e-9 km/s²); the post-solve stability propagation uses the physical
  differential tide (`frame.accel_tidal`).
- The final-orbit rp/ra bounds are path constraints on the osculating elements
  at every node of the final phase.
- The guess supports `co` and `counter` encounters in Triton's plane.

## 5. Initial guess and staged solve

1. **Triton pass (Triton frame, scipy):** from v_rel∞ given by the patched conic
   (default co-moving), fly entry states over γ_E with the ballute and an adaptive
   release event (release when the osculating Triton orbit reaches ra_target),
   then the probe to exit. Corridor = [shallowest γ that captures with the ballute
   retained, steepest γ with post-release rp ≥ R_T + h_min]; guess at
   `guess_corridor_fraction` (0.5). The corridor is reported.
2. **Capture coast + burn:** propagate (Triton + Neptune tide) to apoapsis,
   tangential burn to rp_final, propagate `final_coast_h`.
3. **Neptune leg:** plane = Triton's plane (free asymptote) or the plane through
   the fixed asymptote; rp = band middle; periapsis on the node line (ring-safe).
4. **Backward shooting** (shared propagation and shooting numerics of `core/guess.py`): unknowns
   (|v_rel| at interface, entry time, out-of-plane heading); residuals (v∞ at r₀,
   Neptune periapsis radius, ring proxy); full NEQ dynamics from the Triton
   interface back through Neptune to r₀; `state_to_bplane` gives the arrival
   parameters; periapsis time snapped by whole Triton periods.
5. Cut arcs at phase boundaries; Triton-frame arcs = NEQ − analytic Triton.

| stage | what | purpose |
|---|---|---|
| S0 | Triton-only problem (phases 5–9, fixed Triton v∞ vector, free Triton B-plane) | robust aerocapture solve; also the engine of ballute trades (`scope = "triton_only"`) |
| S1 | full problem, `tcm_max_kms` = 0.2 | decouples the sensitive Neptune-B-plane → Triton-aim map |
| S2 | continuation of the TCM bound → 0 (halving on failure) | the actual requirement (no TCM) |
| S3 | fixed-direction continuation (≤ 10° steps) when `direction` ≠ "free" | as in the Neptune sim |
| — | simulate check + `stability_days` propagation | verification |

## 6. Code layout and workflow

The mission is the subpackage `aerocapture.triton`, beside the shared
`aerocapture.core` and the Neptune mission `aerocapture.neptune`; imports are
package-absolute. Outputs go to `results/aerocapture/triton/` (`core/paths.py`).

| file | role |
|---|---|
| `core/config_base.py` | `ConfigBase`: validation, opt-flag resolution, margins-and-assumptions registry, record, echo |
| `triton/config.py` | `Config`: every input in CONFIG-cell groups (§9), `ASSUMPTIONS`, automatic opt-flag rule, warnings, derived bounds |
| `triton/frame.py` | NEQ ↔ Triton-frame transforms, Triton-frame gravity, ω_T, i_T, Triton Hill radius |
| `triton/atmosphere.py` | Bates hydrostatic model, optional table loader, log-ρ Akima, `TritonDensityComp` |
| `triton/ballute.py` | `Ballute` / `Probe` records, Kn, bridged C_D, Sutton–Graves N₂ + free-molecular heating, film temperature, Tauber–Sutton bound |
| `core/ode.py`, `triton/ode.py` | shared ODE building blocks; `TritonVacuumODE`, `TritonAtmosphereODE`, frame-match, impulse and ring components |
| `core/guess.py`, `triton/guess.py` | shared propagation and shooting numerics; patched conics, scipy pass with adaptive release, backward shooting |
| `triton/problem.py` | `TritonProblem`: builds, seeds and solves the trajectory from a `Config` |
| `core/solve.py`, `triton/case.py` | IPOPT driver and stage helpers; Config → epoch/atmosphere (`resolve`) → guess → staged `solve` (S0–S3); `warm_start`, `solve_chain` |
| `core/io.py`, `triton/results.py` | JSON/CSV/table writers and the printed assumptions block; summary, ring/stability/simulate checks, active constraints, CSV rows (`COLUMNS`) |
| `core/plotting.py`, `triton/plots.py` | plot primitives; system view, Neptune flyby, Triton frame, pass panels, corridor, stability, guess, sweep and trade figures |
| `triton/run_triton.py` | `# %%` cells: one case |
| `triton/sweep_triton.py` | `# %%` cells: v∞ × rp band × encounter sense |
| `triton/trade_triton.py` | `# %%` cells: diameter, masses, C_D model, release mode, density scale, film limit |
| `tests/triton/` | `test_physics.py`, `test_components.py` (fixtures in `tests/conftest.py`) |

Workflow: edit the CONFIG cell of `run_triton.py` (every field with value,
unit and meaning, tagged [physics], [numerics], [guess] or [margin]) and run
the cells: imports → CONFIG → initial guess (+ figures) → solve → summary →
figures → save. The sweep and trade scripts list the same fields as BASE
(per-case fields marked SET PER CASE), then the SWEEP / TRADE lists, a CASES
cell with the only overrides, run, table and figure — the same cells as the
Neptune mission's scripts.

Outputs, `results/aerocapture/triton/<tag>/`: `solution.json` holds
- inputs (opt flags resolved, derived bounds), constants and atmosphere parameters;
- arrival B-plane;
- Neptune periapsis (radius, altitude, time, latitude), ring crossings and the GRAM drag estimate;
- Triton encounter (v_rel∞, sense, entry speed / FPA / latitude) and the direct-intercept comparison;
- ballute pass: peak q (continuum, free-molecular, bridged, Tauber–Sutton bound), film T, g, dynamic pressure, Kn at peak, release time / altitude / speed;
- probe pass, capture orbit, burn Δv, final Triton orbit (a, e, i_T, rp, ra, period, sense);
- stability check, corridor, solver stages, simulate check;
- `assumptions` (every [margin] field with value and reason), `active_constraints`, `active_box_bounds`, warnings.

Figures: `guess_system`, `guess_neptune_flyby`, `guess_triton`, `guess_pass`,
`guess_corridor`, `system` (rings, keep-out band, Triton's orbit),
`neptune_flyby` (full scope), `triton`, `pass`, `pass_ballute`, `corridor`,
`stability`. Sweeps and trades write per-case directories under
`sweep/<name>/` or `trade/<name>/` plus `<name>.csv` and `<name>.png`.

The printed summary follows the section order shared with the Neptune
mission: header, solver (IPOPT status, iterations, wall time, stages),
arrival, Neptune flyby, entry (with the direct-intercept comparison, corridor
and release), loads, heating, final orbit (capture orbit, burn, final orbit,
stability), a checks line (converged, simulate, rings, stability), the
"Assumptions & margins in effect" block (§9, active constraints, numerical
box bounds hit) and every warning.

Library modules stay side-effect free apart from kernel loading. Docstrings
follow `.claude/skills/write-docstrings`. No multiprocessing.

## 7. Verification

| test | pass criterion |
|---|---|
| atmosphere anchors | p(1026 km) = 2.8e-9 Pa ± 1 %; n(575 km) within 3.6–4.4e8 cm⁻³; p(47 km) within 10 % of 1.18 μbar (at p₀ = 16 μbar); monotone; spline exact at nodes |
| frame consistency | one state integrated 2 d in NEQ and in the Triton frame agrees < 1 m; transforms round-trip exactly |
| Triton-frame Kepler | tide off, Dymos coast vs analytic < 0.1 km |
| partials | `check_partials` on every new component |
| bridging limits | C_D → C_c (Kn ≪ 1e-3), C_fm (Kn ≫ 10); q → q_SG / q_fm in the limits; film T: 3.5 W/cm², 2 sides, ε 0.85 → 776 K |
| patched conics | §2.1 values (v∞ 5, 1.10 R_N: 8.21 / 9.10 / 9.91 km/s); Brisby & Lyne 4.01 / 9.41 km/s |
| ring nodes | analytic node radii = brute-force z-crossings of a propagated conic |
| guess corridor | D 70 m, 530 kg, v_E 8.3 → corridor ≈ −52.0…−53.0° |
| every solve | simulate check: final Triton ra within `sim_tol` (1 %), rp within `sim_rp_tol_km` (5 km); ring post-check; stability report |

## 8. Decisions

| # | topic | decision |
|---|---|---|
| 1 | Neptune flyby | kept despite the ~4.4 km/s entry-speed penalty (requirement); the direct-intercept comparison is printed every run |
| 2 | Ballute | enveloping 70 m sphere (NIAC concept); a towed toroid fits through the C_D / R_n fields |
| 3 | Film limit | 773 K, two-sided, ε 0.85 (Kapton continuous use ~673 K is a trade case) |
| 4 | Final orbit | retrograde, i_T ≥ 141°, ra ≤ 0.3 R_H, rp ≥ 1000 km altitude; polar mapping orbits are Kozai-unstable (60–180 d) |
| 5 | Atmosphere | built-in hydrostatic N₂ fit (§3.1); a digitised table plugs in through `atm_table_path` |
| 6 | Radiative heating | Tauber–Sutton bound reported, not constrained (`rad_in_limits=False`) |
| 7 | TCM / Oberth burn | 0 / 0 in the final answer; the TCM is only a solve homotopy (S1) |
| 8 | g limit | 25 g (NIAC Pluto 16–36 g) |
| 9 | Inputs | one `Config` dataclass, every field spelled out in the scripts' CONFIG cell, echoed resolved and saved with each run; no CLI |
| 10 | Margins | every margin, assumed value, weight and check tolerance in one [margin] group, printed with each summary next to the active constraints |

## 9. Inputs (`config.Config`; every field spelled out in the CONFIG cell)

Groups in CONFIG-cell order. Tags: [physics], [numerics], [guess], [margin].

| group | fields | defaults |
|---|---|---|
| arrival [physics] | `vinf_kms`, `direction`, `ra_deg`, `dec_deg`, `from_run`, `point`, `epoch`, `opt_direction`, `opt_t_peri`, `t_peri_window_periods`, `r0_km` | 5.0, "free", 0, 0, "capture/EJN", "p13116", 2060-01-01 TDB, None, None, 0.5, 1e6 |
| Neptune flyby [physics] | `rp_min_rn`, `rp_max_rn`, `ring_keepout_km`, `neptune_dv_max_kms` | 1.10, 1.25, (38 000, 85 000), 0.0 |
| Triton approach [physics] | `frame_switch_km` [numerics], `interface_alt_km`, `tcm_max_kms` | 30 000, 1000, 0.0 |
| Triton atmosphere [physics] | `atm_model`, `atm_table_path`, `p_surface_ubar`, `t_surface_k`, `t_inf_k`, `t_scale_km`, `atm_rotation` | "bates", None, 16, 38, 92, 161.6, True |
| ballute [physics] | `Ballute(name, diameter_m, envelope_mass_kg, cd_continuum, cd_free_molecular, cd_model, emissivity, radiating_sides, t_film_max_k)` | "ead-70", 70, 330, 0.9, 2.0, "wilmoth", 0.85, 2, 773 |
| probe [physics] | `Probe(mass_kg, cd_area_m2, nose_radius_m)` | 200, 2.0 (β 100), 0.5 |
| release and limits [physics] | `release_mode`, `g_max`, `q_max_w_cm2` | "free", 25, None |
| heating [physics] | `conv_model`, `bridging`, `rad_model` | "sutton_graves_n2", "smooth_min", "tauber_sutton_air" |
| final Triton orbit [physics] | `ra_mode`, `final_sense`, `burn_mode`, `burn_dv_max_kms`, `final_coast_h` | "max", "retrograde", "free", 0.3, 24 |
| objective [physics] | `objective` | "qpeak" ("heat", "dv", "g") |
| margins and assumptions [margin] | table below | |
| run [physics] | `scope` | "full" ("triton_only": S0) |
| transcription and solver [numerics] | `order`, `segments`, `r_split_km`, `tcm_homotopy_kms`, `max_iter`, `tol`, `constr_viol_tol`, `print_level`, `linear_solver` | 3, §4.4, 2e5, 0.2, 500, 1e-5, 1e-6, 5 (runner: 0), mumps |
| initial guess [guess] | `guess_corridor_fraction`, `guess_plane`, `guess_encounter`, `guess_ra_fraction`, `guess_rp_margin_km` | 0.5, "triton", "co", 0.92, 50 |
| output | `tag` | None |

Margins and assumptions (`config.ASSUMPTIONS`; printed with every summary):

| field | default | kind | why this value / what it costs |
|---|---|---|---|
| `h_min_km` | 10 km | margin | altitude floor through the passes (Triton's relief ~1 km); active only on steep passes |
| `rp_final_alt_km` | 1000 km | margin | final periapsis floor 50–150 km above the exobase (850–950 km) so drag does not decay the orbit; enforced by the stability check |
| `final_rp_margin_km` | 50 km | margin | the optimiser holds rp ≥ floor + 50 km: Neptune's tide moves rp ~−10 km over 120 d (§10); costs a slightly larger burn |
| `ra_max_hill` | 0.3 R_H | margin | apoapsis limit (4379 km), inside the circular-orbit stability limits 0.93 R_H retrograde / 0.49 R_H prograde |
| `final_ra_margin` | 0.05 | margin | the optimiser holds ra ≤ 0.95 · limit: the tide adds ~+60 km over 120 d (§10); costs a slightly hotter capture |
| `i_retro_min_deg` | 141° | margin | 0.2° outside the Kozai band (140.8°); circular-orbit theory, not sufficient at e ≳ 0.25 (§10) |
| `i_pro_max_deg` | 39° | margin | 0.2° inside the Kozai band's lower edge (39.2°) |
| `atm_density_scale` | 1.0 | model | Bates N₂ density multiplier (dispersions 0.5–1.5); the corridor moves with it |
| `fm_accommodation` | 1.0 | model | free-molecular energy accommodation; 1 absorbs all incident energy (conservative) |
| `rad_in_limits` | False | model | Tauber–Sutton air radiation is an optically thin upper bound far outside its fit range (LAURA: N₂ radiation minimal): reported, not constrained |
| `heat_weight` | 0 per J/cm² | weight | heat-load term in the objective; 0 leaves the objective pure |
| `dv_weight` | 1e-3 per m/s | weight | pins the burn and the capture-orbit plane (0: IPOPT drifts to a plane-change burn onto the Kozai edge, §10); moves peak q < 2 % |
| `stability_days` | 120 d | check | horizon of the post-solve tidal propagation; spans the 60–180 d Kozai growth time |
| `sim_tol` | 0.01 | check | largest relative final-apoapsis error of the simulate re-integration |
| `sim_rp_tol_km` | 5 km | check | largest final-periapsis error of the simulate re-integration |

Fixed model assumptions (`results.NOTES`): circular Triton fit to nep097;
Sutton–Graves N₂ continuum bridged to free-molecular heating; Sun tide and
Triton J2 neglected; ballute inflation, flutter and Triton winds out of scope.

## 10. Findings

All runs: defaults of §9 unless stated (70 m / 330 kg ballute + 200 kg probe,
co-moving encounter, retrograde final orbit), `OMP_NUM_THREADS=1`. Results in
`results/aerocapture/triton/`.

| case | Neptune rp | v_rel∞ / entry (km/s) | entry FPA | release (s, alt) | peak q: continuum / FM / bridged / T–S bound (W/cm²) | film T | peak g | burn | final orbit (after burn) | 120 d stability |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 `triton_only`, v∞ 5 | — | 8.171 / 8.229 | −52.41° | 312 s, 72 km | 0.657 / 8.42 / 0.654 / 5e-5 | 510 K | 9.6 | 82 m/s | rp 1050 km alt, ra 4160 km, i_T 179.95° | min alt 1049 km, max r 4209 km: PASS |
| full, v∞ 5 (`full_vinf5_ead-70`) | 1.250 R_N | 8.138 / 8.197 | −52.38° | 316 s, 73 km | 0.648 / 8.25 / 0.645 / 4e-5 | 509 K | 9.4 | 82 m/s | rp 1050 km alt, ra 4160 km, i_T 179.99°, P 8.7 h | min alt 1053 km, max r 4205 km: PASS |
| full, v∞ 3 (sweep) | 1.25 | 7.109 / 7.178 | −52.21° | 356 s, 77 km | bridged 0.429 | 459 K | 7.0 | 82 m/s | as above | PASS |
| full, v∞ 7 (sweep) | 1.25 | 9.475 / 9.525 | −52.55° | 276 s, 70 km | bridged 1.021, T–S 0.016 | 571 K | 13.2 | 83 m/s | as above | PASS |

Every full run: S0 → S1 (TCM ≤ 0.2 km/s) → S2 (TCM removed) converge with IPOPT
success (~4–7 min in total); simulate check passes (ra error 8e-4, rp error
0.4 km); the single ring-plane crossing is at ~30 800 km (inside the 38 000 km
keep-in); the GRAM drag of the inflated ballute through the flyby is ~1e-5 m/s.
Direct-intercept comparison (printed every run): 2.74 / 3.75 / 5.09 km/s entry
without the close pass at v∞ 3 / 5 / 7, i.e. the close pass costs ~4.4 km/s.

Ballute trade (S0, v∞ 5, `trade/ballute_trade`):

| D / envelope (kg) | β (kg/m²) | corridor | release alt | bridged peak q | film T | peak g | 120 d |
|---|---|---|---|---|---|---|---|
| 30 / 200 | 0.63 | 0.65° | 46 km | 2.16 | 688 K | 10.4 | FAIL (final i_T on the 141° bound, e 0.22→0.33, min alt 900 km) |
| 70 / 330 | 0.15 | 1.13° | 72 km | 0.654 | 510 K | 9.6 | PASS |
| 100 / 680 | 0.12 | 1.13° | 80 km | 0.489 | 475 K | 9.2 | PASS |

Findings:

- **The model reproduces the sizing numbers (§2).** Atmosphere anchors (p(1026 km)
  2.798e-9 Pa, n(575 km) 3.97e8 cm⁻³, p(47 km) 1.106 µbar), patched-conic speeds
  (8.21 / 9.10 / 9.91 km/s), Brisby & Lyne 4.01 / 9.41 km/s, the 776 K film
  calibration, and the scipy corridor −51.96…−53.11° at 8.23 km/s (prototype
  −52.0…−53.0°) all match.
- **The optimum is a late, shallow release.** The optimiser keeps the ballute
  past periapsis (~315 s after entry, ~73 km) at the shallow side of the
  corridor; peak heating sits at ~115 km, Kn ≈ 3e-3 (continuum-dominated: the
  bridged rate is within 1 % of Sutton–Graves). Peak q 0.64 W/cm² and a 510 K film
  leave ~260 K of margin to the 773 K limit at v∞ 5; at v∞ 7 the film is 571 K.
- **The radiative bound is negligible here.** Peak heating happens below 9 km/s
  airspeed, where the Tauber–Sutton f(V) is tiny (≤ 0.016 W/cm² even at v∞ 7);
  the "~19 W/cm² at 10 km/s" sizing estimate applies only to counter-moving
  encounters.
- **The full problem prefers rp = 1.25 R_N** (top of the band), which trims v_rel∞
  by ~0.04 km/s; the Neptune leg in Triton's plane with periapsis near the node
  line clears the rings with ~7 000 km margin.
- **`dv_weight` must be non-zero (1e-3 per m/s).** With 0, the
  burn vector and the capture-orbit plane are free directions of the heat
  objective: IPOPT drifts to a 300 m/s (bound) plane-change burn that parks i_T
  on the 141° bound, S2 needs repeated TCM halving (49 min), and the 120-day
  propagation fails (e 0.03–0.45, min alt 563 km). With 1e-3 the burn is the
  ~82 m/s periapsis raise, i_T stays ~180°, S2 converges in ~50 iterations, and
  peak q is unchanged within 2 %.
- **Final-orbit design margins (`final_rp_margin_km` = 50,
  `final_ra_margin` = 0.05).** Constraining the bare limits puts the osculating
  rp and ra exactly on them; Neptune's tide then moves the orbit −10 km in
  altitude and +60 km in distance over 120 days (stability FAIL: 990.6 km /
  4438 km). The optimiser therefore uses rp ≥ 1050 km alt and ra ≤ 0.95 ra_max;
  the stability check keeps the bare limits.
- **i_T ≥ 141° is not sufficient for eccentric orbits.** 141° is the Kozai edge
  for circular orbits; at e ≈ 0.27 the 30 m case, whose optimum tilts the entry
  out of Triton's plane (entry latitude 22°), lands on i_T = 141° and drifts below
  the floor within 120 days. The default stays 141° (user requirement); retrograde
  orbits near 180° are stable. An S0 retry of the 30 m case with
  `i_retro_min_deg=150` hit the iteration limit (not pursued).
- **Fixed direction and Oberth burn:** the fixed-direction continuation (S3, `direction="radec"`
  6.6° from the free asymptote) converges in one step with all checks passing
  (peak q 0.642); a 50 m/s Neptune periapsis (Oberth) burn
  (`neptune_dv_max_kms=0.05`) lowers peak q to 0.615 W/cm² (−4.6 %).
- **Transcription:** the 24 h final phase needs 40 segments to resolve
  2.7 eccentric orbits (defects ~20 km at 20 segments, < 1 m/s at 40).
- **Optimum flatness:** repeated full runs land within 1 % of each other in peak q
  (0.639–0.648 W/cm²), i.e. at IPOPT tolerance on a flat optimum.

Out of scope: film aeroelasticity / flutter, inflation dynamics, radiation
belts at 1.1 R_N, Triton winds, an AMAT-style cross-check case.

## References

- Brisby & Lyne 2024, J. Spacecraft & Rockets 61(4):1124–1127, doi:10.2514/1.A35964.
- Strobel & Zhu 2017, Icarus 291:55–64, doi:10.1016/j.icarus.2017.03.013.
- Benne et al. 2022, A&A, doi:10.1051/0004-6361/202244447, arXiv:2209.04324 (Triton-3 profile, model top 1026 km).
- Benne et al. 2024, A&A, doi:10.1051/0004-6361/202346699, arXiv:2407.16549 (n_N₂ = 4.0 ± 0.4e8 cm⁻³ at 575 km, Voyager UVS).
- Marques Oliveira et al. 2022, A&A 659:A136, doi:10.1051/0004-6361/202141443 (1.18 ± 0.03 μbar, 2017 occultation).
- Sicardy et al. 2024, arXiv:2402.02476.
- Pluto Hop, Skip, and Jump, NIAC Phase I final report (Global Aerospace, 2018), NTRS 20180007811; Phase II report NTRS 20250000440.
- Rohrschneider & Braun, "Survey of ballute technology for aerocapture", JSR 44(1):10–23, 2007, doi:10.2514/1.19288.
- Putnam & Braun, "Drag-modulation flight-control system options for planetary aerocapture", JSR 51(1), 2014, doi:10.2514/1.A32589.
- Wilmoth, Mitcheltree & Moss, "Low-density aerodynamics of the Stardust sample return capsule", JSR 36(3):436–441, 1999, doi:10.2514/2.3464.
- Sutton & Graves 1971, NASA TR R-376; AMAT `vehicle.py` (Titan/N₂ constant).
- Tauber & Sutton 1991, JSR 28(1):40–42, doi:10.2514/3.26206.
- Pedersen, Meyer-Vernet, Aubier & Zarka 1991, JGR 96:19187–19196, doi:10.1029/91JA01601 (Voyager 2 ring-plane crossings).
- Hamilton & Burns 1991, Icarus 92:118, doi:10.1016/0019-1035(91)90039-V; Domingos, Winter & Yokoyama 2006, MNRAS 373:1227, doi:10.1111/j.1365-2966.2006.11104.x; Scheeres, Guman & Villac 2001, JGCD 24(4):778, doi:10.2514/2.4778.
- SPICE kernels and the Neptune-GRAM table as in `aerocapture/neptune/PLAN.md`.
