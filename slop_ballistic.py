"""Explore ballistic capture by Triton from a Neptune transfer ellipse.

The spacecraft starts at periapsis on a Neptune-centred ellipse with
``rp = 1.5 R_Neptune`` and apoapsis near Triton's orbit radius. The trajectory is
propagated in the Neptune-centred inertial frame used by ``aerocapture``: km,
s, km/s; Neptune gravity (including J2); and Triton's direct + indirect
third-body gravity using the repo's fitted SPICE ephemeris.

Run with ``python slop_ballistic.py``. The script samples the transfer
ellipse's 3D orientation and apoapsis with a Latin hypercube, screens samples
by closest Triton distance in parallel, then repropagates the closest cases at
higher accuracy.
"""
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc

# Share the constants, reference frame, epoch, and ephemeris with the
# aerocapture code rather than maintaining a second set of Neptune/Triton data.
sys.path.insert(0, str(Path(__file__).resolve().parent / "aerocapture"))
import dynamics  # noqa: E402
import system  # noqa: E402


# ------------------------------ run configuration (km, s, km/s) --
EPOCH = system.DEFAULT_EPOCH
PERIAPSIS_RADIUS = 1.25 * system.R_NEPTUNE
TRANSFER_PERIODS = 10.0
SEARCH_SAMPLES = 512
REFINE_COUNT = 16
WORKERS = 10
COARSE_STEP = 900.0
COARSE_MAX_STEP = 900.0


def transfer_initial_state(phase, inclination, node, apoapsis_radius):
    """Periapsis state for the transfer ellipse at a 3D orientation."""
    rp = PERIAPSIS_RADIUS
    ra = apoapsis_radius
    semi_major_axis = 0.5 * (rp + ra)
    speed = np.sqrt(system.MU_NEPTUNE * (2.0 / rp - 1.0 / semi_major_axis))

    # Build a right-handed orbital basis. Inclination and node are measured
    # against the shared NEQ frame; the search includes prograde and retrograde
    # planes, including Triton's own plane.
    h = np.array([np.sin(inclination) * np.sin(node),
                  -np.sin(inclination) * np.cos(node), np.cos(inclination)])
    p = np.array([np.cos(node), np.sin(node), 0.0])
    q = np.cross(h, p)
    radial = np.cos(phase) * p + np.sin(phase) * q
    transverse = -np.sin(phase) * p + np.cos(phase) * q
    return np.r_[rp * radial, speed * transverse]


def propagate(parameters, epoch, *, output_step=COARSE_STEP, max_step=COARSE_MAX_STEP,
              rtol=1e-8, atol=1e-9):
    """Propagate one oriented transfer ellipse through two transfer periods."""
    phase, inclination, node, apoapsis_ratio = parameters
    apoapsis_radius = apoapsis_ratio * epoch.triton.a
    a_transfer = 0.5 * (PERIAPSIS_RADIUS + apoapsis_radius)
    period = 2.0 * np.pi * np.sqrt(a_transfer**3 / system.MU_NEPTUNE)
    duration = TRANSFER_PERIODS * period
    times = np.arange(0.0, duration, output_step)
    times = np.r_[times, duration]

    def rhs(t, y):
        r, v = y[:3], y[3:]
        r_dot, v_dot = dynamics.vacuum_rhs(t, r, v, epoch.triton)
        return np.r_[r_dot, v_dot]

    return solve_ivp(
        rhs, (0.0, duration),
        transfer_initial_state(phase, inclination, node, apoapsis_radius),
        method="DOP853", rtol=rtol, atol=atol, max_step=max_step,
        t_eval=times, dense_output=False,
    )


def encounter_data(solution, triton):
    """Return Triton-relative distance and two-body specific energy histories."""
    t = solution.t
    r, v = solution.y[:3].T, solution.y[3:].T
    relative_r = r - triton.position(t)
    relative_v = v - triton.velocity(t)
    distance = np.linalg.norm(relative_r, axis=1)
    energy = 0.5 * np.sum(relative_v**2, axis=1) - system.MU_TRITON / distance
    return distance, energy


def evaluate_candidate(job):
    """Coarse screen returning the candidate and its minimum Triton distance."""
    parameters, epoch = job
    solution = propagate(parameters, epoch)
    distance, _ = encounter_data(solution, epoch.triton)
    i_min = int(np.argmin(distance))
    return tuple(parameters), float(distance[i_min])


def main():
    epoch = system.make_epoch(EPOCH)
    bounds = np.array([[0.0, 2.0 * np.pi], [0.0, np.pi], [0.0, 2.0 * np.pi], [0.75, 1.25]])
    sampler = qmc.LatinHypercube(d=len(bounds), seed=2060)
    unit_samples = sampler.random(SEARCH_SAMPLES)
    samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
    jobs = [(tuple(sample), epoch) for sample in samples]

    print(f"Screening {SEARCH_SAMPLES} Latin-hypercube cases over "
          f"{TRANSFER_PERIODS:g} transfer periods ({WORKERS} workers) ...")
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        screened = list(pool.map(evaluate_candidate, jobs, chunksize=1))
    screened.sort(key=lambda result: result[1])

    # Rerun the closest coarse cases with smaller steps and tighter tolerances.
    refined = []
    for parameters, _ in screened[:REFINE_COUNT]:
        solution_i = propagate(parameters, epoch, output_step=60.0, max_step=300.0,
                               rtol=2e-10, atol=1e-11)
        distance_i, energy_i = encounter_data(solution_i, epoch.triton)
        i_min = int(np.argmin(distance_i))
        refined.append((float(distance_i[i_min]), float(energy_i[i_min]),
                        parameters, solution_i, distance_i, energy_i, i_min))

    # Any actual Hill-sphere capture must be among the closest approaches.
    # Prefer a refined bound encounter if one exists; otherwise plot the closest.
    hill = epoch.triton.a * (system.MU_TRITON / (3.0 * system.MU_NEPTUNE)) ** (1.0 / 3.0)
    captures = [row for row in refined if row[0] <= hill and np.any((row[4] <= hill) & (row[5] < 0.0))]
    best = min(captures or refined, key=lambda row: row[0])
    distance_ca, energy_ca, best_parameters, solution, distance, energy, i_ca = best
    phase, inclination, node, apoapsis_ratio = best_parameters
    apoapsis_radius = apoapsis_ratio * epoch.triton.a
    hill = epoch.triton.a * (system.MU_TRITON / (3.0 * system.MU_NEPTUNE)) ** (1.0 / 3.0)
    print(f"Epoch: {epoch.label}")
    print(f"Transfer radii: rp={PERIAPSIS_RADIUS:.1f} km, ra={apoapsis_radius:.1f} km "
          f"({apoapsis_ratio:.4f} × Triton's orbit radius)")
    print(f"Closest coarse-screen approach: {screened[0][1]:.1f} km")
    print(f"Refined {len(refined)} of the closest {REFINE_COUNT} cases")
    print(f"Initial orientation: phase={np.degrees(phase):.2f} deg, "
          f"inclination={np.degrees(inclination):.2f} deg, node={np.degrees(node):.2f} deg")
    print(f"Triton Hill radius: {hill:.1f} km")
    print(f"Closest Triton approach: {distance_ca:.1f} km at t={solution.t[i_ca] / 86400:.3f} days")
    print(f"Triton-relative osculating energy at closest approach: {energy_ca:.6f} km^2/s^2")
    close_mask = distance <= hill
    bound_mask = close_mask & (energy < 0.0)
    if np.any(bound_mask):
        duration_bound = solution.t[np.flatnonzero(bound_mask)[-1]] - solution.t[np.flatnonzero(bound_mask)[0]]
        print(f"Temporary ballistic capture: yes; sampled bound interval {duration_bound:.1f} s")
    else:
        print("Temporary ballistic capture: not found by this 3D orientation search")

    r_sc = solution.y[:3].T
    r_tr = epoch.triton.position(solution.t)
    fig, (ax_system, ax_encounter, ax_energy) = plt.subplots(1, 3, figsize=(16, 5.2))

    # Neptune-centred system view, projected onto Triton's orbital plane.
    p, q = np.asarray(epoch.triton.p), np.asarray(epoch.triton.q)
    scale = 1e3  # axes in 1000 km
    ax_system.plot(r_sc @ p / scale, r_sc @ q / scale, color="tab:blue", label="spacecraft")
    ax_system.plot(r_tr @ p / scale, r_tr @ q / scale, color="tab:orange", linestyle="--", label="Triton")
    ax_system.scatter(*(r_sc[0] @ np.array([p, q]).T / scale), color="tab:blue", s=24, label="start")
    ax_system.scatter(*(r_sc[i_ca] @ np.array([p, q]).T / scale), color="black", s=30, label="closest approach")
    neptune = plt.Circle((0, 0), system.R_NEPTUNE / scale, color="royalblue", alpha=0.75)
    ax_system.add_patch(neptune)
    ax_system.set_aspect("equal", adjustable="datalim")
    ax_system.set_xlabel("along Triton p (1000 km)")
    ax_system.set_ylabel("along Triton q (1000 km)")
    ax_system.set_title("Neptune-centred transfer")
    ax_system.legend(fontsize=8)
    ax_system.grid(alpha=0.25)

    # Triton-centred encounter geometry and the relative-energy sign.
    relative = r_sc - r_tr
    ax_encounter.plot(relative @ p, relative @ q, color="tab:blue", label="spacecraft relative to Triton")
    ax_encounter.scatter(0, 0, color="tab:orange", s=30, label="Triton")
    theta = np.linspace(0, 2 * np.pi, 361)
    for radius, label, color, style in (
        (hill, "Triton Hill radius", "tab:green", "--"),
        (system.R_TRITON, "Triton radius", "tab:orange", "-"),
        (distance_ca, "closest approach", "black", ":"),
    ):
        ax_encounter.plot(radius * np.cos(theta), radius * np.sin(theta), color=color,
                          linestyle=style, linewidth=1, label=label)
    ax_encounter.scatter(relative[i_ca] @ p, relative[i_ca] @ q, color="black", s=25)
    ax_encounter.set_aspect("equal", adjustable="datalim")
    ax_encounter.set_xlabel("along Triton p (km)")
    ax_encounter.set_ylabel("along Triton q (km)")
    ax_encounter.set_title("Triton encounter; " + ("bound" if energy_ca < 0 else "unbound") +
                           f" at closest approach")
    ax_encounter.legend(fontsize=8)
    ax_encounter.grid(alpha=0.25)

    ax_energy.plot(solution.t / 86400.0, energy, color="tab:purple")
    ax_energy.axhline(0.0, color="black", linewidth=1, linestyle="--", label="capture threshold")
    ax_energy.axvline(solution.t[i_ca] / 86400.0, color="tab:orange", linewidth=1,
                      linestyle=":", label="closest approach")
    ax_energy.set_xlabel("time from departure (days)")
    ax_energy.set_ylabel("Triton-relative energy (km²/s²)")
    ax_energy.set_title("Instantaneous Triton-relative energy")
    ax_energy.legend(fontsize=8)
    ax_energy.grid(alpha=0.25)
    fig.suptitle("Ballistic Triton-capture search — " + epoch.label)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
