"""Reporting helpers shared by the mission ``results`` modules.

Each mission's ``results.py`` computes its own summary; this module holds
what both use: the JSON writer for ``solution.json``, the CSV and text
tables of sweeps and trades, the solver and guess records, the Neptune-system
constants and Triton-orbit records, the active-constraint test and the
printed blocks every summary ends with (checks, assumptions and margins in
effect, warnings).
"""
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from aerocapture.core import system
from aerocapture.core.config_base import format_value

BOX_BOUND_TEXT = "numerical box bound active — an unstated constraint"


# ------------------------------------------------------------------ JSON
def jsonable(o):
    """Return ``o`` with numpy types converted, tuples as lists and non-finite floats as None."""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        o = o.item()
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def save(summary, out_dir):
    """Write ``solution.json`` (keys starting with ``_`` left out) into ``out_dir``.

    Args:
        summary (dict): Summary from a mission's ``results.summarize``.
        out_dir (str or Path): Output directory, created if missing.

    Returns:
        Path: The written file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "solution.json"
    public = {k: v for k, v in summary.items() if not k.startswith("_")}
    path.write_text(json.dumps(jsonable(public), indent=2))
    return path


# ------------------------------------------------------------------ tables
def write_csv(rows, path, columns):
    """Write table rows to CSV in the order of ``columns`` (only columns some row has).

    Args:
        rows (list): One dict per case.
        path (str or Path): CSV file, parent created if missing.
        columns (dict or list): Column names in order (a mission's ``COLUMNS``).

    Returns:
        Path: The written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in columns if any(c in r for r in rows)]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def print_table(rows, cols, stream=print):
    """Print the selected columns of table rows as fixed-width text."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return "-" if v is None else str(v)
    cells = [list(cols)] + [[fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(cols))]
    for row in cells:
        stream("  ".join(x.rjust(w) for x, w in zip(row, widths)))


# ------------------------------------------------------------------ records
def ipopt_iterations(p):
    """Return the IPOPT iteration count of the last solve of an OpenMDAO problem, or None."""
    try:
        text = (Path(p.get_outputs_dir()) / "IPOPT.out").read_text()
    except Exception:        # no output directory or IPOPT.out yet
        return None
    m = re.findall(r"Number of Iterations\.+:\s+(\d+)", text)
    return int(m[-1]) if m else None


def solver_record(solve_info, p):
    """Return the solver record of a summary: the final-stage info plus totals.

    Args:
        solve_info (dict): Info returned by a mission's ``case.solve``.
        p (openmdao.api.Problem): The solved problem (for ``IPOPT.out``).

    Returns:
        dict: ``solve_info`` with ``stages`` (list), ``ipopt_iterations`` of
        the last solve, ``total_wall_s`` over every stage and
        ``active_box_bounds`` (list of str; empty when none or not reported).
    """
    stages = list(solve_info.get("stages", []))
    total = sum(st.get("wall_s", 0.0) for st in stages) + solve_info.get("sim_wall_s", 0.0)
    return dict(solve_info, stages=stages, ipopt_iterations=ipopt_iterations(p),
                total_wall_s=total or solve_info.get("wall_s"),
                active_box_bounds=list(solve_info.get("active_box_bounds") or []))


def guess_record(guess):
    """Return the guess's entry corridor and the rest of its info dict (None-safe)."""
    info = dict(guess.info) if guess is not None else {}
    return dict(corridor=info.pop("corridor", None), info=info)


def constants_record():
    """Return the Neptune-system constants used, with their sources."""
    return dict(
        mu_neptune_km3s2=system.MU_NEPTUNE, mu_triton_km3s2=system.MU_TRITON,
        mu_sun_km3s2=system.GM_SUN, r_neptune_alt_ref_km=system.R_NEPTUNE,
        r_neptune_eq_pck_km=system.R_NEPTUNE_EQ, r_triton_km=system.R_TRITON,
        j2=system.J2_NEPTUNE, r_j2_km=system.R_J2, j2_source="Jacobson (2009) AJ 137, 4322",
        omega_neptune_rads=system.OMEGA_NEPTUNE, omega_source="pck00011 PM rate",
        r_hill_km=system.R_HILL, a_neptune_helio_km=system.A_NEPTUNE_HELIO,
        g0_kms2=system.G0, kernels=list(system.KERNELS),
        atmosphere_table=str(system.ATMOSPHERE_TABLE.name))


def triton_record(epoch):
    """Return the circular Triton orbit fitted at the epoch."""
    tr = epoch.triton
    return dict(a_km=tr.a, n_rads=tr.n, period_days=tr.period / 86400.0,
                inclination_deg=math.degrees(math.acos(tr.h_hat[2])),
                theta0_rad=tr.theta0, p=list(tr.p), q=list(tr.q), h_hat=list(tr.h_hat))


# ------------------------------------------------------------------ active constraints
def active_constraints(checks, rtol=1e-3):
    """Return a description of every inequality that sits on (or beyond) its bound.

    Args:
        checks (list): Tuples ``(label, value, bound, side, unit, source)``:
            ``side`` is ``"min"`` (value >= bound) or ``"max"`` (value <=
            bound), ``source`` names the config field of the bound. Entries
            with a None value or bound are skipped.
        rtol (float, optional): Distance to the bound, relative to the bound
            magnitude (absolute below 1), that counts as active. Defaults to 1e-3.

    Returns:
        list: One text per active constraint.
    """
    out = []
    for label, value, bound, side, unit, source in checks:
        if value is None or bound is None:
            continue
        gap = (value - bound) if side == "min" else (bound - value)
        if gap <= rtol * max(1.0, abs(bound)):
            where = "lower" if side == "min" else "upper"
            out.append(f"{label} {format_value(float(value), unit)} on its {where} bound "
                       f"{format_value(float(bound), unit)} ({source})")
    return out


# ------------------------------------------------------------------ printed blocks
def print_solver(sv, stream=print):
    """Print the solver line: final IPOPT status, iterations, total wall time and stages."""
    it = sv.get("ipopt_iterations") or sv.get("iterations")
    wall = sv.get("total_wall_s") or 0.0
    stages = "; ".join(f"{st['stage']}: {st.get('ipopt_text')}" for st in sv.get("stages", []))
    stream(f"  solver: IPOPT {sv.get('ipopt_text')} ({it} it, {wall:.0f} s in total); "
           f"stages: {stages or '-'}")


def print_tail(s, flags, stream=print):
    """Print the checks line, the assumptions-and-margins block and every warning.

    Args:
        s (dict): Summary with ``solver``, ``assumptions``, ``notes``,
            ``active_constraints``, ``active_box_bounds`` and ``warnings``.
        flags (dict): Check name -> printable result, shown on the checks line.
        stream (callable, optional): Line printer. Defaults to print.
    """
    stream("  checks: " + ", ".join(f"{k} {v}" for k, v in flags.items()))
    stream("  Assumptions & margins in effect (CONFIG group 'margins and assumptions'):")
    for a in s.get("assumptions", []):
        value = format_value(a["value"], a["unit"])
        stream(f"    [{a['kind']}] {a['field']} = {value}: {a['why']}")
    for note in s.get("notes", []):
        stream(f"    [fixed] {note}")
    active = s.get("active_constraints") or []
    stream("    active constraints at the solution:" + ("" if active else " none"))
    for c in active:
        stream(f"      {c}")
    box = s.get("active_box_bounds") or []
    if box:
        stream(f"    WARNING: {BOX_BOUND_TEXT}: {', '.join(box)}")
    else:
        stream("    numerical box bounds active: none")
    for w in s.get("warnings", []):
        stream(f"  WARNING: {w}")
