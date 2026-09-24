# %%
# Analysis notebook. The global search does NOT live here -- it lives in
# optimize.py, behind an `if __name__ == "__main__"` guard, because pygmo has
# to drive mga_1dsm through mp_island and mp_island forces multiprocessing's
# "spawn" context. spawn rebuilds each child by re-importing __main__, and in
# an Interactive cell __main__ is this file, so an evolve() at module level
# makes every child re-run the notebook and spawn grandchildren until Python
# aborts. Run the search first:
#
#     python orbit_planning/optimize.py --islands 12 --rounds 16
#
# then execute these cells to inspect the champion it wrote.
import json
import sys
from pathlib import Path

try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # interactive cell with no __file__
    _HERE = Path.cwd() / "orbit_planning"
sys.path.insert(0, str(_HERE))

import matplotlib.pyplot as plt
import pykep as pk

import mission  # loads the SPICE kernels on import

if not mission.CHAMPION_PATH.is_file():
    raise FileNotFoundError(
        f"No champion at {mission.CHAMPION_PATH}. Run optimize.py first.")

champion = json.loads(mission.CHAMPION_PATH.read_text())
best_x = champion['x']
seq, LABELS = mission.seq, mission.LABELS

udp = mission.build_udp()                                # capture-burn objective
udp_vinf = mission.build_udp(orbit_insertion=False)      # raw-vinf objective

# Always re-score the decision vector against the CURRENT objective rather than
# trusting the stored fitness. A champion saved under add_vinf_arr scoring has a
# different fitness under orbit_insertion scoring, even though the x vector
# itself is still perfectly valid (get_bounds does not depend on the objective).
best_dv = udp.fitness(best_x)[0]
_stored = champion['best_dv_ms']
print(f"loaded champion: {best_dv:.0f} m/s (re-scored)   settings: {champion['settings']}")
if abs(_stored - best_dv) > 1.0:
    print(f"  NOTE: stored fitness was {_stored:.0f} m/s -- this champion was optimised "
          f"under a different objective.\n"
          f"        The trajectory is valid, but it was not optimised for the "
          f"objective now in force. Re-run optimize.py to search against it.")

# %%
# Decode through pykep rather than by hand: under 'alpha' the times of flight
# are *not* stored directly in x, they are recovered from the alpha variables
# and the total-T gene at x[-1].
DV, _lamberts, T, _bal, _bal_ep = udp._compute_dvs(best_x)

# With orbit_insertion on, DV[-1] is the Neptune CAPTURE BURN, not the arrival
# vinf. Re-score the same x against the raw-vinf objective to recover the vinf
# itself, which is what stage two needs as its entry condition.
DV_vinf, *_ = udp_vinf._compute_dvs(best_x)
arrival_vinf = DV_vinf[-1]          # hyperbolic excess speed at Neptune, m/s
capture_dv = DV[-1]                 # periapsis burn into the capture ellipse
dsm_total = sum(DV[:-1])            # deep-space maneuvers only

launch_epoch = pk.epoch(best_x[0], 'mjd2000')
departure_vinf = best_x[3]          # m/s, already scaled by get_bounds
total_tof = sum(T)

print(f"Minimum Total Delta-V Found : {best_dv:.2f} m/s")
print(f"  - Optimal Launch Date     : {launch_epoch}")
print(f"  - Earth Departure Vinf    : {departure_vinf:.2f} m/s")
print(f"  - Deep-space maneuvers    : {dsm_total:.2f} m/s")
print(f"  - Neptune Arrival Vinf    : {arrival_vinf:.2f} m/s  (not charged; "
      f"it is the approach speed, not a cost)")
print(f"  - Neptune Capture Burn    : {capture_dv:.2f} m/s  -> rp "
      f"{mission.NEPTUNE_CAPTURE_RP / 1e3:,.0f} km, e {mission.NEPTUNE_CAPTURE_E}")
print(f"  - Total Time of Flight    : {total_tof:.1f} days ({total_tof / 365.25:.2f} yr)")

_lo, _hi = mission.MISSION_TOF_DAYS
print(f"  - TOF budget used         : {100 * total_tof / _hi:.1f}% of the {_hi:.0f} d cap")

print("\nInterplanetary Leg Breakdown:")
epoch_mjd2000 = best_x[0]
for i, leg_tof in enumerate(T):
    epoch_mjd2000 += leg_tof
    print(f"  Leg {i + 1} [{LABELS[i]} -> {LABELS[i + 1]}]:")
    print(f"    - Duration    : {leg_tof:.1f} days")
    print(f"    - Arrival     : {pk.epoch(epoch_mjd2000, 'mjd2000')}")
    print(f"    - Leg DV      : {DV[i]:.2f} m/s")

# %%
udp.pretty(best_x)   # prints internally, returns None

# %%
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

udp.plot(best_x, ax=ax)

ax.set_title(f"{'-'.join(LABELS)} Gravity Assist Trajectory\n"
             f"Total Delta-V: {best_dv:.2f} m/s", fontsize=12)
ax.set_xlabel('X (AU)', labelpad=10)
ax.set_ylabel('Y (AU)', labelpad=10)
ax.set_zlabel('Z (AU)', labelpad=10)
ax.set_box_aspect([1, 1, 1])

plt.show()

# %%
# ------------------------------------------------------------- stage two ----
# Neptune-centred arrival, handed the entry conditions from stage one:
#   arrival_epoch  -- when we cross Neptune's sphere of influence
#   arrival_vinf   -- hyperbolic excess speed on arrival, m/s
arrival_epoch = pk.epoch(best_x[0] + total_tof, 'mjd2000')

r_tri, v_tri = mission.triton.eph(arrival_epoch)
print(f"Neptune-centric handover at {arrival_epoch}")
print(f"  - Arrival vinf        : {arrival_vinf:.2f} m/s")
print(f"  - Triton radius       : {sum(c * c for c in r_tri) ** 0.5 / 1e3:,.0f} km")
print(f"  - Triton orbital speed: {sum(c * c for c in v_tri) ** 0.5:,.1f} m/s")

# TODO: capture burn + Triton rendezvous, solved in this Neptune-centred frame.

# %%
# --------------------------------------------------------------- porkchop ---
# Requires the multi-objective search to have run first:
#     python orbit_planning/pareto.py --islands 10 --rounds 12
import porkchop

_pareto = json.loads(porkchop.PARETO_PATH.read_text())
_fig, _info = porkchop.plot(_pareto, theme='light')
plt.show()

# Text counterpart to the chart -- the front, sampled.
print(porkchop.front_table(_pareto))
