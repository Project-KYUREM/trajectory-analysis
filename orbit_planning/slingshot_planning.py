# %%
"""Inspect one saved champion trajectory in interactive ``# %%`` cells."""
# Interactive notebook (VS Code "# %%" cells) for inspecting one champion.
#
# The search itself never runs here: pygmo drives pykep problems through
# mp_island, which forces multiprocessing's spawn context, and spawn re-imports
# __main__ -- in a notebook that is this file, so a search in a cell would
# re-run the notebook in every worker. Produce a champion first with
#
#     python orbit_planning/optimize.py --model capture --sequence EVEEJN
#
# then run these cells. The website (docs/) shows the same information for
# every point of every run; this notebook is for poking at one in Python.
import json
import sys
from pathlib import Path

MODEL, SEQUENCE = 'capture', 'EVEEJN'

try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # interactive cell without __file__
    _HERE = Path.cwd() / 'orbit_planning'
sys.path.insert(0, str(_HERE))

import matplotlib.pyplot as plt
import pykep as pk

import mission
import models
import paths
import trajectory

champion = json.loads(paths.results_path(MODEL, SEQUENCE, 'champion.json').read_text())
x = champion['x']
desc = trajectory.describe(MODEL, SEQUENCE, x)
model = models.get(MODEL)

print(f"{SEQUENCE} / {MODEL}: {desc['cost']:.0f} {model.units}  ({model.label})")
print(f"  launch  {desc['launch']['date'][:10]}   C3 {desc['launch']['c3_km2s2']:.1f} km2/s2   "
      f"asymptote RA/dec {desc['launch']['rla_deg']:.1f}/{desc['launch']['dla_deg']:.1f} deg")
for leg in desc['legs']:
    dsm = f"{leg['dsm_ms']:.0f} m/s" if leg['dsm_ms'] is not None else '-'
    print(f"  {leg['from']:8s} -> {leg['to']:8s} {leg['tof_days']:7.0f} d   maneuver {dsm}   "
          f"arrive {leg['arr_date'][:10]}")
arr = desc['arrival']
print(f"  arrival at {arr['body']}: vinf {arr['vinf_ms']:.0f} m/s, atmospheric entry "
      f"{arr['entry_speed_ms'] / 1e3:.1f} km/s"
      + (f", capture burn {arr['capture_dv_ms']:.0f} m/s" if arr['capture_dv_ms'] else ''))
print(f"  total {desc['totals']['tof_years']:.2f} yr")

# %%
# pykep's own narrative of the same vector.
udp = models.build(MODEL, SEQUENCE)
udp.pretty(x)

# %%
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
udp.plot(x, ax=ax)
ax.set_title(f"{SEQUENCE} ({MODEL}): {desc['cost']:.0f} {model.units}")
ax.set_box_aspect([1, 1, 1])
plt.show()

# %%
# Stage two sketch: Triton's state at arrival, in a Neptune-centred frame.
# mu_central_body and the ephemeris observer must describe the same frame.
neptune = mission.BODIES['N']
triton = pk.planet.spice('TRITON', 'NEPTUNE', 'ECLIPJ2000', 'NONE',
                         mission.gm('NEPTUNE'), mission.gm('TRITON'), 1353400.0, 1353400.0 * 1.1)
arrival_epoch = pk.epoch(desc['arrival']['epoch_mjd2000'], 'mjd2000')
r_tri, v_tri = triton.eph(arrival_epoch)
print(f"Triton at {arrival_epoch}: radius {sum(c * c for c in r_tri) ** 0.5 / 1e3:,.0f} km, "
      f"orbital speed {sum(c * c for c in v_tri) ** 0.5:,.0f} m/s; "
      f"capture ellipse rp = {neptune['capture']['rp_ratio']} R, e = {neptune['capture']['e']}")

# %%
# The static porkchop of this pair's Pareto cloud (needs pareto.py to have run).
import porkchop

cloud = json.loads(paths.results_path(MODEL, SEQUENCE, 'pareto.json').read_text())
porkchop.plot(cloud, theme='light')
plt.show()
print(porkchop.front_table(cloud))
