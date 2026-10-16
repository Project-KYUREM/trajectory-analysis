# %%
"""Inspect one saved champion trajectory in interactive ``# %%`` cells."""
# Interactive notebook (VS Code "# %%" cells) for inspecting one champion.
#
# The search itself never runs here: pygmo drives pykep problems through
# mp_island, which forces multiprocessing's spawn context, and spawn re-imports
# __main__ -- in a notebook that is this file, so a search in a cell would
# re-run the notebook in every worker. Produce a champion first with
#
#     python orbit_planning/optimize.py --model chem --sequence EVEEJN
#
# then run these cells. The website (docs/) shows the same information for
# every point of every run; this notebook is for poking at one in Python.
import json
import sys
from pathlib import Path

MODEL, SEQUENCE = 'chem', 'EVEEJN'

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
for leg in desc['legs']:
    print(f"  {leg['from']:8s} -> {leg['to']:8s} {leg['tof_days']:7.0f} d   arrive {leg['arr_date'][:10]}")
budget = desc['budget']
for e in budget['events']:
    print(f"  {e['date'][:10]}  {e['label']:55s} {e['dv_ms']:7.0f} m/s {e['propellant_kg']:8.0f} kg"
          f"  -> {e['mass_after_kg']:8.0f} kg")
arr = desc['arrival']
print(f"  arrival vinf {arr['vinf_ms']:.0f} m/s; in a {arr['orbit_periapsis_km']:,.0f} x "
      f"{arr['orbit_apoapsis_km']:,.0f} km orbit with {budget['final_mass_kg']:.0f} kg after "
      f"{desc['totals']['tof_years']:.2f} yr")

# %%
# The trajectory as pykep's mga_1dsm builds it (chem models).
udp = models.build(MODEL, SEQUENCE)
udp.inner.pretty(x)

# %%
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
udp.inner.plot(x, ax=ax)
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
      f"target orbit {neptune['target_orbit']}")

# %%
# The static porkchop of this pair's Pareto cloud (needs pareto.py to have run).
import porkchop

cloud = json.loads(paths.results_path(MODEL, SEQUENCE, 'pareto.json').read_text())
porkchop.plot(cloud, theme='light')
plt.show()
print(porkchop.front_table(cloud))
