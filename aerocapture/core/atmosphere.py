"""Neptune-GRAM mean density profile, interpolated smoothly in altitude.

Source: the Neptune-GRAM mean profile as tabulated by AMAT
(``atmdata/Neptune/neptune-gram-avg.dat``; column 1 height in km above the
24 622 km reference radius, 5 km steps from -85 to 4000 km; column 6 density
in kg/m^3). ``scripts/fetch_kernels.sh`` downloads it into ``data/``; it is
GPL-licensed and not committed. The file repeats its first row, which is
dropped.

Model: ln rho(h) is an Akima spline through the table (C1, no overshoot where
the scale height changes). Outside the table ln rho continues linearly with
the end slope, i.e. an exponential atmosphere with the end scale height, so
density stays positive, monotone and C1 for any optimizer iterate:

    ln rho(h) = S(h_c) + S'(h_c) (h - h_c),    h_c = clip(h, h_min, h_max)
"""
import numpy as np
import openmdao.api as om
from scipy.interpolate import Akima1DInterpolator

from aerocapture.core import system


def load_table(path=None):
    """Return ``(height_km, density_kg_m3)`` from the GRAM table, sorted and de-duplicated.

    Args:
        path (Path, optional): Table file. Defaults to ``system.ATMOSPHERE_TABLE``.

    Returns:
        tuple: Heights in km and densities in kg/m^3.

    Raises:
        FileNotFoundError: If the table is missing.
    """
    path = path or system.ATMOSPHERE_TABLE
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run scripts/fetch_kernels.sh")
    data = np.loadtxt(path, comments="#")
    h, first = np.unique(data[:, 0], return_index=True)
    return h, data[first, 5]


class Atmosphere:
    """Log-density Akima spline with log-linear extrapolation.

    Args:
        path (Path, optional): GRAM-format table. Defaults to ``system.ATMOSPHERE_TABLE``.
    """

    def __init__(self, path=None):
        self._set_table(*load_table(path))

    @classmethod
    def from_arrays(cls, h_km, rho_kg_m3):
        """Return an ``Atmosphere`` through increasing heights (km) and densities (kg/m^3)."""
        atm = cls.__new__(cls)
        atm._set_table(np.asarray(h_km, dtype=float), np.asarray(rho_kg_m3, dtype=float))
        return atm

    def _set_table(self, h_km, rho_kg_m3):
        """Store the table and build the log-density spline and its derivative."""
        self.h_table, self.rho_table = h_km, rho_kg_m3
        self._log = Akima1DInterpolator(self.h_table, np.log(self.rho_table))
        self._dlog = self._log.derivative()

    def _log_and_slope(self, h_km):
        """Return ``(ln rho, d ln rho / dh)`` at altitude(s) ``h_km``."""
        h = np.asarray(h_km, dtype=float)
        hc = np.clip(h, self.h_table[0], self.h_table[-1])
        slope = self._dlog(hc)
        return self._log(hc) + slope * (h - hc), slope

    def density(self, h_km):
        """Return the density in kg/m^3 at altitude(s) ``h_km`` in km."""
        return np.exp(self._log_and_slope(h_km)[0])

    def density_and_slope(self, h_km):
        """Return ``(rho, d rho / dh)`` in kg/m^3 and kg/m^3/km at altitude(s) ``h_km``."""
        log_rho, slope = self._log_and_slope(h_km)
        rho = np.exp(log_rho)
        return rho, rho * slope


_DEFAULT = None


def default():
    """Return the process-wide shared ``Atmosphere``, reading the table once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Atmosphere()
    return _DEFAULT


class DensityComp(om.ExplicitComponent):
    """Map ``alt`` (km) to ``rho`` (kg/m^3) with analytic partials; real-valued only.

    Option ``atmosphere`` takes any object with ``density`` and
    ``density_and_slope`` (None: the shared Neptune-GRAM ``default()``).
    """

    def initialize(self):
        """Declare options."""
        self.options.declare("num_nodes", types=int)
        self.options.declare("atmosphere", default=None, allow_none=True,
                             desc="Atmosphere-like object; None uses default()")

    def setup(self):
        """Declare the input, output and diagonal partials."""
        nn = self.options["num_nodes"]
        self.add_input("alt", shape=(nn,), units="km")
        self.add_output("rho", shape=(nn,), units="kg/m**3")
        ar = np.arange(nn)
        self.declare_partials("rho", "alt", rows=ar, cols=ar)
        self._atm = self.options["atmosphere"] or default()

    def compute(self, inputs, outputs):
        """Evaluate the density."""
        outputs["rho"] = self._atm.density(inputs["alt"])

    def compute_partials(self, inputs, partials):
        """Evaluate d rho / d alt."""
        partials["rho", "alt"] = self._atm.density_and_slope(inputs["alt"])[1]
