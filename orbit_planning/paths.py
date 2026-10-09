"""Define the filesystem layout and the default search matrix.

This module imports neither SPICE nor pykep, so ``batch.py`` uses it without
loading kernels.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"            # SPICE kernels (gitignored; scripts/fetch_kernels.sh)
RESULTS_DIR = ROOT / "results"      # raw search output (gitignored, large)
DOCS_DIR = ROOT / "docs"            # the static site
DOCS_DATA_DIR = DOCS_DIR / "data"   # compacted site payload (committed)

DEFAULT_MODELS = ("capture", "ballistic")
DEFAULT_SEQUENCES = ("EVEEJN", "EVEJN", "EVJN", "EVVEJN", "EJN")

# Default gravity-assist sequences of the low-thrust (DFD) sequence sweep:
# direct, then Jupiter / Saturn / Uranus assists, then inner-planet pumps.
# sequence_sweep.py's SEQUENCES list starts from this; edit either.
DFD_SEQUENCES = ("EN", "EJN", "ESN", "EUN", "EJSN", "EJUN", "EEJN", "EVEJN", "EMJN", "EVEEJN")


def csv_list(text, upper=False):
    """Split a comma-separated CLI value into a list of items.

    Args:
        text (str): Comma-separated value.
        upper (bool, optional): Upper-case every item. Defaults to False.

    Returns:
        list[str]: The stripped, non-empty items.
    """
    items = [t.strip() for t in text.split(",") if t.strip()]
    return [t.upper() for t in items] if upper else items


# Result files carry this so a consumer can tell an old layout from the current one.
RESULT_SCHEMA = 2


def results_path(model, sequence, filename):
    """Return the path of one raw result file without creating its directory."""
    return RESULTS_DIR / model / sequence / filename


def docs_run_dir(model, sequence):
    """Return the directory of one run's site payload, creating it if needed."""
    directory = DOCS_DATA_DIR / model / sequence
    directory.mkdir(parents=True, exist_ok=True)
    return directory
