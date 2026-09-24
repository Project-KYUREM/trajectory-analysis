"""Filesystem layout and the default search matrix.

Deliberately kernel-free: ``batch.py`` imports this without paying the SPICE /
pykep import cost, and every other module reads the layout from here so the
``results/<model>/<sequence>/`` convention is spelled in exactly one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"            # SPICE kernels (gitignored; scripts/fetch_kernels.sh)
RESULTS_DIR = ROOT / "results"      # raw search output (gitignored, large)
DOCS_DIR = ROOT / "docs"            # the static site
DOCS_DATA_DIR = DOCS_DIR / "data"   # compacted site payload (committed)

DEFAULT_MODELS = ("capture", "ballistic")
DEFAULT_SEQUENCES = ("EVEEJN", "EVEJN", "EVJN", "EVVEJN", "EJN")


def csv_list(text, upper=False):
    """Split a comma-separated CLI value into a clean list."""
    items = [t.strip() for t in text.split(",") if t.strip()]
    return [t.upper() for t in items] if upper else items


# Result files carry this so a consumer can tell an old layout from the current one.
RESULT_SCHEMA = 2


def results_path(model, sequence, filename):
    """Path of one raw result file (pure; writers create the directory)."""
    return RESULTS_DIR / model / sequence / filename


def docs_run_dir(model, sequence):
    """Directory of one run's site payload, created on demand (publish writes here)."""
    directory = DOCS_DATA_DIR / model / sequence
    directory.mkdir(parents=True, exist_ok=True)
    return directory
