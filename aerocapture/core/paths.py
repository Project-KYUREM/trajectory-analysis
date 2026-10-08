"""Define the aerocapture filesystem layout: data, published runs, mission outputs, work dirs."""
import contextlib
import os
from pathlib import Path

# The same three definitions live in orbit_planning/paths.py; keep both in step.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"            # SPICE kernels and the Neptune-GRAM table (gitignored)
RESULTS_DIR = ROOT / "results"      # raw output (gitignored)
DOCS_DATA_DIR = ROOT / "docs" / "data"   # published kyurem runs (arrival v-inf source)


def mission_dir(mission):
    """Return ``results/aerocapture/<mission>/``, the output root of one mission."""
    return RESULTS_DIR / "aerocapture" / mission


def work_dir(mission):
    """Return ``results/aerocapture/<mission>/_openmdao/``, the default OpenMDAO output dir."""
    return mission_dir(mission) / "_openmdao"


@contextlib.contextmanager
def workdir(path):
    """Point OpenMDAO's output directory at ``path`` for the duration of the block."""
    old = os.environ.get("OPENMDAO_WORKDIR")
    os.environ["OPENMDAO_WORKDIR"] = str(path)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("OPENMDAO_WORKDIR", None)
        else:
            os.environ["OPENMDAO_WORKDIR"] = old
