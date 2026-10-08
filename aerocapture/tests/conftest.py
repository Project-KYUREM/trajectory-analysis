"""Shared fixtures: the default epoch, and each test's OpenMDAO output directory."""
import os

import pytest

from aerocapture.core import paths
from aerocapture.core import system


@pytest.fixture(autouse=True)
def openmdao_workdir(request, monkeypatch):
    """Point OpenMDAO at the test's mission work dir unless ``OPENMDAO_WORKDIR`` is already set."""
    if "OPENMDAO_WORKDIR" not in os.environ:
        mission = "triton" if request.path.parent.name == "triton" else "neptune"
        path = paths.work_dir(mission)
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("OPENMDAO_WORKDIR", str(path))


@pytest.fixture(scope="module")
def epoch():
    """Return the default epoch."""
    return system.make_epoch()
