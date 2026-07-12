import glob
import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _find_pipeline_path():
    """Locate the main pipeline script without hardcoding a version number -
    it has already been renamed once (v2.8 -> v2.9); pinning the filename
    here just breaks the test suite again on the next rename."""
    candidates = glob.glob(os.path.join(REPO_ROOT, "bci_pipeline_v*.py"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No bci_pipeline_v*.py found in {REPO_ROOT}.")
    raise FileNotFoundError(
        f"Multiple bci_pipeline_v*.py files found in {REPO_ROOT}: {candidates}. "
        "Remove stale copies, or update tests/conftest.py to disambiguate."
    )


@pytest.fixture(scope="session")
def bp():
    """The main pipeline script, loaded via importlib since its filename
    isn't a valid Python module name (dot before the version suffix)."""
    path = _find_pipeline_path()
    spec = importlib.util.spec_from_file_location("bci_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
