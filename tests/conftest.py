import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture(scope="session")
def bp():
    """bci_pipeline_v2.8.py, loaded via importlib since its filename isn't a
    valid Python module name (dot before the version suffix)."""
    path = os.path.join(REPO_ROOT, "bci_pipeline_v2.8.py")
    spec = importlib.util.spec_from_file_location("bci_pipeline_v2_8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
