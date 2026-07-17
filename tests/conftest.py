import glob
import importlib.util
import os
import re
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_VERSION_RE = re.compile(r"^bci_pipeline_v(\d+(?:\.\d+)*)\.py$")


def _parse_version(path):
    match = _VERSION_RE.match(os.path.basename(path))
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _find_pipeline_path():
    """Locate the main pipeline script without hardcoding a version number -
    it has already been renamed/duplicated across versions (v2.8, v2.9,
    v2.9.1, ...). When more than one bci_pipeline_v*.py coexists, pick the
    highest parsed version instead of erroring: during active development,
    an older copy sitting alongside a newer one is expected, not exceptional.
    """
    candidates = glob.glob(os.path.join(REPO_ROOT, "bci_pipeline_v*.py"))
    if not candidates:
        raise FileNotFoundError(f"No bci_pipeline_v*.py found in {REPO_ROOT}.")
    if len(candidates) == 1:
        return candidates[0]

    versioned = [(path, _parse_version(path)) for path in candidates]
    if any(version is None for _, version in versioned):
        raise FileNotFoundError(
            f"Multiple bci_pipeline_v*.py files found in {REPO_ROOT}, and at least "
            f"one filename doesn't match the expected vX.Y[.Z].py pattern: {candidates}. "
            "Remove stale copies, or update tests/conftest.py to disambiguate."
        )
    versioned.sort(key=lambda item: item[1])
    chosen_path, _ = versioned[-1]
    print(f"[conftest] Multiple bci_pipeline_v*.py found, using the highest version: "
          f"{os.path.basename(chosen_path)} (candidates: "
          f"{[os.path.basename(p) for p, _ in versioned]})")
    return chosen_path


@pytest.fixture(scope="session")
def bp():
    """The main pipeline script, loaded via importlib since its filename
    isn't a valid Python module name (dot before the version suffix)."""
    path = _find_pipeline_path()
    spec = importlib.util.spec_from_file_location("bci_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
