"""Where driftguard finds its packaged files and its database.

Packaged files (frozen eval sets, detector settings) live inside the installed
package, so they are found the same way from a git checkout and from pip.
User data (the database, calibration output) is never written into the package.
"""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "configs"


def default_db_path() -> str:
    """DRIFTGUARD_DB if set, else driftguard.db in the current directory."""
    return os.environ.get("DRIFTGUARD_DB", "driftguard.db")


def calibration_path() -> Path:
    """DRIFTGUARD_CALIBRATION if set, else the calibration file shipped in the package."""
    env = os.environ.get("DRIFTGUARD_CALIBRATION")
    return Path(env) if env else CONFIG_DIR / "detectors_calibrated.json"
