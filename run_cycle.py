#!/usr/bin/env python3
"""Compatibility shim: `python run_cycle.py` still works. Prefer `driftguard collect`.

Keeps the old behaviour of storing logs/ and driftguard.db next to this file, so a
scheduled task that runs this script from any working directory keeps using the same files.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("DRIFTGUARD_HOME", str(Path(__file__).resolve().parent))

from driftguard.collect import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
