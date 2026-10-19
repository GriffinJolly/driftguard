"""Compatibility shim: `python scripts/make_demo_db.py` still works. Prefer `driftguard demo-data`.

The real code lives in driftguard/demo/make_demo_db.py. Importing `scripts.make_demo_db` returns that module.
"""
import sys

from driftguard.demo import make_demo_db as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
