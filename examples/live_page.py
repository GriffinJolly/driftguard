"""Compatibility shim: `streamlit run examples/live_page.py` still works. Prefer `driftguard live`."""
from driftguard.demo.live_page import main

main()
