"""The `driftguard` command: one entry point for collecting, detecting, capturing and viewing."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from driftguard import __version__
from driftguard._paths import PACKAGE_DIR, default_db_path

# The dashboard's CSS is written for a light page. Streamlit only reads .streamlit/config.toml from the
# current directory, so without these flags a dark-mode browser makes the headings unreadable.
# Keep in sync with .streamlit/config.toml in the repo root.
THEME_FLAGS = [
    "--theme.base", "light",
    "--theme.primaryColor", "#2a78d6",
    "--theme.backgroundColor", "#f4f4f1",
    "--theme.secondaryBackgroundColor", "#ffffff",
    "--theme.textColor", "#0b0b0b",
    "--client.toolbarMode", "minimal",
    "--browser.gatherUsageStats", "false",
]

ENV_TEMPLATE = """\
# driftguard API keys: fill in the providers you use.
GROQ_API_KEY=
OPENROUTER_API_KEY=
NVIDIA_API_KEY=
"""


def _cmd_init(_: argparse.Namespace) -> int:
    env = Path(".env")
    if env.exists():
        print(".env already exists, leaving it alone.")
    else:
        env.write_text(ENV_TEMPLATE, encoding="utf-8")
        print("Wrote .env. Add your API key(s), then run: driftguard eval --provider groq --model openai/gpt-oss-120b")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    from driftguard.detection.pipeline import run_detection
    from driftguard.storage.store import DriftStore

    report = run_detection(DriftStore(args.db_path), args.provider, args.model)
    print(f"{report.written} new events, {report.labelled} labelled")
    return 0


def _run_streamlit(script: Path, port: int, extra_env: dict | None = None) -> int:
    """Open a Streamlit page that ships inside the package, with the light theme forced."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print('This needs extra packages: pip install "llm-driftguard[dashboard]"', file=sys.stderr)
        return 1
    cmd = [sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port), *THEME_FLAGS]
    return subprocess.call(cmd, env=dict(os.environ, **(extra_env or {})))


def _cmd_dashboard(args: argparse.Namespace) -> int:
    return _run_streamlit(PACKAGE_DIR / "dashboard" / "app.py", args.port, {"DRIFTGUARD_DB": str(args.db_path)})


def _cmd_live(args: argparse.Namespace) -> int:
    return _run_streamlit(PACKAGE_DIR / "demo" / "live_page.py", args.port)


def _cmd_demo_data(rest: list[str]) -> int:
    from driftguard.demo.make_demo_db import main
    main(rest)
    return 0


def _cmd_eval(rest: list[str]) -> int:
    from driftguard.evalsuite.runner import _main
    sys.argv = ["driftguard eval", *rest]
    _main()
    return 0


def _cmd_collect(rest: list[str]) -> int:
    from driftguard.collect import main
    return main(rest)


def _cmd_proxy(rest: list[str]) -> int:
    from driftguard.ingest.proxy import _main
    sys.argv = ["driftguard proxy", *rest]
    _main()
    return 0


# Commands that hand their remaining arguments to an existing module's own argparse.
PASSTHROUGH = {"eval": _cmd_eval, "collect": _cmd_collect, "proxy": _cmd_proxy, "demo-data": _cmd_demo_data}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="driftguard", description="Detect silent behavioral drift in deployed LLM APIs.")
    p.add_argument("--version", action="version", version=f"driftguard {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="write a .env template for API keys").set_defaults(func=_cmd_init)

    sub.add_parser("eval", help="run the eval suite once, or on an interval (--provider, --model, ...)", add_help=False)
    sub.add_parser("collect", help="single-shot collector for a scheduler (--target provider:model ...)", add_help=False)
    sub.add_parser("proxy", help="run the reverse proxy that logs your app's LLM traffic (--port, ...)", add_help=False)
    sub.add_parser("demo-data", help="build a SIMULATED database to try the dashboard (--out FILE_demo.db)", add_help=False)

    de = sub.add_parser("detect", help="run detection and classification on stored calls")
    de.add_argument("--provider", required=True)
    de.add_argument("--model", required=True)
    de.add_argument("--db-path", default=default_db_path())
    de.set_defaults(func=_cmd_detect)

    lv = sub.add_parser("live", help="open the live demo page: your prompts, a real API, DriftGuard's detectors")
    lv.add_argument("--port", type=int, default=8501)
    lv.set_defaults(func=_cmd_live)

    da = sub.add_parser("dashboard", help="open the dashboard")
    da.add_argument("--db-path", default=default_db_path())
    da.add_argument("--port", type=int, default=8501)
    da.set_defaults(func=_cmd_dashboard)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)
    if args.command in PASSTHROUGH:
        return int(PASSTHROUGH[args.command](rest) or 0)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
