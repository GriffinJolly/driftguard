"""Guards the things that only break after `pip install`: packaged files and path defaults."""
import json

from driftguard import __version__
from driftguard._paths import CONFIG_DIR, calibration_path, default_db_path
from driftguard.cli import build_parser
from driftguard.detection import baseline, pipeline
from driftguard.evalsuite import refusal_probe, tasks


def test_packaged_config_files_exist():
    for name in ("detectors.yaml", "mmlu_frozen.json", "xstest_frozen.json", "detectors_calibrated.json"):
        assert (CONFIG_DIR / name).is_file(), name


def test_every_module_points_at_packaged_configs():
    for path in (tasks.FROZEN_PATH, refusal_probe.FROZEN_PATH, pipeline.DETECTORS_YAML,
                 pipeline.XSTEST_FROZEN, baseline.DEFAULT_CALIBRATION_PATH):
        assert path.is_file(), path
        assert CONFIG_DIR in path.parents


def test_frozen_sets_are_valid_json():
    assert len(json.loads(tasks.FROZEN_PATH.read_text(encoding="utf-8"))) > 0


def test_db_and_calibration_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DRIFTGUARD_DB", str(tmp_path / "x.db"))
    monkeypatch.setenv("DRIFTGUARD_CALIBRATION", str(tmp_path / "c.json"))
    assert default_db_path() == str(tmp_path / "x.db")
    assert calibration_path() == tmp_path / "c.json"


def test_cli_lists_all_commands():
    parser = build_parser()
    needs_args = {"detect": ["--provider", "p", "--model", "m"]}
    for cmd in ("init", "eval", "collect", "proxy", "demo-data", "live", "detect", "dashboard"):
        assert parser.parse_known_args([cmd, *needs_args.get(cmd, [])])[0].command == cmd


def test_version_is_set():
    assert __version__


def test_dashboard_forces_light_theme():
    from driftguard.cli import THEME_FLAGS
    i = THEME_FLAGS.index("--theme.base")
    assert THEME_FLAGS[i + 1] == "light"


def test_dotenv_in_current_folder_is_found(tmp_path):
    """`driftguard init` writes .env in the current folder; the CLI must read it from there."""
    import subprocess
    import sys
    from pathlib import Path

    (tmp_path / ".env").write_text("GROQ_API_KEY=from_dotenv_file\n", encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text("import os\nfrom driftguard.providers import clients\nprint(os.getenv('GROQ_API_KEY'))\n")
    repo_root = str(Path(__file__).resolve().parent.parent)
    env = {k: v for k, v in __import__("os").environ.items() if k != "GROQ_API_KEY"}
    env["PYTHONPATH"] = repo_root
    out = subprocess.run([sys.executable, str(probe)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "from_dotenv_file", out.stderr


def test_demo_data_command_builds_the_simulated_database(tmp_path):
    """`driftguard demo-data` must work from the installed package alone (no repo checkout)."""
    import sqlite3

    from driftguard.cli import main
    from driftguard.demo import make_demo_db

    out = tmp_path / "t_demo.db"
    assert main(["demo-data", "--out", str(out)]) == 0
    con = sqlite3.connect(out)
    try:
        n = con.execute("SELECT COUNT(*) FROM log_rows").fetchone()[0]
        models = {r[0] for r in con.execute("SELECT DISTINCT model_id FROM log_rows")}
    finally:
        con.close()
    assert n == make_demo_db.PASSES * len(make_demo_db._tasks())
    assert models == {make_demo_db.MODEL_ID}


def test_demo_data_refuses_to_overwrite_a_non_demo_file(tmp_path):
    import pytest

    from driftguard.cli import main

    real = tmp_path / "driftguard.db"
    real.write_text("do not touch")
    with pytest.raises(SystemExit):
        main(["demo-data", "--out", str(real)])
    assert real.read_text() == "do not touch"


def test_live_command_opens_the_packaged_page_with_the_light_theme(monkeypatch):
    import pytest

    pytest.importorskip("streamlit")
    from driftguard import cli

    seen = {}
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, env=None: seen.update(cmd=cmd) or 0)
    assert cli.main(["live", "--port", "8599"]) == 0
    cmd = seen["cmd"]
    assert cmd[:2] == cli.ISOLATED_PYTHON, "Streamlit must start in isolated mode"
    assert cmd[-len(cli.THEME_FLAGS):] == cli.THEME_FLAGS
    assert "8599" in cmd
    page = next(c for c in cmd if c.endswith("live_page.py"))
    assert page.replace("\\", "/").endswith("driftguard/demo/live_page.py")


def test_old_script_paths_still_work():
    """The README and DEMO.md used scripts/ and examples/ before the tools moved into the package."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    env = dict(__import__("os").environ, PYTHONPATH=str(root))
    out = subprocess.run([sys.executable, str(root / "scripts" / "make_demo_db.py"), "--help"],
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0 and "--out" in out.stdout, out.stderr


def test_a_local_driftguard_folder_cannot_shadow_the_installed_package(tmp_path):
    """Running `driftguard live` in a folder that has its own driftguard/ directory (an old clone, a
    half-finished merge) must still use the installed package."""
    import importlib.metadata
    import subprocess

    import pytest

    from driftguard import cli

    try:
        importlib.metadata.version("llm-driftguard")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("needs the package installed (pip install -e .)")
    fake = tmp_path / "driftguard"
    fake.mkdir()
    (fake / "__init__.py").write_text("raise RuntimeError('the local folder shadowed the installed package')\n")
    out = subprocess.run([*cli.ISOLATED_PYTHON, "-c", "import driftguard; print(driftguard.__file__)"],
                         cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert str(tmp_path) not in out.stdout
