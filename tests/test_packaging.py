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
    for cmd in ("init", "eval", "collect", "proxy", "detect", "dashboard"):
        assert parser.parse_known_args([cmd, *needs_args.get(cmd, [])])[0].command == cmd


def test_version_is_set():
    assert __version__
