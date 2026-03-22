"""Tests for config loading/saving and benchmark.py CLI."""

import json
import subprocess
import sys

import pytest

from tests.conftest import REPO_ROOT


def _import_app(monkeypatch):
    """Import benchmark module with streamlit stubbed out."""
    import types

    st = types.ModuleType("streamlit")
    st.error = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.success = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "streamlit", st)

    import importlib
    import benchmark as _app

    importlib.reload(_app)
    return _app


class TestLoadConfig:
    def test_loads_from_config_json(self, config_dir, monkeypatch):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "CONFIG_PATH", config_dir / "config.json")
        monkeypatch.setattr(app, "CONFIG_EXAMPLE_PATH", config_dir / "nonexistent.json")
        cfg = app.load_config()
        assert cfg["execution"]["nersc_user"] == "testuser"

    def test_falls_back_to_example(self, tmp_dir, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        example_path = tmp_dir / "config.example.json"
        with open(example_path, "w") as f:
            json.dump(sample_config, f)
        monkeypatch.setattr(app, "CONFIG_PATH", tmp_dir / "config.json")
        monkeypatch.setattr(app, "CONFIG_EXAMPLE_PATH", example_path)
        cfg = app.load_config()
        assert cfg["execution"]["nersc_user"] == "testuser"


class TestSaveConfig:
    def test_save_and_reload(self, tmp_dir, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        out = tmp_dir / "config.json"
        monkeypatch.setattr(app, "CONFIG_PATH", out)
        app.save_config(sample_config)
        assert out.exists()
        with open(out) as f:
            loaded = json.load(f)
        assert loaded == sample_config

    def test_save_overwrites(self, tmp_dir, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        out = tmp_dir / "config.json"
        monkeypatch.setattr(app, "CONFIG_PATH", out)
        app.save_config(sample_config)
        sample_config["execution"]["nersc_user"] = "newuser"
        app.save_config(sample_config)
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["execution"]["nersc_user"] == "newuser"


class TestPdbIds:
    def test_load_empty(self, tmp_dir, monkeypatch):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "PDB_IDS_PATH", tmp_dir / "pdb_ids.json")
        assert app.load_pdb_ids() == []

    def test_save_and_load(self, tmp_dir, monkeypatch):
        app = _import_app(monkeypatch)
        path = tmp_dir / "pdb_ids.json"
        monkeypatch.setattr(app, "PDB_IDS_PATH", path)
        ids = ["1ABC", "2DEF", "3GHI"]
        app.save_pdb_ids(ids)
        assert app.load_pdb_ids() == ids

    def test_roundtrip_preserves_order(self, tmp_dir, monkeypatch):
        app = _import_app(monkeypatch)
        path = tmp_dir / "pdb_ids.json"
        monkeypatch.setattr(app, "PDB_IDS_PATH", path)
        ids = ["ZZZZ", "AAAA", "MMMM"]
        app.save_pdb_ids(ids)
        assert app.load_pdb_ids() == ids


class TestReadConfigCLI:
    def test_simple_key(self, config_env):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "benchmark.py"), "read-config", "execution.nersc_user"],
            capture_output=True,
            text=True,
            env={**dict(__import__("os").environ), "HELIXNET_CONFIG_DIR": str(config_env)},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "testuser"

    def test_nested_key(self, config_env):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "benchmark.py"), "read-config", "slurm.account"],
            capture_output=True,
            text=True,
            env={**dict(__import__("os").environ), "HELIXNET_CONFIG_DIR": str(config_env)},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "m4229"

    def test_json_value(self, config_env):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "benchmark.py"), "read-config", "openmm.forcefield"],
            capture_output=True,
            text=True,
            env={**dict(__import__("os").environ), "HELIXNET_CONFIG_DIR": str(config_env)},
        )
        assert result.returncode == 0
        parsed = json.loads(result.stdout.strip())
        assert parsed == ["amber14-all.xml", "amber14/tip3pfb.xml"]

    def test_missing_key_fails(self, config_env):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "benchmark.py"), "read-config", "nonexistent.key"],
            capture_output=True,
            text=True,
            env={**dict(__import__("os").environ), "HELIXNET_CONFIG_DIR": str(config_env)},
        )
        assert result.returncode != 0
