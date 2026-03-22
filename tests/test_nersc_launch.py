"""Tests for the NERSC launch path: SSH connection, run_script, run_remote_cmd, scan_wp_dirs.

All paramiko interactions are mocked -- no real SSH connections.
"""

import importlib
import io
import os
import pathlib
import sys
import types
from unittest.mock import MagicMock, patch, call

import pytest


def _import_app(monkeypatch):
    st = types.ModuleType("streamlit")
    st.error = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.success = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "streamlit", st)
    import benchmark as _app
    importlib.reload(_app)
    return _app


class TestGetSshClient:
    def test_connects_with_valid_user(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        mock_paramiko = MagicMock()
        mock_client = MagicMock()
        mock_paramiko.SSHClient.return_value = mock_client
        mock_paramiko.AutoAddPolicy.return_value = "auto"
        monkeypatch.setitem(sys.modules, "paramiko", mock_paramiko)

        client = app._get_ssh_client(sample_config)

        assert client is mock_client
        mock_client.set_missing_host_key_policy.assert_called_once_with("auto")
        mock_client.connect.assert_called_once_with("perlmutter.nersc.gov", username="testuser")

    def test_returns_none_without_user(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        sample_config["execution"]["nersc_user"] = ""
        mock_paramiko = MagicMock()
        monkeypatch.setitem(sys.modules, "paramiko", mock_paramiko)

        client = app._get_ssh_client(sample_config)
        assert client is None

    def test_returns_none_on_connect_failure(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        mock_paramiko = MagicMock()
        mock_client = MagicMock()
        mock_client.connect.side_effect = Exception("Connection refused")
        mock_paramiko.SSHClient.return_value = mock_client
        mock_paramiko.AutoAddPolicy.return_value = "auto"
        monkeypatch.setitem(sys.modules, "paramiko", mock_paramiko)

        client = app._get_ssh_client(sample_config)
        assert client is None

    def test_returns_none_without_paramiko(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        monkeypatch.delitem(sys.modules, "paramiko", raising=False)

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "paramiko":
                raise ImportError("No module named 'paramiko'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        client = app._get_ssh_client(sample_config)
        assert client is None


class TestRunRemoteCmd:
    def test_executes_and_returns_stdout(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        mock_client = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b"hello world\n"
        mock_client.exec_command.return_value = (None, mock_stdout, None)
        monkeypatch.setattr(app, "_get_ssh_client", lambda cfg: mock_client)

        result = app.run_remote_cmd(sample_config, "echo hello world")
        assert result == "hello world\n"
        mock_client.exec_command.assert_called_once_with("echo hello world")
        mock_client.close.assert_called_once()

    def test_returns_empty_on_ssh_failure(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "_get_ssh_client", lambda cfg: None)
        result = app.run_remote_cmd(sample_config, "anything")
        assert result == ""


class TestRunScript:
    def test_ssh_mode_sends_correct_command(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "ssh")

        mock_client = MagicMock()
        stdout_lines = iter(["line1\n", "line2\n"])
        mock_stdout = MagicMock()
        mock_stdout.__iter__ = lambda self: stdout_lines
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.return_value = (None, mock_stdout, mock_stderr)
        monkeypatch.setattr(app, "_get_ssh_client", lambda cfg: mock_client)

        placeholder = MagicMock()
        placeholder.code = MagicMock()

        result = app.run_script(sample_config, "./run.sh batch", placeholder)

        expected_cmd = 'cd /tmp/helixnet_test && bash -lc "./run.sh batch"'
        mock_client.exec_command.assert_called_once_with(expected_cmd, get_pty=True)
        assert "line1" in result
        assert "line2" in result
        mock_client.close.assert_called_once()

    def test_local_mode_runs_subprocess(self, monkeypatch, sample_config, tmp_dir):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "local")
        monkeypatch.setattr(app, "APP_DIR", tmp_dir)

        script = tmp_dir / "test_script.sh"
        script.write_text("#!/bin/bash\necho 'hello from local'\n")
        script.chmod(0o755)

        placeholder = MagicMock()
        placeholder.code = MagicMock()

        result = app.run_script(sample_config, str(script), placeholder)
        assert "hello from local" in result

    def test_ssh_mode_returns_empty_on_failure(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "ssh")
        monkeypatch.setattr(app, "_get_ssh_client", lambda cfg: None)

        placeholder = MagicMock()
        result = app.run_script(sample_config, "./run.sh batch", placeholder)
        assert result == ""


class TestScanWpDirs:
    def test_local_mode_scans_dirs(self, monkeypatch, sample_config, tmp_dir):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "local")
        monkeypatch.setattr(app, "APP_DIR", tmp_dir)

        sample_config["paths"]["out_dir"] = ""
        sample_config["paths"]["project_dir"] = str(tmp_dir)

        wp1 = tmp_dir / "1ABC_WP"
        wp1.mkdir()
        (wp1 / "west.h5").write_bytes(b"fake h5 data")

        wp2 = tmp_dir / "2DEF_WP"
        wp2.mkdir()

        rows = app.scan_wp_dirs(sample_config)
        assert len(rows) == 2
        pdb_ids = {r["PDB ID"] for r in rows}
        assert "1ABC" in pdb_ids
        assert "2DEF" in pdb_ids

    def test_local_mode_empty(self, monkeypatch, sample_config, tmp_dir):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "local")
        monkeypatch.setattr(app, "APP_DIR", tmp_dir)
        sample_config["paths"]["out_dir"] = ""
        sample_config["paths"]["project_dir"] = str(tmp_dir)

        rows = app.scan_wp_dirs(sample_config)
        assert rows == []

    def test_ssh_mode_parses_listing(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "ssh")

        call_count = {"n": 0}
        responses = [
            "/path/out/1ABC_WP\n/path/out/2DEF_WP\n",
            "yes\n",
            "iter_00100 Group\niter_05000 Group\n",
            "no\n",
        ]

        def mock_remote(cfg, cmd):
            idx = call_count["n"]
            call_count["n"] += 1
            if idx < len(responses):
                return responses[idx]
            return ""

        monkeypatch.setattr(app, "run_remote_cmd", mock_remote)

        rows = app.scan_wp_dirs(sample_config)
        assert len(rows) == 2
        assert rows[0]["PDB ID"] == "1ABC"
        assert rows[0]["west.h5"] is True
