"""Tests for app.py helper functions (no Streamlit UI)."""

import importlib
import json
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest
import requests


def _import_app(monkeypatch):
    st = types.ModuleType("streamlit")
    st.error = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.success = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "streamlit", st)
    import app as _app
    importlib.reload(_app)
    return _app


class TestDetectExecutionMode:
    def test_local_on_nersc(self, monkeypatch):
        app = _import_app(monkeypatch)
        monkeypatch.setattr("os.uname", lambda: types.SimpleNamespace(nodename="login01.perlmutter.nersc.gov"))
        assert app.detect_execution_mode() == "local"

    def test_ssh_on_mac(self, monkeypatch):
        app = _import_app(monkeypatch)
        monkeypatch.setattr("os.uname", lambda: types.SimpleNamespace(nodename="MacBook-Pro.local"))
        assert app.detect_execution_mode() == "ssh"

    def test_ssh_on_generic(self, monkeypatch):
        app = _import_app(monkeypatch)
        monkeypatch.setattr("os.uname", lambda: types.SimpleNamespace(nodename="workstation"))
        assert app.detect_execution_mode() == "ssh"


class TestStripAnsi:
    def test_strips_color_codes(self, monkeypatch):
        app = _import_app(monkeypatch)
        assert app.strip_ansi("\x1B[31mRed\x1B[0m") == "Red"

    def test_strips_bold(self, monkeypatch):
        app = _import_app(monkeypatch)
        assert app.strip_ansi("\x1B[1mBold\x1B[0m") == "Bold"

    def test_no_ansi_passthrough(self, monkeypatch):
        app = _import_app(monkeypatch)
        assert app.strip_ansi("plain text") == "plain text"

    def test_empty_string(self, monkeypatch):
        app = _import_app(monkeypatch)
        assert app.strip_ansi("") == ""

    def test_multiple_codes(self, monkeypatch):
        app = _import_app(monkeypatch)
        assert app.strip_ansi("\x1B[1;32mGreen Bold\x1B[0m normal") == "Green Bold normal"


class TestAutoMethod:
    def test_small_payload_uses_get(self, monkeypatch):
        app = _import_app(monkeypatch)
        small = {"query": {"type": "terminal"}, "return_type": "entry"}
        assert app._auto_method(small) == "get"

    def test_large_payload_uses_post(self, monkeypatch):
        app = _import_app(monkeypatch)
        large = {"query": {"type": "group", "nodes": [{"value": "x" * 3000}]}}
        assert app._auto_method(large) == "post"

    def test_boundary(self, monkeypatch):
        app = _import_app(monkeypatch)
        payload = {"x": "a" * 1990}
        method = app._auto_method(payload)
        encoded_len = len(json.dumps(payload))
        if encoded_len <= 2000:
            assert method == "get"
        else:
            assert method == "post"


class TestResolveOutDir:
    def test_empty_returns_base(self, monkeypatch):
        app = _import_app(monkeypatch)
        base = pathlib.Path("/some/base")
        cfg = {"paths": {"out_dir": ""}}
        assert app._resolve_out_dir(cfg, base) == base

    def test_relative_path(self, monkeypatch):
        app = _import_app(monkeypatch)
        base = pathlib.Path("/some/base")
        cfg = {"paths": {"out_dir": "out"}}
        assert app._resolve_out_dir(cfg, base) == base / "out"

    def test_absolute_path(self, monkeypatch):
        app = _import_app(monkeypatch)
        base = pathlib.Path("/some/base")
        cfg = {"paths": {"out_dir": "/absolute/out"}}
        assert app._resolve_out_dir(cfg, base) == pathlib.Path("/absolute/out")

    def test_whitespace_stripped(self, monkeypatch):
        app = _import_app(monkeypatch)
        base = pathlib.Path("/base")
        cfg = {"paths": {"out_dir": "  "}}
        assert app._resolve_out_dir(cfg, base) == base

    def test_missing_out_dir(self, monkeypatch):
        app = _import_app(monkeypatch)
        base = pathlib.Path("/base")
        cfg = {"paths": {}}
        assert app._resolve_out_dir(cfg, base) == base


class TestCredentialsGatePassed:
    def test_local_always_passes(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "local")
        assert app.credentials_gate_passed(sample_config) is True

    def test_ssh_with_user_passes(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "ssh")
        assert app.credentials_gate_passed(sample_config) is True

    def test_ssh_without_user_fails(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "ssh")
        sample_config["execution"]["nersc_user"] = ""
        assert app.credentials_gate_passed(sample_config) is False

    def test_ssh_missing_execution_fails(self, monkeypatch):
        app = _import_app(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "ssh")
        assert app.credentials_gate_passed({}) is False


class TestBuildRcsbPayload:
    def test_default_payload_structure(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        payload = app.build_rcsb_payload(sample_config)
        assert payload["return_type"] == "entry"
        query = payload["query"]
        assert query["type"] == "group"
        assert query["logical_operator"] == "and"
        assert len(query["nodes"]) >= 3

    def test_keyword_nodes(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        payload = app.build_rcsb_payload(sample_config)
        kw_group = payload["query"]["nodes"][0]
        assert kw_group["type"] == "group"
        assert kw_group["logical_operator"] == "or"
        assert len(kw_group["nodes"]) == 2

    def test_min_resolution_added(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        sample_config["rcsb_search"]["min_resolution"] = 1.0
        payload = app.build_rcsb_payload(sample_config)
        nodes = payload["query"]["nodes"]
        res_nodes = [n for n in nodes if n.get("parameters", {}).get("operator") == "greater_or_equal"]
        assert len(res_nodes) == 1
        assert res_nodes[0]["parameters"]["value"] == 1.0

    def test_no_min_resolution(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        sample_config["rcsb_search"]["min_resolution"] = None
        payload = app.build_rcsb_payload(sample_config)
        nodes = payload["query"]["nodes"]
        res_nodes = [n for n in nodes if n.get("parameters", {}).get("operator") == "greater_or_equal"]
        assert len(res_nodes) == 0


class TestRcsbHandle:
    def _make_response(self, status_code, json_data=None, text=""):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = status_code
        resp.text = text
        if json_data is not None:
            resp.json.return_value = json_data
        else:
            resp.json.side_effect = ValueError("No JSON")
        return resp

    def test_200_with_json(self, monkeypatch):
        app = _import_app(monkeypatch)
        resp = self._make_response(200, {"total_count": 5, "result_set": []})
        data, err = app._rcsb_handle(resp)
        assert data == {"total_count": 5, "result_set": []}
        assert err is None

    def test_204_returns_empty(self, monkeypatch):
        app = _import_app(monkeypatch)
        resp = self._make_response(204)
        data, err = app._rcsb_handle(resp)
        assert data["total_count"] == 0
        assert err is None

    def test_400_returns_error(self, monkeypatch):
        app = _import_app(monkeypatch)
        resp = self._make_response(400, {"message": "bad"})
        data, err = app._rcsb_handle(resp)
        assert "Bad Request" in err

    def test_500_returns_error(self, monkeypatch):
        app = _import_app(monkeypatch)
        resp = self._make_response(500, {"message": "fail"})
        data, err = app._rcsb_handle(resp)
        assert "Internal Server Error" in err

    def test_200_non_json(self, monkeypatch):
        app = _import_app(monkeypatch)
        resp = self._make_response(200, text="not json")
        resp.json.side_effect = ValueError
        data, err = app._rcsb_handle(resp)
        assert data is None
        assert "non-JSON" in err
