"""Tests for RCSB API functions in app.py.

Live integration tests hit the real RCSB API.
Mocked tests verify error handling for all documented status codes.
"""

import importlib
import json
import sys
import types

import pytest
import responses

from tests.conftest import SAMPLE_CONFIG

pytestmark = pytest.mark.rcsb


def _import_app(monkeypatch):
    st = types.ModuleType("streamlit")
    st.error = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.success = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "streamlit", st)
    import app as _app
    importlib.reload(_app)
    return _app


# -----------------------------------------------------------------------
# Live integration tests
# -----------------------------------------------------------------------

class TestExecuteRcsbSearchLive:
    def test_post_returns_ids(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, sent = app.execute_rcsb_search(payload, method="post",
                                                  request_options={"paginate": {"start": 0, "rows": 5}})
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, str) for i in ids)
        assert raw.get("total_count", 0) > 0

    def test_get_returns_ids(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, sent = app.execute_rcsb_search(payload, method="get",
                                                  request_options={"paginate": {"start": 0, "rows": 5}})
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_pagination(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="post",
                                               request_options={"paginate": {"start": 0, "rows": 3}})
        assert len(ids) >= 3


class TestRcsbSuggestLive:
    def test_suggest_dna(self, monkeypatch):
        app = _import_app(monkeypatch)
        suggestions, err = app.rcsb_suggest("DNA")
        assert err is None
        assert isinstance(suggestions, dict)

    def test_suggest_empty_query(self, monkeypatch):
        app = _import_app(monkeypatch)
        suggestions, err = app.rcsb_suggest("xyznonexistent999")
        assert err is None or suggestions == {}


class TestRcsbSearchUnreleasedLive:
    def test_unreleased_search(self, monkeypatch):
        app = _import_app(monkeypatch)
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_repository_holdings_unreleased.title",
                "operator": "contains_phrase",
                "value": "DNA",
            },
        }
        ids, raw = app.rcsb_search_unreleased(query)
        assert isinstance(ids, list)
        assert isinstance(raw, dict)


class TestRcsbGetMetadataLive:
    @pytest.mark.parametrize("schema_type", ["structure", "chemical", "unreleased"])
    def test_metadata_schemas(self, monkeypatch, schema_type):
        app = _import_app(monkeypatch)
        schema, err = app.rcsb_get_metadata(schema_type)
        assert err is None
        assert isinstance(schema, dict)
        assert len(schema) > 0


# -----------------------------------------------------------------------
# Mocked error-path tests
# -----------------------------------------------------------------------

class TestExecuteRcsbSearchMocked:
    @responses.activate
    def test_204_no_content(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        responses.add(responses.POST, app.RCSB_SEARCH_URL, status=204)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="post")
        assert ids == []
        assert raw["total_count"] == 0

    @responses.activate
    def test_400_bad_request(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        responses.add(responses.POST, app.RCSB_SEARCH_URL, json={"message": "bad"}, status=400)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="post")
        assert ids == []

    @responses.activate
    def test_500_server_error(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        responses.add(responses.POST, app.RCSB_SEARCH_URL, json={"message": "fail"}, status=500)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="post")
        assert ids == []

    @responses.activate
    def test_408_timeout(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        responses.add(responses.POST, app.RCSB_SEARCH_URL, json={"message": "timeout"}, status=408)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="post")
        assert ids == []

    @responses.activate
    def test_503_unavailable(self, monkeypatch, sample_config):
        app = _import_app(monkeypatch)
        responses.add(responses.POST, app.RCSB_SEARCH_URL, json={"message": "unavailable"}, status=503)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="post")
        assert ids == []

    @responses.activate
    def test_connection_error(self, monkeypatch, sample_config):
        import requests as req
        app = _import_app(monkeypatch)
        responses.add(responses.POST, app.RCSB_SEARCH_URL,
                      body=req.ConnectionError("network down"))
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="post")
        assert ids == []
        assert "error" in raw


class TestRcsbSuggestMocked:
    @responses.activate
    def test_204_no_suggestions(self, monkeypatch):
        app = _import_app(monkeypatch)
        responses.add(responses.GET, app.RCSB_SUGGEST_URL, status=204)
        suggestions, err = app.rcsb_suggest("test")
        assert suggestions == {}
        assert err is None

    @responses.activate
    def test_500_error(self, monkeypatch):
        app = _import_app(monkeypatch)
        responses.add(responses.GET, app.RCSB_SUGGEST_URL, status=500)
        suggestions, err = app.rcsb_suggest("test")
        assert "Internal Server Error" in err


class TestRcsbMetadataMocked:
    @responses.activate
    def test_404_not_found(self, monkeypatch):
        app = _import_app(monkeypatch)
        responses.add(responses.GET, app.RCSB_META_URLS["structure"], status=404)
        schema, err = app.rcsb_get_metadata("structure")
        assert schema is None
        assert "Not Found" in err
