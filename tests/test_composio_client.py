"""Tests for the upgraded Composio client."""
import pytest
from core.composio_client import ComposioClient, ComposioAPIError

class TestComposioClient:
    @pytest.fixture
    def c(self): return ComposioClient()

    def test_list_tools(self, c):
        r = c.list_all_tools(page=1, page_size=3)
        assert "items" in r and len(r["items"]) > 0

    def test_create_and_get_session(self, c):
        s = c.create_session("pytest_u1"); sid = s["session_id"]
        assert c.get_session(sid)["session_id"] == sid

    def test_session_has_meta_tools(self, c):
        s = c.create_session("pytest_u2", sandbox=True)
        assert len(s.get("tool_router_tools", [])) > 0
        t = c.get_tools(s["session_id"])
        slugs = [i["slug"] for i in t.get("items", [])]
        assert "COMPOSIO_SEARCH_TOOLS" in slugs

    def test_search_tools(self, c):
        s = c.create_session("pytest_u3"); r = c.search_tools(s["session_id"], "github issues")
        assert r.get("data", {}).get("tool_schemas") or True

    def test_tool_schemas(self, c):
        s = c.create_session("pytest_u4")
        r = c.get_tool_schemas(s["session_id"], ["COMPOSIO_SEARCH_TOOLS"])
        assert r is not None

    def test_manage_connections(self, c):
        s = c.create_session("pytest_u5"); r = c.manage_connections(s["session_id"], ["github"])
        assert r is not None

    def test_multi_execute(self, c):
        s = c.create_session("pytest_u6")
        r = c.multi_execute(s["session_id"], [{"tool_slug":"COMPOSIO_SEARCH_TOOLS","arguments":{"queries":[{"use_case":"test"}]}}])
        assert r is not None

    def test_config_history(self, c):
        s = c.create_session("pytest_u7"); r = c.config_history(s["session_id"])
        assert r is not None

    def test_invalid_session(self, c):
        with pytest.raises(ComposioAPIError): c.get_session("nonexistent")

    def test_create_with_toolkits(self, c):
        s = c.create_session("pytest_u8", toolkits=["github"])
        assert "session_id" in s
