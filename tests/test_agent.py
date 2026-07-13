"""Tests for the upgraded agent."""
import pytest
from core.agent import ZenAgent
from core.llm_client import LLMResponse

class TestAgent:
    @pytest.fixture
    def a(self): return ZenAgent("pytest_agent", enable_sandbox=False)

    def test_init(self, a):
        i = a.get_info(); assert i["session_id"] and i["user_id"] == "pytest_agent"

    def test_chat(self, a):
        r = a.chat("Say hello in one word"); assert isinstance(r, LLMResponse)
        assert r.content.strip()

    def test_history(self, a):
        a.chat("msg1"); a.chat("msg2"); assert len(a.get_history()) == 4

    def test_clear(self, a):
        a.chat("msg"); a.clear_history(); assert len(a.get_history()) == 0

    def test_info(self, a):
        i = a.get_info(); assert all(k in i for k in ["user_id","session_id","message_count"])

    def test_streaming(self, a):
        tokens = list(a.chat("Say hi", stream=True))
        assert len(tokens) > 0; assert "".join(tokens).strip()

    def test_tool_exec(self, a):
        r = a._exec_composio("COMPOSIO_SEARCH_TOOLS", {"queries":[{"use_case":"test"}]})
        assert r is not None
