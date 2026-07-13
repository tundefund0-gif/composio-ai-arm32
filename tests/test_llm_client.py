"""Tests for the upgraded LLM client."""
import pytest
from core.llm_client import LLMClient, LLMError

class TestLLM:
    @pytest.fixture
    def llm(self): return LLMClient(max_tokens=200)
    @pytest.fixture
    def llm_big(self): return LLMClient(max_tokens=500)

    def test_basic(self, llm):
        r = llm.chat([{"role":"user","content":"Say hi"}])
        assert r.content or r.reasoning

    def test_reasoning(self, llm_big):
        r = llm_big.chat([{"role":"user","content":"Say hello"}])
        assert r.reasoning or r.content

    def test_streaming(self, llm):
        tokens = []
        for t in llm.chat([{"role":"user","content":"Count 1 2 3"}], stream=True):
            tokens.append(t)
            if len(tokens) > 15: break
        assert len(tokens) > 0

    def test_complete(self, llm_big):
        r = llm_big.complete("Say 'test'")
        assert r.strip()

    def test_invalid_model(self):
        with pytest.raises(LLMError): LLMClient(model="bad-model-nonexistent").chat([{"role":"user","content":"hi"}])
