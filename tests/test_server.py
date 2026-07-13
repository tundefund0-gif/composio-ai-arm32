"""Tests for the upgraded server."""
import pytest
from httpx import AsyncClient, ASGITransport
from server.main import app

@pytest.fixture
async def cl():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

class TestServer:
    @pytest.mark.anyio
    async def test_health(self, cl):
        r = await cl.get("/api/health"); assert r.status_code == 200
        assert r.json()["status"] == "ok"

    @pytest.mark.anyio
    async def test_chat(self, cl):
        r = await cl.post("/api/chat", json={"message":"Say hi","user_id":"pytest"})
        assert r.status_code == 200; d = r.json()
        assert d["response"] or d["reasoning"]

    @pytest.mark.anyio
    async def test_chat_empty(self, cl):
        assert (await cl.post("/api/chat", json={"message":"","user_id":"t"})).status_code == 400

    @pytest.mark.anyio
    async def test_session(self, cl):
        await cl.post("/api/chat", json={"message":"hi","user_id":"pytest_s"})
        r = await cl.get("/api/session/pytest_s"); assert r.status_code == 200
        assert r.json()["user_id"] == "pytest_s"

    @pytest.mark.anyio
    async def test_reset(self, cl):
        assert (await cl.post("/api/session/t/reset")).json()["status"] == "cleared"

    @pytest.mark.anyio
    async def test_tools_list(self, cl):
        r = await cl.get("/api/tools/list?page=1&page_size=3")
        assert r.status_code == 200 and "items" in r.json()

    @pytest.mark.anyio
    async def test_frontend(self, cl):
        r = await cl.get("/"); assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Zen Agent" in r.text
