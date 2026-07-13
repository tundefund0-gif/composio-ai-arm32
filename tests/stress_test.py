"""Stress test — every component under maximum load."""
import json, sys, time

PASS = 0; FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  ✅ {name}")
    except Exception as e:
        FAIL += 1
        m = str(e).split('\n')[0][:100]
        print(f"  ❌ {name}: {m}" if m else f"  ❌ {name}")
    sys.stdout.flush()

print("=" * 60)
print("🧪 ZEN AGENT — COMPREHENSIVE STRESS TEST")
print("=" * 60)

# ─── 1. COMPOSIO ────────────────────────────────────────
print("\n═══ 1. COMPOSIO ═══")
from core.composio_client import ComposioClient, ComposioAPIError
c = ComposioClient()

s1 = c.create_session("stress_1")
assert s1.get("session_id")
sid1 = s1["session_id"]
test("session created with meta tools", lambda: None)
assert len(s1.get("tool_router_tools",[])) > 0
assert s1.get("mcp",{}).get("url")
c.get_session(sid1)

t = c.list_all_tools(1, 3)
assert t["total_items"] >= 23000
test("23,790 tools in catalog", lambda: None)

r = c.search_tools(sid1, "manage github issues")
assert len(r.get("data",{}).get("tool_schemas",{})) > 0
test("search github tools", lambda: None)
c.search_tools(sid1, "send emails"); test("search email tools", lambda: None)
c.search_tools(sid1, "slack messages"); test("search slack tools", lambda: None)

c.get_tool_schemas(sid1, ["COMPOSIO_SEARCH_TOOLS","COMPOSIO_REMOTE_WORKBENCH"]); test("meta schemas", lambda: None)
c.get_tool_schemas(sid1, ["GITHUB_GET_A_REPOSITORY"]); test("app schemas", lambda: None)

c.multi_execute(sid1, [
    {"tool_slug":"COMPOSIO_SEARCH_TOOLS","arguments":{"queries":[{"use_case":"github"}]}},
    {"tool_slug":"COMPOSIO_SEARCH_TOOLS","arguments":{"queries":[{"use_case":"gmail"}]}},
]); test("multi-execute", lambda: None)

s2 = c.create_session("stress_2", toolkits=["github","gmail"])
assert s2.get("session_id"); c.delete_session(s2["session_id"])
test("session with toolkits + delete", lambda: None)

s3 = c.create_session("stress_3", sandbox=True)
assert s3.get("session_id")
test("session with sandbox", lambda: None)
c.execute_workbench(s3["session_id"], "print('wb')"); test("execute workbench", lambda: None)

c.manage_connections(sid1, ["github"]); test("manage connections", lambda: None)
c.config_history(sid1); test("config history", lambda: None)
c.get_tools(sid1); test("get session tools", lambda: None)
c.list_all_tools(2, 5); test("paginated tools", lambda: None)

try: c.get_session("nonexistent_session_xyz"); assert False
except ComposioAPIError: test("error on bad session", lambda: None)

# ─── 2. LLM ─────────────────────────────────────────────
print("\n═══ 2. LLM ═══")
from core.llm_client import LLMClient, LLMError
llm = LLMClient(max_tokens=800)

r = llm.chat([{"role":"user","content":"Say ping"}])
assert r.content.strip(); test("basic chat content", lambda: None)
assert r.reasoning; test("reasoning present", lambda: None)
assert r.model; test("model name", lambda: None)
assert r.usage.get("total_tokens",0) > 0; test("usage stats", lambda: None)

assert llm.chat([{"role":"system","content":"Be brief"},{"role":"user","content":"ok"}]).content.strip()
test("system prompt", lambda: None)

tokens = list(llm.chat([{"role":"user","content":"Count 1-3"}], stream=True))
assert len(tokens) > 0; test("streaming tokens", lambda: None)
assert "".join(tokens).strip(); test("streaming complete text", lambda: None)

assert llm.complete("Say complete").strip(); test("complete helper", lambda: None)

try: LLMClient(model="bad").chat([{"role":"user","content":"hi"}]); assert False
except LLMError: test("invalid model error", lambda: None)

# ─── 3. AGENT ───────────────────────────────────────────
print("\n═══ 3. AGENT ═══")
from core.agent import ZenAgent
from core.llm_client import LLMResponse

agent = ZenAgent("stress_agent", enable_sandbox=False)
assert agent.session_id
assert agent.get_info()["message_count"] == 0
test("agent initialized", lambda: None)

r = agent.chat("Say 'hello world'")
assert isinstance(r, LLMResponse) and r.content.strip()
test("agent chat response", lambda: None)

tokens = list(agent.chat("Say 'hello'", stream=True))
assert len(tokens) > 0 and "".join(tokens).strip()
test("agent streaming", lambda: None)

assert len(agent.get_history()) == 4; test("history preserved", lambda: None)
agent.clear_history()
assert len(agent.get_history()) == 0; test("clear history", lambda: None)

agent.chat("test")
assert agent.get_info()["message_count"] == 2; test("message count", lambda: None)

agent2 = ZenAgent("stress_reuse", session_id=agent.session_id, enable_sandbox=False)
assert agent2.session_id == agent.session_id; test("session reuse", lambda: None)

tr = agent2._exec_composio("COMPOSIO_SEARCH_TOOLS", {"queries":[{"use_case":"test"}]})
assert tr is not None; test("tool execution", lambda: None)

# ─── 4. SERVER ──────────────────────────────────────────
print("\n═══ 4. SERVER ═══")
import httpx

SERVER_OK = False
try:
    h = httpx.get("http://localhost:8000/api/health", timeout=5)
    SERVER_OK = h.status_code == 200
except: pass

if SERVER_OK:
    assert h.json()["status"] == "ok"
    assert "agents_active" in h.json()
    test("server health ok", lambda: None)

    r2 = httpx.post("http://localhost:8000/api/chat", json={"message":"Say 'hello server'","user_id":"stress_server"}, timeout=30)
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2.get("response","").strip()
    assert isinstance(d2.get("reasoning"), str)
    assert d2.get("session_id")
    assert isinstance(d2.get("tool_calls"), list)
    test("full chat response with reasoning & tools", lambda: None)

    r3 = httpx.get("http://localhost:8000/api/session/stress_server", timeout=5)
    assert r3.json().get("message_count") is not None; test("session info", lambda: None)

    r4 = httpx.post("http://localhost:8000/api/session/stress_server/reset", timeout=5)
    assert r4.json()["status"] == "cleared"; test("session reset", lambda: None)

    r5 = httpx.get("http://localhost:8000/api/session/stress_server", timeout=5)
    assert r5.json()["message_count"] == 0; test("reset verified", lambda: None)

    r6 = httpx.get("http://localhost:8000/api/tools/list?page=1&page_size=3", timeout=10)
    assert r6.json().get("total_items",0) > 0; test("tools list from server", lambda: None)

    r7 = httpx.post("http://localhost:8000/api/chat", json={"message":"","user_id":"t"}, timeout=5)
    assert r7.status_code == 400; test("empty message validation", lambda: None)

    r8 = httpx.get("http://localhost:8000/", timeout=5)
    assert "text/html" in r8.headers.get("content-type","")
    assert "Zen Agent" in r8.text
    test("frontend dashboard", lambda: None)

else:
    test("server available on :8000", lambda: (_ for _ in ()).throw(AssertionError("Server not running")))

# ─── SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"🏁  FINAL: {PASS} PASSED  |  {FAIL} FAILED  |  {PASS+FAIL} TOTAL TESTS")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
