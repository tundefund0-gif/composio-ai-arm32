"""Zen Agent — FastAPI server with REST + WebSocket streaming, long-text, health."""
from __future__ import annotations
import asyncio

import json
import logging
import os
import signal
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import config
from core.agent import ZenAgent
from core.composio_client import ComposioClient, ComposioAPIError
from core.llm_client import LLMResponse

# Configure structured logging
_log_fmt = config.log_format
if _log_fmt == "json":
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                "ts": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "module": record.module,
                "line": record.lineno,
            })
    _handler = logging.StreamHandler()
    _handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO), handlers=[_handler])
else:
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
logger = logging.getLogger("zen-server")

# ── Agent store ─────────────────────────────────────────────
agents: Dict[str, ZenAgent] = {}

MAX_AGENTS = 200
_start_time = time.time()


def get_agent(user_id: str, session_id: Optional[str] = None) -> ZenAgent:
    if user_id in agents:
        a = agents[user_id]
        if session_id and a.session_id != session_id:
            a = ZenAgent(user_id=user_id, session_id=session_id)
            agents[user_id] = a
        return a
    a = ZenAgent(user_id=user_id, session_id=session_id)
    agents[user_id] = a
    # Evict oldest if over limit
    if len(agents) > MAX_AGENTS:
        agents.pop(next(iter(agents)), None)
    return a


# ── Models ──────────────────────────────────────────────────
class ChatReq(BaseModel):
    message: str
    user_id: str = "web-user"
    session_id: Optional[str] = None


class ChatResp(BaseModel):
    response: str
    reasoning: str = ""
    session_id: str
    user_id: str
    tool_calls: List[Dict[str, Any]] = []
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    sandbox_enabled: bool
    toolkits: Optional[List[str]]
    message_count: int


# ── Rate limiter (in-memory sliding window) ───────────────
_rate_limit_store: Dict[str, List[float]] = {}

def _rate_limit(request: Request) -> bool:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60.0
    if ip not in _rate_limit_store:
        _rate_limit_store[ip] = []
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if now - t < window]
    if len(_rate_limit_store[ip]) >= config.rate_limit_per_minute:
        return False
    _rate_limit_store[ip].append(now)
    return True


# ── Lifespan (graceful startup/shutdown) ──────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Zen Agent server")
    logger.info("Model: %s | Port: %d | Rate limit: %d/min", config.opencode_model, config.port, config.rate_limit_per_minute)
    yield
    logger.info("Shutting down, cleaning up %d agent(s)...", len(agents))
    for uid, agent in list(agents.items()):
        try:
            ComposioClient().delete_session(agent.session_id)
        except Exception:
            pass
    agents.clear()
    logger.info("Shutdown complete")


# ── App ─────────────────────────────────────────────────────
app = FastAPI(
    title="Zen Agent",
    description="AI agent with 23,790+ Composio tools via REST + WebSocket",
    version="3.3.0-arm32",
    lifespan=lifespan,
)

# Parse CORS origins
_cors_origins = config.cors_origins
if _cors_origins and _cors_origins != "*":
    _cors_list = [o.strip() for o in _cors_origins.split(",")]
else:
    _cors_list = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: rate limit + request ID ─────────────────────
@app.middleware("http")
async def middleware(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        if not _rate_limit(request):
            return JSONResponse(status_code=429, content={"error": "Rate limit exceeded", "retry_after": 60})
    req_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── REST API ────────────────────────────────────────────────
@app.get("/api/health")
async def health(request: Request):
    return {
        "status": "ok",
        "version": "3.3.0-arm32",
        "model": config.opencode_model,
        "fallback_model": config.opencode_fallback_model or None,
        "composio": "connected",
        "agents_active": len(agents),
        "max_tokens": config.opencode_max_tokens,
        "rate_limit": config.rate_limit_per_minute,
        "uptime_sec": int(time.time() - _start_time) if _start_time else 0,
    }


@app.post("/api/chat", response_model=ChatResp)
async def chat(req: ChatReq):
    if not req.message.strip():
        raise HTTPException(400, "Message required")
    if len(req.message) > config.max_message_length:
        raise HTTPException(400, f"Message too long (max {config.max_message_length:,} chars)")
    agent = get_agent(req.user_id, req.session_id)
    try:
        resp = agent.chat(req.message)
        if not isinstance(resp, LLMResponse):
            raise HTTPException(500, "Internal error")
        # DEBUG: dump exact content
        import logging as _lg2
        _c = resp.content or ''
        _lg2.getLogger("zen-server").info("API content_len=%d", len(_c))
        _lg2.getLogger("zen-server").info("API content_repr=%s", repr(_c[:300]))
        _lg2.getLogger("zen-server").info("API has_dsml=%s", str('<tool_calls>' in _c or '<tool_call>' in _c))
        return ChatResp(
            response=_c or "",
            reasoning=resp.reasoning[:3000] if resp.reasoning else "",
            session_id=agent.session_id or "",
            user_id=req.user_id,
            tool_calls=[
                {"name": tc["function"]["name"], "args": tc["function"]["arguments"]}
                for tc in (resp.tool_calls or [])
            ],
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            model=resp.model or config.opencode_model,
        )
    except Exception as e:
        logger.exception("Chat error for %s", req.user_id)
        raise HTTPException(500, str(e)[:500])


@app.get("/api/system-prompt/{user_id}")
async def get_system_prompt(user_id: str = "web-user"):
    agent = get_agent(user_id)
    prompt = agent._sysprompt() if hasattr(agent, '_sysprompt') else "N/A"
    return {"prompt": prompt}

@app.get("/api/session/{user_id}", response_model=SessionInfo)
async def get_session(user_id: str):
    agent = get_agent(user_id)
    i = agent.get_info()
    return SessionInfo(
        session_id=i["session_id"],
        user_id=i["user_id"],
        sandbox_enabled=i["sandbox_enabled"],
        toolkits=i["toolkits"],
        message_count=i["message_count"],
    )


@app.post("/api/session/{user_id}/reset")
async def reset_session(user_id: str):
    if user_id in agents:
        agents[user_id].clear_history()
    return {"status": "cleared"}


@app.delete("/api/session/{user_id}")
async def delete_session(user_id: str):
    if user_id in agents:
        a = agents.pop(user_id)
        try:
            ComposioClient().delete_session(a.session_id)
        except Exception:
            pass
    return {"status": "deleted"}


@app.get("/api/models")
async def list_models():
    return {"model": config.opencode_model, "fallback_model": config.opencode_fallback_model or None, "base_url": config.opencode_base_url, "max_tokens": config.opencode_max_tokens}

@app.get("/api/tools/list")
async def list_tools(page: int = 1, page_size: int = 20):
    try:
        return ComposioClient().list_all_tools(page=page, page_size=page_size)
    except ComposioAPIError as e:
        raise HTTPException(502, str(e))


@app.get("/api/tools/search")
async def search_tools(query: str, user_id: str = "web-user"):
    agent = get_agent(user_id)
    try:
        return agent._composio.search_tools(agent.session_id, query)
    except ComposioAPIError as e:
        raise HTTPException(502, str(e))


@app.get("/api/config")
async def get_config():
    return {
        "version": "3.3.0-arm32",
        "model": config.opencode_model,
        "fallback_model": config.opencode_fallback_model or None,
        "max_tokens": config.opencode_max_tokens,
        "max_history": config.max_history_messages,
        "data_dir": config.data_dir,
        "rate_limit": config.rate_limit_per_minute,
        "max_message_length": config.max_message_length,
        "request_timeout": config.request_timeout,
        "llm_timeout": config.llm_timeout,
    }


# ── WebSocket streaming ─────────────────────────────────────
@app.websocket("/ws/chat/{user_id}")
async def ws_chat(websocket: WebSocket, user_id: str):
    await websocket.accept()
    agent = get_agent(user_id)
    try:
        await websocket.send_json({
            "type": "info",
            "session_id": agent.session_id,
            "user_id": user_id,
            "model": config.opencode_model,
            "max_tokens": config.opencode_max_tokens,
        })
    except Exception:
        return
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                msg = data.get("message", "")
            except json.JSONDecodeError:
                msg = raw
            if not msg.strip():
                continue
            if msg.strip().lower() == "/clear":
                agent.clear_history()
                try:
                    await websocket.send_json({"type": "clear"})
                except Exception:
                    pass
                continue
            if msg.strip().lower() == "/ping":
                # Silently ignore - ping is just to keep connection alive
                continue
            if len(msg) > config.max_message_length:
                try:
                    await websocket.send_json({"type": "error", "message": f"Message too long (max {config.max_message_length:,} chars)"})
                except Exception:
                    pass
                continue
            # Run blocking agent.chat in a thread to avoid blocking event loop
            loop = asyncio.get_running_loop()
            full_content = ""
            stream_error = None
            
            def _run_chat():
                nonlocal stream_error, full_content
                try:
                    tokens = []
                    for token in agent.chat(msg, stream=True):
                        tokens.append(token)
                    return tokens
                except Exception as e:
                    stream_error = e
                    return []
            
            chat_tokens = await loop.run_in_executor(None, _run_chat)
            
            if stream_error:
                logger.exception("WS chat error for %s", user_id)
                try:
                    await websocket.send_json({"type": "error", "message": str(stream_error)[:500]})
                except Exception:
                    pass
                try:
                    await websocket.send_json({"type": "done", "content": "", "tokens": 0})
                except Exception:
                    pass
                continue
            
            # Send all tokens
            full_content = ""
            for token in chat_tokens:
                full_content += token
                try:
                    if token.startswith("__reasoning__"):
                        await websocket.send_json({"type": "reasoning", "content": token[13:]})
                    elif token.startswith("__status__:"):
                        await websocket.send_json({"type": "status", "content": token[11:]})
                    else:
                        await websocket.send_json({"type": "token", "content": token})
                except Exception:
                    # Connection closed - stop sending
                    break
            
            usage_info = agent.total_token_usage() if hasattr(agent, 'total_token_usage') else {}
            try:
                await websocket.send_json({"type": "done", "content": full_content, "tokens": usage_info.get("message_count", 0)})
            except Exception:
                pass
    except WebSocketDisconnect:
        logger.info("WS disconnected: %s", user_id)
    except Exception as e:
        logger.exception("WS error for %s", user_id)


# ── Frontend ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    p = os.path.join(static_dir, "index.html")
    if os.path.isfile(p):
        with open(p) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Zen Agent</h1><p>Dashboard not found</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host=config.host, port=config.port, log_level="info")
