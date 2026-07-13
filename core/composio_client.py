"""Composio REST API client — retry, meta tool routing, session management."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from config import config

logger = logging.getLogger("composio-client")


class ComposioAPIError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        self.status_code = status_code
        self.body = body
        super().__init__(message)


class ComposioClient:
    """Direct wrapper around Composio REST API with retry & meta routing."""

    BASE = config.composio_base_url.rstrip("/")

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.composio_api_key

    def _client(self) -> httpx.Client:
        return httpx.Client(
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            timeout=config.request_timeout,
        )

    def _req(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make request with automatic retry on 5xx / timeout."""
        last_err = None
        for attempt in range(3):
            try:
                with self._client() as cl:
                    r = cl.request(method, f"{self.BASE}{path}", **kwargs)
                    if r.status_code >= 500 and attempt < 2:
                        time.sleep(1 * (attempt + 1))
                        continue
                    return r
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_err = e
                logger.warning("Composio req attempt %d/3 failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise ComposioAPIError(f"Request failed after 3 retries: {last_err}")

    def _handle(self, resp: httpx.Response, ctx: str) -> Dict[str, Any]:
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise ComposioAPIError(f"Composio API error ({ctx}): HTTP {resp.status_code}",
                                   status_code=resp.status_code, body=body)
        try:
            return resp.json()
        except Exception as e:
            raise ComposioAPIError(f"Invalid JSON ({ctx}): {e}")

    # ── Sessions ────────────────────────────────────────────
    def create_session(self, user_id: str, toolkits: Optional[List[str]] = None,
                       sandbox: bool = False) -> Dict[str, Any]:
        body: Dict[str, Any] = {"user_id": user_id}
        if toolkits:
            body["toolkits"] = {"enable": toolkits}
        if sandbox:
            body["workbench"] = {"enable": True}
        r = self._req("POST", "/api/v3.1/tool_router/session", json=body)
        return self._handle(r, "create session")

    def get_session(self, session_id: str) -> Dict[str, Any]:
        r = self._req("GET", f"/api/v3.1/tool_router/session/{session_id}")
        return self._handle(r, "get session")

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        r = self._req("DELETE", f"/api/v3.1/tool_router/session/{session_id}")
        return self._handle(r, "delete session")

    # ── Execute ─────────────────────────────────────────────
    def execute_tool(self, session_id: str, tool_slug: str,
                     arguments: Optional[Dict] = None) -> Dict[str, Any]:
        body = {"tool_slug": tool_slug, "arguments": arguments or {}}
        r = self._req("POST", f"/api/v3.1/tool_router/session/{session_id}/execute", json=body)
        return self._handle(r, f"execute {tool_slug}")

    def execute_meta(self, session_id: str, action: str,
                     arguments: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a meta tool. Sends directly to API — COMPOSIO_* tools are valid tool slugs."""
        return self.execute_tool(session_id, tool_slug=action, arguments=arguments)

    # ── Discovery ────────────────────────────────────────────
    def search_tools(self, session_id: str, use_case: str) -> Dict[str, Any]:
        return self.execute_meta(session_id, "COMPOSIO_SEARCH_TOOLS",
                                 {"queries": [{"use_case": use_case}]})

    def get_tools(self, session_id: str) -> Dict[str, Any]:
        r = self._req("GET", f"/api/v3.1/tool_router/session/{session_id}/tools")
        return self._handle(r, "get tools")

    def get_tool_schemas(self, session_id: str, tool_slugs: List[str]) -> Dict[str, Any]:
        return self.execute_meta(session_id, "COMPOSIO_GET_TOOL_SCHEMAS",
                                 {"tool_slugs": tool_slugs})

    def list_all_tools(self, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        r = self._req("GET", "/api/v3/tools", params={"page": page, "pageSize": page_size})
        return self._handle(r, "list tools")

    # ── Connections ──────────────────────────────────────────
    def manage_connections(self, session_id: str, toolkits: List[str],
                           reinitiate: bool = False) -> Dict[str, Any]:
        return self.execute_meta(session_id, "COMPOSIO_MANAGE_CONNECTIONS",
                                 {"toolkits": toolkits, "reinitiate_all": reinitiate})

    def link_account(self, session_id: str, toolkit: str) -> Dict[str, Any]:
        return self.manage_connections(session_id, [toolkit])

    # ── Sandbox ──────────────────────────────────────────────
    def execute_workbench(self, session_id: str, code: str) -> Dict[str, Any]:
        return self.execute_meta(session_id, "COMPOSIO_REMOTE_WORKBENCH",
                                 {"code_to_execute": code, "language": "python"})

    def execute_bash(self, session_id: str, command: str) -> Dict[str, Any]:
        return self.execute_meta(session_id, "COMPOSIO_REMOTE_BASH_TOOL", {"cmd": command})

    def multi_execute(self, session_id: str, tools: List[Dict]) -> Dict[str, Any]:
        return self.execute_meta(session_id, "COMPOSIO_MULTI_EXECUTE_TOOL", {"tools": tools})

    # ── Proxy ───────────────────────────────────────────────
    def proxy_execute(self, session_id: str, endpoint: str, method: str = "GET",
                      headers: Optional[Dict[str, str]] = None,
                      body: Optional[Dict] = None) -> Dict[str, Any]:
        payload = {"endpoint": endpoint, "method": method}
        if headers:
            payload["headers"] = headers
        if body:
            payload["body"] = body
        r = self._req("POST",
                      f"/api/v3.1/tool_router/session/{session_id}/proxy_execute", json=payload)
        return self._handle(r, "proxy execute")

    # ── Config history ───────────────────────────────────────
    def config_history(self, session_id: str) -> Dict[str, Any]:
        r = self._req("GET", f"/api/v3.1/tool_router/session/{session_id}/config_history")
        return self._handle(r, "config history")
