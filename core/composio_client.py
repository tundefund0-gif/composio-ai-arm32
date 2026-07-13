"""Composio REST API client — direct HTTP, no SDK dependency."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from config import config


class ComposioAPIError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        self.status_code = status_code
        self.body = body
        super().__init__(message)


class ComposioClient:
    """Direct wrapper around Composio REST API v3/v3.1."""

    BASE = config.composio_base_url.rstrip("/")

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.composio_api_key

    def _client(self) -> httpx.Client:
        return httpx.Client(
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            timeout=30.0,
        )

    # ── Sessions ──────────────────────────────────────────────────────
    def create_session(self, user_id: str, toolkits: Optional[List[str]] = None, sandbox: bool = False) -> Dict[str, Any]:
        body: Dict[str, Any] = {"user_id": user_id}
        if toolkits:
            body["toolkits"] = {"enable": toolkits}
        if sandbox:
            body["workbench"] = {"enable": True}
        with self._client() as cl:
            return self._handle(cl.post(f"{self.BASE}/api/v3.1/tool_router/session", json=body), "create session")

    def get_session(self, session_id: str) -> Dict[str, Any]:
        with self._client() as cl:
            return self._handle(cl.get(f"{self.BASE}/api/v3.1/tool_router/session/{session_id}"), "get session")

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        with self._client() as cl:
            return self._handle(cl.delete(f"{self.BASE}/api/v3.1/tool_router/session/{session_id}"), "delete session")

    # ── Tool execution ────────────────────────────────────────────────
    def execute_tool(self, session_id: str, tool_slug: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = {"tool_slug": tool_slug, "arguments": arguments or {}}
        with self._client() as cl:
            return self._handle(cl.post(f"{self.BASE}/api/v3.1/tool_router/session/{session_id}/execute", json=body), "execute tool")

    def execute_meta(self, session_id: str, action: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a meta tool by slug (SEARCH, WORKBENCH, MANAGE_CONNECTIONS, etc.)."""
        return self.execute_tool(session_id, tool_slug=action, arguments=arguments)

    # ── Tool discovery ────────────────────────────────────────────────
    def search_tools(self, session_id: str, use_case: str) -> Dict[str, Any]:
        return self.execute_meta(session_id, "COMPOSIO_SEARCH_TOOLS", {"queries": [{"use_case": use_case}]})

    def get_tools(self, session_id: str) -> Dict[str, Any]:
        with self._client() as cl:
            return self._handle(cl.get(f"{self.BASE}/api/v3.1/tool_router/session/{session_id}/tools"), "get tools")

    def get_tool_schemas(self, session_id: str, tool_slugs: List[str]) -> Dict[str, Any]:
        return self.execute_meta(session_id, "COMPOSIO_GET_TOOL_SCHEMAS", {"tool_slugs": tool_slugs})

    def list_all_tools(self, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        with self._client() as cl:
            return self._handle(cl.get(f"{self.BASE}/api/v3/tools", params={"page": page, "pageSize": page_size}), "list tools")

    # ── Auth / connections ────────────────────────────────────────────
    def manage_connections(self, session_id: str, toolkits: List[str], reinitiate: bool = False) -> Dict[str, Any]:
        return self.execute_meta(session_id, "COMPOSIO_MANAGE_CONNECTIONS", {"toolkits": toolkits, "reinitiate_all": reinitiate})

    def link_account(self, session_id: str, toolkit: str) -> Dict[str, Any]:
        return self.manage_connections(session_id, [toolkit])

    # ── Sandbox / multi ───────────────────────────────────────────────
    def execute_workbench(self, session_id: str, code: str) -> Dict[str, Any]:
        return self.execute_meta(session_id, "COMPOSIO_REMOTE_WORKBENCH", {"code": code, "language": "python"})

    def execute_bash(self, session_id: str, command: str) -> Dict[str, Any]:
        return self.execute_meta(session_id, "COMPOSIO_REMOTE_BASH_TOOL", {"cmd": command})

    def multi_execute(self, session_id: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.execute_meta(session_id, "COMPOSIO_MULTI_EXECUTE_TOOL", {"tools": tools})

    # ── Proxy ─────────────────────────────────────────────────────────
    def proxy_execute(self, session_id: str, endpoint: str, method: str = "GET", headers: Optional[Dict[str, str]] = None, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"endpoint": endpoint, "method": method}
        if headers: payload["headers"] = headers
        if body: payload["body"] = body
        with self._client() as cl:
            return self._handle(cl.post(f"{self.BASE}/api/v3.1/tool_router/session/{session_id}/proxy_execute", json=payload), "proxy execute")

    # ── Config history ────────────────────────────────────────────────
    def config_history(self, session_id: str) -> Dict[str, Any]:
        with self._client() as cl:
            return self._handle(cl.get(f"{self.BASE}/api/v3.1/tool_router/session/{session_id}/config_history"), "config history")

    # ── Internal ──────────────────────────────────────────────────────
    @staticmethod
    def _handle(resp: httpx.Response, ctx: str) -> Dict[str, Any]:
        if resp.status_code >= 400:
            try: body = resp.json()
            except Exception: body = resp.text
            raise ComposioAPIError(f"Composio API error ({ctx}): HTTP {resp.status_code}", status_code=resp.status_code, body=body)
        try: return resp.json()
        except Exception as e: raise ComposioAPIError(f"Invalid JSON ({ctx}): {e}")
