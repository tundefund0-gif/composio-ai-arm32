"""Zen Agent — orchestrates LLM + Composio tool ecosystem with long-text support."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from config import config
from core.composio_client import ComposioClient, ComposioAPIError
from core.llm_client import LLMClient, LLMResponse, LLMError

logger = logging.getLogger("zen-agent")

META_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_SEARCH_TOOLS",
            "description": "Search for Composio tools relevant to a task. Returns tool slugs, descriptions, and schemas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "use_case": {"type": "string", "description": "NL description of what the user wants"}
                            },
                            "required": ["use_case"],
                        },
                    }
                },
                "required": ["queries"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_GET_TOOL_SCHEMAS",
            "description": "Get input/output schemas for specific tool slugs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_slugs": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["tool_slugs"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_MANAGE_CONNECTIONS",
            "description": "Check or create OAuth connections for toolkits. Returns auth links if user needs to connect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "toolkits": {"type": "array", "items": {"type": "string"}},
                    "reinitiate_all": {"type": "boolean", "description": "Force reconnection even if active"},
                },
                "required": ["toolkits"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_REMOTE_WORKBENCH",
            "description": "Execute Python code in remote sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_to_execute": {"type": "string", "description": "Python code to execute"},
                    "language": {"type": "string", "enum": ["python"]},
                },
                "required": ["code_to_execute"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_REMOTE_BASH_TOOL",
            "description": "Run shell commands in remote sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["cmd"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_MULTI_EXECUTE_TOOL",
            "description": "Execute multiple tools in parallel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tools": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool_slug": {"type": "string"},
                                "arguments": {"type": "object"},
                            },
                            "required": ["tool_slug"],
                        },
                    }
                },
                "required": ["tools"],
            },
        }
    },
]


def _truncate_msgs(messages: List[Dict], max_count: int = 60) -> List[Dict]:
    """Keep system prompt + last N messages to stay within context limits."""
    if len(messages) <= max_count:
        return messages
    # Keep system (first) + last max_count-1
    return [messages[0]] + messages[-(max_count - 1):]


class ZenAgent:
    """AI agent with 1,000+ Composio tools, long-text, streaming & tool loops."""

    def __init__(self, user_id: str, session_id: Optional[str] = None,
                 toolkits: Optional[List[str]] = None, enable_sandbox: bool = True):
        self.user_id = user_id
        self.session_id = session_id
        self.toolkits = toolkits
        self.enable_sandbox = enable_sandbox
        self._composio = ComposioClient()
        self._llm = LLMClient()
        self._messages: List[Dict] = []
        self._init_session()

    def _init_session(self):
        if self.session_id:
            try:
                self._composio.get_session(self.session_id)
                logger.info("Reusing session %s", self.session_id)
                return
            except ComposioAPIError:
                logger.info("Session %s not found, creating new", self.session_id)
                self.session_id = None
        s = self._composio.create_session(
            user_id=self.user_id, toolkits=self.toolkits, sandbox=self.enable_sandbox
        )
        self.session_id = s["session_id"]
        logger.info("Created session %s", self.session_id)

    def _sysprompt(self) -> str:
        return f"""You are Zen Agent, an AI assistant with access to 1,000+ tools via Composio.

**Capabilities:**
- Search and execute tools from 1,000+ apps (Gmail, GitHub, Slack, Notion, etc.)
- Write and run Python code in a remote sandbox
- Connect user accounts (OAuth) for any toolkit
- Make direct HTTP requests through connected accounts

**Workflow:**
1. Use COMPOSIO_SEARCH_TOOLS to find relevant tools for the user's request.
2. Check connections with COMPOSIO_MANAGE_CONNECTIONS. If not active, show the user the auth link.
3. Execute tools via COMPOSIO_MULTI_EXECUTE_TOOL.
4. For complex tasks, use COMPOSIO_REMOTE_WORKBENCH.

Session: {self.session_id} | User: {self.user_id}
Current UTC time: {datetime.now(timezone.utc).isoformat()}
"""

    # ── Public API ──────────────────────────────────────────
    def chat(self, message: str, stream: bool = False) -> LLMResponse | Generator[str, None, None]:
        self._messages.append({"role": "user", "content": message})
        if stream:
            return self._stream()
        return self._sync()

    def get_history(self) -> List[Dict]:
        return list(self._messages)

    def clear_history(self):
        self._messages = []

    def get_info(self) -> Dict:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "sandbox_enabled": self.enable_sandbox,
            "toolkits": self.toolkits,
            "message_count": len(self._messages),
        }

    # ── Sync path (REST) ─────────────────────────────────────
    def _sync(self) -> LLMResponse:
        msgs = _truncate_msgs(self._build_msgs(), config.max_history_messages)
        resp = self._llm.chat(msgs, tools=META_TOOL_DEFS, retries=2)
        if resp.tool_calls:
            return self._handle_tools(resp, msgs)
        self._messages.append({"role": "assistant", "content": resp.content})
        return resp

    def _handle_tools(self, resp: LLMResponse, msgs: List[Dict]) -> LLMResponse:
        msgs.append(resp.message)
        for tc in resp.tool_calls or []:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            result = self._exec_composio(fn, args)
            result_str = json.dumps(result, default=str)[:config.max_tool_results_length]
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result_str})
        final = self._llm.chat(msgs, retries=1)
        self._messages.append({"role": "assistant", "content": final.content})
        return final

    # ── Streaming path (WebSocket) ───────────────────────────
    def _stream(self) -> Generator[str, None, None]:
        """Stream with full tool call loop support."""
        msgs = _truncate_msgs(self._build_msgs(), config.max_history_messages)
        gen = self._llm.chat(msgs, stream=True, tools=META_TOOL_DEFS, retries=2)
        if not isinstance(gen, Generator):
            return

        while True:
            full_content = ""
            tool_calls_data = None

            for token in gen:
                if token.startswith("__reasoning__"):
                    yield token
                elif token.startswith("__tool_calls__:"):
                    try:
                        tool_calls_data = json.loads(token[len("__tool_calls__:"):])
                    except json.JSONDecodeError:
                        tool_calls_data = None
                else:
                    full_content += token
                    yield token

            if not tool_calls_data:
                self._messages.append({"role": "assistant", "content": full_content})
                break

            # Execute tools and loop
            if full_content.strip():
                msgs.append({"role": "assistant", "content": full_content})

            assistant_msg = {
                "role": "assistant",
                "content": full_content.strip() or None,
                "tool_calls": [],
            }
            for tc in tool_calls_data:
                tc_id = tc.get("id", f"call_{abs(hash(str(tc))) % 10**6}")
                entry = {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                assistant_msg["tool_calls"].append(entry)

            msgs.append(assistant_msg)

            for tc in tool_calls_data:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                result = self._exec_composio(fn, args)
                result_str = json.dumps(result, default=str)[:config.max_tool_results_length]
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{abs(hash(str(tc))) % 10**6}"),
                    "content": result_str,
                })

            gen = self._llm.chat(msgs, stream=True, retries=1)
            if not isinstance(gen, Generator):
                break

    # ── Tool execution ───────────────────────────────────────
    def _exec_composio(self, action: str, args: Dict) -> Dict[str, Any]:
        try:
            return self._composio.execute_meta(self.session_id, action, args)
        except ComposioAPIError as e:
            logger.error("Tool %s failed: %s", action, e)
            return {"error": str(e), "details": str(e.body)[:500] if e.body else None}

    def _build_msgs(self) -> List[Dict]:
        return [{"role": "system", "content": self._sysprompt()}, *self._messages]
