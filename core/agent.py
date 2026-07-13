"""Zen Agent — orchestrates LLM + Composio tool ecosystem."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from config import config
from core.composio_client import ComposioClient, ComposioAPIError
from core.llm_client import LLMClient, LLMResponse, LLMError

logger = logging.getLogger("zen-agent")

# OpenAI-compatible tool definitions for the meta tools
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
                                "use_case": {"type": "string", "description": "Natural language description of what the user wants to accomplish"}
                            },
                            "required": ["use_case"]
                        }
                    }
                },
                "required": ["queries"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_GET_TOOL_SCHEMAS",
            "description": "Get input and output schemas for specific tool slugs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_slugs": {"type": "array", "items": {"type": "string"}, "description": "Tool slugs to get schemas for"}
                },
                "required": ["tool_slugs"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_MANAGE_CONNECTIONS",
            "description": "Check or create OAuth connections for toolkits. Returns auth links if the user needs to connect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "toolkits": {"type": "array", "items": {"type": "string"}, "description": "Toolkit slugs to check or connect"},
                    "reinitiate_all": {"type": "boolean", "description": "Force reconnection even if active"}
                },
                "required": ["toolkits"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_REMOTE_WORKBENCH",
            "description": "Execute Python code in a remote sandbox. Use for data processing, file generation, complex logic, or any task that needs code execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_to_execute": {"type": "string", "description": "Python code to execute"},
                    "language": {"type": "string", "enum": ["python"], "description": "Language"}
                },
                "required": ["code_to_execute"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_REMOTE_BASH_TOOL",
            "description": "Run shell commands in the remote sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["cmd"]
            }
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
                                "arguments": {"type": "object"}
                            },
                            "required": ["tool_slug"]
                        }
                    }
                },
                "required": ["tools"]
            }
        }
    }
]


class ZenAgent:
    """AI agent with 1,000+ Composio tools, per-user sessions, and streaming."""

    def __init__(self, user_id: str, session_id: Optional[str] = None, toolkits: Optional[List[str]] = None, enable_sandbox: bool = True):
        self.user_id = user_id
        self.session_id = session_id
        self.toolkits = toolkits
        self.enable_sandbox = enable_sandbox
        self._composio = ComposioClient()
        self._llm = LLMClient()
        self._messages: List[Dict[str, Any]] = []
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
        s = self._composio.create_session(user_id=self.user_id, toolkits=self.toolkits, sandbox=self.enable_sandbox)
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

    def chat(self, message: str, stream: bool = False) -> LLMResponse | Generator[str, None, None]:
        self._messages.append({"role": "user", "content": message})
        if stream:
            return self._stream()
        return self._sync()

    def _sync(self) -> LLMResponse:
        msgs = self._build_msgs()
        resp = self._llm.chat(msgs, tools=META_TOOL_DEFS)
        if resp.tool_calls:
            return self._handle_tools(resp, msgs)
        self._messages.append({"role": "assistant", "content": resp.content})
        return resp

    def _handle_tools(self, resp: LLMResponse, msgs: List[Dict[str, Any]]) -> LLMResponse:
        msgs.append(resp.message)
        for tc in (resp.tool_calls or []):
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            result = self._exec_composio(fn, args)
            result_str = json.dumps(result, default=str)[:10000]
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result_str})
        final = self._llm.chat(msgs)
        self._messages.append({"role": "assistant", "content": final.content})
        return final

    def _stream(self) -> Generator[str, None, None]:
        """Streaming path with tool call handling."""
        msgs = self._build_msgs()
        gen = self._llm.chat(msgs, stream=True, tools=META_TOOL_DEFS)
        
        if not isinstance(gen, Generator):
            return
        
        # Collect stream — if tool calls are found, execute them and loop
        while True:
            current_gen = gen
            full_content = ""
            tool_calls_data = None
            has_reasoning = False
            
            for token in current_gen:
                if token.startswith("__reasoning__"):
                    has_reasoning = True
                    yield token
                elif token.startswith("__tool_calls__:"):
                    # Tool calls detected! Parse them
                    try:
                        tool_calls_data = json.loads(token[len("__tool_calls__:"):])
                    except json.JSONDecodeError:
                        tool_calls_data = None
                else:
                    full_content += token
                    yield token
            
            if tool_calls_data:
                # Execute tools and add results to messages
                # If we had content streamed before tool calls, add it as assistant message
                if full_content.strip():
                    msgs.append({"role": "assistant", "content": full_content})
                
                # Add tool call messages and results
                assistant_msg = {"role": "assistant", "content": full_content.strip() or None, "tool_calls": []}
                for tc in tool_calls_data:
                    tc_entry = {
                        "id": tc.get("id", f"call_{hash(str(tc)) % 1000000}"),
                        "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                    }
                    assistant_msg["tool_calls"].append(tc_entry)
                
                msgs.append(assistant_msg)
                
                for tc in tool_calls_data:
                    fn = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    result = self._exec_composio(fn, args)
                    result_str = json.dumps(result, default=str)[:10000]
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{hash(str(tc)) % 1000000}"),
                        "content": result_str
                    })
                
                # Make follow-up call - stream the response
                gen = self._llm.chat(msgs, stream=True)
                if not isinstance(gen, Generator):
                    break
                # Continue loop to stream the follow-up response
                continue
            else:
                # No tool calls — store the response and done
                self._messages.append({"role": "assistant", "content": full_content})
                break

    def _exec_composio(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._composio.execute_meta(self.session_id, action, args)
        except ComposioAPIError as e:
            logger.error("Tool failed: %s", e)
            return {"error": str(e), "details": str(e.body)[:500] if e.body else None}

    def _build_msgs(self) -> List[Dict[str, Any]]:
        return [{"role": "system", "content": self._sysprompt()}, *self._messages]

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._messages)

    def clear_history(self):
        self._messages = []

    def get_info(self) -> Dict[str, Any]:
        return {"user_id": self.user_id, "session_id": self.session_id, "sandbox_enabled": self.enable_sandbox, "toolkits": self.toolkits, "message_count": len(self._messages)}
