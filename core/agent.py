"""Zen Agent — orchestrates LLM + Composio tool ecosystem with long-text support."""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from config import config
from core.composio_client import ComposioClient, ComposioAPIError
from core.llm_client import LLMClient, LLMResponse, LLMError

logger = logging.getLogger("zen-agent")



def _parse_dsml_tool_calls(text):
    """Parse DSML/XML formatted tool calls from model text output.
    Handles formats like:
      <tool_calls><invoke name="TOOL">...</invoke></tool_calls>
      <||DSML||tool_calls>...</||DSML||tool_calls>
    """
    if text:
        text = text.replace(chr(0xFF5C), chr(0x007C))
    import re as _re
    if not text:
        return None
    
    # Match common patterns for tool call wrappers
    # Pattern: <tool_calls>...</tool_calls> or <||DSML||tool_calls>...</||DSML||tool_calls>
    outer = _re.search(
        _re.compile(
            "<" + "(?:[" + chr(124) + "][" + chr(124) + "]DSML[" + chr(124) + "][" + chr(124) + "])?"
            "tool_calls[^>]*>(.*?)</"
            "(?:[" + chr(124) + "][" + chr(124) + "]DSML[" + chr(124) + "][" + chr(124) + "])?tool_calls[^>]*>",
            _re.DOTALL
        ),
        text
    )
    inner = outer.group(1) if outer else text
    
    # Parse individual <invoke> tags
    tag_re = _re.compile(
        "<" + "(?:[" + chr(124) + "][" + chr(124) + "]DSML[" + chr(124) + "][" + chr(124) + "])?"
        "invoke" + chr(32) + "+name=" + chr(34) + "([a-zA-Z_]\\w*)" + chr(34)
        + "[^>]*>(.*?)</"
        + "(?:[" + chr(124) + "][" + chr(124) + "]DSML[" + chr(124) + "][" + chr(124) + "])?invoke[^>]*>",
        _re.DOTALL
    )
    invokes = tag_re.findall(inner)
    if not invokes:
        return None
    
    param_re = _re.compile(
        "<" + "(?:[" + chr(124) + "][" + chr(124) + "]DSML[" + chr(124) + "][" + chr(124) + "])?"
        "parameter" + chr(32) + "+name=" + chr(34) + "([a-zA-Z_]\\w*)" + chr(34)
        + "[^>]*>(.*?)</"
        + "(?:[" + chr(124) + "][" + chr(124) + "]DSML[" + chr(124) + "][" + chr(124) + "])?parameter[^>]*>",
        _re.DOTALL
    )
    
    calls = []
    for name, body in invokes:
        params = param_re.findall(body)
        args = {}
        for pname, pval in params:
            pval = pval.strip()
            try:
                args[pname] = json.loads(pval)
            except Exception:
                args[pname] = pval
        calls.append({
            "id": "call_" + str(abs(hash(name)) % 10**6),
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    return calls if calls else None


def _strip_dsml_tags(text):
    """Remove DSML/XML markup from text, keeping only natural language content."""
    if text:
        text = text.replace(chr(0xFF5C), chr(0x007C))
    import re as _re
    # Match tags with optional pipe decoration: <||DSML||tool_calls||>, <||tool_calls>, <tool_calls>
    _pipes = "[" + chr(124) + chr(0xFF5C) + "]*"  # zero or more pipe chars (standard or fullwidth)
    _dsml = "(?:" + _pipes + "DSML" + _pipes + ")?"  # optional ||DSML|| prefix
    _tag = lambda name: "<" + _pipes + _dsml + name + _pipes + "[^>]*>.*?</" + _pipes + _dsml + name + _pipes + "[^>]*>"
    text = _re.sub(_re.compile(_tag("tool_calls"), _re.DOTALL), "", text)
    text = _re.sub(_re.compile(_tag("invoke"), _re.DOTALL), "", text)
    return text.strip()


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
        return f"""You are Zen Agent, an AI assistant with access to 23,790+ tools via Composio.

**CRITICAL - You can ONLY call the 6 meta tools listed below.**
You CANNOT call individual app tools directly (like GMAIL_*, GITHUB_*, SLACK_*, etc.).
Those must be executed through COMPOSIO_MULTI_EXECUTE_TOOL.

**Available Functions (only these 6):**
1. COMPOSIO_SEARCH_TOOLS - Search for tools by use case description
2. COMPOSIO_GET_TOOL_SCHEMAS - Get input/output schemas for specific tool slugs
3. COMPOSIO_MANAGE_CONNECTIONS - Check or create OAuth connections for toolkits
4. COMPOSIO_REMOTE_WORKBENCH - Execute Python code in remote sandbox
5. COMPOSIO_REMOTE_BASH_TOOL - Run shell commands in remote sandbox
6. COMPOSIO_MULTI_EXECUTE_TOOL - Execute multiple tools in parallel

**Workflow:**
1. Use COMPOSIO_SEARCH_TOOLS to find relevant tools for the user's request.
2. Check connections with COMPOSIO_MANAGE_CONNECTIONS. If not active, show the user the auth link.
3. Execute tools via COMPOSIO_MULTI_EXECUTE_TOOL (pass tool_slug and arguments).
4. For complex tasks, use COMPOSIO_REMOTE_WORKBENCH.

**Example of executing a Gmail tool:**
COMPOSIO_MULTI_EXECUTE_TOOL with tools=[{{"tool_slug": "GMAIL_LIST_LABELS", "arguments": {{"userId": "me"}}}}]

**NEVER invent function names that are not listed above.**
Only call the 6 functions listed above. If you need a tool not listed, use COMPOSIO_SEARCH_TOOLS first.

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
        usage = self.total_token_usage()
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "sandbox_enabled": self.enable_sandbox,
            "toolkits": self.toolkits,
            "model": self._llm.model,
            "message_count": len(self._messages),
            "total_input_chars": usage["input_chars"],
            "total_output_chars": usage["output_chars"],
            "total_tool_calls": usage["tool_calls"],
        }

    # ── Sync path (REST) ─────────────────────────────────────
    def _sync(self) -> LLMResponse:
        msgs = _truncate_msgs(self._build_msgs(), config.max_history_messages)
        resp = self._llm.chat(msgs, tools=META_TOOL_DEFS, retries=2)
        if resp.tool_calls:
            return self._handle_tools(resp, msgs)
        # Strip DSML/XML markup from response text (model may include it)
        if resp.content:
            cleaned = _strip_dsml_tags(resp.content)
            if cleaned != resp.content:
                logger.info("DSML: stripped from initial response")
                resp.message["content"] = cleaned or None
                resp.content = cleaned or ""
        self._messages.append({"role": "assistant", "content": resp.content})
        return resp

    def _handle_tools(self, resp: LLMResponse, msgs: List[Dict], _depth: int = 0) -> LLMResponse:
        if _depth > 5:
            logger.warning("Tool call depth exceeded, stopping")
            final = resp
            self._messages.append({"role": "assistant", "content": final.content or ""})
            return final
        msgs.append(resp.message)
        self._messages.append(resp.message)
        for tc in resp.tool_calls or []:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            result = self._exec_composio(fn, args)
            result_str = json.dumps(result, default=str)[:config.max_tool_results_length]
            tool_msg = {"role": "tool", "tool_call_id": tc["id"], "content": result_str}
            msgs.append(tool_msg)
            self._messages.append(tool_msg)
        final = self._llm.chat(msgs, retries=1)
        # Clean DSML/XML markup from final response (model may include it as text)
        if final.content:
            cleaned = _strip_dsml_tags(final.content)
            if cleaned != final.content:
                logger.info("DSML: stripped tags from response (depth=%d)", _depth)
                final.message["content"] = cleaned or None
                final.content = cleaned or ""
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
                    # Strip DSML from each token before sending to client
                    yield _strip_dsml_tags(token) or token

            if not tool_calls_data:
                # Strip DSML/XML from streamed text and store clean version in history
                cleaned = _strip_dsml_tags(full_content) if full_content else full_content
                if cleaned != full_content:
                    logger.info("DSML: stripped from streamed response")
                self._messages.append({"role": "assistant", "content": cleaned or full_content})
                break

            # Execute tools and loop
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
            self._messages.append(assistant_msg)

            for tc in tool_calls_data:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                result = self._exec_composio(fn, args)
                result_str = json.dumps(result, default=str)[:config.max_tool_results_length]
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{abs(hash(str(tc))) % 10**6}"),
                    "content": result_str,
                }
                msgs.append(tool_msg)
                self._messages.append(tool_msg)

            gen = self._llm.chat(msgs, stream=True, retries=1)
            if not isinstance(gen, Generator):
                break

    # ── Session persistence ──────────────────────────────────
    @property
    def _history_dir(self) -> Path:
        p = Path(config.data_dir) / "conversations"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_session(self, path: Optional[str] = None) -> str:
        """Save conversation history to a JSON file. Returns the file path."""
        if path is None:
            path = str(self._history_dir / f"{self.user_id}_{uuid.uuid4().hex[:12]}.json")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "toolkits": self.toolkits,
            "sandbox_enabled": self.enable_sandbox,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "messages": self._messages,
        }
        p.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Session saved to %s (%d messages)", p, len(self._messages))
        return str(p)

    def load_session(self, path: str) -> None:
        """Load conversation history from a JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Session file not found: {path}")
        data = json.loads(p.read_text())
        ver = data.get("version", 1)
        self._messages = data.get("messages", [])
        if ver < 2:
            self._messages = [m for m in self._messages if m.get("role") != "system"]
        logger.info("Session loaded from %s (%d messages)", p, len(self._messages))

    def export_markdown(self, path: Optional[str] = None) -> str:
        """Export conversation as Markdown. Returns the file path."""
        if path is None:
            path = str(self._history_dir / f"{self.user_id}_{uuid.uuid4().hex[:12]}.md")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        parts = [f"# Zen Agent - Conversation", f"User: {self.user_id}"]
        for msg in self._messages:
            role = msg.get("role", "unknown")
            text = msg.get("content", "") or ""
            if role == "user":
                parts.append("\n## 👤 You\n\n" + text + "\n")
            elif role == "assistant":
                tc = msg.get("tool_calls")
                if tc:
                    fn_list = ", ".join(t.get("function", {}).get("name", "?") for t in tc)
                    parts.append("\n## 🤖 AI (calling " + fn_list + ")\n\n" + text + "\n")
                else:
                    parts.append("\n## 🤖 AI\n\n" + text + "\n")
            elif role == "tool":
                snippet = text[:200] + "..." if len(text) > 200 else text
                parts.append("\n**Tool result:** `" + snippet + "`\n")
        p.write_text("".join(parts))
        return str(p)

    def total_token_usage(self) -> Dict[str, int]:
        """Calculate cumulative token usage from all LLM responses stored in messages."""
        total_input = 0
        total_output = 0
        total_tool_calls = 0
        for msg in self._messages:
            if msg.get("role") == "assistant":
                tc = msg.get("tool_calls")
                if tc:
                    total_tool_calls += len(tc)
        # Estimate: count chars as rough proxy when we don't have actual token counts
        for msg in self._messages:
            content = msg.get("content", "") or ""
            if msg.get("role") == "user":
                total_input += len(content)
            elif msg.get("role") == "assistant":
                total_output += len(content)
        return {
            "input_chars": total_input,
            "output_chars": total_output,
            "tool_calls": total_tool_calls,
            "message_count": len(self._messages),
        }

    # ── Tool execution ───────────────────────────────────────
    def _exec_composio(self, action: str, args: Dict) -> Dict[str, Any]:
        try:
            return self._composio.execute_meta(self.session_id, action, args)
        except ComposioAPIError as e:
            logger.error("Tool %s failed: %s", action, e)
            err_body = str(e.body)[:500] if e.body else ""
            return {
                "error": str(e),
                "details": err_body,
                "available_functions": ["COMPOSIO_SEARCH_TOOLS", "COMPOSIO_GET_TOOL_SCHEMAS",
                    "COMPOSIO_MANAGE_CONNECTIONS", "COMPOSIO_REMOTE_WORKBENCH",
                    "COMPOSIO_REMOTE_BASH_TOOL", "COMPOSIO_MULTI_EXECUTE_TOOL"],
                "hint": f"Tool '{action}' failed. If this is an app tool (like GMAIL_*), "
                        f"you must execute it via COMPOSIO_MULTI_EXECUTE_TOOL with tool_slug='{action}'."
            }

    def _build_msgs(self) -> List[Dict]:
        return [{"role": "system", "content": self._sysprompt()}, *self._messages]
